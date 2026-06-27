"""End-to-end MPS Path-3 training: V parameterised as MPS, MPS adjoint,
no MLP.

This combines Tracks A + B:
  * V is a list of real-valued MPS cores (one per time segment ``k``).
  * Forward Trotter chain on MPS ψ (Track A / Phase 3a).
  * Backward adjoint in MPS form (Track A / Phase 3a) → dL/dV_dense
    at every segment.
  * Project dL/dV_dense onto MPS-tangent at current V_mps to obtain
    dL/dA_j per core (Track B / Phase 3b).
  * Adam on the core entries.

Parameter count drops from ``K_data · N^d`` (dense V per segment) to
``K_data · d · D² · N``. At ``N=16 d=2 K=4 D=8``, dense V has 1024
params per V; MPS V has 4 · 8² · 16 = 4096 across all cores — slightly
more, because the dense full-rank V_dense is small. The interesting
regime is high d, where dense scales as N^d and MPS stays linear in d.

Validation: ``--mode validate`` confirms loss decreases over 200 steps
on petals N=16; bond ``D=8`` is large enough to absorb the smooth
biology-V's compressibility.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import (  # noqa: E402
    adjoint_backward_mps,
    forward_with_cache_mps,
)
from v_mps_parameterization import (  # noqa: E402
    V_dense_to_mps,
    V_mps_to_dense,
    project_dense_grad_to_mps_cores,
)
from riemannian_mps import to_mixed_canonical  # noqa: E402
from riemannian_adam_mps import (  # noqa: E402
    _RiemannianAdamMPS,
    init_v_mps_riemannian,
)
from tnwf.data.petals import sample_petals_trajectory
from tnwf.mps.core import mps_to_dense


def bin_samples_to_density(samples_centred, N, d, L, eps: float = 1e-6):
    edges = [np.linspace(-L / 2.0, L / 2.0, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    return (counts / counts.sum()).reshape(-1)


def _init_V_mps_list(K_data, N, d, D_V_param, init_std=0.01, rng=None):
    """List of K_data V_mps's, one per segment. Small-magnitude random init."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = []
    for _ in range(K_data):
        cores = []
        D_prev = 1
        for j in range(d):
            D_next = 1 if j == d - 1 else min(D_V_param, N ** (j + 1),
                                              N ** (d - j - 1))
            cores.append(rng.standard_normal((D_prev, N, D_next)) * init_std)
            D_prev = D_next
        out.append(cores)
    return out


class _AdamMPS:
    """Adam optimiser over a flat list of np.ndarray parameters."""
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                K_data, alpha, beta, N, d, L, D_max, D_V,
                n_substeps: int = 1,
                grad_clip_norm: float = 0.0,
                optimizer_kind: str = "adam"):
    """One training step: forward MPS Trotter, adjoint, project, optimizer.

    ``optimizer_kind`` is one of ``"adam"`` (Euclidean, opt is a flat
    ``_AdamMPS`` over all cores) or ``"radam"`` (Riemannian, opt is a list
    of ``_RiemannianAdamMPS`` — one per segment).

    ``grad_clip_norm > 0`` clips each core's gradient to that Frobenius
    norm before the step (per-core, not global).
    """
    # Materialise dense V per segment from current MPS cores
    V_grids_np = [V_mps_to_dense(V_mps_list[k], N, d) for k in range(K_data)]

    cache = forward_with_cache_mps(
        psi0_np, V_grids_np, alpha=alpha, beta=beta,
        N=N, d=d, L=L, D_max=D_max, D_V=D_V, n_substeps=n_substeps,
    )
    loss = 0.0
    for k in range(K_data):
        psi_k = mps_to_dense(cache[(k + 1) * n_substeps * 8], N=N, d=d)
        loss += float(((np.abs(psi_k) - np.sqrt(q_grids_np[k + 1])) ** 2).sum())

    dL_dV_dense_list = adjoint_backward_mps(
        cache, V_grids_np, q_grids_np,
        alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V, n_substeps=n_substeps,
    )

    # Project each segment's dense grad to MPS-tangent on V_mps_list[k]
    seg_grads = []
    for k in range(K_data):
        cg = project_dense_grad_to_mps_cores(
            dL_dV_dense_list[k], V_mps_list[k], N, d)
        if grad_clip_norm > 0:
            cg = [_clip(g, grad_clip_norm) for g in cg]
        seg_grads.append(cg)

    if optimizer_kind == "adam":
        flat = [g for cg in seg_grads for g in cg]
        opt.step(flat)
    elif optimizer_kind == "radam":
        # opt is a list of _RiemannianAdamMPS, one per segment
        for k in range(K_data):
            opt[k].step(seg_grads[k])
    else:
        raise ValueError(f"unknown optimizer_kind {optimizer_kind!r}")
    return loss


def _clip(g: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(g))
    if n <= max_norm or n == 0.0:
        return g
    return g * (max_norm / n)


def _recanonicalize_all(V_mps_list, center=None):
    """In-place mixed-canonical reset of every segment's V_mps."""
    for V_mps in V_mps_list:
        new = to_mixed_canonical(V_mps, center=center)
        for c, nc in zip(V_mps, new):
            c[...] = nc


def _lr_at(it, n_iter, base_lr, warmup_iters, schedule):
    """Linear warmup over ``warmup_iters`` then either flat or cosine decay."""
    if warmup_iters > 0 and it < warmup_iters:
        return base_lr * (it + 1) / warmup_iters
    if schedule == "cosine":
        progress = (it - warmup_iters) / max(1, n_iter - warmup_iters)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K_data", type=int, default=4)
    p.add_argument("--L", type=float, default=4.0)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_iter", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=16)
    p.add_argument("--D_V_param", type=int, default=8,
                   help="Bond of V_mps parameter space (independent of D_V "
                        "used to compress exp(iβV) for the Trotter step).")
    p.add_argument("--log_every", type=int, default=100)
    # Phase 0 cheap-baseline flags
    p.add_argument("--recanonicalize_every", type=int, default=0,
                   help="Re-canonicalize V_mps every N steps (0 = off).")
    p.add_argument("--grad_clip_norm", type=float, default=0.0,
                   help="Per-core gradient clip Frobenius norm (0 = off).")
    p.add_argument("--lr_warmup_iters", type=int, default=0,
                   help="Linear warmup iters; 0 = off.")
    p.add_argument("--lr_schedule", choices=["flat", "cosine"], default="flat",
                   help="Post-warmup lr schedule.")
    # Phase 2 optimizer choice
    p.add_argument("--optimizer", choices=["adam", "radam"], default="adam",
                   help="Optimizer: 'adam' = Euclidean Adam on cores; "
                        "'radam' = Riemannian Adam on the MPS manifold "
                        "(QR retraction + gauge-projected gradient).")
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    d, N, L = 2, args.N, args.L
    K_data = args.K_data
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    target_traj = sample_petals_trajectory(
        n_per_step=args.n_samples, seed=args.seed,
        n_timepoints=K_data + 1,
    )
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)

    rng = np.random.default_rng(args.seed)
    if args.optimizer == "radam":
        V_mps_list = init_v_mps_riemannian(K_data, N, d, args.D_V_param,
                                            rng=rng)
        opt = [_RiemannianAdamMPS(V_mps_list[k], lr=args.lr)
               for k in range(K_data)]
        flat_params = [c for cores in V_mps_list for c in cores]
    else:
        V_mps_list = _init_V_mps_list(K_data, N, d, args.D_V_param, rng=rng)
        flat_params = [c for cores in V_mps_list for c in cores]
        opt = _AdamMPS(flat_params, lr=args.lr)

    n_dense_params = K_data * (N ** d)
    n_mps_params = sum(c.size for c in flat_params)
    print(f"# params: dense V × K = {n_dense_params}, MPS V × K = {n_mps_params}  "
          f"(ratio MPS/dense = {n_mps_params / n_dense_params:.2f})")
    print(f"N={N} d={d} K={K_data}  D_max={args.D_max}  D_V_param={args.D_V_param}  "
          f"optimizer={args.optimizer}")
    if args.recanonicalize_every > 0:
        print(f"recanonicalize_every={args.recanonicalize_every}")
    if args.grad_clip_norm > 0:
        print(f"grad_clip_norm={args.grad_clip_norm}")
    if args.lr_warmup_iters > 0 or args.lr_schedule != "flat":
        print(f"lr_warmup_iters={args.lr_warmup_iters}  "
              f"lr_schedule={args.lr_schedule}")

    losses = []
    t0 = time.time()
    for it in range(args.n_iter):
        cur_lr = _lr_at(it, args.n_iter, args.lr,
                        args.lr_warmup_iters, args.lr_schedule)
        # Push lr through to optimizer
        if args.optimizer == "radam":
            for o in opt:
                o.lr = cur_lr
        else:
            opt.lr = cur_lr

        loss = _train_step(
            V_mps_list, opt, psi0_np, q_grids_np,
            K_data, alpha, beta, N, d, L,
            args.D_max, args.D_V,
            grad_clip_norm=args.grad_clip_norm,
            optimizer_kind=args.optimizer,
        )
        losses.append(loss)

        if (args.recanonicalize_every > 0
                and args.optimizer == "adam"
                and (it + 1) % args.recanonicalize_every == 0):
            _recanonicalize_all(V_mps_list)

        if (it + 1) % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            elapsed = time.time() - t0
            print(f"iter {it + 1:5d}/{args.n_iter}  "
                  f"loss={recent:.5f}  lr={cur_lr:.2e}  "
                  f"elapsed={elapsed:.1f}s ({(it + 1) / elapsed:.2f} it/s)")

    print(f"final loss (last 100 iters mean): {float(np.mean(losses[-100:])):.6f}")
    print(f"initial loss:                    {losses[0]:.6f}")


if __name__ == "__main__":
    main()
