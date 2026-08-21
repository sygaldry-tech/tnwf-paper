"""Train an MPS-V (tensor-train velocity potential) by JAM / action matching.

The velocity potential V_t is parameterized as a `(d+1)`-site tensor train
(`MPSScalarPotentialTimeSite`) and fit with the same self-contained losses used
for the MLP JAM trainer — no teacher model required. The resulting cores can be
fed directly to the 2-site TDVP V-step (the MPS-V "bypass"), see
`tnwf.pipelines.run_evolution.run(method="mps_v_tdvp2", ...)`.

Usage:
    python -m tnwf.mps_v.train --dataset gmm_8d --N 16 --D 8 --seed 0 \\
        --out examples/checkpoints/mps_v_gmm_d8.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tnwf.jam.scalar_potential import (action_matching_loss,
                                        jam_conservative_loss,
                                        normalize_loss_name)
from tnwf.jam.train import DATASET_DEFAULTS, sample_target
from tnwf.mps_v.model import MPSScalarPotentialTimeSite


def train(
    dataset: str,
    seed: int,
    out_path: str,
    *,
    N: int = 16,
    D: int = 8,
    N_t: int = 12,
    n_iter: int = 20_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    d: int | None = None,
    L: float | None = None,
    sigma0: float | None = None,
    n_samples: int | None = None,
    val_frac: float = 0.1,
    lr_schedule: str | None = None,
    sgdr_t0: int = 10_000,
    lr_min: float = 0.0,
    patience: int | None = None,
    eval_every: int = 500,
    grad_clip: float | None = None,
    loss_name: str = "jam",
    log_every: int = 1000,
) -> dict:
    """Train an MPS-V on an endpoint dataset and save a checkpoint to out_path.

    The checkpoint dict is `{"model", "args", "model_type"}` — the format read
    by `tnwf.mps_v.load_mps_v`. Source samples are N(0, sigma0^2 I) with sigma0
    defaulting to the width the evaluation path uses; the target GMM is in
    the centered frame

    Args beyond the model shape:
        n_samples:   size of the target pool. Defaults to the dataset's own
            value (10,000), which is thin for tail-sensitive fits: at ~4% mass
            beyond 4 sigma that is only ~400 distinct tail points, reused across
            every iteration. The published checkpoints used 200,000.
        val_frac:    fraction of the pool held out to drive early stopping. The
            trainer previously had no validation signal at all.
        lr_schedule: None (constant) or "cosine" for cosine annealing with warm
            restarts of period `sgdr_t0`.
        patience:    stop after this many consecutive evaluations without
            improvement on held-out loss. None disables early stopping.
        grad_clip:   max global grad norm, or None.

    Not restored: `unitarity_weight`, which the published checkpoints record as
    0.1. Its definition is not in this repository and guessing at a regularizer
    would be worse than omitting it, so the learned cores are centered (the V-step provider
    re-centers them to the [0, L) grid).
    """
    loss_name = normalize_loss_name(loss_name)
    cfg = dict(DATASET_DEFAULTS[dataset])
    if d is not None:
        cfg["d"] = d
    if L is not None:
        cfg["L"] = L
    d = int(cfg["d"])
    L = float(cfg["L"])
    # Width of the Gaussian source the oracle is fit against. This has to match
    # the source the flow is later driven from: a velocity potential is only
    # valid for the source it saw in training, and a mismatch costs about a
    # factor 5 in endpoint SW. It used to be hardcoded to 1.0 here while the
    # evaluation path used L/6, so anything trained with this function and run
    # through `run(method="mps_v_tdvp2")` was driven from a source it had never
    # seen. Defaults to the evaluation value and is recorded in the checkpoint.
    if sigma0 is None:
        from tnwf.pipelines.run_evolution import resolve_source_sigma
        sigma0 = resolve_source_sigma("mps_v_tdvp2", L, None)
    sigma0 = float(sigma0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    _META = {"d", "n_samples", "L", "kind", "trajectory_K", "training"}
    sample_kw = {k: v for k, v in cfg.items() if k not in _META}
    n_pool = int(n_samples if n_samples is not None else cfg["n_samples"])
    target = sample_target(dataset, n_pool, d=d, seed=seed, **sample_kw)
    target_t = torch.from_numpy(target).float().to(device)

    # Held-out split for early stopping. Without it there is nothing to stop on,
    # and the loss on the training pool keeps drifting down while the fit to the
    # tails does not improve.
    n_val = int(round(val_frac * n_pool)) if patience else 0
    if n_val:
        perm = torch.randperm(n_pool, device=device)
        val_t, target_t = target_t[perm[:n_val]], target_t[perm[n_val:]]
        # A *fixed* validation batch: the loss is stochastic in x0 and t, and
        # resampling them each evaluation swamps the signal early stopping is
        # meant to read.
        gv = torch.Generator(device="cpu").manual_seed(seed + 9973)
        vb = min(4096, n_val)
        v_idx = torch.randint(0, val_t.shape[0], (vb,), generator=gv).to(device)
        v_x1 = val_t[v_idx]
        v_x0 = (sigma0 * torch.randn(vb, d, generator=gv)).to(device)
        v_t = torch.rand(vb, 1, generator=gv).to(device)

    model = MPSScalarPotentialTimeSite(d=d, N=N, D=D, L=L, N_t=N_t).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, sgdr_t0), eta_min=lr_min)

    def _loss_on(x0, x1, t):
        if loss_name == "jam":
            return jam_conservative_loss(model, x0, x1, t)
        z = torch.zeros(x1.shape[0], 1, device=x1.device)
        return action_matching_loss(model, x0, x1, t, z, torch.ones_like(z))

    losses: list[float] = []
    val_history: list[tuple[int, float]] = []
    best_val, best_state, best_iter, stale = float("inf"), None, 0, 0
    stopped_at = n_iter
    for it in range(n_iter):
        idx = torch.randint(0, target_t.shape[0], (batch_size,), device=device)
        x1 = target_t[idx]
        x0 = sigma0 * torch.randn(batch_size, d, device=device)
        t = torch.rand(batch_size, 1, device=device)
        opt.zero_grad()
        loss = _loss_on(x0, x1, t)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if sched is not None:
            sched.step(it)
        losses.append(loss.item())

        if n_val and (it + 1) % eval_every == 0:
            with torch.enable_grad():          # the losses differentiate w.r.t. x
                v = float(_loss_on(v_x0, v_x1, v_t).item())
            val_history.append((it + 1, v))
            if v < best_val - 1e-6:
                best_val, best_iter, stale = v, it + 1, 0
                best_state = {k: t_.detach().clone()
                              for k, t_ in model.state_dict().items()}
            else:
                stale += 1
                if patience and stale >= patience:
                    stopped_at = it + 1
                    print(f"[mps_v {dataset} seed={seed}] early stop at {stopped_at}"
                          f" (best {best_val:.4f} @ {best_iter})")
                    break

        if (it + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            extra = f"  val={val_history[-1][1]:.4f}" if val_history else ""
            print(f"[mps_v {dataset} seed={seed}] iter {it + 1}/{n_iter}  "
                  f"loss={recent:.4f}{extra}")

    # Restore the best checkpoint rather than the last one -- with a warm-restart
    # schedule the final iterate can sit just after a restart spike.
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model": model.state_dict(),
        "args": {
            "d": d, "N_grid": N, "D_mps": D, "L": L, "N_t": N_t,
            "dataset": dataset, "seed": seed, "loss_name": loss_name,
            "sigma_0": sigma0,
            "n_samples": n_pool, "n_iter": n_iter, "batch_size": batch_size,
            "lr": lr, "lr_schedule": lr_schedule, "sgdr_t0": sgdr_t0,
            "lr_min": lr_min, "patience": patience, "val_frac": val_frac,
            "grad_clip": grad_clip, "stopped_at": stopped_at,
            "best_val": best_val if best_state is not None else None,
            "std": cfg.get("std"), "scale": cfg.get("scale"),
            "arrangement": "orthogonal", "qtt": False,
        },
        "model_type": "mps_potential_time_site",
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
        "val_history": val_history,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return ckpt


def _main() -> None:
    p = argparse.ArgumentParser(description="Train an MPS-V velocity potential.")
    p.add_argument("--dataset", default="gmm_8d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=16, help="grid points per dimension")
    p.add_argument("--D", type=int, default=8, help="MPS bond dimension")
    p.add_argument("--N_t", type=int, default=12, help="time-axis discretization")
    p.add_argument("--n_iter", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--L", type=float, default=None)
    p.add_argument("--n_samples", type=int, default=None,
                   help="target pool size; default is the dataset's 10000, "
                        "which gives only ~400 distinct >4-sigma points")
    p.add_argument("--lr_schedule", choices=["cosine"], default=None)
    p.add_argument("--sgdr_t0", type=int, default=10_000)
    p.add_argument("--patience", type=int, default=None,
                   help="early-stopping patience in evaluations; enables the "
                        "held-out split")
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--sigma0", type=float, default=None,
                   help="Gaussian source width; defaults to the value the "
                        "evaluation path uses (L/6), so training and the flow "
                        "cannot drift apart.")
    # "cfm" is the legacy spelling of "jam"; both select jam_conservative_loss.
    p.add_argument("--loss_name", choices=["jam", "cfm", "am"], default="jam")
    args = p.parse_args()
    train(
        dataset=args.dataset, seed=args.seed, out_path=args.out,
        N=args.N, D=args.D, N_t=args.N_t, n_iter=args.n_iter,
        batch_size=args.batch_size, lr=args.lr, d=args.d, L=args.L,
        sigma0=args.sigma0, n_samples=args.n_samples,
        lr_schedule=args.lr_schedule, sgdr_t0=args.sgdr_t0,
        patience=args.patience, grad_clip=args.grad_clip,
        loss_name=args.loss_name,
    )


if __name__ == "__main__":
    _main()
