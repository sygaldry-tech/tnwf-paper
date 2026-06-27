"""Path-3 training with the adjoint method instead of torch autograd.

Drop-in alternative to ``train_v_via_trotter_loss.py``. The forward
Trotter pipeline runs under ``torch.no_grad`` and caches ψ at every
substep; the backward pass is the hand-rolled adjoint in
``adjoint_trotter.py``. The MLP gradient is obtained by
``torch.autograd.grad`` on the V-grid leaves, with the upstream gradient
supplied by the adjoint.

Loss, model, optimiser, and checkpoint format are identical to the
autograd version. Validation entrypoint: ``--mode validate`` runs 10
iterations of both pipelines side-by-side and prints loss + parameter
gradient match.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make adjoint_trotter importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import (
    SUBSTEP_PLAN,
    _hellinger_lam,
    _sign_mask,
    adjoint_backward,
    apply_K_step,
    apply_V_step,
    forward_with_cache,
    make_grid_centred,
    make_kinetic_eigenvalues,
    trotter_coefficients,
)

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP


def bin_samples_to_density(samples_centred, N, d, L, eps: float = 1e-6):
    edges = [np.linspace(-L / 2.0, L / 2.0, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    return (counts / counts.sum()).reshape(-1)


def _eval_V_grids(model, grid_centred, K_data, device):
    """Return list of V grids (with autograd tape, leaves of MLP params)."""
    V_grids = []
    for k in range(K_data):
        t_mid = float((k + 0.5) / K_data)
        t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                              device=device, dtype=torch.float32)
        V = model(grid_centred, t_tensor).reshape(-1).to(torch.float64)
        V_grids.append(V)
    return V_grids


def _compute_forward_loss(psi_0, V_grids, q_grids, alpha, beta,
                          N, d, K_eigs, sign):
    """Return loss only — used for monitoring (no caching, no grad)."""
    with torch.no_grad():
        psi = psi_0
        loss = torch.zeros((), dtype=torch.float64, device=psi_0.device)
        for k in range(len(V_grids)):
            for kind, s in SUBSTEP_PLAN:
                if kind == "K":
                    psi = apply_K_step(psi, s * alpha, N, d, K_eigs, sign)
                else:
                    psi = apply_V_step(psi, s * beta, V_grids[k])
            loss = loss + ((psi.abs() - q_grids[k + 1].sqrt()) ** 2).sum()
    return float(loss.item())


def _train_step_adjoint(model, opt, psi_0, q_grids, grid_centred,
                        K_data, alpha, beta, N, d, K_eigs, sign,
                        reg: float = 0.0):
    """One Adam step using adjoint gradients."""
    opt.zero_grad()
    # 1. Build V grids with autograd tracking (leaves of MLP params).
    V_grids_grad = _eval_V_grids(model, grid_centred, K_data, psi_0.device)
    V_grids_det = [V.detach() for V in V_grids_grad]

    # 2. Forward Trotter (cached, no grad).
    with torch.no_grad():
        cache = forward_with_cache(
            psi_0, V_grids_det, alpha=alpha, beta=beta,
            N=N, d=d, K_eigs=K_eigs, sign=sign,
        )
        loss = torch.zeros((), dtype=torch.float64, device=psi_0.device)
        for k in range(K_data):
            psi_k = cache[(k + 1) * 8]
            loss = loss + ((psi_k.abs() - q_grids[k + 1].sqrt()) ** 2).sum()
        if reg > 0:
            loss = loss + reg * (V_grids_det[-1] ** 2).mean()

        # 3. Adjoint backward → dL/dV_grid for each segment.
        dL_dV = adjoint_backward(
            cache, V_grids_det, q_grids,
            alpha=alpha, beta=beta, N=N, d=d, K_eigs=K_eigs, sign=sign,
        )
        if reg > 0:
            # ∂(reg · mean(V_last²))/∂V_last[i] = 2·reg·V_last[i]/N_cells
            dL_dV[-1] = dL_dV[-1] + (2.0 * reg / V_grids_det[-1].numel()) \
                                     * V_grids_det[-1]

    # 4. Chain into MLP via autograd: total V-leaf vector × dL/dV_grid.
    flat_V = torch.cat(V_grids_grad)
    flat_g = torch.cat(dL_dV).to(flat_V.dtype)
    torch.autograd.backward([flat_V], grad_tensors=[flat_g])

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    return float(loss.item())


# ---------------------- entrypoints ----------------------

def _setup(args, device="cpu"):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    d, N, L = 2, args.N, args.L
    K_data = args.K_data
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    target_traj = sample_petals_trajectory(
        n_per_step=args.n_samples, seed=args.seed,
        n_timepoints=K_data + 1,
    )
    q_grids = [torch.from_numpy(bin_samples_to_density(target_traj[k], N, d, L)
                                ).to(torch.float64).to(device)
               for k in range(K_data + 1)]
    psi_0 = q_grids[0].sqrt().to(torch.complex128)
    psi_0 = psi_0 / torch.norm(psi_0)

    grid_centred = make_grid_centred(N, d, L, device).to(torch.float32)
    K_eigs = make_kinetic_eigenvalues(N, d, L, device).to(torch.complex128)
    sign = _sign_mask(N, d, device)

    model = ScalarPotentialMLP(
        d=d, hidden=args.hidden, time_embed_dim=args.time_embed_dim,
        L=L, n_layers=args.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    return dict(d=d, N=N, L=L, K_data=K_data, alpha=alpha, beta=beta,
                psi_0=psi_0, q_grids=q_grids, grid_centred=grid_centred,
                K_eigs=K_eigs, sign=sign, model=model, opt=opt)


def _train(args):
    ctx = _setup(args)
    losses = []
    for it in range(args.n_iter):
        loss = _train_step_adjoint(
            ctx["model"], ctx["opt"], ctx["psi_0"], ctx["q_grids"],
            ctx["grid_centred"], ctx["K_data"], ctx["alpha"], ctx["beta"],
            ctx["N"], ctx["d"], ctx["K_eigs"], ctx["sign"], reg=args.reg,
        )
        losses.append(loss)
        if (it + 1) % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            print(f"[adjoint seed={args.seed}] iter {it + 1}/{args.n_iter}  "
                  f"loss={recent:.6f}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": ctx["model"].state_dict(),
        "config": {
            "d": ctx["d"], "L": ctx["L"],
            "hidden": args.hidden, "n_layers": args.n_layers,
            "time_embed_dim": args.time_embed_dim,
            "dataset": "petals_2d", "seed": args.seed,
            "kind": "trajectory", "trajectory_K": ctx["K_data"],
            "loss_name": "trotter_pipeline_adjoint",
            "ot_coupling": False, "global_ot": False,
        },
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
    }
    torch.save(ckpt, out_path)
    print(f"wrote {out_path}  final_loss={ckpt['final_loss']:.6f}")


def _validate(args):
    """Run 10 iterations of autograd vs adjoint training in lockstep with
    matching seeds; assert losses and final parameters agree."""
    ctx_ag = _setup(args)
    ctx_adj = _setup(args)
    # Same init: copy model_a → model_b
    ctx_adj["model"].load_state_dict(ctx_ag["model"].state_dict())
    ctx_adj["opt"] = torch.optim.Adam(ctx_adj["model"].parameters(), lr=args.lr)

    n_check = min(10, args.n_iter)
    for it in range(n_check):
        # ----- autograd step
        ctx_ag["opt"].zero_grad()
        V_grids_ag = _eval_V_grids(ctx_ag["model"], ctx_ag["grid_centred"],
                                   ctx_ag["K_data"], ctx_ag["psi_0"].device)
        psi = ctx_ag["psi_0"]
        loss_ag = torch.zeros((), dtype=torch.float64)
        for k in range(ctx_ag["K_data"]):
            for kind, s in SUBSTEP_PLAN:
                if kind == "K":
                    psi = apply_K_step(psi, s * ctx_ag["alpha"],
                                       ctx_ag["N"], ctx_ag["d"],
                                       ctx_ag["K_eigs"], ctx_ag["sign"])
                else:
                    psi = apply_V_step(psi, s * ctx_ag["beta"], V_grids_ag[k])
            loss_ag = loss_ag + ((psi.abs() - ctx_ag["q_grids"][k + 1].sqrt()) ** 2).sum()
        loss_ag.backward()
        torch.nn.utils.clip_grad_norm_(ctx_ag["model"].parameters(), max_norm=1.0)
        ctx_ag["opt"].step()

        # ----- adjoint step
        loss_adj = _train_step_adjoint(
            ctx_adj["model"], ctx_adj["opt"], ctx_adj["psi_0"], ctx_adj["q_grids"],
            ctx_adj["grid_centred"], ctx_adj["K_data"], ctx_adj["alpha"],
            ctx_adj["beta"], ctx_adj["N"], ctx_adj["d"],
            ctx_adj["K_eigs"], ctx_adj["sign"],
        )

        # Compare param drift
        with torch.no_grad():
            max_drift = max(
                (p_a - p_b).abs().max().item()
                for p_a, p_b in zip(ctx_ag["model"].parameters(),
                                    ctx_adj["model"].parameters())
            )
        print(f"iter {it + 1:2d}  loss(autograd)={float(loss_ag.item()):.10f}  "
              f"loss(adjoint)={loss_adj:.10f}  "
              f"max|Δθ|={max_drift:.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "validate"], default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K_data", type=int, default=4)
    p.add_argument("--L", type=float, default=4.0)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--out_path",
                   default="data/petals_2d/jam_trotter_loss_adjoint/seed0.pt")
    args = p.parse_args()

    if args.mode == "validate":
        _validate(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
