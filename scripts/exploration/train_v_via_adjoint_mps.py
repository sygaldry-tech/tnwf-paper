"""Path-3 training using the MPS adjoint pipeline.

Identical interface to ``train_v_via_adjoint.py``, but the forward
Trotter chain runs on an MPS-encoded ψ and the backward pass uses
``adjoint_trotter_mps.adjoint_backward_mps``. V is still parameterised
by ``ScalarPotentialMLP`` and evaluated densely on the grid — that piece
moves to MPS in Phase 3b. The win here is memory: ψ never materialises
as a dense ``N^d`` tensor inside the chain, only at the K+1 snapshot
points where the Hellinger loss is computed.

Validation entrypoint: ``--mode validate`` runs the MPS-adjoint and the
dense-autograd training in lockstep with matching seeds and prints
loss + parameter drift after each Adam step. Bit-exact match isn't
expected (TT-cross tolerance ~1e-12, SVD truncation at D_max), but
losses should agree to ~1e-6 relative.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import (  # noqa: E402
    SUBSTEP_PLAN,
    _sign_mask,
    apply_K_step,
    apply_V_step,
    make_grid_centred,
    make_kinetic_eigenvalues,
    trotter_coefficients,
)
from adjoint_trotter_mps import (  # noqa: E402
    adjoint_backward_mps,
    forward_with_cache_mps,
)

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP
from tnwf.mps.core import mps_to_dense


def bin_samples_to_density(samples_centred, N, d, L, eps: float = 1e-6):
    edges = [np.linspace(-L / 2.0, L / 2.0, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    return (counts / counts.sum()).reshape(-1)


def _eval_V_grids(model, grid_centred, K_data, device):
    V_grids = []
    for k in range(K_data):
        t_mid = float((k + 0.5) / K_data)
        t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                              device=device, dtype=torch.float32)
        V = model(grid_centred, t_tensor).reshape(-1).to(torch.float64)
        V_grids.append(V)
    return V_grids


def _train_step_adjoint_mps(model, opt, psi0_np, q_grids_np, grid_centred,
                             K_data, alpha, beta, N, d, L, D_max, D_V,
                             reg: float = 0.0):
    """One Adam step using the MPS adjoint."""
    opt.zero_grad()
    V_grids_grad = _eval_V_grids(model, grid_centred, K_data, "cpu")
    V_grids_np = [V.detach().cpu().numpy() for V in V_grids_grad]

    # Forward + adjoint in numpy/MPS
    cache = forward_with_cache_mps(
        psi0_np, V_grids_np,
        alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V,
    )
    loss = 0.0
    for k in range(K_data):
        psi_k_dense = mps_to_dense(cache[(k + 1) * 8], N=N, d=d)
        loss += float(((np.abs(psi_k_dense) - np.sqrt(q_grids_np[k + 1])) ** 2).sum())
    if reg > 0:
        loss += reg * float((V_grids_np[-1] ** 2).mean())

    dL_dV_np = adjoint_backward_mps(
        cache, V_grids_np, q_grids_np,
        alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V,
    )
    if reg > 0:
        dL_dV_np[-1] = dL_dV_np[-1] + (2.0 * reg / V_grids_np[-1].size) * V_grids_np[-1]

    # Chain into MLP: stack V leaves, supply grad from adjoint
    flat_V = torch.cat(V_grids_grad)
    flat_g = torch.from_numpy(np.concatenate(dL_dV_np)).to(flat_V.dtype)
    torch.autograd.backward([flat_V], grad_tensors=[flat_g])

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    return loss


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
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)

    grid_centred = make_grid_centred(N, d, L, device).to(torch.float32)

    model = ScalarPotentialMLP(
        d=d, hidden=args.hidden, time_embed_dim=args.time_embed_dim,
        L=L, n_layers=args.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    return dict(d=d, N=N, L=L, K_data=K_data, alpha=alpha, beta=beta,
                psi0_np=psi0_np, q_grids_np=q_grids_np,
                grid_centred=grid_centred, model=model, opt=opt)


def _train(args):
    ctx = _setup(args)
    losses = []
    t0 = time.time()
    for it in range(args.n_iter):
        loss = _train_step_adjoint_mps(
            ctx["model"], ctx["opt"], ctx["psi0_np"], ctx["q_grids_np"],
            ctx["grid_centred"], ctx["K_data"], ctx["alpha"], ctx["beta"],
            ctx["N"], ctx["d"], ctx["L"], args.D_max, args.D_V,
            reg=args.reg,
        )
        losses.append(loss)
        if (it + 1) % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            elapsed = time.time() - t0
            print(f"[adjoint_mps seed={args.seed}] iter {it + 1}/{args.n_iter}  "
                  f"loss={recent:.6f}  elapsed={elapsed:.1f}s "
                  f"({(it + 1) / elapsed:.1f} it/s)")

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
            "loss_name": "trotter_pipeline_adjoint_mps",
            "D_max": args.D_max, "D_V": args.D_V,
            "ot_coupling": False, "global_ot": False,
        },
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
    }
    torch.save(ckpt, out_path)
    print(f"wrote {out_path}  final_loss={ckpt['final_loss']:.6f}")


def _validate(args):
    """Compare MPS-adjoint training vs dense-autograd in lockstep."""
    from train_v_via_adjoint import _eval_V_grids as _eval_V_grids_ag  # type: ignore

    ctx_mps = _setup(args)
    ctx_ag = _setup(args)
    ctx_ag["model"].load_state_dict(ctx_mps["model"].state_dict())
    ctx_ag["opt"] = torch.optim.Adam(ctx_ag["model"].parameters(), lr=args.lr)

    K_eigs_t = make_kinetic_eigenvalues(
        ctx_ag["N"], ctx_ag["d"], ctx_ag["L"], "cpu").to(torch.complex128)
    sign_t = _sign_mask(ctx_ag["N"], ctx_ag["d"], "cpu")
    psi0_t = torch.from_numpy(ctx_ag["psi0_np"])
    q_grids_t = [torch.from_numpy(q) for q in ctx_ag["q_grids_np"]]

    n_check = min(args.n_iter, 8)
    for it in range(n_check):
        # ----- autograd (dense)
        ctx_ag["opt"].zero_grad()
        V_grids_ag = _eval_V_grids_ag(ctx_ag["model"], ctx_ag["grid_centred"],
                                      ctx_ag["K_data"], psi0_t.device)
        psi = psi0_t
        loss_ag = torch.zeros((), dtype=torch.float64)
        for k in range(ctx_ag["K_data"]):
            for kind, s in SUBSTEP_PLAN:
                if kind == "K":
                    psi = apply_K_step(psi, s * ctx_ag["alpha"],
                                       ctx_ag["N"], ctx_ag["d"], K_eigs_t, sign_t)
                else:
                    psi = apply_V_step(psi, s * ctx_ag["beta"], V_grids_ag[k])
            loss_ag = loss_ag + ((psi.abs() - q_grids_t[k + 1].sqrt()) ** 2).sum()
        loss_ag.backward()
        torch.nn.utils.clip_grad_norm_(ctx_ag["model"].parameters(), max_norm=1.0)
        ctx_ag["opt"].step()

        # ----- MPS adjoint
        loss_mps = _train_step_adjoint_mps(
            ctx_mps["model"], ctx_mps["opt"], ctx_mps["psi0_np"],
            ctx_mps["q_grids_np"], ctx_mps["grid_centred"],
            ctx_mps["K_data"], ctx_mps["alpha"], ctx_mps["beta"],
            ctx_mps["N"], ctx_mps["d"], ctx_mps["L"],
            args.D_max, args.D_V,
        )

        # Param drift
        with torch.no_grad():
            max_drift = max(
                (p_a - p_b).abs().max().item()
                for p_a, p_b in zip(ctx_ag["model"].parameters(),
                                    ctx_mps["model"].parameters())
            )
        print(f"iter {it + 1:2d}  loss(autograd)={float(loss_ag.item()):.10f}  "
              f"loss(MPS-adj)={loss_mps:.10f}  "
              f"Δloss={loss_mps - float(loss_ag.item()):+.2e}  "
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
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=16)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--out_path",
                   default="data/petals_2d/jam_trotter_loss_adjoint_mps/seed0.pt")
    args = p.parse_args()
    if args.mode == "validate":
        _validate(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
