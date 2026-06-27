"""Path-3 proof of concept: train V_t end-to-end through a differentiable
Trotter pipeline so the wavefunction-flow `|ψ_{t_k}(V)|²` matches the bio
data marginals `q_{t_k}`.

Loss (natural WF-paper objective):
  L(V) = Σ_k ‖|ψ_{t_k}(V)| − √q_{t_k}‖₂²  + λ·‖V‖₂² regularizer

where ψ evolves under the same 8-step product formula our pipeline uses:
  ψ_{t+Δt} = e^{iβV} e^{iαK} e^{-iβV} e^{-iαK} e^{-iβV} e^{-iαK} e^{iβV} e^{iαK} · ψ_t
with αβ = Δt/2. K is the pseudospectral kinetic operator on the periodic
[0, L)^d torus; V is the trained scalar potential evaluated on the
N^d grid at the segment midpoint t_{k+1/2}.

Dense torch tensors throughout — appropriate for small grids
(petals N=16 d=2 → 256 cells). Initialises ψ_0 = √q̂_0 where q̂_0 is
the histogram of the bio data's first snapshot, then evolves under
the trained V and minimises the snapshot mismatch via Adam.

Writes the trained model to data/petals_2d/jam_trotter_loss/seed{seed}.pt
with the same ScalarPotentialMLP checkpoint format as the other JAM
training paths, so the resulting V_t plugs into the downstream tnWF
pipeline (Dense, TDVP-1, TDVP-2, …) via the existing make_V_fn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP


def make_grid_centred(N: int, d: int, L: float, device) -> torch.Tensor:
    """(N^d, d) tensor of centred grid points covering [-L/2, L/2)^d."""
    lin = torch.linspace(0.0, L, N + 1, device=device)[:-1] - L / 2.0
    grids = torch.meshgrid(*([lin] * d), indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, d)


def make_kinetic_eigenvalues(N: int, d: int, L: float, device) -> torch.Tensor:
    """``(N^d,)`` eigenvalues of the pseudospectral kinetic operator ½p²."""
    # p_j = (2π/L)·(j − N/2)
    p = (2.0 * np.pi / L) * (torch.arange(N, device=device) - N // 2).float()
    p2 = p * p
    # K_total = ½ Σ_j p_j², separable in coordinates
    if d == 1:
        K = 0.5 * p2
    else:
        # Build the d-D sum via broadcasting
        K = torch.zeros((N,) * d, device=device)
        for j in range(d):
            shape = [1] * d
            shape[j] = N
            K = K + 0.5 * p2.reshape(shape)
    return K.reshape(-1)


def bin_samples_to_density(samples_centred: np.ndarray, N: int, d: int,
                            L: float, eps: float = 1e-6) -> np.ndarray:
    """Histogram (centred) samples onto N^d grid; return float density that
    sums to 1, with a uniform floor to avoid log-zero."""
    edges = [np.linspace(-L / 2.0, L / 2.0, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    return (counts / counts.sum()).reshape(-1)


_SIGN_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _sign_mask(N: int, d: int, device) -> torch.Tensor:
    """(-1)^(j1+j2+...) reshaped to (N^d,). Reproduces apply_K_step's mask."""
    key = (N, d, str(device))
    if key in _SIGN_MASK_CACHE:
        return _SIGN_MASK_CACHE[key]
    g = torch.zeros((N,) * d, device=device)
    for j in range(d):
        shape = [1] * d
        shape[j] = N
        idx = torch.arange(N, device=device).reshape(shape)
        g = g + idx
    mask = ((-1.0) ** g).reshape(-1).to(torch.complex128)
    _SIGN_MASK_CACHE[key] = mask
    return mask


def apply_K_step(psi: torch.Tensor, alpha: float, N: int, d: int,
                 K_eigs: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """exp(iαK)·ψ via FFT with the same sign-mask convention as the np code."""
    psi_nd = (psi * sign).reshape((N,) * d)
    psi_fft = torch.fft.fftn(psi_nd).reshape(-1)
    psi_fft = psi_fft * torch.exp(1j * alpha * K_eigs)
    psi_nd = torch.fft.ifftn(psi_fft.reshape((N,) * d)).reshape(-1)
    return psi_nd * sign


def apply_V_step(psi: torch.Tensor, beta: float,
                 V_grid: torch.Tensor) -> torch.Tensor:
    """Pointwise e^{iβV}·ψ."""
    return psi * torch.exp(1j * beta * V_grid)


def product_formula_step(psi, V_grid, alpha, beta, N, d, K_eigs, sign):
    """8-step BCH product ≈ exp([K, V] Δt) with αβ = Δt/2."""
    psi = apply_K_step(psi, alpha, N, d, K_eigs, sign)
    psi = apply_V_step(psi, beta, V_grid)
    psi = apply_K_step(psi, -alpha, N, d, K_eigs, sign)
    psi = apply_V_step(psi, -beta, V_grid)
    psi = apply_K_step(psi, -alpha, N, d, K_eigs, sign)
    psi = apply_V_step(psi, -beta, V_grid)
    psi = apply_K_step(psi, alpha, N, d, K_eigs, sign)
    psi = apply_V_step(psi, beta, V_grid)
    return psi


def trotter_coefficients(delta_t: float, N: int, d: int, L: float):
    alpha = (L / (np.pi * N)) * np.sqrt(delta_t / d)
    beta = (np.pi * N / (2.0 * L)) * np.sqrt(d * delta_t)
    return float(alpha), float(beta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="petals_2d")
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
    p.add_argument("--reg", type=float, default=0.0,
                   help="L2 regulariser on V values, lambda·‖V‖².")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--out_path", default="data/petals_2d/jam_trotter_loss/seed0.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"      # small grid, dense — CPU fine
    d, N, L = 2, args.N, args.L
    K_data = args.K_data
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    # Bio data target snapshots (centred frame, then bin to grid density)
    target_traj = sample_petals_trajectory(
        n_per_step=args.n_samples, seed=args.seed,
        n_timepoints=K_data + 1,
    )                                                          # (K+1, n, d)
    q_grids = []
    for k in range(K_data + 1):
        q_k = bin_samples_to_density(target_traj[k], N=N, d=d, L=L)
        q_grids.append(torch.from_numpy(q_k).to(torch.float64).to(device))
    psi_0 = torch.sqrt(q_grids[0]).to(torch.complex128)
    psi_0 = psi_0 / torch.norm(psi_0)

    # Grid for V evaluation (centred frame, what the MLP expects)
    grid_centred = make_grid_centred(N, d, L, device).to(torch.float32)
    K_eigs = make_kinetic_eigenvalues(N, d, L, device).to(torch.complex128)
    sign = _sign_mask(N, d, device)

    # Model — same ScalarPotentialMLP architecture as other JAM checkpoints
    model = ScalarPotentialMLP(
        d=d, hidden=args.hidden, time_embed_dim=args.time_embed_dim,
        L=L, n_layers=args.n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    for it in range(args.n_iter):
        opt.zero_grad()

        psi = psi_0
        loss = torch.zeros((), dtype=torch.float64, device=device)
        # Forward through K_data Trotter segments
        for k in range(K_data):
            t_mid = float((k + 0.5) / K_data)
            t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                                  device=device, dtype=torch.float32)
            V_grid = model(grid_centred, t_tensor).reshape(-1).to(torch.float64)
            psi = product_formula_step(psi, V_grid, alpha, beta, N, d,
                                       K_eigs, sign)
            rho = (psi.conj() * psi).real                       # (N^d,) probability
            # WF-paper natural loss: ‖|ψ| − √q‖₂² on amplitudes
            amp_pred = torch.sqrt(rho.clamp(min=1e-20))
            amp_targ = torch.sqrt(q_grids[k + 1])
            loss = loss + ((amp_pred - amp_targ) ** 2).sum()

        if args.reg > 0:
            loss = loss + args.reg * (V_grid ** 2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(loss.item()))

        if (it + 1) % args.log_every == 0:
            recent = float(np.mean(losses[-args.log_every:]))
            print(f"[trotter-loss seed={args.seed}] iter {it + 1}/{args.n_iter}  "
                  f"loss={recent:.6f}")

    # Save checkpoint in the same format as other JAM training paths
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "d": d, "L": L,
            "hidden": args.hidden, "n_layers": args.n_layers,
            "time_embed_dim": args.time_embed_dim,
            "dataset": args.dataset, "seed": args.seed,
            "kind": "trajectory", "trajectory_K": K_data,
            "loss_name": "trotter_pipeline",
            "ot_coupling": False, "global_ot": False,
        },
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
    }
    torch.save(ckpt, out_path)
    print(f"wrote {out_path}  final_loss={ckpt['final_loss']:.6f}")


if __name__ == "__main__":
    main()
