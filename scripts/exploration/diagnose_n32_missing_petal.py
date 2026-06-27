"""Diagnose the missing-petal asymmetry at N=32.

Both V_mps and MLP panels at N=32 missed the leftward petal in the
W₂@t=1 figure. Both pipelines use the same psi_0 (from binning the bio
data target at t=0) and the same MPS forward. So the asymmetry comes
from one of:

  (a) the binned q_grids[k] themselves — sampling noise breaks 4-fold
      symmetry, or
  (b) dense_to_mps(√q_0) — sequential SVD compression introduces
      symmetry-breaking ordering bias, or
  (c) the MPS forward — bond-D truncation prefers some directions.

This script prints all three:
  - q_grids[k] densities at k=0,1,2,3,4 (sum per quadrant)
  - |ψ_0_mps|² densities (sum per quadrant) after dense_to_mps round-trip
  - |ψ_k|² densities at each k (using the existing MLP V_t)

If quadrant sums for q_grids are already asymmetric → fix is more
samples / smoother target. Otherwise it's a pipeline issue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import make_grid_centred, trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import forward_with_cache_mps  # noqa: E402
from train_v_mps_end_to_end import bin_samples_to_density  # noqa: E402
from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP
from tnwf.mps.core import dense_to_mps, mps_to_dense


def quadrant_mass(rho_2d: np.ndarray, N: int) -> dict:
    """Sum of mass in each spatial quadrant (centred grid, with origin at
    cell N/2). Returns dict with TL/TR/BL/BR keys."""
    half = N // 2
    return {
        "BL (-x,-y)": float(rho_2d[:half, :half].sum()),
        "BR (+x,-y)": float(rho_2d[half:, :half].sum()),
        "TL (-x,+y)": float(rho_2d[:half, half:].sum()),
        "TR (+x,+y)": float(rho_2d[half:, half:].sum()),
    }


def arm_mass(rho_2d: np.ndarray, N: int, half_width: int = 4) -> dict:
    """Mass in a thin strip along each arm (±x, ±y from origin)."""
    half = N // 2
    band = slice(half - half_width, half + half_width)
    return {
        "+x arm": float(rho_2d[half:, band].sum()),
        "-x arm": float(rho_2d[:half, band].sum()),
        "+y arm": float(rho_2d[band, half:].sum()),
        "-y arm": float(rho_2d[band, :half].sum()),
    }


def main():
    np.random.seed(0); torch.manual_seed(0)
    d, N, L = 2, 32, 4.0
    K_data = 4
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max, D_V = 16, 16

    # --- 1. binned q_grids (target)
    target_traj = sample_petals_trajectory(n_per_step=2000, seed=0,
                                            n_timepoints=K_data + 1)
    q_grids_flat = [bin_samples_to_density(target_traj[k], N, d, L)
                    for k in range(K_data + 1)]
    print(f"=== Binned target q_grids (N={N}, n_per_step=2000, total mass=1) ===")
    for k, q in enumerate(q_grids_flat):
        q2d = q.reshape(N, N)
        arms = arm_mass(q2d, N, half_width=4)
        total_arms = sum(arms.values())
        print(f"\nt = {k / K_data:.2f}   arm bands (±4 cells around axes):")
        for label, m in arms.items():
            pct = 100 * m / total_arms if total_arms > 0 else 0
            print(f"   {label}: mass {m:.4f}  ({pct:.1f}% of arms)")

    # --- 2. |ψ_0_mps|² after dense_to_mps
    psi0_dense = np.sqrt(q_grids_flat[0]).astype(np.complex128)
    psi0_dense /= np.linalg.norm(psi0_dense)
    psi0_mps = dense_to_mps(psi0_dense, N=N, d=d, D_max=D_max)
    psi0_recon = mps_to_dense(psi0_mps, N=N, d=d)
    rho0_mps = (np.abs(psi0_recon) ** 2).reshape(N, N)
    rho0_dense = (np.abs(psi0_dense) ** 2).reshape(N, N)
    arms_mps = arm_mass(rho0_mps, N)
    arms_dense = arm_mass(rho0_dense, N)
    print(f"\n=== |ψ_0|² before vs after dense_to_mps (D={D_max}) ===")
    print(f"   |ψ_0_dense|²  arm sum: {sum(arms_dense.values()):.4f}")
    print(f"   |ψ_0_mps|²    arm sum: {sum(arms_mps.values()):.4f}  "
          f"(diff = {abs(sum(arms_dense.values()) - sum(arms_mps.values())):.2e})")

    # --- 3. |ψ_K|² from full MPS forward with MLP V_t
    ckpt = torch.load("data/petals_2d/jam_trotter_loss_N32/seed0.pt",
                      weights_only=False)
    cfg = ckpt["config"]
    model = ScalarPotentialMLP(
        d=d, hidden=cfg.get("hidden", 256),
        time_embed_dim=cfg.get("time_embed_dim", 64),
        L=L, n_layers=cfg.get("n_layers", 5),
    )
    model.load_state_dict(ckpt["model_state"]); model.eval()
    grid_centred = make_grid_centred(N, d, L, "cpu").to(torch.float32)
    V_grids_mlp = []
    with torch.no_grad():
        for k in range(K_data):
            t_mid = float((k + 0.5) / K_data)
            t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                                  dtype=torch.float32)
            V_grids_mlp.append(model(grid_centred, t_tensor).reshape(-1)
                                .to(torch.float64).numpy())
    cache = forward_with_cache_mps(
        psi0_dense, V_grids_mlp,
        alpha=alpha, beta=beta, N=N, d=d, L=L, D_max=D_max, D_V=D_V,
    )
    print(f"\n=== |ψ_k|² from MPS forward (MLP V_t, N={N}) ===")
    for k in range(K_data + 1):
        psi_k = mps_to_dense(cache[k * 8], N=N, d=d)
        rho = (np.abs(psi_k) ** 2).reshape(N, N)
        arms = arm_mass(rho, N, half_width=4)
        total = sum(arms.values())
        print(f"\nt = {k / K_data:.2f}   arm-band mass:")
        for label, m in arms.items():
            pct = 100 * m / total if total > 0 else 0
            print(f"   {label}: mass {m:.4f}  ({pct:.1f}% of arms)")


if __name__ == "__main__":
    main()
