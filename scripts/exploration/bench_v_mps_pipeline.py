"""Two-part benchmark of the end-to-end MPS Path-3 pipeline.

Part 1 — W₂ leaderboard at each timepoint, V_mps trained vs the
existing autograd-MLP Trotter-loss checkpoint. Both pipelines are
evaluated by running the MPS forward to get ψ at each snapshot, then
drawing 200 samples per snapshot via autoregressive sampling and
computing W₂@200 (3-repeat mean) against ``target_traj[k]``.

Part 2 — D_V_param sweep at N=16 K=4 (1500 iters each). Reports final
loss and parameter count per ``D_V_param ∈ {2, 4, 8, 16, 32}``. The
"natural" Schmidt bond for a smooth 2D petal V is small; we expect
loss to plateau quickly with D.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import forward_with_cache_mps  # noqa: E402
from train_v_mps_end_to_end import (  # noqa: E402
    _AdamMPS,
    _init_V_mps_list,
    _train_step,
    bin_samples_to_density,
)
from v_mps_parameterization import V_mps_to_dense  # noqa: E402

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.mps.core import sample_mps_indices


def _autoreg_sample(mps, n, N, L, rng):
    from tnwf.mps.core import right_canonicalize
    rc = right_canonicalize(mps)
    idx = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, idx.shape)
    return ((idx + jitter) * dx).astype(np.float64)


def _eval_pipeline_w2(V_grids_np, psi0_np, target_traj_world, N, d, L,
                       K_data, alpha, beta, D_max, D_V, n_samples=200, seed=0):
    """Run forward MPS Trotter with given V grids; sample at each snapshot;
    return per-snapshot W₂."""
    cache = forward_with_cache_mps(
        psi0_np, V_grids_np,
        alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V,
    )
    rng = np.random.default_rng(seed)
    w2_per_step = []
    for k in range(K_data + 1):
        mps_k = cache[k * 8]
        samples = _autoreg_sample(mps_k, n=n_samples, N=N, L=L, rng=rng)
        target_k = (target_traj_world[k]).astype(np.float64)
        w2, _ = wasserstein_2_subsampled(
            samples, target_k, n_sub=min(200, samples.shape[0]),
            n_repeats=3, seed=seed)
        w2_per_step.append(float(w2))
    return w2_per_step


def part1_w2_compare():
    """Compare V_mps-trained vs MLP-autograd-trained V_t on full WF pipeline."""
    print("\n" + "=" * 70)
    print("PART 1 — W₂ comparison: V_mps vs MLP-autograd Trotter-loss V_t")
    print("=" * 70)

    np.random.seed(0); torch.manual_seed(0)
    d, N, L = 2, 16, 4.0
    K_data = 4
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max, D_V = 16, 16

    target_traj = sample_petals_trajectory(n_per_step=2000, seed=0,
                                            n_timepoints=K_data + 1)
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)
    target_world = target_traj + L / 2.0

    # --- A. Train V_mps end-to-end (2500 iters at D=8)
    print("\n[A] Training V_mps end-to-end (D_V_param=8, 2500 iters)...")
    rng = np.random.default_rng(0)
    V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param=8, rng=rng)
    flat = [c for cores in V_mps_list for c in cores]
    opt = _AdamMPS(flat, lr=1e-2)
    losses_A = []
    t0 = time.time()
    for it in range(2500):
        loss = _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                           K_data, alpha, beta, N, d, L, D_max, D_V)
        losses_A.append(loss)
    print(f"    final loss: {np.mean(losses_A[-100:]):.5f}  "
          f"({time.time() - t0:.1f}s, {2500 / (time.time() - t0):.1f} it/s)")
    V_grids_A = [V_mps_to_dense(V_mps_list[k], N, d) for k in range(K_data)]
    w2_A = _eval_pipeline_w2(V_grids_A, psi0_np, target_world,
                              N, d, L, K_data, alpha, beta, D_max, D_V)

    # --- B. Load MLP-autograd-trained V_t (the existing canonical Trotter-loss ckpt)
    print("\n[B] Loading existing MLP-autograd Trotter-loss V_t...")
    ckpt_path = "data/petals_2d/jam_trotter_loss/seed0.pt"
    try:
        ckpt = torch.load(ckpt_path, weights_only=False)
        cfg = ckpt["config"]
        model = ScalarPotentialMLP(
            d=d, hidden=cfg.get("hidden", 256),
            time_embed_dim=cfg.get("time_embed_dim", 64),
            L=L, n_layers=cfg.get("n_layers", 5),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        # Evaluate V on grid at segment midpoints
        from adjoint_trotter import make_grid_centred
        grid_centred = make_grid_centred(N, d, L, "cpu").to(torch.float32)
        V_grids_B = []
        with torch.no_grad():
            for k in range(K_data):
                t_mid = float((k + 0.5) / K_data)
                t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                                      dtype=torch.float32)
                V_grids_B.append(model(grid_centred, t_tensor)
                                 .reshape(-1).to(torch.float64).numpy())
        w2_B = _eval_pipeline_w2(V_grids_B, psi0_np, target_world,
                                  N, d, L, K_data, alpha, beta, D_max, D_V)
    except FileNotFoundError:
        print(f"    {ckpt_path} not found — skipping")
        w2_B = None

    print(f"\n{'snapshot':>10}  {'W₂  V_mps (A)':>14}  {'W₂  MLP (B)':>14}")
    for k in range(K_data + 1):
        b = f"{w2_B[k]:.4f}" if w2_B is not None else "—"
        print(f"{f't = {k / K_data:.2f}':>10}  {w2_A[k]:>14.4f}  {b:>14}")


def part2_D_sweep():
    """How loss varies with V_mps bond at fixed problem size."""
    print("\n" + "=" * 70)
    print("PART 2 — D_V_param sweep at N=16 d=2 K=4 (1500 iters each)")
    print("=" * 70)

    np.random.seed(0); torch.manual_seed(0)
    d, N, L = 2, 16, 4.0
    K_data = 4
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max, D_V = 16, 16

    target_traj = sample_petals_trajectory(n_per_step=2000, seed=0,
                                            n_timepoints=K_data + 1)
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)

    print(f"\n{'D_V_param':>10}  {'n_params':>10}  {'final loss':>14}  "
          f"{'iters/s':>10}  {'wall (s)':>10}")
    for D_V_param in [2, 4, 8, 16, 32]:
        rng = np.random.default_rng(0)
        V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param=D_V_param,
                                       rng=rng)
        flat = [c for cores in V_mps_list for c in cores]
        n_params = sum(c.size for c in flat)
        opt = _AdamMPS(flat, lr=1e-2)
        losses = []
        t0 = time.time()
        for it in range(1500):
            loss = _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                               K_data, alpha, beta, N, d, L, D_max, D_V)
            losses.append(loss)
        wall = time.time() - t0
        final = float(np.mean(losses[-100:]))
        print(f"{D_V_param:>10d}  {n_params:>10d}  {final:>14.5f}  "
              f"{1500 / wall:>10.1f}  {wall:>10.1f}")


if __name__ == "__main__":
    part1_w2_compare()
    part2_D_sweep()
