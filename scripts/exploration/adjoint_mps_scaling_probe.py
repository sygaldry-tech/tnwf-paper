"""Scaling probe: gradient accuracy of the MPS adjoint vs bond ``D_max``,
and behaviour at higher ``d`` where dense becomes infeasible.

Two reports:

1. **Truncation effect** (fixed ``N=8`` ``d=2`` ``K_data=4``): sweep
   ``D_max ∈ {2, 4, 8, 16, 32, 64}``; for each, run MPS forward + adjoint
   and report ``‖∇_MPS − ∇_dense‖ / ‖∇_dense‖`` and the forward-loss
   delta. Reveals at what bond the MPS pipeline becomes a faithful
   stand-in for dense.

2. **Higher-d demonstration** (``d=3``, ``N=6``, dense ``N^d=216`` — still
   computable as reference): MPS adjoint with ``D_max=16`` vs dense.
   This shows the MPS adjoint generalises beyond d=2 without code
   changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import (  # noqa: E402
    _adjoint_loss_and_grad,
    _sign_mask,
    make_kinetic_eigenvalues,
    trotter_coefficients,
)
from adjoint_trotter_mps import (  # noqa: E402
    adjoint_backward_mps,
    forward_with_cache_mps,
)
from tnwf.mps.core import mps_to_dense  # noqa: E402


def _mps_loss(cache, q_grids_np, K_data, N, d):
    loss = 0.0
    for k in range(K_data):
        psi_k_dense = mps_to_dense(cache[(k + 1) * 8], N=N, d=d)
        loss += float(((np.abs(psi_k_dense) - np.sqrt(q_grids_np[k + 1])) ** 2).sum())
    return loss


def _run_case(N, d, K_data, L, D_sweep, label):
    np.random.seed(0)
    torch.manual_seed(0)
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    psi0_raw = (np.random.randn(N**d) + 1j * np.random.randn(N**d))
    psi0 = (psi0_raw / np.linalg.norm(psi0_raw)).astype(np.complex128)
    q_grids_np = []
    for _ in range(K_data + 1):
        q = np.random.rand(N**d).clip(min=1e-6)
        q_grids_np.append(q / q.sum())
    V_init = [np.random.randn(N**d) * 0.5 for _ in range(K_data)]

    # Dense reference
    K_eigs_t = make_kinetic_eigenvalues(N, d, L, "cpu").to(torch.complex128)
    sign_t = _sign_mask(N, d, "cpu")
    psi0_t = torch.from_numpy(psi0)
    q_grids_t = [torch.from_numpy(q) for q in q_grids_np]
    V_grids_t = [torch.from_numpy(V) for V in V_init]
    loss_ref, dL_dV_ref = _adjoint_loss_and_grad(
        V_grids_t, psi0_t, q_grids_t, alpha, beta, N, d, K_eigs_t, sign_t)
    grad_ref_norm = sum(g.norm().item() ** 2 for g in dL_dV_ref) ** 0.5

    print(f"\n=== {label}  (N={N}, d={d}, K_data={K_data}, dense N^d={N**d}) ===")
    print(f"reference loss = {loss_ref.item():.10f}    ‖∇_ref‖ = {grad_ref_norm:.4f}")
    print(f"{'D_max':>6}  {'loss MPS':>15}  {'Δloss':>10}  {'rel ‖∇‖':>10}  {'max |Δg|':>10}")
    print(f"{'-' * 6:>6}  {'-' * 15:>15}  {'-' * 10:>10}  {'-' * 10:>10}  {'-' * 10:>10}")
    for D_max in D_sweep:
        cache = forward_with_cache_mps(
            psi0, V_init, alpha=alpha, beta=beta, N=N, d=d, L=L,
            D_max=D_max, D_V=D_max,
        )
        loss_mps = _mps_loss(cache, q_grids_np, K_data, N, d)
        dL_dV_mps = adjoint_backward_mps(
            cache, V_init, q_grids_np,
            alpha=alpha, beta=beta, N=N, d=d, L=L,
            D_max=D_max, D_V=D_max,
        )
        # Stack and compare
        g_mps_flat = np.concatenate(dL_dV_mps)
        g_ref_flat = np.concatenate([g.numpy() for g in dL_dV_ref])
        rel = np.linalg.norm(g_mps_flat - g_ref_flat) / (
            np.linalg.norm(g_ref_flat) + 1e-30)
        absdiff = np.abs(g_mps_flat - g_ref_flat).max()
        print(f"{D_max:>6d}  {loss_mps:>15.10f}  {loss_mps - loss_ref.item():>+10.2e}  "
              f"{rel:>10.2e}  {absdiff:>10.2e}")


def main():
    # 1. Truncation effect at N=8 d=2 K=4 (dense N^d = 64; natural MPS bond = 8)
    _run_case(N=8, d=2, K_data=4, L=4.0,
              D_sweep=[2, 4, 8, 16, 32, 64],
              label="Truncation effect, d=2")

    # 2. Higher-d demonstration: d=3 N=6 K=4 (dense N^d = 216; natural bond ~36)
    _run_case(N=6, d=3, K_data=4, L=4.0,
              D_sweep=[4, 8, 16, 32, 64],
              label="Higher-d demonstration, d=3")


if __name__ == "__main__":
    main()
