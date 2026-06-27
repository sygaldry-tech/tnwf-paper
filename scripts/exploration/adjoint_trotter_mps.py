"""MPS-form adjoint for the Path-3 Trotter pipeline (Phase 3a).

Replaces the dense ψ vector with an MPS throughout the forward and
backward sweeps. K-substeps use ``apply_K_step_mps`` (bond-preserving,
per-site FFT). V-substeps build ``exp(i·s·β·V)`` as an MPS via TT-cross
and Hadamard-multiply into ψ with SVD truncation to ``D_max``.

Adjoint:
  K-substep   λ ← K^†(α) λ           (just K-step with negated α)
  V-substep   dL/dV[x] += s·2β · Im(λ[x] · ψ_after[x]*)
              λ ← exp(-i·s·β·V) ⊙ λ  (Hadamard with phase, truncate)

Phase 3a leaves V as a dense ``N^d`` vector (no MLP yet) so the V-grad
is gathered as a dense tensor. The benefit visible at this stage:
ψ never appears as a dense ``N^d`` tensor inside the chain — only at
the terminal loss step, where we still need ψ on the grid to compute
the Hellinger ratio. Phase 3b would replace V's dense parameterisation
with an MPS to remove the last ``O(N^d)`` choke point.

Validation: ``__main__`` runs gradient match against the dense adjoint
in ``adjoint_trotter.py`` at high D (no truncation). Should agree to
machine precision when bond truncation is exact.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import (  # noqa: E402  (sys.path mutation above)
    SUBSTEP_PLAN,
    make_kinetic_eigenvalues,
    trotter_coefficients,
)

# Existing MPS utilities
from tnwf.mps.core import (  # noqa: E402
    apply_K_step_mps,
    dense_to_mps,
    mps_to_dense,
    truncate_mps,
)
from tnwf.mps.tt_cross import elementwise_product_mps, tt_cross  # noqa: E402


# ---------------------------------------------------------------------------
# Building the phase MPS exp(i·sign·β·V) from V on the grid via TT-cross.
# ---------------------------------------------------------------------------

def _phase_mps(V_grid: np.ndarray, gamma: complex, N: int, d: int,
               D_V: int) -> list[np.ndarray]:
    """TT-cross compression of x ↦ exp(γ · V_grid[x]) into an MPS.

    γ = i·s·β for the forward V-substep, or γ = -i·s·β for the adjoint.
    """
    V_grid = V_grid.reshape((N,) * d)

    def fn(indices: np.ndarray) -> np.ndarray:
        # indices: (M, d) int32 → values at those grid points
        flat_v = V_grid[tuple(indices[:, j] for j in range(d))]
        return np.exp(gamma * flat_v)

    return tt_cross(fn, N=N, d=d, D_max=D_V, n_sweeps=3, tol=1e-12)


# ---------------------------------------------------------------------------
# Forward + backward in MPS form.
# ---------------------------------------------------------------------------

def forward_with_cache_mps(
    psi0_dense: np.ndarray,
    V_grids: Sequence[np.ndarray],
    *,
    alpha: float,
    beta: float,
    N: int,
    d: int,
    L: float,
    D_max: int,
    D_V: int,
    n_substeps: int = 1,
) -> list[list[np.ndarray]]:
    """Forward Trotter pipeline with MPS ψ; cache MPS at each substep.

    ``alpha`` and ``beta`` are the *substep* coefficients (computed from
    ``Δt / n_substeps``). Each segment k applies the 8-step product
    formula ``n_substeps`` times with the same V_grids[k].

    Returns ``cache`` of length ``K_data * n_substeps * 8 + 1``.
    """
    K_data = len(V_grids)
    psi = dense_to_mps(psi0_dense, N=N, d=d, D_max=D_max)
    cache = [psi]
    for k in range(K_data):
        V = V_grids[k]
        for _ in range(n_substeps):
            for kind, s in SUBSTEP_PLAN:
                if kind == "K":
                    psi = apply_K_step_mps(psi, alpha=s * alpha, N=N, L=L)
                else:
                    phase_mps = _phase_mps(V, 1j * s * beta, N, d, D_V=D_V)
                    psi = elementwise_product_mps(psi, phase_mps, D_max=D_max)
                cache.append(psi)
    return cache


def _hellinger_lam_dense(psi_dense: np.ndarray, q_grid: np.ndarray,
                          floor: float = 1e-20) -> np.ndarray:
    """∂L/∂ψ* for L = ‖|ψ| − √q‖² (dense)."""
    abs_psi = np.maximum(np.abs(psi_dense), floor)
    return psi_dense - np.sqrt(q_grid) * (psi_dense / abs_psi)


def adjoint_backward_mps(
    psi_cache: list[list[np.ndarray]],
    V_grids: Sequence[np.ndarray],
    q_grids: Sequence[np.ndarray],
    *,
    alpha: float,
    beta: float,
    N: int,
    d: int,
    L: float,
    D_max: int,
    D_V: int,
    D_lam_max: int | None = None,
    n_substeps: int = 1,
) -> list[np.ndarray]:
    """Adjoint backward over MPS cache; returns dL/dV_grid[k] (dense N^d each).

    ``alpha`` and ``beta`` are the *substep* coefficients matching the
    forward pass. Each segment k contributes ``n_substeps * 4`` V-grad
    accumulations (4 V-substeps per 8-step formula × M substeps), all
    into the same dL/dV_grid[k] since V is constant across the segment.
    """
    K_data = len(V_grids)
    assert len(q_grids) == K_data + 1
    assert len(psi_cache) == K_data * n_substeps * 8 + 1
    if D_lam_max is None:
        D_lam_max = D_max

    dL_dV = [np.zeros_like(V) for V in V_grids]

    # Terminal λ: dense → MPS
    psi_K_dense = mps_to_dense(psi_cache[K_data * n_substeps * 8], N=N, d=d)
    lam_dense = _hellinger_lam_dense(psi_K_dense, q_grids[K_data])
    lam = dense_to_mps(lam_dense, N=N, d=d, D_max=D_lam_max)

    for k in range(K_data - 1, -1, -1):
        V = V_grids[k]
        for m in range(n_substeps - 1, -1, -1):
            for sub_idx in range(7, -1, -1):
                kind, s = SUBSTEP_PLAN[sub_idx]
                cache_pos = k * n_substeps * 8 + m * 8 + 1 + sub_idx
                psi_after_mps = psi_cache[cache_pos]

                if kind == "V":
                    lam_grid = mps_to_dense(lam, N=N, d=d)
                    psi_after_grid = mps_to_dense(psi_after_mps, N=N, d=d)
                    dL_dV[k] = dL_dV[k] + (
                        s * 2.0 * beta * (lam_grid * np.conj(psi_after_grid)).imag
                    )
                    phase_dag = _phase_mps(V, -1j * s * beta, N, d, D_V=D_V)
                    lam = elementwise_product_mps(lam, phase_dag, D_max=D_lam_max)
                else:
                    lam = apply_K_step_mps(lam, alpha=-s * alpha, N=N, L=L)

        # Add intermediate-snapshot λ contribution at start of segment k
        if k > 0:
            psi_k_dense = mps_to_dense(psi_cache[k * n_substeps * 8], N=N, d=d)
            lam_contrib_dense = _hellinger_lam_dense(psi_k_dense, q_grids[k])
            lam_contrib_mps = dense_to_mps(lam_contrib_dense, N=N, d=d,
                                           D_max=D_lam_max)
            lam = _mps_add(lam, lam_contrib_mps, D_max=D_lam_max)

    return dL_dV


def _mps_add(mps_a: list[np.ndarray], mps_b: list[np.ndarray],
             D_max: int) -> list[np.ndarray]:
    """A + B as MPS via block-diagonal stacking, then truncate to D_max.

    Bond grows D_a + D_b on each interior bond; first/last edges stay 1.
    """
    d = len(mps_a)
    summed: list[np.ndarray] = []
    for j in range(d):
        A = mps_a[j]
        B = mps_b[j]
        D_La, N_loc, D_Ra = A.shape
        D_Lb, _, D_Rb = B.shape
        if j == 0:
            # Stack columns on right; left edge stays 1
            assert D_La == 1 and D_Lb == 1
            new = np.zeros((1, N_loc, D_Ra + D_Rb), dtype=np.complex128)
            new[:, :, :D_Ra] = A
            new[:, :, D_Ra:] = B
        elif j == d - 1:
            # Stack rows on left; right edge stays 1
            assert D_Ra == 1 and D_Rb == 1
            new = np.zeros((D_La + D_Lb, N_loc, 1), dtype=np.complex128)
            new[:D_La, :, :] = A
            new[D_La:, :, :] = B
        else:
            new = np.zeros((D_La + D_Lb, N_loc, D_Ra + D_Rb),
                           dtype=np.complex128)
            new[:D_La, :, :D_Ra] = A
            new[D_La:, :, D_Ra:] = B
        summed.append(new)
    out, _ = truncate_mps(summed, D_max)
    return out


# ---------------------------------------------------------------------------
# Validation: gradient match against dense adjoint.
# ---------------------------------------------------------------------------

def _self_test():
    import torch
    from adjoint_trotter import _adjoint_loss_and_grad, _sign_mask  # noqa: E402

    np.random.seed(0)
    torch.manual_seed(0)
    N, d, L = 4, 2, 4.0
    K_data = 3
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max = N ** (d // 2 + 1)  # large enough that truncation is exact at d=2
    D_V = D_max

    # Common inputs
    psi0_raw = (np.random.randn(N**d) + 1j * np.random.randn(N**d))
    psi0 = (psi0_raw / np.linalg.norm(psi0_raw)).astype(np.complex128)
    q_grids_np = []
    for _ in range(K_data + 1):
        q = np.random.rand(N**d).clip(min=1e-6)
        q_grids_np.append(q / q.sum())
    V_init = [np.random.randn(N**d) * 0.5 for _ in range(K_data)]

    # --- MPS adjoint
    cache_mps = forward_with_cache_mps(
        psi0, V_init,
        alpha=alpha, beta=beta, N=N, d=d, L=L, D_max=D_max, D_V=D_V,
    )
    dL_dV_mps = adjoint_backward_mps(
        cache_mps, V_init, q_grids_np,
        alpha=alpha, beta=beta, N=N, d=d, L=L, D_max=D_max, D_V=D_V,
    )
    # Forward loss in MPS form
    loss_mps = 0.0
    for k in range(K_data):
        psi_k_dense = mps_to_dense(cache_mps[(k + 1) * 8], N=N, d=d)
        loss_mps += float(((np.abs(psi_k_dense) - np.sqrt(q_grids_np[k + 1])) ** 2).sum())

    # --- Dense adjoint (reference)
    K_eigs_t = torch.from_numpy(make_kinetic_eigenvalues(N, d, L, "cpu").numpy()
                                 if hasattr(make_kinetic_eigenvalues(N, d, L, "cpu"), "numpy")
                                 else None)
    # Use torch path consistently with adjoint_trotter
    K_eigs_t = make_kinetic_eigenvalues(N, d, L, "cpu").to(torch.complex128)
    sign_t = _sign_mask(N, d, "cpu")
    psi0_t = torch.from_numpy(psi0)
    q_grids_t = [torch.from_numpy(q) for q in q_grids_np]
    V_grids_t = [torch.from_numpy(V) for V in V_init]
    loss_dense, dL_dV_dense = _adjoint_loss_and_grad(
        V_grids_t, psi0_t, q_grids_t, alpha, beta, N, d, K_eigs_t, sign_t)

    print(f"loss     MPS    = {loss_mps:.12f}")
    print(f"loss     dense  = {loss_dense.item():.12f}")
    print(f"loss     |Δ|    = {abs(loss_mps - loss_dense.item()):.2e}")
    for k in range(K_data):
        g_mps = dL_dV_mps[k]
        g_dense = dL_dV_dense[k].numpy()
        rel = np.linalg.norm(g_mps - g_dense) / (np.linalg.norm(g_dense) + 1e-30)
        absdiff = np.abs(g_mps - g_dense).max()
        print(f"grad[k={k}] rel={rel:.3e}  max|Δ|={absdiff:.3e}")


if __name__ == "__main__":
    _self_test()
