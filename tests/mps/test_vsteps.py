"""Cross-check: every MPS V-step matches the Dense V-step on a small grid."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.dense.evolution import apply_V_step
from tnwf.mps.core import dense_to_mps, mps_to_dense
from tnwf.mps.vstep_tdvp import (
    apply_V_step_mps_tci_tdvp1,
    apply_V_step_mps_tci_tdvp2,
)


def _build_dense_V_grid(V_fn, N: int, d: int, L: float) -> np.ndarray:
    """Evaluate V_fn on [0,L)^d at left-edge grid points (matches tt_cross oracle)."""
    grid_1d = np.linspace(0.0, L, N, endpoint=False)
    mesh = np.meshgrid(*([grid_1d] * d), indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=-1)
    return np.asarray(V_fn(pts, t=0.5)).ravel()


def _smooth_V_fn(x: np.ndarray, t: float) -> np.ndarray:
    """Reference smooth potential (low-rank in TT sense): cos / quadratic."""
    return (np.cos(np.pi * x).sum(axis=-1) + 0.1 * (x**2).sum(axis=-1)).astype(np.float64)


def _setup(seed: int = 0):
    N, d, L = 4, 2, 4.0
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
    psi /= np.linalg.norm(psi)
    return psi, N, d, L


def _dense_V_evolved(psi, beta: float, N: int, d: int, L: float):
    V_grid = _build_dense_V_grid(_smooth_V_fn, N=N, d=d, L=L)
    return apply_V_step(psi, beta=beta, V_grid=V_grid)


@pytest.mark.needle
class TestTciTdvpMatchesDense:
    def test_tdvp1_match(self):
        psi, N, d, L = _setup(seed=2)
        beta = 0.2
        psi_dense_evolved = _dense_V_evolved(psi, beta, N, d, L)
        mps = dense_to_mps(psi, N=N, d=d, D_max=N**d)
        out = apply_V_step_mps_tci_tdvp1(
            mps, V_fn=_smooth_V_fn, beta=beta, t_k=0.5, N=N, d=d, L=L, D_V=8,
        )
        out_dense = mps_to_dense(out, N=N, d=d)
        np.testing.assert_allclose(out_dense, psi_dense_evolved, atol=1e-2)

    def test_tdvp2_match(self):
        psi, N, d, L = _setup(seed=3)
        beta = 0.2
        psi_dense_evolved = _dense_V_evolved(psi, beta, N, d, L)
        mps = dense_to_mps(psi, N=N, d=d, D_max=N**d)
        out = apply_V_step_mps_tci_tdvp2(
            mps, V_fn=_smooth_V_fn, beta=beta, t_k=0.5, N=N, d=d, L=L,
            D_max=N**d, D_V=8,
        )
        out_dense = mps_to_dense(out, N=N, d=d)
        np.testing.assert_allclose(out_dense, psi_dense_evolved, atol=1e-2)


