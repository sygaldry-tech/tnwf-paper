"""Numerical equivalence: numpy `tdvp.py` vs torch `tdvp_torch.py`.

Both implementations should produce the same updated MPS to within
Krylov truncation error (~1e-9) on the same inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tnwf.mps.tdvp import tdvp_1site_diagonal, tdvp_2site_diagonal
from tnwf.mps.tdvp_torch import (
    tdvp_1site_diagonal_torch,
    tdvp_2site_diagonal_torch,
)


def _random_mps(d: int, N: int, D: int, seed: int = 0) -> list[np.ndarray]:
    """Random left-canonical MPS — matches the conditioning of real pipeline
    states arriving at a TDVP V-step (which always come from a prior K-step
    that left them canonical). Without normalisation, the random tensors are
    pathologically ill-conditioned and SVD fails inside the 2-site sweep."""
    rng = np.random.default_rng(seed)
    mps = []
    for j in range(d):
        D_L = 1 if j == 0 else D
        D_R = 1 if j == d - 1 else D
        x = rng.standard_normal((D_L, N, D_R)) + 1j * rng.standard_normal((D_L, N, D_R))
        mps.append(x.astype(np.complex128))
    # Left-canonicalise via successive QR.
    for j in range(d - 1):
        D_L, Nj, D_R = mps[j].shape
        mat = mps[j].reshape(D_L * Nj, D_R)
        Q, R = np.linalg.qr(mat)
        K = Q.shape[1]
        mps[j] = Q.reshape(D_L, Nj, K)
        mps[j + 1] = np.einsum("ab,bsc->asc", R, mps[j + 1])
    # Normalise the rightmost core.
    nrm = np.linalg.norm(mps[-1])
    if nrm > 0:
        mps[-1] = mps[-1] / nrm
    return mps


def _random_diagonal_cores(d: int, N: int, D_V: int, seed: int = 1) -> list[np.ndarray]:
    """Real-valued small cores so the effective H is well-conditioned."""
    rng = np.random.default_rng(seed)
    cores = []
    for j in range(d):
        D_L = 1 if j == 0 else D_V
        D_R = 1 if j == d - 1 else D_V
        # Scale small to keep tau·H ‖small‖ for the Krylov.
        x = 0.1 * rng.standard_normal((D_L, N, D_R))
        cores.append(x.astype(np.complex128))
    return cores


def _mps_to_dense(mps: list[np.ndarray]) -> np.ndarray:
    """Contract MPS to dense ψ ∈ ℂ^(N**d). SVD-gauge invariant."""
    out = mps[0]                                 # (1, N, D)
    for core in mps[1:]:
        out = np.einsum("...b,bnc->...nc", out, core)
    # out has shape (1, N, N, ..., N, 1) — flatten
    return out.reshape(-1)


def _mps_diff(mps_a: list[np.ndarray], mps_b: list[np.ndarray]) -> float:
    """Relative Frobenius difference between the CONTRACTED dense wavefunctions.

    MPS representation has a gauge freedom at each bond (SVD column-sign /
    phase). The contracted state ψ ∈ ℂ^(N**d) is gauge-invariant, so it's the
    right object to compare across SVD implementations.
    """
    psi_a = _mps_to_dense(mps_a)
    psi_b = _mps_to_dense(mps_b)
    return float(np.linalg.norm(psi_a - psi_b) / max(np.linalg.norm(psi_a), 1e-30))


@pytest.mark.parametrize("d,N,D", [(3, 2, 4), (4, 3, 6), (5, 4, 8)])
def test_tdvp_2site_cpu_equiv(d, N, D):
    """Torch (device='cpu') matches numpy bit-for-Krylov."""
    mps = _random_mps(d, N, D, seed=42)
    cores = _random_diagonal_cores(d, N, D_V=4, seed=43)
    tau = 1j * 0.05
    D_max = D + 2  # allow some growth
    out_np = tdvp_2site_diagonal(cores, mps, tau, D_max=D_max, n_sweeps=1)
    out_t = tdvp_2site_diagonal_torch(cores, mps, tau, D_max=D_max,
                                       n_sweeps=1, device="cpu")
    err = _mps_diff(out_np, out_t)
    assert err < 1e-9, f"tdvp_2site equiv: rel_err={err:.2e} (d={d}, N={N}, D={D})"


@pytest.mark.parametrize("d,N,D", [(3, 2, 4), (4, 3, 6), (5, 4, 8)])
def test_tdvp_1site_cpu_equiv(d, N, D):
    """Torch (device='cpu') matches numpy bit-for-Krylov."""
    mps = _random_mps(d, N, D, seed=42)
    cores = _random_diagonal_cores(d, N, D_V=4, seed=43)
    tau = 1j * 0.05
    out_np = tdvp_1site_diagonal(cores, mps, tau, n_sweeps=1)
    out_t = tdvp_1site_diagonal_torch(cores, mps, tau, n_sweeps=1, device="cpu")
    err = _mps_diff(out_np, out_t)
    assert err < 1e-9, f"tdvp_1site equiv: rel_err={err:.2e} (d={d}, N={N}, D={D})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tdvp_2site_cuda_equiv():
    """Torch CUDA path matches numpy CPU."""
    mps = _random_mps(4, 3, 6, seed=42)
    cores = _random_diagonal_cores(4, 3, D_V=4, seed=43)
    tau = 1j * 0.05
    out_np = tdvp_2site_diagonal(cores, mps, tau, D_max=8, n_sweeps=1)
    out_t = tdvp_2site_diagonal_torch(cores, mps, tau, D_max=8,
                                       n_sweeps=1, device="cuda")
    err = _mps_diff(out_np, out_t)
    assert err < 1e-9, f"cuda equiv: rel_err={err:.2e}"
