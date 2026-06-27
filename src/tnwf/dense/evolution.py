"""Dense Trotter evolution for the conservative [K, V] Hamiltonian.

Implements the 8-step product formula from Layden et al. (2025) §3:

    W = e^{iβV} e^{iαK} e^{-iβV} e^{-iαK} e^{-iβV} e^{-iαK} e^{iβV} e^{iαK}

where K = ½(SFD_K F†S†)^⊕d (pseudospectral kinetic) and V = diag(V_t(x_k)).
αβ = Δt/2 ⇒ W ≈ exp(-iH^c Δt) with O(Δt^3) local error.

K-step is applied via FFT in O(N^d log N); V-step is pointwise multiplication.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from tnwf.grid import make_kinetic_eigenvalues


def _build_sign_mask(N: int, d: int) -> np.ndarray:
    """(-1)^{i_0+...+i_{d-1}} on (N,)*d."""
    sign_1d = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    S = sign_1d.copy()
    for _ in range(d - 1):
        S = np.multiply.outer(S, sign_1d)
    return S


def apply_K_step(
    psi: np.ndarray,
    alpha: float,
    N: int,
    d: int,
    L: float,
    eigenvalues: np.ndarray | None = None,
) -> np.ndarray:
    """Apply e^{iα K} via d-dim FFT. Exact (no Trotter), unitary."""
    if eigenvalues is None:
        eigenvalues = make_kinetic_eigenvalues(N, d, L)
    psi_nd = psi.reshape((N,) * d)
    S = _build_sign_mask(N, d)
    psi_nd = psi_nd * S
    psi_fft = np.fft.fftn(psi_nd)
    psi_fft *= np.exp(1j * alpha * eigenvalues)
    psi_out = np.fft.ifftn(psi_fft) * S
    return psi_out.ravel()


def apply_V_step(psi: np.ndarray, beta: float, V_grid: np.ndarray) -> np.ndarray:
    """Pointwise e^{iβ V}. Exact, unitary."""
    return psi * np.exp(1j * beta * V_grid)


def trotter_coefficients(delta_t: float, N: int, d: int, L: float) -> tuple[float, float]:
    """8-step product-formula angles. αβ = Δt/2 by construction."""
    alpha = (L / (math.pi * N)) * math.sqrt(delta_t / d)
    beta = (math.pi * N / (2.0 * L)) * math.sqrt(d * delta_t)
    return alpha, beta


def apply_product_formula(
    psi: np.ndarray,
    V_grid: np.ndarray,
    alpha: float,
    beta: float,
    N: int,
    d: int,
    L: float,
    eigenvalues: np.ndarray | None = None,
) -> np.ndarray:
    """8-step W = e^{iβV} e^{iαK} e^{-iβV} e^{-iαK} e^{-iβV} e^{-iαK} e^{iβV} e^{iαK}.

    Applied right-to-left: step 1 (e^{iαK}) acts on ψ first.
    Local error O(Δt^3); unitary.
    """
    if eigenvalues is None:
        eigenvalues = make_kinetic_eigenvalues(N, d, L)
    psi = apply_K_step(psi,  alpha, N, d, L, eigenvalues)
    psi = apply_V_step(psi,  beta,  V_grid)
    psi = apply_K_step(psi, -alpha, N, d, L, eigenvalues)
    psi = apply_V_step(psi, -beta,  V_grid)
    psi = apply_K_step(psi, -alpha, N, d, L, eigenvalues)
    psi = apply_V_step(psi, -beta,  V_grid)
    psi = apply_K_step(psi,  alpha, N, d, L, eigenvalues)
    psi = apply_V_step(psi,  beta,  V_grid)
    return psi


def evolve_wavefunction_conservative(
    psi: np.ndarray,
    steps: list[dict],
    N: int,
    d: int,
    L: float,
    eigenvalues: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve ψ through K product-formula steps. Each step:
        {"V_grid": (N^d,) float, "alpha": float, "beta": float}.
    """
    if eigenvalues is None:
        eigenvalues = make_kinetic_eigenvalues(N, d, L)
    for step in steps:
        psi = apply_product_formula(
            psi, step["V_grid"], step["alpha"], step["beta"], N, d, L, eigenvalues
        )
    return psi


def build_steps_from_V_fn(
    V_fn: Callable[[np.ndarray, float], np.ndarray],
    grid: np.ndarray,
    K: int,
    delta_t: float,
    N: int,
    d: int,
    L: float,
    t_eps: float = 1e-5,
) -> list[dict]:
    """Discretize V_fn(x, t) into K Trotter-step dicts at midpoints t_k = (k+½)Δt."""
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    steps = []
    for k in range(K):
        t_mid = max((k + 0.5) * delta_t, t_eps)
        V_grid = np.asarray(V_fn(grid, t_mid)).ravel()
        steps.append({"V_grid": V_grid, "alpha": alpha, "beta": beta})
    return steps
