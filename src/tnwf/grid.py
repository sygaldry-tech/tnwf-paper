"""1D real-space grid + kinetic eigenvalues for the [K, V] conservative Hamiltonian.

Pseudospectral kinetic operator K = ½ (S F D_K F† S†)^⊕d with eigenvalues

    λ[k_0,...,k_{d-1}] = ½ (2π/L)² Σ_j (k_j - N/2)²

Grid points live on the periodic torus [0, L)^d. Sample data is centred at 0
(i.e. lives in [-L/2, L/2)) — see tnwf.metrics.nll.grid_pmf_indices.
"""
from __future__ import annotations

import math

import numpy as np


def make_grid(N: int, d: int, L: float = 2 * math.pi) -> np.ndarray:
    """Cartesian product grid on [0, L)^d. Returns (N^d, d) float64."""
    pts_1d = np.linspace(0.0, L, N, endpoint=False)
    mesh = np.meshgrid(*([pts_1d] * d), indexing="ij")
    coords = [m.ravel() for m in mesh]
    return np.stack(coords, axis=-1)


def make_kinetic_eigenvalues(N: int, d: int, L: float) -> np.ndarray:
    """K eigenvalues on the (N,)*d Fourier grid: ½ (2π/L)² Σ_j (k_j - N/2)²."""
    k_1d = np.arange(N, dtype=float)
    lam_1d = 0.5 * (2.0 * math.pi / L) ** 2 * (k_1d - N / 2) ** 2
    result = np.zeros((N,) * d, dtype=float)
    for j in range(d):
        shape = [1] * d
        shape[j] = N
        result = result + lam_1d.reshape(shape)
    return result
