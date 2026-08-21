"""Grid-density NLL for wavefunction-flow methods.

The NLL of a sample x under a grid-pmf p_k (cells of volume (L/N)^d) is
    -log p(x) = -log(p_k / (L/N)^d)
where k is the cell containing x. The (L/N)^d factor turns the discrete pmf
into a continuous density, so values are comparable across N.
"""
from __future__ import annotations

import numpy as np


def density_from_psi(psi: np.ndarray) -> np.ndarray:
    """|ψ|² normalized to sum to 1. Accepts complex or real ψ; returns flat float64."""
    rho = np.abs(np.asarray(psi).ravel()) ** 2
    total = rho.sum()
    if total <= 0:
        raise ValueError("|ψ|² sums to zero — cannot normalize")
    return rho / total


def grid_pmf_indices(samples: np.ndarray, N: int, d: int, L: float) -> np.ndarray:
    """Map (n, d) samples in [-L/2, L/2)^d to flat cell indices in [0, N^d)."""
    grid_coords = np.asarray(samples, dtype=np.float64) + L / 2.0
    grid_coords = np.clip(grid_coords, 0.0, L - 1e-9)
    idx = np.clip((grid_coords / L * N).astype(int), 0, N - 1)
    return np.ravel_multi_index([idx[:, j] for j in range(d)], (N,) * d)


def nll_from_density_grid(
    samples: np.ndarray,
    density_pmf: np.ndarray,
    N: int,
    d: int,
    L: float,
    eps: float = 1e-12,
) -> float:
    """Mean NLL: -mean log p(x_i) where p(x) = density_pmf[cell(x)] / (L/N)^d."""
    cell_vol = (L / N) ** d
    flat_idx = grid_pmf_indices(samples, N=N, d=d, L=L)
    p_cells = density_pmf.ravel()[flat_idx]
    log_density = np.log(np.maximum(p_cells, eps)) - np.log(cell_vol)
    return float(-np.mean(log_density))
