"""Sliced Wasserstein distance."""
from __future__ import annotations

import numpy as np


def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """1-Wasserstein averaged over n_projections random unit directions in R^d."""
    if rng is None:
        rng = np.random.default_rng(0)
    if x.shape[0] != y.shape[0]:
        # subsample to common length so sorted differences align
        n = min(x.shape[0], y.shape[0])
        x = x[:n]
        y = y[:n]
    d = x.shape[1]
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    px = x @ directions.T
    py = y @ directions.T
    w1 = np.mean(np.abs(np.sort(px, axis=0) - np.sort(py, axis=0)), axis=0)
    return float(np.mean(w1))
