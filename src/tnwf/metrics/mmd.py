"""Maximum Mean Discrepancy with RBF kernel (median heuristic bandwidth)."""
from __future__ import annotations

import numpy as np


def mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    """Unbiased MMD (returns √MMD², clipped to ≥ 0)."""
    xy = np.concatenate([x, y], axis=0)
    if bandwidth is None:
        d2_all = np.sum((xy[:, None] - xy[None, :]) ** 2, axis=-1)
        triu = d2_all[np.triu_indices(len(xy), k=1)]
        bandwidth = float(np.sqrt(np.median(triu) / 2.0 + 1e-12))

    def k(a, b):
        d2 = np.sum((a[:, None] - b[None, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * bandwidth ** 2))

    n, m = len(x), len(y)
    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)
    mmd2 = (
        (kxx.sum() - np.diag(kxx).sum()) / (n * (n - 1))
        + (kyy.sum() - np.diag(kyy).sum()) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(np.sqrt(max(mmd2, 0.0)))
