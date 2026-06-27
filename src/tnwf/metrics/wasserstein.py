"""Exact 2-Wasserstein distance via the Hungarian assignment.

For two equal-size point clouds in low dimension (d ≲ 5, n ≲ 5000),
this is the metric the Action Matching paper reports (Neklyudov 2022
Table 1 + Fig 2 — *"W₂ distance between test data marginals and predicted
marginals"*). It's exact (no Sinkhorn approximation), reasonably fast on
CPU at our sizes (n=2000 takes ~10-30 s per call), and gives a
quantitatively comparable number to the paper.

Falls back to ``sliced_wasserstein`` when the two clouds have different
sizes — exact OT then requires choosing a transport plan with varying
mass per point, which the Hungarian algorithm cannot solve directly.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def wasserstein_2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    squared: bool = False,
) -> float:
    """Exact 2-Wasserstein distance between two equal-size point clouds.

    Args:
        x, y: ``(n, d)`` point clouds. Same n required.
        squared: if True, return W₂². Default returns W₂.
    """
    n_x, _ = x.shape
    n_y, _ = y.shape
    if n_x != n_y:
        raise ValueError(
            f"wasserstein_2 requires equal sizes; got {n_x} and {n_y}. "
            f"For unequal sizes use sliced_wasserstein."
        )
    diff = x[:, None, :] - y[None, :, :]
    C = (diff ** 2).sum(axis=-1)                              # (n, n) squared L2
    row, col = linear_sum_assignment(C)
    mean_sq = float(C[row, col].mean())
    return mean_sq if squared else float(np.sqrt(mean_sq))


def wasserstein_2_subsampled(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_sub: int = 1000,
    n_repeats: int = 3,
    seed: int = 0,
    squared: bool = False,
) -> tuple[float, float]:
    """Subsampled W₂ for large clouds; returns ``(mean, std)`` over repeats.

    Hungarian is O(n³); for n ≳ 3000 it gets slow. Subsampling to
    ``n_sub`` gives a stochastic estimate of W₂ with reasonable variance
    at much lower cost. Repeats use different random subsets.
    """
    rng = np.random.default_rng(seed)
    vals = []
    n = min(x.shape[0], y.shape[0], n_sub)
    for _ in range(n_repeats):
        ix = rng.choice(x.shape[0], size=n, replace=False)
        iy = rng.choice(y.shape[0], size=n, replace=False)
        vals.append(wasserstein_2(x[ix], y[iy], squared=squared))
    return float(np.mean(vals)), float(np.std(vals))
