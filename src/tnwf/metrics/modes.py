"""Per-mode geometry of a sample cloud: assignment, width, and tail fraction.

Why this exists. For a mixture target, the >k-sigma tail fraction is
exponentially sensitive to how wide the modes come out: at d=8 with k=4, a
relative width error eps inflates the tail probability by roughly
(1+eps)^10. So an 8.7% width error becomes a 2.4x tail error, while the
endpoint sliced-Wasserstein — dominated by bulk transport — barely moves.

That makes mode width, not the tail fraction, the right quantity to optimize
against when fitting a transport: it is the same information with roughly a
tenth of the sampling noise, because it averages over every sample rather than
counting the few percent that land in the shell.

`sliced_wasserstein` and `mmd_rbf` say nothing about this; they are bulk
metrics. Use these alongside them, not instead.
"""
from __future__ import annotations

import numpy as np


def assign_modes(samples: np.ndarray, centers: np.ndarray):
    """Nearest-mode assignment with untransported mass separated out.

    Returns ``(nearest, nearest_dist, residual)``. ``residual`` marks samples
    closer to the origin than to any mode — source density the transport never
    moved. Those are not tail events and must not be counted as mode mass
    either, or they bias the width low.
    """
    x = np.asarray(samples, dtype=np.float64)
    d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
    nearest = d.argmin(axis=1)
    nearest_dist = d.min(axis=1)
    residual = np.linalg.norm(x, axis=1) < nearest_dist
    return nearest, nearest_dist, residual


def mode_width(samples: np.ndarray, centers: np.ndarray) -> float:
    """Effective isotropic per-mode width.

    For an isotropic d-dimensional Gaussian, E[r^2] = d*sigma^2, so
    ``sigma_eff = sqrt(mean(r^2)/d)`` over the non-residual samples. This is
    the maximum-likelihood scale estimator and is far better behaved than the
    mean radius, which is biased by the chi distribution's skew.
    """
    x = np.asarray(samples, dtype=np.float64)
    _, dist, residual = assign_modes(x, centers)
    keep = dist[~residual]
    if keep.size == 0:
        return float("nan")
    return float(np.sqrt((keep ** 2).mean() / x.shape[1]))


def width_error(samples: np.ndarray, reference: np.ndarray,
                centers: np.ndarray) -> float:
    """Relative mode-width error against a reference cloud, e.g. +0.087.

    Measured against a *finite reference sample* rather than the nominal sigma
    on purpose: the reference carries the same nearest-mode assignment bias and
    the same residual handling, so those cancel.
    """
    return mode_width(samples, centers) / mode_width(reference, centers) - 1.0


def tail_fraction(samples: np.ndarray, centers: np.ndarray, k: float,
                  sigma: float) -> float:
    """Fraction of samples beyond k*sigma of their nearest mode, residual excluded."""
    _, dist, residual = assign_modes(samples, centers)
    tail = (dist > k * sigma) & (~residual)
    return float(tail.mean())


def predicted_tail_ratio(width_err: float, d: int, k: float) -> float:
    """Tail-probability ratio implied by a relative width error alone.

    Exact for isotropic Gaussian modes: the threshold sits at k*sigma_true, so
    a cloud of width sigma_true*(1+eps) puts P[chi^2_d > (k/(1+eps))^2] beyond
    it. Compare against the measured ratio to test whether an excess is width
    inflation or something else — for the shipped d=8 state, width explains
    96% of it.
    """
    from scipy.stats import chi2
    return float(chi2.sf((k / (1.0 + width_err)) ** 2, d) / chi2.sf(k ** 2, d))
