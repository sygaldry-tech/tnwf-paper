"""Closed-form action-matching potentials for Gaussian-source GMM targets.

For source x_0 ~ N(0, σ_0² I) and target x_1 ~ Σ_k w_k N(c_k, σ_k² I), the
JAM-trained scalar potential V_t(x) admits the closed form

    V_t(x) = ||x||² / (2t)
             + ((1-t) σ_0² / t) · log p_t(x) + C(t)

where p_t(x) = Σ_k w_k N(x; t c_k, Σ_k(t)) is itself a Gaussian mixture
(derivation via Tweedie's identity). We can use this
to skip JAM training entirely on GMM datasets — all error in the V-step
pipeline then comes from Trotter splitting, grid quantisation, and MPS
truncation, with no JAM training noise.

Use:
    from tnwf.theory import make_analytic_V_fn
    V_fn = make_analytic_V_fn("gmm_2d")          # uses dataset defaults
    V_fn(x_world_coords, t=0.5)                  # → (n,) np.float64
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from tnwf.data.gaussian_mixture import gm_mode_centers


def _gmm_log_density(
    x_centred: np.ndarray,
    t: float,
    *,
    centers: np.ndarray,
    weights: np.ndarray,
    component_var: float,
    sigma_0: float,
) -> np.ndarray:
    """Log-density of the Gaussian-mixture interpolant p_t(x) at centred coords.

    p_t(x) = Σ_k w_k · N(x; t c_k, σ_t² I)  with  σ_t² = (1-t)² σ_0² + t² σ_k².

    Args:
        x_centred:      (n, d) points in centred frame (data near origin).
        t:              time in (0, 1].
        centers:        (K, d) mode centres.
        weights:        (K,) mixture weights, sum to 1.
        component_var:  σ_k² (assumed isotropic and equal across modes).
        sigma_0:        source std σ_0.

    Returns:
        (n,) log p_t(x_centred).
    """
    n, d = x_centred.shape
    var_t = (1.0 - t) ** 2 * sigma_0 ** 2 + t ** 2 * component_var
    # diffs[i, k] = x_i - t·c_k  (n, K, d)
    diffs = x_centred[:, None, :] - t * centers[None, :, :]
    sq_dist = (diffs ** 2).sum(axis=-1)              # (n, K)
    log_const = (
        np.log(weights)[None, :]
        - 0.5 * d * math.log(2.0 * math.pi * var_t)
    )                                                 # (1, K)
    log_terms = log_const - 0.5 * sq_dist / var_t    # (n, K)
    return logsumexp(log_terms, axis=-1)             # (n,)


def make_analytic_V_fn(
    dataset: str,
    *,
    L: float | None = None,
    d: int | None = None,
    scale: float | None = None,
    std: float | None = None,
    sigma_0: float = 1.0,
    arrangement: str = "orthogonal",
    t_eps: float = 1e-3,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Build a closed-form V_fn(x_world, t) → (n,) np.float64.

    Drop-in replacement for `tnwf.jam.train.make_V_fn(model)`. The signature
    matches: takes world-frame points (in [0, L)^d) and a time scalar.

    Args:
        dataset:     "gmm_2d" or "gmm_3d" (only Gaussian-source GMM targets
                     have a closed-form V_t).
        L, d, scale, std:
                     dataset config; if any is None, falls back to
                     `DATASET_DEFAULTS` from `tnwf.jam.train`.
        sigma_0:     source standard deviation (default 1.0, matches the JAM
                     trainer's `torch.randn` source).
        arrangement: "orthogonal" (default) — must match the dataset.
        t_eps:       floor on t to avoid the 1/t singularity at t→0.

    Returns:
        V_fn: callable(x_world: (n, d) np.ndarray, t: float) → (n,) np.float64
    """
    from tnwf.jam.train import DATASET_DEFAULTS

    if not (dataset.startswith("gmm_") and dataset.endswith("d")
            and dataset[4:-1].isdigit()):
        raise ValueError(
            f"analytic V_fn is only defined for Gaussian-source GMM datasets, "
            f"got {dataset!r}. swiss_roll has no closed-form V_t."
        )
    cfg = dict(DATASET_DEFAULTS[dataset])
    if d is None: d = cfg["d"]
    if L is None: L = cfg["L"]
    if scale is None: scale = cfg["scale"]
    if std is None: std = cfg["std"]

    centers = gm_mode_centers(d=d, scale=scale, arrangement=arrangement)
    K = centers.shape[0]
    weights = np.full(K, 1.0 / K)
    component_var = float(std ** 2)
    L_local = float(L)

    def V_fn(x_world: np.ndarray, t: float) -> np.ndarray:
        x_centred = np.asarray(x_world, dtype=np.float64) - L_local / 2.0
        t_eff = max(float(t), t_eps)
        log_p = _gmm_log_density(
            x_centred, t_eff,
            centers=centers, weights=weights,
            component_var=component_var, sigma_0=sigma_0,
        )
        # V_t(x) = ||x||² / (2t) + ((1-t)σ_0²/t) · log p_t(x)
        # (additive C(t) dropped — it's a global phase in the V-step.)
        quad = (x_centred ** 2).sum(axis=-1) / (2.0 * t_eff)
        log_coeff = (1.0 - t_eff) * sigma_0 ** 2 / t_eff
        return (quad + log_coeff * log_p).astype(np.float64)

    return V_fn
