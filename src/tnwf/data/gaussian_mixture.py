"""d-dimensional Gaussian mixture sampler."""
from __future__ import annotations

import numpy as np


def gm_mode_centers(
    d: int,
    nmodes: int = 4,
    scale: float = 3.0,
    arrangement: str = "orthogonal",
) -> np.ndarray:
    """Return mode centres. Shape (2d, d) for orthogonal, (nmodes, 2) for symmetric (d=2)."""
    if arrangement == "symmetric":
        if d != 2:
            raise ValueError("arrangement='symmetric' requires d=2")
        angles = 2 * np.pi * np.arange(nmodes) / nmodes
        return scale * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if arrangement == "orthogonal":
        centers = []
        for j in range(d):
            for sign in (+1, -1):
                c = np.zeros(d)
                c[j] = sign * scale
                centers.append(c)
        return np.asarray(centers)
    raise ValueError(f"Unknown arrangement: {arrangement!r}")


def sample_gaussian_mixture(
    n_samples: int,
    d: int = 2,
    nmodes: int = 5,
    std: float = 0.5,
    scale: float = 3.0,
    arrangement: str = "orthogonal",
    seed: int | None = None,
) -> np.ndarray:
    """Sample from a d-dim Gaussian mixture. Returns (n_samples, d) float32."""
    rng = np.random.default_rng(seed)
    centers = gm_mode_centers(d=d, nmodes=nmodes, scale=scale, arrangement=arrangement)
    K = centers.shape[0]
    per_mode = n_samples // K
    remainder = n_samples - per_mode * K

    chunks = []
    for i, c in enumerate(centers):
        n = per_mode + (1 if i < remainder else 0)
        chunks.append(rng.normal(loc=c, scale=std, size=(n, d)))
    data = np.vstack(chunks).astype(np.float32)
    rng.shuffle(data)
    return data


def mode_coverage(
    samples: np.ndarray,
    d: int,
    nmodes: int = 4,
    std: float = 0.5,
    scale: float = 3.0,
    arrangement: str = "orthogonal",
    radius_std: float = 2.0,
) -> dict:
    """Per-mode capture metric — fraction within radius_std·std of nearest centre."""
    centers = gm_mode_centers(d=d, nmodes=nmodes, scale=scale, arrangement=arrangement)
    K = centers.shape[0]
    dists = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dists.argmin(axis=1)
    nearest_dist = dists.min(axis=1)
    captured = nearest_dist < radius_std * std
    per_mode_count = np.bincount(nearest[captured], minlength=K)
    n = samples.shape[0]
    return {
        "coverage_frac": float(captured.mean()),
        "modes_hit": int((per_mode_count > 0).sum()),
        "modes_well_hit": int((per_mode_count >= 0.05 * n).sum()),
        "mean_nn_dist": float(nearest_dist.mean()),
        "per_mode_count": per_mode_count,
        "K": int(K),
    }
