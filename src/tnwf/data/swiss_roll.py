"""2D swiss-roll sampler."""
from __future__ import annotations

import numpy as np


def sample_swiss_roll(n_samples: int, noise: float = 0.1, seed: int | None = None) -> np.ndarray:
    """Return (n_samples, 2) float32 swiss-roll points, normalized to unit per-axis std."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    x += rng.normal(scale=noise, size=n_samples)
    y += rng.normal(scale=noise, size=n_samples)
    data = np.stack([x, y], axis=1).astype(np.float32)
    data = data / np.std(data, axis=0, keepdims=True)
    return data
