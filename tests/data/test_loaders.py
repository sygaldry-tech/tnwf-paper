"""Data-loader determinism + shape tests."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.data.gaussian_mixture import (
    gm_mode_centers,
    sample_gaussian_mixture,
)
from tnwf.data.swiss_roll import sample_swiss_roll


@pytest.mark.needle
class TestSwissRoll:
    def test_shape_dtype(self):
        x = sample_swiss_roll(128, seed=0)
        assert x.shape == (128, 2)
        assert x.dtype == np.float32

    def test_determinism(self):
        a = sample_swiss_roll(64, seed=42)
        b = sample_swiss_roll(64, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = sample_swiss_roll(64, seed=0)
        b = sample_swiss_roll(64, seed=1)
        assert not np.allclose(a, b)

    def test_normalized_scale(self):
        x = sample_swiss_roll(2048, seed=0)
        # data is normalized to unit per-axis std
        np.testing.assert_allclose(x.std(axis=0), 1.0, atol=1e-6)


@pytest.mark.needle
class TestGaussianMixture:
    def test_2d_shape(self):
        x = sample_gaussian_mixture(200, d=2, arrangement="orthogonal", seed=0)
        assert x.shape == (200, 2)
        assert x.dtype == np.float32

    def test_3d_shape(self):
        x = sample_gaussian_mixture(120, d=3, arrangement="orthogonal", seed=0)
        assert x.shape == (120, 3)

    def test_determinism(self):
        a = sample_gaussian_mixture(64, d=3, seed=7)
        b = sample_gaussian_mixture(64, d=3, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_orthogonal_centers_2d(self):
        c = gm_mode_centers(d=2, scale=3.0, arrangement="orthogonal")
        assert c.shape == (4, 2)
        # 2d modes per axis: ±3·e_j
        expected = np.array([[3, 0], [-3, 0], [0, 3], [0, -3]])
        np.testing.assert_allclose(c, expected)

    def test_orthogonal_centers_3d(self):
        c = gm_mode_centers(d=3, scale=2.0, arrangement="orthogonal")
        assert c.shape == (6, 3)

    def test_symmetric_2d_only(self):
        with pytest.raises(ValueError):
            gm_mode_centers(d=3, arrangement="symmetric")

    def test_samples_near_modes(self):
        # at low std, samples should cluster near orthogonal mode centers
        x = sample_gaussian_mixture(2000, d=2, std=0.1, scale=3.0, seed=0)
        c = gm_mode_centers(d=2, scale=3.0, arrangement="orthogonal")
        # nearest mode distance should be small (<= a few std)
        d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        nearest = np.sqrt(d2.min(axis=1))
        assert nearest.mean() < 0.3
