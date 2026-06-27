"""Sliced-Wasserstein, MMD, NLL sanity tests."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.metrics import sliced_wasserstein, mmd_rbf, nll_from_density_grid, density_from_psi


@pytest.mark.needle
class TestSlicedWasserstein:
    def test_zero_on_identical_clouds(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((512, 3))
        assert sliced_wasserstein(x, x, n_projections=64) == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_shift(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((512, 2))
        y = x + np.array([2.0, 0.0])
        sw = sliced_wasserstein(x, y, n_projections=128)
        # mean projection-distance for unit shift along x-axis ≈ 2/π * 2 ≈ 1.27 averaged over directions
        assert sw > 0.5

    def test_determinism(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((128, 2))
        y = rng.standard_normal((128, 2))
        a = sliced_wasserstein(x, y, n_projections=32, rng=np.random.default_rng(7))
        b = sliced_wasserstein(x, y, n_projections=32, rng=np.random.default_rng(7))
        assert a == b


@pytest.mark.needle
class TestMMD:
    def test_near_zero_on_same_distribution(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((400, 2))
        y = rng.standard_normal((400, 2))
        assert mmd_rbf(x, y) < 0.1

    def test_positive_on_shift(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((400, 2))
        y = rng.standard_normal((400, 2)) + np.array([5.0, 0.0])
        assert mmd_rbf(x, y) > 0.5

    def test_nonnegative(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 2))
        y = rng.standard_normal((100, 2))
        assert mmd_rbf(x, y) >= 0.0


@pytest.mark.needle
class TestNLL:
    def test_uniform_vs_delta(self):
        # Uniform grid density: NLL = log(N^d). Delta-on-cell density: NLL=0 if all samples in that cell.
        N, d, L = 8, 2, 4.0
        N_total = N ** d
        uniform = np.full((N_total,), 1.0 / N_total)
        rng = np.random.default_rng(0)
        samples = rng.uniform(-L / 2, L / 2, size=(200, d)).astype(np.float64)
        nll_uniform = nll_from_density_grid(samples, uniform, N=N, d=d, L=L)
        # log(N^d) = log(64) ≈ 4.158 (per sample, plus log(cell-volume) = log((L/N)^d))
        assert nll_uniform > 0.0

    def test_delta_concentration(self):
        N, d, L = 8, 2, 4.0
        delta = np.zeros((N ** d,))
        delta[0] = 1.0
        # all samples mapped to cell 0 (corner) → NLL = -log(p_cell / cell_volume) = -log(1/(L/N)^d)
        samples = np.full((10, d), -L / 2 + 1e-3)  # in cell 0
        nll = nll_from_density_grid(samples, delta, N=N, d=d, L=L)
        cell_vol = (L / N) ** d
        np.testing.assert_allclose(nll, -np.log(1.0 / cell_vol), atol=1e-6)


@pytest.mark.needle
class TestDensityFromPsi:
    def test_normalised(self):
        psi = np.random.default_rng(0).standard_normal(64) + 1j * np.random.default_rng(1).standard_normal(64)
        rho = density_from_psi(psi)
        np.testing.assert_allclose(rho.sum(), 1.0, atol=1e-12)
        assert np.all(rho >= 0)

    def test_real_psi(self):
        psi = np.array([1.0, 2.0, 0.0, 1.0])
        rho = density_from_psi(psi)
        np.testing.assert_allclose(rho, np.array([1, 4, 0, 1]) / 6.0)
