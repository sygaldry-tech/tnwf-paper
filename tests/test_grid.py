"""Grid + kinetic-eigenvalue tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tnwf.grid import make_grid, make_kinetic_eigenvalues


@pytest.mark.needle
class TestMakeGrid:
    def test_shape_2d(self):
        g = make_grid(N=4, d=2, L=2 * math.pi)
        assert g.shape == (16, 2)

    def test_shape_3d(self):
        g = make_grid(N=3, d=3, L=1.0)
        assert g.shape == (27, 3)

    def test_range(self):
        g = make_grid(N=8, d=2, L=4.0)
        assert g.min() >= 0.0
        assert g.max() < 4.0


@pytest.mark.needle
class TestKineticEigenvalues:
    def test_shape(self):
        lam = make_kinetic_eigenvalues(N=4, d=3, L=1.0)
        assert lam.shape == (4, 4, 4)

    def test_1d_analytic(self):
        N, L = 8, 2 * math.pi
        lam = make_kinetic_eigenvalues(N=N, d=1, L=L)
        # ½ (2π/L)² (k - N/2)² — for L=2π, factor is 1/2
        expected = 0.5 * (np.arange(N) - N / 2.0) ** 2
        np.testing.assert_allclose(lam, expected)

    def test_kron_sum_2d(self):
        N, L = 4, 1.0
        lam_1d = make_kinetic_eigenvalues(N=N, d=1, L=L)
        lam_2d = make_kinetic_eigenvalues(N=N, d=2, L=L)
        # Kronecker sum: λ_2d[i,j] = λ_1d[i] + λ_1d[j]
        np.testing.assert_allclose(lam_2d, lam_1d[:, None] + lam_1d[None, :])

    def test_minimum_at_zero_mode(self):
        # Min of (k - N/2)² over k=0..N-1 is at k=N/2 for even N → λ_min = 0
        N = 6
        lam = make_kinetic_eigenvalues(N=N, d=1, L=1.0)
        assert lam.argmin() == N // 2
        assert lam[N // 2] == pytest.approx(0.0)
