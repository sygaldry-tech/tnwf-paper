"""prrLU + cross-interpolation factorisation tests."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.mps.ldu_cross import ci_factorize, ldu_truncate, prrlu


@pytest.mark.needle
class TestPrrlu:
    def test_rank1_picks_one(self):
        u = np.array([1.0, 2.0, -1.0])
        v = np.array([0.5, 1.5, -0.7, 2.0])
        F = np.outer(u, v)
        rows, cols = prrlu(F, max_rank=4, tol=1e-12)
        # Only one truly significant pivot for rank-1 matrix
        assert len(rows) >= 1
        # Reconstruction with first pivot
        i, j = rows[0], cols[0]
        F_rank1 = np.outer(F[:, j], F[i, :]) / F[i, j]
        np.testing.assert_allclose(F_rank1, F, atol=1e-10)

    def test_zero_matrix(self):
        rows, cols = prrlu(np.zeros((3, 4)), max_rank=2)
        assert rows == [] and cols == []

    def test_max_rank_cap(self):
        rng = np.random.default_rng(0)
        F = rng.standard_normal((5, 6))
        rows, _ = prrlu(F, max_rank=3, tol=0.0)
        assert len(rows) == 3


@pytest.mark.needle
class TestCiFactorize:
    def test_low_rank_round_trip(self):
        # Rank-2 matrix → CI with r=2 should recover exactly
        rng = np.random.default_rng(0)
        u1, u2 = rng.standard_normal(6), rng.standard_normal(6)
        v1, v2 = rng.standard_normal(8), rng.standard_normal(8)
        Pi = np.outer(u1, v1) + np.outer(u2, v2)
        T_l, T_r, rows, cols, _ = ci_factorize(Pi, max_rank=2, tol=1e-12)
        approx = T_l @ T_r
        np.testing.assert_allclose(approx, Pi, atol=1e-9)
        assert len(rows) == 2

    def test_full_rank_recovers_at_capacity(self):
        rng = np.random.default_rng(1)
        Pi = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
        T_l, T_r, _, _, _ = ci_factorize(Pi, max_rank=4, tol=0.0)
        np.testing.assert_allclose(T_l @ T_r, Pi, atol=1e-9)

    def test_returns_pivot_error(self):
        rng = np.random.default_rng(2)
        Pi = rng.standard_normal((4, 5))
        _, _, _, _, err = ci_factorize(Pi, max_rank=2, tol=0.0)
        assert err >= 0.0


@pytest.mark.needle
class TestLduTruncate:
    def test_shape(self):
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((6, 4))
        L, R = ldu_truncate(mat, D_max=3)
        assert L.shape == (6, 3)
        assert R.shape == (3, 4)

    def test_low_rank_recovery(self):
        rng = np.random.default_rng(0)
        u, v = rng.standard_normal(8), rng.standard_normal(5)
        mat = np.outer(u, v)
        L, R = ldu_truncate(mat, D_max=2)
        np.testing.assert_allclose(L @ R, mat, atol=1e-10)
