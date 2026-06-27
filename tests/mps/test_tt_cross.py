"""TT-cross approximation: rank-1 separability + smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.mps.core import mps_to_dense
from tnwf.mps.tt_cross import elementwise_product_mps, tt_cross


def _eval_dense_from_fn(fn, N: int, d: int) -> np.ndarray:
    """Materialise fn over the full (N^d) integer grid for cross-checks."""
    grid = np.array(np.meshgrid(*([np.arange(N)] * d), indexing="ij")).reshape(d, -1).T
    return np.asarray(fn(grid)).ravel()


@pytest.mark.needle
class TestTtCrossSeparable:
    def test_rank1_recovery(self):
        # f(i, j) = g(i) * h(j) is rank-1 → TT-cross should recover exactly.
        N, d = 4, 2
        g = np.array([1.0, 0.5, -0.3, 2.0])
        h = np.array([0.7, -1.1, 0.2, 0.4])

        def fn(idx: np.ndarray) -> np.ndarray:
            return g[idx[:, 0]] * h[idx[:, 1]]

        cores = tt_cross(fn, N=N, d=d, D_max=2, n_sweeps=2, seed=0)
        dense_approx = mps_to_dense(cores, N=N, d=d).real
        dense_true = _eval_dense_from_fn(fn, N, d)
        np.testing.assert_allclose(dense_approx, dense_true, atol=1e-9)

    def test_rank1_3d(self):
        N, d = 3, 3
        rng = np.random.default_rng(0)
        g = [rng.standard_normal(N) for _ in range(d)]

        def fn(idx: np.ndarray) -> np.ndarray:
            return np.prod([g[k][idx[:, k]] for k in range(d)], axis=0)

        cores = tt_cross(fn, N=N, d=d, D_max=2, n_sweeps=2, seed=1)
        dense_approx = mps_to_dense(cores, N=N, d=d).real
        dense_true = _eval_dense_from_fn(fn, N, d)
        np.testing.assert_allclose(dense_approx, dense_true, atol=1e-9)


@pytest.mark.needle
class TestTtCrossLowRank:
    def test_rank2_function(self):
        # f(i,j) = g1(i)*h1(j) + g2(i)*h2(j) — exactly rank 2
        N, d = 4, 2
        rng = np.random.default_rng(2)
        g1, g2 = rng.standard_normal(N), rng.standard_normal(N)
        h1, h2 = rng.standard_normal(N), rng.standard_normal(N)

        def fn(idx: np.ndarray) -> np.ndarray:
            return g1[idx[:, 0]] * h1[idx[:, 1]] + g2[idx[:, 0]] * h2[idx[:, 1]]

        cores = tt_cross(fn, N=N, d=d, D_max=2, n_sweeps=3, seed=0)
        dense_approx = mps_to_dense(cores, N=N, d=d).real
        dense_true = _eval_dense_from_fn(fn, N, d)
        np.testing.assert_allclose(dense_approx, dense_true, atol=1e-8)


@pytest.mark.needle
class TestTtCrossComplex:
    def test_complex_separable(self):
        N, d = 3, 3
        rng = np.random.default_rng(0)
        g = [rng.standard_normal(N) + 1j * rng.standard_normal(N) for _ in range(d)]

        def fn(idx: np.ndarray) -> np.ndarray:
            return np.prod([g[k][idx[:, k]] for k in range(d)], axis=0)

        cores = tt_cross(fn, N=N, d=d, D_max=2, n_sweeps=2, seed=0)
        dense_approx = mps_to_dense(cores, N=N, d=d)
        dense_true = _eval_dense_from_fn(fn, N, d)
        np.testing.assert_allclose(dense_approx, dense_true, atol=1e-9)


@pytest.mark.needle
class TestElementwiseProduct:
    def test_hadamard_round_trip(self):
        rng = np.random.default_rng(0)
        d, N = 3, 3
        a = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        b = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)

        from tnwf.mps.core import dense_to_mps

        mps_a = dense_to_mps(a, N=N, d=d, D_max=N**d)
        mps_b = dense_to_mps(b, N=N, d=d, D_max=N**d)
        out = elementwise_product_mps(mps_a, mps_b, D_max=N**d)
        out_dense = mps_to_dense(out, N=N, d=d)
        np.testing.assert_allclose(out_dense, a * b, atol=1e-10)
