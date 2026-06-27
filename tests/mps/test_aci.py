"""ACI Hadamard product tests vs dense reference."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.mps.aci import aci_hadamard
from tnwf.mps.core import dense_to_mps, mps_to_dense, truncate_mps


@pytest.mark.needle
class TestAciHadamard:
    def test_2d_round_trip(self):
        rng = np.random.default_rng(0)
        d, N = 2, 4
        a = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        b = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps_a = dense_to_mps(a, N=N, d=d, D_max=N**d)
        mps_b = dense_to_mps(b, N=N, d=d, D_max=N**d)
        out = aci_hadamard(mps_a, mps_b, D_max=N**d, tol=1e-12, n_sweeps=4)
        out, _ = truncate_mps(out, D_max=N**d)
        approx = mps_to_dense(out, N=N, d=d)
        np.testing.assert_allclose(approx, a * b, atol=1e-3)

    def test_3d_low_rank(self):
        rng = np.random.default_rng(1)
        d, N = 3, 3
        # Build separable tensors → product is rank-1
        gs_a = [rng.standard_normal(N) + 1j * rng.standard_normal(N) for _ in range(d)]
        gs_b = [rng.standard_normal(N) + 1j * rng.standard_normal(N) for _ in range(d)]

        def sep_dense(gs):
            v = gs[0]
            for g in gs[1:]:
                v = np.outer(v, g).ravel()
            return v.astype(np.complex128)

        a = sep_dense(gs_a)
        b = sep_dense(gs_b)
        mps_a = dense_to_mps(a, N=N, d=d, D_max=2)
        mps_b = dense_to_mps(b, N=N, d=d, D_max=2)
        out = aci_hadamard(mps_a, mps_b, D_max=4, tol=1e-12, n_sweeps=4)
        out, _ = truncate_mps(out, D_max=4)
        approx = mps_to_dense(out, N=N, d=d)
        np.testing.assert_allclose(approx, a * b, atol=1e-3)

    def test_d1_edge_case(self):
        N = 4
        a = np.array([[1, 2, 3, 4]], dtype=np.complex128).reshape(1, N, 1)
        b = np.array([[2, 1, -1, 0.5]], dtype=np.complex128).reshape(1, N, 1)
        out = aci_hadamard([a], [b], D_max=4)
        np.testing.assert_allclose(out[0], a * b)
