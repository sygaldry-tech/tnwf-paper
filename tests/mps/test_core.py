"""MPS core helpers: dense_to_mps round-trip, mps_norm, chi_max, truncate_mps,
right_canonicalize, sample_mps_indices.
"""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.mps.core import (
    chi_max,
    dense_to_mps,
    mps_norm,
    mps_to_dense,
    right_canonicalize,
    sample_mps_indices,
    truncate_mps,
)


@pytest.mark.needle
class TestDenseToMpsRoundTrip:
    def test_d2_n4_round_trip(self):
        rng = np.random.default_rng(0)
        d, N = 2, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        v_back = mps_to_dense(mps, N=N, d=d)
        np.testing.assert_allclose(v_back, v, atol=1e-10)

    def test_d3_n3_round_trip(self):
        rng = np.random.default_rng(1)
        d, N = 3, 3
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        np.testing.assert_allclose(mps_to_dense(mps, N=N, d=d), v, atol=1e-10)

    def test_truncated_compresses(self):
        rng = np.random.default_rng(2)
        d, N, D_max = 3, 4, 2
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=D_max)
        assert chi_max(mps) <= D_max


@pytest.mark.needle
class TestMpsNorm:
    def test_norm_random(self):
        rng = np.random.default_rng(0)
        d, N = 3, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        np.testing.assert_allclose(mps_norm(mps), np.linalg.norm(v), rtol=1e-10)

    def test_norm_normalised(self):
        rng = np.random.default_rng(1)
        d, N = 2, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        v = v / np.linalg.norm(v)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        np.testing.assert_allclose(mps_norm(mps), 1.0, rtol=1e-10)


@pytest.mark.needle
class TestTruncateMps:
    def test_truncation_preserves_full_norm_when_d_max_is_full(self):
        rng = np.random.default_rng(3)
        d, N = 3, 3
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        truncated, err = truncate_mps(mps, D_max=N**d)
        np.testing.assert_allclose(mps_norm(truncated), mps_norm(mps), rtol=1e-10)
        assert err < 1e-20

    def test_truncation_returns_error(self):
        rng = np.random.default_rng(4)
        d, N = 3, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        truncated, err = truncate_mps(mps, D_max=2)
        assert err >= 0
        assert chi_max(truncated) <= 2


@pytest.mark.needle
class TestRightCanonicalize:
    def test_preserves_state(self):
        # Right-canonicalisation is gauge-only: |ψ⟩ unchanged.
        rng = np.random.default_rng(0)
        d, N = 3, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        rc = right_canonicalize(mps)
        np.testing.assert_allclose(mps_to_dense(rc, N=N, d=d), v, atol=1e-10)

    def test_right_canonical_property(self):
        # For sites j ≥ 1: Σ_{n,b} A_j[a,n,b] · A_j*[a',n,b] = δ_{a,a'}
        rng = np.random.default_rng(1)
        d, N = 3, 4
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        mps = dense_to_mps(v, N=N, d=d, D_max=N**d)
        rc = right_canonicalize(mps)
        for j in range(1, d):
            A = rc[j]
            D_L = A.shape[0]
            G = np.einsum("anb,cnb->ac", A, A.conj())
            np.testing.assert_allclose(G, np.eye(D_L), atol=1e-10)


@pytest.mark.needle
class TestSampleMpsIndicesGivesCorrectMarginals:
    def test_matches_psi_squared(self):
        rng = np.random.default_rng(0)
        N, d = 6, 2
        v = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
        v /= np.linalg.norm(v)
        rho_true = (np.abs(v) ** 2).reshape(N, N)

        mps = dense_to_mps(v, N=N, d=d, D_max=N)
        rc = right_canonicalize(mps)
        n = 50_000
        idx = sample_mps_indices(rc, n, np.random.default_rng(1))
        flat = idx[:, 0] * N + idx[:, 1]
        hist = np.bincount(flat, minlength=N**d).reshape(N, N) / n
        # Should match within sampling noise (~1/sqrt(n) ≈ 0.005)
        assert np.abs(hist - rho_true).max() < 0.02
