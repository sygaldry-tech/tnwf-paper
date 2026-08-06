"""Amplitude estimation primitives behind Figure 8(a,b).

Figure 8's central claim is a slope: MLQAE error falls as O(1/Q) against Monte
Carlo's O(1/sqrt(Q)). Nothing exercised that claim before, so a regression in
the estimator would have shown up only as a changed figure.
"""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.qae import classical_mc_estimate, mlqae_estimate


@pytest.mark.needle
class TestClassicalMonteCarlo:
    def test_unbiased_and_counts_queries(self):
        a = 0.0424  # the d=8, >4-sigma tail probability used in the paper
        rng = np.random.default_rng(0)
        est = [classical_mc_estimate(a, 20_000, rng)[0] for _ in range(40)]
        assert abs(np.mean(est) - a) < 0.002
        assert classical_mc_estimate(a, 1234, rng)[1] == 1234

    def test_error_follows_shot_noise(self):
        """Halving the error should cost ~4x the samples."""
        a = 0.0424
        errs = {}
        for n in (2_000, 32_000):
            rng = np.random.default_rng(1)
            e = [abs(classical_mc_estimate(a, n, rng)[0] - a) for _ in range(200)]
            errs[n] = float(np.median(e))
        ratio = errs[2_000] / errs[32_000]
        # 16x more samples => ~4x lower error.
        assert 2.5 < ratio < 6.0, errs


@pytest.mark.needle
class TestMLQAE:
    def test_recovers_the_amplitude(self):
        a = 0.0424
        rng = np.random.default_rng(0)
        res = mlqae_estimate(a, M=5, n_shots_per_k=200, rng=rng)
        assert abs(res.a_hat - a) < 0.01
        assert res.total_grover_queries > 0

    def test_beats_monte_carlo_at_equal_query_budget(self):
        """The claim Figure 8(a) makes, stated as a test."""
        a = 0.0424
        mlqae_err, mc_err = [], []
        for trial in range(30):
            rng = np.random.default_rng(trial)
            res = mlqae_estimate(a, M=5, n_shots_per_k=50, rng=rng)
            mlqae_err.append(abs(res.a_hat - a))
            mc_hat, _ = classical_mc_estimate(a, res.total_grover_queries, rng)
            mc_err.append(abs(mc_hat - a))
        assert np.median(mlqae_err) < np.median(mc_err), (
            f"MLQAE {np.median(mlqae_err):.5f} vs MC {np.median(mc_err):.5f}"
        )

    def test_error_falls_faster_than_shot_noise_with_M(self):
        """Heisenberg scaling: variance ~ 2^(-2M), so error roughly halves per M."""
        a = 0.0424
        med = {}
        for M in (3, 6):
            errs = [abs(mlqae_estimate(a, M=M, n_shots_per_k=50,
                                       rng=np.random.default_rng(t)).a_hat - a)
                    for t in range(30)]
            med[M] = float(np.median(errs))
        # Three extra rounds of amplification: error must fall substantially
        # faster than the sqrt(query) rate a classical sampler would manage.
        assert med[6] < med[3] / 2.0, med
