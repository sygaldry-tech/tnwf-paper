"""Mode geometry, and the exact-oracle floor it is measured against.

The rare-event tail fraction is exponentially sensitive to mode width, so these
tests pin both the estimator and the accuracy an *exact* velocity oracle
achieves. That floor is what says whether a learned oracle has room to improve
or has hit the formulation's limit.
"""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.metrics.modes import (
    assign_modes, mode_width, predicted_tail_ratio, tail_fraction, width_error,
)

D, SIGMA, SCALE, K = 8, 0.5, 3.0, 4.0


def centers(d=D, scale=SCALE):
    C = np.zeros((2 * d, d))
    for j in range(d):
        C[2 * j, j] = scale
        C[2 * j + 1, j] = -scale
    return C


def gmm(n, rng, sigma=SIGMA, d=D):
    C = centers(d)
    return C[rng.integers(0, len(C), n)] + sigma * rng.standard_normal((n, d))


@pytest.mark.needle
class TestModeWidth:
    def test_recovers_the_true_width(self):
        rng = np.random.default_rng(0)
        C = centers()
        assert mode_width(gmm(20_000, rng), C) == pytest.approx(SIGMA, rel=0.02)

    def test_tracks_a_deliberate_inflation(self):
        rng = np.random.default_rng(0)
        C = centers()
        ref = gmm(20_000, rng)
        wide = gmm(20_000, rng, sigma=SIGMA * 1.10)
        assert width_error(wide, ref, C) == pytest.approx(0.10, abs=0.02)

    def test_residual_mass_is_excluded(self):
        """Untransported mass at the origin must not drag the width up."""
        rng = np.random.default_rng(0)
        C = centers()
        clean = gmm(8000, rng)
        stuck = 0.3 * rng.standard_normal((800, D))          # sits at the origin
        mixed = np.vstack([clean, stuck])
        _, _, resid = assign_modes(mixed, C)
        assert resid[len(clean):].mean() > 0.9               # stuck mass flagged
        assert mode_width(mixed, C) == pytest.approx(mode_width(clean, C), rel=0.03)


@pytest.mark.needle
class TestAmplification:
    def test_width_error_predicts_the_tail_ratio(self):
        """The whole argument rests on this: small width error, large tail error."""
        rng = np.random.default_rng(0)
        C = centers()
        ref = gmm(200_000, rng)
        base = tail_fraction(ref, C, K, SIGMA)
        for eps in (0.05, 0.10):
            wide = gmm(200_000, rng, sigma=SIGMA * (1 + eps))
            got = tail_fraction(wide, C, K, SIGMA) / base
            assert got == pytest.approx(predicted_tail_ratio(eps, D, K), rel=0.10)

    def test_amplification_tracks_rarity_not_dimension(self):
        """The accuracy bar tightens with how rare the event is.

        Not with dimension at fixed k: for chi^2_d the mean is d, so k=4 is a
        genuine tail at d=8 (4.2%) but sits near the median at d=16 (45%), and
        the amplification is correspondingly weaker there. What drives it is
        the threshold. At d=8 the exponent runs 3.8, 9.6, 17.6, 27.5 for
        k=3,4,5,6 — so targeting rarer events demands proportionally tighter
        state preparation, which is the opposite of convenient.
        """
        exps = [np.log(predicted_tail_ratio(0.087, D, k)) / np.log(1.087)
                for k in (3, 4, 5, 6)]
        assert all(b > a for a, b in zip(exps, exps[1:])), exps
        assert exps[1] == pytest.approx(9.6, abs=1.0)   # k=4: the measured case
        assert exps[3] > 2 * exps[1]                    # 6 sigma far harsher

        # And the k=4, d=8 case is the 2.4x we measured on the prepared state.
        assert 2.0 < predicted_tail_ratio(0.087, D, K) < 2.8
        # At fixed k, higher d is a *weaker* amplification, not stronger.
        assert predicted_tail_ratio(0.087, 16, K) < predicted_tail_ratio(0.087, D, K)


@pytest.mark.medium
class TestExactOracleFloor:
    """An exact closed-form velocity, integrated by RK4 with no learning, no
    grid and no tensor network, is the best this flow formulation can do. Every
    trained oracle is measured against it, so it is pinned here."""

    def test_exact_oracle_reproduces_width_and_tail(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from make_fig_gmm_ode import integrate_rk4

        rng = np.random.default_rng(0)
        C = centers()
        ref = gmm(4000, rng)
        x0 = rng.standard_normal((4000, D))
        _, xs = integrate_rk4(x0, n_steps=400, centers=C,
                              component_var=SIGMA ** 2, sigma_0=1.0)
        flowed = xs[-1]

        err = width_error(flowed, ref, C)
        ratio = tail_fraction(flowed, C, K, SIGMA) / tail_fraction(ref, C, K, SIGMA)
        assert abs(err) < 0.025, f"exact-oracle width error {err:+.3f}"
        assert 0.85 < ratio < 1.20, f"exact-oracle tail ratio {ratio:.2f}"
