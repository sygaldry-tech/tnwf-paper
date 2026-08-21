"""Tests for the amplitude-amplification cost model."""
import numpy as np
import pytest

from tnwf.amp import AMP_PREFACTOR, AMP_PREFACTOR_QUERY, amp_cost


def test_prefactor_solves_its_defining_equation():
    """AMP_PREFACTOR must be m*/sin^2(m*) with tan(m*) = 2 m*, not pi/2."""
    m = 1.1655611852072114                     # first positive root of tan m = 2m
    assert np.tan(m) == pytest.approx(2 * m, abs=1e-9)
    assert AMP_PREFACTOR == pytest.approx(m / np.sin(m) ** 2, abs=1e-9)
    assert AMP_PREFACTOR == pytest.approx(1.380050, abs=1e-5)
    # Guard against a regression to the max-P constant, which is 13.8% higher.
    assert AMP_PREFACTOR < 0.9 * (np.pi / 2)
    assert AMP_PREFACTOR_QUERY == pytest.approx(AMP_PREFACTOR / 2.0, rel=1e-12)


def test_cost_approaches_the_asymptote_from_below():
    """cost * sqrt(a) -> AMP_PREFACTOR as a -> 0, approached from below."""
    for a in (1e-3, 1e-4, 1e-5):
        scaled = amp_cost(a) * np.sqrt(a)
        assert scaled == pytest.approx(AMP_PREFACTOR, rel=0.02)
        assert scaled <= AMP_PREFACTOR


def test_never_worse_than_rejection_on_the_same_state():
    """n=0 is always feasible at cost 1/a, so the optimum can never exceed it."""
    a = np.geomspace(1e-4, 0.9, 40)
    assert np.all(amp_cost(a) <= 1.0 / a + 1e-9)


def test_falls_with_amplitude_apart_from_round_quantization():
    """Cost falls as the event becomes commoner, but NOT strictly monotonically.

    The round count is an integer, so cost can tick up slightly near an
    amplitude where the current n is perfectly tuned. The clearest case is
    a = 1/4: there theta = pi/6, so sin^2(3 theta) = 1 and n=1 succeeds with
    certainty at cost exactly 3. Moving a away from 1/4 lowers P and raises
    cost until dropping to n=0 becomes worth it. Those excursions are under 1%.
    """
    a = np.geomspace(1e-4, 0.5, 400)
    c = amp_cost(a)
    assert c[0] > 50 * c[-1], "cost must fall by orders of magnitude overall"

    rises = np.diff(c)
    rises = rises[rises > 0]
    assert rises.size < 0.05 * len(a), "quantization steps should be rare"
    assert (rises / c[:-1][np.diff(c) > 0]).max() < 0.02, "and small"

    # The perfectly-tuned point is exact, which pins the mechanism.
    assert amp_cost(0.25) == pytest.approx(3.0, abs=1e-12)


def test_units_differ_and_query_is_cheaper():
    """Queries count n per trial, preparations 2n+1, so queries cost less."""
    a = np.geomspace(1e-4, 0.5, 30)
    prep, query = amp_cost(a, unit="prep"), amp_cost(a, unit="query")
    assert np.all(query <= prep)
    # Asymptotically the ratio tends to 2; at the rare end it should be close.
    assert amp_cost(1e-5, unit="prep") / amp_cost(1e-5, unit="query") == pytest.approx(
        2.0, rel=0.05
    )


def test_shape_and_scalar_handling():
    assert isinstance(amp_cost(0.1), float)
    assert amp_cost(np.array([0.1, 0.01])).shape == (2,)
    assert amp_cost(np.array([[0.1, 0.01], [0.2, 0.02]])).shape == (2, 2)


def test_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unit must be"):
        amp_cost(0.1, unit="gates")
