"""Amplitude-amplification cost model.

Single source of truth for what amplitude amplification costs per accepted rare
sample. Figure and analysis code must import from here rather than reimplement
it: divergent copies previously put inconsistent lift numbers into the figure
and its caption.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = ["AMP_PREFACTOR", "AMP_PREFACTOR_QUERY", "amp_cost"]

# Asymptotic cost prefactor for the rule used here (minimize expected
# preparations). Writing m = (2n+1)*theta, the cost (2n+1)/sin^2((2n+1)theta)
# becomes m/(theta sin^2 m); minimizing over m gives sin m = 2m cos m, i.e.
# tan m = 2m, whose first positive root is m* = 1.165561. The cost then
# approaches
#     AMP_PREFACTOR / sqrt(a),   AMP_PREFACTOR = m*/sin^2(m*) = 1.38005.
#
# This is NOT pi/2 = 1.5708. That is the prefactor for the max-P rule -- n ~
# (pi/4)/sqrt(a) rounds at two preparations each, driving P -> 1 -- which costs
# 13.8% more. Drawing a pi/2 reference against cost-minimizing data makes the
# data appear to beat its own model by ~11%; it does not, the line is simply the
# wrong asymptote.
_M_STAR = brentq(lambda x: np.tan(x) - 2.0 * x, 1.0, 1.4)
AMP_PREFACTOR = float(_M_STAR / np.sin(_M_STAR) ** 2)

# The same optimum in ORACLE-QUERY units. A Grover round is one oracle (S_f)
# call but two state-preparation applications, so n = (m/theta - 1)/2 ~
# m/(2 theta) and the prefactor is exactly half. Provided for completeness; the
# paper quotes preparations, because a preparation is a full tensor-network
# transport and because the clean -1/2 power law survives only in that unit
# (fitted slopes over p <= 0.2: -0.500 for 2n+1, -0.604 and -0.436 for the two
# natural query conventions).
AMP_PREFACTOR_QUERY = AMP_PREFACTOR / 2.0


def amp_cost(a, nmax: int = 400, unit: str = "prep"):
    """Expected cost per accepted rare sample at good-subspace amplitude ``a``.

    The Grover round count is chosen to MINIMIZE expected cost, ``(2n+1)/P(n)``,
    not to maximize the single-shot success probability ``P``.

    Maximizing ``P`` is the textbook "optimal number of Grover iterations", but
    it is the wrong objective when preparations are the unit of cost and a
    failed trial can simply be repeated. At ``a = 0.38`` it spends 3
    preparations for ``P = 0.83`` where 1 preparation gives ``P = 0.38`` at 28%
    lower expected cost. Using it overstates the quantum cost by ~11% on average
    and makes the cost curve step for no benefit.

    ``n = 0`` is always in the feasible set and costs exactly ``1/a`` -- plain
    rejection sampling on the prepared state -- so this rule can never be more
    expensive than the classical arm on the same state, and the advantage
    switches on smoothly as ``a`` shrinks rather than dipping below unity.

    Args:
        a:    good-subspace amplitude (scalar or array), clipped to [0, 1].
        nmax: largest round count searched. The default is ample for
              ``a >~ 1e-5``; below that the optimum exceeds it and the returned
              cost is an over-estimate.
        unit: ``"prep"`` counts applications of A and A-dagger, ``2n+1`` per
              trial; ``"query"`` counts oracle calls, ``max(n, 1)`` per trial
              (at least one indicator evaluation is needed to test the sample).

    Returns:
        float for scalar input, else an ndarray of the same shape.
    """
    if unit not in ("prep", "query"):
        raise ValueError(f"unit must be 'prep' or 'query', got {unit!r}")
    arr = np.asarray(a, float)
    # Flatten so the round-count axis broadcasts against any input shape; the
    # caller's shape is restored at the end.
    th = np.arcsin(np.sqrt(np.clip(np.atleast_1d(arr).ravel(), 0.0, 1.0)))
    n = np.arange(nmax + 1)[:, None]
    P = np.sin((2 * n + 1) * th[None, :]) ** 2
    per_trial = (2 * n + 1) if unit == "prep" else np.maximum(n, 1)
    out = np.min(per_trial / np.maximum(P, 1e-12), axis=0)
    return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)
