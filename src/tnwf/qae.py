"""
Quantum Amplitude Estimation (QAE) — Maximum-Likelihood variant.

Implements Maximum-Likelihood QAE (Suzuki, Uno, Raymond, Tanaka,
Onodera, Yamamoto 2020, arXiv:1904.10246). Given a state-preparation
unitary A satisfying
    A|0⟩ = √(1-a) |bad⟩|0⟩ + √a |good⟩|1⟩,
MLQAE estimates a ∈ [0, 1] with absolute precision ε using
~O(1/ε · log log(1/ε)) total Grover applications — the Heisenberg
regime, slope ≈ -1 of `|â − a|` vs total Grover applications. Classical
Monte Carlo gives slope -1/2 on the same axes.

This module is a numpy *simulation* of MLQAE. We don't build a real
quantum circuit; instead we model the (Grover-amplified) outcome
probability `p_k = sin²((2k+1)θ)` exactly and draw Bernoulli samples.
This is the right thing to do for Spike A — we're testing whether the
algorithm achieves its claimed scaling on a synthetic state, not
benchmarking circuit-level depth.

Why MLQAE rather than QPE-based QAE or IQAE:
  • No ancilla register / QFT (vs canonical Brassard QAE).
  • Fixed geometric schedule of k ∈ {0, 1, 2, 4, ..., 2^M}; no
    adaptive bracket-narrowing logic to get wrong.
  • Maximum-likelihood estimator is a simple 1-D optimisation.
  • Established to achieve Heisenberg scaling in the original paper.

References:
  Suzuki, Uno, Raymond, Tanaka, Onodera, Yamamoto (2020).
    "Amplitude estimation without phase estimation."
    Quantum Information Processing 19:75. arXiv:1904.10246.
  Brassard, Høyer, Mosca, Tapp (2002). "Quantum Amplitude
    Amplification and Estimation." Contemp. Math. 305:53-74.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ============================================================================
# Algorithm primitive: simulate one round of (Q^k A |0⟩) measurement
# ============================================================================

def _sample_grover_round(
    a_true: float, k: int, n_shots: int, rng: np.random.Generator,
) -> int:
    """Return the number of '1' outcomes in n_shots Bernoulli draws with
    success probability sin²((2k+1) · arcsin(√a_true)).

    Models the measurement of (Q^k A |0⟩) projected onto the "good"
    subspace. This is the *only* place where a_true is consulted.
    """
    theta = math.asin(math.sqrt(a_true))
    p = math.sin((2 * k + 1) * theta) ** 2
    p = min(max(p, 0.0), 1.0)
    return int(rng.binomial(n_shots, p))


# ============================================================================
# Maximum-Likelihood QAE
# ============================================================================

def _neg_log_likelihood(
    theta: float, schedule: list[int], hits: list[int], shots: list[int],
) -> float:
    """Negative log likelihood of θ given observed (hits, shots) at each
    schedule point. Returns +∞ outside [0, π/2]."""
    if theta <= 0.0 or theta >= math.pi / 2.0:
        return float("inf")
    total = 0.0
    for k, h, n in zip(schedule, hits, shots):
        p = math.sin((2 * k + 1) * theta) ** 2
        p = min(max(p, 1e-15), 1.0 - 1e-15)
        total -= h * math.log(p) + (n - h) * math.log(1.0 - p)
    return total


def _mlqae_estimate_theta(
    schedule: list[int], hits: list[int], shots: list[int],
    grid_size: int = 4096,
) -> float:
    """Return the θ ∈ (0, π/2) that maximizes the joint likelihood.

    Uses a 1-D grid search followed by golden-section refinement.
    Multi-modal due to the periodicity of sin²; the grid catches the
    correct mode and the polish refines it.
    """
    # Coarse grid search
    thetas = np.linspace(1e-6, math.pi / 2.0 - 1e-6, grid_size)
    nll = np.array([
        _neg_log_likelihood(t, schedule, hits, shots) for t in thetas
    ])
    i_min = int(np.argmin(nll))
    # Bracket around the minimum
    lo = thetas[max(i_min - 1, 0)]
    hi = thetas[min(i_min + 1, grid_size - 1)]
    # Golden-section refinement
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = lo, hi
    c = b_ - phi * (b_ - a_)
    d = a_ + phi * (b_ - a_)
    f_c = _neg_log_likelihood(c, schedule, hits, shots)
    f_d = _neg_log_likelihood(d, schedule, hits, shots)
    for _ in range(64):
        if f_c < f_d:
            b_, d, f_d = d, c, f_c
            c = b_ - phi * (b_ - a_)
            f_c = _neg_log_likelihood(c, schedule, hits, shots)
        else:
            a_, c, f_c = c, d, f_d
            d = a_ + phi * (b_ - a_)
            f_d = _neg_log_likelihood(d, schedule, hits, shots)
        if abs(b_ - a_) < 1e-10:
            break
    return 0.5 * (a_ + b_)


# ============================================================================
# Public API
# ============================================================================

@dataclass
class QAEResult:
    """Output of `mlqae_estimate`."""
    a_hat: float                       # estimated amplitude
    total_grover_queries: int          # Σ_i shots_i · (2 k_i + 1)
    total_a_queries: int               # Σ_i shots_i · 2(k_i + 1)
    schedule: list[int] = field(default_factory=list)
    hits: list[int] = field(default_factory=list)
    shots: list[int] = field(default_factory=list)
    theta_hat: float = 0.0


def mlqae_estimate(
    a_true: float,
    M: int = 5,
    n_shots_per_k: int = 100,
    rng: Optional[np.random.Generator] = None,
    schedule: Optional[list[int]] = None,
) -> QAEResult:
    """Maximum-likelihood QAE estimate of `a_true`.

    Args:
        a_true: ground-truth amplitude (consulted only inside
                `_sample_grover_round`).
        M:      determines the largest Grover power k_max = 2^M.
                Total Grover queries ≈ n_shots_per_k · 2^(M+1).
        n_shots_per_k: shots drawn at each k in the schedule.
        rng:    numpy RNG; None ⇒ default.
        schedule: optional explicit list of k values. Default is the
                geometric schedule [0, 1, 2, 4, ..., 2^M] from
                Suzuki et al. 2020.

    Returns:
        QAEResult with `a_hat` and oracle-call accounting.

    Heisenberg scaling: for fixed `n_shots_per_k` and the default
    geometric schedule, the variance of â scales as 2^(-2M), so
    error ≈ 1/k_max ≈ 1/(2^M). Total queries scale as 2^M, so error
    vs total queries scales as 1/Q_total — slope -1.
    """
    if rng is None:
        rng = np.random.default_rng()
    if schedule is None:
        schedule = [0] + [2 ** i for i in range(M + 1)]

    hits: list[int] = []
    shots: list[int] = []
    total_grover = 0
    total_a = 0
    for k in schedule:
        h = _sample_grover_round(a_true, k, n_shots_per_k, rng)
        hits.append(h)
        shots.append(n_shots_per_k)
        total_grover += n_shots_per_k * (2 * k + 1)
        total_a += n_shots_per_k * (2 * k + 2)

    theta_hat = _mlqae_estimate_theta(schedule, hits, shots)
    a_hat = math.sin(theta_hat) ** 2

    return QAEResult(
        a_hat=a_hat,
        total_grover_queries=total_grover,
        total_a_queries=total_a,
        schedule=schedule,
        hits=hits,
        shots=shots,
        theta_hat=theta_hat,
    )


# ============================================================================
# Classical Monte Carlo baseline for comparison
# ============================================================================

def classical_mc_estimate(
    a_true: float, n_samples: int, rng: Optional[np.random.Generator] = None,
) -> tuple[float, int]:
    """Estimate `a` via plain Bernoulli sampling. Returns (â, n_samples)."""
    if rng is None:
        rng = np.random.default_rng()
    n_one = int(rng.binomial(n_samples, a_true))
    return float(n_one) / float(n_samples), n_samples
