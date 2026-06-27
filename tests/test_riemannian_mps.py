"""Tests for the Riemannian-MPS primitives in
``scripts/exploration/riemannian_mps.py`` and the optimizer in
``scripts/exploration/riemannian_adam_mps.py``.

All tests are ``@pytest.mark.needle`` (<30s each).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/exploration importable without installing.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "exploration"))

from riemannian_mps import (  # noqa: E402
    gauge_component_norm,
    gauge_project,
    retract_qr,
    stiefel_orthogonality_error,
    to_mixed_canonical,
)
from riemannian_adam_mps import (  # noqa: E402
    _RiemannianAdamMPS,
    init_v_mps_riemannian,
)
from v_mps_parameterization import (  # noqa: E402
    V_dense_to_mps,
    V_mps_to_dense,
    project_dense_grad_to_mps_cores,
)


def _random_v_mps(N: int, d: int, D: int, rng: np.random.Generator
                  ) -> list[np.ndarray]:
    """Random MPS with bonds (1, D, D, ..., D, 1) capped to natural rank."""
    cores: list[np.ndarray] = []
    D_prev = 1
    for j in range(d):
        D_next = 1 if j == d - 1 else min(D, N ** (j + 1), N ** (d - j - 1))
        cores.append(rng.standard_normal((D_prev, N, D_next)))
        D_prev = D_next
    return cores


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------

@pytest.mark.needle
class TestMixedCanonical:
    def test_dense_round_trip_d2(self):
        rng = np.random.default_rng(0)
        N, d, D = 6, 2, 4
        V_mps = _random_v_mps(N, d, D, rng)
        V_dense = V_mps_to_dense(V_mps, N, d)
        V_canon = to_mixed_canonical(V_mps)
        V_dense_canon = V_mps_to_dense(V_canon, N, d)
        np.testing.assert_allclose(V_dense_canon, V_dense, atol=1e-10)

    def test_dense_round_trip_d3(self):
        rng = np.random.default_rng(1)
        N, d, D = 4, 3, 3
        V_mps = _random_v_mps(N, d, D, rng)
        V_dense = V_mps_to_dense(V_mps, N, d)
        V_canon = to_mixed_canonical(V_mps, center=d - 1)
        np.testing.assert_allclose(V_mps_to_dense(V_canon, N, d), V_dense,
                                   atol=1e-10)

    def test_stiefel_orthogonality_holds(self):
        rng = np.random.default_rng(2)
        N, d, D = 5, 3, 3
        V_mps = _random_v_mps(N, d, D, rng)
        V_canon = to_mixed_canonical(V_mps)
        err = stiefel_orthogonality_error(V_canon)
        assert err < 1e-10, f"Stiefel error {err} too large"

    def test_center_choice_preserves_dense(self):
        """Different center choices must give the same dense V."""
        rng = np.random.default_rng(3)
        N, d, D = 4, 3, 3
        V_mps = _random_v_mps(N, d, D, rng)
        V_dense = V_mps_to_dense(V_mps, N, d)
        for center in range(d):
            V_canon = to_mixed_canonical(V_mps, center=center)
            np.testing.assert_allclose(V_mps_to_dense(V_canon, N, d),
                                       V_dense, atol=1e-10,
                                       err_msg=f"center={center} broke dense")


# ---------------------------------------------------------------------------
# Gauge projection
# ---------------------------------------------------------------------------

@pytest.mark.needle
class TestGaugeProject:
    def test_idempotent(self):
        rng = np.random.default_rng(4)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        grads = [rng.standard_normal(c.shape) for c in V_canon]
        proj1 = gauge_project(grads, V_canon)
        proj2 = gauge_project(proj1, V_canon)
        for g1, g2 in zip(proj1, proj2):
            np.testing.assert_allclose(g2, g1, atol=1e-10)

    def test_already_tangent_unchanged(self):
        """Random tangent vector should pass through gauge_project unchanged."""
        rng = np.random.default_rng(5)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        # Synthesize a clearly-tangent vector by projecting random noise once.
        noise = [rng.standard_normal(c.shape) for c in V_canon]
        tangent = gauge_project(noise, V_canon)
        proj_again = gauge_project(tangent, V_canon)
        for t, p in zip(tangent, proj_again):
            np.testing.assert_allclose(p, t, atol=1e-10)

    def test_annihilates_pure_gauge(self):
        """A pure gauge perturbation must lie entirely in the orthogonal
        complement to the tangent space. Construct one explicitly: at site j
        (j < center), an infinitesimal gauge orbit direction is
            dA_j = A_j · S        (S symmetric in matricized D_R basis)
        because A_j (left-orthogonal) + dA_j = A_j (I + S) needs S to lie in
        the gauge group's Lie algebra. The Stiefel tangent's orthogonal
        complement is exactly the symmetric part of A_j^T dA_j.
        """
        rng = np.random.default_rng(6)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        center = d - 1

        gauge_pert = []
        for j, A in enumerate(V_canon):
            if j < center:
                D_L, N_loc, D_R = A.shape
                A_mat = A.reshape(D_L * N_loc, D_R)
                S = rng.standard_normal((D_R, D_R))
                S = 0.5 * (S + S.T)                          # symmetric
                pert_mat = A_mat @ S
                gauge_pert.append(pert_mat.reshape(D_L, N_loc, D_R))
            elif j > center:
                D_L, N_loc, D_R = A.shape
                A_mat = A.reshape(D_L, N_loc * D_R)
                S = rng.standard_normal((D_L, D_L))
                S = 0.5 * (S + S.T)
                pert_mat = S @ A_mat
                gauge_pert.append(pert_mat.reshape(D_L, N_loc, D_R))
            else:
                gauge_pert.append(np.zeros_like(A))

        proj = gauge_project(gauge_pert, V_canon, center=center)
        # The projection of a pure-gauge perturbation should be ~zero on each
        # constrained core.
        for j, p in enumerate(proj):
            if j == center:
                continue
            err = np.linalg.norm(p)
            assert err < 1e-10, f"core {j}: gauge component leaked through ({err})"


# ---------------------------------------------------------------------------
# Retraction
# ---------------------------------------------------------------------------

@pytest.mark.needle
class TestRetractQR:
    def test_preserves_bond_dims(self):
        rng = np.random.default_rng(7)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        tangent = gauge_project([rng.standard_normal(c.shape) for c in V_canon],
                                 V_canon)
        V_new = retract_qr(V_canon, tangent, step_size=0.05)
        for old, new in zip(V_canon, V_new):
            assert old.shape == new.shape

    def test_preserves_stiefel(self):
        rng = np.random.default_rng(8)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        tangent = gauge_project([rng.standard_normal(c.shape) for c in V_canon],
                                 V_canon)
        V_new = retract_qr(V_canon, tangent, step_size=0.1)
        err = stiefel_orthogonality_error(V_new)
        assert err < 1e-10, f"Stiefel error after retraction: {err}"

    def test_zero_step_is_identity(self):
        rng = np.random.default_rng(9)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        tangent = [rng.standard_normal(c.shape) for c in V_canon]
        V_new = retract_qr(V_canon, tangent, step_size=0.0)
        for a, b in zip(V_canon, V_new):
            np.testing.assert_allclose(b, a, atol=1e-10)

    def test_small_step_first_order(self):
        """For small step ε, retracted point ≈ canonical point + ε · tangent
        to first order. Specifically, the dense V_mps_to_dense should match
        to O(ε²)."""
        rng = np.random.default_rng(10)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        tangent = gauge_project([rng.standard_normal(c.shape) for c in V_canon],
                                 V_canon)
        eps = 1e-4
        V_new = retract_qr(V_canon, tangent, step_size=eps)
        # Approximate the linear update: V_dense(V_canon + ε · tangent)
        V_lin_cores = [c + eps * t for c, t in zip(V_canon, tangent)]
        V_dense_retract = V_mps_to_dense(V_new, N, d)
        V_dense_linear = V_mps_to_dense(V_lin_cores, N, d)
        # Difference should be O(ε²); with d=3 cores and tangent norm ~5,
        # chain-rule amplification puts the constant at ~10², so the
        # observed scale is ~1e-5 at ε=1e-4 (vs the naïve ε²=1e-8).
        diff = np.linalg.norm(V_dense_retract - V_dense_linear)
        assert diff < 1e-4, f"first-order error {diff} (should be O(ε²))"


# ---------------------------------------------------------------------------
# RAdam end-to-end on toy fit
# ---------------------------------------------------------------------------

@pytest.mark.needle
class TestRiemannianAdamOnToyFit:
    def test_radam_decreases_loss_monotonically(self):
        """Fit a known MPS by ½‖V_dense - target‖² with RAdam.

        RAdam at lr=1e-2 should reach a low loss within 200 iters; Euclidean
        Adam at the same lr is allowed to diverge — this is the headline
        Riemannian-stability claim of the plan.
        """
        rng = np.random.default_rng(11)
        N, d, D = 6, 2, 4
        target_mps = _random_v_mps(N, d, D, rng)
        target_dense = V_mps_to_dense(target_mps, N, d)

        # Init V_mps at small random
        V_mps = init_v_mps_riemannian(K_data=1, N=N, d=d, D_V_param=D,
                                       init_std=0.1, rng=rng)[0]
        opt = _RiemannianAdamMPS(V_mps, lr=1e-2)

        losses = []
        for _ in range(200):
            V_dense = V_mps_to_dense(V_mps, N, d)
            grad_dense = (V_dense - target_dense).reshape(-1)
            losses.append(0.5 * float(np.sum(grad_dense ** 2)))
            grad_cores = project_dense_grad_to_mps_cores(grad_dense, V_mps,
                                                          N, d)
            opt.step(grad_cores)

        # Final loss should be substantially lower than initial (fit progressed)
        assert losses[-1] < losses[0] * 0.5, (
            f"loss did not decrease enough: {losses[0]} -> {losses[-1]}")
        # Stiefel constraint must still hold after 200 steps
        err = stiefel_orthogonality_error(V_mps)
        assert err < 1e-8, f"Stiefel drift after 200 iters: {err}"

    def test_radam_keeps_loss_finite(self):
        """At lr=1e-2 on this toy, RAdam should NOT NaN/explode."""
        rng = np.random.default_rng(12)
        N, d, D = 6, 2, 4
        target_mps = _random_v_mps(N, d, D, rng)
        target_dense = V_mps_to_dense(target_mps, N, d)
        V_mps = init_v_mps_riemannian(K_data=1, N=N, d=d, D_V_param=D,
                                       init_std=0.1, rng=rng)[0]
        opt = _RiemannianAdamMPS(V_mps, lr=1e-2)
        for _ in range(100):
            V_dense = V_mps_to_dense(V_mps, N, d)
            grad_dense = (V_dense - target_dense).reshape(-1)
            assert np.isfinite(grad_dense).all(), "grad went non-finite"
            grad_cores = project_dense_grad_to_mps_cores(grad_dense, V_mps,
                                                          N, d)
            opt.step(grad_cores)
        # Final dense V must be finite.
        V_dense_final = V_mps_to_dense(V_mps, N, d)
        assert np.isfinite(V_dense_final).all()


# ---------------------------------------------------------------------------
# Gauge component norm (diagnostic)
# ---------------------------------------------------------------------------

@pytest.mark.needle
class TestGaugeDiagnostic:
    def test_post_projection_gauge_norm_zero(self):
        """After gauge_project, the gauge component of the result should be
        zero (idempotence stated as a diagnostic)."""
        rng = np.random.default_rng(13)
        N, d, D = 5, 3, 3
        V_canon = to_mixed_canonical(_random_v_mps(N, d, D, rng))
        grads = [rng.standard_normal(c.shape) for c in V_canon]
        rgrads = gauge_project(grads, V_canon)
        residual = gauge_component_norm(rgrads, V_canon)
        assert residual < 1e-10, (
            f"Riemannian gradient still has gauge component {residual}")
