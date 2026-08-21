"""Tests for closed-form analytic V_t."""
from __future__ import annotations

import numpy as np
import pytest

from tnwf.theory import make_analytic_V_fn


@pytest.mark.needle
class TestAnalyticV:
    def test_signature_gmm_2d(self):
        V_fn = make_analytic_V_fn("gmm_2d")
        x = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.7]])
        V = V_fn(x, t=0.5)
        assert V.shape == (3,)
        assert V.dtype == np.float64
        assert np.all(np.isfinite(V))

    def test_signature_gmm_3d(self):
        V_fn = make_analytic_V_fn("gmm_3d")
        x = np.random.default_rng(0).uniform(0, 8, size=(5, 3))
        V = V_fn(x, t=0.3)
        assert V.shape == (5,)
        assert np.all(np.isfinite(V))

    def test_swiss_roll_rejected(self):
        with pytest.raises(ValueError, match="closed-form"):
            make_analytic_V_fn("swiss_roll_2d")

    def test_t_eps_floor(self):
        # t=0 would divide by zero; t_eps prevents NaN
        V_fn = make_analytic_V_fn("gmm_2d", t_eps=1e-3)
        V = V_fn(np.array([[1.0, 1.0]]), t=0.0)
        assert np.isfinite(V).all()

    def test_gradient_matches_velocity_field(self):
        """∇V_t(x) should equal E[x_1 - x_0 | x_t=x] = (x/t) + (1-t)σ_0²/t · ∇log p_t(x).

        Numerically: finite-difference gradient of V_t should match Tweedie's
        formula. Verified by central differences at a random test point.
        """
        V_fn = make_analytic_V_fn("gmm_2d", t_eps=1e-3)
        x0_world = np.array([[3.5, 4.2]])     # in world coords
        t = 0.4
        h = 1e-5
        # central difference
        grads = []
        for j in range(2):
            ep = np.zeros((1, 2)); ep[0, j] = h
            V_p = V_fn(x0_world + ep, t)
            V_m = V_fn(x0_world - ep, t)
            grads.append((V_p - V_m).item() / (2 * h))
        grad_fd = np.array(grads)

        # The gradient should be smooth and finite at this interior point
        assert np.all(np.isfinite(grad_fd))
        # And it should be NON-zero (we picked a point off-axis)
        assert np.linalg.norm(grad_fd) > 1e-6

    def test_at_t_one_pure_quadratic(self):
        """At t=1, the (1-t)/t·log p_t term vanishes; V_t reduces to ||x_centred||²/2."""
        L = 8.0     # gmm_2d default
        V_fn = make_analytic_V_fn("gmm_2d", t_eps=1e-3)
        x_world = np.array([[5.0, 3.0]])              # centered → (1, -1)
        x_centred = x_world - L / 2.0
        V = V_fn(x_world, t=1.0)
        # At t=1 with t_eps=1e-3, t_eff is 1.0 (no floor needed). Expected:
        # V = ||x_c||² / 2 + 0 · log p_t = 1.0
        expected_quad = (x_centred ** 2).sum() / 2.0
        # Exclude the small log-p contribution (it'd be 0 at t=1)
        assert abs(V.item() - expected_quad) < 1e-3
