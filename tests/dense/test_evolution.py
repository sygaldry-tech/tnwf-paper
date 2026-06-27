"""Dense Trotter evolution: unitarity + free-particle dispersion sanity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tnwf.dense.evolution import (
    apply_K_step,
    apply_V_step,
    apply_product_formula,
    evolve_wavefunction_conservative,
    trotter_coefficients,
)
from tnwf.grid import make_kinetic_eigenvalues


def _random_psi(N: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(N**d) + 1j * rng.standard_normal(N**d)
    psi /= np.linalg.norm(psi)
    return psi


@pytest.mark.needle
class TestApplyKStep:
    def test_unitary(self):
        N, d, L = 4, 2, 4.0
        psi = _random_psi(N, d, seed=0)
        psi_out = apply_K_step(psi, alpha=0.1, N=N, d=d, L=L)
        np.testing.assert_allclose(np.linalg.norm(psi_out), 1.0, atol=1e-10)

    def test_zero_alpha_is_identity(self):
        N, d, L = 4, 2, 4.0
        psi = _random_psi(N, d, seed=1)
        psi_out = apply_K_step(psi, alpha=0.0, N=N, d=d, L=L)
        np.testing.assert_allclose(psi_out, psi, atol=1e-10)

    def test_eigenvalue_propagation(self):
        # If ψ is a Fourier eigenmode, apply_K_step rotates it by exp(iα λ_k).
        N, d, L = 8, 1, 2 * math.pi
        lam = make_kinetic_eigenvalues(N, d, L)
        # Build a single Fourier mode at k=2 in the SF basis (after S†F†)
        k = 2
        # Construct ψ = S F† e_k (canonical eigenvector of K)
        e_k = np.zeros(N, dtype=np.complex128)
        e_k[k] = 1.0
        psi = np.fft.ifft(e_k) * np.array([(-1.0) ** j for j in range(N)])
        # Apply K rotation
        alpha = 0.3
        psi_out = apply_K_step(psi, alpha=alpha, N=N, d=d, L=L)
        # The mode should have rotated by exp(iα λ_k)
        expected = psi * np.exp(1j * alpha * lam[k])
        np.testing.assert_allclose(psi_out, expected, atol=1e-10)


@pytest.mark.needle
class TestApplyVStep:
    def test_unitary(self):
        N, d = 4, 2
        psi = _random_psi(N, d, seed=2)
        V = np.random.default_rng(0).standard_normal(N**d)
        psi_out = apply_V_step(psi, beta=0.5, V_grid=V)
        np.testing.assert_allclose(np.linalg.norm(psi_out), 1.0, atol=1e-10)

    def test_zero_beta_is_identity(self):
        psi = _random_psi(4, 2, seed=3)
        V = np.zeros(16)
        np.testing.assert_allclose(apply_V_step(psi, beta=0.0, V_grid=V), psi)


@pytest.mark.needle
class TestTrotterCoefficients:
    def test_alpha_beta_product(self):
        # αβ = Δt / 2 by construction
        for delta_t in (0.01, 0.1, 1.0):
            for N, d, L in [(4, 2, 4.0), (8, 3, 1.0)]:
                a, b = trotter_coefficients(delta_t, N, d, L)
                np.testing.assert_allclose(a * b, delta_t / 2.0, rtol=1e-12)


@pytest.mark.needle
class TestProductFormula:
    def test_unitary(self):
        N, d, L = 4, 2, 4.0
        psi = _random_psi(N, d, seed=4)
        V = np.random.default_rng(0).standard_normal(N**d)
        a, b = trotter_coefficients(0.05, N, d, L)
        out = apply_product_formula(psi, V, a, b, N, d, L)
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-9)

    def test_zero_potential_returns_psi_up_to_K_phase(self):
        # When V≡0, all V-steps are identities and K-steps cancel pairwise.
        # → product formula reduces to identity (W with V=0 evaluates to e^0).
        N, d, L = 4, 2, 4.0
        psi = _random_psi(N, d, seed=5)
        V = np.zeros(N**d)
        a, b = trotter_coefficients(0.05, N, d, L)
        out = apply_product_formula(psi, V, a, b, N, d, L)
        # K rotations sum to: +α -α -α +α = 0  → identity
        np.testing.assert_allclose(out, psi, atol=1e-9)


@pytest.mark.needle
class TestEvolveConservative:
    def test_unitarity_K_steps(self):
        N, d, L, K = 4, 2, 4.0, 4
        psi0 = _random_psi(N, d, seed=6)
        rng = np.random.default_rng(0)
        a, b = trotter_coefficients(0.05, N, d, L)
        steps = [
            {"V_grid": rng.standard_normal(N**d), "alpha": a, "beta": b}
            for _ in range(K)
        ]
        psi_T = evolve_wavefunction_conservative(psi0, steps, N=N, d=d, L=L)
        np.testing.assert_allclose(np.linalg.norm(psi_T), 1.0, atol=1e-9)
