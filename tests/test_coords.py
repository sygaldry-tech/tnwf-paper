"""Born-sample coordinate convention.

The bug these guard against: `sample_from_psi_grid` / `sample_from_mps` used to
dither to the *right* of the grid node (`x = (i + U(0,1))·dx`) while ψ is stored
as its values *at* the nodes. Every sample came out a rigid `+dx/2` high on every
axis. It was invisible at the error scale of the hyperparameter sweep and only
obvious near the finite-sample floor, so it survived into published numbers.

`tests/mps/test_core.py` already covered `sample_mps_indices`, but it stops at the
integer indices — one line short of the index→coordinate conversion where the
convention lives. These tests cover that last step.
"""
from __future__ import annotations

import numpy as np
import pytest

from tnwf import coords
from tnwf.grid import make_grid
from tnwf.metrics.sw import sliced_wasserstein
from tnwf.mps.core import dense_to_mps
from tnwf.pipelines.run_evolution import sample_from_mps, sample_from_psi_grid


def _psi_from_density(rho: np.ndarray) -> np.ndarray:
    """A (real, positive) wavefunction whose Born density is `rho`."""
    return np.sqrt(rho / rho.sum()).astype(np.complex128)


@pytest.mark.needle
class TestSamplersAreNodeCentered:
    """Samples must be centered on the node, not on the cell to its right."""

    def test_psi_grid_recovers_node_mean(self):
        # ψ concentrated on a single node: the sample mean must be that node's
        # coordinate. Under the old convention it landed at node + dx/2.
        N, d, L = 8, 2, 8.0
        dx = L / N
        rho = np.zeros((N,) * d)
        rho[3, 5] = 1.0
        samples = sample_from_psi_grid(
            _psi_from_density(rho).ravel(), N=N, d=d, L=L, n=20_000,
            rng=np.random.default_rng(0),
        )
        expected = np.array([3, 5]) * dx
        # Dither is U(-dx/2, dx/2), so the mean converges to the node itself.
        np.testing.assert_allclose(samples.mean(axis=0), expected, atol=0.02)
        # And the support is the cell centered on the node.
        assert np.all(np.abs(samples - expected) <= dx / 2 + 1e-12)

    def test_mps_sampler_recovers_node_mean(self):
        N, d, L = 8, 2, 8.0
        dx = L / N
        rho = np.zeros((N,) * d)
        rho[2, 6] = 1.0
        mps = dense_to_mps(_psi_from_density(rho).ravel(), N=N, d=d, D_max=N**d)
        samples = sample_from_mps(
            mps, N=N, d=d, L=L, n=20_000, rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(samples.mean(axis=0), np.array([2, 6]) * dx,
                                   atol=0.02)

    def test_zero_node_dithers_below_zero(self):
        """The sharpest discriminator: cell 0 must straddle the origin.

        Under `x = (i + U(0,1))·dx` every coordinate is >= 0 by construction, so
        a single negative sample is proof of the node-centered convention.
        """
        N, d, L = 4, 1, 4.0
        rho = np.zeros(N)
        rho[0] = 1.0
        samples = sample_from_psi_grid(
            _psi_from_density(rho), N=N, d=d, L=L, n=2_000,
            rng=np.random.default_rng(0),
        )
        assert samples.min() < 0.0
        assert samples.max() < L / N / 2 + 1e-12

    def test_sampling_grid_matches_evaluation_grid(self):
        """The samplers must agree with `make_grid`, which defines where ψ lives."""
        N, d, L = 16, 1, 8.0
        grid = make_grid(N=N, d=1, L=L).ravel()
        for node in (0, 7, N - 1):
            rho = np.zeros(N)
            rho[node] = 1.0
            samples = sample_from_psi_grid(
                _psi_from_density(rho), N=N, d=d, L=L, n=5_000,
                rng=np.random.default_rng(node),
            )
            assert abs(samples.mean() - grid[node]) < 0.02


@pytest.mark.needle
class TestExactStateReachesTheFiniteSampleFloor:
    """An exact ψ on the grid, with no dynamics and no truncation, must score at
    the target--target floor. Under the old convention it sat far above it —
    which is precisely how a rigid offset hid inside "method error"."""

    def test_exact_gaussian_scores_at_floor(self):
        N, d, L, std = 32, 2, 8.0, 0.7
        rng = np.random.default_rng(0)
        ax = make_grid(N=N, d=1, L=L).ravel() - L / 2
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        rho = np.exp(-(gx**2 + gy**2) / (2 * std**2))
        rho /= rho.sum()

        n = 4_000
        samples = sample_from_psi_grid(
            _psi_from_density(rho).ravel(), N=N, d=d, L=L, n=n, rng=rng,
        ) - L / 2
        target = std * rng.standard_normal((n, d))
        other = std * rng.standard_normal((n, d))

        proj = np.random.default_rng(7)
        floor = sliced_wasserstein(other, target, n_projections=128,
                                   rng=np.random.default_rng(7))
        got = sliced_wasserstein(samples, target, n_projections=128, rng=proj)
        biased = sliced_wasserstein(samples + (L / N) / 2, target,
                                    n_projections=128,
                                    rng=np.random.default_rng(7))

        # Within twice the finite-sample floor, and unambiguously better than
        # the same samples read off under the old convention.
        assert got < 2 * floor, f"SW {got:.4f} vs floor {floor:.4f}"
        assert got < biased


@pytest.mark.needle
class TestConventionResolver:
    def test_cell_centre_needs_no_shift(self):
        z = _FakeNpz({coords.CONV_KEY: np.asarray(coords.CONV_CELL)})
        assert coords.resolve_shift(z, dx=0.25) == 0.0

    def test_node_convention_gets_half_cell(self):
        z = _FakeNpz({coords.CONV_KEY: np.asarray(coords.CONV_NODE)})
        assert coords.resolve_shift(z, dx=0.25) == pytest.approx(-0.125)

    def test_unresolvable_raises_rather_than_defaulting(self):
        """Fail closed. A silent 0.0 here is exactly how a stale archive would
        reach a de-compensated consumer and reintroduce the bias."""
        with pytest.raises(coords.CoordConventionError):
            coords.resolve_shift(_FakeNpz({}), dx=0.25, path="/nowhere/x.npz")

    def test_migration_is_idempotent_by_construction(self):
        """Applying the shift and stamping CONV_CELL is one operation, so a
        second pass is a no-op."""
        z = _FakeNpz({coords.CONV_KEY: np.asarray(coords.CONV_NODE)})
        first = coords.resolve_shift(z, dx=0.25)
        migrated = _FakeNpz({coords.CONV_KEY: np.asarray(coords.CONV_CELL)})
        second = coords.resolve_shift(migrated, dx=0.25)
        assert first != 0.0 and second == 0.0

    def test_unknown_method_raises(self):
        """An allowlist, not a denylist: a new method must not be silently
        assumed to be grid-sampled."""
        assert coords.shift_for_method("dense") == 0.5
        assert coords.shift_for_method("jam") == 0.0
        with pytest.raises(coords.CoordConventionError):
            coords.shift_for_method("some_future_method")


class _FakeNpz:
    """Minimal stand-in for an opened NpzFile."""

    def __init__(self, data: dict):
        self._data = data
        self.files = list(data)

    def __getitem__(self, key):
        return self._data[key]
