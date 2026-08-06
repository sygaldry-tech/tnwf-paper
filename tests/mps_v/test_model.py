"""Trained MPS-V velocity potential: the Table 2 / Figure 8 pipeline.

This module had no test coverage at all, despite `method="mps_v_tdvp2"` being the
path behind Table 2 and the rare-event figure. The realignment in
`make_mps_v_provider` is the specific place an off-by-one would hide: the model
represents V on a grid centred at the origin while the V-step indexes [0, L), and
the two are reconciled by a single `np.roll(c, N//2)`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tnwf.mps_v import make_mps_v_provider
from tnwf.mps_v.model import MPSScalarPotentialTimeSite


def _model(d=3, N=8, D=4, L=8.0, seed=0):
    torch.manual_seed(seed)
    m = MPSScalarPotentialTimeSite(d=d, N=N, D=D, L=L, N_t=6)
    # Random cores make the contraction non-degenerate.
    with torch.no_grad():
        for c in m.cores:
            c.copy_(0.3 * torch.randn_like(c))
    return m


def _contract(cores: list[np.ndarray], idx: np.ndarray) -> complex:
    """Contract an MPS at a fixed multi-index."""
    v = cores[0][:, idx[0], :]
    for j in range(1, len(cores)):
        v = v @ cores[j][:, idx[j], :]
    return complex(v.reshape(()))


@pytest.mark.needle
class TestModelForward:
    def test_forward_shape_and_finiteness(self):
        m = _model()
        x = torch.rand(17, 3) * 8.0
        t = torch.rand(17)
        out = m(x, t)
        assert out.shape == (17, 1)
        assert torch.isfinite(out).all()

    def test_periodic_in_x(self):
        """boundary_mode="periodic" must make V(x + L) == V(x)."""
        m = _model(L=8.0)
        x = torch.rand(11, 3) * 8.0
        t = torch.full((11,), 0.3)
        torch.testing.assert_close(m(x, t), m(x + 8.0, t), atol=1e-5, rtol=1e-4)

    def test_grid_nodes_hit_cores_exactly(self):
        """At x = i·dx the interpolation weight must be exactly 0.

        This is the node convention (`tnwf.coords`) as the trained model sees
        it: grid point i is the sample point, not the left edge of a cell.
        """
        m = _model(N=8, L=8.0)
        dx = 8.0 / 8
        x = torch.arange(8, dtype=torch.float32).reshape(-1, 1) * dx
        x = x.repeat(1, 3)
        x0, x1, alpha = m._x_grid_indices(x)
        torch.testing.assert_close(alpha, torch.zeros_like(alpha), atol=1e-5,
                                   rtol=0)
        assert torch.equal(x0[:, 0], torch.arange(8))


@pytest.mark.needle
class TestCoresMatchForward:
    def test_get_mps_cores_reproduces_forward_at_nodes(self):
        """The extracted cores must evaluate to the same V as the model itself.

        If they didn't, the direct-core V-step would be silently driving the
        flow with a different potential than the one that was trained.
        """
        d, N, L, t = 3, 8, 8.0, 0.4
        m = _model(d=d, N=N, L=L)
        cores = m.get_mps_cores(t)
        dx = L / N
        rng = np.random.default_rng(0)
        for _ in range(8):
            idx = rng.integers(0, N, size=d)
            x = torch.tensor((idx * dx)[None, :], dtype=torch.float32)
            with torch.no_grad():
                direct = float(m(x, torch.tensor([t]))[0, 0])
            assert np.isclose(_contract(cores, idx), direct, atol=1e-4)


@pytest.mark.needle
class TestProviderRealignment:
    def test_roll_maps_world_index_to_centred_coordinate(self):
        """provider(t) cores, indexed on [0, L), must equal the model at the
        corresponding centred coordinate x - L/2.

        A wrong roll direction (or an N//2 vs (N+1)//2 slip) would displace the
        potential by half the box and is exactly the class of error that would
        not show up in any smoke test.
        """
        d, N, L, t = 3, 8, 8.0, 0.25
        m = _model(d=d, N=N, L=L)
        provider = make_mps_v_provider(m, N=N, L=L)
        cores = provider(t)
        assert all(c.dtype == np.complex128 for c in cores)

        dx = L / N
        rng = np.random.default_rng(1)
        for _ in range(8):
            idx = rng.integers(0, N, size=d)
            # World-frame grid point i·dx maps to the centred coordinate
            # i·dx - L/2, which is what the model was trained on.
            x_centred = idx * dx - L / 2
            x = torch.tensor(x_centred[None, :], dtype=torch.float32)
            with torch.no_grad():
                expected = float(m(x, torch.tensor([t]))[0, 0])
            assert np.isclose(_contract(cores, idx).real, expected, atol=1e-4), (
                f"idx={idx} rolled core != model at centred x={x_centred}"
            )

    def test_grid_mismatch_is_rejected(self):
        m = _model(N=8, L=8.0)
        with pytest.raises(ValueError):
            make_mps_v_provider(m, N=16, L=8.0)
