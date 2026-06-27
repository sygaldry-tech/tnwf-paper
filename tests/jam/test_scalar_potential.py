"""ScalarPotentialMLP forward + JAM loss tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tnwf.jam.scalar_potential import ScalarPotentialMLP, jam_conservative_loss


@pytest.mark.needle
class TestScalarPotentialMLP:
    def test_forward_shape(self):
        model = ScalarPotentialMLP(d=2, hidden=16, time_embed_dim=8, n_layers=2)
        x = torch.randn(8, 2)
        t = torch.rand(8, 1)
        V = model(x, t)
        assert V.shape == (8, 1)

    def test_forward_3d(self):
        model = ScalarPotentialMLP(d=3, hidden=8, time_embed_dim=4, n_layers=2)
        x = torch.randn(4, 3)
        t = torch.rand(4, 1)
        assert model(x, t).shape == (4, 1)

    def test_periodic(self):
        # V(x + L*e_j) = V(x) up to network nonlinearity (input encoding is periodic)
        L = 4.0
        model = ScalarPotentialMLP(d=2, L=L, hidden=8, time_embed_dim=4, n_layers=2)
        torch.manual_seed(0)
        x = torch.tensor([[0.3, -1.1]])
        t = torch.tensor([[0.5]])
        # Shift by full period along axis 0
        x_shift = torch.tensor([[0.3 + L, -1.1]])
        V0 = model(x, t)
        V1 = model(x_shift, t)
        torch.testing.assert_close(V0, V1, atol=1e-5, rtol=1e-5)


@pytest.mark.needle
class TestJamLoss:
    def test_loss_returns_scalar(self):
        torch.manual_seed(0)
        model = ScalarPotentialMLP(d=2, hidden=8, time_embed_dim=4, n_layers=2)
        x0 = torch.randn(16, 2)
        x1 = torch.randn(16, 2)
        t = torch.rand(16, 1)
        loss = jam_conservative_loss(model, x0, x1, t)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_loss_decreases_after_step(self):
        torch.manual_seed(1)
        model = ScalarPotentialMLP(d=2, hidden=16, time_embed_dim=4, n_layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x0 = torch.randn(64, 2)
        x1 = torch.randn(64, 2) * 2.0 + 1.0
        t = torch.rand(64, 1)

        loss0 = jam_conservative_loss(model, x0, x1, t).item()
        for _ in range(20):
            opt.zero_grad()
            loss = jam_conservative_loss(model, x0, x1, t)
            loss.backward()
            opt.step()
        loss_final = jam_conservative_loss(model, x0, x1, t).item()
        assert loss_final < loss0
