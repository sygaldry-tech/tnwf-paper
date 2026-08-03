"""End-to-end JAM training smoke test."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tnwf.jam.train import load_jam, make_V_fn, train


@pytest.mark.medium
def test_train_swiss_roll_smoke(tmp_path):
    out = tmp_path / "swiss_roll_seed0.pt"
    ckpt = train(
        dataset="swiss_roll_2d", seed=0, out_path=str(out),
        n_iter=200, batch_size=64,
        hidden=32, n_layers=2, time_embed_dim=16, log_every=200,
    )
    assert os.path.isfile(out)
    assert np.isfinite(ckpt["final_loss"])

    model, cfg = load_jam(str(out), device="cpu")
    assert cfg["d"] == 2
    assert cfg["dataset"] == "swiss_roll_2d"

    V_fn = make_V_fn(model, device="cpu")
    pts = np.random.default_rng(0).uniform(0, cfg["L"], size=(16, 2))
    V = V_fn(pts, t=0.5)
    assert V.shape == (16,)
    assert np.all(np.isfinite(V))
