"""End-to-end JAM training smoke test."""
from __future__ import annotations

import os
from pathlib import Path

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


@pytest.mark.medium
@pytest.mark.parametrize("loss_name", ["cfm", "am"])
def test_train_petals_trajectory_smoke(tmp_path, loss_name):
    """Trajectory-mode JAM trains finitely on petals and yields a usable V_fn.

    Parametrised over CFM-with-grad (the legacy default) and Action Matching
    (Neklyudov et al. 2022) — both should converge to finite loss and
    produce a finite V_fn at every interior timepoint.
    """
    out = tmp_path / f"petals_seed0_{loss_name}.pt"
    ckpt = train(
        dataset="petals_2d", seed=0, out_path=str(out),
        n_iter=200, batch_size=64,
        hidden=32, n_layers=2, time_embed_dim=16, log_every=200,
        loss_name=loss_name,
    )
    assert os.path.isfile(out)
    assert np.isfinite(ckpt["final_loss"])
    assert ckpt["config"]["kind"] == "trajectory"
    assert ckpt["config"]["trajectory_K"] == 4
    assert ckpt["config"]["loss_name"] == loss_name

    model, cfg = load_jam(str(out), device="cpu")
    assert cfg["d"] == 2
    assert cfg["dataset"] == "petals_2d"
    assert cfg["kind"] == "trajectory"

    V_fn = make_V_fn(model, device="cpu")
    pts = np.random.default_rng(0).uniform(0, cfg["L"], size=(16, 2))
    for t_eval in (0.0, 0.25, 0.5, 0.75, 1.0):
        V = V_fn(pts, t=t_eval)
        assert V.shape == (16,)
        assert np.all(np.isfinite(V)), t_eval
