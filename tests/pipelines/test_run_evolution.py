"""End-to-end pipeline cross-check: all 6 non-Dense methods within tol of Dense."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tnwf.jam.train import train
from tnwf.pipelines.run_evolution import run


@pytest.fixture(scope="module")
def jam_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("jam") / "swiss_roll_seed0.pt"
    train(
        dataset="swiss_roll_2d", seed=0, out_path=str(out),
        n_iter=100, batch_size=64,
        hidden=16, n_layers=2, time_embed_dim=8, log_every=100, lr=5e-3,
    )
    return str(out)


@pytest.fixture(scope="module")
def jam_ckpt_petals(tmp_path_factory):
    """Tiny JAM trained in trajectory mode on petals (K_data=4)."""
    out = tmp_path_factory.mktemp("jam") / "petals_seed0.pt"
    train(
        dataset="petals_2d", seed=0, out_path=str(out),
        n_iter=100, batch_size=64,
        hidden=16, n_layers=2, time_embed_dim=8, log_every=100, lr=5e-3,
    )
    return str(out)


@pytest.mark.medium
def test_dense_runs(jam_ckpt, tmp_path):
    out = run(
        method="dense", dataset="swiss_roll_2d", jam_ckpt=jam_ckpt,
        seed=0, N=4, K=2, n_samples=200, save=True, out_dir=tmp_path,
    )
    assert out["sw"].shape == (3,)            # K+1 snapshots
    assert np.all(np.isfinite(out["sw"]))
    assert out["samples_T"].shape == (200, 2)
    assert os.path.isfile(tmp_path / "seed0.npz")


@pytest.mark.medium
def test_jam_runs(jam_ckpt, tmp_path):
    out = run(
        method="jam", dataset="swiss_roll_2d", jam_ckpt=jam_ckpt,
        seed=0, N=4, K=2, n_samples=200, save=True, out_dir=tmp_path,
    )
    assert out["sw"].shape == (3,)
    assert np.all(np.isfinite(out["sw"]))
    assert np.all(np.isfinite(out["nll"]))         # JAM uses bin-density NLL
    assert out["samples_T"].shape == (200, 2)
    assert os.path.isfile(tmp_path / "seed0.npz")


@pytest.mark.medium
@pytest.mark.parametrize("method", ["tci_als", "aci", "tci_tdvp1", "tci_tdvp2"])
def test_mps_method_runs(jam_ckpt, method, tmp_path):
    method_kwargs = {"D_max": 16, "D_V": 8, "D_out": 16}
    out = run(
        method=method, dataset="swiss_roll_2d", jam_ckpt=jam_ckpt,
        seed=0, N=4, K=2, n_samples=200, save=False,
        method_kwargs=method_kwargs,
    )
    assert out["sw"].shape == (3,)
    assert np.all(np.isfinite(out["sw"]))
    assert out["samples_T"].shape == (200, 2)


# ── Trajectory-dataset path ───────────────────────────────────────────────


@pytest.mark.medium
@pytest.mark.parametrize("method",
                         ["jam", "dense", "tci_als", "aci", "tci_tdvp1", "tci_tdvp2"])
def test_petals_trajectory_runs(jam_ckpt_petals, method, tmp_path):
    """Trajectory path: sw[k] is compared against bio snapshot k, length K+1."""
    method_kwargs = ({} if method == "dense"
                     else {"D_max": 16, "D_V": 8, "D_out": 16})
    out = run(
        method=method, dataset="petals_2d", jam_ckpt=jam_ckpt_petals,
        seed=0, N=4, K=4, n_samples=200, save=True, out_dir=tmp_path,
        method_kwargs=method_kwargs,
    )
    # trajectory_K=4 ⇒ K+1=5 SW values, one per bio snapshot
    assert out["sw"].shape == (5,)
    assert np.all(np.isfinite(out["sw"]))
    assert out["samples_T"].shape == (200, 2)
    # target_traj must be persisted and have the right shape
    saved = np.load(tmp_path / "seed0.npz")
    assert "target_traj" in saved.files
    assert saved["target_traj"].shape == (5, 200, 2)


@pytest.mark.medium
def test_petals_K_mismatch_raises(jam_ckpt_petals):
    """K != trajectory_K must raise immediately to prevent silent misalignment."""
    with pytest.raises(ValueError, match="trajectory_K"):
        run(
            method="dense", dataset="petals_2d", jam_ckpt=jam_ckpt_petals,
            seed=0, N=4, K=2, n_samples=100, save=False,
        )
