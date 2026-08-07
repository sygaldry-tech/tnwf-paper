"""MPS-V trainer. Previously untested, despite producing the Table 2 oracles.

The cases here are the ones whose failure would be silent: a source width that
disagrees with the evaluation path, a target pool smaller than requested, or
early stopping that never fires. Each of those degrades the fit without raising
anything.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tnwf.mps_v import load_mps_v
from tnwf.mps_v.train import train
from tnwf.pipelines.run_evolution import (
    checkpoint_source_sigma, resolve_source_sigma,
)


def _train(tmp_path, **kw):
    kw.setdefault("N", 8)
    kw.setdefault("D", 4)
    kw.setdefault("N_t", 4)
    kw.setdefault("n_iter", 200)
    kw.setdefault("batch_size", 128)
    kw.setdefault("n_samples", 2000)
    kw.setdefault("log_every", 10**9)
    return train(dataset="gmm_2d", seed=0,
                 out_path=str(tmp_path / "o.pt"), **kw)


@pytest.mark.medium
class TestTrainer:
    def test_loss_decreases(self, tmp_path):
        ck = _train(tmp_path, n_iter=400)
        assert np.isfinite(ck["final_loss"])
        assert ck["final_loss"] < 0.0          # JAM loss goes negative as it fits

    def test_checkpoint_round_trips(self, tmp_path):
        ck = _train(tmp_path)
        model, cfg = load_mps_v(str(tmp_path / "o.pt"))
        assert (cfg["d"], cfg["N"], cfg["D"], cfg["N_t"]) == (
            int(ck["args"]["d"]), 8, 4, 4)
        x = torch.zeros(3, cfg["d"])
        with torch.no_grad():
            out = model(x, torch.full((3, 1), 0.5))
        assert out.shape == (3, 1) and torch.isfinite(out).all()

    def test_source_width_matches_the_evaluation_path(self, tmp_path):
        """A potential is only valid for the source it was fit against, and the
        mismatch is silent -- it costs ~5x in endpoint SW with nothing raised."""
        ck = _train(tmp_path)
        L = float(ck["args"]["L"])
        assert ck["args"]["sigma_0"] == pytest.approx(
            resolve_source_sigma("mps_v_tdvp2", L, None))
        assert checkpoint_source_sigma(ck["args"], L) == pytest.approx(
            ck["args"]["sigma_0"])

    def test_explicit_sigma0_is_honoured_and_recorded(self, tmp_path):
        ck = _train(tmp_path, sigma0=0.7)
        assert ck["args"]["sigma_0"] == pytest.approx(0.7)

    def test_pool_size_is_honoured(self, tmp_path):
        """The default comes from DATASET_DEFAULTS (10,000), which is thin for a
        tail-sensitive fit; an override that silently did nothing would be
        invisible in the loss."""
        ck = _train(tmp_path, n_samples=3000)
        assert ck["args"]["n_samples"] == 3000

    def test_early_stopping_fires_and_keeps_the_best_iterate(self, tmp_path):
        ck = _train(tmp_path, n_iter=5000, patience=1, eval_every=20,
                    val_frac=0.2)
        a = ck["args"]
        assert a["stopped_at"] < 5000, "patience=1 should stop well short"
        assert ck["val_history"], "no validation signal recorded"
        best = min(v for _, v in ck["val_history"])
        assert a["best_val"] == pytest.approx(best, abs=1e-6)

    def test_no_validation_split_when_early_stopping_is_off(self, tmp_path):
        ck = _train(tmp_path)
        assert ck["val_history"] == []
        assert ck["args"]["stopped_at"] == ck["args"]["n_iter"]


@pytest.mark.medium
def test_cosine_schedule_runs_and_changes_nothing_structural(tmp_path):
    """The schedule must not alter the checkpoint contract."""
    ck = _train(tmp_path, lr_schedule="cosine", sgdr_t0=50)
    assert ck["args"]["lr_schedule"] == "cosine"
    assert set(ck) >= {"model", "args", "model_type", "final_loss"}
