"""Smoke tests for figure scripts: run pipeline → make plot, no crashes."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tnwf.jam.train import train
from tnwf.pipelines.run_evolution import run

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "swiss_roll_2d"


@pytest.mark.medium
def test_make_fig1_runs(tmp_path):
    # Train tiny JAM checkpoint
    ckpt = tmp_path / "jam.pt"
    train(
        dataset="swiss_roll_2d", seed=0, out_path=str(ckpt),
        n_iter=50, batch_size=64,
        hidden=8, n_layers=2, time_embed_dim=4, log_every=200, lr=5e-3,
    )

    # Run dense + 2 mps methods at tiny config
    results_dir = tmp_path / "results" / "swiss_roll_2d"
    for method in ("dense", "tci_tdvp1", "tci_tdvp2"):
        run(
            method=method, dataset="swiss_roll_2d", jam_ckpt=str(ckpt),
            seed=0, N=4, K=2, n_samples=200, save=True,
            out_dir=results_dir / method,
            method_kwargs={"D_max": 8, "D_V": 4, "D_out": 8},
        )

    # Run make_fig1.py against this directory
    out_pdf = tmp_path / "fig1.pdf"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "make_fig1.py"),
         "--results_dir", str(results_dir),
         "--out", str(out_pdf)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"make_fig1.py failed: {proc.stderr}"
    assert out_pdf.exists()
