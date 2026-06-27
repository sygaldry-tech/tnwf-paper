"""Leaderboard aggregation + sort tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tnwf.leaderboard import rebuild, to_markdown, write_csv, write_markdown


def _make_npz(path: Path, *, dataset, method, seed, N=4, d=2, K=2, sw=0.5, mmd=0.3, nll=2.0, chi=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method=np.asarray(method), dataset=np.asarray(dataset),
        seed=np.asarray(seed), N=np.asarray(N), d=np.asarray(d), K=np.asarray(K), L=np.asarray(4.0),
        sw=np.asarray([sw]), mmd=np.asarray([mmd]), nll=np.asarray([nll]),
        chi_max=np.asarray([chi]),
        samples_T=np.zeros((1, d), dtype=np.float32),
        target=np.zeros((1, d), dtype=np.float32),
        samples_per_step=np.zeros((1, 1, d), dtype=np.float32),
    )


@pytest.mark.needle
def test_rebuild_aggregates_per_method(tmp_path):
    root = tmp_path / "results"
    for s in range(3):
        _make_npz(root / "swiss_roll_2d" / "dense" / f"seed{s}.npz",
                  dataset="swiss_roll_2d", method="dense", seed=s, sw=0.10 + 0.01 * s)
        _make_npz(root / "swiss_roll_2d" / "aci" / f"seed{s}.npz",
                  dataset="swiss_roll_2d", method="aci", seed=s, sw=0.20 + 0.01 * s)

    rows = rebuild(root)
    assert len(rows) == 2
    by_method = {r.method: r for r in rows}
    assert abs(by_method["dense"].sw_mean - 0.11) < 1e-6
    assert abs(by_method["aci"].sw_mean - 0.21) < 1e-6
    assert by_method["dense"].n_seeds == 3
    assert by_method["aci"].n_seeds == 3


@pytest.mark.needle
def test_to_markdown_sort(tmp_path):
    root = tmp_path / "results"
    _make_npz(root / "ds" / "fast" / "seed0.npz", dataset="ds", method="fast", seed=0, sw=0.1)
    _make_npz(root / "ds" / "slow" / "seed0.npz", dataset="ds", method="slow", seed=0, sw=0.5)
    rows = rebuild(root)
    md = to_markdown(rows, sort_by="sw_mean")
    # fast should come first, slow second
    assert md.find("fast") < md.find("slow")


@pytest.mark.needle
def test_subdir_grouping(tmp_path):
    root = tmp_path / "results"
    _make_npz(root / "gmm_3d" / "dense" / "N8_K8" / "seed0.npz",
              dataset="gmm_3d", method="dense", seed=0, sw=0.1, N=8, K=8)
    _make_npz(root / "gmm_3d" / "dense" / "N16_K16" / "seed0.npz",
              dataset="gmm_3d", method="dense", seed=0, sw=0.05, N=16, K=16)

    rows = rebuild(root)
    assert len(rows) == 2
    subs = {r.sub for r in rows}
    assert subs == {"N8_K8", "N16_K16"}


@pytest.mark.needle
def test_write_csv_md_smoke(tmp_path):
    root = tmp_path / "results"
    _make_npz(root / "ds" / "m1" / "seed0.npz", dataset="ds", method="m1", seed=0)
    rows = rebuild(root)
    write_csv(rows, path=tmp_path / "lb.csv")
    write_markdown(rows, path=tmp_path / "lb.md")
    assert (tmp_path / "lb.csv").exists()
    assert (tmp_path / "lb.md").read_text().startswith("# tnWF leaderboard")
