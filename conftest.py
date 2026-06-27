"""
conftest.py — test infrastructure for tnWF.

Tiered markers (mirrors the research prototype convention):
  needle     — quick smoke tests (<30s)
  medium     — pipeline tests (30s-5min)
  heavy_duty — full sweeps (>5min, Modal-only by default)

Usage:
  pytest                                 # all non-heavy_duty
  pytest -m needle                       # only needle
  pytest -m "needle or medium"           # both
  pytest --save-plots                    # persist plots to results/experiment_tests/
"""
from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save-plots",
        action="store_true",
        default=False,
        help="Persist experiment PNG outputs to results/experiment_tests/",
    )


@pytest.fixture
def plot_output_dir(request, tmp_path):
    """Per-test directory for PNG output. With --save-plots, persists to results/."""
    save = request.config.getoption("--save-plots")
    if save:
        repo_root = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(repo_root, "results", "experiment_tests", request.node.name)
        os.makedirs(out, exist_ok=True)
        return out
    out = str(tmp_path / "plots")
    os.makedirs(out, exist_ok=True)
    return out
