"""The package version is declared twice; keep the copies in step.

`pyproject.toml` and `src/tnwf/__init__.py` each state it, and nothing else
reads `__version__`, so a drift between them would go unnoticed until it
appeared in a published citation.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import tnwf

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.mark.needle
class TestVersion:
    def test_matches_pyproject(self):
        declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
        assert tnwf.__version__ == declared, (
            f"tnwf.__version__ is {tnwf.__version__!r} but pyproject.toml "
            f"declares {declared!r}; update both."
        )
