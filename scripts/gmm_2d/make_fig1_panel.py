"""Fig 1 bottom half: gmm_2d demo. Same structure as swiss_roll_2d/make_fig1.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Re-use the swiss_roll_2d Fig-1 driver with a different dataset_label.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swiss_roll_2d.make_fig1 import main as _shared_main  # noqa: E402


if __name__ == "__main__":
    # Default: results/gmm_2d/, output to results/gmm_2d/fig1_panel.pdf
    sys.argv = (
        ["make_fig1_panel.py"]
        + ["--results_dir", "data/gmm_2d"]
        + ["--dataset_label", "gmm_2d"]
        + ["--out", "results/gmm_2d/fig1_panel.pdf"]
        + [a for a in sys.argv[1:] if a not in ("--results_dir", "--dataset_label", "--out")]
    )
    _shared_main()
