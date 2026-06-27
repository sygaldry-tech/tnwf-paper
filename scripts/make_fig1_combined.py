"""Combined overview figure (former Figures 1 and 2 merged into one).

A single 2x2 figure:
  Top row -- the wavefunction-flow concept (from make_fig1_concept.py):
    (a) wavefunction-flow lift: source -> target
    (b) classical / quantum bifurcation of the shared velocity oracle
  Bottom row -- the tensor-network realization (from
  make_fig2_tensor_networks.py):
    (c) MPS encoding of the flow state
    (d) generator-TT velocity step via TDVP

The two source scripts remain usable standalone; this script reuses their
panel-drawing functions so the panels stay in sync.

Usage:
    uv run python scripts/make_fig1_combined.py \\
        --out figures/fig1_overview.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import make_fig1_concept as f1            # noqa: E402
import make_fig2_tensor_networks as f2    # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig1_overview.pdf")
    args = p.parse_args()

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6.6), dpi=300, constrained_layout=True,
        gridspec_kw={"height_ratios": [3.8, 3.0]})
    fig.set_constrained_layout_pads(w_pad=0.30, h_pad=0.45)

    # ── Top row: the concept ────────────────────────────────────────────
    f1._draw_panel_a(axes[0, 0])
    f1._draw_panel_b(axes[0, 1])

    # ── Bottom row: the tensor network realization ──────────────────────
    # Reuse the Fig-2 panels but drop their titles, relabel (a)/(b) -> (c)/(d)
    # with a small corner letter (matching panels a/b).
    f2._draw_panel_a(axes[1, 0])
    axes[1, 0].set_title("", loc="left")   # clear the left-aligned panel title
    axes[1, 0].text(0.0, 1.0, "(c)", transform=axes[1, 0].transAxes,
                    fontsize=16, fontweight="bold", va="top", ha="left")
    f2._draw_panel_b(axes[1, 1])
    axes[1, 1].set_title("", loc="left")
    axes[1, 1].text(0.0, 1.0, "(d)", transform=axes[1, 1].transAxes,
                    fontsize=16, fontweight="bold", va="top", ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
