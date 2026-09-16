"""Supplementary Figure: cost-vs-accuracy Pareto frontiers.

Full-page 2 rows x 3 cols:
  Row 1 (memory):   SW vs MPS/Dense memory ratio  at d = 3, 4, 5.
  Row 2 (wall-clock): SW vs wall-clock time per run (s)  at d = 3, 4, 5.

Each cell plots SW (log y) against the cost axis (log x). Points are
hyperparameter cells; per-method Pareto staircases are overlaid;
dashed/dotted lines mark Dense/JAM reference SW. The d=3, 4, 5
sweeps each have ~30+ HP cells per method, so each panel has a
substantive Pareto staircase; d>=6 was sampled too sparsely for a
meaningful Pareto.

Usage:
    uv run python scripts/make_fig_supp_pareto.py \\
        --out figures/figS1_supp_pareto.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Computer Modern for math, as in the other figure generators; this figure's
# axis labels carry $d=...$ and SW and were rendering in DejaVu Sans.
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _hp_utils import (
    METHOD_COLORS,
    METHOD_LABEL,
    METHOD_MARKERS,
    WAVE_METHODS,
    best_classical_sw,
    collect,
    memory_fraction,
)

# Font sizes are quoted at the size they RENDER in the paper, not at the size
# matplotlib is handed. This figure is drawn 16 in wide and placed at
# 0.95\textwidth of IOP's 153 mm text block, i.e. 5.73 in, so everything is
# scaled down by a factor of nearly three on the page. Raw sizes put the axis
# labels at about 5 pt against IOP's 8-to-12 pt guidance for figure text.
# Keep _FIG_W_IN in step with the figsize below and _RENDERED_W_IN with the
# includegraphics width in appendices.tex; if either moves, every font moves.
_FIG_W_IN = 17.0
_RENDERED_W_IN = 5.73


def _pt(rendered: float) -> float:
    """The matplotlib fontsize that renders at `rendered` points in the paper."""
    return round(rendered * _FIG_W_IN / _RENDERED_W_IN, 1)


FS_LABEL = _pt(8.0)     # axis labels, at IOP's floor for figure text
FS_TICK = _pt(8.0)      # tick labels; previously the rcParams default
FS_ANNOT = _pt(8.0)     # in-axes reference annotations
FS_TITLE = _pt(8.5)     # per-panel titles, which carry the panel letter
FS_LEGEND = _pt(8.0)

DATASETS = {
    3: "data/gmm_3d_hp_v2",
    4: "data/gmm_4d_hp",
    5: "data/gmm_5d_hp",
}
D_VALUES = [3, 4, 5]

PARETO_METHODS = list(WAVE_METHODS)


def pareto_min_min(xs: np.ndarray, ys: np.ndarray) -> tuple:
    if len(xs) == 0:
        return xs, ys
    order = np.argsort(xs)
    x_sorted, y_sorted = xs[order], ys[order]
    keep_idx, best = [], np.inf
    for i, y in enumerate(y_sorted):
        if y < best:
            keep_idx.append(i)
            best = y
    return x_sorted[keep_idx], y_sorted[keep_idx]


def draw_pareto_panel(ax, d, results_dir, refs, xkind, *,
                       show_legend=False, panel_letter="",
                       show_ylabel=True):
    """Draw a single d=fixed cost-vs-accuracy Pareto panel."""
    by_method = {m: collect(results_dir, m) for m in PARETO_METHODS}
    for method in PARETO_METHODS:
        cells = by_method.get(method, [])
        if not cells:
            continue
        if xkind == "mem":
            if method == "dense":
                xs = np.ones(len(cells))
            else:
                xs = np.array(
                    [memory_fraction(c["N"], c["chi"], d) for c in cells]
                )
        elif xkind == "time":
            xs = np.array([c["time"] for c in cells])
        else:
            raise ValueError(xkind)
        ys = np.array([c["sw"] for c in cells])
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
        if mask.sum() == 0:
            continue
        ax.scatter(xs[mask], ys[mask], s=42, alpha=0.55,
                   color=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method],
                   edgecolors="black", linewidths=0.5,
                   label=METHOD_LABEL[method] if show_legend else None,
                   zorder=2)
        xp, yp = pareto_min_min(xs[mask], ys[mask])
        ax.plot(xp, yp, "-", lw=2.0, color=METHOD_COLORS[method],
                alpha=0.85, drawstyle="steps-post", zorder=3)
    # The two references nearly coincide at d=5, so their labels sit at
    # opposite ends of the axes. Both stay above their own line: dropping one
    # below puts it into the tick row wherever the line sits low.
    for name, val, ls, xpos, halign in (
            ("Dense-grid", refs["dense"], "--", 0.01, "left"),
            ("JAM", refs["jam"], ":", 0.99, "right")):
        if val == val:
            ax.axhline(val, color="dimgray", lw=1, ls=ls, alpha=0.7,
                       zorder=1)
            ax.text(xpos, val * 1.04, name, fontsize=FS_ANNOT,
                    color="dimgray",
                    transform=ax.get_yaxis_transform(), ha=halign)
    if xkind == "mem":
        ax.axvline(1.0, color="dimgray", lw=1, ls="--", alpha=0.5,
                   zorder=1)
    ax.set_yscale("log")
    ax.set_xscale("log")
    if show_ylabel:
        # Same quantity and same symbol as Figure 6's accuracy panel: the
        # endpoint sliced Wasserstein. "SW (final)" said it in words while
        # Figure 6 said it in math.
        ax.set_ylabel("$\\mathrm{SW}(\\Psi_T)$", fontsize=FS_LABEL)
    ax.grid(True, which="both", alpha=0.25)
    if xkind == "mem":
        ax.set_xlabel("MPS / Dense-grid memory",
                      fontsize=FS_LABEL)
        title_kind = "Memory"
    else:
        ax.set_xlabel("Wall-clock time per run (s)", fontsize=FS_LABEL)
        title_kind = "Wall-clock"
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax.tick_params(axis="both", which="minor", labelsize=FS_TICK)
    ax.set_title(rf"{panel_letter}  {title_kind},  $d={d}$",
                 fontsize=FS_TITLE, fontweight="bold", loc="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/figS1_supp_pareto.pdf")
    args = p.parse_args()

    refs_by_d = {
        d: best_classical_sw(Path(DATASETS[d])) for d in D_VALUES
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.22, h_pad=0.30)

    # Lower case, matching the rest of the paper and IOP's requirement that
    # figure parts carry "a lower-case letter in parentheses".
    panel_letters = [
        ["(a)", "(b)", "(c)"],
        ["(d)", "(e)", "(f)"],
    ]
    row_kinds = ["mem", "time"]

    for row, xkind in enumerate(row_kinds):
        for col, d in enumerate(D_VALUES):
            ax = axes[row, col]
            d_path = Path(DATASETS[d])
            refs = refs_by_d[d]
            draw_pareto_panel(
                ax, d, d_path, refs, xkind=xkind,
                show_legend=(row == 0 and col == 0),
                panel_letter=panel_letters[row][col],
                show_ylabel=(col == 0),
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg_top = fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=FS_LEGEND, frameon=False,
               handlelength=1.2, columnspacing=1.8)
    for h in leg_top.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([90])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
