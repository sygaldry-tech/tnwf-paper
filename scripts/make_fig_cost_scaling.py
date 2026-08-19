"""Figure 6 of the paper: cross-d scaling of MPS cost relative to Dense.

1 row x 3 cols:

  (A) MPS / Dense memory  vs d   (log y).  Markers only.
  (B) Evolution Walltime / Dense  vs d  (log y).  Markers only.
  (C) Best-cell accuracy (SW)  vs d  (linear y).  Markers only; dashed
      line at the ~0.1 good-reconstruction rule of thumb.

Filtered to {Dense, TDVP1, TDVP2}.

Usage:
    uv run python scripts/make_fig_cost_scaling.py \\
        --out figures/fig_cost_scaling.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Computer Modern for math. The default fontset, "dejavusans", draws \mathcal from a
# cursive face (mathtext.cal) but everything else from DejaVu Sans italic
# (mathtext.it = "sans:italic"), so R and S came out script while psi, chi and d came
# out sans -- one expression, two families, and the psi looked nothing like LaTeX's.
# "cm" is the face LaTeX itself uses, so the labels now match the manuscript's math.
# Non-math text (titles, ticks, legend) stays sans-serif by design.
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _hp_utils import (
    METHOD_COLORS,
    METHOD_LABEL,
    METHOD_MARKERS,
    collect,
    memory_fraction,
)

SCALING_METHODS = ["dense", "tci_tdvp1", "tci_tdvp2"]

# ── Font sizes, quoted at the size they RENDER in the paper ──────────────
# This figure is 18.5 in wide and sn-article.tex includes it at
# 0.98\textwidth, i.e. about 6.2 in, so the PDF is scaled down by ~0.335 and
# every font shrinks with it. Sizes are therefore written below as the point
# size they should end up at on the printed page, and _pt() converts to the
# matplotlib value. The converted numbers look far too large in isolation --
# a 9 pt axis label becomes fontsize 27 -- which is exactly why the original
# hand-picked values were wrong: fontsize=14 rendered at 4.7 pt and
# fontsize=10 at 3.3 pt, against a caption set in roughly 9 pt. Chad's review
# on 17 August 2026 flagged the result as unreadable.
#
# Keep _FIG_W_IN in step with the figsize below, and _RENDERED_W_IN with the
# includegraphics width in sn-article.tex; if either moves, every font moves.
_FIG_W_IN = 18.5
_RENDERED_W_IN = 6.2


def _pt(rendered: float) -> float:
    """The matplotlib fontsize that renders at `rendered` points in the paper."""
    return round(rendered * _FIG_W_IN / _RENDERED_W_IN, 1)


FS_LABEL = _pt(9.0)     # axis labels, at parity with the caption
FS_TICK = _pt(8.0)      # tick labels; previously the rcParams default
FS_ANNOT = _pt(8.0)     # in-axes annotations ("Dense-grid feasible")
FS_PANEL = _pt(9.0)     # (a)/(b)/(c) panel letters; bold at caption size, not above it
FS_LEGEND = _pt(9.0)
FS_TITLE = _pt(8.0)     # per-panel titles, matching fig_ksweep.py's FS_GLOSS

# Panel-letter placement, in axes coordinates. Left of x=0 and above y=1 puts the
# letter northwest of the plot area's top-left corner rather than inside it, which
# is where it used to sit. Kept in step with the same constants in fig_ksweep.py so
# the two figures place their letters identically.
#
# Y was 1.02, which sat the letter level with the topmost y tick label and, in panel
# (a) where that label is the widest (10^7), the two collided. Raised to clear the
# tick row entirely; the letter now sits alongside the panel title instead.
PANEL_LABEL_X = -0.155
PANEL_LABEL_Y = 1.12

# Per-panel titles. Same role as PANEL_GLOSS in fig_ksweep.py: name the quantity in
# words so the panel is readable without decoding the axis label first.
# Titles name the quantity in words and stay free of notation; the y-axis carries the
# bare symbol and the caption defines it. R and S were the available letters: A is the
# action functional and F the discrete Fourier transform in tn-wf supplementary/si.tex,
# so neither mnemonic (A for acceleration, F for factor) could be used.
PANEL_TITLE = {"A": "Compression Ratio",
               "B": "Acceleration Factor",
               "C": "Accuracy"}

DATASETS = [
    (2, "data/gmm_2d_hp"),
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
    (6, "data/gmm_6d_hp"),
    (7, "data/gmm_7d_hp"),
    (8, "data/gmm_8d_hp"),
]
#: Which timer backs panel (b). See the comment at the fit below.
TIMER = "evolve_time"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig_cost_scaling.pdf")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.22, h_pad=0.22)
    ax_mem_s, ax_wall_s, ax_acc = axes

    # ── Cross-d scaling (best-cell trajectory) ──────────────────────────
    table = {}
    for d, results_dir in DATASETS:
        for m in SCALING_METHODS:
            cells = collect(Path(results_dir), m)
            if not cells:
                continue
            best = min(cells, key=lambda c: c["sw"])
            mem = (1.0 if m == "dense"
                   else memory_fraction(best["N"], best["chi"], d))
            # "mem" stays as the MPS/Dense fraction because make_fig_supp_pareto.py
            # and the Pareto panels reason in that direction; "compression" is the
            # reciprocal, which is what Figure 6(a) reports.
            table[(d, m)] = {**best, "mem": mem, "d": d,
                             "compression": (1.0 / mem if mem and mem == mem and mem > 0
                                             else float("nan"))}

    # Time ratio: extrapolate Dense walltime if absent at high d.
    #
    # TIMER is `evolve_time` (the sum of step_times), not `total_time`.
    # total_time brackets the whole Trotter loop and so includes the per-step
    # metric callback (sampling, SW, MMD, NLL), which is dominated by mmd_rbf's
    # three n x n kernels. That overhead is 0.5-3% at d >= 4 but over 90% at
    # d=2, enough to make d=2 look *more* expensive than d=3 -- impossible for
    # an N^d method, and it flattened the fit the d=8 projection rests on.
    # Figure 6(b)'s caption calls this the evolution walltime: the summed
    # per-Trotter-step wall clock, excluding the per-step metric callback.
    dense_pts = [(d, table[(d, "dense")][TIMER])
                 for d, _ in DATASETS
                 if (d, "dense") in table
                 and table[(d, "dense")][TIMER] == table[(d, "dense")][TIMER]]
    if len(dense_pts) >= 2:
        ds_d, ts_d = zip(*dense_pts)
        slope_d, intercept_d = np.polyfit(ds_d, np.log10(ts_d), 1)
        print(f"[fig6b] Dense {TIMER} fit over d={list(ds_d)}: "
              f"log10 t = {slope_d:.3f} d + {intercept_d:.3f}")
        for d, t in dense_pts:
            print(f"         d={d}: {t:10.1f} s")
    else:
        slope_d, intercept_d = (float("nan"), float("nan"))
    # Which d have no measured Dense timing and are therefore projected. Stated
    # out loud: a silently extrapolated point reads as a measurement.
    projected = [d for d, _ in DATASETS
                 if (d, "dense") not in table
                 or table[(d, "dense")][TIMER] != table[(d, "dense")][TIMER]]
    if projected:
        print(f"[fig6b] Dense walltime PROJECTED (not measured) at d={projected}")

    def t_dense_at(d):
        c = table.get((d, "dense"))
        if c is not None and c[TIMER] == c[TIMER]:
            return c[TIMER]
        if slope_d == slope_d:
            return 10 ** (slope_d * d + intercept_d)
        return float("nan")

    for d, _ in DATASETS:
        t_d = t_dense_at(d)
        for m in SCALING_METHODS:
            cell = table.get((d, m))
            if cell is None:
                continue
            cell["time_ratio"] = (cell[TIMER] / t_d
                                  if (t_d and t_d == t_d
                                      and cell[TIMER] == cell[TIMER])
                                  else float("nan"))
            tr = cell["time_ratio"]
            # Acceleration factor: Dense evolution walltime divided by the tensor
            # network's, i.e. the reciprocal of time_ratio. Panel (b) reports it in
            # this direction so it rises above unity as d grows, matching panel (a)'s
            # compression factor; "time_ratio" itself stays MPS/Dense because the
            # Pareto panels reason in that direction.
            cell["acceleration"] = (1.0 / tr if tr == tr and tr > 0 else float("nan"))

    ds_all = sorted({d for d, _ in DATASETS})
    dense_ds = [d for d, _ in DATASETS if (d, "dense") in table]
    dense_max_d = max(dense_ds) if dense_ds else max(ds_all)

    for ax, key, ylabel, drop_d2, panel_letter in (
        # The y-label is the bare symbol; the title above spells out what it means.
        # Writing the words on both would name the axis twice, and panel (c) already
        # carries a symbol -- SW(psi_data) -- so this also makes the three consistent.
        (ax_mem_s,  "compression",  "$\\mathcal{R}$", True,  "A"),
        (ax_wall_s, "acceleration", "$\\mathcal{S}$", False, "B"),
    ):
        is_memory_panel = (ax is ax_mem_s)
        for m in SCALING_METHODS:
            xs = []; ys = []
            for d in ds_all:
                if (d, m) not in table:
                    continue
                v = table[(d, m)].get(key, float("nan"))
                if isinstance(v, float) and v != v:
                    continue
                xs.append(d); ys.append(v)
            if not xs:
                continue
            ax.plot(xs, ys, ls="none",
                    marker=METHOD_MARKERS[m], ms=18,
                    markerfacecolor=METHOD_COLORS[m],
                    markeredgecolor="black", markeredgewidth=0.7,
                    alpha=0.9, label=METHOD_LABEL[m] if is_memory_panel else None,
                    zorder=3)
        ax.axhline(1.0, color="dimgray", lw=1, ls="--", alpha=0.7, zorder=0)
        ax.set_xticks(list(ds_all))
        ax.set_xticklabels([str(int(d)) for d in ds_all])
        # Math, not plain text: the manuscript sets the spatial dimension as $d$
        # (italic) throughout, and fig_ksweep.py already labels its axes in mathtext.
        ax.set_xlabel("$d$", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.axvspan(min(ds_all) - 0.3, dense_max_d + 0.3, color="gray",
                   alpha=0.08, zorder=0)
        # Top of the axes, not the bottom. Panels (a) and (b) now report factors
        # greater than one, so the Dense series sits on the y=1 line at the very
        # bottom -- exactly where this annotation used to go, and it was struck
        # through by the markers. Panel (c) still plots absolute SW and keeps its
        # label low, where there is room.
        ax.text((min(ds_all) + dense_max_d) / 2, 0.98, "Dense-grid feasible",
                transform=ax.get_xaxis_transform(), va="top", ha="center",
                fontsize=FS_ANNOT, color="dimgray")
        ax.set_title(PANEL_TITLE[panel_letter], fontsize=FS_TITLE,
                     color="0.25", pad=10)
        ax.text(PANEL_LABEL_X, PANEL_LABEL_Y, f"({panel_letter.lower()})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_PANEL, fontweight="bold")

    # ── Accuracy panel (C): best-cell SW vs d (measured d only) ─────────
    sw_vals = []
    for m in SCALING_METHODS:
        xs = []; ys = []
        for d in ds_all:
            cell = table.get((d, m))
            if cell is None:
                continue
            sw = cell.get("sw", float("nan"))
            if isinstance(sw, float) and sw != sw:
                continue
            xs.append(d); ys.append(sw); sw_vals.append(sw)
        if not xs:
            continue
        ax_acc.plot(xs, ys, ls="none", marker=METHOD_MARKERS[m], ms=18,
                    markerfacecolor=METHOD_COLORS[m], markeredgecolor="black",
                    markeredgewidth=0.7, alpha=0.9, zorder=3)
    ax_acc.axhline(0.1, color="dimgray", lw=1.2, ls="--", alpha=0.8, zorder=1)
    ax_acc.axvspan(min(ds_all) - 0.3, dense_max_d + 0.3, color="gray",
                   alpha=0.08, zorder=0)
    ax_acc.text((min(ds_all) + dense_max_d) / 2, 0.98, "Dense-grid feasible",
                transform=ax_acc.get_xaxis_transform(), va="top",
                ha="center", fontsize=FS_ANNOT, color="dimgray")
    ymax = max(sw_vals) if sw_vals else 0.15
    ax_acc.set_ylim(0, ymax * 1.35)
    ax_acc.set_xticks(list(ds_all))
    ax_acc.set_xticklabels([str(int(d)) for d in ds_all])
    ax_acc.set_xlabel("$d$", fontsize=FS_LABEL)
    ax_acc.tick_params(labelsize=FS_TICK)
    # psi_T, not psi_data: the quantity is measured on the state the flow ends at,
    # and psi_T is what the manuscript calls it (SW^wf_T in Table 2). Set as one math
    # expression so SW is upright, matching \mathrm{SW} in the text.
    ax_acc.set_ylabel("$\\mathrm{SW}(\\Psi_T)$", fontsize=FS_LABEL)
    ax_acc.grid(True, which="both", alpha=0.25)
    ax_acc.set_title(PANEL_TITLE["C"], fontsize=FS_TITLE, color="0.25", pad=10)
    ax_acc.text(PANEL_LABEL_X, PANEL_LABEL_Y, "(c)", transform=ax_acc.transAxes,
                ha="left", va="bottom", fontsize=FS_PANEL, fontweight="bold")

    handles, labels = ax_mem_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=FS_LEGEND, frameon=False,
               handlelength=1.2, columnspacing=2.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
