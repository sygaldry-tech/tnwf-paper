"""Figure 4 of the paper: cross-d scaling of MPS cost relative to Dense.

1 row x 3 cols:

  (A) MPS / Dense memory  vs d   (log y).  Markers only.
  (B) Walltime / Dense walltime  vs d  (log y).  Markers only.
  (C) Best-cell accuracy (SW)  vs d  (linear y).  Markers only; dashed
      line at the ~0.1 good-reconstruction rule of thumb.

Filtered to {Dense, TDVP1, TDVP2}.

Usage:
    uv run python scripts/make_fig4.py \\
        --out figures/fig4_cost_scaling.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
                   default="figures/fig4_cost_scaling.pdf")
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
            table[(d, m)] = {**best, "mem": mem, "d": d}

    # Time ratio: extrapolate Dense walltime if absent at high d.
    #
    # TIMER is `evolve_time` (the sum of step_times), not `total_time`.
    # total_time brackets the whole Trotter loop and so includes the per-step
    # metric callback (sampling, SW, MMD, NLL), which is dominated by mmd_rbf's
    # three n x n kernels. That overhead is 0.5-3% at d >= 4 but over 90% at
    # d=2, enough to make d=2 look *more* expensive than d=3 -- impossible for
    # an N^d method, and it flattened the fit the d=8 projection rests on.
    # Figure 6(b)'s caption says "total evolution time", which is this quantity.
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

    ds_all = sorted({d for d, _ in DATASETS})
    dense_ds = [d for d, _ in DATASETS if (d, "dense") in table]
    dense_max_d = max(dense_ds) if dense_ds else max(ds_all)

    for ax, key, ylabel, drop_d2, panel_letter in (
        (ax_mem_s,  "mem",        "MPS / Dense memory",        True,  "A"),
        (ax_wall_s, "time_ratio", "Walltime / Dense walltime", False, "B"),
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
        ax.set_xlabel("d  (spatial dimension)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.axvspan(min(ds_all) - 0.3, dense_max_d + 0.3, color="gray",
                   alpha=0.08, zorder=0)
        ax.text((min(ds_all) + dense_max_d) / 2, 0.02, "Dense feasible",
                transform=ax.get_xaxis_transform(), va="bottom", ha="center",
                fontsize=10, color="dimgray")
        ax.text(0.02, 1.02, f"({panel_letter.lower()})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=18, fontweight="bold")

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
    ax_acc.text((min(ds_all) + dense_max_d) / 2, 0.02, "Dense feasible",
                transform=ax_acc.get_xaxis_transform(), va="bottom",
                ha="center", fontsize=10, color="dimgray")
    ymax = max(sw_vals) if sw_vals else 0.15
    ax_acc.set_ylim(0, ymax * 1.35)
    ax_acc.set_xticks(list(ds_all))
    ax_acc.set_xticklabels([str(int(d)) for d in ds_all])
    ax_acc.set_xlabel("d  (spatial dimension)", fontsize=14)
    ax_acc.set_ylabel("Best-cell accuracy (SW)", fontsize=14)
    ax_acc.grid(True, which="both", alpha=0.25)
    ax_acc.text(0.02, 1.02, "(c)", transform=ax_acc.transAxes,
                ha="left", va="bottom", fontsize=18, fontweight="bold")

    handles, labels = ax_mem_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=18, frameon=False,
               handlelength=1.2, columnspacing=2.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
