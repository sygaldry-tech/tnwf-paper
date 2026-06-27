"""Fig B: Cost-vs-accuracy Pareto — memory and walltime, per d.

Two stacked panels:
  - Top: x = MPS memory / Dense memory (log), y = SW (log).
  - Bottom: x = total_time (log s), y = SW (log).
Per-method scatter + Pareto staircase. Reference horizontal lines for
SW_dense_best and SW_jam_best. Reference vertical x=1 line on memory panel
(MPS = Dense).

Usage:
    uv run python scripts/make_fig_cost_pareto.py \
        --results_dir data/gmm_5d_hp --d 5 \
        --out results/gmm_5d_hp/cost_pareto.pdf
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
    METHOD_COLORS, METHOD_LABEL, METHOD_MARKERS, WAVE_METHODS,
    best_classical_sw, collect, memory_fraction,
)


def pareto_min_min(xs: np.ndarray, ys: np.ndarray) -> tuple:
    """Return (xs_p, ys_p) on the lower-left Pareto frontier of points
    (xs, ys), sorted by ascending x. Both axes minimised.
    """
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    by_method = {m: collect(res_dir, m) for m in WAVE_METHODS}
    refs = best_classical_sw(res_dir)

    # Diagnostics
    print(f"d={args.d}, results_dir={res_dir}", flush=True)
    for m, cells in by_method.items():
        if cells:
            sws = [c["sw"] for c in cells]
            times = [c["time"] for c in cells if c["time"] == c["time"]]
            t_str = f"{min(times):.0f}–{max(times):.0f}s" if times else "(no timing)"
            print(f"  {m:10s}: {len(cells)} cells | SW {min(sws):.3f}–{max(sws):.3f} | wall {t_str}",
                  flush=True)
    print(f"  best classical: jam={refs['jam']:.4f}  dense={refs['dense']:.4f}",
          flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 11), dpi=300,
                              constrained_layout=True)
    ax_mem, ax_time = axes
    n_dom_total = 0
    n_pareto_total = 0

    for method in WAVE_METHODS:
        cells = by_method.get(method, [])
        if not cells:
            continue
        # Memory: Dense by definition uses N^d cells → ratio = 1. For MPS
        # methods we use the observed χ from chi_max[-1].
        if method == "dense":
            xs_mem = np.ones(len(cells))
        else:
            xs_mem = np.array(
                [memory_fraction(c["N"], c["chi"], args.d) for c in cells]
            )
        # Walltime
        xs_time = np.array([c["time"] for c in cells])
        ys = np.array([c["sw"] for c in cells])

        for ax, xs, label in ((ax_mem, xs_mem, "memory"),
                              (ax_time, xs_time, "wall")):
            mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
            if mask.sum() == 0:
                continue
            ax.scatter(xs[mask], ys[mask], s=22, alpha=0.55,
                       color=METHOD_COLORS[method],
                       marker=METHOD_MARKERS[method],
                       edgecolors="black", linewidths=0.4,
                       label=METHOD_LABEL[method] if ax is ax_mem else None,
                       zorder=2)
            xp, yp = pareto_min_min(xs[mask], ys[mask])
            ax.plot(xp, yp, "-", lw=1.8, color=METHOD_COLORS[method],
                    alpha=0.85, drawstyle="steps-post", zorder=3)
            if ax is ax_mem:
                n_dom_total += int(mask.sum())
                n_pareto_total += len(xp)

    # Reference lines
    for ax in (ax_mem, ax_time):
        for name, val, ls in (("Dense best", refs["dense"], "--"),
                              ("JAM best",  refs["jam"],  ":")):
            if val == val:
                ax.axhline(val, color="dimgray", lw=1, ls=ls, alpha=0.7,
                           zorder=1)
                ax.text(0.99, val * 1.04, name, fontsize=8, color="dimgray",
                        transform=ax.get_yaxis_transform(), ha="right")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel("SW (final, log)", fontsize=11)
        ax.grid(True, which="both", alpha=0.25)

    # Memory: x=1 reference. Annotation parked at the very top of the
    # axes-area so it doesn't sit on top of the data cluster at x=1.
    ax_mem.axvline(1.0, color="dimgray", lw=1, ls="--", alpha=0.5, zorder=1)
    ax_mem.annotate("MPS = Dense memory", xy=(1.0, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points",
                    fontsize=8, color="dimgray", ha="left", va="top")

    ax_mem.set_xlabel("MPS memory / Dense memory  (log)", fontsize=11)
    ax_time.set_xlabel("Walltime per run (s, log)", fontsize=11)
    ax_mem.set_title("(A)  Memory Pareto", fontsize=12, fontweight="bold",
                     loc="left")
    ax_time.set_title("(B)  Walltime Pareto", fontsize=12, fontweight="bold",
                      loc="left")
    # Legend on a strip above the figure so it doesn't occlude the Dense+TDVP
    # cluster at the high-x / low-SW corner.
    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               fontsize=10, frameon=False)

    fig.suptitle(
        f"Cost vs accuracy Pareto at d={args.d}  —  GMM  "
        f"(scatter = (N,K,D) cell; staircase = per-method Pareto front)",
        fontsize=12, fontweight="bold", y=1.04,
    )
    print(f"  total cells plotted: {n_dom_total}, on Pareto: {n_pareto_total}",
          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
