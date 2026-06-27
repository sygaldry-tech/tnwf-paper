"""Fig C: minimum cost to hit each accuracy threshold across d.

Single figure, 2 rows × 3 columns (d=3, 4, 5):
  - Top row: minimum MEMORY (MPS / Dense ratio) needed to reach each of 3
    SW thresholds.
  - Bottom row: minimum WALLTIME (s) needed.

Thresholds are relative multiples of the best classical SW per d:
  - 1.00× = match the best
  - 1.25× = within 25%
  - 1.50× = within 50%

For each (d, method, threshold), plot the minimum cost cell that satisfies
SW ≤ threshold. Methods that fail to meet the threshold get a hollow bar
at their best-achievable cost (annotated as "no cell ≤ threshold").

Usage:
    uv run python scripts/make_fig_cost_to_target.py \
        --out results/cost_to_target.pdf
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
    METHOD_COLORS, METHOD_LABEL, WAVE_METHODS,
    best_classical_sw, collect, memory_fraction,
)


DATASETS = [
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
]
THRESHOLD_MULTIPLIERS = [1.00, 1.25, 1.50]


def memory_ratio(method: str, c: dict, d: int) -> float:
    if method == "dense":
        return 1.0
    return memory_fraction(c["N"], c["chi"], d)


def min_cost_to_target(cells: list[dict], target: float, cost_fn) -> tuple:
    """Return (cost, cell_or_None, met) — cost is min over satisfying cells
    if any satisfy SW ≤ target; otherwise the cost of the best-SW cell as a
    lower-bound proxy.
    """
    if not cells:
        return float("nan"), None, False
    sat = [c for c in cells if c["sw"] <= target]
    if sat:
        c = min(sat, key=cost_fn)
        return cost_fn(c), c, True
    c = min(cells, key=lambda x: x["sw"])
    return cost_fn(c), c, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/cost_to_target.pdf")
    args = p.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300,
                             sharey="row")
    fig.subplots_adjust(hspace=0.45, wspace=0.10, top=0.90, bottom=0.10)

    methods = WAVE_METHODS
    method_x = np.arange(len(methods))
    bar_width = 0.27

    summary_rows = []
    for col, (d, results_dir) in enumerate(DATASETS):
        res_dir = Path(results_dir)
        by_method = {m: collect(res_dir, m) for m in methods}
        ref = best_classical_sw(res_dir)
        sw_base = ref["best"]
        thresholds = [sw_base * mult for mult in THRESHOLD_MULTIPLIERS]
        print(f"\nd={d}: best classical SW={sw_base:.4f}  → thresholds {thresholds}",
              flush=True)

        for row, (cost_label, cost_fn, ax_units) in enumerate([
            ("memory", lambda c, m=None: memory_ratio(m, c, d), "MPS / Dense memory"),
            ("walltime", lambda c, m=None: c["time"], "Walltime (s)"),
        ]):
            ax = axes[row, col]
            for ti, (mult, thresh) in enumerate(zip(THRESHOLD_MULTIPLIERS, thresholds)):
                offset = (ti - 1) * bar_width
                xpos = method_x + offset
                heights = []
                hatches = []
                colors = []
                edges = []
                for m in methods:
                    cells = by_method[m]
                    cost_fn_m = (
                        (lambda c, _m=m: memory_ratio(_m, c, d))
                        if cost_label == "memory" else
                        (lambda c, _m=m: c["time"])
                    )
                    cost, cell, met = min_cost_to_target(cells, thresh, cost_fn_m)
                    heights.append(cost if cost == cost else 0.0)
                    if met:
                        hatches.append(None)
                        colors.append(METHOD_COLORS[m])
                        edges.append("black")
                    else:
                        hatches.append("//")
                        colors.append("white")
                        edges.append(METHOD_COLORS[m])
                    summary_rows.append({
                        "d": d, "method": m, "mult": mult, "cost_label": cost_label,
                        "cost": cost, "met": met,
                        "cell": cell,
                    })
                bars = ax.bar(xpos, heights, bar_width,
                              color=colors, edgecolor=edges, linewidth=1.2,
                              hatch=hatches[0] if all(h == hatches[0] for h in hatches) else None,
                              alpha=0.85,
                              label=f"{mult:.2f}×" if row == 0 and col == 0 else None)
                # Per-bar hatch handling: matplotlib bar() doesn't take per-bar hatches; reapply.
                for bar, h in zip(bars, hatches):
                    if h:
                        bar.set_hatch(h)

            ax.set_xticks(method_x)
            ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                               rotation=20, ha="right", fontsize=9)
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", alpha=0.25)
            if col == 0:
                ax.set_ylabel(ax_units, fontsize=11)
            ax.set_title(
                f"{cost_label} — d={d}  (best classical SW={sw_base:.3f})"
                if row == 0 else f"{cost_label} — d={d}",
                fontsize=11, fontweight="bold")

    # Threshold legend (top of figure)
    handles = []
    for mult in THRESHOLD_MULTIPLIERS:
        h = plt.Rectangle((0, 0), 1, 1, fc="lightgray", ec="black",
                          label=f"≤ {mult:.2f}× best classical")
        handles.append(h)
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec="dimgray",
                                  hatch="//",
                                  label="LB (no cell met threshold)"))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=10,
               frameon=False)

    fig.suptitle(
        "Minimum cost to reach SW thresholds  —  GMM, by dimension",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nsaved {out}", flush=True)

    # Print summary table
    n_met = sum(1 for r in summary_rows if r["met"])
    n_total = len(summary_rows)
    print(f"\nthresholds met: {n_met}/{n_total} (d × method × thresh × cost-axis)",
          flush=True)
    print("\n--- Cost-to-target table (cells that met threshold) ---", flush=True)
    print(f"{'d':>2} {'method':<11} {'mult':>5} {'axis':>8} {'cost':>10}  cell",
          flush=True)
    for r in summary_rows:
        if not r["met"]:
            continue
        c = r["cell"]
        cell_s = f"N={c['N']} K={c['K']} D={int(c['D'])} χ={c['chi']:.0f} SW={c['sw']:.3f}"
        print(f"{r['d']:>2} {r['method']:<11} {r['mult']:>5.2f} {r['cost_label']:>8} "
              f"{r['cost']:>10.3g}  {cell_s}", flush=True)


if __name__ == "__main__":
    main()
