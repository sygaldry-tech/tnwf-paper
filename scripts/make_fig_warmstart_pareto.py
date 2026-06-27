"""Fig: warm-start variant Pareto overlay.

Same dual-panel layout as ``make_fig_cost_pareto.py`` (memory and
walltime vs SW, log-log) but reads from the warm-start sweep results
directory which has an extra <variant> level:

  <results_dir>/<variant>/<method>/N{N}_K{K}_D{D}/seed{s}.npz

For each (method, variant) pair we draw scatter points + a lower-left
Pareto staircase. Method = color; variant = linestyle/alpha. The
``cold`` variant uses the same style as ``make_fig_cost_pareto.py``
for visual continuity.

Usage:
    uv run python scripts/make_fig_warmstart_pareto.py \\
        --results_dir data/gmm_5d_hp_warmstart --d 5 \\
        --out results/gmm_5d_hp_warmstart/warmstart_pareto.pdf
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
    collect, memory_fraction,
)


# Variant → (linestyle, alpha, label-suffix). The cold variant matches
# make_fig_cost_pareto.py's existing visual conventions.
VARIANT_STYLE: dict[str, tuple[str, float, str]] = {
    "cold":   ("-",  1.00, ""),
    "P":      ("--", 0.85, " +P"),
    "P+A":    ("-.", 0.85, " +P+A"),
    "P+O":    (":",  0.85, " +P+O"),
    "P+A+O":  ("--", 0.65, " +all"),
}

# Which variants each method supports — must match
# experiments/make_warmstart_grid.py.
VARIANTS_BY_METHOD: dict[str, list[str]] = {
    "tci_als":   ["cold", "P", "P+A", "P+A+O"],
    "aci":       ["cold", "P", "P+O"],
    "dense":     ["cold"],
    "tci_tdvp1": ["cold"],
    "tci_tdvp2": ["cold"],
}


def pareto_min_min(xs: np.ndarray, ys: np.ndarray) -> tuple:
    """Lower-left Pareto frontier (both axes minimised)."""
    if len(xs) == 0:
        return xs, ys
    order = np.argsort(xs)
    x_sorted, y_sorted = xs[order], ys[order]
    keep, best = [], np.inf
    for i, y in enumerate(y_sorted):
        if y < best:
            keep.append(i)
            best = y
    return x_sorted[keep], y_sorted[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True,
                   help="Root of the warm-start sweep. Expected layout: "
                        "<results_dir>/<variant>/<method>/N{N}_K{K}_D{D}/seed{s}.npz")
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.results_dir)
    print(f"d={args.d}  root={root}", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 11.5), dpi=300,
                              constrained_layout=True)
    ax_mem, ax_time = axes

    legend_handles: list = []
    legend_labels: list[str] = []

    for method in WAVE_METHODS:
        for variant in VARIANTS_BY_METHOD.get(method, []):
            variant_root = root / variant
            if not variant_root.exists():
                continue
            cells = collect(variant_root, method)
            if not cells:
                continue
            mems = np.array([
                memory_fraction(c["N"], c["chi"], args.d) for c in cells
            ])
            times = np.array([c["time"] for c in cells if np.isfinite(c["time"])])
            time_idx = [i for i, c in enumerate(cells) if np.isfinite(c["time"])]
            sws_all = np.array([c["sw"] for c in cells])
            sws_time = np.array([cells[i]["sw"] for i in time_idx])
            mems_time = np.array([mems[i] for i in time_idx])

            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            ls, alpha, suffix = VARIANT_STYLE[variant]
            label = METHOD_LABEL[method] + suffix
            print(f"  {label:<22s}: {len(cells):3d} cells  "
                  f"SW {sws_all.min():.3f}–{sws_all.max():.3f}",
                  flush=True)

            # Scatter (light) + Pareto staircase (bold).
            ax_mem.scatter(mems, sws_all, color=color, marker=marker, s=18,
                           alpha=0.20 * alpha)
            xp, yp = pareto_min_min(mems, sws_all)
            line, = ax_mem.plot(xp, yp, color=color, linestyle=ls,
                                alpha=alpha, linewidth=1.6, marker=marker,
                                markersize=4)

            ax_time.scatter(times, sws_time, color=color, marker=marker, s=18,
                            alpha=0.20 * alpha)
            xpt, ypt = pareto_min_min(times, sws_time)
            ax_time.plot(xpt, ypt, color=color, linestyle=ls,
                         alpha=alpha, linewidth=1.6, marker=marker,
                         markersize=4)

            legend_handles.append(line)
            legend_labels.append(label)

    for ax, xlabel in [
        (ax_mem,  "MPS memory / Dense memory"),
        (ax_time, "Total walltime (s)"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sliced Wasserstein")
        ax.grid(True, which="both", alpha=0.3)
    ax_mem.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="MPS = Dense")

    fig.legend(legend_handles, legend_labels, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=9, frameon=False)
    fig.suptitle(
        f"Warm-start Pareto overlay (d={args.d}, gmm_{args.d}d)",
        y=1.08, fontsize=12,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
