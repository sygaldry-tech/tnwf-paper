"""Fig A: HP sensitivity ribbons — SW vs N, K, D_max per d.

Three horizontal panels per dataset. For each panel and each method:
  - Every (N, K, D) cell shown as a faint scatter dot.
  - A LOWESS-smoothed curve through the marginal traces the trend.
Reference horizontal lines at the best classical SW (Dense and JAM).

Usage:
    uv run python scripts/make_fig_hp_sensitivity.py \
        --results_dir data/gmm_5d_hp --d 5 \
        --out results/gmm_5d_hp/hp_sensitivity.pdf
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
    best_classical_sw, collect,
)


def lowess_log(x: np.ndarray, y: np.ndarray, frac: float = 0.6) -> tuple:
    """Tiny local-regression smoother in log-x / log-y.

    Returns (xs, ys) on a regular log-x grid spanning [min(x), max(x)].
    Falls back to a straight line through (mean log x, mean log y) if too
    few points.
    """
    if len(x) < 2:
        return np.array(x), np.array(y)
    lx = np.log2(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    xs = np.linspace(lx.min(), lx.max(), 50)
    ys = np.empty_like(xs)
    h = max(frac * (lx.max() - lx.min()), 0.5)
    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((lx - xi) / h) ** 2)
        sw = w.sum()
        if sw < 1e-9:
            ys[i] = ly.mean()
            continue
        # Local linear regression
        wm = w / sw
        mx = (wm * lx).sum()
        my = (wm * ly).sum()
        vx = (wm * (lx - mx) ** 2).sum()
        cov = (wm * (lx - mx) * (ly - my)).sum()
        slope = cov / vx if vx > 1e-12 else 0.0
        ys[i] = my + slope * (xi - mx)
    return 2 ** xs, np.exp(ys)


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
    n_total = sum(len(v) for v in by_method.values())
    print(f"loaded {n_total} cells across {len(by_method)} methods", flush=True)
    for m, cells in by_method.items():
        if cells:
            sws = [c["sw"] for c in cells]
            print(f"  {m:10s}: {len(cells)} cells, SW {min(sws):.3f}–{max(sws):.3f}",
                  flush=True)
    print(f"  best classical: jam={refs['jam']:.4f}  "
          f"dense={refs['dense']:.4f}  min={refs['best']:.4f}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300, sharex="col",
                              constrained_layout=True)
    HP_AXES = [("N", "Spatial grid resolution N"),
               ("K", "Trotter steps K"),
               ("D", "MPS bond cap D_max")]
    METRIC_ROWS = [
        ("sw",   "SW (final, lower = better)", True),
        ("time", "Walltime per run (s)",       True),
    ]

    for row, (metric_key, ylabel, log_y) in enumerate(METRIC_ROWS):
        for col, (key, hp_label) in enumerate(HP_AXES):
            ax = axes[row, col]
            for method in WAVE_METHODS:
                cells = by_method.get(method, [])
                if not cells:
                    continue
                xs = np.array([c[key] for c in cells], dtype=float)
                ys = np.array([c[metric_key] for c in cells], dtype=float)
                mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
                if mask.sum() == 0:
                    continue
                ax.scatter(xs[mask], ys[mask], s=18, alpha=0.35,
                           color=METHOD_COLORS[method],
                           marker=METHOD_MARKERS[method],
                           edgecolors="none", zorder=2)
                xx, yy = lowess_log(xs[mask], ys[mask], frac=0.55)
                ax.plot(xx, yy, "-", lw=2.2,
                        color=METHOD_COLORS[method], alpha=0.95,
                        label=METHOD_LABEL[method] if (row == 0 and col == 0) else None,
                        zorder=3)

            # SW row: reference lines for best classical
            if metric_key == "sw":
                for name, val, ls in (("Dense best", refs["dense"], "--"),
                                      ("JAM best",  refs["jam"], ":")):
                    if val == val:  # not nan
                        ax.axhline(val, color="dimgray", lw=1, ls=ls, alpha=0.7,
                                   zorder=1)
                        ax.text(0.02, val * 1.04, name, fontsize=8,
                                color="dimgray",
                                transform=ax.get_yaxis_transform(), ha="left")

            ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if row == 1:
                ax.set_xlabel(hp_label, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)

    # Legend on a strip above the panels so it doesn't occlude data.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(labels),
               fontsize=10, frameon=False)
    fig.suptitle(
        f"HP sensitivity at d={args.d}  —  GMM target  "
        f"(top row: accuracy;  bottom row: walltime;  scatter = each (N, K, D) cell; "
        f"line = LOWESS through marginal)",
        fontsize=12, fontweight="bold", y=1.04,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # constrained_layout already handled spacing; don't call tight_layout
    # afterwards (it fights with constrained_layout and can over-squeeze).
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
