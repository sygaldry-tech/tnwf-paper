"""Per-d scatter of Dense + TDVP1 + TDVP2 results, all cells (not just optima).

Three vertically stacked panels: SW, MPS-memory ratio, and walltime against d.
Each (N, K, D, seed-mean) cell shows up as one dot — so each d gets a vertical
column of dots per method. Useful complement to optimum_vs_d.pdf, which shows
only the best-SW cell at each d.

Usage:
    uv run python scripts/make_fig_dense_tdvp_scatter.py --out results/dense_tdvp_scatter.pdf
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
    METHOD_COLORS, METHOD_LABEL, METHOD_MARKERS,
    collect, collect_reference, mps_param_count,
)

BYTES_PER_COMPLEX128 = 16

DATASETS = [
    (2, "data/gmm_2d_hp"),
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
    (6, "data/gmm_6d_hp"),
    (7, "data/gmm_7d_hp"),
    (8, "data/gmm_8d_hp"),
]
METHODS = ["dense", "tci_tdvp1", "tci_tdvp2"]
# Each method's points sit exactly at integer d — no horizontal jitter.
METHOD_OFFSET = {m: 0.0 for m in METHODS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/dense_tdvp_scatter.pdf")
    args = p.parse_args()

    # Single best-SW cell per (d, method).
    by_dm = {}
    for d, results_dir in DATASETS:
        for m in METHODS:
            cells = collect(Path(results_dir), m)
            if not cells:
                continue
            best = min(cells, key=lambda c: c["sw"])
            # Absolute state memory in MB (complex128 = 16 B per amplitude).
            if m == "dense":
                params = best["N"] ** d
            else:
                params = mps_param_count(best["N"], best["chi"], d)
            mem_mb = BYTES_PER_COMPLEX128 * params / (1 << 20)
            by_dm.setdefault((m, "x"),     []).append(d + METHOD_OFFSET[m])
            by_dm.setdefault((m, "sw"),    []).append(best["sw"])
            by_dm.setdefault((m, "mem"),   []).append(mem_mb)
            by_dm.setdefault((m, "time"),  []).append(best["time"])
            by_dm.setdefault((m, "label"), []).append(
                f"N={best['N']} K={best['K']} D={int(best['D'])}"
            )
        # JAM lives on a different per-(N, K) path (no D bond); pull best-SW per d.
        # JAM has no MPS memory or comparable walltime, so it goes on the SW
        # panel only.
        jam_cells = collect_reference(Path(results_dir), "jam")
        if jam_cells:
            best_jam = min(jam_cells, key=lambda c: c["sw"])
            by_dm.setdefault(("jam", "x"),  []).append(d)
            by_dm.setdefault(("jam", "sw"), []).append(best_jam["sw"])

    # Exact (sample-size SW noise floor) — d-by-d from data/exact_sw_floor.json.
    import json as _json
    exact_path = Path("data/exact_sw_floor.json")
    exact_floor = {}
    if exact_path.exists():
        exact_floor = {int(k): v["mean"]
                       for k, v in _json.loads(exact_path.read_text()).items()}

    # 1×2 layout. Force each panel's BOX aspect to 0.7 (height/width), i.e.
    # ~1.4:1 W:H per panel — gives the d=2..7 axis enough horizontal space
    # to breathe while keeping the log-y range readable. figsize is generous
    # (16×6); panel aspect is set explicitly below so it doesn't depend on
    # how constrained_layout absorbs label/legend space.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.20, h_pad=0.10)
    ax_mem, ax_wall = axes
    for ax in axes:
        ax.set_box_aspect(0.7)
    # tuple: (axis, key, ylabel, log_y, do_fit)
    panel_specs = [
        (ax_mem,  "mem",  "State memory (MB, log)", True,  True),
        (ax_wall, "time", "Walltime per run (s)",  True,  True),
    ]

    # Best-cell summary
    print(f"{'method':<11} {'d':>2}  {'best (N, K, D)':<18} {'SW':>7} {'mem':>10} {'wall':>10}")
    for m in METHODS:
        for i, x in enumerate(by_dm.get((m, "x"), [])):
            d = round(x - METHOD_OFFSET[m])
            label = by_dm[(m, "label")][i]
            sw   = by_dm[(m, "sw")][i]
            mem  = by_dm[(m, "mem")][i]
            wall = by_dm[(m, "time")][i]
            wall_s = f"{wall:>10.0f}" if np.isfinite(wall) else f"{'—':>10}"
            print(f"  {m:<11} {d:>2}  {label:<18} {sw:>7.4f} {mem:>10.4g} {wall_s}")

    print()  # blank line before fits
    for ax, key, ylabel, log_y, do_fit in panel_specs:
        annot_lines = []
        # SW panel: overlay Exact noise-floor + JAM (both below the wave-method
        # markers so Dense/TDVP1/TDVP2 stay legible). Plot Exact first, then
        # JAM, so the legend order is: Exact, JAM, Dense, TDVP1, TDVP2.
        if key == "sw":
            if exact_floor:
                xs_floor = sorted(exact_floor)
                ys_floor = [exact_floor[d] for d in xs_floor]
                ax.plot(xs_floor, ys_floor, ls="--", lw=1.6,
                        color="dimgray", alpha=0.85, zorder=1,
                        label="Exact (sample-size floor)")
            jam_x = by_dm.get(("jam", "x"), [])
            jam_y = by_dm.get(("jam", "sw"), [])
            if jam_x:
                ax.scatter(jam_x, jam_y, s=130,
                           color=METHOD_COLORS["jam"],
                           marker=METHOD_MARKERS["jam"],
                           alpha=0.9, edgecolors="black", linewidths=0.8,
                           label=METHOD_LABEL["jam"], zorder=2)
        for m in METHODS:
            xs = np.asarray(by_dm.get((m, "x"), []))
            ys = np.asarray(by_dm.get((m, key), []))
            mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
            if mask.sum() == 0:
                continue
            ax.scatter(xs[mask], ys[mask], s=90,
                       color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                       alpha=0.95, edgecolors="black", linewidths=0.6,
                       label=METHOD_LABEL[m] if ax is panel_specs[0][0] else None,
                       zorder=3)
            if not do_fit:
                continue
            # Skip fits for the TDVP methods — only Dense gets a fit line +
            # annotation. The TDVP markers carry the message themselves and
            # the fit overlays were visual clutter in the small-d range.
            if m != "dense":
                continue
            xs_int = np.round(xs[mask] - METHOD_OFFSET[m])
            if len(xs_int) < 2:
                continue
            # log10 fit ⇒ y = A · B^d  where A=10^intercept, B=10^slope.
            slope, intercept = np.polyfit(xs_int, np.log10(ys[mask]), 1)
            # Dense was infeasible at d=6,7 (memory) so the fit is built from
            # d=3..5 only. Extrapolate the projected line out to d=7 (matching
            # the TDVP markers' x-range) so readers can compare the
            # would-have-been Dense cost against TDVP at high d.
            d_max_dense = float(xs_int.max())
            d_max_extrap = max(d_max_dense, 7.0)
            # Solid segment over the fitted range, dashed continuation past it.
            xx_solid = np.linspace(xs_int.min() - 0.2, d_max_dense, 30)
            xx_dash  = np.linspace(d_max_dense, d_max_extrap + 0.2, 20)
            ax.plot(xx_solid + METHOD_OFFSET[m],
                    10 ** (slope * xx_solid + intercept),
                    "-", lw=1.8, color=METHOD_COLORS[m], alpha=0.85, zorder=2)
            ax.plot(xx_dash + METHOD_OFFSET[m],
                    10 ** (slope * xx_dash + intercept),
                    "--", lw=1.8, color=METHOD_COLORS[m], alpha=0.6, zorder=2)
            # Open hollow markers at every integer d past the fitted range
            # (d=6 and d=7 here) — Dense was infeasible there, so these are
            # extrapolated points; the open style flags them as projections.
            for d_x in range(int(d_max_dense) + 1, int(d_max_extrap) + 1):
                d_y = 10 ** (slope * d_x + intercept)
                ax.plot(d_x + METHOD_OFFSET[m], d_y,
                        marker=METHOD_MARKERS[m], ms=12, lw=0,
                        markerfacecolor="none", markeredgecolor=METHOD_COLORS[m],
                        markeredgewidth=1.5, alpha=0.85, zorder=4)
            A = 10 ** intercept
            B = 10 ** slope
            ynam = "memory" if key == "mem" else "walltime"
            unit = "MB" if key == "mem" else "s"
            annot_lines.append(
                f"{METHOD_LABEL[m]}:  {ynam}(d) = {A:.3g} · {B:.2f}^d {unit}"
            )
            print(f"  {METHOD_LABEL[m]:<11} {ynam:<9} = "
                  f"{A:.3g} · {B:.3f}^d {unit}  "
                  f"(slope log₁₀ = {slope:+.3f}, intercept = {intercept:+.3f})")
        if annot_lines:
            ax.text(0.02, 0.98, "\n".join(annot_lines),
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="lightgray", alpha=0.85))
        ax.set_ylabel(ylabel, fontsize=11)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)

    # Side-by-side: each panel needs its own x-label and ticks.
    for ax in axes:
        ax.set_xlabel("d  (spatial dimension)", fontsize=11)
        ax.set_xticks([d for d, _ in DATASETS])

    # Position the legend manually just above the figure box, NOT via
    # "outside upper center" — that reserves a huge vertical band when
    # combined with a single row of panels (works fine with multi-row
    # figures like optimum_vs_d.pdf, but stretches the layout here).
    # bbox_inches="tight" expands the saved figure to include this anchor.
    handles, labels = ax_mem.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=11, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
