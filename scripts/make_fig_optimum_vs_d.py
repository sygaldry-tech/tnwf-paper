"""Optimum-cell scaling across d: per-method best-SW configuration vs d
with simple log-linear fits and extrapolation to d=6.

For each (d ∈ {2, 3, 4, 5}, method ∈ {dense, tci_tdvp1, tci_tdvp2}) we pick the cell with the lowest SW (mean over seeds). We
then plot, against d:
  - SW achieved at that cell
  - MPS / Dense memory ratio (Dense = 1)
  - Walltime (s)
  - N, K, D_max of the best cell

Per-method log-linear fits (in d) extrapolate to d=6, with a clear
"extrapolated" marker so it's not confused with measured data.

Usage:
    uv run python scripts/make_fig_optimum_vs_d.py \
        --out results/optimum_vs_d.pdf
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
    collect, collect_reference, memory_fraction,
)

# Per-method linestyle so curves remain distinguishable when several methods
# pick the same HP at the same d (common at d=2 where everything saturates
# the search-grid max).
METHOD_LINESTYLE = {
    "dense":      "-",
    "tci_tdvp1":  (0, (1, 1)),          # dotted
    "tci_tdvp2":  (0, (4, 1, 1, 1, 1, 1)),  # long-dash-dot-dot
}

# Maximum HP value sampled by the Sobol grid at each d. Bottom-row panels
# show this as a dashed staircase so the reader can tell whether the optimum
# was bounded by the search box.
SEARCH_MAX = {
    2: {"N": 256, "K": 64, "D": 128},
    3: {"N": 64,  "K": 64, "D": 64},   # K extended to 64 by the d=3 K-fillin
    4: {"N": 32,  "K": 64, "D": 64},
    5: {"N": 32,  "K": 64, "D": 32},
    6: {"N": 32,  "K": 64, "D": 32},   # N=64 cells OOM-ed; effective max
    7: {"N": 32,  "K": 64, "D": 32},   # probe row only at this point
    8: {"N": 32,  "K": 64, "D": 32},   # probe row only at this point
}

DATASETS = [
    (2, "data/gmm_2d_hp"),
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
    (6, "data/gmm_6d_hp"),
    (7, "data/gmm_7d_hp"),
    (8, "data/gmm_8d_hp"),
]
EXTRAPOLATE_TO = 9


def best_cell(cells: list[dict]) -> dict | None:
    """The lowest-SW cell."""
    return min(cells, key=lambda c: c["sw"]) if cells else None


def fit_loglinear(ds, ys):
    """log-y = a + b·d. Returns (slope, intercept) on log10 scale."""
    ds = np.asarray(ds, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ds) & np.isfinite(ys) & (ys > 0)
    if mask.sum() < 2:
        return None
    return np.polyfit(ds[mask], np.log10(ys[mask]), 1)


def fit_linear(ds, ys):
    """y = a + b·d. Returns (slope, intercept)."""
    ds = np.asarray(ds, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ds) & np.isfinite(ys)
    if mask.sum() < 2:
        return None
    return np.polyfit(ds[mask], ys[mask], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/optimum_vs_d.pdf")
    args = p.parse_args()

    # ── Build the per-(d, method) best-cell table ─────────────────────────
    table = {}
    for d, results_dir in DATASETS:
        for m in WAVE_METHODS:
            cells = collect(Path(results_dir), m)
            best = best_cell(cells)
            if best is None:
                continue
            mem = (1.0 if m == "dense"
                   else memory_fraction(best["N"], best["chi"], d))
            table[(d, m)] = {
                **best,
                "mem": mem,
                "d": d,
            }

    # Add a "time_ratio" field (method walltime / Dense walltime at the same d).
    # If Dense isn't measured at d (e.g. d=6, where Dense is infeasible at the
    # required N for SW parity), extrapolate Dense walltime from a log-linear
    # fit on the d's that DO have Dense — otherwise the ratio panel silently
    # drops the d=6 row for every method.
    dense_pts = [(d, table[(d, "dense")]["time"])
                 for d, _ in DATASETS
                 if (d, "dense") in table
                 and table[(d, "dense")]["time"] == table[(d, "dense")]["time"]]
    if len(dense_pts) >= 2:
        ds_d, ts_d = zip(*dense_pts)
        slope_d, intercept_d = np.polyfit(ds_d, np.log10(ts_d), 1)
    else:
        slope_d, intercept_d = (float("nan"), float("nan"))

    def t_dense_at(d):
        c = table.get((d, "dense"))
        if c is not None and c["time"] == c["time"]:
            return c["time"]
        if slope_d == slope_d:
            return 10 ** (slope_d * d + intercept_d)
        return float("nan")

    for d, _ in DATASETS:
        t_d = t_dense_at(d)
        for m in WAVE_METHODS:
            cell = table.get((d, m))
            if cell is None:
                continue
            cell["time_ratio"] = (cell["time"] / t_d
                                  if (t_d and t_d == t_d
                                      and cell["time"] == cell["time"])
                                  else float("nan"))

    # JAM cells live on a separate D-invariant path — pull best-SW per d.
    # Only used in the SW panel; JAM has no MPS bond / walltime for ratios.
    for d, results_dir in DATASETS:
        jam_cells = collect_reference(Path(results_dir), "jam")
        if jam_cells:
            best = min(jam_cells, key=lambda c: c["sw"])
            table[(d, "jam")] = {
                **best, "d": d, "D": 0, "chi": 0,
                "mem": float("nan"), "time": float("nan"),
                "time_ratio": float("nan"),
            }

    # Print summary
    print(f"{'d':>2} {'method':<11} {'N':>4} {'K':>4} {'D_max':>5} {'χ':>4} "
          f"{'SW':>7} {'mem':>9} {'wall':>8}")
    for (d, m), c in sorted(table.items()):
        print(f"{d:>2} {m:<11} {c['N']:>4} {c['K']:>4} {int(c['D']):>5} "
              f"{int(c['chi']):>4} {c['sw']:>7.4f} {c['mem']:>9.4g} "
              f"{c['time']:>8.0f}")

    # ── Lay out the figure ────────────────────────────────────────────────
    panels = [
        ("sw",         "SW (lower = better)",        "log",   fit_loglinear),
        ("mem",        "MPS / Dense memory",         "log",   fit_loglinear),
        ("time_ratio", "Walltime / Dense walltime",  "log",   fit_loglinear),
        # HP panels: log-2 y axis with extrapolation to d=6.
        ("N",          "N (grid resolution)",        "log2",  fit_loglinear),
        ("K",          "K (Trotter steps)",          "log2",  fit_loglinear),
        ("D",          "D_max (bond cap)",           "log2",  fit_loglinear),
    ]
    # In the SW panel only, show fits for these methods only (Dense as the
    # classical reference and the two TDVP methods we focus on).
    SW_FIT_METHODS = {"dense", "tci_tdvp1", "tci_tdvp2"}
    n_panels = len(panels)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.22)
    axes = axes.flatten()

    ds_all = sorted({d for d, _ in DATASETS})
    d_extrap = np.linspace(min(ds_all), EXTRAPOLATE_TO, 50)

    HP_KEYS = {"N", "K", "D"}
    TDVP_METHODS = {"tci_tdvp1", "tci_tdvp2"}
    MPS_METHODS = TDVP_METHODS
    # N panel: all MPS methods + Dense + JAM (each has a meaningful grid res).
    # K panel: all MPS methods + Dense (JAM's K is a sampling discretization,
    # not directly comparable to wave-method Trotter K).
    # D panel: MPS methods only (Dense and JAM don't truncate an MPS bond).
    N_METHODS = MPS_METHODS | {"dense", "jam"}
    K_METHODS = MPS_METHODS | {"dense"}
    D_METHODS = MPS_METHODS
    METHODS_WITH_JAM = ["jam", *WAVE_METHODS]

    # Stacking: Exact line on top, then wave methods, then JAM at the bottom
    # (so the wave-method markers read clearly even when SW values cluster).
    # JAM gets a slightly larger marker (set elsewhere) so it still peeks out.
    LEGEND_ORDER = ["__exact__", *WAVE_METHODS]
    Z_BASE = 3
    Z_OF = {name: Z_BASE + (len(LEGEND_ORDER) - i)
            for i, name in enumerate(LEGEND_ORDER)}
    Z_OF["jam"] = Z_BASE - 1                       # below all wave methods

    # SW noise floor: SW(target_a, target_b) at our (n_samples, n_projections).
    # See scripts/compute_exact_sw_floor.py.
    exact_floor_path = Path("data/exact_sw_floor.json")
    exact_floor = {}
    if exact_floor_path.exists():
        import json as _json
        exact_floor = {int(k): v["mean"]
                       for k, v in _json.loads(exact_floor_path.read_text()).items()}

    # Per-panel method roster for the HP-row grouped bars. Order mirrors the
    # legend (JAM, Dense, TDVP1, TDVP2) so reading the bars
    # left-to-right within a group matches reading the legend top-to-bottom.
    HP_PANEL_METHODS = {
        "N": ["jam", "dense", "tci_tdvp1", "tci_tdvp2"],
        "K": ["dense", "tci_tdvp1", "tci_tdvp2"],
        "D": ["tci_tdvp1", "tci_tdvp2"],
    }

    for ax, (key, label, yscale, fit_fn) in zip(axes, panels):
        # SW panel: overlay the irreducible SW noise floor as a horizontal
        # reference (mean across d, since the floor is roughly d-invariant).
        if key == "sw" and exact_floor:
            xs_floor = sorted(exact_floor)
            ys_floor = [exact_floor[d] for d in xs_floor]
            ax.plot(xs_floor, ys_floor, ls="--", lw=1.5,
                    color="dimgray", alpha=0.8, zorder=Z_OF["__exact__"],
                    label="Exact (sample-size floor)")

        # ── Grouped-bar layout for the HP row ────────────────────────────
        if key in HP_KEYS:
            mlist = [m for m in HP_PANEL_METHODS[key]
                     if any((d, m) in table for d in ds_all)]
            n_methods = len(mlist)
            # Narrower group_width leaves more whitespace BETWEEN d-clusters,
            # so adjacent d's read as distinct.
            group_width = 0.62
            bar_w = group_width / max(n_methods, 1)
            for i, m in enumerate(mlist):
                offset = (i - (n_methods - 1) / 2) * bar_w
                xs, ys = [], []
                for d in ds_all:
                    if (d, m) not in table:
                        continue
                    v = table[(d, m)].get(key, float("nan"))
                    if isinstance(v, float) and v != v:
                        continue
                    xs.append(d + offset)
                    ys.append(v)
                if not xs:
                    continue
                ax.bar(xs, ys, width=bar_w * 0.95,
                       color=METHOD_COLORS[m],
                       edgecolor="black", linewidth=0.4,
                       alpha=0.85, label=METHOD_LABEL[m],
                       zorder=Z_OF.get(m, Z_BASE))
            # Search-grid budget drawn as a dashed *rectangle outline* around
            # each d-group: top edge at SEARCH_MAX[d][key], bottom edge at the
            # x-axis (y = 1 on the log2 axis, which is the visual floor). The
            # reader sees at a glance which bars reach the cap and which sit
            # interior — and the box visually delineates each d-group.
            from matplotlib.patches import Rectangle
            ybot = 1.0                                      # log2(1) = 0 → x-axis
            for j, d in enumerate(ds_all):
                ymax = SEARCH_MAX[d][key]
                rect = Rectangle((d - group_width / 2, ybot),
                                 group_width, ymax - ybot,
                                 fill=False, edgecolor="dimgray",
                                 linestyle="--", linewidth=1.3,
                                 alpha=0.75, zorder=1,
                                 label="Search-grid box" if j == 0 else None)
                ax.add_patch(rect)

        # SW and the ratio panels: keep the scatter-marker layout.
        else:
            method_list = ([*WAVE_METHODS, "jam"] if key == "sw"
                           else WAVE_METHODS)
        for m in [] if key in HP_KEYS else method_list:
            xs = []
            ys = []
            for d in ds_all:
                if (d, m) not in table:
                    continue
                v = table[(d, m)].get(key, float("nan"))
                if isinstance(v, float) and v != v:    # NaN
                    continue
                xs.append(d)
                ys.append(v)
            if not xs:
                continue
            xs_arr = np.asarray(xs)
            ys_arr = np.asarray(ys)
            is_jam = (m == "jam")
            ax.plot(xs_arr, ys_arr, ls="none",
                    marker=METHOD_MARKERS[m], ms=16 if is_jam else 12,
                    markerfacecolor=METHOD_COLORS[m],
                    markeredgecolor="black",
                    markeredgewidth=1.2 if is_jam else 0.5,
                    alpha=0.9, label=METHOD_LABEL[m],
                    zorder=Z_OF.get(m, Z_BASE))
            # Fit overlay + extrapolation:
            #   - SW panel: NO fit lines (the markers carry the message).
            #   - mem panel: drop d=2 from the fit. The d=2 search-box maxes
            #     out N and D (256 / 128) so its memory ratio sits well off
            #     the trend at d≥3 and pulls the slope.
            #   - time_ratio panel: dashed fit line for ALL methods.
            if key in ("mem", "time_ratio") and fit_fn is not None:
                if key == "mem":
                    fit_xs = [x for x in xs if x >= 3]
                    fit_ys = [y for x, y in zip(xs, ys) if x >= 3]
                else:
                    fit_xs, fit_ys = xs, ys
                fit = fit_fn(fit_xs, fit_ys)
                if fit is not None:
                    slope, intercept = fit
                    yhat = 10 ** (slope * d_extrap + intercept)
                    ax.plot(d_extrap, yhat, "--", lw=1.8,
                            color=METHOD_COLORS[m], alpha=0.65, zorder=2)
                    y6 = 10 ** (slope * EXTRAPOLATE_TO + intercept)
                    ax.plot(EXTRAPOLATE_TO, y6,
                            marker=METHOD_MARKERS[m], ms=12, lw=0,
                            markerfacecolor="none",
                            markeredgecolor=METHOD_COLORS[m],
                            markeredgewidth=1.5, alpha=0.85, zorder=4)

        # Only the panels that actually fit + extrapolate (mem, time_ratio)
        # get the d=7 tick and the gray "extrapolated" axvspan. SW, N, K, D
        # show only measured d.
        is_extrap_panel = key in ("mem", "time_ratio")
        if is_extrap_panel:
            ax.set_xticks(list(ds_all) + [EXTRAPOLATE_TO])
            ax.set_xticklabels([str(int(d)) for d in ds_all] + [f"{EXTRAPOLATE_TO}*"])
            ax.set_xlabel("d  (spatial dimension; * = extrapolated)", fontsize=14)
        else:
            ax.set_xticks(list(ds_all))
            ax.set_xticklabels([str(int(d)) for d in ds_all])
            ax.set_xlabel("d  (spatial dimension)", fontsize=14)
        ax.set_ylabel(label, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        # log2 panels (N, K, D) get power-of-2 ticks and labels.
        if yscale == "log2":
            ax.set_yscale("log", base=2)
            from matplotlib.ticker import LogLocator, FuncFormatter
            ax.yaxis.set_major_locator(LogLocator(base=2.0))
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _: f"{int(v)}" if v >= 1 else f"{v:g}"))
            ax.yaxis.set_minor_locator(LogLocator(base=2.0,
                                                   subs=[1.5], numticks=20))
        else:
            ax.set_yscale(yscale)
        ax.grid(True, which="both", alpha=0.25)
        # HP panels: pin the y-axis lower bound to 1 (= log2(0)) so the
        # search-grid box outline extends visually all the way down to the
        # x-axis. Without this, matplotlib auto-scales the y-min to the
        # smallest bar (typically 4 or 8), clipping the box bottom.
        if key in HP_KEYS:
            ax.set_ylim(bottom=1)
        if is_extrap_panel:
            ax.axvspan(max(ds_all), EXTRAPOLATE_TO, color="gray", alpha=0.08)
        # Reference y=1 line on the ratio panels (mem and time_ratio).
        if key in ("mem", "time_ratio"):
            ax.axhline(1.0, color="dimgray", lw=1, ls="--", alpha=0.7,
                       zorder=0)

    # Legend in constrained-layout-reserved space above the panels.
    # Pull from SW panel (axes[0]) since it includes JAM in addition to the
    # 5 wave methods. Move JAM to the 2nd slot (right after the Exact floor
    # reference) so the two non-wave references sit together at the front.
    handles, labels = axes[0].get_legend_handles_labels()
    if "JAM" in labels:
        i = labels.index("JAM")
        handles.insert(1, handles.pop(i))
        labels.insert(1, labels.pop(i))
    fig.legend(handles, labels, loc="outside upper center",
               ncol=len(labels), fontsize=13, frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
