"""HP heatmaps per (d, method): χ_state / SW / MMD over (N, K), one column per D_max.

Lets you eye-ball the Pareto frontier of (N, K, D_max) for each method at each d.

For each {dataset, method}, we:
  1. Walk data/{dataset}/{method}/N{N}_K{K}_D{D}/seed*.npz, mean over seeds.
  2. For each unique D_max, build a (K × N) heatmap of χ_state, SW, MMD.
  3. Lay out as 3 rows (one per metric) × len(D_max) columns.
  4. Shared colorscale across the row so cells are comparable across D_max.

Usage:
    uv run python scripts/make_fig_hp_heatmaps.py \
        --results_dir data/gmm_3d_hp_v2 --d 3 --method tci_tdvp1 \
        --out results/gmm_3d_hp_v2/heatmap_tdvp1.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np

NK_D = re.compile(r"^N(\d+)_K(\d+)_D(\d+)$")
NK   = re.compile(r"^N(\d+)_K(\d+)$")


def collect(results_dir: Path, method: str) -> list[dict]:
    """Load all (N, K, D) cells for an MPS method (mean over seeds)."""
    base = results_dir / method
    if not base.exists():
        return []
    rows = []
    for sub in sorted(base.iterdir()):
        m = NK_D.match(sub.name)
        if not m:
            continue
        N, K, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sw, mmd, chi = [], [], []
        for f in sorted(sub.glob("seed*.npz")):
            try:
                z = np.load(f, allow_pickle=True)
                sw.append(float(z["sw"][-1]))
                mmd.append(float(z["mmd"][-1]))
                chi.append(int(z["chi_max"][-1]))
            except Exception:
                continue
        if sw:
            rows.append({
                "N": N, "K": K, "D": D,
                "sw": float(np.mean(sw)),
                "mmd": float(np.mean(mmd)),
                "chi": float(np.mean(chi)),
                "n_seeds": len(sw),
            })
    return rows


def collect_reference(results_dir: Path, method: str) -> list[dict]:
    """Load (N, K) cells for a D-invariant reference method.

    JAM lives at `N{N}_K{K}/` (no D suffix). Dense lives at `N{N}_K{K}_D{D}/`
    but its result doesn't depend on D — so we average across all D values per
    (N, K) to reduce noise.
    """
    base = results_dir / method
    if not base.exists():
        return []
    bins: dict[tuple[int, int], dict] = {}
    for sub in sorted(base.iterdir()):
        m_nkd = NK_D.match(sub.name)
        m_nk = NK.match(sub.name)
        if m_nkd:
            N, K = int(m_nkd.group(1)), int(m_nkd.group(2))
        elif m_nk:
            N, K = int(m_nk.group(1)), int(m_nk.group(2))
        else:
            continue
        for f in sorted(sub.glob("seed*.npz")):
            try:
                z = np.load(f, allow_pickle=True)
                d = bins.setdefault((N, K), {"sw": [], "mmd": []})
                d["sw"].append(float(z["sw"][-1]))
                d["mmd"].append(float(z["mmd"][-1]))
            except Exception:
                continue
    return [
        {"N": N, "K": K, "sw": float(np.mean(d["sw"])), "mmd": float(np.mean(d["mmd"]))}
        for (N, K), d in bins.items()
    ]


def make_grid(rows, Ns, Ks, key):
    """Return a (len(Ks), len(Ns)) array; rows=K (y), cols=N (x). NaN where missing."""
    M = np.full((len(Ks), len(Ns)), np.nan)
    for r in rows:
        if r["N"] in Ns and r["K"] in Ks:
            M[Ks.index(r["K"]), Ns.index(r["N"])] = r[key]
    return M


def annotate_cells(ax, M, fmt, vnorm):
    for ki in range(M.shape[0]):
        for ni in range(M.shape[1]):
            v = M[ki, ni]
            if np.isnan(v):
                continue
            text_c = "white" if vnorm(v) > 0.55 else "black"
            ax.text(ni, ki, fmt.format(v), ha="center", va="center",
                    fontsize=9, color=text_c, fontweight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--method", required=True, choices=["tci_tdvp1", "tci_tdvp2"])
    p.add_argument("--out", required=True)
    p.add_argument("--label", default=None,
                   help="Display label, defaults to '{method} d={d}'")
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    rows = collect(res_dir, args.method)
    if not rows:
        raise SystemExit(f"No data at {res_dir}/{args.method}")

    # Reference methods: JAM (no MPS, gradient flow on trained net) and Dense
    # (analytical V_t, full N^d state). Both are D-independent.
    jam_rows   = collect_reference(res_dir, "jam")
    dense_rows = collect_reference(res_dir, "dense")

    Ns = sorted({r["N"] for r in rows} | {r["N"] for r in jam_rows} | {r["N"] for r in dense_rows})
    Ks = sorted({r["K"] for r in rows} | {r["K"] for r in jam_rows} | {r["K"] for r in dense_rows})
    Ds = sorted({r["D"] for r in rows})
    print(f"  found {len(rows)} TDVP cells, {len(jam_rows)} jam, {len(dense_rows)} dense")
    print(f"  axes: N={Ns}, K={Ks}, D_max={Ds}")

    # Build grids per metric per D_max
    metrics = ["chi", "sw", "mmd"]
    metric_titles = {"chi": "Effective state bond  χ", "sw": "SW (lower=better)", "mmd": "MMD (lower=better)"}
    metric_log = {"chi": False, "sw": True, "mmd": True}
    metric_fmt = {"chi": "{:.0f}", "sw": "{:.3f}", "mmd": "{:.3f}"}

    grids = {}        # grids[(metric, D)] -> 2D array
    row_vals = {m: [] for m in metrics}
    for D in Ds:
        rows_D = [r for r in rows if r["D"] == D]
        for met in metrics:
            M = make_grid(rows_D, Ns, Ks, met)
            grids[(met, D)] = M
            row_vals[met].extend(M[~np.isnan(M)].tolist())

    # Reference grids (only sw/mmd; no χ for non-MPS methods)
    ref_grids = {}    # ref_grids[(metric, ref_method)] -> 2D
    for ref_method, ref_rows in (("jam", jam_rows), ("dense", dense_rows)):
        for met in ("sw", "mmd"):
            if not ref_rows:
                continue
            M = make_grid(ref_rows, Ns, Ks, met)
            ref_grids[(met, ref_method)] = M
            row_vals[met].extend(M[~np.isnan(M)].tolist())

    label = args.label or f"{args.method.upper()} (d={args.d})"

    REFS = ["jam", "dense"]
    REF_LABEL = {"jam": "JAM\n(reference)", "dense": "Dense\n(analytical V_t)"}
    n_cols_tdvp = len(Ds)
    n_cols = n_cols_tdvp + len(REFS)
    fig = plt.figure(figsize=(3.4 * n_cols + 1.2, 11), dpi=200)
    gs = GridSpec(
        4, n_cols + 1, figure=fig,
        width_ratios=[1.0] * n_cols + [0.05],
        height_ratios=[0.08, 1.0, 1.0, 1.0],
        hspace=0.35, wspace=0.18,
    )

    # ── Top strip: column titles for each D_max + reference panels ──────────
    for ci, D in enumerate(Ds):
        ax_strip = fig.add_subplot(gs[0, ci])
        ax_strip.axis("off")
        ax_strip.text(0.5, 0.5, f"D_max = {D}",
                      ha="center", va="center", fontsize=13, fontweight="bold",
                      transform=ax_strip.transAxes)
    for off, ref in enumerate(REFS):
        ax_strip = fig.add_subplot(gs[0, n_cols_tdvp + off])
        ax_strip.axis("off")
        ax_strip.text(0.5, 0.5, REF_LABEL[ref],
                      ha="center", va="center", fontsize=13, fontweight="bold",
                      color="dimgray", transform=ax_strip.transAxes)

    # ── For each metric row, plot heatmaps with shared colorbar ─────────────
    for ri, met in enumerate(metrics, start=1):
        if not row_vals[met]:
            continue
        if metric_log[met]:
            v = np.array(row_vals[met])
            v = v[v > 0]
            vmin, vmax = float(v.min()), float(v.max())
            norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
        else:
            vmin, vmax = float(np.min(row_vals[met])), float(np.max(row_vals[met]))
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)

        last_im = None
        # TDVP heatmaps per D_max
        for ci, D in enumerate(Ds):
            ax = fig.add_subplot(gs[ri, ci])
            M = grids[(met, D)]
            cmap = "viridis" if met == "chi" else "plasma_r"
            im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            last_im = im
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels([str(n) for n in Ns], fontsize=10)
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels([str(k) for k in Ks], fontsize=10)
            if ci == 0:
                ax.set_ylabel("K (Trotter steps)", fontsize=11)
            if ri == 3:
                ax.set_xlabel("N (grid points / dim)", fontsize=11)
            if ci == 0:
                ax.text(-0.45, 0.5, metric_titles[met],
                        transform=ax.transAxes,
                        ha="right", va="center", fontsize=12, fontweight="bold",
                        rotation=90)

            def vnorm(x):
                if metric_log[met]:
                    return (np.log(max(x, 1e-12)) - np.log(max(vmin, 1e-12))) \
                           / max(np.log(vmax) - np.log(max(vmin, 1e-12)), 1e-12)
                else:
                    return (x - vmin) / max(vmax - vmin, 1e-12)
            annotate_cells(ax, M, metric_fmt[met], vnorm)

        # Reference panels (sw/mmd only) — share colorscale with TDVP row
        for off, ref in enumerate(REFS):
            ax = fig.add_subplot(gs[ri, n_cols_tdvp + off])
            if met == "chi" or (met, ref) not in ref_grids:
                ax.axis("off")
                if met == "chi":
                    ax.text(0.5, 0.5, "n/a\n(non-MPS)",
                            ha="center", va="center", fontsize=10, color="gray",
                            transform=ax.transAxes)
                continue
            M = ref_grids[(met, ref)]
            im = ax.imshow(M, cmap="plasma_r", norm=norm, origin="lower", aspect="auto")
            last_im = im
            ax.set_xticks(range(len(Ns)))
            ax.set_xticklabels([str(n) for n in Ns], fontsize=10)
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels([str(k) for k in Ks], fontsize=10)
            if ri == 3:
                ax.set_xlabel("N", fontsize=11)
            # Frame ref panels in gray to visually separate from TDVP block
            for s in ax.spines.values():
                s.set_edgecolor("dimgray"); s.set_linewidth(1.2)

            def vnorm_ref(x):
                if metric_log[met]:
                    return (np.log(max(x, 1e-12)) - np.log(max(vmin, 1e-12))) \
                           / max(np.log(vmax) - np.log(max(vmin, 1e-12)), 1e-12)
                else:
                    return (x - vmin) / max(vmax - vmin, 1e-12)
            annotate_cells(ax, M, metric_fmt[met], vnorm_ref)

        # Shared colorbar for the row
        cax = fig.add_subplot(gs[ri, n_cols])
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(metric_titles[met].split(" (")[0], fontsize=10)

    fig.suptitle(
        f"{label}  —  HP heatmaps over (N, K), one column per D_max",
        fontsize=14, fontweight="bold", y=0.995,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
