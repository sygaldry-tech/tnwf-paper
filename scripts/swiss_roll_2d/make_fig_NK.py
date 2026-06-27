"""Figure 4: Swiss roll (N, K) phase diagram.

For each method, a heatmap of mean SW over (N, K). Shared colorscale across
methods so the heatmaps are directly comparable.

Layout: 2×3 grid of heatmaps (6 methods) + 1 small bottom panel showing
SW vs N at fixed K=K_max (convergence curves, all methods overlaid).
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

METHODS = [
    "jam", "dense",     "tci_tdvp1", "tci_tdvp2",
]
METHOD_LABEL = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+TDVP1",
    "tci_tdvp2":  "TCI+TDVP2",
}
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}

NK_PATTERN = re.compile(r"^N(\d+)_K(\d+)$")
_NEEDED_KEYS = ("sw", "mmd", "chi_max")


def _load_eager(path: Path) -> dict | None:
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        return {k: np.asarray(z[k]) for k in _NEEDED_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        print(f"[skip {path}] {type(e).__name__}: {e}")
        return None


def load_grid(results_dir: Path) -> dict:
    """Return {(method, N, K): list[seed runs]}."""
    out: dict[tuple[str, int, int], list] = {}
    for method in METHODS:
        method_dir = results_dir / method
        if not method_dir.exists():
            continue
        for run_dir in method_dir.iterdir():
            m = NK_PATTERN.match(run_dir.name)
            if not m:
                continue
            N, K = int(m.group(1)), int(m.group(2))
            seeds = []
            for f in sorted(run_dir.glob("seed*.npz")):
                rec = _load_eager(f)
                if rec is not None:
                    seeds.append(rec)
            if seeds:
                out[(method, N, K)] = seeds
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/swiss_roll_NK")
    p.add_argument("--out", default="results/swiss_roll_NK/fig_NK.pdf")
    p.add_argument("--metric", default="sw", choices=["sw", "mmd"])
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    grid = load_grid(res_dir)
    if not grid:
        raise SystemExit(f"No results at {res_dir}/{{method}}/N*_K*/seed*.npz")

    Ns = sorted({N for (_, N, _) in grid})
    Ks = sorted({K for (_, _, K) in grid})

    # Shared colorscale across all methods (ignore JAM-only outliers if huge?)
    all_vals = []
    per_method_M: dict[str, np.ndarray] = {}
    for method in METHODS:
        M = np.full((len(Ks), len(Ns)), np.nan)
        for ki, K in enumerate(Ks):
            for ni, N in enumerate(Ns):
                runs = grid.get((method, N, K), [])
                if runs:
                    M[ki, ni] = float(np.nanmean([r[args.metric][-1] for r in runs]))
        per_method_M[method] = M
        all_vals.extend(M[~np.isnan(M)].tolist())
    if not all_vals:
        raise SystemExit("All cells empty — check --results_dir")
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # 2×3 heatmap grid + bottom convergence panel + legend bar
    n_methods = len(METHODS)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols     # ceil

    fig = plt.figure(figsize=(5.0 * n_cols + 1.5, 3.6 * n_rows + 4))
    gs = GridSpec(
        n_rows + 2, n_cols, figure=fig, hspace=0.45, wspace=0.35,
        height_ratios=[1.0] * n_rows + [1.4, 0.07],
    )

    last_im = None
    for idx, method in enumerate(METHODS):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        M = per_method_M[method]
        im = ax.imshow(np.clip(M, max(vmin, 1e-4), None),
                       cmap="plasma_r",
                       norm=LogNorm(vmin=max(vmin, 1e-4), vmax=vmax),
                       origin="lower", aspect="auto")
        last_im = im
        ax.set_xticks(np.arange(len(Ns)))
        ax.set_xticklabels([str(n) for n in Ns], fontsize=10)
        ax.set_yticks(np.arange(len(Ks)))
        ax.set_yticklabels([str(k) for k in Ks], fontsize=10)
        ax.set_xlabel("N (grid resolution)", fontsize=10)
        if c == 0:
            ax.set_ylabel("K (Trotter steps)", fontsize=10)
        ax.set_title(METHOD_LABEL[method], fontsize=12, fontweight="bold",
                     color=METHOD_COLORS[method])
        for ki in range(M.shape[0]):
            for ni in range(M.shape[1]):
                v = M[ki, ni]
                if not np.isnan(v):
                    norm_v = (v - vmin) / max(vmax - vmin, 1e-12)
                    text_c = "white" if norm_v > 0.55 else "black"
                    ax.text(ni, ki, f"{v:.3f}",
                            ha="center", va="center",
                            fontsize=10, color=text_c, fontweight="bold")

    # Shared colorbar on the right
    if last_im is not None:
        cax = fig.add_axes([0.93, 0.55, 0.013, 0.30])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(f"{args.metric.upper()}  (lower = better)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Bottom: SW vs N at fixed K=K_max — convergence curves with error bars
    ax_conv = fig.add_subplot(gs[n_rows, :])
    K_max = max(Ks)
    for method in METHODS:
        means, stds = [], []
        for N in Ns:
            runs = grid.get((method, N, K_max), [])
            if runs:
                vals = np.array([float(r[args.metric][-1]) for r in runs])
                means.append(vals.mean())
                stds.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan)
                stds.append(0.0)
        ax_conv.errorbar(Ns, means, yerr=stds, marker="o", lw=1.6, capsize=4,
                         color=METHOD_COLORS[method], label=METHOD_LABEL[method])
    ax_conv.set_xscale("log", base=2)
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("N (grid resolution)", fontsize=11)
    ax_conv.set_ylabel(f"{args.metric.upper()}  (mean ± SE)", fontsize=11)
    ax_conv.set_title(
        f"Convergence with N at K = {K_max}  —  Swiss roll (2D)",
        fontsize=12,
    )
    ax_conv.legend(fontsize=9, loc="best", ncol=3)
    ax_conv.grid(True, which="both", alpha=0.3)

    # Legend bar at the very bottom
    lax = fig.add_subplot(gs[n_rows + 1, :])
    lax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=3, label=METHOD_LABEL[m])
               for m in METHODS]
    lax.legend(handles=handles, ncol=len(METHODS), loc="center", frameon=False, fontsize=10)

    fig.suptitle(
        "Swiss roll (2D) — N × K phase diagram (per method)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
