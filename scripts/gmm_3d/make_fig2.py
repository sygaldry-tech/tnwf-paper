"""Figure 2: gmm_3d demo.

Layout:
  Row 1: 2D scatter (first 2 coords) per method at one fixed (N, K).
  Rows 2-4: per-method (N × K) heatmap of SW / MMD / NLL.
            Shared color scale within each metric row.
  Row 5: NLL_method/NLL_dense vs χ/N^d compression-ratio scatter (single panel).
  Row 6: legend bar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np

METHODS = [
    "jam", "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]
METHOD_LABEL = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_als":    "TCI+ALS",
    "aci":        "ACI",
    "tci_tdvp1":  "TCI+TDVP1",
    "tci_tdvp2":  "TCI+TDVP2",
}
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_als":    "tab:blue",
    "aci":        "tab:orange",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}


_NEEDED_KEYS = ("sw", "mmd", "chi_max", "samples_T", "target", "d")


def _load_eager(path: Path) -> dict | None:
    """Open + force-read needed arrays so we hit any decompression errors here."""
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        return {k: np.asarray(z[k]) for k in _NEEDED_KEYS}
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        print(f"[skip {path}] {type(e).__name__}: {e}")
        return None


def load_runs_grid(results_dir: Path) -> dict:
    """Load all (method, N, K) → list[seed runs]. Skip files mid-written by a
    concurrent runner (bad-zip / OSError) so partial state doesn't kill the plot."""
    out: dict[tuple[str, int, int], list] = {}
    for method in METHODS:
        method_dir = results_dir / method
        if not method_dir.exists():
            continue
        for run_dir in method_dir.glob("N*_K*"):
            try:
                _, N_str, K_str = ("_" + run_dir.name).rsplit("_", 2)
                N = int(N_str.lstrip("N"))
                K = int(K_str.lstrip("K"))
            except (ValueError, IndexError):
                continue
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
    p.add_argument("--results_dir", default="data/gmm_3d")
    p.add_argument("--out", default="results/gmm_3d/fig2.pdf")
    p.add_argument("--N_show", type=int, default=16, help="N for top-row scatter")
    p.add_argument("--K_show", type=int, default=16, help="K for top-row scatter")
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    grid = load_runs_grid(res_dir)
    if not grid:
        raise SystemExit(f"No results at {res_dir}/{{method}}/N*_K*/seed*.npz")

    Ns = sorted({N for (_, N, _) in grid})
    Ks = sorted({K for (_, _, K) in grid})
    n_methods = len(METHODS)

    # 5-row layout: scatter / SW heatmaps / MMD heatmaps / accuracy-vs-comp / legend
    fig = plt.figure(figsize=(3.6 * n_methods, 20), dpi=200)
    gs = GridSpec(
        5, n_methods, figure=fig, hspace=0.55, wspace=0.45,
        height_ratios=[1.0, 1.0, 1.0, 1.2, 0.06],
    )

    # ── Row 1: t-SNE projection per method ──────────────────────────────────
    # Each panel: t-SNE of (target ∪ method_samples) jointly so target/method
    # share the same embedding within the panel. Cross-panel coords are NOT
    # directly comparable (t-SNE is per-fit), but the cluster structure is.
    from sklearn.manifold import TSNE
    for j, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, j])
        runs = grid.get((method, args.N_show, args.K_show), [])
        if not runs:
            ax.set_title(f"{METHOD_LABEL[method]} (no data)", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        x = np.asarray(runs[0]["samples_T"])[:600]              # cap for t-SNE speed
        tgt = np.asarray(runs[0]["target"])[:600]
        joined = np.concatenate([tgt, x], axis=0)
        perplexity = min(30, max(5, joined.shape[0] // 5))
        emb = TSNE(
            n_components=2, init="pca", perplexity=perplexity,
            random_state=0, learning_rate="auto",
        ).fit_transform(joined)
        emb_tgt = emb[: tgt.shape[0]]
        emb_x = emb[tgt.shape[0]:]
        ax.scatter(emb_tgt[:, 0], emb_tgt[:, 1], s=8, alpha=0.25, c="gray", label="target")
        ax.scatter(emb_x[:, 0], emb_x[:, 1], s=8, alpha=0.7,
                   c=METHOD_COLORS[method], label=METHOD_LABEL[method])
        ax.set_title(METHOD_LABEL[method], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"t-SNE samples\nN={args.N_show}, K={args.K_show}", fontsize=9)

    # ── Rows 2-3: per-method N×K heatmap, one row per metric ────────────────
    metrics = ["sw", "mmd"]
    metric_titles = {"sw": "SW (sliced Wasserstein)",
                     "mmd": "MMD (RBF)"}
    for row_offset, metric in enumerate(metrics, start=1):
        # Pre-collect all values for shared colormap range
        all_vals = []
        per_method_grid = {}
        for method in METHODS:
            M = np.full((len(Ks), len(Ns)), np.nan)        # rows = K (y), cols = N (x)
            for ki, K in enumerate(Ks):
                for ni, N in enumerate(Ns):
                    runs = grid.get((method, N, K), [])
                    if runs:
                        M[ki, ni] = float(np.nanmean([r[metric][-1] for r in runs]))
            per_method_grid[method] = M
            all_vals.extend(M[~np.isnan(M)].tolist())
        if not all_vals:
            continue
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

        last_im = None
        for j, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[row_offset, j])
            M = per_method_grid[method]
            im = ax.imshow(np.clip(M, max(vmin, 1e-4), None),
                           cmap="plasma_r",
                           norm=LogNorm(vmin=max(vmin, 1e-4), vmax=vmax),
                           origin="lower", aspect="auto")
            last_im = im
            ax.set_xticks(np.arange(len(Ns)))
            ax.set_xticklabels([f"N={n}" for n in Ns], fontsize=9)
            ax.set_yticks(np.arange(len(Ks)))
            ax.set_yticklabels([f"K={k}" for k in Ks], fontsize=9)
            for ki in range(M.shape[0]):
                for ni in range(M.shape[1]):
                    v = M[ki, ni]
                    if not np.isnan(v):
                        # Pick text colour for contrast
                        norm_v = (v - vmin) / max(vmax - vmin, 1e-12)
                        text_c = "white" if norm_v > 0.55 else "black"
                        ax.text(ni, ki, f"{v:.3f}", ha="center", va="center",
                                fontsize=11, color=text_c, fontweight="bold")
            if j == 0:
                ax.set_ylabel(metric_titles[metric], fontsize=10, fontweight="bold")
            ax.set_title(METHOD_LABEL[method], fontsize=10)
        # One shared colorbar at the right of the row
        if last_im is not None:
            cax = fig.add_axes([
                0.92,
                0.78 - 0.18 * (row_offset - 1),
                0.012,
                0.12,
            ])
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(metric.upper(), fontsize=9)

    # ── Row 4: SW gap from Dense vs compression, with error bars ────────────
    # y = SW_method - SW_dense  (positive ⇒ method is worse than Dense; 0 = par)
    # x = χ / N^d (compression ratio); log scale.
    # error bar = sqrt(σ²_method / n_seeds_method  +  σ²_dense / n_seeds_dense)
    ax_ratio = fig.add_subplot(gs[3, 1:max(2, n_methods - 1)])
    for method in METHODS:
        if method == "dense":
            continue
        gaps, gap_errs, comps = [], [], []
        for (m, N, K), runs in grid.items():
            if m != method:
                continue
            dense_runs = grid.get(("dense", N, K), [])
            if not dense_runs:
                continue
            sw_m = np.array([float(r["sw"][-1]) for r in runs])
            sw_d = np.array([float(r["sw"][-1]) for r in dense_runs])
            mu_m, mu_d = sw_m.mean(), sw_d.mean()
            se_m = sw_m.std(ddof=1) / np.sqrt(len(sw_m)) if len(sw_m) > 1 else 0.0
            se_d = sw_d.std(ddof=1) / np.sqrt(len(sw_d)) if len(sw_d) > 1 else 0.0
            chi = float(np.nanmean([r["chi_max"][-1] for r in runs]))
            d = int(runs[0]["d"])
            gaps.append(mu_m - mu_d)
            gap_errs.append(np.sqrt(se_m ** 2 + se_d ** 2))
            comps.append(chi / max(N ** d, 1))
        if gaps:
            ax_ratio.errorbar(
                comps, gaps, yerr=gap_errs, fmt="o", ms=8,
                c=METHOD_COLORS[method], label=METHOD_LABEL[method],
                alpha=0.85, capsize=4, ecolor=METHOD_COLORS[method],
                markeredgecolor="black", markeredgewidth=0.5, lw=1.0,
            )
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("compression ratio  χ / N^d  (log scale)", fontsize=11)
    ax_ratio.set_ylabel("SW gap from Dense\n(SW$_\\mathrm{method}$ − SW$_\\mathrm{Dense}$)",
                        fontsize=11)
    ax_ratio.set_title(
        "Accuracy gap vs compression — Gaussian Mixture (3D)\n"
        "0 = matches Dense ◇ positive = worse ◇ negative = better",
        fontsize=11,
    )
    ax_ratio.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7,
                     label="matches Dense")
    ax_ratio.legend(fontsize=9, loc="best", ncol=2)
    ax_ratio.grid(alpha=0.3)

    # ── Row 5: method legend bar ────────────────────────────────────────────
    lax = fig.add_subplot(gs[4, :])
    lax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=3, label=METHOD_LABEL[m])
               for m in METHODS]
    lax.legend(handles=handles, ncol=len(METHODS), loc="center", frameon=False, fontsize=10)

    fig.suptitle("Figure 2 — Gaussian Mixture (3D)", fontsize=14, fontweight="bold", y=0.995)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
