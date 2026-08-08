"""Figure 5 of the paper: t-SNE overlays for d ∈ {3, 5, 7}.

3 columns × 3 rows grid. Each column is one d. Rows: Exact / JAM / TCI+1TDVP.
Each panel fits its own t-SNE on (target ∪ method_samples) so cluster
structure is comparable WITHIN a panel; cross-panel coords are NOT comparable.

The "Exact" row uses an independent target draw (visual sample-size floor).

Usage:
    uv run python scripts/make_fig_tsne_grid.py \\
        --out figures/fig_tsne_grid.pdf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Magma ramp, matching the submitted figure and the rest of the paper's palette
# (Fig 4 / fig_scaling_bounds uses magma at 0.18 / 0.40 / 0.76).
#
# This file previously carried a Wong/Okabe-Ito palette (gray #cccccc target,
# black Exact, vermillion #D55E00 TDVP1) predating the manuscript's switch to
# magma. That was the cause of the mismatch against the committed fig_tsne_grid.pdf:
# the released script drew a stale palette, so the script needed correcting, not
# the figure.
#
# Pinned as explicit triples rather than plt.cm.magma(f) calls: magma is a
# 256-entry lookup table, so magma(f) is piecewise constant in f and a
# "close enough" f silently lands on a neighboring stop. These are LUT indices
# 38 / 128 / 199, read back out of the committed figure, so they reproduce it
# regardless of any future change to matplotlib's colormap data.
COLOR_TARGET = (0.146785, 0.068738, 0.334011)   # magma 38/255 — first-drawn series
COLOR_EXACT  = (0.716387, 0.214982, 0.475290)   # magma 128/255
COLOR_TDVP1  = (0.992196, 0.587502, 0.406299)   # magma 199/255
COLOR_JAM    = "#009E73"       # unused by this figure; kept for callers

# Which (cell, seed) each panel plots. Derived, not carried by hand: the argmin
# moved when the half-cell sampling bias was corrected, so a constant copied
# from a previous release would silently plot a run that is no longer the one
# the panel claims to show.
#
# Note this is a per-*run* argmin (a single seed), whereas Table 1 reports the
# best cell by seed-averaged SW. The two need not agree, and the annotated
# value is the plotted run's own SW -- which is what the caption now says.
DATASET_DIR = {3: "data/gmm_3d_hp_v2", 5: "data/gmm_5d_hp", 7: "data/gmm_7d_hp"}
PANEL_METHODS = ("jam", "tci_tdvp1")
_CELL_PATTERN = re.compile(r"^N(\d+)_K(\d+)(?:_D(\d+))?$")


def derive_cells(dims=(3, 5, 7)) -> dict:
    """Locate, per (d, method), the single run with the lowest endpoint SW."""
    out: dict[int, dict] = {}
    for d in dims:
        out[d] = {}
        for method in PANEL_METHODS:
            base = Path(DATASET_DIR[d]) / method
            best = None
            for sub in (sorted(base.iterdir()) if base.exists() else []):
                if not _CELL_PATTERN.match(sub.name):
                    continue
                for f in sorted(sub.glob("seed*.npz")):
                    z = np.load(f, allow_pickle=True)
                    if "sw_endpoint" not in z.files:
                        raise SystemExit(
                            f"{f} predates the node-centered coordinate "
                            f"convention; run scripts/migrate_archive.py first.")
                    sw = float(np.asarray(z["sw_endpoint"]))
                    seed = int(f.stem.replace("seed", ""))
                    if best is None or sw < best[0]:
                        best = (sw, str(sub), seed)
            if best is None:
                raise SystemExit(f"no runs found for d={d} {method}")
            out[d][method] = (best[1], best[2])
    return out
N_SHOW = 500


def _load(dir_path: Path, seed: int) -> dict:
    z = np.load(dir_path / f"seed{seed}.npz", allow_pickle=True)
    return {
        "samples_T": np.asarray(z["samples_T"]),
        "target":    np.asarray(z["target"]),
        "sw":        float(np.asarray(z["sw_endpoint"])),
        "N":         int(z["N"]),
        "K":         int(z["K"]),
    }


def _tsne(joined: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, joined.shape[0] // 5))
    return TSNE(
        n_components=2, init="pca", perplexity=perplexity,
        random_state=seed, learning_rate="auto",
    ).fit_transform(joined)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/fig_tsne_grid.pdf")
    args = p.parse_args()

    ds = [3, 5, 7]
    CELLS = derive_cells(tuple(ds))
    for d in ds:
        for m, (dir_, seed) in CELLS[d].items():
            print(f"[fig5] d={d} {m}: {dir_} seed{seed}")

    # 1 row x 3 cols: each panel overlays three sample sets via a single
    # joint t-SNE: "true" target reference, "Exact" independent target
    # draw (noise floor), and TCI+1TDVP generated samples.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.20, h_pad=0.10)

    for col_idx, d in enumerate(ds):
        tdvp1 = _load(Path(CELLS[d]["tci_tdvp1"][0]), CELLS[d]["tci_tdvp1"][1])
        n = N_SHOW
        true_target = tdvp1["target"][:n]
        seed_b = 1 - CELLS[d]["tci_tdvp1"][1]
        try:
            exact_draw = _load(Path(CELLS[d]["tci_tdvp1"][0]), seed_b)["target"][:n]
        except FileNotFoundError:
            exact_draw = _load(Path(CELLS[d]["jam"][0]),
                               1 - CELLS[d]["jam"][1])["target"][:n]
        tdvp1_pts = tdvp1["samples_T"][:n]

        # Joint t-SNE on the union so the three clouds share coordinates.
        joined = np.concatenate([true_target, exact_draw, tdvp1_pts], axis=0)
        emb = _tsne(joined, seed=0)
        n_t = true_target.shape[0]; n_e = exact_draw.shape[0]
        emb_true  = emb[:n_t]
        emb_exact = emb[n_t : n_t + n_e]
        emb_tdvp1 = emb[n_t + n_e:]

        ax = axes[col_idx]
        # Bottom layer: "true" target reference.
        ax.scatter(emb_true[:, 0], emb_true[:, 1], s=24, alpha=0.45,
                   c=COLOR_TARGET, edgecolors="none", label="True",
                   zorder=2)
        # Middle layer: Exact independent draw (sample-size floor).
        ax.scatter(emb_exact[:, 0], emb_exact[:, 1], s=20, alpha=0.65,
                   c=COLOR_EXACT, edgecolors="none", label="Exact",
                   zorder=3)
        # Top layer: TCI+1TDVP generated samples.
        ax.scatter(emb_tdvp1[:, 0], emb_tdvp1[:, 1], s=20, alpha=0.65,
                   c=COLOR_TDVP1, edgecolors="none", label="TCI+1TDVP",
                   zorder=4)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$d = {d}$", fontsize=24, fontweight="bold")
        # SW annotation chip below the cloud.
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - 0.18 * (ylim[1] - ylim[0]), ylim[1])
        ax.text(0.5, 0.035,
                f"TCI+1TDVP SW = {tdvp1['sw']:.3f}",
                transform=ax.transAxes, va="bottom", ha="center",
                fontsize=17, color="black",
                bbox=dict(boxstyle="round,pad=0.30", fc="white",
                          ec="lightgray", alpha=0.95))

    # Single legend for all three panels.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=20, frameon=False,
               handlelength=1.0, columnspacing=1.6)
    for h in leg.legend_handles:
        h.set_sizes([90])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
