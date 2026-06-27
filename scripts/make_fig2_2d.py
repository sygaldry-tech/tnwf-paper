"""Figure 2 of the paper: 2D wavefunction-flow visualisations at N=64.

Two rows × three columns:
  Row 1: Swiss roll (d=2)
  Row 2: 2D Gaussian mixture (d=2)
Columns: Target / JAM / Dense (TN baseline). Each panel is a 2D
density estimated from sample points via Gaussian KDE on the
[0, L)^2 frame.

Sources (all at N=64, best-K within the cell):
  Swiss roll target/JAM/Dense:
    data/swiss_roll_2d/{dense,jam}/N64/seed0.npz
  GMM target/JAM/Dense:
    data/gmm_2d_hp/{dense,jam}/N64_K32_D32/seed0.npz   (best dense)
    data/gmm_2d_hp/{dense,jam}/N64_K16_D8/seed0.npz    (best jam)

The "Target" column re-uses the target-samples field stored in the
chosen cell's npz (same target draw as the method run).

Usage:
    uv run python scripts/make_fig2_2d.py \\
        --out figures/fig2_2d.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Best-cell paths at N=64 for each (target, method) pair.
SOURCES = {
    "swiss_roll": {
        "target": ("data/swiss_roll_2d/dense/N64/seed0.npz", "target"),
        "jam":    ("data/swiss_roll_2d/jam/N64/seed0.npz",   "samples_T"),
        "dense":  ("data/swiss_roll_2d/dense/N64/seed0.npz", "samples_T"),
        "label":  "Swiss roll",
        "L":      5.0,
    },
    "gmm_2d": {
        "target": ("data/gmm_2d_hp/dense/N64_K32_D32/seed0.npz", "target"),
        "jam":    ("data/gmm_2d_hp/jam/N64_K16_D8/seed0.npz",    "samples_T"),
        "dense":  ("data/gmm_2d_hp/dense/N64_K32_D32/seed0.npz", "samples_T"),
        "label":  "Gaussian mixture",
        "L":      8.0,
    },
}


def _load_samples(npz_path: str, key: str) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    return np.asarray(z[key], dtype=np.float64)


def _density_grid(samples: np.ndarray, L: float, N: int = 64,
                  bandwidth: float | None = None) -> np.ndarray:
    """Gaussian-KDE density on an N×N grid covering [0, L)^2."""
    from scipy.stats import gaussian_kde
    # samples already shifted into [0, L)^2 frame.
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    XX, YY = np.meshgrid(x, y, indexing="xy")
    pts = np.vstack([XX.ravel(), YY.ravel()])
    kde = gaussian_kde(samples.T, bw_method=bandwidth)
    return kde(pts).reshape(N, N), x, y


def _read_metric(npz_path: str, key: str) -> float | None:
    z = np.load(npz_path, allow_pickle=True)
    if key in z.files:
        v = np.asarray(z[key])
        return float(v[-1]) if v.ndim else float(v)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/fig2_2d.pdf")
    p.add_argument("--N_kde", type=int, default=96,
                   help="grid resolution for KDE density visualisation")
    args = p.parse_args()

    cmap = "magma"

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.18, h_pad=0.30)

    col_titles = ["target", "JAM", "Dense (TN baseline)"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight="bold",
                                    pad=8)

    for row_idx, key in enumerate(["swiss_roll", "gmm_2d"]):
        spec = SOURCES[key]
        L = spec["L"]
        # --- target ---
        npz, fld = spec["target"]
        samples = _load_samples(npz, fld)
        rho, x, y = _density_grid(samples, L=L, N=args.N_kde)
        ax = axes[row_idx, 0]
        ax.imshow(rho, extent=[0, L, 0, L], origin="lower",
                  cmap=cmap, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(spec["label"], fontsize=13, fontweight="bold")

        # --- JAM ---
        npz, fld = spec["jam"]
        samples = _load_samples(npz, fld)
        rho, _, _ = _density_grid(samples, L=L, N=args.N_kde)
        sw = _read_metric(npz, "sw"); mmd = _read_metric(npz, "mmd")
        ax = axes[row_idx, 1]
        ax.imshow(rho, extent=[0, L, 0, L], origin="lower",
                  cmap=cmap, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if sw is not None:
            ax.text(0.97, 0.04, f"SW={sw:.3f}\nMMD={mmd:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black",
                              alpha=0.55, ec="none"))

        # --- Dense ---
        npz, fld = spec["dense"]
        samples = _load_samples(npz, fld)
        rho, _, _ = _density_grid(samples, L=L, N=args.N_kde)
        sw = _read_metric(npz, "sw"); mmd = _read_metric(npz, "mmd")
        ax = axes[row_idx, 2]
        ax.imshow(rho, extent=[0, L, 0, L], origin="lower",
                  cmap=cmap, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if sw is not None:
            ax.text(0.97, 0.04, f"SW={sw:.3f}\nMMD={mmd:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc="black",
                              alpha=0.55, ec="none"))

    # Per-row x-axis label.
    axes[1, 1].set_xlabel(r"position grid, $N=64$", fontsize=11)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
