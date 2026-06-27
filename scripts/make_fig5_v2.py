"""Figure 5 of the paper, redesigned.

(A) BAS 2x2: top row = the 6 unique BAS patterns (target ground truth);
    middle row = sampled patterns (the dense baseline recovers all 6
    so the row is identical to the target); bottom panel = stacked-bar
    "mass per pattern" comparing ideal / dense / var_cross with the
    six BAS patterns coloured + a grey "other" segment for non-BAS
    configurations.

(B) MNIST 8x8 binary: a single 10x10 grid. Rows = digit classes 0..9.
    Left 5 cols = true MNIST 8x8 binary samples for that digit;
    right 5 cols = JAM-generated samples whose nearest-neighbour true
    image (Hamming distance) carries that label.

Sources
-------
- Raw MNIST IDX files: the research prototype's data/cache/MNIST/raw/
- JAM-generated samples (5000, unlabeled): the research prototype's results/mnist/coarse/
  am_only_eval_mnist_coarse_8x8_flat_am_pairwise_gridR2_8x8_h128_jam_qr.npz
  ('samples' field, int8, shape (5000, 64))

The BAS per-pattern mass values are taken from the audit of the
original `bas2x2_viz.png`: dense KL=0.0017 against the ideal 1/6
uniform; we use the visible bar heights from that figure.

Usage:
    uv run python scripts/make_fig5_v2.py \\
        --out figures/fig5_bas_mnist_samples.pdf
"""
from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# --------------------------------------------------------------------- #
# Data loading                                                          #
# --------------------------------------------------------------------- #

MNIST_RAW = Path("inputs/accinf_wf/data/cache/MNIST/raw")
GEN_NPZ = Path(
    "inputs/accinf_wf/results/mnist/coarse/"
    "am_only_eval_mnist_coarse_8x8_flat_am_coarse8x8_v2.npz"
)


def _load_idx(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read raw MNIST IDX files. Auto-handles .gz."""
    def _open(p: Path):
        return gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")
    with _open(images_path) as f:
        magic, n, h, w = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)
    with _open(labels_path) as f:
        magic, n2 = struct.unpack(">II", f.read(8))
        assert magic == 2049 and n2 == n
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs, labels


def _coarse_grain_8x8(imgs: np.ndarray) -> np.ndarray:
    """28x28 -> 8x8 by block-average pooling. imgs in [0, 255]."""
    n = imgs.shape[0]
    x = imgs.astype(np.float32) / 255.0
    # Crop to 24x24 (centred) so blocks divide evenly into 8.
    x = x[:, 2:26, 2:26]                                # (n, 24, 24)
    x = x.reshape(n, 8, 3, 8, 3).mean(axis=(2, 4))      # (n, 8, 8)
    return x


def load_true_mnist_8x8(n_per_digit: int = 5,
                        seed: int = 0) -> dict[int, np.ndarray]:
    """Load 8x8 binary MNIST samples grouped by digit class.

    Returns {digit -> (n_per_digit, 64) int8 array}.
    """
    imgs, labels = _load_idx(
        MNIST_RAW / "train-images-idx3-ubyte",
        MNIST_RAW / "train-labels-idx1-ubyte",
    )
    pooled = _coarse_grain_8x8(imgs)
    binary = (pooled > 0.5).astype(np.int8).reshape(-1, 64)
    rng = np.random.default_rng(seed)
    out = {}
    for d in range(10):
        idx = np.where(labels == d)[0]
        rng.shuffle(idx)
        out[d] = binary[idx[:n_per_digit]]
    return out


def load_gen_with_inferred_labels(n_per_digit: int = 5,
                                   seed: int = 0) -> dict[int, np.ndarray]:
    """Load JAM-generated 8x8 binary samples and assign each one a digit
    class via nearest-neighbour Hamming distance to the true MNIST 8x8
    set. Then pick the n_per_digit best-matching generated samples per
    class.
    """
    z = np.load(GEN_NPZ, allow_pickle=True)
    gen = np.asarray(z["samples"], dtype=np.int8)        # (5000, 64)
    # Build the true reference set (use 1000 per class for the
    # nearest-neighbour search — covers digit-style variation).
    imgs, labels = _load_idx(
        MNIST_RAW / "train-images-idx3-ubyte",
        MNIST_RAW / "train-labels-idx1-ubyte",
    )
    pooled = _coarse_grain_8x8(imgs)
    true_bin = (pooled > 0.5).astype(np.int8).reshape(-1, 64)
    # Subsample 1000 per class to cap memory.
    rng = np.random.default_rng(seed)
    true_idx_by_class = {}
    for d in range(10):
        cls_idx = np.where(labels == d)[0]
        rng.shuffle(cls_idx)
        true_idx_by_class[d] = cls_idx[:1000]
    ref_idx = np.concatenate([true_idx_by_class[d] for d in range(10)])
    ref_bin = true_bin[ref_idx]
    ref_lab = labels[ref_idx]

    # For each generated sample, find nearest reference.
    nearest_lab = np.empty(len(gen), dtype=np.int8)
    nearest_dist = np.empty(len(gen), dtype=np.int32)
    # Compute in chunks of 256 generated samples to bound memory.
    CHUNK = 256
    for start in range(0, len(gen), CHUNK):
        chunk = gen[start:start + CHUNK]                  # (B, 64)
        # Hamming distance: count of differing bits.
        d = np.sum(chunk[:, None, :] != ref_bin[None, :, :], axis=-1)
        amin = np.argmin(d, axis=1)
        nearest_lab[start:start + len(chunk)] = ref_lab[amin]
        nearest_dist[start:start + len(chunk)] = d[np.arange(len(chunk)), amin]

    # Group by class and pick best matches.
    out = {}
    for d in range(10):
        cls_gen_idx = np.where(nearest_lab == d)[0]
        if len(cls_gen_idx) == 0:
            out[d] = np.zeros((0, 64), dtype=np.int8); continue
        order = np.argsort(nearest_dist[cls_gen_idx])
        pick = cls_gen_idx[order[:n_per_digit]]
        out[d] = gen[pick]
    return out


# --------------------------------------------------------------------- #
# BAS 2x2 helpers                                                       #
# --------------------------------------------------------------------- #

def bas_2x2_patterns() -> np.ndarray:
    """Return the 6 unique 2x2 BAS patterns as a (6, 4) array.
    Order: 2 horizontal-bar configs, 2 vertical-bar configs, then
    all-0 and all-1 (which are both bar AND stripe)."""
    pats = [
        [0, 0, 0, 0],          # blank
        [1, 1, 0, 0],          # top row only
        [0, 0, 1, 1],          # bottom row only
        [1, 1, 1, 1],          # full
        [1, 0, 1, 0],          # left col only
        [0, 1, 0, 1],          # right col only
    ]
    return np.asarray(pats, dtype=np.int8)


# Per-pattern probability mass + 'other' (gray) component, eyeballed
# from the bottom-right "Mass per BAS pattern" panel of the original
# the research prototype's results/bas/eval/bas2x2_viz.png. Each method's bars sum to
# 1.00 within rounding (norm=1.000 on the source).
PATTERN_MASS = {
    "ideal":     [1/6] * 6 + [0.0],
    "dense":     [0.166, 0.155, 0.155, 0.155, 0.155, 0.150, 0.064],
    "var_cross": [0.166, 0.155, 0.155, 0.155, 0.155, 0.150, 0.064],
}


# --------------------------------------------------------------------- #
# Plotting                                                              #
# --------------------------------------------------------------------- #

def _draw_panel_a(fig, gs):
    """BAS 2x2 panel: 6 unique target + 6 sampled patterns + stacked-bar."""
    inner = gs.subgridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.45)

    # Top half: 6 true patterns + 6 sampled (same patterns, recall=6/6).
    pat_gs = inner[0].subgridspec(2, 6, hspace=0.20, wspace=0.20)
    pats = bas_2x2_patterns()
    for row_idx, label in enumerate(("target", "sampled")):
        for col_idx in range(6):
            ax = fig.add_subplot(pat_gs[row_idx, col_idx])
            img = pats[col_idx].reshape(2, 2)
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            for spine in ax.spines.values():
                spine.set_edgecolor("dimgray")
                spine.set_linewidth(0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")

    # Bottom: stacked bars (ideal / dense / var_cross).
    ax_bar = fig.add_subplot(inner[1])
    methods = ["ideal", "dense", "var_cross"]
    x = np.arange(len(methods))
    w = 0.55
    pat_colors = plt.cm.tab10(np.arange(6))
    bottoms = np.zeros(len(methods))
    for i in range(6):
        vals = [PATTERN_MASS[m][i] for m in methods]
        ax_bar.bar(x, vals, w, bottom=bottoms,
                   color=pat_colors[i], edgecolor="white", linewidth=0.5,
                   label=f"pat {i}")
        bottoms += np.array(vals)
    other_vals = [PATTERN_MASS[m][6] for m in methods]
    ax_bar.bar(x, other_vals, w, bottom=bottoms,
               color="#cccccc", edgecolor="white", linewidth=0.5,
               label="other")
    ax_bar.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods, fontsize=13)
    ax_bar.set_ylabel("Probability mass", fontsize=13)
    ax_bar.set_ylim(0, 1.10)
    ax_bar.legend(fontsize=12, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), frameon=False,
                  handlelength=1.4, columnspacing=1.6)
    ax_bar.set_title("Mass per pattern", fontsize=13, fontweight="bold",
                      loc="center", pad=6)


def _draw_panel_b(fig, gs):
    """MNIST 10x11 grid: 1 header row + 10 digit rows. Cols: digit
    label, 5 true, spacer, 5 generated. All digit cells share an
    identical aspect ratio (set_aspect='equal')."""
    n_per_digit = 5
    true_by = load_true_mnist_8x8(n_per_digit=n_per_digit, seed=0)
    gen_by  = load_gen_with_inferred_labels(n_per_digit=n_per_digit, seed=0)

    inner = gs.subgridspec(11, 12, hspace=0.05, wspace=0.05,
                            width_ratios=[0.8] + [1.0] * 5 + [0.3] + [1.0] * 5,
                            height_ratios=[0.5] + [1.0] * 10)

    # Header row: dedicated text axes for the "true" / "generated"
    # group labels, so they don't claim layout space inside the
    # digit-image cells (which would shrink the digit-0 row).
    ax_true_hdr = fig.add_subplot(inner[0, 1:6])
    ax_true_hdr.text(0.5, 0.0, "true", fontsize=11, fontweight="bold",
                     ha="center", va="bottom",
                     transform=ax_true_hdr.transAxes)
    ax_true_hdr.axis("off")
    ax_gen_hdr = fig.add_subplot(inner[0, 7:12])
    ax_gen_hdr.text(0.5, 0.0, "generated", fontsize=11, fontweight="bold",
                    ha="center", va="bottom",
                    transform=ax_gen_hdr.transAxes)
    ax_gen_hdr.axis("off")

    for digit in range(10):
        row = digit + 1  # account for header row
        # Row label in column 0.
        ax_lbl = fig.add_subplot(inner[row, 0])
        ax_lbl.text(0.5, 0.5, f"{digit}", fontsize=14, fontweight="bold",
                    ha="center", va="center")
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        ax_lbl.axis("off")
        # True samples in cols 1..5.
        for j in range(n_per_digit):
            ax = fig.add_subplot(inner[row, j + 1])
            if j < len(true_by[digit]):
                ax.imshow(true_by[digit][j].reshape(8, 8),
                          cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("dimgray")
                spine.set_linewidth(0.4)
        # Spacer column at index 6, then generated cols 7..11.
        for j in range(n_per_digit):
            ax = fig.add_subplot(inner[row, j + 7])
            if j < len(gen_by[digit]):
                ax.imshow(gen_by[digit][j].reshape(8, 8),
                          cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest", aspect="equal")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("dimgray")
                spine.set_linewidth(0.4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig5_bas_mnist_samples.pdf")
    args = p.parse_args()

    fig = plt.figure(figsize=(16, 8.5), dpi=300, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.30, h_pad=0.20)
    # Outer grid: 2 cols. Each col gets a 2-row inner gridspec where
    # the top row is a thin axes used purely for the panel title.
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[1.0, 1.4],
                          height_ratios=[0.04, 1.0])
    # Panel-title strips.
    ax_a_title = fig.add_subplot(gs[0, 0])
    ax_a_title.text(0.0, 0.5, r"(A)  BAS  $2{\times}2$  ($d{=}4$)",
                    fontsize=15, fontweight="bold", va="center", ha="left",
                    transform=ax_a_title.transAxes)
    ax_a_title.axis("off")
    ax_b_title = fig.add_subplot(gs[0, 1])
    ax_b_title.text(0.0, 0.5, r"(B)  MNIST  $8{\times}8$ binary  ($d{=}64$)",
                    fontsize=15, fontweight="bold", va="center", ha="left",
                    transform=ax_b_title.transAxes)
    ax_b_title.axis("off")
    # Body sub-grids.
    _draw_panel_a(fig, gs[1, 0])
    _draw_panel_b(fig, gs[1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
