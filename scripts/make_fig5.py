"""Figure 5 of the paper: BAS 2x2 + 8x8 binary MNIST samples.

Two-panel composite:
  (A) BAS 2x2: top-row pmf bars from the research prototype's results/bas/eval/bas2x2_viz.png
      (the source figure has 6 sub-panels in 2 rows × 3 cols; we crop the
      top row only — pmf comparisons of dense, var_cross, ideal).
  (B) 8x8 binary MNIST samples: a vertically cropped subset of
      the research prototype's results/mnist/coarse/mnist_8x8_digit_grid.png so a few
      digit rows are visible.

Usage:
    uv run python scripts/make_fig5.py \\
        --out figures/fig5_bas_mnist_samples.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BAS_SRC   = Path("inputs/accinf_wf/results/bas/eval/bas2x2_viz.png")
MNIST_SRC = Path("inputs/accinf_wf/results/mnist/coarse/mnist_8x8_digit_grid.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/fig5_bas_mnist_samples.pdf")
    args = p.parse_args()

    # Load source images.
    bas_img = np.asarray(Image.open(BAS_SRC).convert("RGB"))
    mnist_img = np.asarray(Image.open(MNIST_SRC).convert("RGB"))

    # Crop BAS: top half (pmf bars, sub-panels A1-A3); bottom half is
    # sample grids and per-pattern mass — keep the pmf bars (most legible
    # quantitative summary) and the mass-per-pattern panel at right.
    h_bas = bas_img.shape[0]
    bas_top = bas_img[: int(0.55 * h_bas), :, :]

    # Crop MNIST: first ~5 digits (rows 0-4), each spans 2 rows of the grid
    # (gen + true). Source image has 10 digits × 2 rows = 20 sub-rows total.
    h_mnist = mnist_img.shape[0]
    mnist_crop = mnist_img[: int(0.55 * h_mnist), :, :]

    fig = plt.figure(figsize=(14, 9), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_bas = fig.add_subplot(gs[0, 0])
    ax_mnist = fig.add_subplot(gs[0, 1])

    ax_bas.imshow(bas_top)
    ax_bas.set_xticks([]); ax_bas.set_yticks([])
    ax_bas.set_title("(A)  BAS  $2{\\times}2$  (d=4)",
                     fontsize=14, fontweight="bold", loc="left")

    ax_mnist.imshow(mnist_crop)
    ax_mnist.set_xticks([]); ax_mnist.set_yticks([])
    ax_mnist.set_title("(B)  MNIST  $8{\\times}8$ binary  (d=64)",
                       fontsize=14, fontweight="bold", loc="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
