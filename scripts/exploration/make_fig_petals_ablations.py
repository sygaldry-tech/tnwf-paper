"""Petals AM-Fig-3-style ablation grid (5 panels).

Renders the same conservative-gradient-flow trajectory plot for five
JAM-training variants:

  baseline    — CFM-grad + per-minibatch Sinkhorn OT (the published
                CFM+OT, hidden=128, n_layers=3, batch=256, n_iter=10k)
  +global_ot  — replaces minibatch OT with one precomputed full-dataset
                Sinkhorn permutation per segment
  +big_model  — hidden=256, n_layers=5 (otherwise baseline)
  +more_iters — n_iter = 50k (otherwise baseline)
  +big_batch  — batch_size = 1024 (otherwise baseline)

Each panel integrates 80 particles from q_{t=0} (the petals' innermost
ring) forward under ẋ = ∇V_t for 4 × 25 = 100 Euler steps and shows the
resulting smooth yellow threads, with samples coloured by snapshot time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS, load_jam, sample_gradient_flow


VARIANTS = [
    ("baseline",    "data/petals_2d/jam_cfm_ot/seed0.pt",      "Baseline\n(CFM + minibatch OT)"),
    ("global_ot",   "data/petals_2d/jam_abl_globot/seed0.pt",  "+ global OT\n(full-dataset Sinkhorn)"),
    ("big_model",   "data/petals_2d/jam_abl_bigmodel/seed0.pt","+ bigger MLP\n(h=256, L=5)"),
    ("more_iters",  "data/petals_2d/jam_abl_moreiter/seed0.pt","+ more iters\n(50k vs 10k)"),
    ("big_batch",   "data/petals_2d/jam_abl_bigbatch/seed0.pt","+ bigger batch\n(1024 vs 256)"),
]


def _integrate(ckpt: str, L: float, K: int, n_particles: int,
               steps_per_segment: int, q0_centred: np.ndarray, seed: int):
    model, _ = load_jam(ckpt, device="cpu")
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, q0_centred.shape[0], size=n_particles)
    z0 = q0_centred[idx].astype(np.float32)
    n_steps = K * steps_per_segment
    snaps = sample_gradient_flow(model, torch.from_numpy(z0),
                                 n_steps=n_steps, save_every=1, L=L)
    smooth = snaps.astype(np.float32) + L / 2.0
    return smooth, smooth[::steps_per_segment]


def _panel(ax, smooth, snapshots, bg, label, K, axis_box):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.4, c="#5b8a6c", alpha=0.45,
               linewidths=0, rasterized=True)
    for i in range(smooth.shape[1]):
        ax.plot(smooth[:, i, 0], smooth[:, i, 1], color="#f5b800",
                alpha=0.45, lw=0.55, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=10,
                   c=[cmap(k / K)], alpha=0.85, edgecolors="none",
                   rasterized=True)
    ax.set_title(label, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_particles", type=int, default=80)
    p.add_argument("--steps_per_segment", type=int, default=25)
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)
    q0_centred = target_traj[0]                                 # already centred
    bg = (target_traj.reshape(-1, 2) + L / 2.0)                 # world frame for plot
    rng = np.random.default_rng(0)
    idx = rng.choice(bg.shape[0], size=min(3000, bg.shape[0]), replace=False)
    bg = bg[idx]
    pad = 0.2
    axis_box = (bg[:, 0].min() - pad, bg[:, 0].max() + pad,
                bg[:, 1].min() - pad, bg[:, 1].max() + pad)

    n = len(VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.7, 3.1))
    last_smooth = None
    for ax, (key, ckpt, label) in zip(axes, VARIANTS):
        if not Path(ckpt).exists():
            ax.text(0.5, 0.5, f"missing\n{key}", transform=ax.transAxes,
                    ha="center", va="center", color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        smooth, snapshots = _integrate(ckpt, L=L, K=K,
                                        n_particles=args.n_particles,
                                        steps_per_segment=args.steps_per_segment,
                                        q0_centred=q0_centred, seed=args.seed)
        _panel(ax, smooth, snapshots, bg, label, K, axis_box)

    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                       markerfacecolor=cmap(i / K), markeredgecolor="none",
                       markersize=7, label=f"$t = {i / K:.2f}$")
               for i in range(K + 1)]
    handles.append(Line2D([], [], color="#f5b800", lw=1.4, label="trajectories"))
    handles.append(Line2D([], [], marker="o", linestyle="",
                           markerfacecolor="#5b8a6c", markeredgecolor="none",
                           markersize=4, alpha=0.7, label="bio data target"))
    fig.legend(handles=handles, loc="center right", frameon=False,
               bbox_to_anchor=(0.998, 0.5), fontsize=9, labelspacing=0.5)

    fig.suptitle("Petals (2D) — JAM-training ablations vs baseline "
                 "(seed=0, q$_{t=0}$ init, 100 Euler steps)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.92, 1.0))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "petals_ablations.pdf"
    png = out_dir / "petals_ablations.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
