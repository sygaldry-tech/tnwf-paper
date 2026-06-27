"""Visualise the reach ablation — does each variant push particles to the petal tips?

Five AM-Fig-3-style panels side-by-side, one per JAM-training variant.
Each integrates 80 particles from q_{t=0} for 100 Euler sub-steps under
`ẋ = ∇V_t(x)` and plots the smooth trajectories + coloured snapshot dots.

Axes are SHARED across panels so over/under-reach is visually obvious.
The panel range is widened to include the global-OT overshoot
(r ≲ 1.4 in centred frame) rather than tight-cropping to the petal hull.
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
    ("baseline",     "data/petals_2d/jam_paper_cfg/seed0.pt",
     "Baseline\n(minibatch OT, ε=0.05)"),
    ("sharp",        "data/petals_2d/jam_abl_sharpot/seed0.pt",
     "+ sharp ε=0.005\n(minibatch OT)"),
    ("global",       "data/petals_2d/jam_abl_globot_paper/seed0.pt",
     "+ global OT\n(ε=0.05) — overshoots"),
    ("soft_global",  "data/petals_2d/jam_abl_softglobot/seed0.pt",
     "+ soft global OT\n(ε=0.5)"),
    ("biglong",      "data/petals_2d/jam_abl_biglong/seed0.pt",
     "+ bigger MLP + more iters\n(7×384, 100k)"),
]


def _integrate(ckpt: str, L: float, K: int, n_particles: int,
               n_steps_per_segment: int, q0_centred: np.ndarray, seed: int):
    model, _ = load_jam(ckpt, device="cpu")
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, q0_centred.shape[0], size=n_particles)
    z0 = q0_centred[idx].astype(np.float32)
    n_steps = K * n_steps_per_segment
    snaps = sample_gradient_flow(model, torch.from_numpy(z0),
                                 n_steps=n_steps, save_every=1, L=L)
    smooth = snaps.astype(np.float32) + L / 2.0                # world frame
    return smooth, smooth[::n_steps_per_segment]


def _panel(ax, smooth, snapshots, bg, label, K, axis_box):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.4, c="#5b8a6c", alpha=0.5,
               linewidths=0, rasterized=True)
    for i in range(smooth.shape[1]):
        ax.plot(smooth[:, i, 0], smooth[:, i, 1], color="#f5b800",
                alpha=0.45, lw=0.55, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=10,
                   c=[cmap(k / K)], alpha=0.85, edgecolors="none",
                   rasterized=True)
    ax.set_title(label, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_particles", type=int, default=80)
    p.add_argument("--steps_per_segment", type=int, default=25)
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--out_stem", default="petals_reach_ablations")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)
    q0_centred = target_traj[0]
    bg = (target_traj.reshape(-1, 2) + L / 2.0)
    rng = np.random.default_rng(0)
    bg = bg[rng.choice(bg.shape[0], size=min(3000, bg.shape[0]), replace=False)]

    # Widen axis box to include global-OT overshoots (which reach the
    # box corner at distance √2·L/2 ≈ 2.83 in centred frame).
    pad = 0.4
    cx = L / 2.0
    half_extent = 1.6                                          # ±1.6 in centred frame
    axis_box = (cx - half_extent, cx + half_extent,
                cx - half_extent, cx + half_extent)

    n = len(VARIANTS)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n + 0.6, 3.1))
    for ax, (_, ckpt, label) in zip(axes, VARIANTS):
        if not Path(ckpt).exists():
            ax.text(0.5, 0.5, f"missing\n{Path(ckpt).parent.name}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(axis_box[0], axis_box[1])
            ax.set_ylim(axis_box[2], axis_box[3])
            continue
        smooth, snapshots = _integrate(
            ckpt, L=L, K=K,
            n_particles=args.n_particles,
            n_steps_per_segment=args.steps_per_segment,
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
               bbox_to_anchor=(0.997, 0.5), fontsize=9, labelspacing=0.5)

    fig.suptitle(
        f"Petals reach ablations — JAM grad-flow from q$_{{t=0}}$, "
        f"5×256 MLP, 50k iters "
        f"(q$_0$ init, {args.steps_per_segment * K} Euler sub-steps)",
        fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.93, 1.0))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
