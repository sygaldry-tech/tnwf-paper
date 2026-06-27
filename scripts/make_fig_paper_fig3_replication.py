"""Closest possible replication of Neklyudov 2022 Fig 3 (petals trajectories).

Two large square panels side-by-side, matching the paper's visual idiom:
  - LEFT  : "MIOFlow-equivalent"  → our CFM-grad baseline (paper-cfg JAM)
  - RIGHT : "entropic AM-equivalent" → our AM-loss + Sinkhorn-OT JAM

Each panel:
  - 80 particles initialised from q_{t=0} (innermost petals ring)
  - integrated under ẋ = ∇V_t(x) with 1000 Euler sub-steps (fine grid;
    avoids the under-integration artifact diagnosed in
    commit a75b4a5)
  - smooth yellow trajectory lines (sampled every step)
  - viridis-coloured snapshot dots at the K+1 reference times
  - bio-data target as small green dots in background
  - tight axis crop (±1.6 in centred frame, padding for any outliers)

Reads paper-cfg JAM (data/petals_2d/jam_paper_cfg/seed0.pt) and the AM
checkpoint (data/petals_2d/jam_am_<tag>/seed0.pt where <tag> defaults
to ``paper`` = AM-loss + minibatch OT + 5×256 MLP + 50k iters).

Writes ``results/figures/paper_fig3_replication.{pdf,png}``.
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
    snapshots = smooth[::steps_per_segment]
    return smooth, snapshots


def _panel(ax, smooth, snapshots, bg, title, K, axis_box):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
                linewidths=0, rasterized=True)
    for i in range(smooth.shape[1]):
        ax.plot(smooth[:, i, 0], smooth[:, i, 1], color="#f5b800",
                alpha=0.50, lw=0.6, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=15,
                    c=[cmap(k / K)], alpha=0.92, edgecolors="white",
                    linewidths=0.3, rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_particles", type=int, default=80)
    p.add_argument("--steps_per_segment", type=int, default=250,
                    help="1000 total Euler steps by default — fine integration.")
    p.add_argument("--left_ckpt", default="data/petals_2d/jam_paper_cfg/seed0.pt")
    p.add_argument("--left_title", default="CFM (our flow-matching baseline)")
    p.add_argument("--right_ckpt", default="data/petals_2d/jam_am_paper/seed0.pt")
    p.add_argument("--right_title", default="Action Matching (entropic, ours)")
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--out_stem", default="paper_fig3_replication")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)
    q0_centred = target_traj[0]
    bg = (target_traj.reshape(-1, 2) + L / 2.0)
    rng = np.random.default_rng(0)
    bg = bg[rng.choice(bg.shape[0], size=min(2500, bg.shape[0]), replace=False)]

    # Tight crop matching the petal bounding box + small padding
    pad = 0.25
    cx = L / 2.0
    half = 1.25
    axis_box = (cx - half, cx + half, cx - half, cx + half)

    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.32], wspace=0.07)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2]); ax_legend.axis("off")

    for ax, ckpt, title in (
        (ax_l, args.left_ckpt, args.left_title),
        (ax_r, args.right_ckpt, args.right_title),
    ):
        if not Path(ckpt).exists():
            ax.text(0.5, 0.5, f"missing\n{Path(ckpt).name}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            continue
        smooth, snapshots = _integrate(ckpt, L=L, K=K,
                                        n_particles=args.n_particles,
                                        steps_per_segment=args.steps_per_segment,
                                        q0_centred=q0_centred, seed=args.seed)
        _panel(ax, smooth, snapshots, bg, title, K, axis_box)

    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                        markerfacecolor=cmap(i / K), markeredgecolor="white",
                        markersize=8, markeredgewidth=0.3,
                        label=f"$t = {i / K:.2f}$")
                for i in range(K + 1)]
    handles.append(Line2D([], [], color="#f5b800", lw=1.6, label="trajectories"))
    handles.append(Line2D([], [], marker="o", linestyle="",
                            markerfacecolor="#5b8a6c", markeredgecolor="none",
                            markersize=5, alpha=0.7, label="bio data target"))
    ax_legend.legend(handles=handles, loc="center left", frameon=False,
                     bbox_to_anchor=(0.0, 0.5), fontsize=10.5,
                     labelspacing=0.7, handletextpad=0.5)

    fig.suptitle(
        "Generated trajectories on petals — paper Fig 3 replication "
        f"({args.n_particles} particles from q$_{{t=0}}$, "
        f"{args.steps_per_segment * K} Euler steps, q$_0$ init)",
        fontsize=12, y=1.00)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
