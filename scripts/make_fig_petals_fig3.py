"""Single-panel headline petals figure — closest analogue to AM Fig 3.

One large square panel showing the JAM-grad-flow generated trajectories on
petals_2d using a CFM-trained-with-Sinkhorn-OT-coupling checkpoint. This is
the cleanest output of our conservative-V pipeline:

  - V_t(x) is the learned scalar potential
  - ẋ = ∇V_t(x) is the conservative velocity field
  - Pair sampling during training uses Sinkhorn-OT coupling rather than
    random pairs (commit 3e0c656), which removes the
    interpolation-vs-q_t bias that the AM-paper-style vanilla AM loss
    suffers from

Reads ``data/petals_2d/jam_cfm_ot/seed{seed}.pt`` for the JAM model and
``data/petals_2d/jam/N{N}_K{K}/seed{seed}.npz`` for the bio-data target.
Writes ``results/figures/petals_fig3.{pdf,png}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from tnwf.jam.train import DATASET_DEFAULTS, load_jam, sample_gradient_flow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="data/petals_2d/jam_cfm_ot/seed0.pt")
    p.add_argument("--run", default="data/petals_2d/jam/N16_K4/seed0.npz",
                   help="Run npz to pull target_traj from (background).")
    p.add_argument("--title", default=("JAM grad-flow with Sinkhorn-OT pair "
                                       "coupling (CFM-trained, 10k iters)"))
    p.add_argument("--n_particles", type=int, default=120)
    p.add_argument("--steps_per_segment", type=int, default=30)
    p.add_argument("--n_data_bg", type=int, default=4500)
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--out_stem", default="petals_fig3")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K_data = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    if not Path(args.ckpt).exists():
        raise FileNotFoundError(args.ckpt)
    if not Path(args.run).exists():
        raise FileNotFoundError(args.run)

    model, _cfg = load_jam(args.ckpt, device="cpu")
    rng = np.random.default_rng(args.seed + 1)

    # Initial particles from q_{t=0} (the petals' innermost ring), matching
    # the Action Matching paper's Fig 3 setup. Sampling from N(0, I) instead
    # would start the cloud at a much wider spread than the actual q_0,
    # producing "exploded" trajectories. target_traj is in centered frame.
    target_traj = np.load(args.run)["target_traj"]                  # world frame
    q0_centred = target_traj[0] - L / 2.0
    idx0 = rng.integers(0, q0_centred.shape[0], size=args.n_particles)
    z0 = q0_centred[idx0].astype(np.float32)
    snaps_c = sample_gradient_flow(model, torch.from_numpy(z0),
                                    n_steps=K_data * args.steps_per_segment,
                                    save_every=1, L=L)
    smooth = snaps_c.astype(np.float32) + L / 2.0
    snapshots = smooth[::args.steps_per_segment]

    bg = target_traj.reshape(-1, 2)
    rng = np.random.default_rng(0)
    idx = rng.choice(bg.shape[0], size=min(args.n_data_bg, bg.shape[0]),
                     replace=False)
    bg = bg[idx]

    x0, x1 = bg[:, 0].min() - 0.2, bg[:, 0].max() + 0.2
    y0, y1 = bg[:, 1].min() - 0.2, bg[:, 1].max() + 0.2

    fig = plt.figure(figsize=(8.5, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.3], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[0, 1]); ax_legend.axis("off")

    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.55, c="#5b8a6c", alpha=0.55,
               linewidths=0, rasterized=True)
    for i in range(args.n_particles):
        ax.plot(smooth[:, i, 0], smooth[:, i, 1], color="#f5b800",
                alpha=0.45, lw=0.6, rasterized=True)
    for k in range(K_data + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=14,
                   c=[cmap(k / K_data)], alpha=0.9, edgecolors="none",
                   rasterized=True)

    ax.set_title(args.title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    handles = [
        Line2D([], [], marker="o", linestyle="",
               markerfacecolor=cmap(i / K_data), markeredgecolor="none",
               markersize=9, label=f"$t = {i / K_data:.2f}$")
        for i in range(K_data + 1)
    ]
    handles.append(Line2D([], [], color="#f5b800", lw=1.4, label="trajectories"))
    handles.append(Line2D([], [], marker="o", linestyle="",
                          markerfacecolor="#5b8a6c", markeredgecolor="none",
                          markersize=5, alpha=0.7, label="bio data target"))
    ax_legend.legend(handles=handles, loc="center left", frameon=False,
                     bbox_to_anchor=(0.0, 0.5), fontsize=11,
                     labelspacing=0.75, handletextpad=0.5)

    fig.suptitle(r"Petals (2D) — conservative gradient flow "
                 r"$\dot x = \nabla V_t(x)$",
                 fontsize=13, y=0.99)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
