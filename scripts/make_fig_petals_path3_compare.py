"""Comparison figure: AM gradient flow vs Trotter-loss wavefunction flow.

Two panels:
  LEFT  : JAM gradient-flow under AM-trained V_t — particles integrated
          classically via 100 Euler steps from q_{t=0}. Trajectories
          visible as yellow lines.
  RIGHT : Wavefunction-flow output (TCI + TDVP-2) under Trotter-loss
          V_t — samples drawn from |ψ_{t_k}|² at each of the K+1
          reference times. No trajectories (independent samples per
          snapshot under autoregressive MPS sampling).

Both panels coloured by snapshot time (viridis). Bio data target in
faint green underneath. Same axis crop and styling as Fig 3.

This is the *structurally correct* comparison: each V_t paired with
its semantically-matching downstream pipeline.
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
from tnwf.pipelines.run_evolution import run


def _integrate_jam(ckpt_path, L, K, n_particles, steps_per_segment,
                   q0_centred, seed):
    model, _ = load_jam(str(ckpt_path), device="cpu")
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, q0_centred.shape[0], size=n_particles)
    z0 = q0_centred[idx].astype(np.float32)
    snaps_c = sample_gradient_flow(
        model, torch.from_numpy(z0), n_steps=K * steps_per_segment,
        save_every=1, L=L,
    )
    smooth = snaps_c.astype(np.float32) + L / 2.0
    return smooth, smooth[::steps_per_segment]


def _wf_pipeline_samples(ckpt_path, dataset, N, K, n_samples, seed):
    """Run TCI+TDVP-2 wavefunction pipeline; return samples_per_step in world frame."""
    out = run(
        method="tci_tdvp2", dataset=dataset, jam_ckpt=str(ckpt_path),
        seed=seed, N=N, K=K, n_samples=n_samples, save=False,
        method_kwargs={"D_max": 16, "D_V": 16, "D_out": 16},
    )
    return out["samples_per_step"]                             # (K+1, ≤200, d)


def _panel_with_trajectories(ax, smooth, snapshots, bg, title, K, axis_box):
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
    _finish_panel(ax, title, axis_box)


def _panel_scatter_only(ax, snapshots, bg, title, K, axis_box):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
                linewidths=0, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=15,
                    c=[cmap(k / K)], alpha=0.92, edgecolors="white",
                    linewidths=0.3, rasterized=True)
    _finish_panel(ax, title, axis_box)


def _finish_panel(ax, title, axis_box):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--n_particles", type=int, default=120)
    p.add_argument("--steps_per_segment", type=int, default=250)
    p.add_argument("--am_ckpt", default="data/petals_2d/jam_am_softglobot/seed0.pt")
    p.add_argument("--trotter_ckpt", default="data/petals_2d/jam_trotter_loss/seed0.pt")
    p.add_argument("--out_stem", default="petals_path3_compare")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)
    q0_centred = target_traj[0]
    bg = (target_traj.reshape(-1, 2) + L / 2.0)
    rng = np.random.default_rng(0)
    bg = bg[rng.choice(bg.shape[0], size=min(2500, bg.shape[0]), replace=False)]

    half = 1.25
    cx = L / 2.0
    axis_box = (cx - half, cx + half, cx - half, cx + half)

    fig = plt.figure(figsize=(12.0, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.32], wspace=0.07)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2]); ax_legend.axis("off")

    # LEFT: AM V_t + JAM gradient flow
    smooth, snapshots = _integrate_jam(
        Path(args.am_ckpt), L=L, K=K,
        n_particles=args.n_particles,
        steps_per_segment=args.steps_per_segment,
        q0_centred=q0_centred, seed=args.seed,
    )
    _panel_with_trajectories(
        ax_l, smooth, snapshots, bg,
        "AM V$_t$ + JAM grad-flow\n(classical gradient flow)", K, axis_box)

    # RIGHT: Trotter-loss V_t + wavefunction pipeline (TDVP-2 samples)
    snapshots_wf = _wf_pipeline_samples(
        Path(args.trotter_ckpt), dataset="petals_2d",
        N=args.N, K=K, n_samples=200, seed=args.seed,
    )
    _panel_scatter_only(
        ax_r, snapshots_wf, bg,
        "Trotter-loss V$_t$ + WF pipeline\n(TDVP-2 samples from |ψ$_t$|²)",
        K, axis_box)

    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                        markerfacecolor=cmap(i / K), markeredgecolor="white",
                        markersize=8, markeredgewidth=0.3,
                        label=f"$t = {i / K:.2f}$")
                for i in range(K + 1)]
    handles.append(Line2D([], [], color="#f5b800", lw=1.6,
                          label="JAM trajectories (left)"))
    handles.append(Line2D([], [], marker="o", linestyle="",
                            markerfacecolor="#5b8a6c", markeredgecolor="none",
                            markersize=5, alpha=0.7, label="bio data target"))
    ax_legend.legend(handles=handles, loc="center left", frameon=False,
                     bbox_to_anchor=(0.0, 0.5), fontsize=10.5,
                     labelspacing=0.7, handletextpad=0.5)

    fig.suptitle(
        "Petals — each V_t paired with its semantically-matched pipeline  "
        "(N=16, K=4, q$_{t=0}$ init, 3-seed mean W₂@t=1: AM-grad 0.33, "
        "Trotter-loss-WF 0.19)",
        fontsize=11.5, y=1.00)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
