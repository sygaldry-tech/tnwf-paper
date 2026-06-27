"""Visualise the quantum V-step methods under the Path-3 Trotter-loss V_t.

Panels (left → right, top → bottom):
  Dense, TCI+TDVP-1, TCI+TDVP-2.

Each panel scatters wavefunction-pipeline samples drawn from |ψ_{t_k}|²
at each of the K+1 reference times, coloured by snapshot time (viridis).
Bio data target in faint green underneath. Same axis crop / styling as
the Path-3 comparison figure.

A 6th panel hosts the legend and a methods table with W₂[t=1] readouts
filled in at run-time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.pipelines.run_evolution import run


METHODS = [
    ("dense",      "Dense (exact)",           {}),
    ("tci_tdvp1",  "TCI + TDVP-1",            {"D_V": 16}),
    ("tci_tdvp2",  "TCI + TDVP-2",            {"D_V": 16, "D_max": 16, "D_out": 16}),
]


def _run_method(method, method_kwargs, ckpt_path, dataset, N, K, n_samples, seed):
    out = run(
        method=method, dataset=dataset, jam_ckpt=str(ckpt_path),
        seed=seed, N=N, K=K, n_samples=n_samples, save=False,
        method_kwargs=method_kwargs,
    )
    return out["samples_per_step"]                          # (K+1, ≤n, d)


def _panel(ax, snapshots, bg, title, K, axis_box, w2_t1):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
               linewidths=0, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=14,
                   c=[cmap(k / K)], alpha=0.90, edgecolors="white",
                   linewidths=0.3, rasterized=True)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])
    ax.text(0.02, 0.97, f"W₂@t=1: {w2_t1:.3f}",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--trotter_ckpt",
                   default="data/petals_2d/jam_trotter_loss/seed0.pt")
    p.add_argument("--out_stem", default="petals_quantum_methods")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    # Bio data target + background scatter
    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)
    bg = (target_traj.reshape(-1, 2) + L / 2.0)
    rng = np.random.default_rng(0)
    bg = bg[rng.choice(bg.shape[0], size=min(2500, bg.shape[0]), replace=False)]

    half = 1.25
    cx = L / 2.0
    axis_box = (cx - half, cx + half, cx - half, cx + half)

    # Run every method, collect samples + W₂@t=1
    panels = []
    target_t1_world = target_traj[K] + L / 2.0
    for method, label, kw in METHODS:
        print(f"[run {method}] kwargs={kw}")
        snaps = _run_method(
            method, kw, ckpt_path=Path(args.trotter_ckpt),
            dataset="petals_2d", N=args.N, K=K,
            n_samples=args.n_samples, seed=args.seed,
        )
        w2_mean, _ = wasserstein_2_subsampled(
            snaps[K].astype(np.float64),
            target_t1_world.astype(np.float64),
            n_sub=200, n_repeats=3, seed=args.seed,
        )
        panels.append((label, snaps, float(w2_mean)))
        print(f"  W₂@t=1 = {w2_mean:.3f}")

    # Figure: 2x2 grid, last cell is legend
    n_methods = len(panels)
    fig = plt.figure(figsize=(10.0, 10.0))
    gs = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.18)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_methods)]
    ax_legend = fig.add_subplot(gs[1, 1]); ax_legend.axis("off")

    for ax, (label, snaps, w2) in zip(axes, panels):
        _panel(ax, snaps, bg, label, K, axis_box, w2)

    # Legend / methods key
    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                      markerfacecolor=cmap(i / K), markeredgecolor="white",
                      markersize=8, markeredgewidth=0.3,
                      label=f"$t = {i / K:.2f}$")
               for i in range(K + 1)]
    handles.append(Line2D([], [], marker="o", linestyle="",
                          markerfacecolor="#5b8a6c", markeredgecolor="none",
                          markersize=5, alpha=0.7, label="bio data target"))
    leg = ax_legend.legend(handles=handles, loc="upper left", frameon=False,
                           bbox_to_anchor=(0.0, 1.0), fontsize=11,
                           labelspacing=0.7, handletextpad=0.5,
                           title="snapshot time",
                           title_fontsize=11.5)
    leg.get_title().set_fontweight("bold")

    # Method/W₂ summary table below the legend
    lines = ["", "W₂@t=1 by method:"]
    for label, _snaps, w2 in panels:
        lines.append(f"  {label:<18s} {w2:.3f}")
    ax_legend.text(0.0, 0.45, "\n".join(lines),
                   transform=ax_legend.transAxes, fontsize=10.5,
                   va="top", family="monospace")

    fig.suptitle(
        "Petals — quantum V-step methods, Trotter-loss V$_t$, N=16 K=4 "
        "(seed 0, 200 samples / snapshot)",
        fontsize=12.5, y=0.995)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
