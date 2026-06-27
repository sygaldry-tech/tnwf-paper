"""Figure 1: 2D demo. Dense |ψ|² snapshots + Dense sample scatters over time
+ SW / MMD / NLL trajectories with shaded ±1σ over seeds (all 7 methods).

Reads results/{dataset}/{method}/seed*.npz produced by tnwf.pipelines.run_evolution.
Per-step ψ snapshots are stored under "psi_per_step" (Dense only) and per-step
samples under "samples_per_step".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

METHODS = [
    "jam", "dense",     "tci_tdvp1", "tci_tdvp2",
]
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}


def load_seeds(results_dir: Path, method: str) -> list:
    return [np.load(f, allow_pickle=True) for f in sorted((results_dir / method).glob("seed*.npz"))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/swiss_roll_2d")
    p.add_argument("--dataset_label", default="swiss_roll_2d")
    p.add_argument("--out", default="results/swiss_roll_2d/fig1.pdf")
    p.add_argument("--n_time_panels", type=int, default=5,
                   help="Number of time snapshots to display in rows 1–2.")
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    dense_runs = load_seeds(res_dir, "dense")
    if not dense_runs:
        raise SystemExit(f"No dense runs at {res_dir}/dense/seed*.npz")

    # Use first seed for snapshot rows
    dense = dense_runs[0]
    N = int(dense["N"]); d = int(dense["d"]); K = int(dense["K"]); L = float(dense["L"])
    has_psi = "psi_per_step" in dense.files
    has_samples_steps = "samples_per_step" in dense.files

    # Sub-sample time slices uniformly across the K+1 snapshots
    n_steps = K + 1
    t_idx = np.linspace(0, n_steps - 1, args.n_time_panels).astype(int)

    fig = plt.figure(figsize=(4.5 * args.n_time_panels, 18), dpi=200)
    gs = GridSpec(4, args.n_time_panels, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1, 1, 1.2, 0.05])

    # Row 1: Dense |ψ|² heatmaps over time
    if has_psi and d == 2:
        psi_steps = dense["psi_per_step"]                          # (n_steps, N^d) cmplx
        rho_steps = (np.abs(psi_steps) ** 2).reshape(n_steps, N, N)
        rho_max = rho_steps.max()
        for i, ti in enumerate(t_idx):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(rho_steps[ti].T, origin="lower", extent=(0, L, 0, L),
                           cmap="plasma", vmin=0, vmax=rho_max)
            ax.set_title(f"|ψ|²  t={ti / K:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel("Dense\n|ψ|² heatmap")
    else:
        for i in range(args.n_time_panels):
            ax = fig.add_subplot(gs[0, i])
            ax.text(0.5, 0.5, "(no |ψ|² stored)", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])

    # Row 2: Dense sample scatters over time
    if has_samples_steps and d == 2:
        samples_steps = dense["samples_per_step"]                  # (n_steps, n_samples, d)
        target = dense["target"]
        for i, ti in enumerate(t_idx):
            ax = fig.add_subplot(gs[1, i])
            ax.scatter(target[:, 0], target[:, 1], s=2, alpha=0.15, c="gray", label="target")
            ax.scatter(samples_steps[ti, :, 0], samples_steps[ti, :, 1],
                       s=2, alpha=0.5, c="black")
            ax.set_xlim(0, L); ax.set_ylim(0, L)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"samples  t={ti / K:.2f}")
            if i == 0:
                ax.set_ylabel("Dense\nsamples")

    # Row 3: SW / MMD vs t with shaded ±1σ over seeds — all methods
    half = max(1, args.n_time_panels // 2)
    metric_axes = [
        fig.add_subplot(gs[2, 0:half]),
        fig.add_subplot(gs[2, half:]),
    ]
    metrics = ["sw", "mmd"]
    for ax, metric in zip(metric_axes, metrics):
        for method in METHODS:
            runs = load_seeds(res_dir, method)
            if not runs:
                continue
            arr = np.stack([r[metric] for r in runs], axis=0)
            t_axis = np.linspace(0, 1, arr.shape[1])
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            ax.plot(t_axis, mean, label=method, color=METHOD_COLORS[method], lw=1.5)
            ax.fill_between(t_axis, mean - std, mean + std, alpha=0.2,
                             color=METHOD_COLORS[method])
        ax.set_xlabel("t")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs t  ({args.dataset_label})")

    # Legend in row 4
    lax = fig.add_subplot(gs[3, :])
    lax.axis("off")
    handles = [plt.Line2D([0], [0], color=METHOD_COLORS[m], lw=2, label=m) for m in METHODS]
    lax.legend(handles=handles, ncol=len(METHODS), loc="center", frameon=False)

    fig.suptitle(f"Figure 1 — {args.dataset_label}", y=0.995)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
