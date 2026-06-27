"""Diagnose the petal-whitespace artefact.

Three rows × (K+1) columns:
  row 1: target density q_{t_k} on the same N grid (histogram of bio data)
  row 2: Dense-evolution |ψ_{t_k}(V_t)|² with the Trotter-loss V_t
  row 3: 200 samples drawn from |ψ_{t_k}|² (what the figure shows)

The point: if rows 1/2 look identical on the arm-quadrants but row 3
shows "whitespace violation", the artefact is sampling (intra-cell
uniform draw at dx=L/N). If row 2 has mass in the gap quadrants
already, the artefact is wavefunction spreading / Trotter-loss
optimisation residual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS
from tnwf.pipelines.run_evolution import run


def _bin_density(samples_centred, N, L, d=2):
    edges = [np.linspace(-L / 2, L / 2, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    return counts / counts.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--trotter_ckpt",
                   default="data/petals_2d/jam_trotter_loss/seed0.pt")
    p.add_argument("--out_stem", default="petals_whitespace_diagnose")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    L = DATASET_DEFAULTS["petals_2d"]["L"]
    K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])

    # Target density per snapshot
    traj = sample_petals_trajectory(n_per_step=3000, seed=args.seed)
    q_targets = [_bin_density(traj[k], N=args.N, L=L) for k in range(K + 1)]

    # Run Dense pipeline; the pipeline already stores per-step samples,
    # but we want the dense PROBABILITY ARRAY too. Recompute |ψ|² from
    # the dense state at each step by calling run() with save=False.
    # The run() loop computes samples but discards psi; easiest path is
    # to re-run the Trotter steps externally. Cheap alternative: bin the
    # returned samples_per_step at high count to approximate |ψ|².
    out = run(
        method="dense", dataset="petals_2d", jam_ckpt=str(args.trotter_ckpt),
        seed=args.seed, N=args.N, K=K, n_samples=5000, save=False,
    )
    samples_per_step = out["samples_per_step"]                # (K+1, ≤n, d)
    # Convert from world frame back to centred frame for density binning
    samples_centred = samples_per_step - L / 2.0
    # 200 of these for the sample row
    samples_200 = samples_centred[:, :200]
    # Approximate |ψ|² by binning all 5000 samples
    psi2_approx = [_bin_density(samples_centred[k], N=args.N, L=L)
                   for k in range(K + 1)]

    half = 1.25
    extent = (-half, half, -half, half)
    # Common log-scale colour limits
    vmin = 1e-5
    vmax = max(np.max(q_targets), np.max(psi2_approx))

    fig, axes = plt.subplots(3, K + 1, figsize=(3 * (K + 1), 9))
    for k in range(K + 1):
        # crop to ±half × ±half region centred on origin
        cmin = int((L / 2 - half) / (L / args.N))
        cmax = args.N - cmin

        axes[0, k].imshow(q_targets[k][cmin:cmax, cmin:cmax].T,
                          origin="lower", extent=extent,
                          cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, k].set_title(f"target  t={k / K:.2f}", fontsize=10)

        axes[1, k].imshow(psi2_approx[k][cmin:cmax, cmin:cmax].T,
                          origin="lower", extent=extent,
                          cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, k].set_title(f"|ψ|² (Dense)  t={k / K:.2f}", fontsize=10)

        axes[2, k].scatter(samples_200[k, :, 0], samples_200[k, :, 1],
                           s=6, alpha=0.7, c="#3366aa")
        axes[2, k].set_xlim(-half, half); axes[2, k].set_ylim(-half, half)
        axes[2, k].set_aspect("equal", adjustable="box")
        axes[2, k].set_title(f"200 samples  t={k / K:.2f}", fontsize=10)

        for r in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])

    axes[0, 0].set_ylabel("target density\n(bio data histogram)", fontsize=10)
    axes[1, 0].set_ylabel("|ψ|² density\n(5000-sample bin)", fontsize=10)
    axes[2, 0].set_ylabel("200 samples", fontsize=10)

    fig.suptitle(f"Petal whitespace diagnostic — N={args.N}  L={L}  "
                 f"dx={L / args.N:.3f}  (arm std ≈ 0.03)", fontsize=12)
    fig.tight_layout()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{args.out_stem}.pdf"
    png = out_dir / f"{args.out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
