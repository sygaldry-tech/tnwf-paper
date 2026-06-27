"""Petals — corrected AM-paper-Fig-2 analogue.

Faithful port of the visual idiom of Neklyudov et al. 2023 Fig 2:
SW (left) and MMD (right) vs time, with one line per *temporal
resolution of the dataset* — NOT per integration step count. The AM
paper generates three petals datasets with K_data + 1 ∈ {5, 10, 15}
timepoints, trains a separate model on each, and overlays the three
curves to test robustness to dataset granulation.

For each K_data ∈ {4, 9, 14}:
  - load the JAM checkpoint trained on petals_2d{,_K9,_K14} with the
    paper-matched config (MLP 5×256, CFM+Sinkhorn-OT, 50k iters)
  - sample n_particles from q_{t=0} of that dataset variant
  - Euler-integrate ẋ = ∇V_t(x) with fine n_steps and save every step
  - at each reference time t_k = k / K_data, compute SW + MMD vs the
    bio data at t_k

Reads ``data/petals_2d{,_K9,_K14}/jam_paper_cfg/seed{seed}.pt``.
Writes ``results/figures/petals_fig2.{pdf,png}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tnwf.jam.train import (DATASET_DEFAULTS, load_jam, sample_gradient_flow,
                             sample_target_trajectory)
from tnwf.metrics import mmd_rbf, wasserstein_2_subsampled


def _eval_one_variant(ckpt_path: Path, dataset_name: str, n_particles: int,
                      n_steps_per_segment: int, seed: int):
    """Returns (ref_t, sw_at_ref, mmd_at_ref) for one K_data variant."""
    cfg = DATASET_DEFAULTS[dataset_name]
    L = cfg["L"]
    K_data = int(cfg["trajectory_K"])

    target_traj = sample_target_trajectory(
        dataset_name, n_per_step=n_particles, d=cfg["d"], seed=seed)
    target_world = target_traj + L / 2.0
    q0_centred = target_traj[0]

    rng = np.random.default_rng(seed + 1)
    idx0 = rng.integers(0, q0_centred.shape[0], size=n_particles)
    z0 = q0_centred[idx0].astype(np.float32)

    model, _ = load_jam(str(ckpt_path), device="cpu")
    n_steps_total = K_data * n_steps_per_segment
    snaps_c = sample_gradient_flow(
        model, torch.from_numpy(z0), n_steps=n_steps_total,
        save_every=1, L=L,
    )                                                          # (n_steps_total+1, n, 2)
    snaps_w = snaps_c.astype(np.float64) + L / 2.0

    # Reference times t_k = k/K_data → integration step index k * n_steps_per_segment
    ref_t = np.linspace(0.0, 1.0, K_data + 1)
    w2s, mmds = [], []
    for k in range(K_data + 1):
        idx = k * n_steps_per_segment
        pred = snaps_w[idx]
        tgt = target_world[k]
        w2_mean, _ = wasserstein_2_subsampled(
            pred, tgt, n_sub=min(1000, pred.shape[0], tgt.shape[0]),
            n_repeats=3, seed=seed,
        )
        w2s.append(w2_mean)
        mmds.append(mmd_rbf(pred, tgt))
    return ref_t, np.array(w2s), np.array(mmds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_particles", type=int, default=2000)
    p.add_argument("--steps_per_segment", type=int, default=20)
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    variants = [
        ("petals_2d",     "K_data + 1 = 5 timepoints",  "#1f77b4", "o"),
        ("petals_2d_K9",  "K_data + 1 = 10 timepoints", "#2ca02c", "s"),
        ("petals_2d_K14", "K_data + 1 = 15 timepoints", "#d62728", "^"),
    ]

    fig, (ax_w2, ax_mmd) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, label, color, marker in variants:
        ckpt = Path(f"data/{name}/jam_paper_cfg/seed{args.seed}.pt")
        if not ckpt.exists():
            print(f"[skip] missing checkpoint at {ckpt}")
            continue
        ref_t, w2, mmd = _eval_one_variant(
            ckpt, name,
            n_particles=args.n_particles,
            n_steps_per_segment=args.steps_per_segment,
            seed=args.seed,
        )
        ax_w2.plot(ref_t, w2, marker=marker, color=color, lw=1.7, ms=5.5,
                   label=label)
        ax_mmd.plot(ref_t, mmd, marker=marker, color=color, lw=1.7, ms=5.5,
                    label=label)

    for ax, ylab, title in (
        (ax_w2, "$W_2$ distance (lower better)", "$W_2$ vs time"),
        (ax_mmd, "MMD (lower better)", "MMD vs time"),
    ):
        ax.set_xlabel("time $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=10, frameon=False)

    fig.suptitle(
        "Petals — JAM grad-flow (paper-matched MLP 5×256, CFM+Sinkhorn OT, 50k iters)\n"
        "Robustness to dataset temporal resolution (q$_{t=0}$ init, $W_2$ + MMD)",
        fontsize=11, y=1.02)
    fig.tight_layout()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "petals_fig2.pdf"
    png = out_dir / "petals_fig2.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
