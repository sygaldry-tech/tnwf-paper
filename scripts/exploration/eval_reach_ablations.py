"""Evaluate the three reach-ablation JAM checkpoints — does any of them
push particles further outward into the petal arms?

For each variant + the paper-cfg baseline:
  - integrate 2000 particles from q_{t=0} for K_data=4 segments × 25
    Euler sub-steps = 100 total steps
  - report r_mean, r_p95 at the K+1 reference times vs the bio data
  - report W_2 at each reference time

Prints a side-by-side comparison table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS, load_jam, sample_gradient_flow
from tnwf.metrics import wasserstein_2_subsampled

L = DATASET_DEFAULTS["petals_2d"]["L"]
K = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])
target = sample_petals_trajectory(n_per_step=2000, seed=0)
target_world = target + L / 2.0

print("Bio data radius progression (the goal):")
for k in range(K + 1):
    pts = target[k]
    r = np.linalg.norm(pts, axis=-1)
    print(f"  t={k}/{K}: r_mean={r.mean():.3f}  r_p95={np.percentile(r,95):.3f}")
print()

variants = [
    ("paper_cfg (baseline)",     "data/petals_2d/jam_paper_cfg/seed0.pt"),
    ("+ sharp OT (ε=0.005)",     "data/petals_2d/jam_abl_sharpot/seed0.pt"),
    ("+ global OT",              "data/petals_2d/jam_abl_globot_paper/seed0.pt"),
    ("+ bigger MLP + more iters","data/petals_2d/jam_abl_biglong/seed0.pt"),
]

for label, ckpt in variants:
    if not Path(ckpt).exists():
        print(f"--- {label}: ckpt MISSING ({ckpt}) ---\n")
        continue
    model, _ = load_jam(ckpt, device="cpu")
    rng = np.random.default_rng(1)
    z0 = target[0][rng.integers(0, target.shape[1], size=2000)].astype(np.float32)
    snaps = sample_gradient_flow(model, torch.from_numpy(z0),
                                  n_steps=100, save_every=1, L=L)
    print(f"--- {label} ---")
    for k in range(K + 1):
        pts = snaps[k * 25]
        r = np.linalg.norm(pts, axis=-1)
        w2_m, _ = wasserstein_2_subsampled(
            pts.astype(np.float64) + L / 2.0,
            target_world[k], n_sub=1000, n_repeats=3, seed=0,
        )
        data_r_mean = np.linalg.norm(target[k], axis=-1).mean()
        reach = r.mean() / data_r_mean if data_r_mean > 0 else 1.0
        print(f"  t={k}/{K}: r_mean={r.mean():.3f}  r_p95={np.percentile(r,95):.3f}  "
              f"reach={reach*100:.0f}%  W2={w2_m:.3f}")
    print()
