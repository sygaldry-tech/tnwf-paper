"""Run all V-step methods × N seeds on swiss_roll_2d.

Assumes per-seed JAM checkpoints already exist at
data/swiss_roll_2d/jam/seed{S}.pt (run scripts/swiss_roll_2d/train_jam.py first).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tnwf.leaderboard import update_after_batch_subprocess as update_after_batch
from tnwf.pipelines.run_evolution import run

METHODS = [
    "jam", "dense",
    "tci_tdvp1", "tci_tdvp2",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                   help="Comma-separated seed list.")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--ckpt_dir", default="data/swiss_roll_2d/jam")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    for seed in seeds:
        ckpt = Path(args.ckpt_dir) / f"seed{seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no JAM checkpoint at {ckpt}")
            continue
        for method in methods:
            out_file = Path(f"data/swiss_roll_2d/{method}/seed{seed}.npz")
            if out_file.exists() and not args.force:
                print(f"[skip] {out_file}")
                continue
            print(f"--> swiss_roll_2d / {method} / seed={seed}")
            method_kwargs = {"D_max": args.D_max, "D_V": args.D_V, "D_out": args.D_max}
            run(
                method=method, dataset="swiss_roll_2d", jam_ckpt=str(ckpt),
                seed=seed, N=args.N, K=args.K,
                n_samples=args.n_samples, method_kwargs=method_kwargs,
                snapshot_psi=(method == "dense"),
            )
            update_after_batch()


if __name__ == "__main__":
    main()
