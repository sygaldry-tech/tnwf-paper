"""Compression sweep on swiss_roll_2d: vary D_max ∈ {2,4,8,16} to differentiate methods.

Writes results to results/swiss_roll_2d/{method}/D{D_max}/seed{S}.npz so the
leaderboard groups them as separate config rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tnwf.leaderboard import update_after_batch
from tnwf.pipelines.run_evolution import run

METHODS = [
    "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--D_list", type=str, default="2,4,8,16")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--ckpt_dir", default="data/swiss_roll_2d/jam")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ds = [int(d) for d in args.D_list.split(",") if d]

    for seed in seeds:
        ckpt = Path(args.ckpt_dir) / f"seed{seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no JAM checkpoint at {ckpt}")
            continue
        for D_max in Ds:
            for method in methods:
                if method == "dense" and D_max != Ds[-1]:
                    # Dense has no D_max; only run once at the largest D
                    continue
                tag = f"D{D_max}"
                print(f"--> swiss_roll_2d / {method} / D={D_max} / seed={seed}")
                method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                                 "D_init": D_max}
                out_dir = Path(f"data/swiss_roll_2d/{method}/{tag}")
                run(
                    method=method, dataset="swiss_roll_2d", jam_ckpt=str(ckpt),
                    seed=seed, N=args.N, K=args.K,
                    n_samples=args.n_samples, method_kwargs=method_kwargs,
                    out_dir=out_dir,
                )
                update_after_batch()


if __name__ == "__main__":
    main()
