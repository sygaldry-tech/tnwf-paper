"""Train JAM checkpoints for gmm_2d × N seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from tnwf.jam.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--n_iter", type=int, default=10_000)
    p.add_argument("--out_dir", default="data/gmm_2d/jam")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]

    for seed in seeds:
        out = f"{args.out_dir}/seed{seed}.pt"
        print(f"=== training gmm_2d seed={seed} → {out} ===")
        train(dataset="gmm_2d", seed=seed, out_path=out, n_iter=args.n_iter)


if __name__ == "__main__":
    main()
