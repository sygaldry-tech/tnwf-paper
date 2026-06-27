"""Train JAM checkpoints for petals_2d (trajectory mode) × N seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from tnwf.jam.train import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n_iter", type=int, default=10_000)
    p.add_argument("--out_dir", default="data/petals_2d/jam")
    p.add_argument("--loss", choices=("cfm", "am"), default="cfm",
                   help="cfm: CFM-with-grad (legacy default); "
                        "am: Action Matching (Neklyudov 2022).")
    p.add_argument("--ot_coupling", action="store_true",
                   help="Entropic-OT pair re-coupling per minibatch "
                        "(Sinkhorn-permuted x1). Trajectory data only.")
    p.add_argument("--ot_epsilon", type=float, default=0.05)
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]

    for seed in seeds:
        out = f"{args.out_dir}/seed{seed}.pt"
        tag = f"loss={args.loss}, ot={args.ot_coupling}"
        print(f"=== training petals_2d seed={seed} ({tag}) → {out} ===")
        train(dataset="petals_2d", seed=seed, out_path=out,
              n_iter=args.n_iter, loss_name=args.loss,
              ot_coupling=args.ot_coupling, ot_epsilon=args.ot_epsilon)


if __name__ == "__main__":
    main()
