"""Run all V-step methods on petals_2d (trajectory dataset, K=4)."""
from __future__ import annotations

import argparse
from pathlib import Path

from tnwf.leaderboard import update_after_batch_subprocess as update_after_batch
from tnwf.pipelines.run_evolution import run

METHODS = [
    "jam", "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--Ns", type=str, default="8,16")
    # K is fixed to trajectory_K=4 for petals_2d (per-snapshot SW alignment).
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--ckpt_dir", default="data/petals_2d/jam")
    p.add_argument("--out_suffix", default="",
                   help="Optional suffix appended to method dir so AM vs CFM "
                        "results can coexist, e.g. '_am' writes to "
                        "data/petals_2d/<method>_am/N{N}_K{K}/.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ns = [int(n) for n in args.Ns.split(",") if n]

    for seed in seeds:
        ckpt = Path(args.ckpt_dir) / f"seed{seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no JAM checkpoint at {ckpt}")
            continue
        for N in Ns:
            for method in methods:
                out_dir = Path(f"data/petals_2d/{method}{args.out_suffix}/N{N}_K{args.K}")
                out_file = out_dir / f"seed{seed}.npz"
                if out_file.exists() and not args.force:
                    print(f"[skip] {out_file} already exists")
                    continue
                print(f"--> petals_2d / {method} / seed={seed} / N={N} / K={args.K}")
                method_kwargs = {"D_max": args.D_max, "D_V": args.D_V, "D_out": args.D_max}
                run(
                    method=method, dataset="petals_2d", jam_ckpt=str(ckpt),
                    seed=seed, N=N, K=args.K,
                    n_samples=args.n_samples, method_kwargs=method_kwargs,
                    out_dir=out_dir,
                )
                update_after_batch()


if __name__ == "__main__":
    main()
