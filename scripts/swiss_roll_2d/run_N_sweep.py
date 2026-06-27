"""Swiss roll N-sweep: vary grid resolution N for all methods.

The premise: at low N the wavefunction methods are bottlenecked by grid
quantization (cells of size L/N). JAM (continuous Euler integration) is
unaffected by N. As N grows, the wavefunction methods should catch up.

For each N: D_max = min(N, 32) so MPS retains full bond expressivity at small
N and compresses at large N. Saves to
    results/swiss_roll_2d/{method}/N{N}/seed{S}.npz
"""
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
    p.add_argument("--Ns", type=str, default="16,32,64,128")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--methods", type=str, default=",".join(METHODS))
    p.add_argument("--ckpt_dir", default="data/swiss_roll_2d/jam")
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
                out_dir = Path(f"data/swiss_roll_2d/{method}/N{N}")
                out_file = out_dir / f"seed{seed}.npz"
                if out_file.exists() and not args.force:
                    print(f"[skip] {out_file}")
                    continue
                D_max = min(N, 32)
                method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                                 "D_init": D_max}
                print(f"--> swiss_roll_2d / {method} / N={N} / seed={seed} / D_max={D_max}")
                run(
                    method=method, dataset="swiss_roll_2d", jam_ckpt=str(ckpt),
                    seed=seed, N=N, K=args.K,
                    n_samples=args.n_samples, method_kwargs=method_kwargs,
                    out_dir=out_dir,
                )
                update_after_batch()


if __name__ == "__main__":
    main()
