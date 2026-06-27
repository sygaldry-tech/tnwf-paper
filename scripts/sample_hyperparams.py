"""Sobol-sample hyperparameters over (N, K, D_max), snap to dyadic, write CSV.

Usage:
    python scripts/sample_hyperparams.py \\
        --n_points 50 --seed 0 --methods tci_tdvp2 --n_seeds 5 \\
        --out data/hp_pilot.csv

Output CSV: method,N,K,D_max,seed (one row per run).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import qmc


def sobol_dyadic(n_points: int, axes: dict[str, tuple[int, int]],
                 seed: int = 0) -> list[dict[str, int]]:
    """Sobol-sample n_points in log space over each axis, snap to powers of 2.

    `axes` is a dict like {"N": (16, 256), "K": (4, 64), "D_max": (8, 128)}.
    """
    keys = list(axes.keys())
    lo = np.array([np.log2(axes[k][0]) for k in keys])
    hi = np.array([np.log2(axes[k][1]) for k in keys])
    d = len(keys)

    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    raw = sampler.random(n=n_points)                # (n, d) in [0,1]
    log2_vals = lo + raw * (hi - lo)                # log2 in [lo, hi]
    snapped = np.power(2, np.round(log2_vals).astype(int))
    return [
        {k: int(v) for k, v in zip(keys, row)} for row in snapped
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_points", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", type=str, default="tci_tdvp2",
                   help="Comma-separated method names")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--N_range", type=str, default="16,256")
    p.add_argument("--K_range", type=str, default="4,64")
    p.add_argument("--D_range", type=str, default="8,128")
    p.add_argument("--out", default="data/hp_pilot.csv")
    args = p.parse_args()

    axes = {
        "N":     tuple(int(x) for x in args.N_range.split(",")),
        "K":     tuple(int(x) for x in args.K_range.split(",")),
        "D_max": tuple(int(x) for x in args.D_range.split(",")),
    }
    points = sobol_dyadic(args.n_points, axes, seed=args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = []
    for pt in points:
        for method in methods:
            for s in range(args.n_seeds):
                rows.append({"method": method, **pt, "seed": s})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "N", "K", "D_max", "seed"])
        w.writeheader()
        w.writerows(rows)

    # Brief summary
    Ns = sorted({r["N"] for r in rows})
    Ks = sorted({r["K"] for r in rows})
    Ds = sorted({r["D_max"] for r in rows})
    print(f"Wrote {len(rows)} run rows to {out}")
    print(f"  unique N values: {Ns}")
    print(f"  unique K values: {Ks}")
    print(f"  unique D values: {Ds}")
    print(f"  methods × seeds: {len(methods)} × {args.n_seeds}")


if __name__ == "__main__":
    main()
