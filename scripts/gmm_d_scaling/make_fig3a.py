"""Figure 3 panel A: scaling with d. SW (or NLL) vs d for each method.

Reads from results/gmm_d_scaling/{method}/d{D}/seed{S}.npz.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]
METHOD_COLORS = {
    "dense":      "black",
    "tci_als":    "tab:blue",
    "aci":        "tab:orange",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}
D_PATTERN = re.compile(r"d(\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/gmm_d_scaling")
    p.add_argument("--out", default="results/gmm_d_scaling/fig3a.pdf")
    p.add_argument("--metric", default="sw", choices=["sw", "mmd"])
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHODS:
        method_dir = res_dir / method
        if not method_dir.exists():
            continue
        d_to_vals: dict[int, list[float]] = {}
        for sub in method_dir.iterdir():
            m = D_PATTERN.match(sub.name)
            if not m:
                continue
            d_val = int(m.group(1))
            for seed_file in sub.glob("seed*.npz"):
                run = np.load(seed_file, allow_pickle=True)
                d_to_vals.setdefault(d_val, []).append(float(run[args.metric][-1]))

        if not d_to_vals:
            continue
        ds = sorted(d_to_vals.keys())
        means = np.array([np.mean(d_to_vals[d]) for d in ds])
        stds = np.array([np.std(d_to_vals[d]) for d in ds])
        ax.errorbar(ds, means, yerr=stds, marker="o", label=method,
                    color=METHOD_COLORS[method], capsize=3)

    ax.set_xlabel("dimension d")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(f"Figure 3a — scaling with d ({args.metric.upper()})")
    ax.legend(fontsize=8, loc="best")
    ax.set_yscale("log")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
