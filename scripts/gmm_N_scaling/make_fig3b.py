"""Figure 3 panel B: scaling with N. SW (or NLL) vs N for each method."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    "dense",     "tci_tdvp1", "tci_tdvp2",
]
METHOD_COLORS = {
    "dense":      "black",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}
N_PATTERN = re.compile(r"N(\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/gmm_N_scaling")
    p.add_argument("--out", default="results/gmm_N_scaling/fig3b.pdf")
    p.add_argument("--metric", default="sw", choices=["sw", "mmd"])
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    for method in METHODS:
        method_dir = res_dir / method
        if not method_dir.exists():
            continue
        n_to_vals: dict[int, list[float]] = {}
        for sub in method_dir.iterdir():
            m = N_PATTERN.match(sub.name)
            if not m:
                continue
            N_val = int(m.group(1))
            for seed_file in sub.glob("seed*.npz"):
                run = np.load(seed_file, allow_pickle=True)
                n_to_vals.setdefault(N_val, []).append(float(run[args.metric][-1]))

        if not n_to_vals:
            continue
        Ns = sorted(n_to_vals.keys())
        means = np.array([np.mean(n_to_vals[N]) for N in Ns])
        stds = np.array([np.std(n_to_vals[N]) for N in Ns])
        ax.errorbar(Ns, means, yerr=stds, marker="o", label=method,
                    color=METHOD_COLORS[method], capsize=3)

    ax.set_xlabel("grid resolution N")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(f"Figure 3b — scaling with N ({args.metric.upper()})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
