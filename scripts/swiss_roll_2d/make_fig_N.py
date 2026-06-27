"""SW (and MMD) vs N for swiss_roll, all methods, fixed K.

Reads results/swiss_roll_2d/{method}/N{N}/seed*.npz. Two side-by-side panels
(SW, MMD) with error bars across seeds.
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
    "jam", "dense",     "tci_tdvp1", "tci_tdvp2",
]
METHOD_LABEL = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+TDVP1",
    "tci_tdvp2":  "TCI+TDVP2",
}
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}
N_PATTERN = re.compile(r"^N(\d+)$")


def collect(results_dir: Path) -> dict:
    """Return {method: {N: list[(metric, vals_array)]}}."""
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for method in METHODS:
        method_dir = results_dir / method
        if not method_dir.exists():
            continue
        for sub in method_dir.iterdir():
            m = N_PATTERN.match(sub.name)
            if not m:
                continue
            N = int(m.group(1))
            sw_vals, mmd_vals = [], []
            for f in sorted(sub.glob("seed*.npz")):
                try:
                    z = np.load(f, allow_pickle=True)
                    sw_vals.append(float(z["sw"][-1]))
                    mmd_vals.append(float(z["mmd"][-1]))
                except (OSError, KeyError, EOFError):
                    pass
            if sw_vals:
                out.setdefault(method, {})[N] = {
                    "sw": np.asarray(sw_vals),
                    "mmd": np.asarray(mmd_vals),
                }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/swiss_roll_2d")
    p.add_argument("--out", default="results/swiss_roll_2d/fig_N.pdf")
    args = p.parse_args()

    data = collect(Path(args.results_dir))
    if not data:
        raise SystemExit(f"No N-sweep results at {args.results_dir}/{{method}}/N*/")

    Ns_all = sorted({N for d in data.values() for N in d})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=200)
    for ax, metric, ylabel in zip(axes, ("sw", "mmd"),
                                  ("SW (sliced Wasserstein)", "MMD (RBF)")):
        for method in METHODS:
            if method not in data:
                continue
            xs, means, errs = [], [], []
            for N in Ns_all:
                if N not in data[method]:
                    continue
                vals = data[method][N][metric]
                xs.append(N)
                means.append(vals.mean())
                errs.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, means, yerr=errs, marker="o", lw=1.8, capsize=4,
                            ms=8, color=METHOD_COLORS[method],
                            label=METHOD_LABEL[method])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (grid resolution)", fontsize=12)
        ax.set_ylabel(ylabel + "  (mean ± SE)", fontsize=12)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=10, loc="best", ncol=2)

    fig.suptitle(
        "Swiss roll (2D): convergence with grid resolution N\n"
        "K = 8 fixed,  D_max = 32 fixed (compressed at N ≥ 64)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
