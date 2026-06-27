"""Compare CFM-trained vs AM-trained vs CFM+OT-coupled JAM on biology datasets.

Reads three families of run npzs and prints per-snapshot SW side-by-side:

  cfm     :  data/<dataset>/<method>/N{N}_K{K}/seed*.npz
  am      :  data/<dataset>/<method>_am/N{N}_K{K}/seed*.npz
  cfm_ot  :  data/<dataset>/<method>_cfm_ot/N{N}_K{K}/seed*.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_sw(dataset: str, method_dir: str, N: int, K: int) -> np.ndarray | None:
    arrs = []
    for p in sorted(Path(f"data/{dataset}/{method_dir}/N{N}_K{K}").glob("seed*.npz")):
        arrs.append(np.load(p)["sw"])
    return np.stack(arrs) if arrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    args = p.parse_args()

    variants = [("cfm", ""), ("am", "_am"), ("cfm_ot", "_cfm_ot")]

    for dataset, methods in (
        ("petals_2d", ["jam", "dense", "tci_tdvp1", "tci_tdvp2"]),
        ("eb_5d", ["jam", "tci_tdvp1", "tci_tdvp2"]),
    ):
        print()
        print("=" * 100)
        print(f"{dataset} (N={args.N}, K={args.K}, D=16, 3 seeds): "
              f"CFM vs AM vs CFM+OT JAM training")
        print("=" * 100)
        for method in methods:
            sws = {}
            for variant_name, suffix in variants:
                arr = _load_sw(dataset, f"{method}{suffix}", args.N, args.K)
                if arr is not None:
                    sws[variant_name] = arr.mean(axis=0)
            if not sws:
                continue
            print(f"\n  {method}")
            print(f"  {'variant':<10}  " + "  ".join(f"sw[{k}]" for k in range(args.K + 1)))
            print(f"  {'-' * 10}  " + "  ".join(["------"] * (args.K + 1)))
            for variant_name in ("cfm", "am", "cfm_ot"):
                if variant_name not in sws:
                    continue
                row = f"  {variant_name:<10}  " + "  ".join(
                    f"{x:>6.3f}" for x in sws[variant_name])
                print(row)


if __name__ == "__main__":
    main()
