"""Generate ``data/gmm_5d_hp_warmstart.csv`` from the existing Sobol grid
``data/gmm_5d_hp_wave.csv``.

For each (method, N, K, D_max, seed) row in the source grid, this script
emits one row per warm-start variant that the method supports:

  • tci_als → cold, P (1.1), P+A (1.1+2.1), P+A+O (all three).
  • aci     → cold, P (1.1), P+O (1.1+4.1).
  • dense / tci_tdvp1 / tci_tdvp2 → cold only.

Each output row carries a "variant" column. The Modal worker
``modal/gmm_5d_hp_warmstart.py`` parses that and sets the matching
TNWF_WARMSTART_* environment variables before calling ``run()``.

Usage:
    python3 experiments/make_warmstart_grid.py
"""
from __future__ import annotations

import csv
import os
import sys


VARIANTS_BY_METHOD: dict[str, list[str]] = {
    "tci_als":   ["cold", "P", "P+A", "P+A+O"],
    "aci":       ["cold", "P", "P+O"],
    "dense":     ["cold"],
    "tci_tdvp1": ["cold"],
    "tci_tdvp2": ["cold"],
}


def main() -> None:
    src_path = "data/gmm_5d_hp_wave.csv"
    dst_path = "data/gmm_5d_hp_warmstart.csv"
    if not os.path.exists(src_path):
        print(f"missing {src_path}", file=sys.stderr)
        sys.exit(1)

    with open(src_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    out_rows = []
    for r in rows:
        method = r["method"]
        variants = VARIANTS_BY_METHOD.get(method, ["cold"])
        for v in variants:
            out_rows.append({
                "method":  method,
                "N":       r["N"],
                "K":       r["K"],
                "D_max":   r["D_max"],
                "seed":    r["seed"],
                "variant": v,
            })

    with open(dst_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "N", "K", "D_max", "seed", "variant"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows → {dst_path}")
    # Summary by variant for sanity.
    by_variant: dict[str, int] = {}
    for r in out_rows:
        by_variant[r["variant"]] = by_variant.get(r["variant"], 0) + 1
    print("rows per variant:")
    for v, n in sorted(by_variant.items()):
        print(f"  {v:>8s}: {n}")


if __name__ == "__main__":
    main()
