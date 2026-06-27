"""One-shot summary of all EB sweep cells under data/eb_5d/.

Iterates over every ``data/eb_5d/<method>/N<N>_K<K>[*]/seed*.npz`` and prints
the per-snapshot SW mean ± std and mean walltime grouped by (cell, method).
Picks up ``N{N}_K{K}_warm`` and any other annotated cell directory so the
warm-start vs cold comparison is visible.

Usage: ``uv run python scripts/summarize_eb_sweep.py``.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

CELL_RE = re.compile(r"^N(\d+)_K(\d+)(_.*)?$")


def main():
    root = Path("data/eb_5d")
    if not root.exists():
        print(f"no EB sweep data at {root}")
        return

    # Collect (cell_label, method, sw_seeds, walls)
    rows = []
    for method_dir in sorted(root.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for cell_dir in sorted(method_dir.iterdir()):
            m = CELL_RE.match(cell_dir.name)
            if not m:
                continue
            N, K = int(m.group(1)), int(m.group(2))
            suffix = m.group(3) or ""
            sws, walls = [], []
            for p in sorted(cell_dir.glob("seed*.npz")):
                d = np.load(p)
                sws.append(d["sw"])
                walls.append(float(d["total_time"]))
            if not sws:
                continue
            rows.append((N, K, suffix, method, np.stack(sws), np.array(walls)))

    # Group by (N, K, suffix)
    cells = {}
    for N, K, suffix, method, sws, walls in rows:
        cells.setdefault((N, K, suffix), []).append((method, sws, walls))

    method_order = ["jam", "dense", "tci_als", "aci", "tci_tdvp1", "tci_tdvp2"]

    def _key(meth):
        return method_order.index(meth) if meth in method_order else 99

    for (N, K, suffix), methods in sorted(cells.items()):
        tag = f"N={N}, K={K}" + (f" [{suffix.lstrip('_')}]" if suffix else "")
        print()
        print("=" * 90)
        print(f"EB (5D PCA, scRNA-seq) — {tag}, D=16, 3 seeds")
        print("=" * 90)
        print(f"{'method':<12} {'sw[0]':>7} {'sw[1]':>7} {'sw[2]':>7} "
              f"{'sw[3]':>7} {'sw[4]':>7}  {'chi':>3}  {'wall(s)':>7}")
        print("-" * 78)
        for method, sws, walls in sorted(methods, key=lambda r: _key(r[0])):
            sw_mean = sws.mean(axis=0)
            sw_std = sws.std(axis=0)
            # chi_max not stored per-seed here; use last loaded run's chi
            print(f"{method:<12} " + " ".join(f"{x:>7.4f}" for x in sw_mean)
                  + f"        {walls.mean():>6.1f}")


if __name__ == "__main__":
    main()
