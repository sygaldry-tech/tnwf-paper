"""Emit Table 1 of the paper: best-cell SW per (d, method) across d=2..8.

Output is a LaTeX `tabular` block ready for `\\input{}` from the main TeX file.
Row selection uses the same `_hp_utils.collect` helpers as `make_fig4.py`, so the
table's numbers are consistent with Fig 6.

Layout: rows are Target--target / JAM / Dense / TCI+1TDVP / TCI+2TDVP; columns are
d in 2..8. Cells hold the best-cell SW (mean across seeds where more than one
exists). Infeasible cells (Dense at d>=6) are typeset ``--''.

Requires the hyperparameter sweeps under `data/gmm_*_hp*/` (published as a
separate data archive, unpacked at the repo root) plus `data/exact_sw_floor.json`
for the target--target finite-sample floor.

Usage:
    make table1
    uv run python scripts/make_table1_sw.py [--out figures/table1_sw.tex]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _hp_utils import WAVE_METHODS, collect, collect_reference

DATASETS = [
    (2, "data/gmm_2d_hp"),
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
    (6, "data/gmm_6d_hp"),
    (7, "data/gmm_7d_hp"),
    (8, "data/gmm_8d_hp"),
]
DS_ALL = [d for d, _ in DATASETS]

# Visual grouping for the LaTeX table: each tuple is (group label, [row keys]),
# with a \midrule inserted between groups.
# Two groups, one \midrule between them — matching the submitted Table 1.
# The archive version of this script also carried a "cross-interp" group
# (TCI+ALS, ACI); those methods are not in the current paper, and the public
# `_hp_utils.WAVE_METHODS` does not include them either.
ROW_GROUPS = [
    ("reference", ["__exact__", "jam"]),
    ("wave",      ["dense", "tci_tdvp1", "tci_tdvp2"]),
]
# No bolding. Cells cluster tightly near the minimum and the gaps between the
# two TDVP integrators (0.001-0.02) sit at or below the selection noise, so
# bolding a winner per column asserted a resolution the data does not have.
# The reported uncertainty makes that visible instead.
ROW_LABEL = {
    "__exact__":  "Target--target (finite-sample)",
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+1TDVP",
    "tci_tdvp2":  "TCI+2TDVP",
}
ROW_ORDER = [m for _, ms in ROW_GROUPS for m in ms]


def best_cell(cells: list[dict]) -> dict | None:
    """Cell with the lowest seed-mean SW."""
    return min(cells, key=lambda c: c["sw"]) if cells else None


def uncertainty(cell: dict) -> float:
    """Spread to quote on a best-cell SW.

    Two things move a cell: the seed-to-seed spread of the run itself, and the
    Monte-Carlo noise of the sliced-Wasserstein estimator (which the migration
    records per file as `sw_endpoint_mc_std`). We quote the larger, since a
    single-seed cell has no seed spread to measure but still carries estimator
    noise. Both are of order 0.005, which is the scale on which the argmin
    over ~35 cells moves — hence no bolding.
    """
    cands = [v for v in (cell.get("sw_std"), cell.get("sw_mc_std"))
             if v is not None and not math.isnan(v)]
    return max(cands) if cands else float("nan")


def fmt(v: float | None, err: float = float("nan"), show_spread: bool = False) -> str:
    """Format a cell.

    The spread is still computed and reported on stdout, because the caption's
    statement that the two integrators are separated by less than it needs to
    stay checkable -- but it is kept out of the table itself, which reads
    better at seven columns. Pass --spread to typeset it as `0.115(6)`, the
    parenthesised digit being the 1-sigma spread in units of the last decimal.
    """
    if v is None:
        return "--"
    if not show_spread or math.isnan(err):
        return f"{v:.3f}"
    return f"{v:.3f}({max(1, min(9, round(err * 1000)))})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/table1_sw.tex")
    p.add_argument("--spread", action="store_true",
                   help="typeset the 1-sigma spread as 0.115(6); off by default, "
                        "but always reported on stdout")
    args = p.parse_args()

    # Load Exact noise floor (precomputed by compute_exact_sw_floor.py).
    exact_path = Path("data/exact_sw_floor.json")
    exact_floor = {}
    if exact_path.exists():
        exact_floor = {int(k): float(v["mean"])
                       for k, v in json.loads(exact_path.read_text()).items()}

    # Build the table dict: table[method][d] = (SW, spread).
    table: dict[str, dict[int, tuple | None]] = {m: {} for m in ROW_ORDER}
    for d, dir_ in DATASETS:
        rd = Path(dir_)
        for m in WAVE_METHODS:
            # Dense results are D-independent but were stored under N*_K*_D*,
            # so `collect` handed Dense a min over replicate runs that the
            # bond-dimension-dependent methods never got. `collect_reference`
            # averages over D first, putting every row on the same footing.
            cells = (collect_reference(rd, m) if m == "dense" else collect(rd, m))
            c = best_cell(cells)
            table[m][d] = (c["sw"], uncertainty(c)) if c else None
        c = best_cell(collect_reference(rd, "jam"))
        table["jam"][d] = (c["sw"], uncertainty(c)) if c else None
        floor = exact_floor.get(d)
        table["__exact__"][d] = (floor, float("nan")) if floor is not None else None

    # ── Emit the LaTeX tabular ──────────────────────────────────────────
    col_spec = "l" + "c" * len(DS_ALL)
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(f"$d={d}$" for d in DS_ALL) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for g_idx, (group_name, ms) in enumerate(ROW_GROUPS):
        if g_idx > 0:
            lines.append(r"\midrule")
        for m in ms:
            label = ROW_LABEL[m]
            cells = []
            for d in DS_ALL:
                v = table[m].get(d)
                cells.append(fmt(*v, show_spread=args.spread) if v is not None
                             else "--")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    # Report the spreads regardless, so the caption's claim about the
    # integrators being separated by less than the noise stays verifiable.
    spreads = [e for m in ROW_ORDER for v in table[m].values()
               if v is not None for e in (v[1],) if not math.isnan(e)]
    if spreads:
        print(f"1-sigma spread across cells: min {min(spreads):.4f}  "
              f"median {sorted(spreads)[len(spreads) // 2]:.4f}  "
              f"max {max(spreads):.4f}")
    print(f"saved {out}")
    print()
    print("Preview:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
