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
# Bolding marks the better of the two TDVP variants per column. Dense is
# deliberately excluded: at d=2 Dense (0.112) beats both TDVP variants, but the
# paper bolds TCI+1TDVP (0.127) there — the comparison is within TDVP only.
BOLD_KEYS = ["tci_tdvp1", "tci_tdvp2"]
ROW_LABEL = {
    "__exact__":  "Target--target (finite-sample)",
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_tdvp1":  "TCI+1TDVP",
    "tci_tdvp2":  "TCI+2TDVP",
}
ROW_ORDER = [m for _, ms in ROW_GROUPS for m in ms]


def best_sw(cells: list[dict]) -> float | None:
    return min(c["sw"] for c in cells) if cells else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figures/table1_sw.tex")
    args = p.parse_args()

    # Load Exact noise floor (precomputed by compute_exact_sw_floor.py).
    exact_path = Path("data/exact_sw_floor.json")
    exact_floor = {}
    if exact_path.exists():
        exact_floor = {int(k): float(v["mean"])
                       for k, v in json.loads(exact_path.read_text()).items()}

    # Build the table dict: table[method][d] = SW.
    table: dict[str, dict[int, float | None]] = {m: {} for m in ROW_ORDER}
    for d, dir_ in DATASETS:
        rd = Path(dir_)
        # Wave methods (Dense, TDVP1, TDVP2 — see _hp_utils.WAVE_METHODS).
        for m in WAVE_METHODS:
            cells = collect(rd, m)
            table[m][d] = best_sw(cells)
        # JAM (separate D-invariant path).
        jam_cells = collect_reference(rd, "jam")
        table["jam"][d] = best_sw(jam_cells)
        # Exact noise floor.
        table["__exact__"][d] = exact_floor.get(d)

    # ── Emit the LaTeX tabular ──────────────────────────────────────────
    # Row groups separated by \midrule. Within the TDVP group, only the
    # cell of the TDVP variant with the best (lowest) SW at each d is
    # bolded — so each column highlights the better of TDVP1 / TDVP2.
    tdvp_keys = BOLD_KEYS
    best_tdvp_per_d = {}
    for d in DS_ALL:
        candidates = [(m, table[m].get(d)) for m in tdvp_keys
                      if table[m].get(d) is not None]
        if candidates:
            best_tdvp_per_d[d] = min(candidates, key=lambda kv: kv[1])[0]

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
        in_tdvp = group_name == "wave"
        for m in ms:
            label = ROW_LABEL[m]
            cells = []
            for d in DS_ALL:
                v = table[m].get(d)
                cell = f"{v:.3f}" if v is not None else "--"
                # Bold only when this row has the best SW among TDVP1/TDVP2 at d.
                if in_tdvp and v is not None and best_tdvp_per_d.get(d) == m:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"saved {out}")
    print()
    print("Preview:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
