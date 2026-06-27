"""Table 3: hyperparameter cells on the memory and walltime Pareto
frontiers at d=5 (Fig 4 panels A and B).

For each method we list every (N, K, D_max) cell that is non-dominated
on the (memory_ratio, SW) frontier (panel A) and on the (walltime, SW)
frontier (panel B). Cells appearing on both frontiers are bolded.

Output: a single LaTeX `tabular` block ready for `\\input{}`.

Usage:
    uv run python scripts/make_table3_pareto_configs.py \\
        --out figures/table3_pareto_configs.tex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _hp_utils import (
    METHOD_LABEL, WAVE_METHODS, collect, memory_fraction,
)


# Methods in display order (matches Fig 4 legend).
METHOD_ORDER = ["dense", "tci_tdvp1", "tci_tdvp2"]


def pareto_indices(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """Indices of points (xs, ys) on the lower-left Pareto frontier
    (both axes minimised). Returns the subset of input indices in
    ascending-x order."""
    if len(xs) == 0:
        return []
    order = np.argsort(xs)
    keep = []
    best = np.inf
    for j in order:
        if ys[j] < best:
            keep.append(int(j))
            best = ys[j]
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/table3_pareto_configs.tex")
    p.add_argument("--results_dir", default="data/gmm_5d_hp")
    p.add_argument("--d", type=int, default=5)
    args = p.parse_args()

    res_dir = Path(args.results_dir)
    rows = []                     # collected cell records for the LaTeX table
    for m in METHOD_ORDER:
        cells = collect(res_dir, m)
        if not cells:
            continue
        # x-axis values for the two Pareto fronts.
        if m == "dense":
            xs_mem = np.ones(len(cells))
        else:
            xs_mem = np.array(
                [memory_fraction(c["N"], c["chi"], args.d) for c in cells]
            )
        xs_time = np.array([c["time"] for c in cells])
        ys = np.array([c["sw"] for c in cells])

        mask_mem  = np.isfinite(xs_mem)  & np.isfinite(ys) & (xs_mem  > 0) & (ys > 0)
        mask_time = np.isfinite(xs_time) & np.isfinite(ys) & (xs_time > 0) & (ys > 0)
        mem_idx = pareto_indices(xs_mem[mask_mem], ys[mask_mem])
        time_idx = pareto_indices(xs_time[mask_time], ys[mask_time])
        # Translate back to indices in the original `cells` list.
        mem_orig = list(np.where(mask_mem)[0][mem_idx])
        time_orig = list(np.where(mask_time)[0][time_idx])
        on_both = set(mem_orig) & set(time_orig)
        on_mem_only  = set(mem_orig)  - on_both
        on_time_only = set(time_orig) - on_both

        # Build a deduplicated row set, preserving ascending SW order.
        row_idx = sorted(on_both | on_mem_only | on_time_only,
                         key=lambda j: cells[j]["sw"])
        for j in row_idx:
            c = cells[j]
            tag = ("M+W" if j in on_both
                   else "M  " if j in on_mem_only
                   else "  W")
            rows.append({
                "method":  m,
                "method_label": METHOD_LABEL[m],
                "N":       int(c["N"]),
                "K":       int(c["K"]),
                "D":       int(c["D"]),
                "chi":     int(c["chi"]),
                "sw":      float(c["sw"]),
                "mem":     float(xs_mem[j]) if j < len(xs_mem) else float("nan"),
                "time":    float(c["time"]),
                "front":   tag,
                "on_both": j in on_both,
            })

    # ── Emit the LaTeX tabular ──────────────────────────────────────────
    lines = []
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & $N$ & $K$ & $D_{\max}$ & $\chi_{\max}$ & SW & "
        r"$\frac{\text{MPS}}{\text{Dense}}$ & Walltime (s) & Front \\"
    )
    lines.append(r"\midrule")
    last_method = None
    for r in rows:
        if last_method is not None and r["method"] != last_method:
            lines.append(r"\midrule")
        last_method = r["method"]
        method_cell = r["method_label"] if r["method"] != last_method else ""
        method_cell = r["method_label"]                       # always show
        sw  = f"{r['sw']:.3f}"
        mem = f"{r['mem']:.2g}"
        wall = f"{r['time']:.0f}"
        front_pretty = {"M+W": r"M\,+\,W", "M  ": "M", "  W": "W"}[r["front"]]
        cells_tex = (f"{method_cell} & {r['N']} & {r['K']} & {r['D']} & "
                     f"{r['chi']} & {sw} & {mem} & {wall} & "
                     f"{front_pretty}")
        # Bold cells that are on BOTH fronts.
        if r["on_both"]:
            cells_tex = r"\textbf{" + cells_tex.replace("&", r"} & \textbf{") + "}"
        lines.append(cells_tex + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")
    print()
    print("Preview (first 30 lines):")
    for ln in lines[:30]:
        print(ln)


if __name__ == "__main__":
    main()
