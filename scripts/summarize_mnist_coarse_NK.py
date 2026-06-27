"""Aggregate mnist_coarse_NK Modal sweep results.

Reads /results/mnist_coarse_NK/{method}/grid{g}_N{N}_K{K}/seed{S}.npz from a
local mirror (download via `modal run modal/mnist_coarse_NK.py::download`) and
emits:

  - SUMMARY.md  (markdown table grouped by grid → N → K → method)
  - sw_vs_K_grid{2,4,8}.png  (one plot per grid: x=K, y=sw, lines per (method, N),
                              with JAM ceiling as a horizontal reference)

Usage:
    python scripts/summarize_mnist_coarse_NK.py \
        --root results/mnist_coarse_NK \
        --out  results/mnist_coarse_NK
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_CELL_RE = re.compile(r"^grid(?P<g>\d+)_N(?P<N>\d+)_K(?P<K>\d+)$")


def _load_cells(root: Path) -> list[dict]:
    """Walk root for {method}/grid{g}_N{N}_K{K}/seed{S}.npz; return one dict per file."""
    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        for cell_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            m = _CELL_RE.match(cell_dir.name)
            if m is None:
                continue
            g, N, K = int(m["g"]), int(m["N"]), int(m["K"])
            for npz_path in sorted(cell_dir.glob("seed*.npz")):
                seed = int(npz_path.stem.replace("seed", ""))
                try:
                    d = np.load(npz_path, allow_pickle=False)
                    sw = d["sw"]
                    chi = d["chi_max"]
                    total_t = float(d["total_time"])
                except Exception as exc:
                    rows.append({"method": method, "grid": g, "N": N, "K": K,
                                  "seed": seed, "sw_final": float("nan"),
                                  "chi_final": -1, "total_time": float("nan"),
                                  "error": str(exc)})
                    continue
                rows.append({
                    "method": method, "grid": g, "N": N, "K": K, "seed": seed,
                    "sw_final": float(sw[-1]),
                    "sw_best": float(np.min(sw)),
                    "chi_final": int(chi[-1]),
                    "chi_max": int(np.max(chi)),
                    "total_time": total_t,
                })
    return rows


def _write_markdown(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["grid"], r["method"], r["N"], r["K"]))
    by_grid: dict[int, list[dict]] = {}
    for r in rows:
        by_grid.setdefault(r["grid"], []).append(r)

    lines = ["# mnist_coarse_NK sweep summary", "",
             "Columns: SW_final = sliced Wasserstein at t=1 (lower is better).",
             "         chi_final = bond dim at t=1; chi_max = max over the trajectory.",
             "         total_time = wall-clock seconds.",
             "         For `jam` rows: N and K are nominal (gradient flow uses K Euler steps).",
             ""]
    for g in sorted(by_grid):
        lines += [f"## grid = {g}×{g}  (d = {g*g})", ""]
        lines += ["| method      | N  | K   | SW_final | SW_best | χ_final | χ_max | t (s) |",
                  "|-------------|----|-----|---------:|--------:|--------:|------:|------:|"]
        # JAM row first (ceiling), then TDVP grouped by method/N/K
        jam_rows = [r for r in by_grid[g] if r["method"] == "jam"]
        tdvp_rows = [r for r in by_grid[g] if r["method"] != "jam"]
        for r in jam_rows + tdvp_rows:
            if "error" in r:
                lines.append(f"| {r['method']:11s} | {r['N']:2d} | {r['K']:3d} | "
                              f"  ERROR | {r['error'][:30]} | | | |")
                continue
            sw_str = f"{r['sw_final']:.4f}"
            if r['chi_final'] >= 64:
                sw_str += " ⚠️"   # cap-saturated
            lines.append(
                f"| {r['method']:11s} | {r['N']:2d} | {r['K']:3d} | "
                f"  {sw_str} | {r['sw_best']:.4f} | "
                f"{r['chi_final']:7d} | {r['chi_max']:5d} | {r['total_time']:5.0f} |"
            )
        lines.append("")
        lines.append("⚠️ = χ_final saturated the D_max=64 cap (SW is a lower bound)")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def _make_plots(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grids = sorted(set(r["grid"] for r in rows))
    for g in grids:
        rows_g = [r for r in rows if r["grid"] == g and "error" not in r]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        tdvp_rows = [r for r in rows_g if r["method"] != "jam"]
        methods = sorted(set(r["method"] for r in tdvp_rows))
        Ns = sorted(set(r["N"] for r in tdvp_rows))
        markers = {"tci_tdvp1": "o", "tci_tdvp2": "s"}
        for method in methods:
            for N in Ns:
                pts = sorted(
                    [(r["K"], r["sw_final"], r["chi_final"])
                     for r in tdvp_rows if r["method"] == method and r["N"] == N],
                    key=lambda x: x[0],
                )
                if not pts:
                    continue
                Ks = [p[0] for p in pts]
                sws = [p[1] for p in pts]
                label = f"{method} N={N}"
                ax.plot(Ks, sws, marker=markers.get(method, "."),
                        label=label, linestyle="-" if N == 4 else "--")
        # JAM ceiling
        jam = [r["sw_final"] for r in rows_g if r["method"] == "jam"]
        if jam:
            ax.axhline(np.mean(jam), color="black", linestyle=":",
                       label=f"JAM ceiling ({np.mean(jam):.3f})")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("K (Trotter steps)")
        ax.set_ylabel("Sliced Wasserstein at t=1")
        ax.set_title(f"coarse MNIST {g}×{g} (d={g*g}) — TDVP vs JAM")
        ax.set_xticks([8, 16, 32, 64])
        ax.set_xticklabels(["8", "16", "32", "64"])
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = out_dir / f"sw_vs_K_grid{g}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="local mirror of /results/mnist_coarse_NK")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir for SUMMARY.md + PNGs (default: --root)")
    args = parser.parse_args()
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_cells(args.root)
    print(f"loaded {len(rows)} cells from {args.root}")
    if not rows:
        return
    _write_markdown(rows, out / "SUMMARY.md")
    _make_plots(rows, out)


if __name__ == "__main__":
    main()
