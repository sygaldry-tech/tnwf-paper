"""Aggregate the 5-backend GPU kernel benchmark.

Reads `results/bench_gpu_kernels/{cpu,t4,a10g,l40s,a100}.json` and emits:
  - SUMMARY.md            table of (op, D, N) → time per backend + speedup vs CPU
  - speedup_heatmap_{op}.png   one per op (4 PNGs)

Usage:
    python scripts/summarize_gpu_bench.py --root results/bench_gpu_kernels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BACKENDS_ORDER = ["cpu", "t4", "a10g", "l40s", "a100"]
OPS_ORDER = ["matmul", "full_svd", "truncated_svd", "expm_multiply"]


def _load_backends(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for backend in BACKENDS_ORDER:
        path = root / f"{backend}.json"
        if not path.exists():
            print(f"  missing: {path.name}")
            continue
        with open(path) as fh:
            out[backend] = json.load(fh)
    return out


def _make_cell_index(backend_data: dict) -> dict[tuple, dict]:
    """Map (op, D, N) → cell dict."""
    return {(c["op"], c["D"], c["N"]): c for c in backend_data["cells"]}


def _cell_ok(c: dict | None) -> bool:
    """Cell has a usable timing — either fully ok, or only-one-iter (still good)."""
    if c is None:
        return False
    s = c.get("status", "ok")
    return s == "ok" or s.startswith("slow_one_iter_only")


def _write_markdown(backends: dict[str, dict], out_path: Path) -> None:
    if "cpu" not in backends:
        raise SystemExit("CPU baseline missing — can't compute speedups")

    cpu_index = _make_cell_index(backends["cpu"])
    gpu_indices = {b: _make_cell_index(backends[b]) for b in BACKENDS_ORDER
                   if b != "cpu" and b in backends}

    lines = ["# GPU kernel benchmark — tnWF MPS-TDVP hot ops", ""]
    lines += [f"- CPU baseline: `{backends['cpu']['device_name']}`, "
              f"torch={backends['cpu']['torch_version']}, "
              f"dtype={backends['cpu']['dtype']}",
              f"- GPU backends: " + ", ".join(
                  f"{b.upper()} (`{backends[b]['device_name']}`)" for b in gpu_indices),
              "",
              "Per-cell value = mean wall-clock time in seconds; "
              "GPU columns include the speedup vs CPU `(× CPU)`.",
              ""]

    for op in OPS_ORDER:
        lines.append(f"## op = `{op}`")
        lines.append("")
        header = ["D", "N", "n=D·N", "CPU (s)"] + [
            f"{b.upper()} (s) [× CPU]" for b in gpu_indices]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---:" for _ in header]) + "|")

        for D in sorted({k[1] for k in cpu_index if k[0] == op}):
            for N in sorted({k[2] for k in cpu_index if k[0] == op and k[1] == D}):
                cpu_cell = cpu_index.get((op, D, N))
                if not _cell_ok(cpu_cell):
                    continue
                t_cpu = cpu_cell["mean_s"]
                row = [str(D), str(N), str(D * N), f"{t_cpu:.4f}"]
                for b, idx in gpu_indices.items():
                    c = idx.get((op, D, N))
                    if c is None:
                        row.append("—")
                    elif not _cell_ok(c):
                        row.append(f"FAIL ({c['status'][:20]})")
                    else:
                        speedup = t_cpu / c["mean_s"] if c["mean_s"] > 0 else float("inf")
                        row.append(f"{c['mean_s']:.4f} [{speedup:.1f}×]")
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Headline summary
    lines += ["## Headline speedups (geometric mean across cells, only D·N ≥ 1024)", ""]
    lines += ["| backend | geomean speedup | max speedup | min speedup |",
              "|---|---:|---:|---:|"]
    for b, idx in gpu_indices.items():
        ratios = []
        for (op, D, N), gpu_cell in idx.items():
            if D * N < 1024:
                continue
            cpu_cell = cpu_index.get((op, D, N))
            if not (_cell_ok(cpu_cell) and _cell_ok(gpu_cell)):
                continue
            ratios.append(cpu_cell["mean_s"] / gpu_cell["mean_s"])
        if ratios:
            geom = float(np.exp(np.mean(np.log(ratios))))
            lines.append(
                f"| {b.upper()} | {geom:.1f}× | {max(ratios):.1f}× | {min(ratios):.1f}× |")
        else:
            lines.append(f"| {b.upper()} | n/a | n/a | n/a |")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def _make_heatmaps(backends: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "cpu" not in backends:
        return
    cpu_index = _make_cell_index(backends["cpu"])
    gpu_backends = [b for b in BACKENDS_ORDER if b != "cpu" and b in backends]
    gpu_indices = {b: _make_cell_index(backends[b]) for b in gpu_backends}

    D_vals = sorted({c["D"] for c in backends["cpu"]["cells"]})
    N_vals = sorted({c["N"] for c in backends["cpu"]["cells"]})

    for op in OPS_ORDER:
        fig, axes = plt.subplots(1, len(gpu_backends), figsize=(4 * len(gpu_backends), 4),
                                 squeeze=False)
        for ax, b in zip(axes[0], gpu_backends):
            grid = np.full((len(D_vals), len(N_vals)), np.nan)
            for i, D in enumerate(D_vals):
                for j, N in enumerate(N_vals):
                    cpu_c = cpu_index.get((op, D, N))
                    gpu_c = gpu_indices[b].get((op, D, N))
                    if not (_cell_ok(cpu_c) and _cell_ok(gpu_c)):
                        continue
                    if gpu_c["mean_s"] > 0:
                        grid[i, j] = cpu_c["mean_s"] / gpu_c["mean_s"]
            im = ax.imshow(grid, cmap="RdYlGn", origin="lower",
                            vmin=0.5, vmax=max(20.0, np.nanmax(grid) if np.any(np.isfinite(grid)) else 20.0),
                            aspect="auto")
            ax.set_xticks(range(len(N_vals)))
            ax.set_xticklabels([str(n) for n in N_vals])
            ax.set_yticks(range(len(D_vals)))
            ax.set_yticklabels([str(d) for d in D_vals])
            ax.set_xlabel("N")
            ax.set_ylabel("D")
            ax.set_title(f"{b.upper()}")
            for i in range(len(D_vals)):
                for j in range(len(N_vals)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                                color="black" if val > 5 else "white")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="speedup × CPU")

        fig.suptitle(f"GPU speedup vs CPU — op = {op}", fontsize=13)
        plt.tight_layout()
        out_png = out_dir / f"speedup_heatmap_{op}.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/bench_gpu_kernels"))
    args = parser.parse_args()
    backends = _load_backends(args.root)
    print(f"loaded {len(backends)} backends from {args.root}: {list(backends)}")
    if not backends:
        return
    _write_markdown(backends, args.root / "SUMMARY.md")
    _make_heatmaps(backends, args.root)


if __name__ == "__main__":
    main()
