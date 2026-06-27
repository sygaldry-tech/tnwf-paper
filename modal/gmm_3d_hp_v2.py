"""Modal: gmm_3d HP sweep v2 — optimised for cost.

Changes from v1:
  - 3 methods only (Dense, TCI+TDVP1, TCI+TDVP2) — drop JAM/ACI/TCI+ALS
    (JAM is N-invariant and tested separately; ACI/TCI+ALS fail at d=3
    high-N anyway and aren't informative).
  - Analytic V_t (no JAM oracle) — eliminates training noise + 2× faster oracle.
  - Tighter axes: N ∈ [16, 64], K ∈ [4, 32], D ∈ [8, 64] (drop the expensive
    high ends; we already see saturation in the 2D sweep).
  - 32 Sobol points (vs 64) — pilot validated GP fit at this density.
  - 2 seeds (vs 3) — Sobol density compensates for lower seed count.
  - 600s per-container timeout — runaway cells terminate early.
  - n_samples=500 (vs 2000) — SW noise still small at √(500/2000)≈ 50%
    relative SE, dominated by seed variance anyway.

Total: 3 methods × ~30 unique (N,K,D) cells × 2 seeds ≈ 180 containers.
Estimated cost: $1–3 vs $10–30 for v1.
"""
from __future__ import annotations

import csv
from pathlib import Path

import modal

APP_NAME = "tnwf-gmm-3d-hp-v2"
VOLUME_NAME = "tnwf-results"
CSV_PATH = "data/gmm_3d_hp_v2.csv"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2",
                 "matplotlib>=3.8", "scikit-learn>=1.4")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(cpu=8.0, memory=16384, timeout=7200, volumes={"/results": volume})
def run_one(method: str, N: int, K: int, D_max: int, seed: int,
            D_V: int = -1, D_out: int = -1, n_v_substeps: int = 1,
            n_global: int = 0) -> str:
    from tnwf.pipelines.run_evolution import run
    from tnwf.modal_resourced import resourced_kwargs

    method_kwargs, suffix = resourced_kwargs(method, D_max, D_V, D_out,
                                             n_v_substeps, n_global)
    out_dir = Path(f"/results/gmm_3d_hp_v2/{method}/N{N}_K{K}_D{D_max}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_K{K}_D{D_max}{suffix}_s{seed}_skip"

    run(
        method=method, dataset="gmm_3d", jam_ckpt=None,
        seed=seed, N=N, K=K, n_samples=500,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
        V_source="analytic",
    )
    return f"{method}_N{N}_K{K}_D{D_max}{suffix}_s{seed}_done"


@app.function(cpu=0.5, memory=1024, timeout=24 * 3600, volumes={"/results": volume})
def dispatch(grid: list) -> str:
    """Modal-side starmap dispatcher — survives local-client disconnects."""
    done = timed_out = other_errors = 0
    grid = [tuple(row) for row in grid]
    for result in run_one.starmap(grid, return_exceptions=True):
        if isinstance(result, BaseException):
            kind = type(result).__name__
            if "Timeout" in kind:
                timed_out += 1
            else:
                other_errors += 1
        else:
            done += 1
    summary = f"{done} done, {timed_out} timed out, {other_errors} other errors"
    print(summary)
    return summary


@app.local_entrypoint()
def main(csv_path: str = CSV_PATH):
    """Local-orchestrated starmap (dies if local client disconnects)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    from tnwf.modal_resourced import grid_row
    grid = [grid_row(r) for r in rows]
    print(f"launching {len(grid)} containers from {csv_path}")
    done = timed_out = other_errors = 0
    for result in run_one.starmap(grid, return_exceptions=True):
        if isinstance(result, BaseException):
            kind = type(result).__name__
            if "Timeout" in kind:
                timed_out += 1
                print(f"TIMEOUT: {kind}")
            else:
                other_errors += 1
                print(f"ERROR: {kind}: {result}")
        else:
            done += 1
            print(result)
    print(f"=== {done} done, {timed_out} timed out, {other_errors} other errors ===")


@app.local_entrypoint()
def main_remote(csv_path: str = CSV_PATH):
    """Modal-side dispatcher: spawn() and exit. Survives local-client disconnects."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    from tnwf.modal_resourced import grid_row
    grid = [list(grid_row(r)) for r in rows]
    handle = dispatch.spawn(grid)
    print(f"dispatched {len(grid)} cells from {csv_path}")
    print(f"FunctionCall id: {handle.object_id}")
    print("dispatcher runs independently — local client can exit safely")
