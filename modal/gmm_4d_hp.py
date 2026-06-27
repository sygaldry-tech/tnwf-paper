"""Modal: gmm_4d HP sweep.

Sobol grid: N ∈ [8, 32], K ∈ [4, 64], D ∈ [8, 64], 48 points × 2 seeds.
Wavefunction methods (Dense, TCI+TDVP1, TCI+TDVP2) use
analytic V_t — no JAM ckpt required. JAM rows are launched separately by
modal/gmm_4d_hp_jam.py once /jam_checkpoints/gmm_4d/ is populated.

Outputs land in /results/gmm_4d_hp/{method}/N{N}_K{K}_D{D}/seed{S}.npz, with
JAM at /results/gmm_4d_hp/jam/N{N}_K{K}/seed{S}.npz (D-invariant).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import modal

APP_NAME = "tnwf-gmm-4d-hp"
VOLUME_NAME = "tnwf-results"
DATASET = "gmm_4d"
RESULT_ROOT = "/results/gmm_4d_hp"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2",
                 "matplotlib>=3.8", "scikit-learn>=1.4")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Resourced-cross params (D_V decoupled from D_out, sub-step M, global pivots)
# default to the original coupled/off behaviour, so old CSVs reproduce exactly.
@app.function(cpu=8.0, memory=16384, timeout=7200, volumes={"/results": volume})
def run_one(method: str, N: int, K: int, D_max: int, seed: int,
            D_V: int = -1, D_out: int = -1, n_v_substeps: int = 1,
            n_global: int = 0) -> str:
    from tnwf.pipelines.run_evolution import run

    if D_V < 0:
        D_V = D_max
    if D_out < 0:
        D_out = D_max
    # Suffix encodes resourced knobs so non-default cells don't collide on disk
    # with the original D_max-keyed paths (kept identical when knobs are default).
    resourced = (D_V != D_max) or (D_out != D_max) or (n_v_substeps != 1) or (n_global != 0)
    suffix = f"_DV{D_V}_DO{D_out}_M{n_v_substeps}_g{n_global}" if resourced else ""

    if method == "jam":
        out_dir = Path(f"{RESULT_ROOT}/{method}/N{N}_K{K}")
        V_source = "jam"
        jam_ckpt = f"/results/jam_checkpoints/{DATASET}/seed{seed}.pt"
        method_kwargs = {}
    else:
        out_dir = Path(f"{RESULT_ROOT}/{method}/N{N}_K{K}_D{D_max}{suffix}")
        V_source = "analytic"
        jam_ckpt = None
        method_kwargs = {"D_max": D_max, "D_V": D_V, "D_out": D_out,
                         "D_init": D_out, "n_v_substeps": n_v_substeps,
                         "n_global": n_global}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_K{K}_D{D_max}{suffix}_s{seed}_skip"

    run(
        method=method, dataset=DATASET, jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=500,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
        V_source=V_source,
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


def _grid_row(r: dict) -> tuple:
    """Build a run_one starmap tuple from a CSV row. Optional resourced columns
    (D_V, D_out, n_v_substeps, n_global) default to coupled/off when absent or
    blank, so legacy CSVs reproduce the original behaviour exactly."""
    Dm = int(r["D_max"])

    def g(key, default):
        v = r.get(key)
        return int(v) if v not in (None, "") else default

    return (r["method"], int(r["N"]), int(r["K"]), Dm, int(r["seed"]),
            g("D_V", Dm), g("D_out", Dm), g("n_v_substeps", 1), g("n_global", 0))


@app.local_entrypoint()
def main(csv_path: str = "data/gmm_4d_hp_wave.csv"):
    """Local-orchestrated starmap (dies if local client disconnects)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    grid = [_grid_row(r) for r in rows]
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
def main_remote(csv_path: str = "data/gmm_4d_hp_wave.csv"):
    """Modal-side dispatcher: spawn() and exit. Survives local-client disconnects."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    grid = [list(_grid_row(r)) for r in rows]
    handle = dispatch.spawn(grid)
    print(f"dispatched {len(grid)} cells from {csv_path}")
    print(f"FunctionCall id: {handle.object_id}")
    print("dispatcher runs independently — local client can exit safely")
