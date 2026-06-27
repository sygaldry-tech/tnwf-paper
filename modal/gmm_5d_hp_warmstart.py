"""Modal: gmm_5d HP sweep × warm-start variants.

Mirrors ``modal/gmm_5d_hp.py`` but expects an extra ``variant`` column on
the input CSV (see ``experiments/make_warmstart_grid.py``). For each row
the worker sets the corresponding TNWF_WARMSTART_* env flags before
calling ``run()``.

Variants:
  cold     — no warm-start (parity with gmm_5d_hp.py reference).
  P        — PIVOTS (1.1) only.
  P+A      — PIVOTS (1.1) + ALS_INIT (2.1).        [tci_als only]
  P+O      — PIVOTS (1.1) + ORACLE (4.1).          [aci only]
  P+A+O    — all three.                            [tci_als only]

Results are written to:
  /results/gmm_5d_hp_warmstart/<variant>/<method>/N{N}_K{K}_D{D}/seed{s}.npz
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import modal

APP_NAME = "tnwf-gmm-5d-hp-warmstart"
VOLUME_NAME = "tnwf-results"
DATASET = "gmm_5d"
RESULT_ROOT = "/results/gmm_5d_hp_warmstart"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2",
                 "matplotlib>=3.8", "scikit-learn>=1.4")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# Map shorthand variant → env-var dict the worker sets before calling run().
_VARIANT_ENV: dict[str, dict[str, str]] = {
    "cold":  {},
    "P":     {"TNWF_WARMSTART_PIVOTS": "1"},
    "P+A":   {"TNWF_WARMSTART_PIVOTS": "1", "TNWF_WARMSTART_ALS_INIT": "1"},
    "P+O":   {"TNWF_WARMSTART_PIVOTS": "1", "TNWF_WARMSTART_ORACLE": "1"},
    "P+A+O": {"TNWF_WARMSTART_PIVOTS": "1", "TNWF_WARMSTART_ALS_INIT": "1",
              "TNWF_WARMSTART_ORACLE": "1"},
}


@app.function(cpu=4.0, memory=16384, timeout=24 * 3600, volumes={"/results": volume})
def run_one(
    method: str, N: int, K: int, D_max: int, seed: int, variant: str,
) -> str:
    # Set warm-start env vars BEFORE importing tnwf (the dispatcher reads them
    # at _make_v_step time, which happens inside run()).
    for k, v in _VARIANT_ENV[variant].items():
        os.environ[k] = v

    from tnwf.pipelines.run_evolution import run

    if method == "jam":
        out_dir = Path(f"{RESULT_ROOT}/{variant}/{method}/N{N}_K{K}")
        V_source = "jam"
        jam_ckpt = f"/results/jam_checkpoints/{DATASET}/seed{seed}.pt"
        method_kwargs = {}
    else:
        out_dir = Path(f"{RESULT_ROOT}/{variant}/{method}/N{N}_K{K}_D{D_max}")
        V_source = "analytic"
        jam_ckpt = None
        method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                         "D_init": D_max}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_{variant}_N{N}_K{K}_D{D_max}_s{seed}_skip"

    ckpt_path = out_dir / f"seed{seed}.checkpoint.pkl"
    run(
        method=method, dataset=DATASET, jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=500,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
        V_source=V_source,
        checkpoint_path=str(ckpt_path),
        checkpoint_callback=lambda: volume.commit(),
        checkpoint_every=1,
    )
    return f"{method}_{variant}_N{N}_K{K}_D{D_max}_s{seed}_done"


@app.function(cpu=0.5, memory=1024, timeout=24 * 3600, volumes={"/results": volume})
def dispatch(grid: list) -> str:
    """Modal-side starmap dispatcher (parity with gmm_5d_hp.py.dispatch)."""
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
def main(csv_path: str = "data/gmm_5d_hp_warmstart.csv"):
    """Local client orchestrates the starmap. Dies if local disconnects."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    grid = [(r["method"], int(r["N"]), int(r["K"]), int(r["D_max"]),
             int(r["seed"]), r["variant"]) for r in rows]
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
def main_remote(csv_path: str = "data/gmm_5d_hp_warmstart.csv"):
    """Modal-side dispatcher: spawn() and exit. Survives local-client disconnects."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    grid = [[r["method"], int(r["N"]), int(r["K"]), int(r["D_max"]),
             int(r["seed"]), r["variant"]] for r in rows]
    handle = dispatch.spawn(grid)
    print(f"dispatched {len(grid)} cells from {csv_path}")
    print(f"FunctionCall id: {handle.object_id}")
    print("dispatcher runs independently — local client can exit safely")
    print(f"poll status:  uv run modal call-graph {handle.object_id}")
