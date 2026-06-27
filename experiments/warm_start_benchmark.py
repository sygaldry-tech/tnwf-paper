"""Warm-start A/B benchmark for the V-step pipeline.

Each piece in the warm-start audit is opt-in via an environment-variable
flag, so we can A/B test by toggling. Each run uses the same seed +
identical (d, N, K, D) so timing and SW are directly comparable.

Pieces (set TNWF_WARMSTART_<PIECE>=1 to enable):
    PIVOTS    : TT-cross pivot (right_idx) reuse across steps         (1.1+1.2)
    ALS_INIT  : ALS initial guess from previous psi                   (2.1)
    ORACLE    : V-grid oracle cache across steps                      (4.1)

Usage:
    # Baseline:
    python3 experiments/warm_start_benchmark.py

    # Test piece 1.1 alone:
    TNWF_WARMSTART_PIVOTS=1 python3 experiments/warm_start_benchmark.py

    # Stack pieces:
    TNWF_WARMSTART_PIVOTS=1 TNWF_WARMSTART_ALS_INIT=1 \\
        python3 experiments/warm_start_benchmark.py

The flags are read inside the V-step modules (see warm_start.py).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tnwf.pipelines.run_evolution import run


# ---------------------------------------------------------------------------
# Cells: small enough to iterate quickly, big enough to stress the pipeline.
# ---------------------------------------------------------------------------

CELLS = [
    # Small A/B cell for fast iteration on piece 1.1 (TT-cross pivot warm-start).
    # d=3 N=16 K=20 D=16 — should run in seconds per pass.
    {"name": "tci_als_d3_N16",
     "method": "tci_als", "dataset": "gmm_3d",
     "N": 16, "K": 20,
     "method_kwargs": {"D_V": 16, "D_out": 16, "n_sweeps": 2,
                       "n_sweeps_cross": 2}},
    # Same cell, ACI V-step — checks that pivot/oracle warm-starts also
    # benefit the prrLU pivot path (the ALS-init piece does not apply).
    {"name": "aci_d3_N16",
     "method": "aci", "dataset": "gmm_3d",
     "N": 16, "K": 20,
     "method_kwargs": {"D_V": 16, "D_max": 16, "n_sweeps": 4,
                       "n_sweeps_cross": 2, "tol": 1e-6}},
    # Reference: TDVP-1 at the same cell. Not a warm-start variant
    # (TDVP-1's V-step doesn't use the same exp(iβV) MPO build path
    # for accuracy-critical operations); included so the piece-1.1
    # conclusion has a comparison to the paper's headline method.
    {"name": "tdvp1_d3_N16",
     "method": "tci_tdvp1", "dataset": "gmm_3d",
     "N": 16, "K": 20,
     "method_kwargs": {"D_V": 16, "n_sweeps": 1, "n_sweeps_cross": 2}},
]


def warm_start_summary() -> str:
    flags = []
    for piece in ("PIVOTS", "ALS_INIT", "ORACLE"):
        env = f"TNWF_WARMSTART_{piece}"
        if os.environ.get(env, "0") == "1":
            flags.append(piece)
    return ",".join(flags) if flags else "baseline"


def run_cell(cell: dict, n_samples: int = 1500) -> dict:
    print(f"\n=== {cell['name']} | warm-start: {warm_start_summary()} ===",
          flush=True)
    t0 = time.perf_counter()
    out = run(
        method=cell["method"],
        dataset=cell["dataset"],
        V_source="analytic",
        seed=0,
        N=cell["N"],
        K=cell["K"],
        n_samples=n_samples,
        out_dir=None,
        save=False,
        method_kwargs=cell["method_kwargs"],
    )
    wall = time.perf_counter() - t0
    sw = float(out["sw"][-1]) if isinstance(out["sw"], (list, np.ndarray)) and len(out["sw"]) else float(out["sw"])
    chi = int(out["chi_max"][-1]) if isinstance(out["chi_max"], (list, np.ndarray)) and len(out["chi_max"]) else int(out["chi_max"])
    step_times = list(out.get("step_times", [])) or []
    median_step_s = float(np.median(step_times)) if step_times else float("nan")
    print(f"  wall={wall:.2f}s  median_step={median_step_s:.3f}s  "
          f"SW={sw:.4f}  chi={chi}",
          flush=True)
    return {
        "cell": cell["name"],
        "warm_start": warm_start_summary(),
        "wall_s": wall,
        "median_step_s": median_step_s,
        "step_times": step_times,
        "sw": sw,
        "chi_max": chi,
    }


def main():
    rows = []
    for cell in CELLS:
        rows.append(run_cell(cell))
    out_dir = "experiments/results"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"warm_start_{warm_start_summary()}.json".replace(",", "+")
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
