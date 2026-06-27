"""Modal sweep: swiss_roll grid-resolution N ∈ {16, 32, 64, 128, 256}.

Question: do wavefunction methods catch up to (or beat) JAM as cell size L/N
shrinks? At N=16 JAM clearly wins (SW=0.092 vs 0.165 for wavefunction at full
bond), but the wavefunction methods are bottlenecked by grid quantisation.

Grid: 4 methods × 5 N × 5 seeds = 100 containers (well under the 1000 cap).

Each container trains its own JAM checkpoint inline (~30 s) and runs the
8-step product formula at the given N. Output → Modal volume tnwf-results.

Launch:
    modal run modal/swiss_roll_N.py

Download:
    modal volume get tnwf-results /swiss_roll_N/* ./results/swiss_roll_2d/N_sweep/
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-swiss-roll-N"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2",
        "matplotlib>=3.8", "scikit-learn>=1.4",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

METHODS = [
    "jam", "dense",
    "tci_tdvp1", "tci_tdvp2",
]
N_VALUES = [16, 32, 64, 128, 256]
SEEDS = list(range(10))

K_FIXED = 8


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=3600,
    volumes={"/results": volume},
)
def run_one(method: str, N: int, seed: int) -> str:
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/swiss_roll_N/{method}/N{N}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_seed{seed}_skip"

    # Load pre-trained JAM (run modal/train_jam.py first)
    jam_ckpt = f"/results/jam_checkpoints/swiss_roll_2d/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_N{N}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run `modal run modal/train_jam.py` first.")

    D_max = min(N, 32)
    method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                     "D_init": D_max}
    run(
        method=method, dataset="swiss_roll_2d", jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K_FIXED, n_samples=2000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    # rely on implicit commit at exit (avoids server contention)
    return f"{method}_N{N}_seed{seed}_done"


@app.local_entrypoint()
def main():
    """Launch all (method, N, seed) cells in parallel via starmap."""
    grid = [(m, n, s) for m in METHODS for n in N_VALUES for s in SEEDS]
    print(f"launching {len(grid)} containers "
          f"({len(METHODS)} methods × {len(N_VALUES)} N × {len(SEEDS)} seeds)")
    done = 0
    for result in run_one.starmap(grid):
        print(result)
        done += 1
    print(f"=== ALL {done} jobs complete ===")
