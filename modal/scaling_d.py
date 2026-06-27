"""Modal sweep for Fig 3 panel A: scaling with d.

Grid: 6 methods × 7 d-values × 10 seeds = 420 containers (well under the
1000 concurrent-container cap on the user's Modal account).

Dense excluded (would OOM at d ≥ 5 with N=16).

Outputs land on a Modal Volume; download with:
    modal volume get tnwf-results /scaling_d/* ./results/gmm_d_scaling/

Launch:
    modal run modal/scaling_d.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-scaling-d"
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

METHODS = ["tci_tdvp1", "tci_tdvp2"]
D_VALUES = [2, 3, 4, 5, 6, 7, 8]
SEEDS = list(range(10))

N_FIXED = 16
K_FIXED = 16


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=3600,
    volumes={"/results": volume},
)
def run_one(method: str, d: int, seed: int) -> str:
    """One container = one (method, d, seed) cell. Loads pre-trained JAM (d-specific)."""
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/scaling_d/{method}/d{d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JAM checkpoints keyed by spatial dim (since d is overridden from the
    # dataset's default). Cache populated by `modal run modal/train_jam.py`
    # with the gmm_d{d} dataset variant.
    jam_ckpt = f"/results/jam_checkpoints/gmm_d{d}/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_d{d}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Train d-specific JAMs first.")

    method_kwargs = {"D_max": 16, "D_V": 16, "D_out": 16}
    run(
        method=method, dataset="gmm_d_scaling", jam_ckpt=jam_ckpt,
        seed=seed, N=N_FIXED, d=d, K=K_FIXED, n_samples=2_000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    # rely on implicit commit at exit (avoids server contention)
    return f"{method}_d{d}_seed{seed}_done"


@app.local_entrypoint()
def main():
    """Launch the full grid in parallel via Function.starmap."""
    grid = [(m, d, s) for m in METHODS for d in D_VALUES for s in SEEDS]
    print(f"launching {len(grid)} containers ({len(METHODS)} methods × "
          f"{len(D_VALUES)} d × {len(SEEDS)} seeds)")
    for result in run_one.starmap(grid):
        print(result)
