"""Modal sweep for Fig 3 panel B: scaling with N.

Grid: 6 methods × 6 N-values × 10 seeds = 360 containers.

Launch:
    modal run modal/scaling_N.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-scaling-N"
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
N_VALUES = [8, 16, 32, 64, 128, 256]
SEEDS = list(range(10))

D_FIXED = 3
K_FIXED = 16


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=7200,
    volumes={"/results": volume},
)
def run_one(method: str, N: int, seed: int) -> str:
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/scaling_N/{method}/N{N}")
    out_dir.mkdir(parents=True, exist_ok=True)

    jam_ckpt = f"/results/jam_checkpoints/gmm_3d/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_N{N}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run `modal run modal/train_jam.py` first.")

    # Cap D_max so memory stays bounded as N grows
    D_max = min(32, max(8, N // 2))
    method_kwargs = {"D_max": D_max, "D_V": min(D_max, 16), "D_out": D_max}
    run(
        method=method, dataset="gmm_N_scaling", jam_ckpt=jam_ckpt,
        seed=seed, N=N, d=D_FIXED, K=K_FIXED, n_samples=2_000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    # rely on implicit commit at exit (avoids server contention)
    return f"{method}_N{N}_seed{seed}_done"


@app.local_entrypoint()
def main():
    grid = [(m, n, s) for m in METHODS for n in N_VALUES for s in SEEDS]
    print(f"launching {len(grid)} containers ({len(METHODS)} methods × "
          f"{len(N_VALUES)} N × {len(SEEDS)} seeds)")
    for result in run_one.starmap(grid):
        print(result)
