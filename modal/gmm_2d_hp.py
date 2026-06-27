"""Modal: Sobol-sampled HP sweep for gmm_2d (mirror of swiss_roll_hp_pilot)."""
from __future__ import annotations

import csv
from pathlib import Path

import modal

APP_NAME = "tnwf-gmm-2d-hp"
VOLUME_NAME = "tnwf-results"
CSV_PATH = "data/gmm_2d_hp.csv"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2",
                 "matplotlib>=3.8", "scikit-learn>=1.4")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(cpu=2.0, memory=8192, timeout=3600, volumes={"/results": volume})
def run_one(method: str, N: int, K: int, D_max: int, seed: int) -> str:
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/gmm_2d_hp/{method}/N{N}_K{K}_D{D_max}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_K{K}_D{D_max}_s{seed}_skip"

    jam_ckpt = f"/results/jam_checkpoints/gmm_2d/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_N{N}_K{K}_D{D_max}_s{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run `modal run modal/train_jam.py` first.")

    method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                     "D_init": D_max}
    run(
        method=method, dataset="gmm_2d", jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=2000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    # rely on implicit commit at exit (avoids server contention)
    return f"{method}_N{N}_K{K}_D{D_max}_s{seed}_done"


@app.local_entrypoint()
def main():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    grid = [(r["method"], int(r["N"]), int(r["K"]), int(r["D_max"]), int(r["seed"]))
            for r in rows]
    print(f"launching {len(grid)} containers from {CSV_PATH}")
    done = 0
    for result in run_one.starmap(grid):
        print(result)
        done += 1
    print(f"=== ALL {done} jobs complete ===")
