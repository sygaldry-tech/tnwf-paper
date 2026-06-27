"""Train JAM checkpoints once and cache on Modal volume.

This is stage 1 of the two-stage Modal pipeline:
  1. modal/train_jam.py   — train JAMs per (dataset, seed), save to volume.
  2. modal/{N,NK,hp}.py   — load cached ckpts, run V-step pipelines (no retrain).

Without caching, every V-step container retrains JAM from scratch (~30-60 s each),
which dominates wall time when the V-step itself is fast. With this cache, a
1500-run sweep skips ~25 min of redundant training.

Usage:
    modal run modal/train_jam.py
    modal run modal/train_jam.py --datasets swiss_roll_2d --seeds 10
    modal run modal/train_jam.py --gpu T4    # ~3-5x speedup at small extra cost

Output layout on Modal volume `tnwf-results`:
    /jam_checkpoints/{dataset}/seed{S}.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-train-jam"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DEFAULT_DATASETS = ["swiss_roll_2d", "gmm_2d", "gmm_3d"]
DEFAULT_N_SEEDS = 10


def _train_one_impl(dataset: str, seed: int) -> str:
    """Common impl shared by CPU and GPU function variants."""
    from tnwf.jam.train import train as train_jam

    out_dir = Path(f"/results/jam_checkpoints/{dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.pt"
    if out_file.exists():
        volume.commit()
        return f"{dataset}_seed{seed}_skip"

    train_jam(
        dataset=dataset, seed=seed, out_path=str(out_file),
        n_iter=8000, batch_size=256, hidden=128,
        n_layers=3, time_embed_dim=64, lr=1e-3, log_every=4000,
    )
    volume.commit()
    return f"{dataset}_seed{seed}_done"


@app.function(cpu=2.0, memory=4096, timeout=1800, volumes={"/results": volume})
def train_one_cpu(dataset: str, seed: int) -> str:
    return _train_one_impl(dataset, seed)


@app.function(gpu="T4", cpu=2.0, memory=8192, timeout=1200,
              volumes={"/results": volume})
def train_one_gpu(dataset: str, seed: int) -> str:
    return _train_one_impl(dataset, seed)


@app.local_entrypoint()
def main(
    datasets: str = ",".join(DEFAULT_DATASETS),
    seeds: int = DEFAULT_N_SEEDS,
    gpu: str = "off",
):
    """Train JAMs in parallel.

    Args:
        datasets: comma-separated dataset list (default: swiss_roll_2d,gmm_2d,gmm_3d)
        seeds:    number of seeds per dataset (default: 10)
        gpu:      'off' (CPU) or 't4' (T4 GPU, ~3-5x faster, slight extra cost)
    """
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    grid = [(d, s) for d in ds_list for s in range(seeds)]
    print(f"launching {len(grid)} JAM training containers "
          f"({len(ds_list)} datasets × {seeds} seeds, gpu={gpu})")
    fn = train_one_gpu if gpu.lower() == "t4" else train_one_cpu
    done = 0
    for result in fn.starmap(grid):
        print(result)
        done += 1
    print(f"=== ALL {done} JAM checkpoints in /jam_checkpoints/ ===")
