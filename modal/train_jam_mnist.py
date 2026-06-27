"""Train coarse-MNIST JAM checkpoints once and cache on Modal volume.

Stage 1 of the MNIST coarse-grain TDVP-vs-JAM comparison. Trains one JAM
ScalarPotentialMLP per grid in {2, 4, 8} and writes checkpoints to the
`tnwf-results` volume at `/jam_checkpoints/mnist_coarse_{g}/seed0.pt`.

Usage:
    modal run modal/train_jam_mnist.py
    modal run modal/train_jam_mnist.py --gpu T4    # faster JAM training
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-train-jam-mnist"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2", "torchvision>=0.17",
        "scikit-learn>=1.4",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

DATASETS = ["mnist_coarse_2", "mnist_coarse_4", "mnist_coarse_8"]
SEEDS = [0]


def _train_one_impl(dataset: str, seed: int) -> str:
    from tnwf.jam.train import DATASET_DEFAULTS, train as train_jam

    out_dir = Path(f"/results/jam_checkpoints/{dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.pt"
    if out_file.exists():
        volume.commit()
        return f"{dataset}_seed{seed}_skip"

    n_iter = DATASET_DEFAULTS[dataset]["training"]["n_iter"]
    train_jam(
        dataset=dataset, seed=seed, out_path=str(out_file),
        n_iter=n_iter, log_every=max(1000, n_iter // 10),
    )
    volume.commit()
    return f"{dataset}_seed{seed}_done"


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes={"/results": volume})
def train_one_cpu(dataset: str, seed: int) -> str:
    return _train_one_impl(dataset, seed)


@app.function(gpu="T4", cpu=2.0, memory=8192, timeout=3600,
              volumes={"/results": volume})
def train_one_gpu(dataset: str, seed: int) -> str:
    return _train_one_impl(dataset, seed)


@app.local_entrypoint()
def main(gpu: str = "off"):
    """Train one JAM per grid in {2, 4, 8}. 3 containers total.

    Args:
        gpu: 'off' (CPU, default) or 't4' (T4 GPU, ~3-5x faster).
    """
    grid = [(d, s) for d in DATASETS for s in SEEDS]
    print(f"launching {len(grid)} JAM-MNIST containers (gpu={gpu})")
    fn = train_one_gpu if gpu.lower() == "t4" else train_one_cpu
    done = 0
    for result in fn.starmap(grid):
        print(result)
        done += 1
    print(f"=== ALL {done} JAM checkpoints in /jam_checkpoints/mnist_coarse_* ===")
