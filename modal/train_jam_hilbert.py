"""Train JAM checkpoints for the Hilbert-ordered g=8 MNIST variant.

Mirrors `modal/train_jam_mnist.py` but trains the `mnist_coarse_8_hilbert`
(and `mnist_coarse_8_snake` for completeness) datasets, which differ from
the row-major `mnist_coarse_8` only by site permutation.

Usage:
    modal run --detach modal/train_jam_hilbert.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-train-jam-hilbert"
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


DATASETS = ["mnist_coarse_8_hilbert", "mnist_coarse_8_snake"]
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
def train_one(dataset: str, seed: int) -> str:
    return _train_one_impl(dataset, seed)


@app.local_entrypoint()
def main():
    grid = [(d, s) for d in DATASETS for s in SEEDS]
    print(f"launching {len(grid)} JAM-hilbert/snake containers")
    done = 0
    for result in train_one.starmap(grid):
        print(result)
        done += 1
    print(f"=== {done} JAM checkpoints written ===")
