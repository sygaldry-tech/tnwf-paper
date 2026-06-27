"""g=8 (d=64) MNIST campaign on GPU — pushes N and K within the validated
D ≤ 32 range of the torch port, with optional Hilbert site ordering.

Cells (each spawned independently on A100-80GB, 24 h container budget):

    method ∈ {tci_tdvp1, tci_tdvp2}
    N      ∈ {16, 32}
    K      ∈ {8, 16, 32, 64}
    dataset ∈ {mnist_coarse_8, mnist_coarse_8_hilbert}

D_max = 24 (SCOREBOARD SOTA at 8×8). Sub-bug-region; should produce correct
SW values on GPU per the equivalence tests in tnWF/tests/test_tdvp_torch_equiv.py.

Prerequisites:
  1. JAM checkpoints in /results/jam_checkpoints/{mnist_coarse_8,
     mnist_coarse_8_hilbert}/seed0.pt — produced by
     `modal/train_jam_mnist.py` (row) and `modal/train_jam_hilbert.py`
     (hilbert).

Usage:
    modal run --detach modal/g8_gpu_sweep.py::main
    modal run         modal/g8_gpu_sweep.py::download
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-g8-gpu-sweep"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2", "torchvision>=0.17",
        "matplotlib>=3.8", "scikit-learn>=1.4",
    )
    # PYTHONUNBUFFERED=1 forces print() to flush per line — gives live
    # progress in `modal app logs` rather than only at cell completion.
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

METHODS = ["tci_tdvp1", "tci_tdvp2"]
N_VALUES = [16, 32]
K_VALUES = [8, 16, 32, 64]
SEEDS = [0]
DATASETS = ["mnist_coarse_8", "mnist_coarse_8_hilbert"]


def _method_kwargs(D: int) -> dict:
    # Keep D_V=16 across D waves so the only varying lever is the MPS bond
    # dim. Matches the in-flight D=24 cells.
    return {
        "D_max": D, "D_V": 16, "tol": 1e-3, "D_init": D,
        "n_sweeps": 1, "n_sweeps_cross": 2,
        "device": "cuda",
    }


@app.function(
    volumes={"/results": volume},
    gpu="A100-80GB", cpu=4.0, memory=32768, timeout=86400,
)
def run_cell(dataset: str, method: str, N: int, K: int, seed: int,
             D: int = 24) -> str:
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/g8_gpu_sweep/D{D}/{dataset}/{method}/N{N}_K{K}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"D{D}/{dataset}/{method}_N{N}_K{K}_seed{seed}_skip"

    jam_ckpt = f"/results/jam_checkpoints/{dataset}/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"D{D}/{dataset}/{method}_N{N}_K{K}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt})")

    run(
        method=method, dataset=dataset, jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=2000,
        save=True, out_dir=str(out_dir), method_kwargs=_method_kwargs(D),
    )
    return f"D{D}/{dataset}/{method}_N{N}_K{K}_seed{seed}_done"


def _wave(D: int, label: str):
    cells = [
        (d, m, n, k, s, D)
        for d in DATASETS for m in METHODS
        for n in N_VALUES for k in K_VALUES for s in SEEDS
    ]
    print(f"launching {len(cells)} {label} cells (D={D}, A100-80GB, 24h each)")
    done = ok = err = 0
    for result in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== {label}: {ok} ok, {err} failed (total {done}) ===")


@app.local_entrypoint()
def main():
    """Original D=24 wave (kept for compatibility)."""
    _wave(24, "g=8 GPU sweep D=24")


@app.local_entrypoint()
def d32_wave():
    """Bond-up wave: D=32. ~2.4× wall vs D=24. Same chi cap → likely
    closer to JAM ceiling since D=24 was bond-saturated."""
    _wave(32, "g=8 GPU sweep D=32")


@app.local_entrypoint()
def d48_wave():
    """Bond-up wave: D=48. ~8× wall vs D=24. Last regime before the
    D=64 cuBLAS bug zone; cells may approach 24h budget."""
    _wave(48, "g=8 GPU sweep D=48")


@app.local_entrypoint()
def d64_wave():
    """Bond-up wave: D=64. Path C diagnostic confirmed SW_best is correct
    on GPU at D=64 (matches CPU to 1%) after the tf32-off + matrix_exp-on-CPU
    + Saad-residual fixes. Cells projected 2–20 h on A100; tail cells
    (N=32 K=64) may timeout — return_exceptions=True keeps the others."""
    _wave(64, "g=8 GPU sweep D=64")


@app.local_entrypoint()
def d64_probe():
    """Single-cell probe to test whether the D=64 drift bug manifests at
    g=8 production scale. Cell: row-major tdvp2 N=16 K=8 D=64. SVD shape
    here is only (1024, 1024) vs the diagnostic's (4096, 4096) — if the
    drift is SVD-precision-bound, it should disappear at this size."""
    cells = [("mnist_coarse_8", "tci_tdvp2", 16, 8, 0, 64)]
    print(f"launching 1 g=8 D=64 N=16 K=8 tdvp2 probe cell")
    done = ok = err = 0
    for result in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== d64_probe: {ok} ok, {err} failed ===")


@app.local_entrypoint()
def d64_n16_only():
    """Lean variant: only N=16 cells at D=64. 16 cells, all should fit
    in 24h. Useful to bound the D=64 GPU win before risking heavy
    N=32 cells."""
    cells = [
        (d, m, n, k, s, 64)
        for d in DATASETS for m in METHODS
        for n in [16] for k in K_VALUES for s in SEEDS
    ]
    print(f"launching {len(cells)} g=8 D=64 N=16-only cells")
    done = ok = err = 0
    for result in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== d64_n16_only: {ok} ok, {err} failed (total {done}) ===")


@app.local_entrypoint()
def row_only():
    """Just the row-ordered cells (use after train_jam_mnist.py finishes,
    before train_jam_hilbert.py is done)."""
    cells = [
        ("mnist_coarse_8", m, n, k, s)
        for m in METHODS for n in N_VALUES for k in K_VALUES for s in SEEDS
    ]
    print(f"launching {len(cells)} g=8 row-ordered GPU cells")
    done = ok = err = 0
    for result in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== {ok} ok, {err} failed (total {done}) ===")


# ── Volume helpers ────────────────────────────────────────────────────────

@app.function(volumes={"/results": volume})
def _list_results() -> list:
    import os
    root = "/results/g8_gpu_sweep"
    if not os.path.isdir(root):
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs]


@app.function(volumes={"/results": volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download(dest: str = "results/g8_gpu_sweep"):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    files = _list_results.remote()
    print(f"Downloading {len(files)} files → {dest_path}/")
    for remote_path in files:
        rel = Path(remote_path).relative_to("/results/g8_gpu_sweep")
        local = dest_path / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(_read_file.remote(remote_path))
        print(f"  {rel}")
