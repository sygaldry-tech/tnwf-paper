"""Modal: gmm_5d HP sweep — GPU variant with torch_cuda Krylov backend.

Mirrors gmm_5d_hp.py but runs on Modal L40S with backend='torch_cuda' so the
inner expm_multiply uses the batched Krylov on GPU. Results land at
/results/gmm_5d_hp_gpu/{method}/N{N}_K{K}_D{D}/seed{S}.npz so they don't
collide with the CPU sweep. Enables the d=5 N=32 K=64 D=64 cells which are
borderline-infeasible on CPU.
"""
from __future__ import annotations

import csv
from pathlib import Path

import modal

APP_NAME = "tnwf-gmm-5d-hp-gpu"
VOLUME_NAME = "tnwf-results"
DATASET = "gmm_5d"
RESULT_ROOT = "/results/gmm_5d_hp_gpu"

# Default PyPI torch ships with CUDA bundled for Linux x86_64 + Py3.11.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2",
                 "matplotlib>=3.8", "scikit-learn>=1.4")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu="L40S", cpu=4.0, memory=32768, timeout=24 * 3600,
              volumes={"/results": volume})
def run_one_gpu(method: str, N: int, K: int, D_max: int, seed: int,
                D_V: int = -1, D_out: int = -1, n_v_substeps: int = 1,
                n_global: int = 0) -> str:
    """Same dispatch as gmm_5d_hp.run_one but with backend='torch_cuda' for
    TDVP methods so the inner Krylov expm_multiply runs on the L40S GPU.

    Resourced-cross knobs (D_V/D_out/n_v_substeps/n_global) default to
    coupled/off → original behaviour. exp(iβV) cross is memory-bounded by
    env-ALS, so high D_V is safe even at d=5.
    """
    from tnwf.pipelines.run_evolution import run
    from tnwf.modal_resourced import resourced_kwargs

    method_kwargs, suffix = resourced_kwargs(method, D_max, D_V, D_out,
                                             n_v_substeps, n_global)
    if method == "jam":
        out_dir = Path(f"{RESULT_ROOT}/{method}/N{N}_K{K}")
        V_source = "jam"
        jam_ckpt = f"/results/jam_checkpoints/{DATASET}/seed{seed}.pt"
    else:
        method_kwargs["backend"] = "torch_cuda" if method.startswith("tci_tdvp") else "numpy"
        out_dir = Path(f"{RESULT_ROOT}/{method}/N{N}_K{K}_D{D_max}{suffix}")
        V_source = "analytic"
        jam_ckpt = None

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_K{K}_D{D_max}{suffix}_s{seed}_skip"

    # Checkpoint after each Trotter K-step + commit to volume so a preempted
    # restart picks up where the previous container left off.
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
    return f"{method}_N{N}_K{K}_D{D_max}{suffix}_s{seed}_done"


@app.function(cpu=0.5, memory=1024, timeout=24 * 3600,
              volumes={"/results": volume})
def dispatch(grid: list) -> str:
    """Modal-side starmap dispatcher — survives local-client disconnects."""
    done = timed_out = other_errors = 0
    grid = [tuple(row) for row in grid]
    for result in run_one_gpu.starmap(grid, return_exceptions=True):
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
def main_remote(csv_path: str = "data/gmm_5d_hp_n32_high_D.csv"):
    """Spawn the GPU dispatcher and exit. Output lives at /results/gmm_5d_hp_gpu/."""
    from tnwf.modal_resourced import grid_row
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    grid = [list(grid_row(r)) for r in rows]
    handle = dispatch.spawn(grid)
    print(f"dispatched {len(grid)} GPU cells from {csv_path}")
    print(f"FunctionCall id: {handle.object_id}")
    print(f"results land at /results/gmm_5d_hp_gpu/<method>/N<N>_K<K>_D<D>/seed<S>.npz")
