"""Modal GPU timing benchmark — tnWF MPS-TDVP hot kernels.

Times 4 dense linear-algebra primitives that dominate `tdvp_2site_diagonal`:

  - matmul          : A @ B,  A, B  ∈ ℂ^(n×n)
  - full_svd        : U, s, Vh = svd(A, full_matrices=True),  A ∈ ℂ^(n×n)
  - truncated_svd   : U[:,:D], s[:D], Vh[:D]  with full_matrices=False
  - expm_multiply   : exp(τ·H) v    (scipy Krylov on CPU; torch.matrix_exp @ v on GPU)

across the grid (D, N) ∈ {16, 32, 64, 128}² → matrix dim n = D·N (256 … 16 384).

Runs 5 backends in parallel, each ≤ 30 min:
    bench_cpu   (cpu=8 memory=32 GB)
    bench_t4    (T4)
    bench_a10g  (A10G)
    bench_l40s  (L40S)
    bench_a100  (A100-80GB)

Each writes /bench_gpu_kernels/{backend}.json on the tnwf-results volume.

Usage:
    modal run --detach modal/bench_gpu_kernels.py::main
    modal run         modal/bench_gpu_kernels.py::download --dest results/bench_gpu_kernels
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tnwf-bench-gpu-kernels"
VOLUME_NAME = "tnwf-results"
REMOTE_RES_DIR = "/results"

image_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2")
    .add_local_python_source("tnwf")
)
# Same image works for GPU — `pip install torch>=2.2` pulls the CUDA wheel
# automatically when Modal mounts a GPU.
image_gpu = image_cpu

app = modal.App(APP_NAME, image=image_cpu)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ── Bench grid ────────────────────────────────────────────────────────────

D_VALUES = [16, 32, 64, 128]
N_VALUES = [16, 32, 64, 128]
OPS = ["matmul", "full_svd", "truncated_svd", "expm_multiply"]

# Adaptive-iter targets: keep each (op, size) cell under TARGET_S wall on the
# slowest backend. Floor at MIN_ITERS to dampen noise. If the first call alone
# exceeds MAX_S_PER_CALL we record that single timing and skip the rest.
TARGET_S_PER_CELL = 5.0
MAX_S_PER_CALL = 30.0
MIN_ITERS = 3
MAX_ITERS = 30
DTYPE = "complex128"
TAU_IMAG = 0.05


# ── Core timing impl (shared CPU/GPU) ─────────────────────────────────────

def _run_bench(backend: str, device: str) -> dict:
    """Time all (op, D, N) cells on the given device. Saves JSON to volume."""
    import os
    import time

    import numpy as np
    import torch

    use_torch = device == "cuda"
    if use_torch and not torch.cuda.is_available():
        raise RuntimeError(
            f"backend={backend} expects CUDA but torch.cuda.is_available() is False"
        )

    dev_name = torch.cuda.get_device_name() if use_torch else "cpu"
    print(f"[{backend}] device={device} ({dev_name}), torch={torch.__version__}, "
          f"cuda={torch.version.cuda or 'n/a'}", flush=True)

    rng = np.random.default_rng(0)
    cells: list[dict] = []
    started = time.time()
    BUDGET_S = 1500.0  # leave 5 min slack inside the 1800 s container

    for D in D_VALUES:
        for N in N_VALUES:
            n = D * N
            elapsed = time.time() - started
            if elapsed > BUDGET_S:
                print(f"[{backend}] budget exhausted at D={D} N={N}, stopping",
                      flush=True)
                break

            # Generate test matrices once per size; reuse across ops.
            A_np = (rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            B_np = (rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            H_np = (A_np + A_np.conj().T) * 0.5    # Hermitian
            v_np = (rng.standard_normal(n)
                    + 1j * rng.standard_normal(n)).astype(np.complex128)
            v_np /= np.linalg.norm(v_np)

            if use_torch:
                A = torch.from_numpy(A_np).cuda()
                B = torch.from_numpy(B_np).cuda()
                H = torch.from_numpy(H_np).cuda()
                v = torch.from_numpy(v_np).cuda()
                tau = torch.tensor(1j * TAU_IMAG, dtype=torch.complex128).cuda()
                torch.cuda.synchronize()

            for op in OPS:
                cell = _time_cell(
                    op=op, D=D, N=N, n=n, use_torch=use_torch,
                    A_np=A_np, B_np=B_np, H_np=H_np, v_np=v_np,
                    A=A if use_torch else None,
                    B=B if use_torch else None,
                    H=H if use_torch else None,
                    v=v if use_torch else None,
                    tau=tau if use_torch else (1j * TAU_IMAG),
                )
                cells.append(cell)
                print(f"  [{backend}] op={op:14s} D={D:>3} N={N:>3} "
                      f"n={n:>5} mean={cell['mean_s']:.4f}s iters={cell['n_iters']:>2} "
                      f"{'status='+cell['status'] if cell['status'] != 'ok' else ''}",
                      flush=True)

            if use_torch:
                del A, B, H, v
                torch.cuda.empty_cache()
        else:
            continue
        break  # budget broke out of N loop, also break D loop

    out = {
        "backend": backend,
        "device": device,
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "n/a",
        "dtype": DTYPE,
        "target_s_per_cell": TARGET_S_PER_CELL,
        "min_iters": MIN_ITERS,
        "max_iters": MAX_ITERS,
        "total_time_s": time.time() - started,
        "cells": cells,
    }

    out_dir = os.path.join(REMOTE_RES_DIR, "bench_gpu_kernels")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backend}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print(f"[{backend}] wrote {out_path}  ({len(cells)} cells, "
          f"{out['total_time_s']:.0f}s)", flush=True)
    return out


def _time_cell(
    *, op: str, D: int, N: int, n: int, use_torch: bool,
    A_np, B_np, H_np, v_np, A=None, B=None, H=None, v=None, tau,
) -> dict:
    """Time a single (op, size) cell with adaptive iterations."""
    import time

    import numpy as np
    import torch
    from scipy.sparse.linalg import expm_multiply
    from scipy.linalg import svd as scipy_svd

    # Build the callable for this (op, backend) combo.
    if use_torch:
        sync = torch.cuda.synchronize
        if op == "matmul":
            def call():
                return A @ B
        elif op == "full_svd":
            def call():
                return torch.linalg.svd(A, full_matrices=True)
        elif op == "truncated_svd":
            def call():
                return torch.linalg.svd(A, full_matrices=False)
        elif op == "expm_multiply":
            def call():
                return torch.linalg.matrix_exp(tau * H) @ v
        else:
            raise ValueError(op)
    else:
        def sync():
            return None
        if op == "matmul":
            def call():
                return A_np @ B_np
        elif op == "full_svd":
            def call():
                return np.linalg.svd(A_np, full_matrices=True)
        elif op == "truncated_svd":
            def call():
                return np.linalg.svd(A_np, full_matrices=False)
        elif op == "expm_multiply":
            def call():
                return expm_multiply(tau * H_np, v_np)
        else:
            raise ValueError(op)

    # First call (serves as warmup + first timing).
    try:
        sync()
        t0 = time.perf_counter()
        _ = call()
        sync()
        t_first = time.perf_counter() - t0
    except (RuntimeError, MemoryError, np.linalg.LinAlgError) as exc:
        return {"op": op, "D": D, "N": N, "size": n, "dtype": DTYPE,
                "mean_s": float("nan"), "std_s": float("nan"), "n_iters": 0,
                "status": f"first_call_failed: {type(exc).__name__}: {str(exc)[:80]}"}

    times = [t_first]

    # If the first call alone is >MAX_S_PER_CALL, record that single measurement
    # and skip more iters — protects against multi-minute SVDs eating the
    # container budget before we can write any results.
    if t_first < MAX_S_PER_CALL:
        n_more = max(MIN_ITERS - 1, min(MAX_ITERS - 1,
                                         int(TARGET_S_PER_CELL / max(t_first, 1e-9))))
        try:
            for _ in range(n_more):
                sync()
                t0 = time.perf_counter()
                _ = call()
                sync()
                times.append(time.perf_counter() - t0)
        except (RuntimeError, MemoryError, np.linalg.LinAlgError) as exc:
            # Keep what we have; record the failure.
            arr = np.asarray(times)
            return {"op": op, "D": D, "N": N, "size": n, "dtype": DTYPE,
                    "mean_s": float(arr.mean()), "std_s": float(arr.std()),
                    "min_s": float(arr.min()), "n_iters": int(len(times)),
                    "status": f"partial_then_failed: {type(exc).__name__}: {str(exc)[:80]}"}
        status = "ok"
    else:
        status = f"slow_one_iter_only_{t_first:.1f}s"

    arr = np.asarray(times)
    return {"op": op, "D": D, "N": N, "size": n, "dtype": DTYPE,
            "mean_s": float(arr.mean()), "std_s": float(arr.std()),
            "min_s": float(arr.min()), "n_iters": int(len(times)),
            "status": status}


# ── Backend entrypoints ───────────────────────────────────────────────────

@app.function(image=image_cpu, volumes={REMOTE_RES_DIR: volume},
              cpu=8.0, memory=32768, timeout=1800)
def bench_cpu() -> str:
    out = _run_bench("cpu", "cpu")
    return f"cpu: {len(out['cells'])} cells, {out['total_time_s']:.0f}s"


@app.function(image=image_gpu, volumes={REMOTE_RES_DIR: volume},
              gpu="T4", cpu=4.0, memory=16384, timeout=1800)
def bench_t4() -> str:
    out = _run_bench("t4", "cuda")
    return f"t4: {len(out['cells'])} cells, {out['total_time_s']:.0f}s"


@app.function(image=image_gpu, volumes={REMOTE_RES_DIR: volume},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def bench_a10g() -> str:
    out = _run_bench("a10g", "cuda")
    return f"a10g: {len(out['cells'])} cells, {out['total_time_s']:.0f}s"


@app.function(image=image_gpu, volumes={REMOTE_RES_DIR: volume},
              gpu="L40S", cpu=4.0, memory=16384, timeout=1800)
def bench_l40s() -> str:
    out = _run_bench("l40s", "cuda")
    return f"l40s: {len(out['cells'])} cells, {out['total_time_s']:.0f}s"


@app.function(image=image_gpu, volumes={REMOTE_RES_DIR: volume},
              gpu="A100-80GB", cpu=4.0, memory=32768, timeout=1800)
def bench_a100() -> str:
    out = _run_bench("a100", "cuda")
    return f"a100: {len(out['cells'])} cells, {out['total_time_s']:.0f}s"


@app.local_entrypoint()
def main():
    """Spawn all 5 backends in parallel (detached-compatible)."""
    print("Spawning 5 benchmark containers (cpu + T4 + A10G + L40S + A100-80GB)")
    handles = [
        ("cpu",  bench_cpu.spawn()),
        ("t4",   bench_t4.spawn()),
        ("a10g", bench_a10g.spawn()),
        ("l40s", bench_l40s.spawn()),
        ("a100", bench_a100.spawn()),
    ]
    for name, h in handles:
        print(f"  spawned {name} → FunctionCall {h.object_id}")
    print()
    print("Waiting on results (each ≤ 30 min)...")
    for name, h in handles:
        try:
            result = h.get()
            print(f"  {name}: {result}")
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
    print("=== done — collect via `download` entrypoint ===")


# ── Volume helpers ────────────────────────────────────────────────────────

@app.function(volumes={REMOTE_RES_DIR: volume})
def _list_results() -> list:
    import os
    root = "/results/bench_gpu_kernels"
    if not os.path.isdir(root):
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs]


@app.function(volumes={REMOTE_RES_DIR: volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download(dest: str = "results/bench_gpu_kernels"):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    files = _list_results.remote()
    print(f"Downloading {len(files)} files → {dest_path}/")
    for remote_path in files:
        name = Path(remote_path).name
        local = dest_path / name
        local.write_bytes(_read_file.remote(remote_path))
        print(f"  {name}")
