"""Two GPU-speedup paths benchmarked side-by-side.

Path 1 (complex64): same 4 hot kernels as `bench_gpu_kernels.py` but the GPU
side runs in complex64. CPU stays complex128 to keep the speedup denominator
honest. The question: do Tensor-Core / cuSOLVER fast paths engage at fp32 and
unlock the SVD/matmul speedup that complex128 left on the table?

Path 2 (Krylov on GPU): a 30-dim Lanczos implementation written in pure torch
on the GPU. Krylov is matvec-dominant — and matvec (matmul) is the one op
where torch.cuda already beats CPU (≈ 4× on L40S in complex128). We compare
against scipy's CPU `expm_multiply` and against torch's dense `matrix_exp` ·
v (which lost). Run in both complex128 and complex64.

5 backends × 2 paths runs as parallel Modal jobs:

    bench_cpu_paths   — CPU baseline for both paths
    bench_{t4,a10g,l40s,a100}_paths  — GPU backends

Each job ≤ 30 min. Output:
  /bench_gpu_paths/{backend}.json   on the tnwf-results volume

Usage:
    modal run --detach modal/bench_gpu_paths.py::main
    modal run         modal/bench_gpu_paths.py::download --dest results/bench_gpu_paths
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tnwf-bench-gpu-paths"
VOLUME_NAME = "tnwf-results"
REMOTE_RES_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.12", "torch>=2.2")
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ── Bench grid ────────────────────────────────────────────────────────────

D_VALUES = [16, 32, 64, 128]
N_VALUES = [16, 32, 64, 128]

# Path 1 ops (same as bench_gpu_kernels.py)
PATH1_OPS = ["matmul", "full_svd", "truncated_svd", "expm_dense"]
# Path 2 ops: expm via different methods at two dtypes
PATH2_OPS = ["expm_scipy_cpu_c128", "expm_torch_dense_c128",
             "expm_torch_dense_c64",  "expm_torch_krylov_c128",
             "expm_torch_krylov_c64"]

TARGET_S_PER_CELL = 5.0
MAX_S_PER_CALL = 30.0
MIN_ITERS = 3
MAX_ITERS = 30
TAU_IMAG = 0.05
KRYLOV_M = 30                  # Lanczos subspace size


# ── Krylov-Lanczos implementation in torch ────────────────────────────────

def _krylov_expm_torch(H, v, tau_complex, m: int = KRYLOV_M, tol: float = 1e-12):
    """Compute exp(tau · H) · v via Lanczos (Hermitian H), entirely in torch.

    Runs on whatever device H lives on. Returns a torch.Tensor on the same
    device with the same complex dtype as H.
    """
    import torch
    dev = H.device
    cdtype = H.dtype
    n = v.shape[0]
    beta = torch.linalg.vector_norm(v).real
    if float(beta) == 0.0:
        return v.clone()

    Q = torch.zeros((n, m + 1), dtype=cdtype, device=dev)
    alphas = torch.zeros(m, dtype=torch.float64, device=dev)
    betas = torch.zeros(m, dtype=torch.float64, device=dev)
    Q[:, 0] = v / beta

    m_eff = m
    for j in range(m):
        w = H @ Q[:, j]
        a = torch.vdot(Q[:, j], w).real
        alphas[j] = a
        w = w - a * Q[:, j]
        if j > 0:
            w = w - betas[j - 1] * Q[:, j - 1]
        b = torch.linalg.vector_norm(w).real
        if float(b) < tol:
            m_eff = j + 1
            break
        if j < m - 1:
            betas[j] = b
            Q[:, j + 1] = w / b
    # Build the tridiagonal T (alphas, betas) and exponentiate it.
    T = (torch.diag(alphas[:m_eff].to(cdtype))
         + torch.diag(betas[:m_eff - 1].to(cdtype), 1)
         + torch.diag(betas[:m_eff - 1].to(cdtype), -1))
    expT = torch.linalg.matrix_exp(tau_complex * T)
    return beta * (Q[:, :m_eff] @ expT[:, 0])


# ── Per-cell timer ────────────────────────────────────────────────────────

def _time_call(call, sync, n_iters_first_est: bool):
    """Run one call, then 2-29 more (target ≥5 s). Bails after the first call
    if it took > MAX_S_PER_CALL.

    Returns (mean_s, std_s, n_iters, status_str).
    """
    import time
    import numpy as np

    try:
        sync()
        t0 = time.perf_counter()
        _ = call()
        sync()
        t_first = time.perf_counter() - t0
    except Exception as exc:
        return float("nan"), float("nan"), 0, f"first_failed:{type(exc).__name__}:{str(exc)[:60]}"

    times = [t_first]
    if t_first < MAX_S_PER_CALL:
        n_more = max(MIN_ITERS - 1, min(MAX_ITERS - 1, int(TARGET_S_PER_CELL / max(t_first, 1e-9))))
        try:
            for _ in range(n_more):
                sync()
                t0 = time.perf_counter()
                _ = call()
                sync()
                times.append(time.perf_counter() - t0)
        except Exception as exc:
            arr = np.asarray(times)
            return float(arr.mean()), float(arr.std()), len(times), f"partial_failed:{type(exc).__name__}:{str(exc)[:60]}"
        status = "ok"
    else:
        status = f"slow_one_iter_only_{t_first:.1f}s"

    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std()), len(times), status


# ── Path 1 + Path 2 driver ────────────────────────────────────────────────

def _run_paths(backend: str, device: str) -> dict:
    import os
    import time

    import numpy as np
    import torch
    from scipy.sparse.linalg import expm_multiply as scipy_expm_multiply

    use_torch = device == "cuda"
    if use_torch and not torch.cuda.is_available():
        raise RuntimeError(f"backend={backend} expects CUDA but unavailable")

    dev_name = torch.cuda.get_device_name() if use_torch else "cpu"
    print(f"[{backend}] device={device} ({dev_name})  torch={torch.__version__}"
          f"  cuda={torch.version.cuda or 'n/a'}", flush=True)

    BUDGET_S = 1500.0
    started = time.time()
    rng = np.random.default_rng(0)
    cells: list[dict] = []

    for D in D_VALUES:
        for N in N_VALUES:
            n = D * N
            if time.time() - started > BUDGET_S:
                print(f"[{backend}] budget exhausted at D={D} N={N}", flush=True)
                break

            # Test data — make once per size, reuse across all ops.
            A128 = (rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            B128 = (rng.standard_normal((n, n))
                    + 1j * rng.standard_normal((n, n))).astype(np.complex128)
            H128 = (A128 + A128.conj().T) * 0.5
            v128 = (rng.standard_normal(n)
                    + 1j * rng.standard_normal(n)).astype(np.complex128)
            v128 /= np.linalg.norm(v128)
            tau128 = 1j * TAU_IMAG

            if use_torch:
                A_t128 = torch.from_numpy(A128).cuda()
                B_t128 = torch.from_numpy(B128).cuda()
                H_t128 = torch.from_numpy(H128).cuda()
                v_t128 = torch.from_numpy(v128).cuda()
                tau_t128 = torch.tensor(tau128, dtype=torch.complex128).cuda()
                # Single-precision copies
                A_t64 = A_t128.to(torch.complex64)
                B_t64 = B_t128.to(torch.complex64)
                H_t64 = H_t128.to(torch.complex64)
                v_t64 = v_t128.to(torch.complex64)
                tau_t64 = torch.tensor(tau128, dtype=torch.complex64).cuda()
                torch.cuda.synchronize()
                sync = torch.cuda.synchronize
            else:
                def sync():
                    return None

            # ─ Path 1 cells ─────────────────────────────────────────────
            # Path 1 op set runs at two dtypes on GPU (c128 + c64), and at
            # c128 only on CPU (no point timing fp32 numpy — same speed).
            for op in PATH1_OPS:
                # CPU c128 baseline
                if not use_torch:
                    if op == "matmul":
                        call = lambda: A128 @ B128
                    elif op == "full_svd":
                        call = lambda: np.linalg.svd(A128, full_matrices=True)
                    elif op == "truncated_svd":
                        call = lambda: np.linalg.svd(A128, full_matrices=False)
                    elif op == "expm_dense":
                        call = lambda: scipy_expm_multiply(tau128 * H128, v128)
                    mean, std, ni, status = _time_call(call, sync, True)
                    cells.append({"path": 1, "op": op, "dtype": "complex128",
                                  "device": "cpu", "D": D, "N": N, "size": n,
                                  "mean_s": mean, "std_s": std, "n_iters": ni,
                                  "status": status})
                    print(f"  [{backend}] P1 {op:14s} c128 D={D:>3} N={N:>3} "
                          f"n={n:>5} mean={mean:.4f}s status={status[:20]}", flush=True)
                    continue
                # GPU: time at both c128 and c64
                for dtype_tag, A, B, H, v, tau in [
                    ("complex128", A_t128, B_t128, H_t128, v_t128, tau_t128),
                    ("complex64",  A_t64,  B_t64,  H_t64,  v_t64,  tau_t64),
                ]:
                    if op == "matmul":
                        call = lambda A=A, B=B: A @ B
                    elif op == "full_svd":
                        call = lambda A=A: torch.linalg.svd(A, full_matrices=True)
                    elif op == "truncated_svd":
                        call = lambda A=A: torch.linalg.svd(A, full_matrices=False)
                    elif op == "expm_dense":
                        call = lambda H=H, v=v, tau=tau: torch.linalg.matrix_exp(tau * H) @ v
                    mean, std, ni, status = _time_call(call, sync, True)
                    cells.append({"path": 1, "op": op, "dtype": dtype_tag,
                                  "device": "cuda", "D": D, "N": N, "size": n,
                                  "mean_s": mean, "std_s": std, "n_iters": ni,
                                  "status": status})
                    print(f"  [{backend}] P1 {op:14s} {dtype_tag:>10} "
                          f"D={D:>3} N={N:>3} n={n:>5} mean={mean:.4f}s "
                          f"status={status[:20]}", flush=True)

            # ─ Path 2 cells ─────────────────────────────────────────────
            # Path 2: expm via different methods. CPU times scipy_cpu_c128.
            # GPU times the rest (torch_dense at c128/c64, krylov at c128/c64).
            if not use_torch:
                call = lambda: scipy_expm_multiply(tau128 * H128, v128)
                mean, std, ni, status = _time_call(call, sync, True)
                cells.append({"path": 2, "op": "expm_scipy_cpu_c128",
                              "dtype": "complex128", "device": "cpu",
                              "D": D, "N": N, "size": n,
                              "mean_s": mean, "std_s": std, "n_iters": ni,
                              "status": status})
                print(f"  [{backend}] P2 expm_scipy_cpu  c128 D={D:>3} N={N:>3} "
                      f"n={n:>5} mean={mean:.4f}s status={status[:20]}", flush=True)
            else:
                # GPU: dense matrix_exp + krylov in both dtypes
                gpu_methods = [
                    ("expm_torch_dense", "complex128", H_t128, v_t128, tau_t128),
                    ("expm_torch_dense", "complex64",  H_t64,  v_t64,  tau_t64),
                    ("expm_torch_krylov","complex128", H_t128, v_t128, tau_t128),
                    ("expm_torch_krylov","complex64",  H_t64,  v_t64,  tau_t64),
                ]
                for op_name, dtype_tag, H, v, tau in gpu_methods:
                    if op_name == "expm_torch_dense":
                        call = lambda H=H, v=v, tau=tau: torch.linalg.matrix_exp(tau * H) @ v
                    elif op_name == "expm_torch_krylov":
                        call = lambda H=H, v=v, tau=tau: _krylov_expm_torch(H, v, tau)
                    mean, std, ni, status = _time_call(call, sync, True)
                    cells.append({"path": 2, "op": op_name, "dtype": dtype_tag,
                                  "device": "cuda", "D": D, "N": N, "size": n,
                                  "mean_s": mean, "std_s": std, "n_iters": ni,
                                  "status": status})
                    print(f"  [{backend}] P2 {op_name:18s} {dtype_tag:>10} "
                          f"D={D:>3} N={N:>3} n={n:>5} mean={mean:.4f}s "
                          f"status={status[:20]}", flush=True)

            if use_torch:
                del A_t128, B_t128, H_t128, v_t128, tau_t128
                del A_t64, B_t64, H_t64, v_t64, tau_t64
                torch.cuda.empty_cache()
        else:
            continue
        break

    out = {
        "backend": backend,
        "device": device,
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "n/a",
        "krylov_m": KRYLOV_M,
        "total_time_s": time.time() - started,
        "cells": cells,
    }
    out_dir = os.path.join(REMOTE_RES_DIR, "bench_gpu_paths")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backend}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print(f"[{backend}] wrote {out_path} ({len(cells)} cells, "
          f"{out['total_time_s']:.0f}s)", flush=True)
    return out


# ── Backend entrypoints ───────────────────────────────────────────────────

@app.function(volumes={REMOTE_RES_DIR: volume},
              cpu=8.0, memory=32768, timeout=1800)
def bench_cpu_paths() -> str:
    return f"cpu: {len(_run_paths('cpu', 'cpu')['cells'])} cells"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="T4", cpu=4.0, memory=16384, timeout=1800)
def bench_t4_paths() -> str:
    return f"t4: {len(_run_paths('t4', 'cuda')['cells'])} cells"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def bench_a10g_paths() -> str:
    return f"a10g: {len(_run_paths('a10g', 'cuda')['cells'])} cells"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="L40S", cpu=4.0, memory=16384, timeout=1800)
def bench_l40s_paths() -> str:
    return f"l40s: {len(_run_paths('l40s', 'cuda')['cells'])} cells"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="A100-80GB", cpu=4.0, memory=32768, timeout=1800)
def bench_a100_paths() -> str:
    return f"a100: {len(_run_paths('a100', 'cuda')['cells'])} cells"


@app.local_entrypoint()
def main():
    print("Spawning Path 1+2 benchmarks on 5 backends...")
    handles = [
        ("cpu",  bench_cpu_paths.spawn()),
        ("t4",   bench_t4_paths.spawn()),
        ("a10g", bench_a10g_paths.spawn()),
        ("l40s", bench_l40s_paths.spawn()),
        ("a100", bench_a100_paths.spawn()),
    ]
    for name, h in handles:
        print(f"  spawned {name} → {h.object_id}")
    for name, h in handles:
        try:
            print(f"  {name}: {h.get()}")
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")
    print("=== done — collect via `download` entrypoint ===")


# ── Volume helpers ────────────────────────────────────────────────────────

@app.function(volumes={REMOTE_RES_DIR: volume})
def _list_results() -> list:
    import os
    root = "/results/bench_gpu_paths"
    if not os.path.isdir(root):
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs]


@app.function(volumes={REMOTE_RES_DIR: volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download(dest: str = "results/bench_gpu_paths"):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    files = _list_results.remote()
    print(f"Downloading {len(files)} files → {dest_path}/")
    for remote_path in files:
        name = Path(remote_path).name
        local = dest_path / name
        local.write_bytes(_read_file.remote(remote_path))
        print(f"  {name}")
