"""End-to-end TDVP-on-GPU benchmark.

For a representative MNIST cell, time `run(method='tci_tdvp2', ...)` on
five backends:

    bench_cpu_e2e   — numpy reference (`device='cpu'`)
    bench_t4_e2e    — torch CUDA on T4
    bench_a10g_e2e  — torch CUDA on A10G
    bench_l40s_e2e  — torch CUDA on L40S
    bench_a100_e2e  — torch CUDA on A100-80GB

Cell: mnist_coarse_2 N=8 K=16 D=32 D_V=16 — modest enough to fit CPU
baseline in 30 min, large enough to make the GPU win meaningful. After
proving speedup we follow with a real high-N MNIST sweep on GPU.

Output: /bench_tdvp_torch/{backend}.json on tnwf-results volume.

Usage:
    modal run --detach modal/bench_tdvp_torch.py::main
    modal run         modal/bench_tdvp_torch.py::download --dest results/bench_tdvp_torch
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "tnwf-bench-tdvp-torch"
VOLUME_NAME = "tnwf-results"
REMOTE_RES_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2",
        "torchvision>=0.17", "matplotlib>=3.8", "scikit-learn>=1.4",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

BENCH_CELL = {
    "dataset": "mnist_coarse_2",
    "N": 64,
    "K": 8,
    "n_samples": 500,
    "method_kwargs": {"D_max": 64, "D_V": 32, "tol": 1e-3, "D_init": 64,
                      "n_sweeps": 1, "n_sweeps_cross": 2},
}


def _run_one_e2e(backend: str, device: str) -> dict:
    """Run the BENCH_CELL with the given device. Return timings + SW."""
    import os
    import time

    import torch

    print(f"[{backend}] device={device}, torch={torch.__version__}, "
          f"cuda={torch.version.cuda or 'n/a'}", flush=True)
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(f"backend={backend} expects CUDA")
        print(f"[{backend}] GPU = {torch.cuda.get_device_name()}", flush=True)

    from tnwf.pipelines.run_evolution import run

    jam_ckpt = "/results/jam_checkpoints/mnist_coarse_2/seed0.pt"
    if not os.path.exists(jam_ckpt):
        raise RuntimeError(f"JAM ckpt missing: {jam_ckpt}")

    kw = dict(BENCH_CELL["method_kwargs"])
    kw["device"] = device

    t0 = time.time()
    out = run(
        method="tci_tdvp2", dataset=BENCH_CELL["dataset"], jam_ckpt=jam_ckpt,
        seed=0, N=BENCH_CELL["N"], K=BENCH_CELL["K"],
        n_samples=BENCH_CELL["n_samples"], save=False,
        method_kwargs=kw,
    )
    wall = time.time() - t0

    result = {
        "backend": backend, "device": device,
        "device_name": (torch.cuda.get_device_name() if device != "cpu" else "cpu"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "n/a",
        "cell": {**BENCH_CELL, "method_kwargs": kw},
        "wall_time_s": wall,
        "tdvp_total_time_s": float(out["total_time"]),
        "sw_final": float(out["sw"][-1]),
        "sw_best": float(out["sw"].min()),
        "sw_best_step": int(out["sw"].argmin()),
        "sw_per_step": [float(x) for x in out["sw"]],
        "chi_max_final": int(out["chi_max"][-1]),
        "chi_max_overall": int(out["chi_max"].max()),
        "chi_per_step": [int(x) for x in out["chi_max"]],
        "step_times": [float(x) for x in out["step_times"]],
    }

    out_dir = os.path.join(REMOTE_RES_DIR, "bench_tdvp_torch")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backend}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(f"[{backend}] wall={wall:.1f}s  sw_final={result['sw_final']:.4f}  "
          f"chi={result['chi_max_overall']}  → wrote {out_path}", flush=True)
    return result


# ── Backend entrypoints ───────────────────────────────────────────────────

@app.function(volumes={REMOTE_RES_DIR: volume},
              cpu=8.0, memory=32768, timeout=7200)
def bench_cpu_e2e() -> str:
    r = _run_one_e2e("cpu", "cpu")
    return f"cpu: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="T4", cpu=4.0, memory=16384, timeout=7200)
def bench_t4_e2e() -> str:
    r = _run_one_e2e("t4", "cuda")
    return f"t4: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="A10G", cpu=4.0, memory=16384, timeout=7200)
def bench_a10g_e2e() -> str:
    r = _run_one_e2e("a10g", "cuda")
    return f"a10g: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="L40S", cpu=4.0, memory=16384, timeout=7200)
def bench_l40s_e2e() -> str:
    r = _run_one_e2e("l40s", "cuda")
    return f"l40s: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="A100-80GB", cpu=4.0, memory=32768, timeout=7200)
def bench_a100_e2e() -> str:
    r = _run_one_e2e("a100", "cuda")
    return f"a100: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.local_entrypoint()
def main():
    print(f"End-to-end TDVP-torch bench, cell = {BENCH_CELL}")
    print("Spawning 5 backends...")
    handles = [
        ("cpu",  bench_cpu_e2e.spawn()),
        ("t4",   bench_t4_e2e.spawn()),
        ("a10g", bench_a10g_e2e.spawn()),
        ("l40s", bench_l40s_e2e.spawn()),
        ("a100", bench_a100_e2e.spawn()),
    ]
    for name, h in handles:
        print(f"  spawned {name} → {h.object_id}")
    for name, h in handles:
        try:
            print(f"  {name}: {h.get()}")
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")


@app.local_entrypoint()
def a100_only():
    """Diagnostic-only run on A100 with per-step sw saved (for divergence
    investigation). One container, no L40S/A10G/T4."""
    print(f"A100-only diagnostic bench, cell = {BENCH_CELL}")
    h = bench_a100_e2e.spawn()
    print(f"  spawned a100 → {h.object_id}")
    try:
        print(f"  a100: {h.get()}")
    except Exception as exc:
        print(f"  a100: FAILED {type(exc).__name__}: {exc}")


@app.local_entrypoint()
def gpu_only():
    """N=64 cell: skip CPU (too slow) and T4 (OOMs)."""
    print(f"End-to-end TDVP-torch bench (GPU-only), cell = {BENCH_CELL}")
    handles = [
        ("a10g", bench_a10g_e2e.spawn()),
        ("l40s", bench_l40s_e2e.spawn()),
        ("a100", bench_a100_e2e.spawn()),
    ]
    for name, h in handles:
        print(f"  spawned {name} → {h.object_id}")
    for name, h in handles:
        try:
            print(f"  {name}: {h.get()}")
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")


# ── g=4 N=32 K=32 at D=24 (D=64 didn't fit; D=24 was SCOREBOARD SOTA) ─────

BENCH_CELL_G4 = {
    "dataset": "mnist_coarse_4",
    "N": 32,
    "K": 32,
    "n_samples": 500,
    "method_kwargs": {"D_max": 24, "D_V": 16, "tol": 1e-3, "D_init": 24,
                      "n_sweeps": 1, "n_sweeps_cross": 2},
}

# Debug cell: same N=64 K=8 cell but D=16 (8x smaller matrices). Tests
# whether the SW divergence is matrix-size-dependent or N-dependent.
BENCH_CELL_DEBUG = {
    "dataset": "mnist_coarse_2",
    "N": 64,
    "K": 8,
    "n_samples": 500,
    "method_kwargs": {"D_max": 16, "D_V": 8, "tol": 1e-3, "D_init": 16,
                      "n_sweeps": 1, "n_sweeps_cross": 2},
}


def _run_debug_e2e(backend: str, device: str) -> dict:
    import os, time, torch
    from tnwf.pipelines.run_evolution import run

    jam_ckpt = "/results/jam_checkpoints/mnist_coarse_2/seed0.pt"
    if not os.path.exists(jam_ckpt):
        raise RuntimeError(f"JAM ckpt missing: {jam_ckpt}")

    kw = dict(BENCH_CELL_DEBUG["method_kwargs"])
    kw["device"] = device

    t0 = time.time()
    out = run(method="tci_tdvp2", dataset=BENCH_CELL_DEBUG["dataset"],
              jam_ckpt=jam_ckpt, seed=0,
              N=BENCH_CELL_DEBUG["N"], K=BENCH_CELL_DEBUG["K"],
              n_samples=BENCH_CELL_DEBUG["n_samples"], save=False,
              method_kwargs=kw)
    wall = time.time() - t0
    result = {
        "backend": backend, "device": device,
        "device_name": (torch.cuda.get_device_name() if device != "cpu" else "cpu"),
        "cell": {**BENCH_CELL_DEBUG, "method_kwargs": kw},
        "wall_time_s": wall,
        "sw_final": float(out["sw"][-1]),
        "sw_best": float(out["sw"].min()),
        "chi_max_overall": int(out["chi_max"].max()),
    }
    out_dir = os.path.join(REMOTE_RES_DIR, "bench_tdvp_torch_debug")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{backend}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(f"[{backend}] wall={wall:.1f}s sw={result['sw_final']:.4f}", flush=True)
    return result


@app.function(volumes={REMOTE_RES_DIR: volume},
              cpu=8.0, memory=32768, timeout=7200)
def bench_debug_cpu() -> str:
    r = _run_debug_e2e("cpu", "cpu")
    return f"cpu: sw={r['sw_final']:.4f}, t={r['wall_time_s']:.1f}s"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="L40S", cpu=4.0, memory=16384, timeout=7200)
def bench_debug_l40s() -> str:
    r = _run_debug_e2e("l40s", "cuda")
    return f"l40s: sw={r['sw_final']:.4f}, t={r['wall_time_s']:.1f}s"


@app.local_entrypoint()
def debug_n64_smallD():
    """Diagnose N=64 GPU SW divergence: run at D=16 on CPU + L40S."""
    print(f"Debug bench: {BENCH_CELL_DEBUG}")
    h_cpu = bench_debug_cpu.spawn()
    h_gpu = bench_debug_l40s.spawn()
    print(f"  cpu spawned → {h_cpu.object_id}")
    print(f"  l40s spawned → {h_gpu.object_id}")
    print(f"  cpu: {h_cpu.get()}")
    print(f"  l40s: {h_gpu.get()}")


def _run_g4_e2e(backend: str, device: str) -> dict:
    """Same as _run_one_e2e but uses BENCH_CELL_G4 and mnist_coarse_4 ckpt."""
    import os
    import time
    import torch

    print(f"[{backend}] device={device}, torch={torch.__version__}", flush=True)
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(f"backend={backend} expects CUDA")
        print(f"[{backend}] GPU = {torch.cuda.get_device_name()}", flush=True)

    from tnwf.pipelines.run_evolution import run

    jam_ckpt = "/results/jam_checkpoints/mnist_coarse_4/seed0.pt"
    if not os.path.exists(jam_ckpt):
        raise RuntimeError(f"JAM ckpt missing: {jam_ckpt}")

    kw = dict(BENCH_CELL_G4["method_kwargs"])
    kw["device"] = device

    t0 = time.time()
    out = run(
        method="tci_tdvp2", dataset=BENCH_CELL_G4["dataset"], jam_ckpt=jam_ckpt,
        seed=0, N=BENCH_CELL_G4["N"], K=BENCH_CELL_G4["K"],
        n_samples=BENCH_CELL_G4["n_samples"], save=False,
        method_kwargs=kw,
    )
    wall = time.time() - t0

    result = {
        "backend": backend, "device": device,
        "device_name": (torch.cuda.get_device_name() if device != "cpu" else "cpu"),
        "torch_version": torch.__version__,
        "cell": {**BENCH_CELL_G4, "method_kwargs": kw},
        "wall_time_s": wall,
        "sw_final": float(out["sw"][-1]),
        "sw_best": float(out["sw"].min()),
        "chi_max_final": int(out["chi_max"][-1]),
        "chi_max_overall": int(out["chi_max"].max()),
    }
    out_dir = os.path.join(REMOTE_RES_DIR, "bench_tdvp_torch_g4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{backend}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    volume.commit()
    print(f"[{backend}] wall={wall:.1f}s sw={result['sw_final']:.4f}", flush=True)
    return result


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="L40S", cpu=4.0, memory=16384, timeout=7200)
def bench_g4_l40s() -> str:
    r = _run_g4_e2e("l40s", "cuda")
    return f"l40s: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.function(volumes={REMOTE_RES_DIR: volume},
              gpu="A100-80GB", cpu=4.0, memory=32768, timeout=7200)
def bench_g4_a100() -> str:
    r = _run_g4_e2e("a100", "cuda")
    return f"a100: {r['wall_time_s']:.1f}s, sw={r['sw_final']:.4f}"


@app.local_entrypoint()
def g4_gpu():
    """g=4 N=32 K=128 on L40S + A100. ~30-90 min each."""
    print(f"g=4 GPU bench, cell = {BENCH_CELL_G4}")
    handles = [
        ("l40s", bench_g4_l40s.spawn()),
        ("a100", bench_g4_a100.spawn()),
    ]
    for name, h in handles:
        print(f"  spawned {name} → {h.object_id}")
    for name, h in handles:
        try:
            print(f"  {name}: {h.get()}")
        except Exception as exc:
            print(f"  {name}: FAILED {type(exc).__name__}: {exc}")


# ── Volume helpers ────────────────────────────────────────────────────────

@app.function(volumes={REMOTE_RES_DIR: volume})
def _list_results() -> list:
    import os
    root = "/results/bench_tdvp_torch"
    if not os.path.isdir(root):
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs]


@app.function(volumes={REMOTE_RES_DIR: volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download(dest: str = "results/bench_tdvp_torch"):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    files = _list_results.remote()
    print(f"Downloading {len(files)} files → {dest_path}/")
    for remote_path in files:
        name = Path(remote_path).name
        local = dest_path / name
        local.write_bytes(_read_file.remote(remote_path))
        print(f"  {name}")
