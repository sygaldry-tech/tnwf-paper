"""Modal sweep: coarse-grained continuous MNIST × N × K × method.

Compares tnWF's two MPS-TDVP V-step methods (tci_tdvp1, tci_tdvp2) against the
JAM gradient-flow ceiling on continuous MNIST at three coarse-grain scales:

    grids   = [2, 4, 8]          # d = 4, 16, 64
    N_VALUES = [4, 8]            # per-axis discretization
    K_VALUES = [8, 16, 32, 64]   # Trotter step count
    METHODS  = [tci_tdvp1, tci_tdvp2, jam]
    SEEDS    = [0]

Cell count:
    TDVP cells:  3 grids × 2 N × 4 K × 2 methods  = 48
    JAM ceiling: 1 row per grid (N, K independent) = 3
                                                  ----
                                              total = 51

Stage 0 prerequisite: run `modal run modal/train_jam_mnist.py` first to
populate `/jam_checkpoints/mnist_coarse_{2,4,8}/seed0.pt` on the
`tnwf-results` volume.

Launch:
    modal run modal/mnist_coarse_NK.py
Download:
    modal run modal/mnist_coarse_NK.py::download --dest results/mnist_coarse_NK
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-mnist-coarse-NK"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2", "torchvision>=0.17",
        "matplotlib>=3.8", "scikit-learn>=1.4",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

TDVP_METHODS = ["tci_tdvp1", "tci_tdvp2"]
GRIDS = [2, 4, 8]
N_VALUES = [4, 8]
K_VALUES = [8, 16, 32, 64]
SEEDS = [0]

# Adaptive D with safety cap. D_max=32 matches the SCOREBOARD SOTA at 8×8
# (binary MNIST, D=24 K=80 → SW=0.164). D_V=16 keeps the V-MPO build tractable
# at d=64. tol=1e-3 drives SVD truncation in tdvp2.
# D_init=32 seeds the initial MPS bond at the cap, so tci_tdvp1 (fixed-bond)
# and tci_tdvp2 (adaptive up to D_max) have the same bond budget.
TDVP_KWARGS = {"D_max": 32, "D_V": 16, "tol": 1e-3, "D_init": 32,
               "n_sweeps": 1, "n_sweeps_cross": 2}
TDVP1_KWARGS = {"D_max": 32, "D_V": 16, "D_init": 32,
                "n_sweeps": 1, "n_sweeps_cross": 2}


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=28800,
    volumes={"/results": volume},
)
def run_tdvp_cell(method: str, grid: int, N: int, K: int, seed: int) -> str:
    """One TDVP cell: (method, grid, N, K, seed). 48 such cells."""
    from tnwf.pipelines.run_evolution import run

    dataset = f"mnist_coarse_{grid}"
    out_dir = Path(f"/results/mnist_coarse_NK/{method}/grid{grid}_N{N}_K{K}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_g{grid}_N{N}_K{K}_seed{seed}_skip"

    jam_ckpt = f"/results/jam_checkpoints/{dataset}/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_g{grid}_N{N}_K{K}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run modal/train_jam_mnist.py first.")

    method_kwargs = TDVP1_KWARGS if method == "tci_tdvp1" else TDVP_KWARGS
    run(
        method=method, dataset=dataset, jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=2000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    return f"{method}_g{grid}_N{N}_K{K}_seed{seed}_done"


@app.function(
    cpu=4.0,
    memory=16384,
    timeout=3600,
    volumes={"/results": volume},
)
def run_jam_ceiling(grid: int, seed: int) -> str:
    """JAM AM-only ceiling for one grid. Independent of N, K — saved once per
    grid at the canonical N=8, K=64 cell so the aggregator picks it up
    consistently."""
    from tnwf.pipelines.run_evolution import run

    dataset = f"mnist_coarse_{grid}"
    N, K = 8, 64                      # canonical: use the same (N, K) as the
                                       # finest TDVP cell so plots align
    out_dir = Path(f"/results/mnist_coarse_NK/jam/grid{grid}_N{N}_K{K}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"jam_g{grid}_seed{seed}_skip"

    jam_ckpt = f"/results/jam_checkpoints/{dataset}/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"jam_g{grid}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run modal/train_jam_mnist.py first.")

    run(
        method="jam", dataset=dataset, jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=2000,
        save=True, out_dir=str(out_dir),
    )
    return f"jam_g{grid}_seed{seed}_done"


@app.local_entrypoint()
def main():
    tdvp_cells = [
        (m, g, n, k, s)
        for m in TDVP_METHODS for g in GRIDS
        for n in N_VALUES for k in K_VALUES for s in SEEDS
    ]
    jam_cells = [(g, s) for g in GRIDS for s in SEEDS]
    print(f"launching {len(tdvp_cells) + len(jam_cells)} containers "
          f"({len(tdvp_cells)} TDVP + {len(jam_cells)} JAM)")
    done = ok = err = 0
    for result in run_tdvp_cell.starmap(tdvp_cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    for result in run_jam_ceiling.starmap(jam_cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  JAM CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== {ok} ok, {err} failed (total {done}) ===")


@app.local_entrypoint()
def tdvp_only():
    """Just the 48 TDVP cells (JAM ceilings already exist)."""
    cells = [
        (m, g, n, k, s)
        for m in TDVP_METHODS for g in GRIDS
        for n in N_VALUES for k in K_VALUES for s in SEEDS
    ]
    print(f"launching {len(cells)} TDVP containers")
    done = ok = err = 0
    for result in run_tdvp_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== {ok} ok, {err} failed (total {done}) ===")


def _run_starmap(cells: list[tuple], label: str) -> None:
    print(f"launching {len(cells)} {label} cells")
    done = ok = err = 0
    for result in run_tdvp_cell.starmap(cells, return_exceptions=True):
        if isinstance(result, BaseException):
            print(f"  CELL FAILED: {type(result).__name__}: {result}")
            err += 1
        else:
            print(result); ok += 1
        done += 1
    print(f"=== {label}: {ok} ok, {err} failed (total {done}) ===")


@app.local_entrypoint()
def g2_extended():
    """High-N high-K 2x2 follow-up sweep — does TDVP converge to JAM ceiling?"""
    cells = [
        (m, 2, n, k, 0)
        for m in TDVP_METHODS
        for n in [16, 32, 64]
        for k in [8, 16, 32, 64, 128]
    ]
    _run_starmap(cells, "g=2 extended")


@app.local_entrypoint()
def g4_extended():
    """High-N high-K 4x4 follow-up sweep (full mirror of g2_extended)."""
    cells = [
        (m, 4, n, k, 0)
        for m in TDVP_METHODS
        for n in [16, 32, 64]
        for k in [8, 16, 32, 64, 128]
    ]
    _run_starmap(cells, "g=4 extended")


@app.local_entrypoint()
def g8_extended():
    """8x8 probe — only tdvp1 at N=16 K∈{8,16,32}. Anything more at d=64 will
    almost certainly hit the 8h container timeout; this is a 3-cell sanity
    check that tdvp1 still produces sensible SW at higher N=16 for d=64."""
    cells = [("tci_tdvp1", 8, 16, k, 0) for k in [8, 16, 32]]
    _run_starmap(cells, "g=8 probe")


@app.local_entrypoint()
def jam_only():
    """Just the 3 JAM-ceiling cells."""
    cells = [(g, s) for g in GRIDS for s in SEEDS]
    print(f"launching {len(cells)} JAM-ceiling containers")
    done = 0
    for result in run_jam_ceiling.starmap(cells):
        print(result); done += 1
    print(f"=== ALL {done} JAM jobs complete ===")


# ── Volume helpers ────────────────────────────────────────────────────────

@app.function(volumes={"/results": volume})
def _list_results() -> list[str]:
    import os
    root = "/results/mnist_coarse_NK"
    if not os.path.isdir(root):
        return []
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs]


@app.function(volumes={"/results": volume})
def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


@app.local_entrypoint()
def download(dest: str = "results/mnist_coarse_NK"):
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    files = _list_results.remote()
    print(f"Downloading {len(files)} files → {dest_path}/", flush=True)
    for remote_path in files:
        rel = Path(remote_path).relative_to("/results/mnist_coarse_NK")
        local = dest_path / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(_read_file.remote(remote_path))
        print(f"  {rel}")
