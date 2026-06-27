"""Modal sweep: swiss_roll N × K phase diagram.

For each method, sweep grid resolution N and Trotter step count K to see
where accuracy plateaus.

  - Higher N: less grid quantisation. SW expected to drop monotonically.
  - Higher K: tighter product-formula error O(Δt²). Should saturate once
    that's below grid/MPS error.

Grid: 6 methods × 4 N × 5 K × 5 seeds = 600 containers (under 1000 cap).
D_max = min(N, 32) so MPS retains full-bond expressivity at small N and
compresses at large N.

Launch:
    modal run modal/swiss_roll_NK.py
Download:
    modal volume get tnwf-results /swiss_roll_NK/* ./results/swiss_roll_NK/
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-swiss-roll-NK"
VOLUME_NAME = "tnwf-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26", "scipy>=1.12", "torch>=2.2",
        "matplotlib>=3.8", "scikit-learn>=1.4",
    )
    .add_local_python_source("tnwf")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

METHODS = [
    "jam", "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]
N_VALUES = [16, 32, 64, 128]
K_VALUES = [4, 8, 16, 32, 64]
SEEDS = list(range(10))


@app.function(
    cpu=2.0,
    memory=8192,
    timeout=3600,
    volumes={"/results": volume},
)
def run_one(method: str, N: int, K: int, seed: int) -> str:
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/swiss_roll_NK/{method}/N{N}_K{K}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed{seed}.npz"
    if out_file.exists():
        return f"{method}_N{N}_K{K}_seed{seed}_skip"

    jam_ckpt = f"/results/jam_checkpoints/swiss_roll_2d/seed{seed}.pt"
    if not Path(jam_ckpt).exists():
        return (f"{method}_N{N}_K{K}_seed{seed}_FAIL: missing JAM ckpt "
                f"({jam_ckpt}). Run `modal run modal/train_jam.py` first.")

    D_max = 32                                          # fixed across N
    method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                     "D_init": D_max}
    run(
        method=method, dataset="swiss_roll_2d", jam_ckpt=jam_ckpt,
        seed=seed, N=N, K=K, n_samples=2000,
        save=True, out_dir=str(out_dir), method_kwargs=method_kwargs,
    )
    # rely on implicit commit at exit (avoids server contention)
    return f"{method}_N{N}_K{K}_seed{seed}_done"


@app.local_entrypoint()
def main():
    grid = [(m, n, k, s)
            for m in METHODS for n in N_VALUES for k in K_VALUES for s in SEEDS]
    print(f"launching {len(grid)} containers "
          f"({len(METHODS)} methods × {len(N_VALUES)} N × "
          f"{len(K_VALUES)} K × {len(SEEDS)} seeds)")
    done = 0
    for result in run_one.starmap(grid):
        print(result)
        done += 1
    print(f"=== ALL {done} jobs complete ===")
