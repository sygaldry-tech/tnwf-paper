"""Modal sweep: TCI+TDVP1 scaling to d=10,12,16 (analytic V, no JAM).

Grid = d ∈ {10,12,16} × K ∈ {64,128} × D ∈ {32,48,64}  (single seed; the
analytic flow is deterministic, accuracy CIs come from a bootstrap over the
final Born samples). Tests three questions at once:
  - dimension ladder (does TCI+TDVP1 reach d=16 at fixed resolved grid N=32?)
  - rank sweep D (chi saturates at 32 for d>=10 -> need D >~ 2d = #modes)
  - Trotter phase K (per-step phase ~ N*sqrt(d/K); d=16 may need K=128)

Each cell is checkpointed to the Volume so long (D=64, d=16 ~ tens of hours)
runs survive the 24 h container timeout via retries + resume.

Outputs: /results/dscaling/tci_tdvp1/d{d}_N{N}_K{K}_D{D}/seed{seed}.npz
Download: modal volume get tnwf-results /dscaling ./results/dscaling

Launch:  uv run modal run modal/scaling_d_highdim.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tnwf-scaling-d-highdim"
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

N_FIXED = 32
SEED = 0
CELLS = [
    (d, N_FIXED, K, D)
    for d in (10, 12, 16)
    for K in (64, 128)
    for D in (32, 48, 64)
]


@app.function(cpu=4.0, memory=32768, timeout=86400, retries=3,
              volumes={"/results": volume})
def run_cell(d: int, N: int, K: int, D: int) -> str:
    import numpy as np
    from tnwf.pipelines.run_evolution import run

    out_dir = Path(f"/results/dscaling/tci_tdvp1/d{d}_N{N}_K{K}_D{D}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"seed{SEED}.ckpt.pkl"
    try:
        r = run(
            method="tci_tdvp1", dataset=f"gmm_{d}d", jam_ckpt=None, seed=SEED,
            N=N, K=K, d=d, n_samples=500, save=True, out_dir=str(out_dir),
            V_source="analytic",
            method_kwargs={"D_max": D, "D_V": D, "D_out": D, "D_init": D},
            checkpoint_path=str(ckpt), checkpoint_every=1,
            checkpoint_callback=lambda: volume.commit(),
        )
        chi = int(np.asarray(r["chi_max"]).max())
        sw = float(r["sw"][-1])
        fl = float(np.sum(r["step_times"]))
        volume.commit()
        return (f"d{d}_N{N}_K{K}_D{D}: SW={sw:.3f} chi*={chi}/{D} "
                f"flow_h={fl/3600:.2f}")
    except Exception as e:
        return f"d{d}_N{N}_K{K}_D{D}_FAIL: {type(e).__name__}: {e}"


@app.local_entrypoint()
def main():
    print(f"launching {len(CELLS)} TCI+TDVP1 cells (analytic V): "
          f"d{{10,12,16}} x K{{64,128}} x D{{32,48,64}}")
    for res in run_cell.starmap(CELLS):
        print(res, flush=True)
