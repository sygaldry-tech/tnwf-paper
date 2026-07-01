"""Generate a V-MPS + 2TDVP trajectory from a trained MPS-V checkpoint.

Loads a trained MPS-V (tensor-train velocity potential), runs the 2-site TDVP
V-step with the trained cores fed directly — no runtime tensor-cross — and saves
the sampled trajectory plus metrics (per-step sliced-Wasserstein, bond dimension,
step times, final samples, target) to an ``.npz`` for plotting.

The shipped ``examples/checkpoints/mps_v_gmm_d8_result.npz`` was produced by this
script at the default config (N=32 from the checkpoint, K=40, D_max=16;
final SW ~ 0.12, ~1 h on CPU). Raising ``--K`` toward 160 reproduces the paper's
near-floor d=8 sliced-Wasserstein (Table 2: 2TDVP, N=32, K=160, SW ~ 0.036) —
expect several hours on CPU, minutes on a GPU.

Usage:
    python scripts/mps_v/generate.py \
        --ckpt examples/checkpoints/mps_v_gmm_d8.pt --dataset gmm_8d \
        --K 40 --D_max 16 --out examples/checkpoints/mps_v_gmm_d8_result.npz
    # paper Table 2 (slow):
    python scripts/mps_v/generate.py --ckpt ... --K 160 --D_max 24 --out ...
"""
from __future__ import annotations

import argparse

import numpy as np

from tnwf.pipelines.run_evolution import run


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a V-MPS+2TDVP trajectory.")
    p.add_argument("--ckpt", required=True, help="trained MPS-V checkpoint (.pt)")
    p.add_argument("--dataset", default="gmm_8d")
    p.add_argument("--K", type=int, default=40, help="Trotter steps (160 for paper Table 2)")
    p.add_argument("--D_max", type=int, default=16, help="wavefunction MPS bond cap")
    p.add_argument("--n_samples", type=int, default=1500)
    p.add_argument("--out", required=True, help="output .npz path")
    a = p.parse_args()

    r = run(method="mps_v_tdvp2", dataset=a.dataset, mps_v_ckpt=a.ckpt,
            K=a.K, n_samples=a.n_samples, method_kwargs={"D_max": a.D_max}, save=False)

    np.savez(
        a.out,
        sw=r["sw"], mmd=r["mmd"], nll=r["nll"], chi_max=r["chi_max"],
        step_times=r["step_times"], samples_T=r["samples_T"], target=r["target"],
        N=r["N"], d=r["d"], K=r["K"], L=r["L"],
    )
    print(f"saved {a.out}: N={int(r['N'])} K={int(r['K'])} "
          f"final SW={float(r['sw'][-1]):.3f} chi*={int(r['chi_max'].max())} "
          f"wall={float(r['step_times'].sum()) / 60:.0f} min")


if __name__ == "__main__":
    main()
