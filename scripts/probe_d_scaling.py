"""Single-cell TCI+TDVP1 probe for the d-scaling ladder (d>8).

Runs ONE (d, N, K, D) cell with the analytic GMM velocity, saves the standard
npz, and prints SW / achieved chi / flow time. Used to test whether TCI+TDVP1
scales to d=10,12,16 and what (N,K,D) configs are needed.

    uv run python scripts/probe_d_scaling.py --d 16 --N 32 --K 128 --D 32
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from tnwf.pipelines.run_evolution import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/probe_dscaling")
    a = ap.parse_args()

    od = Path(a.out) / "tci_tdvp1" / f"d{a.d}_N{a.N}_K{a.K}_D{a.D}"
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = run(method="tci_tdvp1", dataset=f"gmm_{a.d}d", jam_ckpt=None, seed=a.seed,
            N=a.N, K=a.K, d=a.d, n_samples=500, save=True, out_dir=str(od),
            V_source="analytic",
            method_kwargs={"D_max": a.D, "D_V": a.D, "D_out": a.D, "D_init": a.D},
            checkpoint_path=str(od / f"seed{a.seed}.ckpt.pkl"), checkpoint_every=1)
    chi = int(np.asarray(r["chi_max"]).max())
    sw = float(r["sw"][-1])
    fl = float(np.sum(r["step_times"]))
    print(f"RESULT d={a.d} N={a.N} K={a.K} D={a.D}: SW={sw:.3f} "
          f"chi*={chi}/{a.D} flow_h={fl/3600:.2f} wall_h={(time.time()-t0)/3600:.2f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
