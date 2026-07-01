"""Generate the paper-grade V-MPS 2TDVP d=8 GMM state (Table-2 config: N=32, K=160, D_max=24).

Writes a fresh result npz into the accInf_WF rare-event experiment folder so that
`experiments/rare_event_qae.py` picks it up. Must run in TNWF-paper's env (tnwf + torch).

    cd /Users/nxkodama/TNWF-paper
    .venv/bin/python scripts/gmm_8d_rare_event/gen_papergrade.py
"""
from __future__ import annotations

import os
import time

import numpy as np

from tnwf.pipelines.run_evolution import run

CKPT = "/Users/nxkodama/TNWF-paper/examples/checkpoints/mps_v_gmm_d8.pt"
OUT_DIR = "/Users/nxkodama/lab-nathan/projects/accInf_WF/results/gmm_8d_rare_event"
OUT = os.path.join(OUT_DIR, "mps_v_gmm_d8_result_K160_D24.npz")

K = 160
D_MAX = 24
N_SAMPLES = 4000
SEED = 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f"[gen] paper-grade V-MPS 2TDVP: K={K}, D_max={D_MAX}, n={N_SAMPLES} "
          f"(N fixed by checkpoint grid)...", flush=True)
    out = run(
        method="mps_v_tdvp2",
        dataset="gmm_8d",
        mps_v_ckpt=CKPT,
        K=K,
        n_samples=N_SAMPLES,
        seed=SEED,
        method_kwargs={"D_max": D_MAX},
        save=False,
    )
    sw = np.asarray(out["sw"], dtype=float)
    chi = np.asarray(out["chi_max"])
    np.savez(
        OUT,
        samples_T=np.asarray(out["samples_T"], dtype=np.float32),
        target=np.asarray(out["target"], dtype=np.float32),
        sw=sw,
        mmd=np.asarray(out["mmd"], dtype=float),
        chi_max=chi,
        step_times=np.asarray(out.get("step_times", []), dtype=float),
        N=int(out.get("N", 32)),
        d=int(out.get("d", 8)),
        K=int(out.get("K", K)),
        L=float(out.get("L", 8.0)),
        D_max=D_MAX,
    )
    print(f"[gen] SAVED {OUT}", flush=True)
    print(f"[gen] final SW={sw[-1]:.4f}  min SW={sw.min():.4f}  "
          f"chi[min,max,final]=[{int(chi.min())},{int(chi.max())},{int(chi[-1])}]  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
