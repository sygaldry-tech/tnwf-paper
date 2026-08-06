"""Generate a V-MPS + 2TDVP trajectory from a trained MPS-V checkpoint.

Loads a trained MPS-V (tensor-train velocity potential), runs the 2-site TDVP
V-step with the trained cores fed directly — no runtime tensor-cross — and saves
the sampled trajectory plus metrics to an ``.npz`` the demo notebook can load.

The reported accuracy is the sliced-Wasserstein on node-centred Born samples
(see ``tnwf.coords``): the sampler dithers symmetrically about the grid node, so
no half-cell correction is applied here. Releases up to v1 dithered to the right
of the node and compensated at this point instead; the ``sw_unbiased`` key name
is kept for compatibility with the shipped artifact and the demo notebook.

The shipped ``examples/checkpoints/mps_v_gmm_d8_result.npz`` holds the paper's
actual Table 2 run (2TDVP, N=32, K=160; SW ~ 0.036). This script reproduces an
equivalent trajectory from the checkpoint; the default K=40 is a ~1 h CPU sanity
run (SW ~ 0.08 unbiased), while ``--K 160`` reproduces the near-floor result
(several hours on CPU, faster on a GPU).

Usage:
    python scripts/mps_v/generate.py \
        --ckpt examples/checkpoints/mps_v_gmm_d8.pt --dataset gmm_8d \
        --K 160 --D_max 24 --out examples/checkpoints/mps_v_gmm_d8_result.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from tnwf.metrics.sw import sliced_wasserstein
from tnwf.pipelines.run_evolution import run

# The paper's estimator: 128 projections, fixed seed so the trajectory is
# reproducible from the saved samples. Previously this module carried its own
# 200-projection copy of sliced_wasserstein with a different length-matching
# rule, which quietly made its numbers incomparable with the pipeline's.
SW_PROJECTIONS = 128
SW_SEED = 0


def _sw(a: np.ndarray, b: np.ndarray) -> float:
    return float(sliced_wasserstein(a, b, n_projections=SW_PROJECTIONS,
                                    rng=np.random.default_rng(SW_SEED)))


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

    L, N = float(r["L"]), int(r["N"])
    target = np.asarray(r["target"], dtype=np.float64)
    # Samples are already node-centred (tnwf.coords), so no half-cell shift is
    # applied here. `sw_unbiased` is retained as a key name because the shipped
    # v1 artifact and the demo notebook read it; under the current convention it
    # is simply the SW, identical to `sw`.
    sw_unbiased = np.array([_sw(np.asarray(s, np.float64), target)
                            for s in r["samples_per_step"]])
    # Sample-size floor: SW between two independent halves of the target draw.
    h = target.shape[0] // 2
    sw_floor = _sw(target[:h], target[h:2 * h])
    wall_cum = np.cumsum(np.asarray(r["step_times"], dtype=np.float64))

    np.savez(
        a.out,
        k=np.arange(len(r["sw"])), t=np.linspace(0.0, 1.0, len(r["sw"])),
        chi=np.asarray(r["chi_max"]), sw=r["sw"], sw_unbiased=sw_unbiased,
        mmd=r["mmd"], wall_cum=wall_cum,
        samples_T=r["samples_T"], target=r["target"],
        N=N, d=int(r["d"]), K=int(r["K"]), L=L,
        sw_unbiased_mean=float(sw_unbiased[-1]), sw_unbiased_ci=0.0,
        n_reps=1, sw_floor=sw_floor,
    )
    print(f"saved {a.out}: N={N} K={int(r['K'])} "
          f"unbiased SW={float(sw_unbiased[-1]):.4f} (floor {sw_floor:.4f}) "
          f"chi*={int(np.asarray(r['chi_max']).max())} "
          f"wall={wall_cum[-1] / 60:.0f} min")


if __name__ == "__main__":
    main()
