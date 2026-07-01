"""Generate a V-MPS + 2TDVP trajectory from a trained MPS-V checkpoint.

Loads a trained MPS-V (tensor-train velocity potential), runs the 2-site TDVP
V-step with the trained cores fed directly — no runtime tensor-cross — and saves
the sampled trajectory plus metrics to an ``.npz`` the demo notebook can load.

The reported accuracy is the **grid-unbiased** sliced-Wasserstein: WF samples land
on grid nodes, so a half-cell shift ``x -> x - dx/2`` (``dx = L/N``) is applied
before the SW, matching the paper's convention (Table 2).

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

from tnwf.pipelines.run_evolution import run


def _sw(a: np.ndarray, b: np.ndarray, n_proj: int = 200, seed: int = 0) -> float:
    """Sliced-Wasserstein between two point clouds (paper convention)."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n_proj, a.shape[1]))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    a_s, b_s = np.sort(a @ p.T, axis=0), np.sort(b @ p.T, axis=0)
    if a_s.shape[0] != b_s.shape[0]:
        idx = np.linspace(0, b_s.shape[0] - 1, a_s.shape[0]).astype(int)
        b_s = b_s[idx]
    return float(np.mean(np.abs(a_s - b_s)))


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
    dx = L / N
    target = np.asarray(r["target"], dtype=np.float64)
    # Grid-unbiased SW per step (half-cell shift) from the per-step sample snapshots.
    sw_unbiased = np.array([_sw(np.asarray(s, np.float64) - dx / 2.0, target)
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
