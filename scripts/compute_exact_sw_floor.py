"""Compute the target--target finite-sample sliced-Wasserstein reference.

This is the ``Target--target (finite-sample)`` row of Table 1: the SW between
two *independent* draws of the same target distribution. It is a scale set by
the sample size and the projection count, not an irreducible error floor --
draw more samples and it shrinks.

`make_table1_sw.py` has always named this script as the generator of
`data/exact_sw_floor.json`, but it was not in the release, which left Table 1's
reference row as a magic constant. That matters more after the half-cell
correction, because the corrected method rows move *closer* to this row.

Protocol (Supplementary Section S2.12): for each d, draw `n_pairs` independent
pairs of size `n_samples` from the orthogonal GMM and average
SW(x_a, x_b) over pairs, with fresh random projections per pair.

Usage:
    uv run python scripts/compute_exact_sw_floor.py
    uv run python scripts/compute_exact_sw_floor.py --check   # verify, don't write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tnwf.data.gaussian_mixture import sample_gaussian_mixture  # noqa: E402
from tnwf.metrics.sw import sliced_wasserstein  # noqa: E402

DIMS = [2, 3, 4, 5, 6, 7, 8]
N_SAMPLES = 500        # matches the per-cell sample count of the sweeps
N_PAIRS = 32
N_PROJECTIONS = 128    # the paper's stated projection count
STD, SCALE = 0.5, 3.0
SEED = 0


def floor_for_d(d: int) -> dict:
    vals = []
    for p in range(N_PAIRS):
        a = sample_gaussian_mixture(N_SAMPLES, d=d, std=STD, scale=SCALE,
                                    arrangement="orthogonal",
                                    seed=SEED + 1000 * d + 2 * p)
        b = sample_gaussian_mixture(N_SAMPLES, d=d, std=STD, scale=SCALE,
                                    arrangement="orthogonal",
                                    seed=SEED + 1000 * d + 2 * p + 1)
        vals.append(sliced_wasserstein(
            a.astype(np.float64), b.astype(np.float64),
            n_projections=N_PROJECTIONS,
            rng=np.random.default_rng(SEED + 7919 * d + p)))
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n_pairs": N_PAIRS, "n_samples": N_SAMPLES,
            "n_projections": N_PROJECTIONS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/exact_sw_floor.json")
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored file instead of writing")
    a = ap.parse_args()

    out = {str(d): floor_for_d(d) for d in DIMS}
    for d in DIMS:
        r = out[str(d)]
        print(f"  d={d}: {r['mean']:.4f} +/- {r['std']:.4f}")

    path = Path(a.out)
    if a.check and path.exists():
        stored = json.loads(path.read_text())
        print("\ncheck against stored values:")
        ok = True
        for d in DIMS:
            s, n = stored[str(d)]["mean"], out[str(d)]["mean"]
            within = abs(s - n) <= out[str(d)]["std"]
            ok &= within
            print(f"  d={d}: stored {s:.4f}  recomputed {n:.4f}  "
                  f"delta {n - s:+.4f}  {'ok' if within else 'OUTSIDE 1 sigma'}")
        raise SystemExit(0 if ok else 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
