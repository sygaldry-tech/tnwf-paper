"""Regenerate the three CSV inputs for Fig. 4 using the public dense solver.

Usage: PYTHONPATH=src python scripts/compute_scaling_theory.py [--panel A|B|D] [--out DIR]
Time runs from source noise at 0 to target data at 1.
"""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from tnwf.dense.evolution import apply_product_formula, trotter_coefficients
from tnwf.grid import make_grid, make_kinetic_eigenvalues
from tnwf.pipelines.run_evolution import initial_psi_dense
from tnwf.theory import make_analytic_V_fn

L, SCALE, SIG0, T0 = 8.0, 3.0, 8.0 / 6.0, 0.05

def segment(psi, potential, grid, eigenvalues, N, d, start, end, steps):
    """Stream the potential and eight-factor formula, retaining one state."""
    dt = (end - start) / steps
    alpha, beta = trotter_coefficients(dt, N, d, L)
    for k in range(steps):
        V = potential(grid, start + (k + 0.5) * dt)
        psi = apply_product_formula(psi, V, alpha, beta, N, d, L,
                                    eigenvalues=eigenvalues)
    return psi

def setup(N, d, reference_steps):
    """Evolve [0, T0] at the reference step count, excluding the 1/t singularity."""
    potential = make_analytic_V_fn("gmm_2d", L=L, d=d, scale=SCALE,
                                   std=0.5, sigma_0=SIG0)
    grid = make_grid(N, d, L)
    eigenvalues = make_kinetic_eigenvalues(N, d, L)
    psi = initial_psi_dense(N, d, L, sigma=SIG0)
    psi = segment(psi, potential, grid, eigenvalues, N, d, 0, T0, reference_steps)
    return psi, potential, grid, eigenvalues

def state_distance(a, b):
    a = a.ravel() / np.linalg.norm(a)
    b = b.ravel() / np.linalg.norm(b)
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * abs(np.vdot(a, b)))))

def temporal(N, d, steps, reference_steps):
    psi, potential, grid, eigenvalues = setup(N, d, reference_steps)
    def evolve(k):
        return segment(psi.copy(), potential, grid, eigenvalues, N, d,
                       T0, 1.0, k)
    reference = evolve(reference_steps)
    return [state_distance(evolve(k), reference) for k in steps]

def truncation_error(f, N, ref=8192):
    x = -L / 2 + np.arange(ref) * (L / ref)
    spectrum = np.fft.fft(f(x))
    keep = np.zeros(ref, bool)
    h = N // 2
    keep[:h + 1], keep[-h:] = True, True
    return float(np.linalg.norm(spectrum - np.where(keep, spectrum, 0))
                 / np.linalg.norm(spectrum))


def panel_b():
    def center(x):
        return (x % L) - L / 2
    def matern(x):
        u = np.sqrt(3) * np.abs(center(x)) / 0.5
        return np.sqrt((1 + u) * np.exp(-u))
    functions = (lambda x: np.exp(-center(x) ** 2 / (4 * 0.5 ** 2)),
                 matern, lambda x: np.exp(-np.abs(center(x)) / (2 * 0.5)))
    return [[N, *(truncation_error(f, N) for f in functions)]
            for N in (16, 32, 64, 128, 256)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/scaling_theory"))
    parser.add_argument("--panel", choices=("A", "B", "D"), help="run one panel")
    parser.add_argument("--max-d", type=int, choices=(2, 3, 4, 5), default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for panel in (args.panel,) if args.panel else ("A", "B", "D"):
        start = perf_counter()
        if panel == "A":
            values = (128, 256, 512, 1024, 2048, 4096)
            rows = zip(values, temporal(16, 2, values, 16384))
            name, header = "panelA_state", ("K", "sdist")
        elif panel == "B":
            rows = panel_b()
            name, header = "panelB_repr", ("N", "gauss", "matern32", "laplace")
        else:
            rows = [[d, temporal(12, d, (512,), 4096)[0]] for d in range(2, args.max_d + 1)]
            name, header = "panelD_state", ("d", "sdist")
        with (args.out / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{panel}: wrote {args.out / (name + '.csv')} in {perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
