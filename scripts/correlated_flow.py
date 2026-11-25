"""Correlated Gaussian mixture flow (paper Appendix B.6): TCI+1TDVP in d dimensions.

Target: two equal-weight Gaussian components with means at +-3 on the central coordinate and
the banded covariance S_ij = s^2 rho^|i-j| (s = 0.5).  The flow uses the analytic potential.
Readout: the correlation recovery R = (c0 - err) / (c0 - err_grid), where err is the RMS
difference between the pooled within-component correlation matrix of the Born samples and
the target's, c0 is the same error for an output with uncorrelated coordinates, and err_grid
is the error of exact target samples rounded to the grid.

Step coefficients: "coordinate" sets alpha = (L / pi N) sqrt(dt), beta = (pi N / 2L) sqrt(dt);
"balanced" is the package default, beta ~ sqrt(d), which fails on these targets for d >= 32
unless --substeps 4 is given.  Both keep alpha * beta = dt / 2.

Usage: uv run python scripts/correlated_flow.py --d 8 [--rho 0.6] [--angles balanced]
       [--substeps 4].  Runtime at N = 32, K = 160: minutes at d = 8, hours at d >= 64.
"""
import argparse
import math

import numpy as np
from scipy.special import logsumexp

from tnwf.dense.evolution import trotter_coefficients
from tnwf.mps.core import chi_max, right_canonicalize
from tnwf.pipelines.run_evolution import (
    _make_v_step,
    _product_formula_step,
    initial_psi_mps,
    sample_from_mps,
)

L, SIGMA0, STD = 8.0, 8.0 / 6.0, 0.5


def target(d, rho):
    i = np.arange(d)
    mus = np.zeros((2, d))
    mus[0, d // 2], mus[1, d // 2] = 3.0, -3.0
    return mus, STD ** 2 * rho ** np.abs(i[:, None] - i[None, :])


def potential(mus, Sigma, t_eps=1e-3):
    """V_t(x) = |x|^2 / 2t + ((1 - t) sigma0^2 / t) log p_t(x), with x measured from L/2."""
    lam, U = np.linalg.eigh(Sigma)
    mus_U, d = mus @ U, len(Sigma)

    def V_fn(x_world, t):
        x = np.asarray(x_world, dtype=np.float64) - L / 2.0
        t = max(float(t), t_eps)
        inv = 1.0 / ((1 - t) ** 2 * SIGMA0 ** 2 + t ** 2 * lam)
        y = x @ U
        maha = ((y ** 2) @ inv)[:, None] - 2 * t * y @ (mus_U * inv).T + t ** 2 * ((mus_U ** 2) @ inv)
        log_p = logsumexp(-0.5 * maha, axis=-1) - 0.5 * (d * math.log(2 * math.pi) - np.log(inv).sum())
        return (x ** 2).sum(-1) / (2 * t) + (1 - t) * SIGMA0 ** 2 / t * (log_p - math.log(len(mus)))

    return V_fn


def flow(V_fn, d, N, K, D_max, D_V, angles, substeps):
    dt = 1.0 / K
    if angles == "balanced":
        alpha, beta = trotter_coefficients(dt, N=N, d=d, L=L)
    else:
        alpha, beta = L / (math.pi * N) * math.sqrt(dt), math.pi * N / (2 * L) * math.sqrt(dt)
    v_step = _make_v_step("tci_tdvp1", D_max=D_max, D_V=D_V, n_sweeps=1, n_sweeps_cross=2, tol=1e-8)
    state = initial_psi_mps(N=N, d=d, L=L, sigma=SIGMA0, D_init=D_max)
    for k in range(K):
        state = _product_formula_step(
            state, v_step, V_fn, alpha=alpha, beta=beta, t_k=max((k + 0.5) * dt, 1e-5), N=N, d=d,
            L=L, eigenvalues=None, is_dense=False, n_substeps=substeps)
        state = right_canonicalize(state)
        if (k + 1) % max(1, K // 8) == 0:
            print(f"step {k + 1}/{K}  bond {chi_max(state)}", flush=True)
    return state


def within_corr(x, mus):
    """Pooled within-component correlation matrix and the component fractions."""
    lab = ((x[:, None, :] - mus[None]) ** 2).sum(-1).argmin(1)
    cov, n = np.zeros((x.shape[1],) * 2), 0
    for j in range(len(mus)):
        r = x[lab == j] - x[lab == j].mean(0)
        cov, n = cov + r.T @ r, n + len(r) - 1
    s = np.sqrt(np.diag(cov / n))
    return cov / n / np.outer(s, s), np.bincount(lab, minlength=len(mus)) / len(x)


def recovery(samples, mus, Sigma, N, n=10000):
    """samples are centered (world coordinate minus L/2)."""
    rng = np.random.default_rng(1000)
    ref = mus[rng.integers(0, len(mus), n)] + rng.standard_normal((n, len(Sigma))) @ np.linalg.cholesky(
        Sigma + 1e-12 * np.eye(len(Sigma))).T
    dx = L / N
    grid = -L / 2 + np.clip(np.round((ref + L / 2) / dx), 0, N - 1) * dx
    grid = grid + (np.random.default_rng(7).random(ref.shape) - 0.5) * dx
    s = np.sqrt(np.diag(Sigma))
    od = np.triu_indices(len(Sigma), 1)
    c_target = (Sigma / np.outer(s, s))[od]
    c_flow, frac = within_corr(samples, mus)
    err = np.sqrt(np.mean((c_flow[od] - c_target) ** 2))
    err_grid = np.sqrt(np.mean((within_corr(grid, mus)[0][od] - c_target) ** 2))
    c0 = np.sqrt(np.mean(c_target ** 2))
    return (c0 - err) / (c0 - err_grid), np.abs(frac - 1 / len(mus)).max()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--K", type=int, default=160)
    ap.add_argument("--D_max", type=int, default=8)
    ap.add_argument("--D_V", type=int, default=32)
    ap.add_argument("--angles", choices=["coordinate", "balanced"], default="coordinate")
    ap.add_argument("--substeps", type=int, default=1)
    a = ap.parse_args()
    mus, Sigma = target(a.d, a.rho)
    state = flow(potential(mus, Sigma), a.d, a.N, a.K, a.D_max, a.D_V, a.angles, a.substeps)
    x = sample_from_mps(state, N=a.N, d=a.d, L=L, n=10000, rng=np.random.default_rng(0)) - L / 2
    R, w_err = recovery(x, mus, Sigma, a.N)
    print(f"d={a.d} rho={a.rho} bond={chi_max(state)} recovery R={R:.4f} component-weight error={w_err:.4f}")
