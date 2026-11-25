"""Exact bond dimension of sqrt(p) for correlated Gaussian components (SI, correlated mixtures).

sqrt(p) for p = N(0, S) is a pure Gaussian state.  Across a cut A|B its Schmidt weights are
products of geometric laws (1-q) q^n, with q = (2 nu - 1) / (2 nu + 1) and nu^2 the eigenvalues
of S_AA (S^-1)_AA / 4.  chi(eps) is the smallest bond with discarded weight <= eps; a mixture of
K well-separated components needs at most K * chi(eps).  Banded S_ij = s^2 rho^|i-j| in the
natural, window-shuffled and random orders (paper Table B2), and along the exact flow path
sqrt(p_t), p_t = N(0, (1-t)^2 sigma0^2 I + t^2 S).  Usage: python scripts/correlated_bond_scaling.py
"""
import heapq

import numpy as np


def cut_q(S, Sinv, c):
    ev = np.linalg.eigvals(S[:c, :c] @ Sinv[:c, :c] / 4).real
    nu = np.sqrt(np.clip(ev, 0.25, None))
    return np.sort((2 * nu - 1) / (2 * nu + 1))[::-1]


def chi(q, eps, nmax=3000):
    """Smallest bond with discarded weight <= eps (nmax = lower bound if reached)."""
    q = q[q > 1e-12]
    if len(q) == 0:
        return 1
    lq, tot = np.log(q), 0.0
    heap = [(-np.log1p(-q).sum(), 0, ())]      # (-log weight, last mode used, occupations)
    for n in range(1, nmax + 1):
        nl, last, occ = heapq.heappop(heap)
        tot += np.exp(-nl)
        if tot >= 1 - eps:
            return n
        for i in range(last, len(q)):
            if -nl + lq[i] < np.log(1e-14):
                break
            heapq.heappush(heap, (nl - lq[i], i, occ + (i,)))
    return nmax


def window_order(d, w, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([s + rng.permutation(min(w, d - s)) for s in range(0, d, w)])


def max_chi(S, eps=1e-3):
    Si = np.linalg.inv(S)
    return max(chi(cut_q(S, Si, c), eps) for c in range(1, len(S)))


def banded(d, rho, s=0.5):
    return s ** 2 * rho ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))


if __name__ == "__main__":
    print("max over cuts of chi(1e-3), one component; '+' = lower bound (>= 3000)")
    for d in (32, 64, 128, 256):
        orders = [("natural", np.arange(d)), ("window 4", window_order(d, 4))]
        if d == 64:
            orders.append(("random", np.random.default_rng(0).permutation(d)))
        for name, o in orders:
            row = [max_chi(banded(d, rho)[np.ix_(o, o)]) for rho in (0.3, 0.6, 0.9)]
            print(f"d={d:3d} {name:9s} rho=0.3/0.6/0.9: " + " / ".join(f"{x}{'+' if x >= 3000 else ''}" for x in row))
    print("along the flow path, d=64, rho=0.6, sigma0 = 8/6:")
    for t in (0.5, 0.8, 0.9, 1.0):
        print(f"  t={t}: {max_chi((1 - t) ** 2 * (8 / 6) ** 2 * np.eye(64) + t ** 2 * banded(64, 0.6))}")
