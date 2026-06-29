"""V-step MPO construction via TT-cross.

**V-only MPO** — bare V_t(x) as TT cores, used by TDVP1/TDVP2 (which
exponentiate locally via expm_multiply on small effective Hamiltonian
blocks). Oracle returns real-valued V(x). Pivots are selected by the
SVD/MAXVOL strategy of :func:`tnwf.mps.tt_cross.tt_cross`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from tnwf.mps.tt_cross import tt_cross


def _make_grid_1d(N: int, L: float) -> np.ndarray:
    """Left-edge 1D grid on [0, L)."""
    return np.linspace(0.0, L, N, endpoint=False)


# Maximum number of points evaluated by V_fn in a single call. The analytic
# V_fn allocates an (n_pts, K_modes, d) tensor of squared-distances; at d=6
# with K_modes=12 a 65k-point batch is ~38 MB, which compounds across many
# in-flight calls and tipped 64 GB hosts into OOM at d=6 N=64. Capping
# per-call batch keeps peak per-call memory bounded regardless of how big
# tt_cross's index batch grows.
ORACLE_CHUNK = 4096


def _chunked_oracle_eval(V_fn: Callable, x: np.ndarray, t: float) -> np.ndarray:
    """Evaluate V_fn(x, t) in fixed-size chunks; output identical to V_fn(x, t)."""
    n = x.shape[0]
    if n <= ORACLE_CHUNK:
        return np.asarray(V_fn(x, t), dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, ORACLE_CHUNK):
        e = min(s + ORACLE_CHUNK, n)
        out[s:e] = np.asarray(V_fn(x[s:e], t), dtype=np.float64)
    return out


def _cached_oracle_eval(
    V_fn: Callable,
    points: np.ndarray,
    grid_1d: np.ndarray,
    t: float,
    cache: dict,
) -> np.ndarray:
    """Same output as ``_chunked_oracle_eval(V_fn, grid_1d[points], t)``, but
    memoizes per-row results keyed by ``(t, tuple(int_idx))``.

    Caching is at the row level so partial overlaps across calls reuse the
    rows that were previously evaluated. Misses are batched into a single
    chunked V_fn call to amortise the per-call overhead (PyTorch forward for
    JAM V_fn, vectorised analytic for closed-form V_t).

    Audit piece 4.1. Cache lifetime is the lifetime of the ``cache`` dict
    object — typically the V-step closure's ``warm_state["V_cache"]`` so it
    survives across K Trotter steps within a single ``run()``.
    """
    n = points.shape[0]
    out = np.empty(n, dtype=np.float64)
    # Walk points, collect misses; populate hits in-place.
    miss_idx: list[int] = []
    miss_keys: list[tuple] = []
    for i in range(n):
        key = (t, tuple(int(p) for p in points[i]))
        cached = cache.get(key)
        if cached is None:
            miss_idx.append(i)
            miss_keys.append(key)
        else:
            out[i] = cached
    # Batched V_fn call on the miss rows.
    if miss_idx:
        miss_pts = grid_1d[points[miss_idx]]
        miss_vals = _chunked_oracle_eval(V_fn, miss_pts, t)
        for i, key, val in zip(miss_idx, miss_keys, miss_vals):
            cache[key] = float(val)
            out[i] = val
    return out


def build_V_mpo(
    V_fn: Callable,
    t: float,
    N: int,
    d: int,
    L: float,
    D_max: int,
    n_sweeps: int = 2,
    seed: int = 0,
    init_right_idx: list[np.ndarray] | None = None,
    return_right_idx: bool = False,
    oracle_cache: dict | None = None,
) -> list[np.ndarray]:
    """Build bare V_t(x) MPO (real → complex). Used by TDVP V-steps.

    If ``init_right_idx`` is given, seeds the TT-cross MAXVOL search.
    If ``oracle_cache`` is given, V_fn rows are memoized.
    """
    grid_1d = _make_grid_1d(N, L)

    if oracle_cache is None:
        def oracle(points: np.ndarray) -> np.ndarray:
            x = grid_1d[points]
            return _chunked_oracle_eval(V_fn, x, t)
    else:
        def oracle(points: np.ndarray) -> np.ndarray:
            return _cached_oracle_eval(V_fn, points, grid_1d, t, oracle_cache)

    result = tt_cross(
        oracle, N=N, d=d, D_max=D_max, n_sweeps=n_sweeps, seed=seed,
        init_right_idx=init_right_idx,
        return_right_idx=return_right_idx,
    )
    if return_right_idx:
        cores, right_idx_out = result
        return [C.astype(np.complex128) for C in cores], right_idx_out
    return [C.astype(np.complex128) for C in result]
