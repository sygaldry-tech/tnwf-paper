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
) -> list[np.ndarray]:
    """Build bare V_t(x) MPO (real → complex). Used by TDVP V-steps.

    If ``init_right_idx`` is given, seeds the TT-cross MAXVOL search.

    Note on cost: no shipped caller warm-starts the search
    (``tnwf.mps.vstep_tdvp`` passes neither ``init_right_idx`` nor a reuse of
    the previous step's index sets), so the cross is rebuilt cold on every
    call -- four times per Trotter step, at identical ``t``. That uncached
    cost is precisely the baseline the trained-MPS-V bypass is measured
    against in Section 2.4, so it should be read as a comparison against the
    cross as actually run, not against an optimally warm-started one. An
    earlier row-level memo for ``V_fn`` lived here but no caller ever enabled
    it, so it has been removed rather than left as dead weight.
    """
    grid_1d = _make_grid_1d(N, L)

    def oracle(points: np.ndarray) -> np.ndarray:
        x = grid_1d[points]
        return _chunked_oracle_eval(V_fn, x, t)

    result = tt_cross(
        oracle, N=N, d=d, D_max=D_max, n_sweeps=n_sweeps, seed=seed,
        init_right_idx=init_right_idx,
        return_right_idx=return_right_idx,
    )
    if return_right_idx:
        cores, right_idx_out = result
        return [C.astype(np.complex128) for C in cores], right_idx_out
    return [C.astype(np.complex128) for C in result]
