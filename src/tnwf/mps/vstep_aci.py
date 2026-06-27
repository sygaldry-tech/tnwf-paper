"""ACI V-step: TCI exp(iβV) + ACI Hadamard product to fuse with ψ."""
from __future__ import annotations

from typing import Callable

import numpy as np

from tnwf.mps.aci import aci_hadamard
from tnwf.mps.core import mps_norm, truncate_mps
from tnwf.mpo.build_evolution_mpo import build_exp_iβV_mpo


def apply_V_step_mps_aci(
    mps: list[np.ndarray],
    V_fn: Callable,
    beta: float,
    t_k: float,
    N: int,
    d: int,
    L: float,
    D_max: int,
    D_V: int | None = None,
    n_sweeps: int = 4,
    tol: float = 1e-6,
    n_sweeps_cross: int = 2,
    warm_state: dict | None = None,
    n_v_substeps: int = 1,
    n_global: int = 0,
    global_pool: int = 0,
) -> list[np.ndarray]:
    """Apply e^{iβ V_t} to MPS via TCI exp(iβV) + ACI Hadamard.

    Honors the same warm_state contract as ``apply_V_step_mps_tci_als``:

      "right_idx_exp" : TT-cross pivot warm-start across Trotter steps
                       (audit 1.1).
      "V_cache"       : V_fn(x, t) row memoisation (audit 4.1).

    The ALS-specific "psi_prev" key (audit 2.1) does not apply here —
    ACI uses prrLU adaptive pivots, not an ALS sweep, so there's no
    init-state to warm-start. Pass ``warm_state=None`` (default) for
    the original cold-start behavior.

    Piece-independence: sentinel KEY presence (not value) gates each
    piece, matching the convention used in vstep_tci_als.
    """
    if D_V is None:
        D_V = D_max
    input_norm = mps_norm(mps)

    # Resourced-cross levers (default-off → M=1, n_global=0 leaves the published
    # behaviour byte-identical). See vstep_tci_als for the rationale.
    M = max(1, int(n_v_substeps))
    beta_sub = beta / M

    pivot_active = warm_state is not None and "right_idx_exp" in warm_state
    oracle_active = warm_state is not None and "V_cache" in warm_state

    oracle_cache = warm_state["V_cache"] if oracle_active else None
    right_idx_carry = warm_state["right_idx_exp"] if pivot_active else None

    cur = mps
    for _m in range(M):
        if pivot_active:
            exp_V, right_idx_carry = build_exp_iβV_mpo(
                V_fn, beta=beta_sub, t=t_k, N=N, d=d, L=L,
                D_max=D_V, n_sweeps=n_sweeps_cross, method="tci",
                init_right_idx=right_idx_carry, return_right_idx=True,
                oracle_cache=oracle_cache, n_global=n_global, global_pool=global_pool,
            )
        else:
            exp_V = build_exp_iβV_mpo(
                V_fn, beta=beta_sub, t=t_k, N=N, d=d, L=L,
                D_max=D_V, n_sweeps=n_sweeps_cross, method="tci",
                oracle_cache=oracle_cache, n_global=n_global, global_pool=global_pool,
            )
        cur = aci_hadamard(exp_V, cur, D_max=D_max, tol=tol, n_sweeps=n_sweeps)
        cur, _ = truncate_mps(cur, D_max=D_max, tol=tol)

    out = cur
    if pivot_active:
        warm_state["right_idx_exp"] = right_idx_carry
    output_norm = mps_norm(out)
    if output_norm > 1e-10 * input_norm:
        out[-1] = out[-1] * (input_norm / output_norm)
    return out
