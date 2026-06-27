"""TCI+ALS V-step: build exp(iβV) via TT-cross, then variational ALS-compress.

Pipeline:
  1. exp_V = TT-cross of  x ↦ exp(iβ V_t(x))   with bond dim D_V.
  2. out = ALS-compress(exp_V ⊙ ψ) to D_out.

The ALS step minimises ‖out − exp_V⊙ψ‖² via per-site closed-form updates,
using full left/right overlap environments. At D_out ≥ D_V·χ(ψ) the result is
exact; below that capacity it converges to the optimal D_out-MPS approximation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from tnwf.mps.core import chi_max, mps_norm, truncate_mps
from tnwf.mps.tt_cross import elementwise_product_mps
from tnwf.mpo.build_evolution_mpo import build_exp_iβV_mpo


def _als_init_compatible(
    psi_prev: list[np.ndarray], d: int, N: int, D_out: int,
) -> bool:
    """Return True if ``psi_prev`` is structurally valid as an ALS init for
    a length-d MPS with physical dim N and bond cap D_out.

    Accepts any ψ whose cores have shapes (D_L, N, D_R) with D_L, D_R ≤ D_out
    and matching physical dim. Tiny shape mismatches (e.g. χ growing/shrinking
    by one) are tolerated since ``_als_compress`` re-projects to D_out on its
    final ``truncate_mps`` call.
    """
    if len(psi_prev) != d:
        return False
    for core in psi_prev:
        if core.ndim != 3:
            return False
        D_L, n_loc, D_R = core.shape
        if n_loc != N:
            return False
        if D_L > D_out or D_R > D_out:
            return False
    return True


def _als_compress(
    target: list[np.ndarray],
    init: list[np.ndarray],
    D_out: int,
    n_sweeps: int = 2,
) -> list[np.ndarray]:
    """ALS-fit a D_out-bounded MPS to target. Updates `init` in place-style."""
    d = len(target)
    out = [c.copy() for c in init]

    for _ in range(n_sweeps):
        # Right environments: R[j] = ⟨out_{>j} | target_{>j}⟩, shape (D_out_j, D_tgt_j)
        R = [None] * (d + 1)
        R[d] = np.ones((1, 1), dtype=np.complex128)
        for j in range(d - 1, 0, -1):
            # R[j] = sum over (n, b_out, b_tgt): out[j][:, n, b_out].conj() * target[j][:, n, b_tgt] * R[j+1][b_out, b_tgt]
            R[j] = np.einsum(
                "anb,cnd,bd->ac", out[j].conj(), target[j], R[j + 1]
            )

        # L→R sweep
        L = np.ones((1, 1), dtype=np.complex128)
        for j in range(d):
            # Optimal out[j] = Σ_{a,b} L[a,a'] target[j][a',n,b'] R[j+1][b,b']  (no out conj — that's solve)
            # For overlap-fit: project target onto out's local basis with envs.
            # Closed-form: out[j][a,n,b] = Σ_{a',b'} L[a,a'] target[j][a',n,b'] R[j+1][b,b']
            new_core = np.einsum("ax,xny,by->anb", L, target[j], R[j + 1])
            D_L, N_loc, D_R_full = new_core.shape

            if j < d - 1:
                # QR + truncate to D_out
                mat = new_core.reshape(D_L * N_loc, D_R_full)
                Q, Rm = np.linalg.qr(mat)
                K = min(Q.shape[1], D_out)
                Q = Q[:, :K]
                Rm = Rm[:K, :]
                out[j] = Q.reshape(D_L, N_loc, K)
                # Absorb R into right neighbour (will be overwritten when sweep hits it,
                # but we need correct shape for L env update)
                out[j + 1] = np.einsum("ab,bnc->anc", Rm, out[j + 1])
                # Update L: L_new[a,a'] = sum_{x,n} out[j][x,n,a].conj() * target[j][x,n,a'] L_old?
                # Actually L_{j+1}[a, a'] = Σ_{x, x', n} L_j[x, x'] · out[j][x, n, a].conj() · target[j][x', n, a']
                L = np.einsum("xy,xna,ynb->ab", L, out[j].conj(), target[j])
            else:
                out[j] = new_core

    out, _ = truncate_mps(out, D_out)
    return out


# Product-bond threshold (= D_V·χ(ψ)) above which we switch from the original
# materialize-then-ALS path to the memory-safe environment-ALS path. Below it,
# the materialised product is small (≤ ~CAP²·N·16 B per core) and we keep the
# exact published numerics; above it (resourced regime, large D_V) materialising
# would OOM (D_V=χ=128 → bond 16384 → ~137 GB/core).
_PRODUCT_BOND_CAP = 2048


def _als_compress_envs(
    A: list[np.ndarray],
    V: list[np.ndarray],
    D_out: int,
    init: list[np.ndarray],
    n_sweeps: int = 2,
) -> list[np.ndarray]:
    """Environment-ALS: best D_out-MPS approximation to the Hadamard product
    A ⊙ V WITHOUT ever materialising the D_A·D_V product. Peak memory is set by
    the 3-leg environments (~D_out·D_A·D_V), not (D_A·D_V)². Numpy port of
    the research prototype's wavefunction/mps_wf_evolution.py:_variational_compress (einsum
    index conventions copied verbatim)."""
    d = len(A)
    out = [c.astype(np.complex128, copy=True) for c in init]
    A = [c.astype(np.complex128) for c in A]
    V = [c.astype(np.complex128) for c in V]

    for _ in range(n_sweeps):
        # Right environments R[j]: (D_out_R, D_A_R, D_V_R)
        R = [None] * (d + 1)
        R[d] = np.ones((1, 1, 1), dtype=np.complex128)
        for j in range(d - 1, 0, -1):
            tmp = np.einsum("anb,cbd->ancd", A[j], R[j + 1])
            tmp2 = np.einsum("ancd,end->aecn", tmp, V[j])
            R[j] = np.einsum("aecn,fnc->fae", tmp2, out[j].conj())

        # Left-to-right sweep
        L = np.ones((1, 1, 1), dtype=np.complex128)
        for j in range(d):
            tmp = np.einsum("oab,anp->onbp", L, A[j])
            tmp2 = np.einsum("onbp,bnq->onpq", tmp, V[j])
            b_eff = np.einsum("onpq,rpq->onr", tmp2, R[j + 1])
            if j < d - 1:
                D_L, N_loc, D_R = b_eff.shape
                Q, Rm = np.linalg.qr(b_eff.reshape(D_L * N_loc, D_R))
                D_keep = min(Q.shape[1], D_out)
                Q = Q[:, :D_keep]
                Rm = Rm[:D_keep, :]
                out[j] = Q.reshape(D_L, N_loc, D_keep)
                out[j + 1] = np.einsum("ab,bnc->anc", Rm, out[j + 1])
                t1 = np.einsum("xab,xnc->nabc", L, out[j].conj())
                t2 = np.einsum("nabc,and->nbcd", t1, A[j])
                L = np.einsum("nbcd,bne->cde", t2, V[j])
            else:
                out[j] = b_eff

    out, _ = truncate_mps(out, D_out)
    return out


def apply_V_step_mps_tci_als(
    mps: list[np.ndarray],
    V_fn: Callable,
    beta: float,
    t_k: float,
    N: int,
    d: int,
    L: float,
    D_V: int,
    D_out: int,
    n_sweeps: int = 2,
    n_sweeps_cross: int = 2,
    warm_state: dict | None = None,
    n_v_substeps: int = 1,
    n_global: int = 0,
    global_pool: int = 0,
) -> list[np.ndarray]:
    """Apply e^{iβ V_t} to an MPS via TCI exp(iβV) + ALS variational compression.

    If ``warm_state`` is a dict, it is read for cross-step warm-start hints
    and updated in place. Currently honored keys (warm-start audit pieces):

      "right_idx_exp" : carries the TT-cross right_idx between Trotter steps
                       (audit 1.1).  When present, seeds tt_cross MAXVOL
                       search; the function writes back the converged set.
      "psi_prev"      : carries the previous Trotter step's converged
                       ALS output (audit 2.1).  When present and shape-
                       compatible, seeds _als_compress instead of the
                       default truncated-target init.  Function writes
                       back the new ALS output.
      "V_cache"       : dict memoising V_fn(x, t) rows across V-steps
                       (audit 4.1).  Passed through to build_exp_iβV_mpo.

    Pass ``warm_state=None`` (default) for the original cold-start behavior.

    Note on independence: each piece is engaged only when its sentinel key is
    *present* in the warm_state dict (the value can be None on first step).
    The dispatcher in ``run_evolution._make_v_step`` is responsible for
    pre-populating the keys for whichever pieces are requested. This is
    intentional so the three audits can A/B independently — passing an empty
    ``{}`` would NOT activate any piece.
    """
    input_norm = mps_norm(mps)

    # Resourced-cross levers (all default-off → M=1, n_global=0 leaves the
    # published behaviour byte-identical):
    #   n_v_substeps M : split exp(iβV) → exp(i(β/M)V) applied M times. Keeps the
    #                    per-application operator low-rank (the product stays
    #                    compressible) when the per-step Trotter angle β is large.
    #   n_global       : global-pivot re-seeding of the exp(iβV) cross.
    M = max(1, int(n_v_substeps))
    beta_sub = beta / M

    pivot_active = warm_state is not None and "right_idx_exp" in warm_state
    oracle_active = warm_state is not None and "V_cache" in warm_state
    psi_active = warm_state is not None and "psi_prev" in warm_state

    oracle_cache = warm_state["V_cache"] if oracle_active else None
    right_idx_carry = warm_state["right_idx_exp"] if pivot_active else None

    cur = mps
    for m in range(M):
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

        product_bond = D_V * max(chi_max(cur), 1)

        # ALS init: first sub-step honours the cross-step psi_prev warm-start
        # (audit 2.1); later sub-steps warm-start from the previous sub-step's
        # output. Materialise path can additionally fall back to the truncated
        # product; env path falls back to truncated ψ (the product is never built).
        init = None
        if m == 0 and psi_active:
            psi_prev = warm_state["psi_prev"]
            if psi_prev is not None and _als_init_compatible(psi_prev, d, N, D_out):
                init = [c.copy() for c in psi_prev]
        elif m > 0 and _als_init_compatible(cur, d, N, D_out):
            init = [c.copy() for c in cur]

        import os
        cap = int(os.environ.get("TNWF_PRODUCT_BOND_CAP", _PRODUCT_BOND_CAP))
        if product_bond <= cap:
            # Small product: materialise then ALS (exact; published numerics).
            target_full = elementwise_product_mps(exp_V, cur, D_max=product_bond)
            init_m = init
            if init_m is None:
                init_m, _ = truncate_mps([c.copy() for c in target_full], D_max=D_out)
            cur = _als_compress(target_full, init_m, D_out=D_out, n_sweeps=n_sweeps)
        else:
            # Large product (resourced regime): environment-ALS, never materialised.
            init_e = init
            if init_e is None:
                init_e, _ = truncate_mps([c.copy() for c in cur], D_max=D_out)
            cur = _als_compress_envs(cur, exp_V, D_out=D_out, init=init_e, n_sweeps=n_sweeps)

    out = cur
    if pivot_active:
        warm_state["right_idx_exp"] = right_idx_carry

    output_norm = mps_norm(out)
    if output_norm > 1e-10 * input_norm:
        out[-1] = out[-1] * (input_norm / output_norm)

    if psi_active:
        # Cache the converged ψ for next-step warm-start. Storing reference is
        # fine: the caller does not mutate cores in-place.
        warm_state["psi_prev"] = out
    return out
