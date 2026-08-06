"""TT-cross approximation of scalar fields.

Builds a TT (MPS) approximation of f: {0,...,N-1}^d → R (or C) using only
O(n_sweeps · d · D_max² · N) function evaluations rather than N^d. Numpy-only
port of the research prototype's wavefunction/tt_cross_approx.py with the TDVP-canonical
backward-sweep pivot update (SciPost Phys. 18, 104).

Pivot rows are selected by SVD-based greedy pivoted-QR row selection.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from tnwf.mps.core import truncate_mps


def _maxvol_pivot_rows(F: np.ndarray, D_max: int) -> np.ndarray:
    """Greedy pivoted-QR-style row selection: up to D_max rows that maximize volume.

    O(D_max · M · min(M, K)) selection on (M, K) matrix. Equivalent to MAXVOL.
    """
    M, K = F.shape
    if M <= D_max:
        return np.arange(M, dtype=np.int32)

    k_sub = min(D_max, M, K)
    try:
        U, s, _ = np.linalg.svd(F, full_matrices=False)
        U_k = U[:, :k_sub]                # (M, k_sub) — rows in dominant subspace
    except np.linalg.LinAlgError:
        norms = np.linalg.norm(F, axis=1)
        return np.argsort(norms)[-D_max:].astype(np.int32)

    selected: list[int] = []
    residual = U_k.copy()
    for _ in range(D_max):
        norms_sq = np.einsum("ij,ij->i", residual, residual.conj()).real
        idx = int(np.argmax(norms_sq))
        if norms_sq[idx] < 1e-14:
            break
        selected.append(idx)
        v = residual[idx].copy()
        v /= math.sqrt(float(norms_sq[idx]))
        proj = residual @ v.conj()
        residual = residual - np.outer(proj, v)

    return np.array(selected, dtype=np.int32)


def _update_right_indices(
    fn: Callable[[np.ndarray], np.ndarray],
    left_idx: list[np.ndarray],
    right_idx_in: list[np.ndarray],
    N: int,
    d: int,
    D_max: int,
    dtype,
) -> list[np.ndarray]:
    """Right-to-left sweep: select MAXVOL columns of F(D_L, N·D_R) per bond."""
    right_idx: list[np.ndarray] = [None] * (d + 1)
    right_idx[d] = np.zeros((1, 0), dtype=np.int32)

    for j in range(d - 1, 0, -1):
        D_L = left_idx[j].shape[0]
        # Use updated right_idx if computed, else fall back to input
        if right_idx[j + 1] is None:
            right_idx[j + 1] = right_idx_in[j + 1]
        D_R = right_idx[j + 1].shape[0]

        il_g, n_g, ir_g = np.meshgrid(
            np.arange(D_L, dtype=np.int32),
            np.arange(N, dtype=np.int32),
            np.arange(D_R, dtype=np.int32),
            indexing="ij",
        )
        il_flat, n_flat, ir_flat = il_g.ravel(), n_g.ravel(), ir_g.ravel()
        left_part = left_idx[j][il_flat]
        local_part = n_flat[:, None]
        right_part = right_idx[j + 1][ir_flat]
        all_idx = np.concatenate([left_part, local_part, right_part], axis=1).astype(np.int32)
        values = np.asarray(fn(all_idx), dtype=dtype)
        F = values.reshape(D_L, N * D_R)

        pivot_cols = _maxvol_pivot_rows(F.T.astype(np.complex128), D_max)
        new_right: list[np.ndarray] = []
        for col in pivot_cols:
            n = int(col) // D_R
            ir = int(col) % D_R
            new_right.append(np.concatenate([[n], right_idx[j + 1][ir]]).astype(np.int32))
        right_idx[j] = np.unique(np.array(new_right, dtype=np.int32), axis=0)[:D_max]

    right_idx[0] = np.zeros((1, 0), dtype=np.int32)
    return right_idx


def _tt_eval(cores: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Reconstruct TT amplitudes at a batch of multi-indices (M, d) → (M,)."""
    acc = np.ones((idx.shape[0], 1), dtype=cores[0].dtype)
    for k in range(len(cores)):
        acc = np.einsum("mr,rms->ms", acc, cores[k][:, idx[:, k], :])
    return acc[:, 0]


def _forward_sweep(fn, right_idx, N, d, D_max, tol, out_dtype):
    """One left-to-right TT-cross pass given right index sets → (cores, left_idx).

    Extracted verbatim from the tt_cross main loop so the main sweeps and the
    global-pivot rounds share identical logic.
    """
    cores: list[np.ndarray] = [None] * d  # type: ignore
    left_idx: list[np.ndarray] = [np.zeros((1, 0), dtype=np.int32)]
    for j in range(d):
        D_L = left_idx[j].shape[0]
        D_R = right_idx[j + 1].shape[0]
        il_g, n_g, ir_g = np.meshgrid(
            np.arange(D_L, dtype=np.int32),
            np.arange(N, dtype=np.int32),
            np.arange(D_R, dtype=np.int32),
            indexing="ij",
        )
        il_flat, n_flat, ir_flat = il_g.ravel(), n_g.ravel(), ir_g.ravel()
        left_part = left_idx[j][il_flat]
        local_part = n_flat[:, None]
        right_part = right_idx[j + 1][ir_flat]
        all_idx = np.concatenate([left_part, local_part, right_part], axis=1).astype(np.int32)
        values = np.asarray(fn(all_idx), dtype=out_dtype)
        F = values.reshape(D_L * N, D_R)

        U, S, Vh = np.linalg.svd(F, full_matrices=False)
        S0 = float(S[0]) if len(S) else 0.0
        if S0 > 0:
            D_keep = max(1, min(D_max, int((S > tol * S0).sum())))
        else:
            D_keep = 1

        pivot_rows = _maxvol_pivot_rows(U[:, :D_keep], D_keep)
        D_keep = len(pivot_rows)

        if j < d - 1:
            F_pivot = F[pivot_rows, :].astype(out_dtype)
            core_mat = F.astype(out_dtype) @ np.linalg.pinv(F_pivot)
            cores[j] = core_mat.reshape(D_L, N, D_keep)
            new_left = []
            for row in pivot_rows:
                il = int(row) // N
                n = int(row) % N
                new_left.append(np.append(left_idx[j][il], n).astype(np.int32))
            left_idx.append(np.array(new_left, dtype=np.int32))
        else:
            S_trunc = S[:D_keep]
            U_k = U[:, :D_keep].astype(out_dtype)
            cores[j] = (U_k * S_trunc[None, :]).reshape(D_L, N, D_keep)
    return cores, left_idx


def _augment_right_idx_global(fn, cores, right_idx, N, d, D_max, rng, dtype,
                              global_pool, extra_candidates):
    """Inject worst-residual configs into right_idx at every bond (the native
    analog of xfac TensorCI2.addPivotsAllBonds). Returns (new_right_idx, n_added).

    Candidate pool is uniform-random over the grid (the operator exp(iβV) is
    ψ-independent, so uniform coverage is the right pool) plus any
    extra_candidates. For each worst config c, suffix c[j:] is appended to
    right_idx[j] for j=1..d-1; capped at D_max + n_add so injected pivots survive.
    """
    pool = int(global_pool) if global_pool and global_pool > 0 else max(2000, 8 * D_max)
    cand = rng.integers(0, N, size=(pool, d), dtype=np.int32)
    if extra_candidates is not None and len(extra_candidates) > 0:
        cand = np.vstack([cand, np.asarray(extra_candidates, dtype=np.int32)])
    cand = np.unique(cand, axis=0)

    approx = _tt_eval(cores, cand)
    true = np.asarray(fn(cand), dtype=dtype)
    err = np.abs(approx - true)
    scale = float(np.max(np.abs(true))) + 1e-300
    if not np.isfinite(err).any() or err.max() <= 1e-12 * scale:
        return right_idx, 0

    n_add = max(1, D_max // 8)
    worst = cand[np.argsort(err)[-n_add:]]
    new_right = [r.copy() for r in right_idx]
    for j in range(1, d):
        suff = worst[:, j:].astype(np.int32)
        merged = np.unique(np.vstack([right_idx[j], suff]), axis=0)
        if merged.shape[0] > D_max + n_add:
            merged = merged[:D_max + n_add]
        new_right[j] = merged
    return new_right, int(n_add)


def tt_cross(
    fn: Callable[[np.ndarray], np.ndarray],
    N: int,
    d: int,
    D_max: int,
    n_sweeps: int = 3,
    tol: float = 1e-6,
    seed: int = 0,
    init_right_idx: list[np.ndarray] | None = None,
    return_right_idx: bool = False,
    n_global: int = 0,
    global_pool: int = 0,
    extra_candidates: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build a TT (MPS) approximation of fn: {0,...,N-1}^d → scalar.

    Args:
        fn:    callable (M, d) int32 → (M,) float or complex
        N, d:  local dim, chain length
        D_max: max bond dimension
        n_sweeps: alternating L→R / R→L pivot refinement passes
        tol:   SVD relative truncation tolerance
        seed:  RNG for initial right-index sets
        init_right_idx: warm-start right-index sets (list of d+1 int arrays)
        return_right_idx: if True, also return the final right_idx list (used
            to warm-start the next call when fn varies smoothly between calls).

    Returns:
        cores: d-list of left-canonical TT cores; dtype follows fn output.
        If return_right_idx, returns (cores, right_idx_final).
    """
    rng = np.random.default_rng(seed)

    if init_right_idx is not None:
        right_idx = [np.asarray(r, dtype=np.int32) for r in init_right_idx]
    else:
        right_idx = []
        for j in range(d + 1):
            n_right = d - j
            if n_right == 0:
                right_idx.append(np.zeros((1, 0), dtype=np.int32))
            else:
                D_R = min(D_max, N**n_right)
                right_idx.append(rng.integers(0, N, size=(D_R, n_right), dtype=np.int32))

    probe_val = np.asarray(fn(rng.integers(0, N, size=(1, d), dtype=np.int32)))
    out_dtype = np.complex128 if np.issubdtype(probe_val.dtype, np.complexfloating) else np.float64

    cores: list[np.ndarray] = [None] * d  # type: ignore
    left_idx: list[np.ndarray] = [np.zeros((1, 0), dtype=np.int32)]

    for sweep in range(n_sweeps):
        cores, left_idx = _forward_sweep(
            fn, right_idx, N, d, D_max, tol, out_dtype)
        if sweep < n_sweeps - 1:
            right_idx = _update_right_indices(fn, left_idx, right_idx, N, d, D_max, out_dtype)

    # Global-pivot refinement rounds (off by default; n_global=0 → behavior
    # unchanged). Native analog of xfac TensorCI2.addPivotsAllBonds: find
    # worst-residual configs over a candidate pool, inject their right suffixes
    # at every bond, and re-sweep. Recovers modes a local sweep misses; bounded
    # by D_max so it cannot fix a genuinely rank-deficient operator.
    for _r in range(int(n_global)):
        right_idx, n_added = _augment_right_idx_global(
            fn, cores, right_idx, N, d, D_max, rng, out_dtype,
            global_pool, extra_candidates)
        if n_added == 0:
            break
        cores, left_idx = _forward_sweep(
            fn, right_idx, N, d, D_max, tol, out_dtype)

    if return_right_idx:
        return cores, right_idx
    return cores


def elementwise_product_mps(
    mps_a: list[np.ndarray],
    mps_b: list[np.ndarray],
    D_max: int,
) -> list[np.ndarray]:
    """Hadamard product C(x) = A(x)·B(x). Bond dim D_a·D_b before SVD truncation."""
    d = len(mps_a)
    cores: list[np.ndarray] = []
    for j in range(d):
        A = mps_a[j].astype(np.complex128)
        B = mps_b[j].astype(np.complex128)
        D_La, N_loc, D_Ra = A.shape
        D_Lb, _, D_Rb = B.shape
        C = np.einsum("anc,bnd->abncd", A, B)
        C = C.reshape(D_La * D_Lb, N_loc, D_Ra * D_Rb)
        cores.append(C)
    truncated, _ = truncate_mps(cores, D_max)
    return truncated
