"""prrLU-based cross interpolation primitive (numerically stable rank-r CUR).

Greedy max-entry pivot selection with rank-1 deflation (Oseledets &
Tyrtyshnikov, 2010). Used by ACI Hadamard for 2-site factorisation without SVD.
"""
from __future__ import annotations

import numpy as np


def prrlu(
    F: np.ndarray,
    max_rank: int,
    tol: float = 1e-10,
) -> tuple[list[int], list[int]]:
    """Greedy partial row/row-restricted LU. Returns (row_pivots, col_pivots).

    Stops when |R[i,j]| ≤ tol·|F|_max or after max_rank pivots. O(r·M·K).
    """
    M, K = F.shape
    r_max = min(max_rank, M, K)
    if r_max == 0:
        return [], []

    R = np.array(F, dtype=np.complex128, copy=True)
    abs_F_max = float(np.max(np.abs(R)))
    if abs_F_max == 0.0:
        return [], []

    threshold = tol * abs_F_max
    row_idx: list[int] = []
    col_idx: list[int] = []

    for _ in range(r_max):
        flat = int(np.argmax(np.abs(R)))
        i, j = divmod(flat, K)
        if abs(R[i, j]) <= threshold:
            break
        row_idx.append(i)
        col_idx.append(j)
        col_vec = R[:, j].copy()
        row_vec = R[i, :].copy()
        R -= np.outer(col_vec, row_vec) / R[i, j]

    return row_idx, col_idx


def ci_factorize(
    Pi: np.ndarray,
    max_rank: int,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], float]:
    """Cross-interpolation factorisation. Pi ≈ Pi[:, J] @ inv(Pi[I,J]) @ Pi[I,:].

    Returns (T_l (M,r), T_r (r,K), row_pivots, col_pivots, |last_pivot|).
    """
    Pi_c = np.asarray(Pi, dtype=np.complex128)
    M, K = Pi_c.shape
    row_idx, col_idx = prrlu(Pi_c, max_rank, tol)

    if len(row_idx) == 0:
        return (
            np.zeros((M, 1), dtype=np.complex128),
            np.zeros((1, K), dtype=np.complex128),
            [0],
            [0],
            0.0,
        )

    C = Pi_c[:, col_idx]                                      # (M, r)
    R_mat = Pi_c[row_idx, :]                                  # (r, K)
    P = Pi_c[np.ix_(row_idx, col_idx)]                        # (r, r) pivot submatrix
    T_r, _, _, _ = np.linalg.lstsq(P, R_mat, rcond=None)      # (r, K)
    T_l = C                                                    # (M, r)
    pivot_error = float(abs(Pi_c[row_idx[-1], col_idx[-1]]))
    return T_l, T_r, row_idx, col_idx, pivot_error


def ldu_truncate(mat: np.ndarray, D_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank-D_max cross factorisation as fallback for failed SVD."""
    T_l, T_r, _, _, _ = ci_factorize(mat, D_max, tol=0.0)
    return T_l, T_r
