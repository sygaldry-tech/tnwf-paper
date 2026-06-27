"""ACI (Alternating Cross Interpolation) Hadamard product on TT/MPS.

Computes Y ≈ tt_a ⊙ tt_b without ever materialising the full N^d Hadamard.
Uses prrLU-based ci_factorize (numerically stable when D >> D_eff).

Reference: arXiv:2604.00037. Numpy-only port of the research prototype's wavefunction/aci_vstep.py.
"""
from __future__ import annotations

import numpy as np

from tnwf.mps.ldu_cross import ci_factorize


def _assemble_pi(
    L_n: np.ndarray,
    A_l: np.ndarray,
    A_r: np.ndarray,
    R_n: np.ndarray,
    r_l: int,
    N: int,
) -> np.ndarray:
    """2-site frame assembly via lazy 3-step matmul. Returns (r_l·N, N·r_r) cmplx."""
    chi_l, _, chi_m = A_l.shape
    _, _, chi_r = A_r.shape
    r_r = R_n.shape[1]
    step1 = (L_n @ A_l.reshape(chi_l, N * chi_m)).reshape(r_l * N, chi_m)
    step2 = step1 @ A_r.reshape(chi_m, N * chi_r)
    return (step2.reshape(r_l * N * N, chi_r) @ R_n).reshape(r_l * N, N * r_r)


def _update_left_frame(
    L_n: np.ndarray, A_l: np.ndarray, row_idx: list[int], r_l: int, N: int
) -> np.ndarray:
    chi_l, _, chi_m = A_l.shape
    full_L = (L_n @ A_l.reshape(chi_l, N * chi_m)).reshape(r_l * N, chi_m)
    return full_L[row_idx, :]


def _update_right_frame(
    A_r: np.ndarray, R_n: np.ndarray, col_idx: list[int], N: int
) -> np.ndarray:
    chi_m, _, chi_r = A_r.shape
    r_r = R_n.shape[1]
    full_R = (A_r.reshape(chi_m * N, chi_r) @ R_n).reshape(chi_m, N * r_r)
    return full_R[:, col_idx]


def aci_hadamard(
    tt_a: list[np.ndarray],
    tt_b: list[np.ndarray],
    D_max: int,
    tol: float = 1e-6,
    n_sweeps: int = 4,
) -> list[np.ndarray]:
    """ACI Hadamard product Y ≈ tt_a ⊙ tt_b. Adaptive bond growth up to D_max."""
    d = len(tt_a)
    if d == 1:
        A = np.asarray(tt_a[0], dtype=np.complex128)
        B = np.asarray(tt_b[0], dtype=np.complex128)
        return [A * B]

    N = tt_a[0].shape[1]
    tt_a = [np.asarray(A, dtype=np.complex128) for A in tt_a]
    tt_b = [np.asarray(A, dtype=np.complex128) for A in tt_b]

    L_a: list[np.ndarray] = [None] * d  # type: ignore
    L_b: list[np.ndarray] = [None] * d  # type: ignore
    R_a: list[np.ndarray] = [None] * (d + 1)  # type: ignore
    R_b: list[np.ndarray] = [None] * (d + 1)  # type: ignore

    L_a[0] = np.ones((1, 1), dtype=np.complex128)
    L_b[0] = np.ones((1, 1), dtype=np.complex128)
    R_a[d] = np.ones((1, 1), dtype=np.complex128)
    R_b[d] = np.ones((1, 1), dtype=np.complex128)

    # Initialise right frames (rank-1 right boundary)
    for ell in range(d - 1, -1, -1):
        A_a = tt_a[ell]
        A_b = tt_b[ell]
        chi_l_a, _, chi_r_a = A_a.shape
        chi_l_b, _, chi_r_b = A_b.shape
        r_r_a = R_a[ell + 1].shape[1]
        r_r_b = R_b[ell + 1].shape[1]
        full_R_a = (A_a.reshape(chi_l_a * N, chi_r_a) @ R_a[ell + 1]).reshape(chi_l_a, N * r_r_a)
        full_R_b = (A_b.reshape(chi_l_b * N, chi_r_b) @ R_b[ell + 1]).reshape(chi_l_b, N * r_r_b)
        R_a[ell] = full_R_a[:, :1]
        R_b[ell] = full_R_b[:, :1]

    Y: list[np.ndarray] = [None] * d  # type: ignore

    for _sweep in range(n_sweeps):
        max_pivot_err = 0.0

        # L→R sweep
        for ell in range(d - 1):
            r_l = L_a[ell].shape[0]
            R_a_right = R_a[ell + 2] if ell + 2 <= d else np.ones((1, 1), dtype=np.complex128)
            R_b_right = R_b[ell + 2] if ell + 2 <= d else np.ones((1, 1), dtype=np.complex128)
            r_r = R_a_right.shape[1]

            Pi_a = _assemble_pi(L_a[ell], tt_a[ell], tt_a[ell + 1], R_a_right, r_l, N)
            Pi_b = _assemble_pi(L_b[ell], tt_b[ell], tt_b[ell + 1], R_b_right, r_l, N)
            Pi = Pi_a * Pi_b

            T_l, T_r, row_idx, col_idx, pivot_err = ci_factorize(Pi, D_max, tol=tol * 1e-2)
            r_new = len(row_idx)
            max_pivot_err = max(max_pivot_err, pivot_err)

            Y[ell] = T_l.reshape(r_l, N, r_new)
            if ell == d - 2:
                Y[d - 1] = T_r.reshape(r_new, N, 1)

            L_a[ell + 1] = _update_left_frame(L_a[ell], tt_a[ell], row_idx, r_l, N)
            L_b[ell + 1] = _update_left_frame(L_b[ell], tt_b[ell], row_idx, r_l, N)

        # R→L sweep
        for ell in range(d - 2, -1, -1):
            r_l = L_a[ell].shape[0]
            R_a_right = R_a[ell + 2] if ell + 2 <= d else np.ones((1, 1), dtype=np.complex128)
            R_b_right = R_b[ell + 2] if ell + 2 <= d else np.ones((1, 1), dtype=np.complex128)
            r_r = R_a_right.shape[1]

            Pi_a = _assemble_pi(L_a[ell], tt_a[ell], tt_a[ell + 1], R_a_right, r_l, N)
            Pi_b = _assemble_pi(L_b[ell], tt_b[ell], tt_b[ell + 1], R_b_right, r_l, N)
            Pi = Pi_a * Pi_b

            T_l, T_r, row_idx, col_idx, pivot_err = ci_factorize(Pi, D_max, tol=tol * 1e-2)
            r_new = len(row_idx)
            max_pivot_err = max(max_pivot_err, pivot_err)

            Y[ell + 1] = T_r.reshape(r_new, N, r_r)
            if ell == 0:
                Y[0] = T_l.reshape(r_l, N, r_new)

            R_a[ell + 1] = _update_right_frame(tt_a[ell + 1], R_a_right, col_idx, N)
            R_b[ell + 1] = _update_right_frame(tt_b[ell + 1], R_b_right, col_idx, N)

        if _sweep >= 1 and max_pivot_err < tol:
            break

    for i, core in enumerate(Y):
        if core is None:
            Y[i] = np.zeros((1, N, 1), dtype=np.complex128)

    return Y
