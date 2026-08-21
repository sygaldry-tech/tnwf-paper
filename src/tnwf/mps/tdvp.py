"""1-site and 2-site TDVP for diagonal MPOs (V_t potential V-step).

Applies exp(τ·V)|ψ⟩ where V is given as TT cores, by projecting the dynamics
onto the MPS tangent manifold. Exploits the block-diagonal structure of the
effective Hamiltonian for ~N⁴× speedup vs a generic-MPO TDVP.

Reference: Haegeman et al., Phys. Rev. B 94, 165116 (2016).
Numpy-only TDVP port (no ThreadPool).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply


def _robust_svd(mat: np.ndarray, full_matrices: bool = False):
    """``np.linalg.svd`` with progressively more aggressive fallbacks.

    At high (N, d) the 2-site theta tensor can become numerically degenerate
    — flat singular spectra, huge dynamic range, occasional NaN/Inf from
    upstream BLAS — and LAPACK's default ``gesdd`` raises
    ``LinAlgError: SVD did not converge`` (with the tell-tale
    ``DLASCL parameter 4 had an illegal value`` from the internal scaler).

    Fallback chain:
      1. ``np.linalg.svd`` (gesdd, fast happy path).
      2. ``scipy.linalg.svd`` with ``gesvd`` driver (slower, more stable).
      3. Sanitise NaN/Inf via ``np.nan_to_num``, retry gesvd.
      4. Normalize by max-abs so the dynamic range fits in float64's
         well-scaled regime, SVD, scale singular values back.
      5. Add a tiny diagonal regulariser (1e-14 · max-abs) and retry.

    Algorithm is unchanged on the happy path — only ill-conditioned cells
    take the fallback branches.
    """
    try:
        return np.linalg.svd(mat, full_matrices=full_matrices)
    except np.linalg.LinAlgError:
        pass

    from scipy.linalg import svd as _scipy_svd
    try:
        return _scipy_svd(mat, full_matrices=full_matrices,
                          lapack_driver="gesvd")
    except np.linalg.LinAlgError:
        pass

    clean = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return _scipy_svd(clean, full_matrices=full_matrices,
                          lapack_driver="gesvd")
    except np.linalg.LinAlgError:
        pass

    scale = float(np.max(np.abs(clean)))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    try:
        U, S, Vh = _scipy_svd(clean / scale, full_matrices=full_matrices,
                              lapack_driver="gesvd")
        return U, S * scale, Vh
    except np.linalg.LinAlgError:
        pass

    eps = 1e-14
    m, n = clean.shape
    reg = np.eye(m, n, dtype=clean.dtype) * eps
    U, S, Vh = _scipy_svd(clean / scale + reg, full_matrices=full_matrices,
                          lapack_driver="gesvd")
    return U, S * scale, Vh


def _diag_build_right_envs(cores: list[np.ndarray], mps: list[np.ndarray]):
    d = len(mps)
    R = [None] * (d + 1)
    R[d] = np.ones((1, 1, 1), dtype=np.complex128)
    for j in range(d - 1, -1, -1):
        A = mps[j]
        C = cores[j]
        Rj1 = R[j + 1]
        AR = np.einsum("bsr,acr->bsac", A, Rj1)
        CAR = np.einsum("csr,bsar->csab", C, AR)
        R[j] = np.einsum("asr,csrb->acb", A.conj(), CAR)
    return R


def _diag_update_left_env(L_env, A, C):
    LA = np.einsum("acb,bsf->acsf", L_env, A)
    CLA = np.einsum("cse,acsf->asef", C, LA)
    return np.einsum("asd,asef->def", A.conj(), CLA)


def _diag_compute_single_right_env(A, C, R_next):
    AR = np.einsum("bsr,acr->bsac", A, R_next)
    CAR = np.einsum("csr,bsar->csab", C, AR)
    return np.einsum("asr,csrb->acb", A.conj(), CAR)


def _diag_effective_H_2site_blocks(L_env, C_j, C_j1, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    N_j = C_j.shape[1]
    N_j1 = C_j1.shape[1]
    CC = np.einsum("csr,rtd->cstd", C_j, C_j1)
    blocks = {}
    for s in range(N_j):
        for s1 in range(N_j1):
            CC_ss1 = CC[:, s, s1, :]
            CCR = np.einsum("cd,bdf->cbf", CC_ss1, R_env)
            H = np.einsum("ace,cbf->abef", L_env, CCR)
            blocks[(s, s1)] = H.reshape(D_L * D_R, D_L * D_R)
    return blocks


def _diag_effective_H_1site_blocks(L_env, C, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    N_loc = C.shape[1]
    blocks = {}
    for s in range(N_loc):
        C_s = C[:, s, :]
        CR = np.einsum("cd,bdf->cbf", C_s, R_env)
        H = np.einsum("ace,cbf->abef", L_env, CR)
        blocks[s] = H.reshape(D_L * D_R, D_L * D_R)
    return blocks


def _diag_effective_H_0site(L_env, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    H = np.einsum("ace,bcf->abef", L_env, R_env)
    return H.reshape(D_L * D_R, D_L * D_R)


def tdvp_1site_diagonal(
    cores: list[np.ndarray],
    mps: list[np.ndarray],
    tau: complex,
    n_sweeps: int = 1,
) -> list[np.ndarray]:
    """1-site TDVP for diagonal MPO. Bond dim is fixed (no SVD)."""
    d = len(mps)
    if d < 2:
        return [c.copy() for c in mps]
    A = [c.copy() for c in mps]

    for _ in range(n_sweeps):
        R = _diag_build_right_envs(cores, A)
        L = np.ones((1, 1, 1), dtype=np.complex128)

        # L→R sweep
        for j in range(d):
            D_Lj, N_j, D_Rj = A[j].shape
            blocks_1 = _diag_effective_H_1site_blocks(L, cores[j], R[j + 1])
            A_new = np.zeros_like(A[j])
            for s, H_block in blocks_1.items():
                vec = A[j][:, s, :].reshape(-1)
                A_new[:, s, :] = expm_multiply(H_block * (tau / 2), vec).reshape(D_Lj, D_Rj)
            A[j] = A_new

            if j < d - 1:
                mat = A[j].reshape(D_Lj * N_j, D_Rj)
                Q, R_mat = np.linalg.qr(mat)
                K = Q.shape[1]
                A[j] = Q.reshape(D_Lj, N_j, K)
                L = _diag_update_left_env(L, A[j], cores[j])

                H_0 = _diag_effective_H_0site(L, R[j + 1])
                R_vec_new = expm_multiply(H_0 * (-tau / 2), R_mat.reshape(-1))
                R_mat = R_vec_new.reshape(K, D_Rj)
                A[j + 1] = np.einsum("ab,bsc->asc", R_mat, A[j + 1])
            else:
                L = _diag_update_left_env(L, A[j], cores[j])

        # R→L sweep
        L_all = [None] * (d + 1)
        L_all[0] = np.ones((1, 1, 1), dtype=np.complex128)
        for j in range(d):
            L_all[j + 1] = _diag_update_left_env(L_all[j], A[j], cores[j])
        R_env_j = np.ones((1, 1, 1), dtype=np.complex128)

        for j in range(d - 1, -1, -1):
            D_Lj, N_j, D_Rj = A[j].shape
            blocks_1 = _diag_effective_H_1site_blocks(L_all[j], cores[j], R_env_j)
            A_new = np.zeros_like(A[j])
            for s, H_block in blocks_1.items():
                vec = A[j][:, s, :].reshape(-1)
                A_new[:, s, :] = expm_multiply(H_block * (tau / 2), vec).reshape(D_Lj, D_Rj)
            A[j] = A_new

            if j > 0:
                mat = A[j].reshape(D_Lj, N_j * D_Rj)
                Qt, Rt = np.linalg.qr(mat.T)
                K = Qt.shape[1]
                L_mat = Rt.T
                Q_mat = Qt.T
                A[j] = Q_mat.reshape(K, N_j, D_Rj)
                R_env_j = _diag_compute_single_right_env(A[j], cores[j], R_env_j)

                H_0 = _diag_effective_H_0site(L_all[j], R_env_j)
                L_vec_new = expm_multiply(H_0 * (-tau / 2), L_mat.reshape(-1))
                L_mat = L_vec_new.reshape(D_Lj, K)
                A[j - 1] = np.einsum("asb,bc->asc", A[j - 1], L_mat)
            else:
                R_env_j = _diag_compute_single_right_env(A[j], cores[j], R_env_j)

    return A


def tdvp_2site_diagonal(
    cores: list[np.ndarray],
    mps: list[np.ndarray],
    tau: complex,
    D_max: int,
    n_sweeps: int = 1,
    tol: float = 1e-8,
) -> list[np.ndarray]:
    """2-site TDVP for diagonal MPO. Bond dim grows to D_max via SVD."""
    d = len(mps)
    if d < 2:
        return [c.copy() for c in mps]
    A = [c.copy() for c in mps]

    for _ in range(n_sweeps):
        R = _diag_build_right_envs(cores, A)
        L = np.ones((1, 1, 1), dtype=np.complex128)

        # L→R sweep
        for j in range(d - 1):
            D_Lj, N_j, D_Rj = A[j].shape
            D_Lj1, N_j1, D_Rj1 = A[j + 1].shape
            theta = np.einsum("anb,bmc->anmc", A[j], A[j + 1])

            blocks = _diag_effective_H_2site_blocks(L, cores[j], cores[j + 1], R[j + 2])
            theta_new = np.zeros_like(theta)
            for (s, s1), H_block in blocks.items():
                vec = theta[:, s, s1, :].reshape(-1)
                theta_new[:, s, s1, :] = expm_multiply(H_block * (tau / 2), vec).reshape(D_Lj, D_Rj1)
            theta = theta_new

            mat = theta.reshape(D_Lj * N_j, N_j1 * D_Rj1)
            U, S, Vh = _robust_svd(mat, full_matrices=False)
            S0 = float(S[0]) if len(S) else 1.0
            if tol > 0 and S0 > 0:
                D_keep = max(1, min(D_max, int(np.sum(S > tol * S0))))
            else:
                D_keep = min(len(S), D_max)
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vh = Vh[:D_keep, :]

            A[j] = U.reshape(D_Lj, N_j, D_keep)
            A[j + 1] = (np.diag(S) @ Vh).reshape(D_keep, N_j1, D_Rj1)

            L = _diag_update_left_env(L, A[j], cores[j])

            if j < d - 2:
                blocks_1 = _diag_effective_H_1site_blocks(L, cores[j + 1], R[j + 2])
                D_Lnew, N_new, D_Rnew = A[j + 1].shape
                A_new = np.zeros_like(A[j + 1])
                for s, H_block in blocks_1.items():
                    vec = A[j + 1][:, s, :].reshape(-1)
                    A_new[:, s, :] = expm_multiply(H_block * (-tau / 2), vec).reshape(D_Lnew, D_Rnew)
                A[j + 1] = A_new

        # R→L sweep
        R = _diag_build_right_envs(cores, A)
        L_all = [None] * (d + 1)
        L_all[0] = np.ones((1, 1, 1), dtype=np.complex128)
        for j in range(d):
            L_all[j + 1] = _diag_update_left_env(L_all[j], A[j], cores[j])

        for j in range(d - 2, -1, -1):
            D_Lj, N_j, D_Rj = A[j].shape
            D_Lj1, N_j1, D_Rj1 = A[j + 1].shape
            theta = np.einsum("anb,bmc->anmc", A[j], A[j + 1])

            R_idx = j + 2
            blocks = _diag_effective_H_2site_blocks(L_all[j], cores[j], cores[j + 1], R[R_idx])
            theta_new = np.zeros_like(theta)
            for (s, s1), H_block in blocks.items():
                vec = theta[:, s, s1, :].reshape(-1)
                theta_new[:, s, s1, :] = expm_multiply(H_block * (tau / 2), vec).reshape(D_Lj, D_Rj1)
            theta = theta_new

            mat = theta.reshape(D_Lj * N_j, N_j1 * D_Rj1)
            U, S, Vh = _robust_svd(mat, full_matrices=False)
            S0 = float(S[0]) if len(S) else 1.0
            if tol > 0 and S0 > 0:
                D_keep = max(1, min(D_max, int(np.sum(S > tol * S0))))
            else:
                D_keep = min(len(S), D_max)
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vh = Vh[:D_keep, :]

            A[j] = (U @ np.diag(S)).reshape(D_Lj, N_j, D_keep)
            A[j + 1] = Vh.reshape(D_keep, N_j1, D_Rj1)

            R[j + 1] = _diag_compute_single_right_env(A[j + 1], cores[j + 1], R[j + 2])

            if j > 0:
                blocks_1 = _diag_effective_H_1site_blocks(L_all[j], cores[j], R[j + 1])
                D_Lnew, N_new, D_Rnew = A[j].shape
                A_new = np.zeros_like(A[j])
                for s, H_block in blocks_1.items():
                    vec = A[j][:, s, :].reshape(-1)
                    A_new[:, s, :] = expm_multiply(H_block * (-tau / 2), vec).reshape(D_Lnew, D_Rnew)
                A[j] = A_new

    return A
