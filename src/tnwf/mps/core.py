"""MPS core helpers: dense ↔ MPS, norm, truncation, sampling, evaluation.

MPS format
----------
A list of complex128 cores [A_0, ..., A_{d-1}] in left-canonical form:
    A_0:           (1,   N, D_1)
    A_j (interior): (D_j, N, D_{j+1})
    A_{d-1}:       (D_{d-1}, N, 1)

The physical index (axis=1) runs over the N local basis states at each site.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# dense ↔ MPS
# ---------------------------------------------------------------------------

def dense_to_mps(v: np.ndarray, N: int, d: int, D_max: int) -> list[np.ndarray]:
    """SVD-compress a dense (N^d,) vector into a left-canonical MPS."""
    v = np.asarray(v, dtype=np.complex128)
    cores: list[np.ndarray] = []
    remainder = v.reshape(1, N**d)

    for j in range(d):
        D_left = remainder.shape[0]
        N_right = max(remainder.shape[1] // N, 1)
        mat = remainder.reshape(D_left * N, N_right)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        D_keep = min(S.shape[0], D_max)
        U = U[:, :D_keep]
        S = S[:D_keep]
        Vh = Vh[:D_keep, :]

        if j < d - 1:
            cores.append(U.reshape(D_left, N, D_keep))
            remainder = S[:, None] * Vh
        else:
            # absorb final S into last core
            cores.append((U * S[None, :]).reshape(D_left, N, D_keep))
    return cores


def mps_to_dense(mps: list[np.ndarray], N: int, d: int) -> np.ndarray:
    """Contract MPS into a dense (N^d,) complex vector."""
    state = mps[0].reshape(N, -1)  # (N, D_1)
    for j in range(1, d):
        D_L, _, D_R = mps[j].shape
        state = state @ mps[j].reshape(D_L, N * D_R)
        state = state.reshape(-1, D_R)
    return state.ravel()


# ---------------------------------------------------------------------------
# Norm + diagnostics
# ---------------------------------------------------------------------------

def mps_norm(mps: list[np.ndarray]) -> float:
    """‖MPS‖ via sequential transfer matrix.

    T_{j+1}[c,d] = Σ_{a,b,n} T_j[a,b] · A_j[a,n,c] · conj(A_j[b,n,d])
    """
    T = np.ones((1, 1), dtype=np.complex128)
    for A in mps:
        T = np.einsum("ab,anc,bnd->cd", T, A, A.conj())
    return math.sqrt(max(float(np.real(T[0, 0])), 0.0))


def chi_max(mps: list[np.ndarray]) -> int:
    """Maximum bond dimension across internal bonds."""
    if len(mps) <= 1:
        return 1
    return max(A.shape[2] for A in mps[:-1])


# ---------------------------------------------------------------------------
# Truncation (left-to-right SVD sweep)
# ---------------------------------------------------------------------------

def truncate_mps(
    mps: list[np.ndarray],
    D_max: int,
    tol: float = 0.0,
) -> tuple[list[np.ndarray], float]:
    """Left-to-right SVD sweep: left-canonicalise + truncate bonds to D_max.

    Returns (truncated_mps, total_truncation_error). When tol>0, the kept rank
    is min(D_max, |{S_i > tol·S_0}|).
    """
    d = len(mps)
    cores = [A.copy() for A in mps]
    total_err = 0.0

    for j in range(d - 1):
        D_L, N_loc, D_R_old = cores[j].shape
        mat = cores[j].reshape(D_L * N_loc, D_R_old)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        if tol > 0.0 and S[0] > 0.0:
            D_keep = max(1, min(D_max, int((S > tol * S[0]).sum())))
        else:
            D_keep = min(len(S), D_max)
        S_norm_sq = float(np.sum(S**2))
        total_err += float(np.sum(S[D_keep:] ** 2)) / max(S_norm_sq, 1e-30)

        U = U[:, :D_keep]
        S_t = S[:D_keep]
        Vh = Vh[:D_keep, :]

        cores[j] = U.reshape(D_L, N_loc, D_keep)
        D_L_next, N_next, D_R_next = cores[j + 1].shape
        SV = S_t[:, None] * Vh
        cores[j + 1] = (SV @ cores[j + 1].reshape(D_L_next, N_next * D_R_next)).reshape(
            D_keep, N_next, D_R_next
        )

    return cores, total_err


# ---------------------------------------------------------------------------
# Autoregressive sampling on a left-canonical MPS
# ---------------------------------------------------------------------------

def sample_mps_indices(
    mps: list[np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples multi-indices from |ψ(x)|² / ‖ψ‖². Returns (n_samples, d) int32."""
    d = len(mps)
    indices = np.zeros((n_samples, d), dtype=np.int32)
    v = np.ones((n_samples, 1), dtype=np.complex128)

    for k, A in enumerate(mps):
        D_L, N_loc, D_R = A.shape
        vA = (v @ A.reshape(D_L, N_loc * D_R)).reshape(n_samples, N_loc, D_R)
        probs = np.einsum("snr,snr->sn", vA, vA.conj()).real
        probs = np.clip(probs, 0.0, None)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs /= np.where(row_sums > 0, row_sums, 1.0)
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(n_samples)[:, None]
        x_k = np.clip((u > cdf).sum(axis=1).astype(np.int32), 0, N_loc - 1)
        indices[:, k] = x_k
        v = vA[np.arange(n_samples), x_k, :]

    return indices


def right_canonicalize(mps: list[np.ndarray]) -> list[np.ndarray]:
    """Right-canonical form via R→L QR sweep. Σ_{n,b} A_j[a,n,b] A_j*[a',n,b] = δ for j ≥ 1."""
    out = [c.copy() for c in mps]
    d = len(out)
    for j in range(d - 1, 0, -1):
        D_L, N_loc, D_R = out[j].shape
        mat = out[j].reshape(D_L, N_loc * D_R)              # (D_L, N*D_R)
        Q, R = np.linalg.qr(mat.T)                          # mat.T = Q · R, Q: (N*D_R, D_new)
        D_new = Q.shape[1]
        out[j] = Q.T.reshape(D_new, N_loc, D_R)
        out[j - 1] = np.einsum("anb,cb->anc", out[j - 1], R)
    return out


def apply_K_step_mps(
    mps: list[np.ndarray],
    alpha: float,
    N: int,
    L: float,
) -> list[np.ndarray]:
    """e^{iαK} on MPS via per-site 1D FFT. K = ⊕_j K^(1) → factorises, no bond growth."""
    lam_1d = 0.5 * (2.0 * math.pi / L) ** 2 * (np.arange(N) - N / 2.0) ** 2
    sign_1d = np.array([(-1.0) ** k for k in range(N)])
    phase = np.exp(1j * alpha * lam_1d)
    out = []
    for A in mps:
        A_s = A * sign_1d[None, :, None]
        A_f = np.fft.fft(A_s, axis=1)
        A_r = A_f * phase[None, :, None]
        A_i = np.fft.ifft(A_r, axis=1)
        out.append(A_i * sign_1d[None, :, None])
    return out


def eval_mps_at_indices(mps: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    """Evaluate ψ(x) at integer multi-indices. indices: (n, d) int. Returns (n,) complex."""
    d = len(mps)
    n = indices.shape[0]
    v = np.ones((n, 1), dtype=np.complex128)
    for k in range(d):
        A = mps[k]                                    # (D_L, N, D_R)
        rows = A[:, indices[:, k], :]                 # (D_L, n, D_R)
        rows = np.transpose(rows, (1, 0, 2))          # (n, D_L, D_R)
        v = np.einsum("ni,nij->nj", v, rows)
    return v[:, 0]
