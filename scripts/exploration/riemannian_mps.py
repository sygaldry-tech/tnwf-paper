"""Riemannian-manifold primitives for real-valued MPS optimization.

For a real-valued MPS with cores ``A_j`` shape ``(D_{j-1}, N, D_j)``, the
parameter space is a smooth manifold modulo gauge: bond-j gauge action is
``A_j → A_j G``, ``A_{j+1} → G^{-1} A_{j+1}`` for invertible G. Fixing a
mixed-canonical form (cores ``j < center`` left-orthogonal, cores
``j > center`` right-orthogonal, gauge absorbed at ``center``) makes the
left/right-canonical cores live on a Stiefel manifold:

    St(N·D_L, D_R) = {A ∈ R^{(N·D_L) × D_R} : A^T A = I_{D_R}}     (left-canon)
    St(N·D_R, D_L) = {A ∈ R^{D_L × (N·D_R)} (matricized)
                     : A A^T = I_{D_L}}                              (right-canon)

The center core is unconstrained.

This module provides:

  * ``to_mixed_canonical`` — QR sweep to put V_mps into mixed-canonical form.
  * ``gauge_project`` — orthogonal projection of a Euclidean core gradient
    onto the Stiefel tangent space at each constrained core.
  * ``retract_qr`` — take a step along a tangent vector and QR-retract back
    onto the manifold; preserves bond dimensions exactly.

All routines operate on lists of float64 numpy arrays.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------

def to_mixed_canonical(V_mps: list[np.ndarray],
                       center: int | None = None) -> list[np.ndarray]:
    """Put V_mps into mixed-canonical form with gauge absorbed at ``center``.

    Cores ``j < center`` become left-orthogonal: matricized as
    ``(D_L · N, D_R)``, ``A_j^T A_j = I``.
    Cores ``j > center`` become right-orthogonal: matricized as
    ``(D_L, N · D_R)``, ``A_j A_j^T = I``.

    Bond dimensions are preserved exactly (no truncation).

    Parameters
    ----------
    V_mps : list of np.ndarray
        Cores shaped ``(D_{j-1}, N, D_j)``.
    center : int, optional
        Index of the unconstrained center core. Defaults to ``d - 1``.

    Returns
    -------
    list of np.ndarray
        New cores in mixed-canonical form (input untouched).
    """
    d = len(V_mps)
    if center is None:
        center = d - 1
    if not 0 <= center < d:
        raise ValueError(f"center must be in [0, {d - 1}], got {center}")

    out = [A.copy() for A in V_mps]

    # Left sweep: cores 0 .. center-1 → left-orthogonal via QR.
    for j in range(center):
        D_L, N_loc, D_R = out[j].shape
        mat = out[j].reshape(D_L * N_loc, D_R)
        Q, R = np.linalg.qr(mat)                              # Q: (D_L*N, D_keep)
        D_keep = Q.shape[1]
        out[j] = Q.reshape(D_L, N_loc, D_keep)
        # Absorb R into next core
        D_L2, N_2, D_R2 = out[j + 1].shape
        assert D_L2 == D_R, (
            f"bond mismatch at site {j}: R is ({R.shape}), next D_L={D_L2}")
        # R: (D_keep, D_R), next core: (D_R, N, D_R2)
        out[j + 1] = np.einsum("ab,bnc->anc", R, out[j + 1])

    # Right sweep: cores d-1 .. center+1 → right-orthogonal via QR.
    for j in range(d - 1, center, -1):
        D_L, N_loc, D_R = out[j].shape
        mat = out[j].reshape(D_L, N_loc * D_R)
        # QR of the transpose: mat^T = Q R → mat = R^T Q^T, Q^T is row-orth.
        Q, R = np.linalg.qr(mat.T)                            # Q: (N*D_R, D_keep)
        D_keep = Q.shape[1]
        out[j] = Q.T.reshape(D_keep, N_loc, D_R)
        # R: (D_keep, D_L); absorb R^T into previous core
        D_Lp, N_p, D_Rp = out[j - 1].shape
        assert D_Rp == D_L, (
            f"bond mismatch at site {j - 1}: prev D_R={D_Rp}, this D_L={D_L}")
        out[j - 1] = np.einsum("anb,cb->anc", out[j - 1], R)

    return out


# ---------------------------------------------------------------------------
# Stiefel tangent projection (gauge_project)
# ---------------------------------------------------------------------------

def _proj_left_stiefel(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project gradient G onto tangent space of left-Stiefel at A.

    A is left-orthogonal: A reshaped to ``(D_L·N, D_R)`` satisfies
    ``A_mat^T A_mat = I``. Tangent vectors Z at A must satisfy
    ``Z_mat^T A_mat + A_mat^T Z_mat = 0`` (skew-symmetric).
    The canonical (Euclidean-metric) projection is:
        proj(G) = G − A · sym(A^T G)
    where sym(X) = (X + X^T) / 2.
    """
    D_L, N_loc, D_R = A.shape
    A_mat = A.reshape(D_L * N_loc, D_R)
    G_mat = G.reshape(D_L * N_loc, D_R)
    AtG = A_mat.T @ G_mat                                     # (D_R, D_R)
    sym = 0.5 * (AtG + AtG.T)
    proj = G_mat - A_mat @ sym
    return proj.reshape(D_L, N_loc, D_R)


def _proj_right_stiefel(A: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Right-canonical version: A_mat = A reshaped to ``(D_L, N·D_R)``
    satisfies ``A_mat A_mat^T = I``. Tangent satisfies
    ``Z_mat A_mat^T + A_mat Z_mat^T = 0``, projection:
        proj(G) = G − sym(G A^T) · A
    """
    D_L, N_loc, D_R = A.shape
    A_mat = A.reshape(D_L, N_loc * D_R)
    G_mat = G.reshape(D_L, N_loc * D_R)
    GAt = G_mat @ A_mat.T                                     # (D_L, D_L)
    sym = 0.5 * (GAt + GAt.T)
    proj = G_mat - sym @ A_mat
    return proj.reshape(D_L, N_loc, D_R)


def gauge_project(grad_cores: list[np.ndarray],
                  V_mps_canon: list[np.ndarray],
                  center: int | None = None) -> list[np.ndarray]:
    """Project Euclidean core gradients onto the Stiefel tangent space.

    Constrained cores (``j != center``) get their gauge-orbit component
    subtracted; the center core's gradient is returned unchanged.

    Parameters
    ----------
    grad_cores : list of np.ndarray
        Euclidean dL/dA_j, same shapes as V_mps_canon.
    V_mps_canon : list of np.ndarray
        MPS in mixed-canonical form (output of ``to_mixed_canonical``).
    center : int, optional
        Index of unconstrained core. Defaults to ``d - 1``.

    Returns
    -------
    list of np.ndarray
        Riemannian gradients on the Stiefel tangent spaces.
    """
    d = len(V_mps_canon)
    if center is None:
        center = d - 1
    out: list[np.ndarray] = []
    for j in range(d):
        if j < center:
            out.append(_proj_left_stiefel(V_mps_canon[j], grad_cores[j]))
        elif j > center:
            out.append(_proj_right_stiefel(V_mps_canon[j], grad_cores[j]))
        else:
            out.append(grad_cores[j].copy())
    return out


# ---------------------------------------------------------------------------
# QR retraction
# ---------------------------------------------------------------------------

def _retract_left_qr(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """QR retraction on the left-Stiefel manifold:
       R(A, dA) = qr(A + dA)
    Returns a new left-orthogonal core of the same shape.
    """
    D_L, N_loc, D_R = A.shape
    M = (A + dA).reshape(D_L * N_loc, D_R)
    Q, R = np.linalg.qr(M)
    # Sign-fix: enforce sign convention on diag(R) so retraction is continuous.
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    Q = Q * sgn[None, :]
    return Q.reshape(D_L, N_loc, D_R)


def _retract_right_qr(A: np.ndarray, dA: np.ndarray) -> np.ndarray:
    """QR retraction on the right-Stiefel manifold (transposed convention)."""
    D_L, N_loc, D_R = A.shape
    M = (A + dA).reshape(D_L, N_loc * D_R)
    Q, R = np.linalg.qr(M.T)
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    Q = Q * sgn[None, :]
    return Q.T.reshape(D_L, N_loc, D_R)


def retract_qr(V_mps_canon: list[np.ndarray],
               tangent_cores: list[np.ndarray],
               step_size: float = 1.0,
               center: int | None = None) -> list[np.ndarray]:
    """Take a step along ``tangent_cores`` and retract back to the manifold.

    Constrained cores use QR retraction; center core takes a Euclidean step.
    Bond dimensions are preserved exactly.

    Parameters
    ----------
    V_mps_canon : list of np.ndarray
        Current point on the manifold (mixed-canonical form).
    tangent_cores : list of np.ndarray
        Tangent vectors (typically the Riemannian gradient or an Adam update
        direction). Shapes match V_mps_canon.
    step_size : float
        Scaling applied to all tangent_cores before retraction.
    center : int, optional
        Index of unconstrained core.

    Returns
    -------
    list of np.ndarray
        New point on the manifold, mixed-canonical form preserved.
    """
    d = len(V_mps_canon)
    if center is None:
        center = d - 1
    out: list[np.ndarray] = []
    for j in range(d):
        if j < center:
            out.append(_retract_left_qr(V_mps_canon[j], step_size * tangent_cores[j]))
        elif j > center:
            out.append(_retract_right_qr(V_mps_canon[j], step_size * tangent_cores[j]))
        else:
            out.append(V_mps_canon[j] + step_size * tangent_cores[j])
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def stiefel_orthogonality_error(V_mps_canon: list[np.ndarray],
                                 center: int | None = None) -> float:
    """Max ‖A^T A − I‖_F (or ‖A A^T − I‖_F) over all constrained cores."""
    d = len(V_mps_canon)
    if center is None:
        center = d - 1
    max_err = 0.0
    for j in range(d):
        if j == center:
            continue
        A = V_mps_canon[j]
        D_L, N_loc, D_R = A.shape
        if j < center:
            M = A.reshape(D_L * N_loc, D_R)
            err = np.linalg.norm(M.T @ M - np.eye(D_R))
        else:
            M = A.reshape(D_L, N_loc * D_R)
            err = np.linalg.norm(M @ M.T - np.eye(D_L))
        max_err = max(max_err, err)
    return float(max_err)


def gauge_component_norm(grad_cores: list[np.ndarray],
                          V_mps_canon: list[np.ndarray],
                          center: int | None = None) -> float:
    """‖grad − Π_tangent(grad)‖ summed over Stiefel cores. Zero if grad is
    already tangent."""
    proj = gauge_project(grad_cores, V_mps_canon, center)
    total = 0.0
    for g, p in zip(grad_cores, proj):
        total += float(np.linalg.norm(g - p) ** 2)
    return float(np.sqrt(total))
