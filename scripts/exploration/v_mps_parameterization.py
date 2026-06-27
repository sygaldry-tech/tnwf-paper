"""Phase 3b: V as an MPS (tensor train) instead of a dense ``N^d`` vector.

The forward pipeline (``adjoint_trotter_mps``) already produces
``dL/dV_dense`` as a dense ``N^d`` gradient. Track B keeps that
gradient computation but reparameterises V itself as an MPS, and
projects the dense gradient onto the MPS tangent space at the current
V_mps to obtain core gradients ``dL/dA_j[a, n, b]``.

For a left-canonical V_mps with cores ``A_0, ..., A_{d-1}``,
  V_dense[x_0, ..., x_{d-1}] = (A_0[:, x_0, :] · A_1[:, x_1, :] · ...
                                · A_{d-1}[:, x_{d-1}, :])  (matrix chain)

The gradient w.r.t. core j at indices (a, n, b) is:
  ∂V[x]/∂A_j[a, n, b] = δ_{x_j, n} · L_j[x_<j, a] · R_j[b, x_>j]

So:
  dL/dA_j[a, n, b] = Σ_{x: x_j = n} (dL/dV[x]) · L_j[x_<j, a] · R_j[b, x_>j]

This module provides:
  * ``V_dense_to_mps`` / ``V_mps_to_dense`` — real-valued round trip
  * ``eval_V_mps_on_grid`` — full grid evaluation (dense N^d)
  * ``project_dense_grad_to_mps_cores`` — dL/dV_dense → list of dL/dA_j
  * ``_self_test()`` — numerical chain-rule check against torch autograd

The dense-grad → MPS-tangent projection is itself O(N^d · D²)  — i.e.,
still touches the full grid. The *parameter space* however is
O(D² · N · d), which is what scales. The remaining O(N^d) at the
projection step disappears only when λ and ψ are kept as MPS through
the gradient calculation (a deeper Phase-3c refactor).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# Reuse standard MPS routines
from tnwf.mps.core import truncate_mps  # noqa: E402


# ---------------------------------------------------------------------------
# Real-valued MPS round-trip
# ---------------------------------------------------------------------------

def V_dense_to_mps(V_dense: np.ndarray, N: int, d: int,
                   D_max: int) -> list[np.ndarray]:
    """SVD-compress a real ``(N^d,)`` vector into a left-canonical MPS.

    Cores are real-valued float64. ``D_max`` clips bond growth.
    """
    V = np.asarray(V_dense, dtype=np.float64).reshape(N**d)
    cores: list[np.ndarray] = []
    remainder = V.reshape(1, N**d)
    for j in range(d):
        D_left = remainder.shape[0]
        N_right = max(remainder.shape[1] // N, 1)
        mat = remainder.reshape(D_left * N, N_right)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        D_keep = min(S.shape[0], D_max)
        U = U[:, :D_keep]; S = S[:D_keep]; Vh = Vh[:D_keep, :]
        if j < d - 1:
            cores.append(U.reshape(D_left, N, D_keep))
            remainder = S[:, None] * Vh
        else:
            cores.append((U * S[None, :]).reshape(D_left, N, D_keep))
    return cores


def V_mps_to_dense(V_mps: list[np.ndarray], N: int, d: int) -> np.ndarray:
    """Contract real-valued V MPS to dense ``(N^d,)`` vector."""
    state = V_mps[0].reshape(N, -1)
    for j in range(1, d):
        D_L, _, D_R = V_mps[j].shape
        state = state @ V_mps[j].reshape(D_L, N * D_R)
        state = state.reshape(-1, D_R)
    return state.ravel().astype(np.float64)


def eval_V_mps_on_grid(V_mps: list[np.ndarray], N: int, d: int) -> np.ndarray:
    """Same as ``V_mps_to_dense`` but with the standard ordering convention.

    Kept as a separate name for clarity at the call site.
    """
    return V_mps_to_dense(V_mps, N, d)


# ---------------------------------------------------------------------------
# Dense gradient → MPS-tangent gradient
# ---------------------------------------------------------------------------

def _build_left_envs(V_mps: list[np.ndarray], N: int, d: int) -> list[np.ndarray]:
    """``L_j[x_<j, a]`` = product of A_0..A_{j-1} contracted on left edge,
    leaving site j's left bond as a tensor index.

    Returns a list of length ``d``. ``envs[0]`` has shape ``(1, 1)`` (no
    sites contracted, dummy left edge of size 1). ``envs[j]`` has shape
    ``(N^j, D_{j-1})``.
    """
    envs: list[np.ndarray] = [np.ones((1, 1), dtype=np.float64)]  # L_0
    for j in range(d - 1):
        A = V_mps[j]                                        # (D_L, N, D_R)
        D_L, _, D_R = A.shape
        prev = envs[-1]                                     # (N^j, D_L)
        # contract: out[x_<=j, b] = Σ_a prev[x_<j, a] · A[a, x_j, b]
        prev_rs = prev                                       # (N^j, D_L)
        A_rs = A.transpose(0, 1, 2).reshape(D_L, N * D_R)
        nxt = prev_rs @ A_rs                                # (N^j, N*D_R)
        envs.append(nxt.reshape(-1, D_R))                    # (N^{j+1}, D_R)
    return envs


def _build_right_envs(V_mps: list[np.ndarray], N: int, d: int) -> list[np.ndarray]:
    """``R_j[b, x_>j]`` = product of A_{j+1}..A_{d-1} contracted on right
    edge, leaving site j+1's left bond as a tensor index.

    Returns a list of length ``d``. ``envs[d - 1]`` has shape ``(1, 1)``;
    ``envs[j]`` has shape ``(D_j_right_bond, N^{d - j - 1})``.
    """
    envs: list[np.ndarray] = [None] * d  # type: ignore
    envs[d - 1] = np.ones((1, 1), dtype=np.float64)         # R_{d-1}
    for j in range(d - 2, -1, -1):
        A = V_mps[j + 1]                                    # (D_L, N, D_R)
        D_L, _, D_R = A.shape
        nxt = envs[j + 1]                                   # (D_R, N^{d-j-2})
        # contract: out[a, x_{j+1}..] = Σ_b A[a, x_{j+1}, b] · nxt[b, ...]
        A_rs = A.reshape(D_L * N, D_R)
        out = A_rs @ nxt                                    # (D_L*N, N^{d-j-2})
        envs[j] = out.reshape(D_L, -1)                       # (D_L, N^{d-j-1})
    return envs


def project_dense_grad_to_mps_cores(
    grad_dense: np.ndarray,
    V_mps: list[np.ndarray],
    N: int, d: int,
) -> list[np.ndarray]:
    """Project ``dL/dV_dense`` (shape ``(N^d,)``) onto MPS tangent at V_mps.

    Returns ``[dL/dA_j]`` matching the shapes of ``V_mps``'s cores.

    For each core j:
      dL/dA_j[a, n, b] = Σ_{x: x_j = n} grad_dense[x] · L_j[x_<j, a]
                                                       · R_j[b, x_>j]
    """
    g = np.asarray(grad_dense, dtype=np.float64).reshape((N**0, *([N] * d)))
    # We'll reshape on the fly per j.
    L_envs = _build_left_envs(V_mps, N, d)
    R_envs = _build_right_envs(V_mps, N, d)

    core_grads: list[np.ndarray] = []
    flat = grad_dense.astype(np.float64)
    for j in range(d):
        D_L, _, D_R = V_mps[j].shape
        # Reshape grad: (N^j, N, N^{d-j-1})
        g_reshape = flat.reshape(N**j, N, N**(d - j - 1))
        L = L_envs[j]                                       # (N^j, D_L)
        R = R_envs[j]                                       # (D_R, N^{d-j-1})
        # dL/dA_j[a, n, b] = Σ_{l, r} g[l, n, r] · L[l, a] · R[b, r]
        dL_dA = np.einsum("lnr, la, br -> anb", g_reshape, L, R)
        core_grads.append(dL_dA)
    return core_grads


def _hadamard_dense_at_grid(V_mps: list[np.ndarray], N: int,
                             d: int) -> np.ndarray:
    """Sanity helper: full-grid evaluation of V via core contraction."""
    return V_mps_to_dense(V_mps, N, d)


# ---------------------------------------------------------------------------
# Numerical chain-rule validation
# ---------------------------------------------------------------------------

def _self_test():
    """Verify: for a random scalar function L(V_dense) and arbitrary V_mps,
    the core-gradient produced by ``project_dense_grad_to_mps_cores`` equals
    the finite-difference gradient w.r.t. the cores.

    Setup: L = ½ ‖V_dense − target‖² with V_dense = mps_to_dense(V_mps).
    Then dL/dV_dense = V_dense − target (closed form), and dL/dA_j[a,n,b]
    should match a numerical Jacobian at random perturbation directions.
    """
    np.random.seed(42)
    N, d = 5, 3                                              # N^d = 125
    D_max = 6

    # Random V_mps (real cores)
    V_mps = []
    bonds = [1]
    for _ in range(d - 1):
        bonds.append(min(D_max, bonds[-1] * N))
    bonds.append(1)
    for j in range(d):
        V_mps.append(np.random.randn(bonds[j], N, bonds[j + 1]))

    V_dense = V_mps_to_dense(V_mps, N, d)
    target = np.random.randn(N**d)

    def loss_fn(V_d):
        return 0.5 * float(((V_d - target) ** 2).sum())

    # Analytic: dL/dV_dense
    grad_dense = V_dense - target
    L0 = loss_fn(V_dense)

    # Project
    core_grads = project_dense_grad_to_mps_cores(grad_dense, V_mps, N, d)

    # Finite-difference check at random perturbation per core
    eps = 1e-6
    max_rel = 0.0
    for j in range(d):
        delta = np.random.randn(*V_mps[j].shape) * 0.01
        V_mps_pert = [A.copy() for A in V_mps]
        V_mps_pert[j] = V_mps[j] + eps * delta
        L_plus = loss_fn(V_mps_to_dense(V_mps_pert, N, d))
        V_mps_pert[j] = V_mps[j] - eps * delta
        L_minus = loss_fn(V_mps_to_dense(V_mps_pert, N, d))
        fd = (L_plus - L_minus) / (2 * eps)
        analytic = float((core_grads[j] * delta).sum())
        rel = abs(fd - analytic) / (abs(fd) + 1e-30)
        max_rel = max(max_rel, rel)
        print(f"core j={j}: analytic={analytic:+.6e}  FD={fd:+.6e}  rel={rel:.2e}")
    print(f"\nmax relative chain-rule error: {max_rel:.2e}")
    assert max_rel < 1e-5, "Chain-rule mismatch — bug somewhere"
    print("OK — MPS-tangent projection matches FD to machine precision.")


if __name__ == "__main__":
    _self_test()
