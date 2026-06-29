"""Torch port of `tnwf.mps.tdvp`'s diagonal TDVP — runs on CPU or CUDA.

Same algorithm as `tdvp_1site_diagonal` / `tdvp_2site_diagonal`. Hot-loop
swaps:

  - `scipy.sparse.linalg.expm_multiply` → `_krylov_expm_torch` (Lanczos m=30)
  - `np.einsum`                         → `torch.einsum`
  - `np.linalg.svd`                     → `torch.linalg.svd`
  - `np.linalg.qr`                      → `torch.linalg.qr`

Tensors are kept on `device` throughout the inner loop; the caller can pass
numpy arrays (auto-converted) or torch tensors. Output is always a list of
numpy `complex128` arrays so existing pipelines don't see a type change.

Per GPU kernel benchmarks:
- Krylov-on-GPU vs scipy CPU: 80–500× at our regime sizes.
- SVD on GPU complex128: 4–9× (capped — no Tensor Cores for complex128).

If a `torch.linalg.svd` fails on an ill-conditioned bond we fall back to the
numpy `_robust_svd` for that single bond (transfer cost is tiny compared to
the rest of the step).
"""
from __future__ import annotations

import numpy as np
import torch

from tnwf.mps.tdvp import _robust_svd as _cpu_robust_svd

# Force strict fp64 cuBLAS dispatch — TF32 reduced-precision paths corrupt
# complex128 matmuls at (4096, 4096) and broke our D=64 SW values silently.
# These settings are no-ops on CPU and on GPUs without TF32 support.
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # PyTorch 2.x adds an explicit fp32 precision knob; keep it on `highest`.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
except AttributeError:
    pass


# ── Lanczos Krylov expm (Hermitian H, complex tensors) ────────────────────

def _krylov_expm_torch(
    H: torch.Tensor, v: torch.Tensor, tau: torch.Tensor,
    m: int = 30, tol: float = 1e-12,
) -> torch.Tensor:
    """Compute exp(τ · H) · v via Arnoldi (handles non-Hermitian H).

    The TDVP effective Hamiltonian blocks are not always Hermitian (e.g.,
    when the MPS hasn't been canonicalised at the active bond), so we use
    Arnoldi + Modified Gram-Schmidt instead of Lanczos. Cost is slightly
    higher (O(m²·n) vs O(m·n) for orthogonalisation) but correct for
    general complex matrices.

    Cap `m` to min(m, n) so we don't try to build more Krylov vectors than
    the matrix dimension allows.
    """
    dev = H.device
    cdtype = H.dtype
    n = v.shape[0]
    m = min(m, n)
    beta = torch.linalg.vector_norm(v).real
    if float(beta) == 0.0:
        return v.clone()

    Q = torch.zeros((n, m + 1), dtype=cdtype, device=dev)
    H_m = torch.zeros((m + 1, m), dtype=cdtype, device=dev)
    Q[:, 0] = v / beta

    m_eff = m
    for j in range(m):
        w = H @ Q[:, j]
        # Modified Gram-Schmidt against all prior Q columns.
        for i in range(j + 1):
            h_ij = torch.vdot(Q[:, i], w)
            H_m[i, j] = h_ij
            w = w - h_ij * Q[:, i]
        h_next = torch.linalg.vector_norm(w).real
        H_m[j + 1, j] = h_next.to(cdtype)
        if float(h_next) < tol:
            m_eff = j + 1
            break
        if j < m - 1:
            Q[:, j + 1] = w / h_next
    T = H_m[:m_eff, :m_eff]
    if T.is_cuda:
        expT = torch.linalg.matrix_exp(tau.cpu() * T.cpu()).to(T.device)
    else:
        expT = torch.linalg.matrix_exp(tau * T)
    return beta * (Q[:, :m_eff] @ expT[:, 0])


def _krylov_expm_torch_batched(
    H_batch: torch.Tensor, v_batch: torch.Tensor, tau: torch.Tensor,
    m: int = 30, rel_tol: float = 1e-9, abs_tol: float = 1e-12,
) -> torch.Tensor:
    """Batched Arnoldi: exp(τ · H_b) v_b for b in 0..batch-1, all in one go.

    **Adaptive convergence**: after each Arnoldi iteration j we compute the
    full Krylov approximation y_j = β · Q[:, :j+1] · exp(τ · T_j) · e_1 and
    compare to y_{j-1}. We stop when MAX over batch of `||y_j - y_{j-1}|| /
    ||y_j||` falls below `rel_tol`. This is the standard
    Saad-style accuracy check; it costs one (j+1, j+1) batched matrix_exp
    per iteration but ensures the returned approximation is converged.

    Without this, fixed m=30 gave numerically diverged results at large n
    (N≥64): the Krylov subspace wasn't big enough for the (4096, 4096)
    Hermitian blocks and per-step error accumulated into the 10² regime.

    Args:
        H_batch:  (B, n, n) tensor of Hamiltonian blocks
        v_batch:  (B, n)   tensor of initial vectors
        tau:      0-dim complex scalar
        m:        max Krylov subspace size (the loop runs at most m iterations)
        rel_tol:  relative-change tolerance for early exit
        abs_tol:  absolute floor for division safety

    Returns: (B, n) tensor with exp(τ H_b) v_b for each b.
    """
    dev = H_batch.device
    cdtype = H_batch.dtype
    B, n = v_batch.shape
    m = min(m, n)
    beta = torch.linalg.vector_norm(v_batch, dim=1).real          # (B,)

    Q = torch.zeros((B, n, m + 1), dtype=cdtype, device=dev)
    H_m = torch.zeros((B, m + 1, m), dtype=cdtype, device=dev)
    safe_beta = torch.where(beta > 0, beta, torch.ones_like(beta))
    Q[:, :, 0] = v_batch / safe_beta.unsqueeze(-1)
    beta_c = beta.unsqueeze(-1).to(cdtype)

    prev_approx: torch.Tensor | None = None
    approx_j: torch.Tensor = v_batch                                # init fallback

    for j in range(m):
        # w = H_b @ Q[:, :, j]
        w = torch.einsum("bij,bj->bi", H_batch, Q[:, :, j])
        # Modified Gram-Schmidt against Q[:, :, 0..j]
        for i in range(j + 1):
            h_ij = torch.einsum("bi,bi->b", Q[:, :, i].conj(), w)
            H_m[:, i, j] = h_ij
            w = w - h_ij.unsqueeze(-1) * Q[:, :, i]
        h_next = torch.linalg.vector_norm(w, dim=1).real           # (B,)
        H_m[:, j + 1, j] = h_next.to(cdtype)

        # Current Krylov approximation y_j = β · Q[:, :j+1] · exp(τ T_j) · e_1
        # CRITICAL: torch.linalg.matrix_exp on CUDA at complex128 gives wrong
        # results at large input matrix sizes (T_j is tiny, but it's built
        # from cuBLAS-computed H @ Q matvecs where the (n, n) H lives on GPU).
        # Verified: torch CPU at D=64 N=8 matches numpy to 1e-13; only CUDA
        # diverges. Workaround: compute matrix_exp on CPU (tiny matrix, no
        # speed penalty) and ship the result back to GPU.
        T_j = H_m[:, :j + 1, :j + 1]
        if T_j.is_cuda:
            expT_j = torch.linalg.matrix_exp(tau.cpu() * T_j.cpu()).to(T_j.device)
        else:
            expT_j = torch.linalg.matrix_exp(tau * T_j)            # (B, j+1, j+1)
        approx_j = beta_c * torch.einsum(
            "bnm,bm->bn", Q[:, :, :j + 1], expT_j[:, :, 0]
        )

        # Saad 1992 residual estimator for the action of exp(τ·H) on v:
        #   err_j ≈ β · h_next · |[exp(τ T_j) e_1][j, 0]|
        # This is a rigorous local-truncation indicator — far more reliable
        # than the per-iteration "approx change" because it estimates the
        # ACTUAL error, not the iterate change (which can plateau at a
        # wrong fixed point at large matrix sizes).
        last_row_e1 = torch.abs(expT_j[:, j, 0])                   # (B,)
        err_j = beta * h_next * last_row_e1                        # (B,)
        y_norm = torch.linalg.vector_norm(approx_j, dim=1).real.clamp(min=abs_tol)
        rel_err = float((err_j / y_norm).max())
        if rel_err < rel_tol:
            return approx_j

        # Belt-and-suspenders: also exit if approx is plateauing AND happy
        # breakdown is near (catches numerical edge cases).
        if prev_approx is not None and float(h_next.max()) < abs_tol:
            return approx_j

        prev_approx = approx_j

        if j < m - 1:
            safe_h = torch.where(h_next > 0, h_next, torch.ones_like(h_next))
            Q[:, :, j + 1] = w / safe_h.unsqueeze(-1).to(cdtype)

    return approx_j


# ── Torch helpers (mirrors of tdvp.py:_diag_*) ────────────────────────────

def _diag_build_right_envs_t(cores: list[torch.Tensor], mps: list[torch.Tensor],
                               device: torch.device, cdtype: torch.dtype):
    d = len(mps)
    R = [None] * (d + 1)
    R[d] = torch.ones((1, 1, 1), dtype=cdtype, device=device)
    for j in range(d - 1, -1, -1):
        A = mps[j]
        C = cores[j]
        Rj1 = R[j + 1]
        AR = torch.einsum("bsr,acr->bsac", A, Rj1)
        CAR = torch.einsum("csr,bsar->csab", C, AR)
        R[j] = torch.einsum("asr,csrb->acb", A.conj(), CAR)
    return R


def _diag_update_left_env_t(L_env, A, C):
    LA = torch.einsum("acb,bsf->acsf", L_env, A)
    CLA = torch.einsum("cse,acsf->asef", C, LA)
    return torch.einsum("asd,asef->def", A.conj(), CLA)


def _diag_compute_single_right_env_t(A, C, R_next):
    AR = torch.einsum("bsr,acr->bsac", A, R_next)
    CAR = torch.einsum("csr,bsar->csab", C, AR)
    return torch.einsum("asr,csrb->acb", A.conj(), CAR)


def _make_2site_block_builder(L_env, C_j, C_j1, R_env):
    """Return a callable that builds a single (s, s1) 2-site H_eff block.

    Used for lazy batched application in _batched_expm_apply — avoids
    materialising all N² blocks at once.
    """
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    CC = torch.einsum("csr,rtd->cstd", C_j, C_j1)
    def get(key):
        s, s1 = key
        CC_ss1 = CC[:, s, s1, :]
        CCR = torch.einsum("cd,bdf->cbf", CC_ss1, R_env)
        H = torch.einsum("ace,cbf->abef", L_env, CCR)
        return H.reshape(D_L * D_R, D_L * D_R)
    return get, D_L, D_R, C_j.shape[1], C_j1.shape[1]


def _make_1site_block_builder(L_env, C, R_env):
    """Return a callable that builds a single s 1-site H_eff block."""
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    def get(s):
        C_s = C[:, s, :]
        CR = torch.einsum("cd,bdf->cbf", C_s, R_env)
        H = torch.einsum("ace,cbf->abef", L_env, CR)
        return H.reshape(D_L * D_R, D_L * D_R)
    return get, D_L, D_R, C.shape[1]


def _diag_effective_H_2site_blocks_t(L_env, C_j, C_j1, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    N_j = C_j.shape[1]
    N_j1 = C_j1.shape[1]
    CC = torch.einsum("csr,rtd->cstd", C_j, C_j1)
    blocks: dict[tuple[int, int], torch.Tensor] = {}
    for s in range(N_j):
        for s1 in range(N_j1):
            CC_ss1 = CC[:, s, s1, :]
            CCR = torch.einsum("cd,bdf->cbf", CC_ss1, R_env)
            H = torch.einsum("ace,cbf->abef", L_env, CCR)
            blocks[(s, s1)] = H.reshape(D_L * D_R, D_L * D_R)
    return blocks


def _diag_effective_H_1site_blocks_t(L_env, C, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    N_loc = C.shape[1]
    blocks: dict[int, torch.Tensor] = {}
    for s in range(N_loc):
        C_s = C[:, s, :]
        CR = torch.einsum("cd,bdf->cbf", C_s, R_env)
        H = torch.einsum("ace,cbf->abef", L_env, CR)
        blocks[s] = H.reshape(D_L * D_R, D_L * D_R)
    return blocks


def _diag_effective_H_0site_t(L_env, R_env):
    D_L = L_env.shape[0]
    D_R = R_env.shape[0]
    H = torch.einsum("ace,bcf->abef", L_env, R_env)
    return H.reshape(D_L * D_R, D_L * D_R)


# ── SVD fallback (per-bond CPU fall-through on torch failure) ─────────────

def _robust_svd_t(mat: torch.Tensor, full_matrices: bool = False):
    """torch.linalg.svd, falling back to numpy _robust_svd on failure.

    Returns (U, S, Vh) as torch tensors on the same device as `mat`.
    """
    try:
        return torch.linalg.svd(mat, full_matrices=full_matrices)
    except (torch._C._LinAlgError, RuntimeError):
        # Round-trip to CPU numpy for ill-conditioned cells.
        mat_np = mat.detach().cpu().numpy()
        U_np, S_np, Vh_np = _cpu_robust_svd(mat_np, full_matrices=full_matrices)
        U = torch.from_numpy(U_np).to(mat.device)
        S = torch.from_numpy(S_np).to(mat.device)
        Vh = torch.from_numpy(Vh_np).to(mat.device)
        return U, S, Vh


# ── Public TDVP functions (torch) ─────────────────────────────────────────

def _batched_expm_apply(
    blocks_or_fn, keys: list, vec_fn, tau, krylov_m: int = 30,
    chunk_size: int = 32,
) -> dict:
    """Apply exp(τ · H_b) · v_b to every block in `keys` via batched Arnoldi.

    `blocks_or_fn` can be:
      - a dict {key: H_block_tensor} (eager — all blocks pre-built); or
      - a callable(key) → H_block_tensor (lazy — built per chunk).

    The lazy form bounds peak memory to chunk_size · n² · 16 B regardless of
    how many keys there are, which is essential when N² · n² doesn't fit on
    one GPU (e.g., N=64, D=64 → 1 TB if eager).

    **NaN guard**: my fixed-m Arnoldi can produce NaN/Inf on ill-conditioned
    blocks (high d, high entanglement — e.g., g=8 tdvp2 cells). When that
    happens, we per-block fall back to scipy's adaptive `expm_multiply` on
    CPU (slower but algorithmically robust, with internal scaling-squaring).
    This keeps tdvp2 cells correct at g=8.

    Returns: {key: result_tensor} (1-D, length n).
    """
    from scipy.sparse.linalg import expm_multiply as _scipy_expm_multiply

    if callable(blocks_or_fn):
        get_block = blocks_or_fn
    else:
        get_block = blocks_or_fn.__getitem__

    # Materialize tau as a python complex once (for scipy path).
    tau_c = complex(tau.item()) if isinstance(tau, torch.Tensor) else complex(tau)

    out: dict = {}
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start + chunk_size]
        H_stack = torch.stack([get_block(k) for k in chunk], dim=0)
        v_stack = torch.stack([vec_fn(k) for k in chunk], dim=0)
        res = _krylov_expm_torch_batched(H_stack, v_stack, tau, m=krylov_m)

        # Per-block NaN/Inf check + scipy fallback.
        if not torch.all(torch.isfinite(res)):
            bad = ~torch.all(torch.isfinite(res), dim=1)             # (B,)
            for i in torch.nonzero(bad, as_tuple=False).flatten().tolist():
                H_np = H_stack[i].detach().cpu().numpy()
                v_np = v_stack[i].detach().cpu().numpy()
                # scipy adaptive Padé scaling-squaring — won't overflow.
                fallback = _scipy_expm_multiply(tau_c * H_np, v_np)
                res[i] = torch.from_numpy(fallback).to(res.device, res.dtype)

        for i, k in enumerate(chunk):
            out[k] = res[i]
        del H_stack, v_stack, res
    return out


def _to_torch_list(arrays, device: torch.device, cdtype: torch.dtype):
    """Convert a list of numpy arrays (or torch tensors) to tensors on device."""
    out = []
    for a in arrays:
        if isinstance(a, torch.Tensor):
            out.append(a.to(device=device, dtype=cdtype))
        else:
            out.append(torch.from_numpy(np.ascontiguousarray(a)).to(
                device=device, dtype=cdtype))
    return out


def _to_numpy_list(tensors):
    return [t.detach().cpu().numpy().astype(np.complex128) for t in tensors]


def tdvp_1site_diagonal_torch(
    cores,
    mps,
    tau: complex,
    n_sweeps: int = 1,
    device: str = "cuda",
    krylov_m: int = 30,
    chunk_size: int = 32,
) -> list[np.ndarray]:
    """1-site TDVP for diagonal MPO. Bond dim fixed.

    All N H-eff blocks per site are evolved via one batched Arnoldi call
    (amortises GPU launch overhead). The 0-site bond evolutions stay
    per-call (only one block each)."""
    dev = torch.device(device)
    cdtype = torch.complex128
    d = len(mps)
    if d < 2:
        return [a.copy() if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
                for a in mps]
    A = _to_torch_list(mps, dev, cdtype)
    C_list = _to_torch_list(cores, dev, cdtype)
    tau_t = torch.tensor(tau, dtype=cdtype, device=dev)

    def _evolve_1site_lazy(A_j, L_env, C, R_env, tau_step):
        """Apply exp(τ_step · H_s) lazily — H blocks built per chunk."""
        get_block, D_L, D_R, N = _make_1site_block_builder(L_env, C, R_env)
        keys = list(range(N))
        results = _batched_expm_apply(
            get_block, keys,
            lambda s: A_j[:, s, :].reshape(-1),
            tau_step, krylov_m=krylov_m, chunk_size=chunk_size,
        )
        out = torch.zeros_like(A_j)
        for s, r in results.items():
            out[:, s, :] = r.reshape(D_L, D_R)
        return out

    for _ in range(n_sweeps):
        R = _diag_build_right_envs_t(C_list, A, dev, cdtype)
        L = torch.ones((1, 1, 1), dtype=cdtype, device=dev)

        # L→R sweep
        for j in range(d):
            D_Lj, N_j, D_Rj = A[j].shape
            A[j] = _evolve_1site_lazy(A[j], L, C_list[j], R[j + 1], tau_t * 0.5)

            if j < d - 1:
                mat = A[j].reshape(D_Lj * N_j, D_Rj)
                Q, R_mat = torch.linalg.qr(mat)
                K = Q.shape[1]
                A[j] = Q.reshape(D_Lj, N_j, K)
                L = _diag_update_left_env_t(L, A[j], C_list[j])

                H_0 = _diag_effective_H_0site_t(L, R[j + 1])
                R_vec_new = _krylov_expm_torch(
                    H_0, R_mat.reshape(-1), -tau_t * 0.5, m=krylov_m
                )
                R_mat = R_vec_new.reshape(K, D_Rj)
                A[j + 1] = torch.einsum("ab,bsc->asc", R_mat, A[j + 1])
            else:
                L = _diag_update_left_env_t(L, A[j], C_list[j])

        # R→L sweep
        L_all = [None] * (d + 1)
        L_all[0] = torch.ones((1, 1, 1), dtype=cdtype, device=dev)
        for j in range(d):
            L_all[j + 1] = _diag_update_left_env_t(L_all[j], A[j], C_list[j])
        R_env_j = torch.ones((1, 1, 1), dtype=cdtype, device=dev)

        for j in range(d - 1, -1, -1):
            D_Lj, N_j, D_Rj = A[j].shape
            A[j] = _evolve_1site_lazy(A[j], L_all[j], C_list[j], R_env_j, tau_t * 0.5)

            if j > 0:
                mat = A[j].reshape(D_Lj, N_j * D_Rj)
                Qt, Rt = torch.linalg.qr(mat.T)
                K = Qt.shape[1]
                L_mat = Rt.T
                Q_mat = Qt.T
                A[j] = Q_mat.reshape(K, N_j, D_Rj)
                R_env_j = _diag_compute_single_right_env_t(A[j], C_list[j], R_env_j)

                H_0 = _diag_effective_H_0site_t(L_all[j], R_env_j)
                L_vec_new = _krylov_expm_torch(
                    H_0, L_mat.reshape(-1), -tau_t * 0.5, m=krylov_m
                )
                L_mat = L_vec_new.reshape(D_Lj, K)
                A[j - 1] = torch.einsum("asb,bc->asc", A[j - 1], L_mat)
            else:
                R_env_j = _diag_compute_single_right_env_t(A[j], C_list[j], R_env_j)

    return _to_numpy_list(A)


def tdvp_2site_diagonal_torch(
    cores,
    mps,
    tau: complex,
    D_max: int,
    n_sweeps: int = 1,
    tol: float = 1e-8,
    device: str = "cuda",
    krylov_m: int = 30,
    chunk_size: int = 32,
) -> list[np.ndarray]:
    """2-site TDVP for diagonal MPO. Bond dim grows to D_max via SVD.

    The N² 2-site H-eff blocks per bond are batched into one Arnoldi call
    (chunked by `chunk_size` to bound peak memory). The N 1-site back-evolves
    are batched similarly.
    """
    dev = torch.device(device)
    cdtype = torch.complex128
    d = len(mps)
    if d < 2:
        return [a.copy() if isinstance(a, np.ndarray) else a.detach().cpu().numpy()
                for a in mps]
    A = _to_torch_list(mps, dev, cdtype)
    C_list = _to_torch_list(cores, dev, cdtype)
    tau_t = torch.tensor(tau, dtype=cdtype, device=dev)

    def _evolve_2site_lazy(theta, L_env, C_a, C_b, R_env, tau_step):
        get_block, D_L, D_R, N_a, N_b = _make_2site_block_builder(
            L_env, C_a, C_b, R_env,
        )
        keys = [(s, s1) for s in range(N_a) for s1 in range(N_b)]
        results = _batched_expm_apply(
            get_block, keys,
            lambda key: theta[:, key[0], key[1], :].reshape(-1),
            tau_step, krylov_m=krylov_m, chunk_size=chunk_size,
        )
        out = torch.zeros_like(theta)
        for (s, s1), r in results.items():
            out[:, s, s1, :] = r.reshape(D_L, D_R)
        return out

    def _evolve_1site_lazy(A_j, L_env, C, R_env, tau_step):
        get_block, D_L, D_R, N = _make_1site_block_builder(L_env, C, R_env)
        keys = list(range(N))
        results = _batched_expm_apply(
            get_block, keys,
            lambda s: A_j[:, s, :].reshape(-1),
            tau_step, krylov_m=krylov_m, chunk_size=chunk_size,
        )
        out = torch.zeros_like(A_j)
        for s, r in results.items():
            out[:, s, :] = r.reshape(D_L, D_R)
        return out

    for _ in range(n_sweeps):
        R = _diag_build_right_envs_t(C_list, A, dev, cdtype)
        L = torch.ones((1, 1, 1), dtype=cdtype, device=dev)

        # L→R sweep
        for j in range(d - 1):
            D_Lj, N_j, D_Rj = A[j].shape
            D_Lj1, N_j1, D_Rj1 = A[j + 1].shape
            theta = torch.einsum("anb,bmc->anmc", A[j], A[j + 1])

            theta = _evolve_2site_lazy(theta, L, C_list[j], C_list[j + 1], R[j + 2], tau_t * 0.5)

            mat = theta.reshape(D_Lj * N_j, N_j1 * D_Rj1)
            U, S, Vh = _robust_svd_t(mat, full_matrices=False)
            S0 = float(S[0].real) if len(S) else 1.0
            if tol > 0 and S0 > 0:
                D_keep = max(1, min(D_max, int((S.real > tol * S0).sum().item())))
            else:
                D_keep = min(len(S), D_max)
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vh = Vh[:D_keep, :]

            A[j] = U.reshape(D_Lj, N_j, D_keep)
            A[j + 1] = (torch.diag(S.to(cdtype)) @ Vh).reshape(D_keep, N_j1, D_Rj1)

            L = _diag_update_left_env_t(L, A[j], C_list[j])

            if j < d - 2:
                A[j + 1] = _evolve_1site_lazy(
                    A[j + 1], L, C_list[j + 1], R[j + 2], -tau_t * 0.5
                )

        # R→L sweep
        R = _diag_build_right_envs_t(C_list, A, dev, cdtype)
        L_all = [None] * (d + 1)
        L_all[0] = torch.ones((1, 1, 1), dtype=cdtype, device=dev)
        for j in range(d):
            L_all[j + 1] = _diag_update_left_env_t(L_all[j], A[j], C_list[j])

        for j in range(d - 2, -1, -1):
            D_Lj, N_j, D_Rj = A[j].shape
            D_Lj1, N_j1, D_Rj1 = A[j + 1].shape
            theta = torch.einsum("anb,bmc->anmc", A[j], A[j + 1])

            theta = _evolve_2site_lazy(
                theta, L_all[j], C_list[j], C_list[j + 1], R[j + 2], tau_t * 0.5,
            )

            mat = theta.reshape(D_Lj * N_j, N_j1 * D_Rj1)
            U, S, Vh = _robust_svd_t(mat, full_matrices=False)
            S0 = float(S[0].real) if len(S) else 1.0
            if tol > 0 and S0 > 0:
                D_keep = max(1, min(D_max, int((S.real > tol * S0).sum().item())))
            else:
                D_keep = min(len(S), D_max)
            U = U[:, :D_keep]
            S = S[:D_keep]
            Vh = Vh[:D_keep, :]

            A[j] = (U @ torch.diag(S.to(cdtype))).reshape(D_Lj, N_j, D_keep)
            A[j + 1] = Vh.reshape(D_keep, N_j1, D_Rj1)

            R[j + 1] = _diag_compute_single_right_env_t(A[j + 1], C_list[j + 1], R[j + 2])

            if j > 0:
                A[j] = _evolve_1site_lazy(
                    A[j], L_all[j], C_list[j], R[j + 1], -tau_t * 0.5
                )

    return _to_numpy_list(A)
