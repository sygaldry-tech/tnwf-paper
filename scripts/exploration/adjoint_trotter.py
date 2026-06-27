"""Adjoint-method backward for the dense Path-3 Trotter pipeline.

Replaces autograd through the K_data·8 product-formula substeps with a
hand-rolled costate ('λ') propagation. The forward pass runs with
``torch.no_grad`` and caches ψ at every substep boundary; the backward
pass walks the cache in reverse, accumulating ``dL/dV_grid[k]`` at each
V-substep and propagating λ through the adjoint of each unitary.

Key formulas, V real and ψ complex:

  K-substep   y = U_K(α) x        (FFT-based pseudospectral)
                lam_before = U_K(-α) lam_after

  V-substep   y = e^{i·s·β·V} ⊙ x    (pointwise diagonal)
                dL/dV[i]      += s · 2β · Im(lam_after[i] · y[i]*)
                lam_before     = e^{-i·s·β·V} ⊙ lam_after

with ``lam = ∂L/∂ψ*`` (torch's conjugate-Wirtinger convention — the same
tensor torch stores in ``.grad``).

The script's ``__main__`` block runs a gradient match against
``torch.autograd`` on a small N=4 case to validate the implementation.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


_SIGN_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _sign_mask(N: int, d: int, device) -> torch.Tensor:
    """(-1)^(Σ_j j_k) flattened to (N^d,) — same convention as the Path-3
    training script's ``apply_K_step``."""
    key = (N, d, str(device))
    if key in _SIGN_MASK_CACHE:
        return _SIGN_MASK_CACHE[key]
    g = torch.zeros((N,) * d, device=device)
    for j in range(d):
        shape = [1] * d
        shape[j] = N
        idx = torch.arange(N, device=device).reshape(shape)
        g = g + idx
    mask = ((-1.0) ** g).reshape(-1).to(torch.complex128)
    _SIGN_MASK_CACHE[key] = mask
    return mask


def make_kinetic_eigenvalues(N: int, d: int, L: float, device) -> torch.Tensor:
    p = (2.0 * np.pi / L) * (torch.arange(N, device=device) - N // 2).float()
    p2 = p * p
    if d == 1:
        K = 0.5 * p2
    else:
        K = torch.zeros((N,) * d, device=device)
        for j in range(d):
            shape = [1] * d
            shape[j] = N
            K = K + 0.5 * p2.reshape(shape)
    return K.reshape(-1)


def make_grid_centred(N: int, d: int, L: float, device) -> torch.Tensor:
    lin = torch.linspace(0.0, L, N + 1, device=device)[:-1] - L / 2.0
    grids = torch.meshgrid(*([lin] * d), indexing="ij")
    return torch.stack(grids, dim=-1).reshape(-1, d)


def apply_K_step(psi, alpha, N, d, K_eigs, sign):
    """Unitary K-substep: y = exp(iα K) ψ via FFT."""
    psi_nd = (psi * sign).reshape((N,) * d)
    psi_fft = torch.fft.fftn(psi_nd).reshape(-1)
    psi_fft = psi_fft * torch.exp(1j * alpha * K_eigs)
    psi_nd = torch.fft.ifftn(psi_fft.reshape((N,) * d)).reshape(-1)
    return psi_nd * sign


def apply_V_step(psi, beta, V_grid):
    """Unitary V-substep: y = exp(iβV) ⊙ ψ pointwise."""
    return psi * torch.exp(1j * beta * V_grid)


# Forward substep plan for the 8-step BCH product formula:
# (kind, sign_factor) — α_eff = sign_factor·α  for K, β_eff = sign_factor·β  for V
SUBSTEP_PLAN: list[tuple[str, int]] = [
    ("K", +1), ("V", +1),
    ("K", -1), ("V", -1),
    ("K", -1), ("V", -1),
    ("K", +1), ("V", +1),
]


def trotter_coefficients(delta_t: float, N: int, d: int, L: float):
    alpha = (L / (np.pi * N)) * np.sqrt(delta_t / d)
    beta = (np.pi * N / (2.0 * L)) * np.sqrt(d * delta_t)
    return float(alpha), float(beta)


def forward_with_cache(
    psi_0: torch.Tensor,
    V_grids: Sequence[torch.Tensor],
    *,
    alpha: float,
    beta: float,
    N: int,
    d: int,
    K_eigs: torch.Tensor,
    sign: torch.Tensor,
) -> list[torch.Tensor]:
    """Run the forward Trotter pipeline and return ψ at every substep boundary.

    Returns a list of length ``K_data * 8 + 1``. ``cache[0]`` is ``psi_0``;
    ``cache[k * 8 + 1 + j]`` is ψ after substep ``j`` of segment ``k``;
    ``cache[(k + 1) * 8]`` is ψ at time ``(k + 1) * Δt``.
    """
    cache = [psi_0]
    psi = psi_0
    for k in range(len(V_grids)):
        V = V_grids[k]
        for kind, s in SUBSTEP_PLAN:
            if kind == "K":
                psi = apply_K_step(psi, s * alpha, N, d, K_eigs, sign)
            else:
                psi = apply_V_step(psi, s * beta, V)
            cache.append(psi)
    return cache


def _hellinger_lam(psi: torch.Tensor, q: torch.Tensor,
                   floor: float = 1e-20) -> torch.Tensor:
    """Conjugate-Wirtinger gradient of ‖|ψ| − √q‖₂² w.r.t. ψ.

    ∂L/∂ψ[x]* = ψ[x] − √q[x] · ψ[x] / |ψ[x]|.
    """
    abs_psi = psi.abs().clamp(min=floor)
    sqrt_q = q.sqrt()
    return psi - sqrt_q * (psi / abs_psi)


def adjoint_backward(
    psi_cache: list[torch.Tensor],
    V_grids: Sequence[torch.Tensor],
    q_grids: Sequence[torch.Tensor],
    *,
    alpha: float,
    beta: float,
    N: int,
    d: int,
    K_eigs: torch.Tensor,
    sign: torch.Tensor,
) -> list[torch.Tensor]:
    """Adjoint backward pass; returns ``dL/dV_grid[k]`` for each segment.

    ``q_grids[k]`` is the data marginal at time ``k * Δt`` (k = 0..K_data).
    The loss is ``Σ_{k=1..K_data} ‖|ψ_k| − √q_k‖²``; ``q_grids[0]`` is not
    used (ψ_0 is initialised from it, not differentiated against it).
    """
    K_data = len(V_grids)
    assert len(q_grids) == K_data + 1
    assert len(psi_cache) == K_data * 8 + 1

    dL_dV = [torch.zeros_like(V) for V in V_grids]

    # λ at terminal time: contribution from final-snapshot loss.
    lam = _hellinger_lam(psi_cache[K_data * 8], q_grids[K_data])

    for k in range(K_data - 1, -1, -1):
        V = V_grids[k]
        # Walk segment k's 8 substeps in reverse.
        for sub_idx in range(7, -1, -1):
            kind, s = SUBSTEP_PLAN[sub_idx]
            # ψ after this substep:
            psi_after = psi_cache[k * 8 + 1 + sub_idx]

            if kind == "V":
                # Gradient contribution: dL/dV[i] += s · 2β · Im(λ[i] · ψ_after[i]*)
                dL_dV[k] = dL_dV[k] + (
                    s * 2.0 * beta * (lam * psi_after.conj()).imag
                ).to(dL_dV[k].dtype)
                # Adjoint of V-substep: λ ← exp(-i s β V) ⊙ λ
                lam = torch.exp(-1j * s * beta * V) * lam
            else:  # K-substep
                # Adjoint of K-substep: λ ← U_K(-s α) λ
                lam = apply_K_step(lam, -s * alpha, N, d, K_eigs, sign)

        # λ now lives at ψ_{k Δt}. Add direct loss contribution from this
        # intermediate snapshot (skip k=0 — ψ_0 is fixed, not optimised).
        if k > 0:
            lam = lam + _hellinger_lam(psi_cache[k * 8], q_grids[k])

    return dL_dV


# --------------------------------------------------------------------
# Validation: gradient match against torch autograd on a small case.
# --------------------------------------------------------------------

def _autograd_loss_and_grad(V_grids_param, psi_0, q_grids,
                            alpha, beta, N, d, K_eigs, sign):
    """Path-3 forward via autograd. ``V_grids_param`` are leaves."""
    psi = psi_0
    K_data = len(V_grids_param)
    loss = torch.zeros((), dtype=torch.float64, device=psi.device)
    for k in range(K_data):
        V = V_grids_param[k]
        for kind, s in SUBSTEP_PLAN:
            if kind == "K":
                psi = apply_K_step(psi, s * alpha, N, d, K_eigs, sign)
            else:
                psi = apply_V_step(psi, s * beta, V)
        # Snapshot loss
        abs_psi = psi.abs()
        sqrt_q = q_grids[k + 1].sqrt()
        loss = loss + ((abs_psi - sqrt_q) ** 2).sum()
    loss.backward()
    return loss.detach(), [V.grad.detach().clone() for V in V_grids_param]


def _adjoint_loss_and_grad(V_grids, psi_0, q_grids,
                           alpha, beta, N, d, K_eigs, sign):
    """Same forward + adjoint backward."""
    with torch.no_grad():
        cache = forward_with_cache(
            psi_0, V_grids, alpha=alpha, beta=beta,
            N=N, d=d, K_eigs=K_eigs, sign=sign,
        )
        loss = torch.zeros((), dtype=torch.float64, device=psi_0.device)
        for k in range(len(V_grids)):
            psi_k = cache[(k + 1) * 8]
            loss = loss + ((psi_k.abs() - q_grids[k + 1].sqrt()) ** 2).sum()
        dL_dV = adjoint_backward(
            cache, V_grids, q_grids,
            alpha=alpha, beta=beta, N=N, d=d, K_eigs=K_eigs, sign=sign,
        )
    return loss, dL_dV


def _self_test():
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"
    N, d, L = 4, 2, 4.0
    K_data = 3
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    K_eigs = make_kinetic_eigenvalues(N, d, L, device).to(torch.complex128)
    sign = _sign_mask(N, d, device)

    # Random normalized ψ_0
    psi_0_raw = (torch.randn(N**d, dtype=torch.float64)
                 + 1j * torch.randn(N**d, dtype=torch.float64))
    psi_0 = psi_0_raw / torch.norm(psi_0_raw)
    # Random target densities (uniform-ish on grid)
    q_grids = []
    for _ in range(K_data + 1):
        q = torch.rand(N**d, dtype=torch.float64).clamp(min=1e-6)
        q_grids.append(q / q.sum())

    # Random V grids per segment (real, with requires_grad for autograd run)
    V_init = [torch.randn(N**d, dtype=torch.float64) * 0.5 for _ in range(K_data)]

    # Autograd run (re-fresh leaves so grads accumulate cleanly)
    V_grids_ag = [V.detach().clone().requires_grad_(True) for V in V_init]
    loss_ag, grads_ag = _autograd_loss_and_grad(
        V_grids_ag, psi_0, q_grids, alpha, beta, N, d, K_eigs, sign)

    # Adjoint run
    V_grids_adj = [V.detach().clone() for V in V_init]
    loss_adj, grads_adj = _adjoint_loss_and_grad(
        V_grids_adj, psi_0, q_grids, alpha, beta, N, d, K_eigs, sign)

    print(f"loss     autograd = {loss_ag.item():.12f}")
    print(f"loss     adjoint  = {loss_adj.item():.12f}")
    print(f"loss     |Δ|      = {abs(loss_ag.item() - loss_adj.item()):.2e}")
    for k, (g_ag, g_adj) in enumerate(zip(grads_ag, grads_adj)):
        rel = (g_ag - g_adj).norm() / (g_ag.norm() + 1e-30)
        absdiff = (g_ag - g_adj).abs().max()
        print(f"grad[k={k}] rel={rel.item():.3e}  max|Δ|={absdiff.item():.3e}")


if __name__ == "__main__":
    _self_test()
