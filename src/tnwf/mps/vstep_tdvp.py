"""TDVP V-step wrappers: TCI+TDVP1 and TCI+TDVP2.

TDVP doesn't form exp(iβV) ⊙ ψ at all — it projects on the MPS tangent space
and exponentiates locally per site via expm_multiply on small effective
Hamiltonian blocks. The MPO carries V_t (not exp(iβV)).

The MPO is built via tt_cross (SVD/MAXVOL pivots).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from tnwf.mps.core import mps_norm
from tnwf.mps.tdvp import tdvp_1site_diagonal, tdvp_2site_diagonal
from tnwf.mpo.build_evolution_mpo import build_V_mpo


def apply_V_step_mps_tci_tdvp1(
    mps, V_fn: Callable, beta: float, t_k: float,
    N: int, d: int, L: float, D_V: int = 32,
    n_sweeps: int = 1, n_sweeps_cross: int = 2,
    device: str = "cpu",
):
    """TCI MPO + 1-site TDVP. Bond dim is fixed (no SVD).

    `device='cuda'` dispatches the inner TDVP loop to the torch port in
    `tnwf.mps.tdvp_torch` (Arnoldi expm + torch.linalg.qr on GPU), which gives
    a large speedup at our regime sizes.
    """
    input_norm = mps_norm(mps)
    V_mpo = build_V_mpo(
        V_fn, t=t_k, N=N, d=d, L=L,
        D_max=D_V, n_sweeps=n_sweeps_cross,
    )
    mps_c = [c.astype(np.complex128) for c in mps]
    if device == "cpu":
        out = tdvp_1site_diagonal(V_mpo, mps_c, tau=1j * beta, n_sweeps=n_sweeps)
    else:
        from tnwf.mps.tdvp_torch import tdvp_1site_diagonal_torch
        out = tdvp_1site_diagonal_torch(
            V_mpo, mps_c, tau=1j * beta, n_sweeps=n_sweeps, device=device,
        )
    output_norm = mps_norm(out)
    if output_norm > 1e-10 * input_norm:
        out[-1] = out[-1] * (input_norm / output_norm)
    return out


def apply_V_step_mps_tci_tdvp2(
    mps, V_fn: Callable, beta: float, t_k: float,
    N: int, d: int, L: float, D_max: int, D_V: int = 32,
    n_sweeps: int = 1, n_sweeps_cross: int = 2, tol: float = 1e-8,
    device: str = "cpu",
):
    """TCI MPO + 2-site TDVP. Bond dim grows up to D_max via SVD split.

    `device='cuda'` dispatches the inner TDVP loop to the torch port.
    """
    input_norm = mps_norm(mps)
    V_mpo = build_V_mpo(
        V_fn, t=t_k, N=N, d=d, L=L,
        D_max=D_V, n_sweeps=n_sweeps_cross,
    )
    mps_c = [c.astype(np.complex128) for c in mps]
    if device == "cpu":
        out = tdvp_2site_diagonal(
            V_mpo, mps_c, tau=1j * beta, D_max=D_max, n_sweeps=n_sweeps, tol=tol,
        )
    else:
        from tnwf.mps.tdvp_torch import tdvp_2site_diagonal_torch
        out = tdvp_2site_diagonal_torch(
            V_mpo, mps_c, tau=1j * beta, D_max=D_max, n_sweeps=n_sweeps,
            tol=tol, device=device,
        )
    output_norm = mps_norm(out)
    if output_norm > 1e-10 * input_norm:
        out[-1] = out[-1] * (input_norm / output_norm)
    return out
