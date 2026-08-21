"""MPS-V: the velocity potential V_t parameterized directly as a tensor train.

`MPSScalarPotentialTimeSite` represents V(x_0, ..., x_{d-1}, t) as a (d+1)-site
MPS: sites 0..d-1 are spatial (local dim N), site d is time (local dim N_t).
Continuous x and t are handled by multilinear interpolation on the respective
grids. The learned real cores can be read out at any t via `get_mps_cores(t)`
and fed *directly* to the 2-site TDVP V-step — no runtime tensor-cross. This is
the "MPS-V bypass" used in the paper.

This is the public-release port of the research model. The quantized-TT (QTT,
binary-folded) variant is not included here; load non-QTT checkpoints
(`qtt=False`, the default).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _core_shape(i: int, d: int, N: int, D: int) -> tuple[int, int, int]:
    if d == 1:
        return (1, N, 1)
    D_L = 1 if i == 0 else D
    D_R = 1 if i == d - 1 else D
    return (D_L, N, D_R)


class MPSScalarPotentialTimeSite(nn.Module):
    """V(x, t) as a (d+1)-site MPS over [x_0, ..., x_{d-1}; t].

    Args:
        d:    spatial dimension.
        N:    grid points per spatial dimension.
        D:    MPS bond cap (both x-x and x-t bonds).
        L:    spatial domain length (torus [0, L)^d).
        N_t:  time-axis discretization (default 16).
        boundary_mode:   "periodic" (default), "clamp", or "padded". Note
                         "padded" stores N+1 amplitudes per site and is
                         incompatible with the standard V-step pipeline.
        linear_baseline: if True, add a learnable additive linear term
                         Σ_k a_k(t)·(x_k/L), breaking V(x=0)=V(x=L).
        qtt:             accepted for checkpoint-arg compatibility; must be
                         False — the QTT path is not ported to this release.
    """

    def __init__(self, d: int, N: int, D: int, L: float = 2.0,
                 N_t: int = 16,
                 boundary_mode: str = "periodic",
                 linear_baseline: bool = False,
                 qtt: bool = False) -> None:
        super().__init__()
        if qtt:
            raise NotImplementedError(
                "QTT (binary-folded) MPS-V is not included in this release port; "
                "use a non-QTT checkpoint (qtt=False)."
            )
        self.d = int(d)
        self.N = int(N)
        self.N_t = int(N_t)
        self.D = int(D)
        self.L = float(L)
        self.dx = float(L) / N
        self.qtt = False
        self.d_eff_x = self.d
        if boundary_mode not in ("periodic", "clamp", "padded"):
            raise ValueError(
                f"boundary_mode must be 'periodic'|'clamp'|'padded', "
                f"got {boundary_mode!r}")
        self.boundary_mode = boundary_mode
        self.linear_baseline = bool(linear_baseline)
        # Number of stored amplitudes per x-site (only padded differs).
        self._N_store = (self.N + 1) if boundary_mode == "padded" else self.N

        d_eff = self.d_eff_x + 1   # x-sites + 1 t-site
        noise_std = 0.01
        self.cores = nn.ParameterList()
        for i in range(d_eff):
            N_loc = self._N_store if i < self.d_eff_x else self.N_t
            shape = _core_shape(i, d_eff, N_loc, D)
            D_L, _, D_R = shape
            tensor = torch.randn(*shape) * noise_std
            if D_L == D_R and D_L > 1:
                eye = torch.eye(D_L)
                for n in range(N_loc):
                    tensor[:, n, :] = tensor[:, n, :] + eye
            elif D_L == 1 and D_R > 1:
                tensor = tensor + 1.0 / math.sqrt(D_R)
            elif D_R == 1 and D_L > 1:
                tensor = tensor + 1.0 / math.sqrt(D_L)
            self.cores.append(nn.Parameter(tensor))

        if self.linear_baseline:
            # (N_t, d): per-time-slice per-dim linear coefficient; init zero.
            self.linear_coeffs = nn.Parameter(torch.zeros(self.N_t, self.d))

    def _x_grid_indices(self, x: torch.Tensor):
        """Return (x0, x1, alpha) lower/upper neighbor indices + frac for x."""
        if self.boundary_mode == "periodic":
            x_grid = (x / self.dx) % self.N
            x0 = x_grid.long() % self.N
            x1 = (x0 + 1) % self.N
        elif self.boundary_mode == "clamp":
            x_grid = (x / self.dx).clamp(0.0, float(self.N - 1))
            x0 = x_grid.long().clamp(0, self.N - 1)
            x1 = (x0 + 1).clamp(0, self.N - 1)
        else:  # padded — N+1 amplitudes
            x_grid = (x / self.dx).clamp(0.0, float(self.N))
            x0 = x_grid.long().clamp(0, self.N - 1)
            x1 = (x0 + 1).clamp(0, self.N)
        alpha = x_grid - x0.float()
        return x0, x1, alpha

    def _mps_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-sample contraction of the (d+1)-site MPS at (x, t). Returns (B,)."""
        B = x.shape[0]
        x0, x1, alpha = self._x_grid_indices(x)
        # t site (per-sample t in [0, 1])
        t_flat = t.reshape(B).clamp(0.0, 1.0)
        t_grid = t_flat * (self.N_t - 1)
        t0 = t_grid.long().clamp(0, self.N_t - 2)
        t1 = (t0 + 1).clamp(0, self.N_t - 1)
        alpha_t = (t_grid - t0.float()).reshape(B, 1, 1)

        # Contract x sites.
        c0 = self.cores[0].squeeze(0)                  # (N_store, D_R)
        v_lo = c0[x0[:, 0]]
        v_hi = c0[x1[:, 0]]
        state = (1.0 - alpha[:, 0:1]) * v_lo + alpha[:, 0:1] * v_hi  # (B, D)

        for k in range(1, self.d):
            core = self.cores[k]                       # (D, N_store, D)
            core_perm = core.permute(1, 0, 2)          # (N_store, D, D)
            c_lo = core_perm[x0[:, k]]                 # (B, D, D)
            c_hi = core_perm[x1[:, k]]                 # (B, D, D)
            a_k = alpha[:, k].reshape(B, 1, 1)
            core_eff = (1.0 - a_k) * c_lo + a_k * c_hi
            state = torch.bmm(state.unsqueeze(1), core_eff).squeeze(1)

        # Contract t site (final, right boundary D_R=1).
        t_core = self.cores[self.d]                    # (D, N_t, 1)
        t_perm = t_core.permute(1, 0, 2)               # (N_t, D, 1)
        c_lo = t_perm[t0]                              # (B, D, 1)
        c_hi = t_perm[t1]                              # (B, D, 1)
        t_core_eff = (1.0 - alpha_t) * c_lo + alpha_t * c_hi
        state = torch.bmm(state.unsqueeze(1), t_core_eff).squeeze(1)  # (B, 1)
        v_mps = state.squeeze(-1)                      # (B,)

        if self.linear_baseline:
            lin_lo = self.linear_coeffs[t0]            # (B, d)
            lin_hi = self.linear_coeffs[t1]            # (B, d)
            alpha_t_flat = alpha_t.reshape(B, 1)
            lin_t = (1.0 - alpha_t_flat) * lin_lo + alpha_t_flat * lin_hi
            x_scaled = (x.clamp(0.0, self.L) / self.L)
            v_mps = v_mps + (lin_t * x_scaled).sum(dim=-1)
        return v_mps

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._mps_forward(x, t).unsqueeze(-1)

    def get_mps_cores(self, t: float) -> list[np.ndarray]:
        """Return d-site MPS cores (local dim N) for V_t at the given t.

        Contracts out the t-site via multilinear interp between adjacent
        t-bins. Output shapes: (1,N,D), (D,N,D), ..., (D,N,1) — the drop-in
        format for the 2-site TDVP V-step bypass.
        """
        with torch.no_grad():
            t = float(t)
            t_grid = max(0.0, min(self.N_t - 1.0, t * (self.N_t - 1)))
            t0 = int(min(self.N_t - 2, int(t_grid)))
            alpha = t_grid - t0
            nx = self.d_eff_x                            # x-sites
            t_core = self.cores[nx]                      # (D, N_t, 1)
            t_vec = torch.zeros(self.N_t, device=t_core.device)
            t_vec[t0] = 1.0 - alpha
            t_vec[t0 + 1] = alpha
            t_contracted = torch.einsum('dnr,n->dr', t_core, t_vec)      # (D, 1)
            last = self.cores[nx - 1]                    # (D, N_loc, D)
            new_last = torch.einsum('dnr,re->dne', last, t_contracted)   # (D, N_loc, 1)
            cores = [c.detach().cpu().numpy().astype(np.float64)
                     for c in list(self.cores[:nx - 1]) + [new_last]]
        return cores
