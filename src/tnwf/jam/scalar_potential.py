"""JAM scalar-potential MLP V_t(x) for the conservative wavefunction-flow pipeline.

The model maps (x, t) → V_t(x) ∈ R, periodic on [0, L)^d via sin/cos input
encoding. The conservative velocity is v_t(x) = ∇_x V_t(x).

JAM loss (Joint Action Matching, Neklyudov et al. 2023 "Action Matching"):

    L = E_{t, x_t}[½‖∇_x V_t(x_t)‖² − ∇_x V_t(x_t)·(x_1 − x_0)]
      = E[½‖∇V − (x_1 − x_0)‖²] + const

Optimal V satisfies ∇V_t*(x) = E[x_1 − x_0 | x_t = x] — the conditional mean
of the CFM velocity field, constrained to be a gradient.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """t → emb of dim `dim` via sin/cos at log-spaced frequencies."""

    def __init__(self, dim: int = 64):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = t * self.freqs
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ScalarPotentialMLP(nn.Module):
    """Periodic MLP V: (x, t) → R, with sin/cos spatial encoding on [0, L)^d."""

    def __init__(
        self,
        d: int = 2,
        hidden: int = 128,
        time_embed_dim: int = 64,
        L: float = 8.0,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d = d
        self.L = L
        self.time_embed = SinusoidalEmbedding(dim=time_embed_dim)
        input_dim = 2 * d + time_embed_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        scale = 2.0 * math.pi / self.L
        return torch.cat([torch.sin(scale * x), torch.cos(scale * x)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x_enc = self._encode_x(x)
        h = torch.cat([x_enc, t_emb], dim=-1)
        return self.net(h)


def sinkhorn_ot_pairs(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    epsilon: float = 0.05,
    n_iters: int = 30,
) -> torch.Tensor:
    """Minibatch entropic OT assignment via Sinkhorn (Cuturi 2013).

    Computes a Sinkhorn-regularised optimal transport plan ``P`` between the
    rows of ``x0`` and ``x1`` (both ``(B, d)``), then returns a permutation
    ``perm`` of length B such that pairing ``(x0[i], x1[perm[i]])`` minimises
    the entropy-regularised cost. The pairing is greedy: for each i, pick
    ``j = argmax_j P[i, j]`` and remove that column. With ``epsilon → 0`` this
    approaches the true OT assignment; with larger ``epsilon`` it's smoother
    and the optimization is more stable.

    Pairs returned this way still keep ``∇V`` from a scalar potential, so the
    conservative-velocity constraint is preserved — the OT coupling only
    changes the *pair distribution*, not the loss or the model class.
    """
    B = x0.shape[0]
    # Squared-Euclidean cost (B, B)
    diff = x0.unsqueeze(1) - x1.unsqueeze(0)
    C = (diff ** 2).sum(dim=-1)
    # Normalize C to median magnitude for numerical stability across batch sizes
    C = C / (C.median() + 1e-12)

    log_K = -C / epsilon                                       # (B, B)
    log_u = torch.zeros(B, device=x0.device)
    log_v = torch.zeros(B, device=x0.device)
    for _ in range(n_iters):
        log_v = -torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
    log_P = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)    # log of transport plan

    # Exact bijective assignment via the Hungarian algorithm on the cost
    # surface ``-log_P`` (i.e. minimize the regularised transport cost).
    # scipy's C implementation is far faster than a Python greedy loop
    # at B≈256 (≪1 ms vs ~100 ms).
    from scipy.optimize import linear_sum_assignment
    cost_np = (-log_P).detach().cpu().numpy()
    _, col = linear_sum_assignment(cost_np)
    return torch.from_numpy(col).long().to(x0.device)


#: Canonical loss names, and the aliases accepted for backward compatibility.
#:
#: The paper calls the trained objective "JAM" (joint action matching) and the
#: shipped Table 2 checkpoints record ``loss_fn='jam'``, but the CLI used to
#: expose that same loss as ``cfm``, so four names were in circulation for one
#: object ("action matching", "JAM", "cfm", and the ``_am_`` in the checkpoint
#: filenames, which is a fixed template rather than a record of the loss).
#: ``jam`` is now canonical; ``cfm`` still resolves to it.
LOSS_ALIASES = {"jam": "jam", "cfm": "jam", "am": "am"}


def normalize_loss_name(name: str) -> str:
    """Map a loss name or legacy alias onto its canonical form."""
    try:
        return LOSS_ALIASES[str(name)]
    except KeyError:
        raise ValueError(
            f"loss_name must be one of {sorted(set(LOSS_ALIASES))}, got {name!r}"
        ) from None


def jam_conservative_loss(
    model: ScalarPotentialMLP,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """CFM-with-gradient-velocity loss: E[½‖∇V‖² − ∇V·(x_1−x_0)] on x_t=(1-t)x_0+t·x_1.

    Implements Conditional Flow Matching (Lipman et al. 2022) restricted to
    conservative (gradient) velocity fields. Optimal V satisfies
    ∇V*(x) = E[x_1 − x_0 | x_t = x], yielding the conditional-mean CFM
    velocity field, projected onto gradients.

    Pairs ``(x0[i], x1[i])`` are taken in given order — usually random
    minibatch pairs. To approximate the *action-optimal* velocity field
    (the AM-paper objective) while staying within the conservative-V class,
    re-order pairs with :func:`sinkhorn_ot_pairs` *before* calling this loss.
    """
    x_t = (1.0 - t) * x0 + t * x1
    x_in = x_t.detach().requires_grad_(True)
    V = model(x_in, t)
    grad_x = torch.autograd.grad(V.sum(), x_in, create_graph=True)[0]
    dx_dt = (x1 - x0).detach()
    kinetic = 0.5 * (grad_x ** 2).sum(dim=-1)
    alignment = (grad_x * dx_dt).sum(dim=-1)
    return (kinetic - alignment).mean()


def action_matching_loss(
    model: ScalarPotentialMLP,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    t_0: torch.Tensor,
    t_1: torch.Tensor,
) -> torch.Tensor:
    """Action Matching loss (Neklyudov et al. 2022, https://arxiv.org/abs/2210.06662).

    Variational form of the action functional for the optimal conservative
    velocity field that drives ``q_{t_0}`` to ``q_{t_1}``. Given samples
    ``x0 ~ q_{t_0}``, ``x1 ~ q_{t_1}`` and an interpolation time
    ``t ∈ [t_0, t_1]``, the loss is::

        L_AM = E[V(t_0, x_0)] − E[V(t_1, x_1)]
             + (t_1 − t_0) · E_{t, x_t}[½‖∇_x V(t, x_t)‖² + ∂_t V(t, x_t)]

    where ``x_t = (1-tau) x_0 + tau x_1`` with ``tau = (t − t_0)/(t_1 − t_0)``
    and the factor ``(t_1 − t_0)`` is the inverse of the uniform time-sampling
    density. Constant weight ``w(t) = 1`` (the ``w'(t) = 0`` branch in the
    upstream JAX reference at github.com/necludov/jam, ``losses.py:get_am_loss``).

    Unlike :func:`jam_conservative_loss` (CFM-with-grad), AM does **not**
    depend on the pair displacement ``x_1 − x_0`` — only on the marginals at
    the boundary timepoints and on the interpolation ``x_t``. This makes it
    the right loss for trajectory data with multiple intermediate marginals.

    Args:
        model:   ScalarPotentialMLP. ``model(x, t)`` returns ``(B, 1)``.
        x0, x1:  ``(B, d)`` boundary samples from ``q_{t_0}``, ``q_{t_1}``.
        t:       ``(B, 1)`` global time in ``[t_0, t_1]`` (uniform sample).
        t_0, t_1: ``(B, 1)`` segment endpoints (broadcast-compatible).

    Returns:
        Scalar loss (mean over the batch).
    """
    dt_segment = (t_1 - t_0).detach()
    tau = ((t - t_0) / dt_segment).detach()                    # (B, 1) in [0, 1]
    x_t = (1.0 - tau) * x0 + tau * x1

    # Boundary terms: V evaluated at the actual marginal samples
    V_0 = model(x0, t_0)                                       # (B, 1)
    V_1 = model(x1, t_1)                                       # (B, 1)
    boundary = (V_0 - V_1).mean()

    # Time loss: needs both ∂V/∂x and ∂V/∂t
    x_in = x_t.detach().requires_grad_(True)
    t_in = t.detach().requires_grad_(True)
    V = model(x_in, t_in)                                      # (B, 1)
    grad_x = torch.autograd.grad(V.sum(), x_in, create_graph=True)[0]
    grad_t = torch.autograd.grad(V.sum(), t_in, create_graph=True)[0]
    kinetic = 0.5 * (grad_x ** 2).sum(dim=-1, keepdim=True)
    time_integrand = (grad_t + kinetic).mean()                 # scalar
    seg_len = dt_segment.mean()                                # scalar (uniform within batch)

    return boundary + seg_len * time_integrand
