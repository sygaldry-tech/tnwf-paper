"""Riemannian Adam optimizer for a real MPS in mixed-canonical form.

Drop-in replacement for ``_AdamMPS`` in ``train_v_mps_end_to_end.py``.
The key differences from Euclidean Adam:

  1. The incoming gradient is projected onto the Stiefel tangent space at
     each constrained core (``gauge_project``). The center core's gradient
     passes through unchanged.

  2. Adam first/second moments are stored as core-shaped arrays in the
     tangent space. Parallel transport is approximated by identity (the
     standard cheap choice; works well when step sizes are moderate).

  3. The Adam update direction is computed in the tangent space, then a QR
     retraction takes the step back onto the manifold so the constrained
     cores stay exactly left/right-orthogonal at every iteration.

This optimizer modifies ``params`` in place (matching ``_AdamMPS`` API) by
replacing each core's contents with the retracted one. Because retraction
returns a new array of the same shape, we copy data back into the original
arrays so that any external references stay valid.
"""
from __future__ import annotations

import numpy as np

from riemannian_mps import (
    gauge_project,
    retract_qr,
    to_mixed_canonical,
)


class _RiemannianAdamMPS:
    """Riemannian Adam on a single MPS V_mps in mixed-canonical form.

    Operates on a SINGLE V_mps (one list of cores). To handle the K_data
    segments used by the training script, instantiate one optimizer per
    segment; the outer training loop is responsible for that bookkeeping.

    Parameters
    ----------
    cores : list of np.ndarray
        The MPS cores. Must already be in mixed-canonical form with the
        given ``center``. Modified in place.
    lr : float
        Learning rate.
    betas : tuple
        Adam (β1, β2) defaults to (0.9, 0.999).
    eps : float
        Adam ε defaults to 1e-8.
    center : int, optional
        Index of unconstrained core. Defaults to ``len(cores) - 1``.
    project_moments : bool
        If True (default), gauge-project the first/second moment estimates
        after the update so they stay in tangent space.
    """

    def __init__(self, cores: list[np.ndarray], lr: float = 1e-2,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 center: int | None = None, project_moments: bool = True):
        self.cores = cores
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.center = center if center is not None else len(cores) - 1
        self.project_moments = project_moments
        self.m = [np.zeros_like(c) for c in cores]
        self.v = [np.zeros_like(c) for c in cores]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1

        # 1. Project Euclidean gradient onto Riemannian tangent space.
        rgrads = gauge_project(grads, self.cores, self.center)

        # 2. Update Adam moments.
        #    m stays in tangent space (it's a moving average of tangent
        #    gradients). v is element-wise nonnegative (per-coordinate
        #    variance estimate) and is NOT a tangent vector — never project
        #    it, that would let entries go negative and break sqrt(v).
        for i, g in enumerate(rgrads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)

        # 3. Compute Adam update direction.
        m_hat = [m / (1 - self.b1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.b2 ** self.t) for v in self.v]
        direction = [m_h / (np.sqrt(v_h) + self.eps)
                     for m_h, v_h in zip(m_hat, v_hat)]

        # 4. Project direction back to tangent. The Hadamard division by
        # sqrt(v_hat) can pull m_hat off the tangent space; re-projecting
        # ensures the QR retraction sees a strictly tangent vector.
        if self.project_moments:
            direction = gauge_project(direction, self.cores, self.center)

        # 5. QR-retract: new_cores = retract(cores, -lr · direction).
        new_cores = retract_qr(self.cores, direction, step_size=-self.lr,
                               center=self.center)

        # 6. Write back in place so external references stay valid.
        for c, nc in zip(self.cores, new_cores):
            assert c.shape == nc.shape, (c.shape, nc.shape)
            c[...] = nc

        # 7. Parallel-transport approximation: re-project m onto new tangent
        # space. v stays as-is (per-coordinate variance, not tangent).
        if self.project_moments:
            self.m = gauge_project(self.m, self.cores, self.center)


def init_v_mps_riemannian(K_data: int, N: int, d: int, D_V_param: int,
                           init_std: float = 0.01,
                           center: int | None = None,
                           rng: np.random.Generator | None = None
                           ) -> list[list[np.ndarray]]:
    """Build K_data V_mps's in mixed-canonical form, ready for Riemannian
    Adam.

    Mirrors ``_init_V_mps_list`` from ``train_v_mps_end_to_end.py`` but runs
    each newly-initialised MPS through ``to_mixed_canonical`` so the Stiefel
    invariants hold from step 0.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    out: list[list[np.ndarray]] = []
    for _ in range(K_data):
        cores: list[np.ndarray] = []
        D_prev = 1
        for j in range(d):
            D_next = 1 if j == d - 1 else min(D_V_param, N ** (j + 1),
                                              N ** (d - j - 1))
            cores.append(rng.standard_normal((D_prev, N, D_next)) * init_std)
            D_prev = D_next
        cores = to_mixed_canonical(cores, center=center)
        out.append(cores)
    return out
