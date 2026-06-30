"""MPS-V: velocity potential as a trained tensor train, fed directly to 2-site
TDVP (no runtime tensor-cross — the "bypass" of the paper).

Public API:
  - `MPSScalarPotentialTimeSite` — the model (see `model.py`).
  - `load_mps_v(ckpt_path)`       — rebuild a trained model + its config.
  - `make_mps_v_provider(model)`  — a callable t → V_t MPS cores for the V-step.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from tnwf.mps_v.model import MPSScalarPotentialTimeSite

__all__ = [
    "MPSScalarPotentialTimeSite",
    "load_mps_v",
    "make_mps_v_provider",
]


def load_mps_v(ckpt_path: str, device: str = "cpu"):
    """Load a trained MPS-V model and its config from a checkpoint.

    The checkpoint is a dict with keys "model" (state_dict) and "args" (the
    training config). Returns ``(model, cfg)`` where ``cfg`` is a normalized
    dict exposing ``d, N, L, N_t, D, dataset, std, scale, arrangement``.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    a = a if isinstance(a, dict) else vars(a)

    d = int(a["d"])
    N = int(a.get("N_grid", a.get("N")))
    D = int(a.get("D_mps", a.get("D")))
    L = float(a.get("L", 2.0))
    N_t = int(a.get("N_t", a.get("n_time_slices", 16)))

    model = MPSScalarPotentialTimeSite(
        d=d, N=N, D=D, L=L, N_t=N_t,
        boundary_mode=a.get("boundary_mode", "periodic"),
        linear_baseline=bool(a.get("linear_baseline", False)),
        qtt=bool(a.get("qtt", False)),
    )
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()

    cfg = {
        "d": d, "N": N, "L": L, "N_t": N_t, "D": D,
        "dataset": a.get("dataset"),
        "std": a.get("std"), "scale": a.get("scale"),
        "arrangement": a.get("arrangement", "orthogonal"),
    }
    return model, cfg


def make_mps_v_provider(
    model: MPSScalarPotentialTimeSite, N: int, L: float
) -> Callable[[float], list[np.ndarray]]:
    """Build a ``provider(t) -> list[np.ndarray]`` of complex128 V_t MPS cores.

    The model represents V_t on a grid centered at the origin, while the
    wavefunction V-step indexes the grid over ``[0, L)``. We realign by rolling
    each core's local (grid) axis by ``N // 2`` and cast to complex128, the
    dtype the 2-site TDVP engine expects.
    """
    if int(model.N) != int(N):
        raise ValueError(
            f"MPS-V grid N={model.N} must match the evolution grid N={N}."
        )
    shift = N // 2

    def provider(t: float) -> list[np.ndarray]:
        cores = model.get_mps_cores(t)
        return [np.roll(c, shift, axis=1).astype(np.complex128) for c in cores]

    return provider
