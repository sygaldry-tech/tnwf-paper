"""Locate the rare-event tail error: grid, learned oracle, or tensor network?

The prepared state over-populates the >4-sigma shell relative to the analytic
tail probability. That could come from any of three places, and the paper's
claim depends on which:

  1. the grid          -- does an *exact* |psi|^2 on the same N^d grid already
                          mis-state the tail?
  2. the learned V_t   -- does the CLASSICAL ODE flow of the same oracle, with
                          a near-ideal integrator (no truncation, no Trotter
                          error), already mis-state it?
  3. the TN transport  -- what does MPS + TDVP add on top of (2)?

Stage 2 is the one that matters. If the classical flow is already off, the
error is in the learned potential and no amount of Trotter refinement or bond
dimension will fix it -- the tensor network is faithfully transporting a
velocity field that is itself wrong in the tails.

Measured on the shipped d=8 checkpoint, the answer is that the oracle carries
essentially all of it: target 4.08%, grid 3.98%, + oracle 8.12%, + transport
9.38%. Sliced-Wasserstein does not see this, because it is dominated by bulk
transport.

Usage:
    uv run python scripts/diagnose_tail_error.py
    uv run python scripts/diagnose_tail_error.py --ckpt path/to/oracle.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

from make_fig_rare_event import (  # noqa: E402
    D, L, N, SCALE, SIGMA, gm_mode_centers, label_tails, load_wf_samples,
)
from tnwf.data.gaussian_mixture import sample_gaussian_mixture  # noqa: E402
from tnwf.mps_v.model import MPSScalarPotentialTimeSite  # noqa: E402

TAIL_K = 4.0
C = gm_mode_centers()


def tail_pct(x) -> tuple[float, float]:
    _, _, tail, resid = label_tails(np.asarray(x, float), C, TAIL_K)
    return 100 * tail.mean(), 100 * resid.mean()


def exact_grid_samples(n: int, rng) -> np.ndarray:
    """Sample an exact |psi|^2 on the N^d grid without forming N^d numbers.

    Each mixture component is a product Gaussian and the grid is a product
    grid, so the discretized component factorizes and can be sampled axis by
    axis. At d=8, N=32 the dense array would be 32^8 entries.
    """
    dx = L / N
    nodes = np.arange(N) * dx - L / 2
    comp = rng.integers(0, len(C), n)
    x = np.empty((n, D))
    for j in range(len(C)):
        m = comp == j
        if not m.any():
            continue
        for k in range(D):
            p = np.exp(-(nodes - C[j, k]) ** 2 / (2 * SIGMA ** 2))
            p /= p.sum()
            x[m, k] = nodes[rng.choice(N, size=int(m.sum()), p=p)]
    return x + rng.uniform(-0.5, 0.5, x.shape) * dx


def classical_flow(ckpt: str, n: int, Kref: int = 160, sub: int = 4) -> np.ndarray:
    """Integrate dx/dt = grad V with the trained oracle. No tensor network."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    a = ck["args"]
    m = MPSScalarPotentialTimeSite(d=D, N=int(a["N_grid"]), D=int(a["D_mps"]),
                                   L=float(a["L"]), N_t=int(a["N_t"]))
    m.load_state_dict(ck["model"])
    m.eval()
    # The source the oracle was fit against; mismatching it costs ~5x in SW.
    s0 = float(a.get("sigma_0", L / 6.0))
    g = torch.Generator().manual_seed(0)
    x = s0 * torch.randn(n, D, generator=g)
    dt = 1.0 / (Kref * sub)
    for k in range(Kref):
        for s in range(sub):
            tk = (k * sub + s) * dt
            xin = x.detach().requires_grad_(True)
            gx = torch.autograd.grad(m(xin, torch.full((n, 1), tk)).sum(), xin)[0]
            with torch.no_grad():
                x = (x + dt * gx).clamp(-L / 2, L / 2)
    return x.detach().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="MPS-V checkpoint; defaults to the shipped d=8 one")
    ap.add_argument("-n", type=int, default=4000)
    a = ap.parse_args()

    ckpt = a.ckpt
    if ckpt is None:
        hits = [f for f in glob.glob(str(ROOT / "data/mps_v_checkpoints/*.pt"))
                if "_d8_" in f]
        if not hits:
            raise SystemExit("no d=8 MPS-V checkpoint found; pass --ckpt")
        ckpt = hits[0]

    rng = np.random.default_rng(0)
    rows = []

    tgt = sample_gaussian_mixture(a.n, d=D, std=SIGMA, scale=SCALE,
                                  arrangement="orthogonal", seed=1).astype(float)
    rows.append(("analytic target", *tail_pct(tgt)))
    rows.append(("+ grid discretization", *tail_pct(exact_grid_samples(a.n, rng))))
    print("  integrating the classical flow ...", flush=True)
    rows.append(("+ learned oracle (classical ODE)", *tail_pct(classical_flow(ckpt, a.n))))
    wf, _, _ = load_wf_samples("paper", 0)
    rows.append(("+ tensor-network transport", *tail_pct(wf)))

    base = rows[0][1]
    print(f"\n  >{TAIL_K:g} sigma tail fraction, d={D} orthogonal GMM, N={N}\n")
    print(f"  {'stage':34s} {'tail':>7s} {'x target':>9s} {'residual':>9s}")
    print("  " + "-" * 62)
    for label, tail, resid in rows:
        print(f"  {label:34s} {tail:6.2f}% {tail/base:8.2f}x {resid:8.2f}%")

    oracle, transport = rows[2][1], rows[3][1]
    print(f"\n  oracle accounts for   {(oracle-base)/(transport-base)*100:.0f}% "
          f"of the excess over target")
    print(f"  transport adds        {transport/oracle:.2f}x on top of the oracle")


if __name__ == "__main__":
    main()
