"""Train an MPS-V (tensor-train velocity potential) by JAM / action matching.

The velocity potential V_t is parameterized as a `(d+1)`-site tensor train
(`MPSScalarPotentialTimeSite`) and fit with the same self-contained losses used
for the MLP JAM trainer — no teacher model required. The resulting cores can be
fed directly to the 2-site TDVP V-step (the MPS-V "bypass"), see
`tnwf.pipelines.run_evolution.run(method="mps_v_tdvp2", ...)`.

Usage:
    python -m tnwf.mps_v.train --dataset gmm_8d --N 16 --D 8 --seed 0 \\
        --out examples/checkpoints/mps_v_gmm_d8.pt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tnwf.jam.scalar_potential import action_matching_loss, jam_conservative_loss
from tnwf.jam.train import DATASET_DEFAULTS, sample_target
from tnwf.mps_v.model import MPSScalarPotentialTimeSite


def train(
    dataset: str,
    seed: int,
    out_path: str,
    *,
    N: int = 16,
    D: int = 8,
    N_t: int = 12,
    n_iter: int = 20_000,
    batch_size: int = 256,
    lr: float = 1e-3,
    d: int | None = None,
    L: float | None = None,
    loss_name: str = "cfm",
    log_every: int = 1000,
) -> dict:
    """Train an MPS-V on an endpoint dataset and save a checkpoint to out_path.

    The checkpoint dict is `{"model", "args", "model_type"}` — the format read
    by `tnwf.mps_v.load_mps_v`. Source samples are N(0, I); the target GMM is in
    the centered frame, so the learned cores are centered (the V-step provider
    re-centers them to the [0, L) grid).
    """
    if loss_name not in ("cfm", "am"):
        raise ValueError(f"loss_name must be 'cfm' or 'am', got {loss_name!r}")
    cfg = dict(DATASET_DEFAULTS[dataset])
    if d is not None:
        cfg["d"] = d
    if L is not None:
        cfg["L"] = L
    d = int(cfg["d"])
    L = float(cfg["L"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    _META = {"d", "n_samples", "L", "kind", "trajectory_K", "training"}
    sample_kw = {k: v for k, v in cfg.items() if k not in _META}
    target = sample_target(dataset, cfg["n_samples"], d=d, seed=seed, **sample_kw)
    target_t = torch.from_numpy(target).float().to(device)

    model = MPSScalarPotentialTimeSite(d=d, N=N, D=D, L=L, N_t=N_t).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for it in range(n_iter):
        idx = torch.randint(0, target_t.shape[0], (batch_size,), device=device)
        x1 = target_t[idx]
        x0 = torch.randn(batch_size, d, device=device)
        t = torch.rand(batch_size, 1, device=device)
        opt.zero_grad()
        if loss_name == "cfm":
            loss = jam_conservative_loss(model, x0, x1, t)
        else:
            t_0 = torch.zeros(batch_size, 1, device=device)
            t_1 = torch.ones(batch_size, 1, device=device)
            loss = action_matching_loss(model, x0, x1, t, t_0, t_1)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (it + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f"[mps_v {dataset} seed={seed}] iter {it + 1}/{n_iter}  loss={recent:.4f}")

    ckpt = {
        "model": model.state_dict(),
        "args": {
            "d": d, "N_grid": N, "D_mps": D, "L": L, "N_t": N_t,
            "dataset": dataset, "seed": seed, "loss_name": loss_name,
            "std": cfg.get("std"), "scale": cfg.get("scale"),
            "arrangement": "orthogonal", "qtt": False,
        },
        "model_type": "mps_potential_time_site",
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return ckpt


def _main() -> None:
    p = argparse.ArgumentParser(description="Train an MPS-V velocity potential.")
    p.add_argument("--dataset", default="gmm_8d")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--N", type=int, default=16, help="grid points per dimension")
    p.add_argument("--D", type=int, default=8, help="MPS bond dimension")
    p.add_argument("--N_t", type=int, default=12, help="time-axis discretization")
    p.add_argument("--n_iter", type=int, default=20_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--L", type=float, default=None)
    p.add_argument("--loss_name", choices=["cfm", "am"], default="cfm")
    args = p.parse_args()
    train(
        dataset=args.dataset, seed=args.seed, out_path=args.out,
        N=args.N, D=args.D, N_t=args.N_t, n_iter=args.n_iter,
        batch_size=args.batch_size, lr=args.lr, d=args.d, L=args.L,
        loss_name=args.loss_name,
    )


if __name__ == "__main__":
    _main()
