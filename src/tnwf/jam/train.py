"""JAM training CLI: train a ScalarPotentialMLP on swiss_roll / GMM datasets.

Usage:
    python -m tnwf.jam.train --dataset swiss_roll_2d --seed 0 \\
        --out data/swiss_roll_2d/jam/seed0.pt
    python -m tnwf.jam.train --dataset gmm_2d --seed 0
    python -m tnwf.jam.train --dataset gmm_3d --d 3 --seed 0
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tnwf.data.gaussian_mixture import sample_gaussian_mixture
from tnwf.data.swiss_roll import sample_swiss_roll
from tnwf.jam.scalar_potential import (
    ScalarPotentialMLP,
    action_matching_loss,
    jam_conservative_loss,
    normalize_loss_name,
    sinkhorn_ot_pairs,
)

DATASET_DEFAULTS: dict[str, dict] = {
    "swiss_roll_2d": {"d": 2, "L": 5.0, "n_samples": 10_000, "noise": 0.1,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_2d":        {"d": 2, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_3d":        {"d": 3, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_4d":        {"d": 4, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_5d":        {"d": 5, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_6d":        {"d": 6, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_7d":        {"d": 7, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_8d":        {"d": 8, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_10d":       {"d": 10, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_12d":       {"d": 12, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
    "gmm_16d":       {"d": 16, "L": 8.0, "n_samples": 10_000, "std": 0.5, "scale": 3.0,
                       "kind": "endpoint", "trajectory_K": None},
}


def sample_target(name: str, n: int, *, d: int, seed: int, **kw) -> np.ndarray:
    """Draw n samples from the named dataset (endpoint datasets only)."""
    if name == "swiss_roll_2d":
        return sample_swiss_roll(n, noise=kw.get("noise", 0.1), seed=seed)
    if name.startswith("gmm_") and name.endswith("d") and name[4:-1].isdigit():
        return sample_gaussian_mixture(
            n, d=d, std=kw.get("std", 0.5), scale=kw.get("scale", 3.0),
            arrangement="orthogonal", seed=seed,
        )
    raise ValueError(f"Unknown dataset: {name}")


def train(
    dataset: str,
    seed: int,
    out_path: str,
    n_iter: int | None = None,
    batch_size: int | None = None,
    hidden: int | None = None,
    n_layers: int | None = None,
    time_embed_dim: int | None = None,
    lr: float | None = None,
    d: int | None = None,
    L: float | None = None,
    log_every: int = 1000,
    loss_name: str | None = None,
    ot_coupling: bool | None = None,
    ot_epsilon: float | None = None,
    global_ot: bool | None = None,
) -> dict:
    """Train V_t for one (dataset, seed) and save checkpoint to out_path.

    Args:
        loss_name: ``"jam"`` (default) uses :func:`jam_conservative_loss` —
            the Conditional Flow Matching loss restricted to gradient
            velocity fields, which is what the paper calls JAM and what the
            shipped Table 2 checkpoints record as ``loss_fn='jam'``. Accepted
            as ``"cfm"`` too, the name this option used to carry.
            ``"am"`` uses :func:`action_matching_loss` —
            the variational Action Matching loss of Neklyudov et al. 2022,
            which matches marginals directly via boundary terms + time
            integral.
        ot_coupling: if True, re-pair each minibatch with
            :func:`sinkhorn_ot_pairs` (entropic OT permutation) before
            evaluating the loss. Approximates the action-optimal velocity
            field for either loss while staying within the conservative-V
            class. Trajectory data only (no-op for endpoint datasets).
        ot_epsilon: Sinkhorn regularisation. Smaller is sharper / more
            OT-like; larger is smoother / more like random pairs.
        global_ot: if True (and ot_coupling=True), precompute *one* OT
            permutation per segment over the FULL n_samples × n_samples
            cost matrix at the start of training (rather than re-running
            Sinkhorn on each minibatch). Removes mode-averaging completely
            for the data we've staged. Trajectory only.
    """
    cfg = dict(DATASET_DEFAULTS[dataset])
    # Pull dataset-specific training defaults; explicit kwargs take precedence.
    train_cfg = cfg.get("training", {})
    if n_iter is None:        n_iter        = train_cfg.get("n_iter", 20_000)
    if batch_size is None:    batch_size    = train_cfg.get("batch_size", 256)
    if hidden is None:        hidden        = train_cfg.get("hidden", 128)
    if n_layers is None:      n_layers      = train_cfg.get("n_layers", 3)
    if time_embed_dim is None: time_embed_dim = train_cfg.get("time_embed_dim", 64)
    if lr is None:            lr            = train_cfg.get("lr", 1e-3)
    if loss_name is None:     loss_name     = train_cfg.get("loss_name", "jam")
    if ot_coupling is None:   ot_coupling   = train_cfg.get("ot_coupling", False)
    if ot_epsilon is None:    ot_epsilon    = train_cfg.get("ot_epsilon", 0.05)
    if global_ot is None:     global_ot     = train_cfg.get("global_ot", False)
    loss_name = normalize_loss_name(loss_name)
    if d is not None:
        cfg["d"] = d
    if L is not None:
        cfg["L"] = L

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    kind = cfg.get("kind", "endpoint")
    # Strip dispatcher-only keys before passing the rest as data-generator kwargs.
    _META_KEYS = {"d", "n_samples", "L", "kind", "trajectory_K", "training"}
    sample_kw = {k: v for k, v in cfg.items() if k not in _META_KEYS}

    if kind == "endpoint":
        # source = N(0, I); target = pre-sampled snapshot in centred frame
        target = sample_target(
            dataset, cfg["n_samples"], d=cfg["d"], seed=seed, **sample_kw
        )
        target_t = torch.from_numpy(target).to(device)
        snapshots_t = None
        K_data = 0
    elif kind == "trajectory":
        K_data = int(cfg["trajectory_K"])
        snapshots = sample_target_trajectory(
            dataset, cfg["n_samples"], d=cfg["d"], seed=seed, **sample_kw
        )
        # (K_data+1, n_per_step, d) on device
        snapshots_t = torch.from_numpy(snapshots).to(device)
        target_t = None
        # Optional: precompute one global OT permutation per segment.
        global_ot_perms = None
        if ot_coupling and global_ot:
            print(f"[{dataset} seed={seed}] precomputing global OT "
                  f"permutations for {K_data} segments...")
            global_ot_perms = []
            for k in range(K_data):
                perm = sinkhorn_ot_pairs(snapshots_t[k], snapshots_t[k + 1],
                                          epsilon=ot_epsilon)
                global_ot_perms.append(perm)
            global_ot_perms = torch.stack(global_ot_perms)     # (K_data, n)
    else:
        raise ValueError(f"Unknown dataset kind {kind!r} for {dataset!r}")

    model = ScalarPotentialMLP(
        d=cfg["d"], hidden=hidden, time_embed_dim=time_embed_dim,
        L=cfg["L"], n_layers=n_layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for it in range(n_iter):
        if kind == "endpoint":
            idx = torch.randint(0, target_t.shape[0], (batch_size,), device=device)
            x1 = target_t[idx]
            x0 = torch.randn(batch_size, cfg["d"], device=device)
            t = torch.rand(batch_size, 1, device=device)
            t_0 = torch.zeros(batch_size, 1, device=device)
            t_1 = torch.ones(batch_size, 1, device=device)
        else:  # trajectory
            n_per_step = snapshots_t.shape[1]
            k = torch.randint(0, K_data, (batch_size,), device=device)
            tau = torch.rand(batch_size, 1, device=device)
            t = (k.unsqueeze(-1).float() + tau) / float(K_data)
            idx0 = torch.randint(0, n_per_step, (batch_size,), device=device)
            if ot_coupling and global_ot_perms is not None:
                # Look up the precomputed OT-coupled idx1 from the per-segment
                # permutation. Each particle in q_{t_k}[idx0] is paired with
                # the SAME q_{t_{k+1}} particle every batch.
                idx1 = global_ot_perms[k, idx0]
            else:
                idx1 = torch.randint(0, n_per_step, (batch_size,), device=device)
            x0 = snapshots_t[k, idx0]
            x1 = snapshots_t[k + 1, idx1]
            t_0 = (k.float() / float(K_data)).unsqueeze(-1)
            t_1 = ((k.float() + 1.0) / float(K_data)).unsqueeze(-1)
            if ot_coupling and global_ot_perms is None:
                # Per-minibatch Sinkhorn within each segment-bucket
                with torch.no_grad():
                    for k_val in range(K_data):
                        mask = (k == k_val)
                        if mask.sum() < 2:
                            continue
                        x0_b = x0[mask]
                        x1_b = x1[mask]
                        perm = sinkhorn_ot_pairs(x0_b, x1_b, epsilon=ot_epsilon)
                        x1_b = x1_b[perm]
                        x1 = x1.clone()
                        x1[mask] = x1_b

        opt.zero_grad()
        if loss_name == "jam":
            loss = jam_conservative_loss(model, x0, x1, t)
        else:  # "am"
            loss = action_matching_loss(model, x0, x1, t, t_0, t_1)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if (it + 1) % log_every == 0:
            recent = float(np.mean(losses[-log_every:]))
            print(f"[{dataset} seed={seed}] iter {it + 1}/{n_iter}  loss={recent:.4f}")

    ckpt = {
        "model_state": model.state_dict(),
        "config": {
            "d": cfg["d"],
            "L": cfg["L"],
            "hidden": hidden,
            "n_layers": n_layers,
            "time_embed_dim": time_embed_dim,
            "dataset": dataset,
            "seed": seed,
            "kind": kind,
            "trajectory_K": cfg.get("trajectory_K"),
            "loss_name": loss_name,
            "ot_coupling": ot_coupling,
            "global_ot": global_ot,
        },
        "final_loss": float(np.mean(losses[-100:])) if losses else float("inf"),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return ckpt


def load_jam(checkpoint_path: str, device: str | None = None) -> tuple[ScalarPotentialMLP, dict]:
    """Reload a trained JAM checkpoint. Returns (eval-mode model, config dict)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = ScalarPotentialMLP(
        d=cfg["d"], hidden=cfg["hidden"], time_embed_dim=cfg["time_embed_dim"],
        L=cfg["L"], n_layers=cfg["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def sample_gradient_flow(
    model: ScalarPotentialMLP,
    z: "torch.Tensor",
    n_steps: int = 200,
    save_every: int | None = None,
    L: float | None = None,
) -> "np.ndarray":
    """Euler integration of the classical conservative ODE dx/dt = ∇_x V_t(x).

    Args:
        model:      trained ScalarPotentialMLP
        z:          (B, d) initial source samples (model coords, centred at 0)
        n_steps:    number of Euler steps over t ∈ [0, 1]
        save_every: if given, return (n_snapshots+1, B, d) including initial state.
                    Otherwise return (B, d) final samples only.
        L:          if given, clamp samples to [-L/2, L/2] (overflow guard).

    Returns:
        Final samples (or stack of snapshots) as a numpy array, model coords.
    """
    import numpy as np
    import torch
    model.eval()
    device = next(model.parameters()).device
    x = z.clone().to(device)
    dt = 1.0 / n_steps
    snaps = [x.detach().cpu().numpy().copy()] if save_every is not None else None

    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0], 1), t_val, device=device, dtype=x.dtype)
        x_in = x.detach().requires_grad_(True)
        V = model(x_in, t)
        grad_x = torch.autograd.grad(V.sum(), x_in)[0]
        with torch.no_grad():
            x = x + dt * grad_x
            if L is not None:
                x = x.clamp(-L / 2.0, L / 2.0)
        if save_every is not None and ((i + 1) % save_every == 0):
            snaps.append(x.detach().cpu().numpy().copy())

    if save_every is not None:
        return np.stack(snaps, axis=0)
    return x.detach().cpu().numpy()


def make_V_fn(model: ScalarPotentialMLP, device: str | None = None):
    """Wrap a torch model as V_fn(x_np, t_float) → (M,) np.float64 (for tt_cross/Dense V_grid).

    Maps grid coordinates in [0, L) to model coordinates centred at 0 (model's
    sin/cos input encoding is periodic so any shift is consistent — but the
    the research prototype convention shifts by L/2, so we follow that).
    """
    if device is None:
        device = next(model.parameters()).device

    def V_fn(x: np.ndarray, t: float) -> np.ndarray:
        x_centred = x - model.L / 2.0
        x_t = torch.from_numpy(np.asarray(x_centred, dtype=np.float32)).to(device)
        t_t = torch.full((len(x_t), 1), float(t), device=device, dtype=torch.float32)
        with torch.no_grad():
            return model(x_t, t_t).squeeze(-1).cpu().numpy().astype(np.float64)

    return V_fn


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="checkpoint path (default: data/{dataset}/jam/seed{S}.pt)")
    parser.add_argument("--n_iter", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--time_embed_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d", type=int, default=None, help="override default d")
    parser.add_argument("--L", type=float, default=None, help="override default L")
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    out = args.out or f"data/{args.dataset}/jam/seed{args.seed}.pt"
    train(
        dataset=args.dataset, seed=args.seed, out_path=out,
        n_iter=args.n_iter, batch_size=args.batch_size,
        hidden=args.hidden, n_layers=args.n_layers,
        time_embed_dim=args.time_embed_dim, lr=args.lr,
        d=args.d, L=args.L, log_every=args.log_every,
    )


if __name__ == "__main__":
    _cli()
