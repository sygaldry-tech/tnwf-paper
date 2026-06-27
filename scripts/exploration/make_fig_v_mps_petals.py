"""Visualise petal samples from the end-to-end MPS Path-3 pipeline.

Two panels:
  LEFT  : MLP-Trotter-loss V_t  +  MPS WF pipeline (existing checkpoint)
  RIGHT : V_mps end-to-end       +  MPS WF pipeline (D_V_param=8, 2500 iters)

Both panels: 200 samples per snapshot, drawn from |ψ_{t_k}|² via
autoregressive sampling on the right-canonical MPS, coloured by
snapshot time (viridis). Bio data target in faint green underneath.
Saves to ``results/figures/petals_v_mps_compare.{pdf,png}``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import make_grid_centred, trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import forward_with_cache_mps  # noqa: E402
from train_v_mps_end_to_end import (  # noqa: E402
    _AdamMPS,
    _init_V_mps_list,
    _train_step,
    bin_samples_to_density,
)
from v_mps_parameterization import V_mps_to_dense  # noqa: E402

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.mps.core import right_canonicalize, sample_mps_indices


def _sample_from_mps(mps, n, N, L, rng):
    rc = right_canonicalize(mps)
    idx = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, idx.shape)
    return ((idx + jitter) * dx).astype(np.float64)


def _run_pipeline_samples(V_grids_np, psi0_np, N, d, L, K_data, alpha, beta,
                          D_max, D_V, n_samples=200, seed=0, n_substeps=1):
    cache = forward_with_cache_mps(
        psi0_np, V_grids_np, alpha=alpha, beta=beta,
        N=N, d=d, L=L, D_max=D_max, D_V=D_V, n_substeps=n_substeps,
    )
    rng = np.random.default_rng(seed)
    snaps = np.stack(
        [_sample_from_mps(cache[k * n_substeps * 8], n_samples, N, L, rng)
         for k in range(K_data + 1)],
        axis=0,
    )
    return snaps


def _panel(ax, snapshots, bg, title, K, axis_box, w2_t1):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
               linewidths=0, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=14,
                   c=[cmap(k / K)], alpha=0.92, edgecolors="white",
                   linewidths=0.3, rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])
    ax.text(0.02, 0.97, f"W₂@t=1: {w2_t1:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3.0))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K_data", type=int, default=4)
    p.add_argument("--n_substeps", type=int, default=1)
    # Default D_max=D_V=N gives full bond at d=2 — no MPS truncation,
    # avoids the missing-petal artefact that appears when D<N (see
    # RESULTS.md "Fifth round"). Override only if you want to study
    # truncation effects.
    p.add_argument("--D_max", type=int, default=None)
    p.add_argument("--D_V", type=int, default=None)
    p.add_argument("--D_V_param", type=int, default=None)
    p.add_argument("--n_iter", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mlp_ckpt",
                   default="data/petals_2d/jam_trotter_loss/seed0.pt")
    p.add_argument("--out_stem", default=None)
    args = p.parse_args()

    np.random.seed(0); torch.manual_seed(0)
    d, N, L = 2, args.N, 4.0
    K_data = args.K_data
    # Trotter coefficients use substep duration when n_substeps > 1
    delta_t_substep = 1.0 / (K_data * args.n_substeps)
    alpha, beta = trotter_coefficients(delta_t_substep, N, d, L)
    # Default bond = N (full rank at d=2). User can override.
    D_max = args.D_max if args.D_max is not None else N
    D_V = args.D_V if args.D_V is not None else N
    D_V_param = (args.D_V_param if args.D_V_param is not None
                 else max(N // 2, 8))

    target_traj = sample_petals_trajectory(n_per_step=2000, seed=0,
                                            n_timepoints=K_data + 1)
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)
    target_world = target_traj + L / 2.0

    half, cx = 1.25, L / 2.0
    axis_box = (cx - half, cx + half, cx - half, cx + half)

    bg = target_world.reshape(-1, 2)
    rng_bg = np.random.default_rng(0)
    bg = bg[rng_bg.choice(bg.shape[0], size=min(2500, bg.shape[0]), replace=False)]

    # ---------------- LEFT: existing MLP-autograd checkpoint ----------------
    print("Loading MLP-autograd Trotter-loss V_t...")
    ckpt = torch.load(args.mlp_ckpt, weights_only=False)
    cfg = ckpt["config"]
    model = ScalarPotentialMLP(
        d=d, hidden=cfg.get("hidden", 256),
        time_embed_dim=cfg.get("time_embed_dim", 64),
        L=L, n_layers=cfg.get("n_layers", 5),
    )
    model.load_state_dict(ckpt["model_state"]); model.eval()
    grid_centred = make_grid_centred(N, d, L, "cpu").to(torch.float32)
    V_grids_mlp = []
    with torch.no_grad():
        for k in range(K_data):
            t_mid = float((k + 0.5) / K_data)
            t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                                  dtype=torch.float32)
            V_grids_mlp.append(model(grid_centred, t_tensor).reshape(-1)
                               .to(torch.float64).numpy())
    snaps_mlp = _run_pipeline_samples(V_grids_mlp, psi0_np, N, d, L, K_data,
                                       alpha, beta, D_max, D_V,
                                       n_substeps=args.n_substeps)
    w2_mlp_t1, _ = wasserstein_2_subsampled(
        snaps_mlp[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)

    # ---------------- RIGHT: V_mps end-to-end training ----------------
    print(f"Training V_mps end-to-end ({args.n_iter} iters, "
          f"D_V_param={D_V_param}, lr={args.lr})...")
    rng_init = np.random.default_rng(0)
    V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param=D_V_param,
                                   rng=rng_init)
    flat = [c for cores in V_mps_list for c in cores]
    opt = _AdamMPS(flat, lr=args.lr)
    t0 = time.time()
    losses = []
    for it in range(args.n_iter):
        loss = _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                           K_data, alpha, beta, N, d, L, D_max, D_V,
                           n_substeps=args.n_substeps)
        losses.append(loss)
        if (it + 1) % 500 == 0:
            recent = float(np.mean(losses[-500:]))
            print(f"   iter {it + 1}/{args.n_iter}  loss={recent:.5f}  "
                  f"({(it + 1) / (time.time() - t0):.1f} it/s)")
    print(f"   done in {time.time() - t0:.1f}s   "
          f"final loss={float(np.mean(losses[-100:])):.5f}")
    V_grids_mps = [V_mps_to_dense(V_mps_list[k], N, d) for k in range(K_data)]
    snaps_mps = _run_pipeline_samples(V_grids_mps, psi0_np, N, d, L, K_data,
                                       alpha, beta, D_max, D_V,
                                       n_substeps=args.n_substeps)
    w2_mps_t1, _ = wasserstein_2_subsampled(
        snaps_mps[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(12.5, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.30], wspace=0.08)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2]); ax_legend.axis("off")

    n_mlp_params = sum(p.numel() for p in model.parameters())
    n_mps_params = sum(c.size for c in flat)
    _panel(ax_l, snaps_mlp, bg,
           f"MLP V$_t$  (~{n_mlp_params // 1000}k params)\n+ MPS WF pipeline",
           K_data, axis_box, float(w2_mlp_t1))
    _panel(ax_r, snaps_mps, bg,
           f"V$_t$ as MPS  ({n_mps_params} params, D=8)\n+ MPS WF pipeline",
           K_data, axis_box, float(w2_mps_t1))

    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                      markerfacecolor=cmap(i / K_data),
                      markeredgecolor="white", markersize=8,
                      markeredgewidth=0.3, label=f"$t = {i / K_data:.2f}$")
               for i in range(K_data + 1)]
    handles.append(Line2D([], [], marker="o", linestyle="",
                          markerfacecolor="#5b8a6c", markeredgecolor="none",
                          markersize=5, alpha=0.7, label="bio data target"))
    ax_legend.legend(handles=handles, loc="center left", frameon=False,
                     bbox_to_anchor=(0.0, 0.5), fontsize=10.5,
                     labelspacing=0.7, handletextpad=0.5,
                     title="snapshot time", title_fontsize=11)

    fig.suptitle(
        "Petals — MLP V$_t$ vs end-to-end MPS V$_t$ in the same WF pipeline  "
        f"(N={N}, K=4, D_max={D_max}, D_V_param={D_V_param}; "
        f"{args.n_iter}-iter MPS training)",
        fontsize=11, y=0.99)

    stem = args.out_stem or f"petals_v_mps_compare_N{N}"
    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
