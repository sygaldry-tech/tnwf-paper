"""Warm-start V_mps from MLP-trained V_t and refine via end-to-end MPS adjoint.

At N>16 training V_mps from random init is slow (needs lr ≪ 1e-2 for
stability + many iters). Warm-starting from the MLP's converged V_grids
shows the architecture can ACCEPT a pre-trained V and refine it
further. Procedure:

1. Load MLP V_t checkpoint trained at this N (matched-grid).
2. Evaluate MLP on the grid at each segment midpoint → V_grids dense.
3. SVD-compress each V_grid → V_mps with bond ``D_V_param`` via
   ``V_dense_to_mps``.
4. Continue end-to-end training with small lr.

Goal: demonstrate that the V_mps representation at moderate bond is
sufficient to express the MLP's optimum, AND that further refinement
under the MPS adjoint at small lr is stable.
"""
from __future__ import annotations

import argparse
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
    _train_step,
    bin_samples_to_density,
)
from v_mps_parameterization import (  # noqa: E402
    V_dense_to_mps,
    V_mps_to_dense,
)
from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.scalar_potential import ScalarPotentialMLP
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.mps.core import (
    mps_to_dense,
    right_canonicalize,
    sample_mps_indices,
)


def _sample_from_mps(mps, n, N, L, rng):
    rc = right_canonicalize(mps)
    idx = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, idx.shape)
    return ((idx + jitter) * dx).astype(np.float64)


def _run_pipeline_samples(V_grids_np, psi0_np, N, d, L, K_data, alpha, beta,
                           D_max, D_V, n_samples=200, seed=0):
    cache = forward_with_cache_mps(
        psi0_np, V_grids_np, alpha=alpha, beta=beta,
        N=N, d=d, L=L, D_max=D_max, D_V=D_V,
    )
    rng = np.random.default_rng(seed)
    return np.stack([_sample_from_mps(cache[k * 8], n_samples, N, L, rng)
                      for k in range(K_data + 1)], axis=0)


def _panel(ax, snapshots, bg, title, K, axis_box, w2_t1):
    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
               linewidths=0, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=14,
                   c=[cmap(k / K)], alpha=0.92, edgecolors="white",
                   linewidths=0.3, rasterized=True)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])
    ax.text(0.02, 0.97, f"W₂@t=1: {w2_t1:.3f}",
            transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=16)
    p.add_argument("--D_V_param", type=int, default=16)
    p.add_argument("--n_iter", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mlp_ckpt", required=True)
    args = p.parse_args()

    np.random.seed(0); torch.manual_seed(0)
    d, L = 2, 4.0
    N, K_data = args.N, 4
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max, D_V = args.D_max, args.D_V

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

    # ---- 1. MLP V_t baseline samples
    print(f"[N={N}] Loading MLP V_t checkpoint...")
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
                                       alpha, beta, D_max, D_V)
    w2_mlp_t1, _ = wasserstein_2_subsampled(
        snaps_mlp[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)

    # ---- 2. Warm-start V_mps from MLP V_grids, evaluate before refining
    print(f"[N={N}] Warm-starting V_mps from MLP V_grids "
          f"(D_V_param={args.D_V_param})...")
    V_mps_list = [V_dense_to_mps(V, N, d, args.D_V_param) for V in V_grids_mlp]
    V_grids_warm = [V_mps_to_dense(cores, N, d) for cores in V_mps_list]
    rec_err = max(np.linalg.norm(V - W) / np.linalg.norm(V)
                  for V, W in zip(V_grids_mlp, V_grids_warm))
    print(f"   max relative reconstruction error (MLP → V_mps): {rec_err:.4f}")

    # ---- 3. Refine via MPS adjoint
    print(f"[N={N}] Refining V_mps for {args.n_iter} iters at lr={args.lr}...")
    flat = [c for cores in V_mps_list for c in cores]
    opt = _AdamMPS(flat, lr=args.lr)
    t0 = time.time()
    losses = []
    for it in range(args.n_iter):
        loss = _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                           K_data, alpha, beta, N, d, L, D_max, D_V)
        losses.append(loss)
        if (it + 1) % 100 == 0:
            print(f"   iter {it + 1}/{args.n_iter}  loss={np.mean(losses[-100:]):.5f}")
    print(f"   done in {time.time() - t0:.1f}s")

    V_grids_refined = [V_mps_to_dense(cores, N, d) for cores in V_mps_list]
    snaps_refined = _run_pipeline_samples(V_grids_refined, psi0_np, N, d, L,
                                            K_data, alpha, beta, D_max, D_V)
    w2_refined_t1, _ = wasserstein_2_subsampled(
        snaps_refined[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)

    # ---- 4. Figure
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
    _panel(ax_r, snaps_refined, bg,
           f"V$_t$ warm-started as MPS  ({n_mps_params} params, D={args.D_V_param})\n"
           f"+ {args.n_iter} iters refinement",
           K_data, axis_box, float(w2_refined_t1))

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
        f"Petals N={N} — V_mps warm-started from MLP V_t (compression × {n_mlp_params / n_mps_params:.0f})  "
        f"reconstruction err {rec_err:.3f}",
        fontsize=11, y=0.99)

    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"petals_v_mps_warmstart_N{N}.pdf"
    png = out_dir / f"petals_v_mps_warmstart_N{N}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}\nwrote {png}")
    print(f"\n=== RESULT ===")
    print(f"  rel. recon error    : {rec_err:.4f}")
    print(f"  W₂@t=1  MLP         : {w2_mlp_t1:.4f}")
    print(f"  W₂@t=1  V_mps warm  : {w2_refined_t1:.4f}")


if __name__ == "__main__":
    main()
