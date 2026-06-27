"""Paper figure: AM (NN) vs Adjoint (V_mps) — trajectories + W2/MMD-vs-t.

Layout (2 rows × 2 cols):

  Top row:
    [Action Matching trajectories]   [Adjoint method scatter]
    (NN; ~281k params)               (V_mps; ~4k params)

  Bottom row (spans full width):
    W2 distance vs time              MMD vs time
    overlaid with eAM, MIOFlow (Neklyudov et al. 2023, petals)

The top-left panel re-renders the AM trajectories using the same JAM
checkpoint as paper_fig3_replication's right panel. The top-right panel
re-renders the Adjoint sample scatter using the cached V_grids from the
N=32 D=32 radam_compare run. The bottom panels read W2/MMD per snapshot
from the same cache.

Inputs (must exist locally):
  data/petals_2d_K14/jam_am_softglobot/seed0.pt           — AM JAM ckpt
  results/figures/radam_compare_cache_N32_D32.npz         — Adjoint cache

Outputs:
  results/figures/paper_method_compare.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent / "exploration"))

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS, load_jam, sample_gradient_flow


def _am_panel(ax, ckpt: str, L: float, K: int, n_particles: int,
               steps_per_segment: int, q0_centred: np.ndarray, bg: np.ndarray,
               axis_box, title: str, seed: int = 0):
    """Render the AM-trajectories panel (mirrors paper_fig3_replication
    right panel). Returns the param count of the loaded JAM model."""
    model, _ = load_jam(ckpt, device="cpu")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, q0_centred.shape[0], size=n_particles)
    z0 = q0_centred[idx].astype(np.float32)
    n_steps = K * steps_per_segment
    snaps = sample_gradient_flow(model, torch.from_numpy(z0),
                                  n_steps=n_steps, save_every=1, L=L)
    smooth = snaps.astype(np.float32) + L / 2.0
    snapshots = smooth[::steps_per_segment]

    cmap = plt.cm.viridis
    ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#5b8a6c", alpha=0.55,
                linewidths=0, rasterized=True)
    for i in range(smooth.shape[1]):
        ax.plot(smooth[:, i, 0], smooth[:, i, 1], color="#f5b800",
                alpha=0.50, lw=0.6, rasterized=True)
    for k in range(K + 1):
        ax.scatter(snapshots[k, :, 0], snapshots[k, :, 1], s=15,
                    c=[cmap(k / K)], alpha=0.92, edgecolors="white",
                    linewidths=0.3, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])
    ax.set_title(f"{title}\n(NN, {n_params:,} params)",
                  fontsize=11, fontweight="bold")
    return n_params


def _adjoint_scatter_panel(ax, snaps_per_k, w2_at_T: float, n_params: int,
                            t_axis, L: float, axis_box, title: str):
    cmap = plt.cm.viridis
    K_plus_1 = len(snaps_per_k)
    for k, samples in enumerate(snaps_per_k):
        c = cmap(k / max(K_plus_1 - 1, 1))
        ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.5,
                    color=c, edgecolors="none", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(axis_box[0], axis_box[1]); ax.set_ylim(axis_box[2], axis_box[3])
    ax.set_title(f"{title}\n(V$_\\mathrm{{mps}}$, {n_params:,} params; "
                  f"$W_2$@$t{{=}}1 = {w2_at_T:.3f}$)",
                  fontsize=11, fontweight="bold")


def _w2_mmd_row(ax_w2, ax_mmd, t_axis, w2_adjoint, mmd_adjoint):
    """Bottom row: W2 and MMD vs time, Adjoint vs eAM, MIOFlow.
    Riemannian dropped per request (we're highlighting only the adjoint
    method on this paper figure)."""
    ax_w2.plot(t_axis, w2_adjoint, color="#1f77b4", lw=1.8, marker="o",
                markersize=5, label="Adjoint (ours)")
    ax_mmd.plot(t_axis, mmd_adjoint, color="#1f77b4", lw=1.8, marker="o",
                 markersize=5)

    # Paper overlays at our 5 timepoints (eyeballed from
    # Neklyudov et al. 2023 Fig 2).
    w2_eam     = np.array([0.02, 0.06, 0.08, 0.10, 0.13])
    w2_mioflow = np.array([0.02, 0.16, 0.22, 0.27, 0.30])
    mmd_eam     = np.array([0.005, 0.020, 0.015, 0.013, 0.015])
    mmd_mioflow = np.array([0.005, 0.13,  0.14,  0.16,  0.17])

    ax_w2.plot(t_axis, w2_eam, color="#ff7f0e", linestyle="--", lw=1.4,
                marker="^", markersize=5, markerfacecolor="white",
                label="entropic AM (Neklyudov et al. 2023)")
    ax_w2.plot(t_axis, w2_mioflow, color="#17becf", linestyle="--", lw=1.4,
                marker="v", markersize=5, markerfacecolor="white",
                label="MIOFlow (Huguet et al. 2022)")
    ax_mmd.plot(t_axis, mmd_eam, color="#ff7f0e", linestyle="--", lw=1.4,
                 marker="^", markersize=5, markerfacecolor="white")
    ax_mmd.plot(t_axis, mmd_mioflow, color="#17becf", linestyle="--", lw=1.4,
                 marker="v", markersize=5, markerfacecolor="white")

    ax_w2.set_xlabel("time"); ax_w2.set_ylabel(r"$W_2$ distance")
    ax_mmd.set_xlabel("time"); ax_mmd.set_ylabel("MMD")
    ax_w2.set_xticks(t_axis); ax_mmd.set_xticks(t_axis)
    ax_w2.legend(loc="best", fontsize=8)
    ax_w2.grid(alpha=0.25); ax_mmd.grid(alpha=0.25)


def main():
    # ---- paths
    # Same checkpoint as the right panel of paper_fig3_replication
    # (K=4 petals; AM-loss + soft global OT JAM training).
    am_ckpt = "data/petals_2d/jam_am_softglobot/seed0.pt"
    cache_path = "results/figures/radam_compare_cache_N32_D32.npz"

    # ---- AM panel inputs
    L_petals = DATASET_DEFAULTS["petals_2d"]["L"]
    K_petals = int(DATASET_DEFAULTS["petals_2d"]["trajectory_K"])
    target_traj = sample_petals_trajectory(n_per_step=400, seed=0)
    q0_centred = target_traj[0]
    bg = (target_traj.reshape(-1, 2) + L_petals / 2.0)
    rng = np.random.default_rng(0)
    bg = bg[rng.choice(bg.shape[0], size=min(2500, bg.shape[0]), replace=False)]
    pad = 0.25
    cx = L_petals / 2.0
    half = 1.25
    axis_box_am = (cx - half, cx + half, cx - half, cx + half)

    # ---- Adjoint panel + bottom-row inputs (from cache)
    cache = np.load(cache_path, allow_pickle=True)
    iter_budgets = list(cache["iter_budgets"])
    best_idx = int(np.argmax(iter_budgets))
    snaps_adjoint = [np.asarray(arr, dtype=np.float64)
                      for arr in cache["snaps_adam"][best_idx]]
    w2_adjoint = np.asarray(cache["w2_adam"][best_idx], dtype=np.float64)
    mmd_adjoint = np.asarray(cache["mmd_adam"][best_idx], dtype=np.float64)
    K_data_adj = len(snaps_adjoint) - 1
    t_axis = np.array([k / K_data_adj for k in range(K_data_adj + 1)])

    # V_mps param count: K_data * (1*N*D + D*N*1) = K_data * 2 * N * D
    # (for d=2 with bond cap D, here N=32, D=16, K=4)
    N_adj = 32
    D_V_param = 16
    n_vmps_params = K_data_adj * (1 * N_adj * D_V_param + D_V_param * N_adj * 1)

    # Pick a tight crop matching the petal extent in world coords
    # (radam_compare_fig3 used 0..L = 0..4 with the cluster centred at L/2).
    L_adj = 4.0
    half_adj = 1.6
    axis_box_adj = (L_adj / 2 - half_adj, L_adj / 2 + half_adj,
                    L_adj / 2 - half_adj, L_adj / 2 + half_adj)

    # ---- Figure layout
    fig = plt.figure(figsize=(12.5, 11.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                           hspace=0.30, wspace=0.18)
    ax_am = fig.add_subplot(gs[0, 0])
    ax_adj = fig.add_subplot(gs[0, 1])

    am_params = _am_panel(
        ax_am, ckpt=am_ckpt, L=L_petals, K=K_petals,
        n_particles=80, steps_per_segment=250,
        q0_centred=q0_centred, bg=bg, axis_box=axis_box_am,
        title="Action Matching + soft global OT (NN, classical baseline)")

    _adjoint_scatter_panel(
        ax_adj, snaps_per_k=snaps_adjoint, w2_at_T=float(w2_adjoint[-1]),
        n_params=n_vmps_params, t_axis=t_axis, L=L_adj,
        axis_box=axis_box_adj,
        title=f"Adjoint method (ours, $N=${N_adj}, $D_V$=${D_V_param}$, 2000 iters)")

    # Bottom row: two side-by-side subplots inside the lower half of gs
    inner_gs = gs[1, :].subgridspec(1, 2, wspace=0.22)
    ax_w2  = fig.add_subplot(inner_gs[0, 0])
    ax_mmd = fig.add_subplot(inner_gs[0, 1])
    _w2_mmd_row(ax_w2, ax_mmd, t_axis, w2_adjoint, mmd_adjoint)

    # Shared time-colour legend in the upper row (one only)
    cmap = plt.cm.viridis
    handles = [Line2D([], [], marker="o", linestyle="",
                        markerfacecolor=cmap(i / 4), markeredgecolor="white",
                        markersize=8, markeredgewidth=0.3,
                        label=f"$t = {i / 4:.2f}$")
                for i in range(5)]
    ax_adj.legend(handles=handles, loc="upper right", fontsize=8,
                   framealpha=0.9)

    fig.suptitle(
        "Petal-trajectory generation: NN-based Action Matching vs. "
        "MPS-adjoint pipeline (ours).  "
        f"~{am_params//1000}k vs. ~{n_vmps_params//1000}k parameters",
        fontsize=12, y=0.995)

    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "paper_method_compare.pdf"
    png = out_dir / "paper_method_compare.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"AM JAM params: {am_params:,}")
    print(f"V_mps params:  {n_vmps_params:,}  "
          f"(compression {am_params/n_vmps_params:.1f}×)")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
