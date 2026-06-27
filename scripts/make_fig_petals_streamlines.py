"""Petals velocity-field streamlines ∇V_t(x) at multiple times.

Loads a JAM checkpoint and renders ``∂_x V_t(x)`` as streamlines on a 2D grid
covering ``[0, L)^2`` for petals_2d, at several representative times spanning
the K=4 trajectory. The bio-data target snapshots are shown as faint
background dots so the reader can see which ring the streamlines should
push samples toward.

By default, renders side-by-side panels for the CFM-trained and AM-trained
JAM checkpoints (looks for ``data/petals_2d/jam/seed{seed}.pt`` and
``data/petals_2d/jam_am/seed{seed}.pt``). Pass ``--ckpts`` to override.

Writes ``results/figures/petals_streamlines{,_compare}.{pdf,png}``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tnwf.data.petals import sample_petals_trajectory
from tnwf.jam.train import DATASET_DEFAULTS, load_jam


def _grad_V_grid(model, L: float, t_eval: float, n_grid: int = 50,
                 margin: float = 0.02):
    """Return ``(X, Y, vx, vy)`` for ``∇V_t`` on a world-frame grid."""
    lo, hi = margin * L, (1.0 - margin) * L
    lin = np.linspace(lo, hi, n_grid)
    X, Y = np.meshgrid(lin, lin)
    pts_world = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    pts_t = torch.from_numpy(pts_world).clone().requires_grad_(True)
    pts_c = pts_t - L / 2.0
    t_t = torch.full((pts_world.shape[0], 1), float(t_eval))
    V = model(pts_c, t_t).sum()
    grad = torch.autograd.grad(V, pts_t)[0].detach().numpy()
    vx = grad[:, 0].reshape(n_grid, n_grid)
    vy = grad[:, 1].reshape(n_grid, n_grid)
    return X, Y, vx, vy


def _plot_streamline_panel(ax, model, L: float, t_eval: float,
                            target_at_t, n_grid: int = 50):
    X, Y, vx, vy = _grad_V_grid(model, L=L, t_eval=t_eval, n_grid=n_grid)
    # Magnitude for color
    mag = np.sqrt(vx ** 2 + vy ** 2)
    strm = ax.streamplot(X, Y, vx, vy, color=mag, cmap="viridis",
                         density=1.4, linewidth=0.8, arrowsize=0.9)
    if target_at_t is not None:
        # Shift target to world frame for the panel
        pts = target_at_t + L / 2.0
        ax.scatter(pts[:, 0], pts[:, 1], s=4.0, c="#cccccc",
                   alpha=0.6, edgecolors="none", rasterized=True)
    ax.set_title(f"$t = {t_eval:.2f}$", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    return strm


def _render_ckpt(ckpt_path: Path, label: str, times, target_traj, L,
                 out_pdf: Path, out_png: Path):
    model, cfg = load_jam(str(ckpt_path), device="cpu")
    n_cols = len(times)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols + 0.4, 3.0))
    if n_cols == 1:
        axes = [axes]
    last_strm = None
    for ax, t_eval in zip(axes, times):
        # Pick the bio snapshot whose normalised t is closest to t_eval
        K = target_traj.shape[0] - 1
        bin_k = int(round(t_eval * K))
        last_strm = _plot_streamline_panel(ax, model, L=L,
                                           t_eval=t_eval,
                                           target_at_t=target_traj[bin_k])
    fig.colorbar(last_strm.lines, ax=axes[-1],
                 label="‖∇V_t(x)‖", fraction=0.045, pad=0.02)
    fig.suptitle(
        f"Petals — JAM velocity-field streamlines  ({label})",
        fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--times", type=str, default="0.05,0.25,0.5,0.75,0.95",
                   help="Comma-separated times in [0,1] to render streamlines for.")
    p.add_argument("--ckpts", type=str, default="data/petals_2d/jam,data/petals_2d/jam_am",
                   help="Comma-separated JAM checkpoint dirs to render. Each "
                        "produces a separate row of streamline panels.")
    p.add_argument("--out_dir", default="results/figures")
    p.add_argument("--n_grid", type=int, default=50)
    args = p.parse_args()

    times = [float(t) for t in args.times.split(",") if t]
    ckpt_dirs = [Path(d) for d in args.ckpts.split(",") if d]
    L = DATASET_DEFAULTS["petals_2d"]["L"]

    target_traj = sample_petals_trajectory(n_per_step=400, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-checkpoint figure
    rendered: list[tuple[str, Path, Path]] = []
    for ckpt_dir in ckpt_dirs:
        ckpt = ckpt_dir / f"seed{args.seed}.pt"
        if not ckpt.exists():
            print(f"[skip] no checkpoint at {ckpt}")
            continue
        label = ckpt_dir.name
        stem = f"petals_streamlines_{label}_seed{args.seed}"
        pdf = out_dir / f"{stem}.pdf"
        png = out_dir / f"{stem}.png"
        _render_ckpt(ckpt, label, times, target_traj, L, pdf, png)
        print(f"wrote {png}")
        rendered.append((label, pdf, png))

    # Combined comparison if more than one
    if len(rendered) >= 2:
        n_rows = len(rendered)
        n_cols = len(times)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.6 * n_cols + 0.6, 3.0 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])
        last_strm = None
        for r, (label, _, _) in enumerate(rendered):
            ckpt = ckpt_dirs[r] / f"seed{args.seed}.pt"
            model, _cfg = load_jam(str(ckpt), device="cpu")
            for c, t_eval in enumerate(times):
                ax = axes[r, c]
                K = target_traj.shape[0] - 1
                bin_k = int(round(t_eval * K))
                last_strm = _plot_streamline_panel(
                    ax, model, L=L, t_eval=t_eval,
                    target_at_t=target_traj[bin_k], n_grid=args.n_grid)
                if c == 0:
                    ax.set_ylabel(label, fontsize=10.5, fontweight="bold")
        fig.colorbar(last_strm.lines, ax=axes[:, -1].tolist(),
                     label="‖∇V_t(x)‖", fraction=0.04, pad=0.02)
        fig.suptitle("Petals — JAM velocity-field streamlines, CFM vs AM training",
                     fontsize=11, y=0.995)
        cmp_pdf = out_dir / "petals_streamlines.pdf"
        cmp_png = out_dir / "petals_streamlines.png"
        fig.savefig(cmp_pdf, bbox_inches="tight")
        fig.savefig(cmp_png, bbox_inches="tight", dpi=150)
        print(f"wrote {cmp_png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
