"""Figure 2: Dense wavefunction evolution at N=64 for Swiss roll.

Runs dense evolution end-to-end (no MPS truncation), capturing the
wavefunction at six time points along the integration path. We plot
two rows for the Swiss-roll target:
  - amplitude+phase visualisation (hue = arg(psi), brightness = |psi|),
  - probability mass function |psi|^2 in a light sequential colormap.

Total 2 rows x 6 cols. The columns step uniformly through t in [0, 1].
The GMM target previously occupied rows 3-4 has been moved to a
flow-matching-style figure (see make_fig_gmm_ode.py).

Usage:
    uv run python scripts/make_fig2_dense_evolution.py \\
        --out figures/fig2_dense.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from tnwf.dense.evolution import (
    apply_K_step,
    apply_V_step,
    make_kinetic_eigenvalues,
    trotter_coefficients,
)
from tnwf.grid import make_grid
from tnwf.jam.train import DATASET_DEFAULTS
from tnwf.pipelines.run_evolution import initial_psi_dense
from tnwf.theory import make_analytic_V_fn


def evolve_dense_with_snapshots(dataset: str, N: int, K: int,
                                 snapshot_ts: list[float],
                                 sigma_0: float = 1.0,
                                 jam_ckpt: str | None = None):
    """Re-implement the per-K-step product formula and snapshot psi at
    the requested t values.

    For GMM datasets the analytic V_t is used; for Swiss roll the
    user-supplied JAM checkpoint provides the velocity oracle.

    Returns a list of (t, psi_2d) tuples, where psi_2d is the (N, N)
    complex wavefunction at the *post-step* time t.
    """
    cfg = DATASET_DEFAULTS[dataset]
    d = cfg["d"]; L = cfg["L"]
    if d != 2:
        raise ValueError(f"Only d=2 supported here; got d={d}")
    psi = initial_psi_dense(N=N, d=d, L=L, sigma=sigma_0)
    if dataset == "swiss_roll_2d":
        if jam_ckpt is None:
            jam_ckpt = "data/swiss_roll_2d/jam/seed0.pt"
        from tnwf.jam.train import load_jam, make_V_fn
        model, _cfg = load_jam(jam_ckpt, device="cpu")
        V_fn = make_V_fn(model, device="cpu")
    else:
        V_fn = make_analytic_V_fn(dataset)
    grid = make_grid(N=N, d=d, L=L)
    eig = make_kinetic_eigenvalues(N=N, d=d, L=L)
    delta_t = 1.0 / K
    alpha, beta = trotter_coefficients(delta_t, N=N, d=d, L=L)

    def _V_2d(t: float) -> np.ndarray:
        # Guard against the t→0 singularity of the analytic GMM
        # potential; the streamline overlay only needs the velocity
        # direction, not its (divergent) magnitude near t=0.
        return np.asarray(V_fn(grid, max(t, 0.05))).reshape(N, N)

    snapshots = [(0.0, psi.copy().reshape(N, N), _V_2d(0.0))]
    snap_set = sorted(snapshot_ts)
    for k in range(K):
        t_mid = max((k + 0.5) * delta_t, 1e-5)
        V_grid = np.asarray(V_fn(grid, t_mid)).ravel()
        # 8-step product formula (matches dense.evolution.apply_product_formula).
        psi = apply_K_step(psi, alpha,  N=N, d=d, L=L, eigenvalues=eig)
        psi = apply_V_step(psi, beta,   V_grid)
        psi = apply_K_step(psi, -alpha, N=N, d=d, L=L, eigenvalues=eig)
        psi = apply_V_step(psi, -beta,  V_grid)
        psi = apply_K_step(psi, -alpha, N=N, d=d, L=L, eigenvalues=eig)
        psi = apply_V_step(psi, -beta,  V_grid)
        psi = apply_K_step(psi, alpha,  N=N, d=d, L=L, eigenvalues=eig)
        psi = apply_V_step(psi, beta,   V_grid)
        t_now = (k + 1) * delta_t
        snapshots.append((t_now, psi.copy().reshape(N, N), _V_2d(t_now)))
    # Pick the snapshots closest to the requested t values.
    chosen = []
    for t_req in snap_set:
        i_closest = int(np.argmin([abs(s[0] - t_req) for s in snapshots]))
        chosen.append(snapshots[i_closest])
    return chosen


def _build_cb_cyclic_cmap() -> mcolors.ListedColormap:
    """Wong colorblind-safe 4-anchor cyclic colormap, identical to
    `_build_cyclic_colormap("cb_cyclic")` in
    `the research prototype's analysis/wf_visualize.py`. The source PNG uses this."""
    blue   = [  0/255, 114/255, 178/255]
    sky    = [ 86/255, 180/255, 233/255]
    orange = [230/255, 159/255,   0/255]
    purple = [204/255, 121/255, 167/255]
    t_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    anchors = np.array([blue, sky, orange, purple, blue])
    t = np.linspace(0.0, 1.0, 256)
    rgb = np.stack([np.interp(t, t_nodes, anchors[:, c])
                    for c in range(3)], axis=1)
    return mcolors.ListedColormap(rgb, name="cb_cyclic")


_CB_CYCLIC = _build_cb_cyclic_cmap()


def complex_to_rgb(psi: np.ndarray) -> np.ndarray:
    """Light-background phase + amplitude visualisation matching
    `wf_visualize.py::psi_to_hsv_rgba(dark_bg=False, colormap="cb_cyclic")`:

        a   = |psi| / max|psi|                # amplitude weight
        rgb = (1 - a) * white + a * cb_cyclic(arg(psi))

    Zero amplitude → white; full amplitude → Wong colorblind-safe hue.
    """
    amp = np.abs(psi).astype(np.float64)
    phase = np.angle(psi)
    phase_norm = (phase + np.pi) / (2.0 * np.pi)
    cmap_rgb = _CB_CYCLIC(phase_norm)[..., :3]
    max_amp = amp.max() if amp.max() > 0 else 1.0
    a = (amp / max_amp)[..., None]
    rgb = (1.0 - a) + a * cmap_rgb
    return np.clip(rgb, 0.0, 1.0)


def amp_colormap() -> mcolors.ListedColormap:
    """Amplitude colourbar: white -> reference colour at arg=0
    (cb_cyclic at t=0.5, which is orange #E69F00). Matches the
    centre colorbar in the source PNG."""
    ref_rgb = np.array(_CB_CYCLIC(0.5)[:3])
    t = np.linspace(0, 1, 256)
    rgb = np.clip((1.0 - t[:, None]) + t[:, None] * ref_rgb[None, :], 0, 1)
    return mcolors.ListedColormap(rgb, name="_amp")


def phase_colormap() -> mcolors.ListedColormap:
    return _CB_CYCLIC


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig2_dense.pdf")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--K", type=int, default=40,
                   help="number of Trotter steps")
    args = p.parse_args()

    snap_ts = [0.0, 0.40, 0.70, 0.85, 0.95, 1.0]

    print(f"Running dense evolution for swiss_roll_2d (N={args.N}, K={args.K})...")
    sr_snaps = evolve_dense_with_snapshots(
        "swiss_roll_2d", N=args.N, K=args.K, snapshot_ts=snap_ts,
    )

    # ── Layout: 3 colorbars (top) + 2 data rows × 6 cols ────────────
    n_cols = len(snap_ts)
    fig = plt.figure(figsize=(2.0 * n_cols + 0.8, 6.2), dpi=300,
                     constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.30)
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[0.13, 1.0], hspace=0.25)
    cbar_gs = outer_gs[0].subgridspec(1, 3, wspace=0.6)
    data_gs = outer_gs[1].subgridspec(2, n_cols)

    pmf_cmap = "Oranges"

    # ── Top-row colorbars (match wf_visualize.py source-PNG style) ───
    cax_phase = fig.add_subplot(cbar_gs[0])
    cax_amp   = fig.add_subplot(cbar_gs[1])
    cax_pmf   = fig.add_subplot(cbar_gs[2])
    phase_cmap_obj = phase_colormap()
    amp_cmap_obj   = amp_colormap()
    for cax, cmap_obj, vmin, vmax, label, ticks, tick_labels in (
        (cax_phase, phase_cmap_obj, -np.pi, np.pi,
         r"phase $\arg(\psi)$  [rad]",
         [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
         [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]),
        (cax_amp,   amp_cmap_obj,   0.0, 1.0,
         r"$|\psi|\,/\,\max|\psi|$",
         [0.0, 0.5, 1.0], ["0", "0.5", "1"]),
        (cax_pmf,   plt.get_cmap(pmf_cmap), 0.0, 1.0,
         r"$|\psi|^{2}\,/\,\max|\psi|^{2}$",
         [0.0, 0.5, 1.0], ["0", "0.5", "1"]),
    ):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        # ticks, tick labels, and the colorbar label all above the bar
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(label, fontsize=10)
        cb.set_ticks(ticks)
        cb.set_ticklabels(tick_labels)
        cb.ax.tick_params(labelsize=9)

    # ── Data rows: 2 rows × n_cols, math-only row labels ───────────
    row_specs = [
        (sr_snaps,  "amp_phase", r"$\psi$"),
        (sr_snaps,  "pmf",       r"$|\psi|^2$"),
    ]
    axes = np.empty((len(row_specs), n_cols), dtype=object)
    for row_idx, (snaps, kind, kind_label) in enumerate(row_specs):
        for col_idx, (t, psi, V_2d) in enumerate(snaps):
            ax = fig.add_subplot(data_gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax
            if kind == "amp_phase":
                ax.imshow(complex_to_rgb(psi), origin="lower",
                          interpolation="bicubic", aspect="equal")
            else:
                pmf = np.abs(psi) ** 2
                pmf_max = pmf.max() if pmf.max() > 0 else 1.0
                ax.imshow(pmf / pmf_max, origin="lower", cmap=pmf_cmap,
                          interpolation="bicubic", aspect="equal",
                          vmin=0.0, vmax=1.0)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"$t = {t:.2f}$", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(kind_label, fontsize=14, rotation=0,
                              ha="right", va="center", labelpad=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
