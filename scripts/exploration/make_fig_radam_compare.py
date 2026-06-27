"""Render Fig-2 / Fig-3 analogues comparing Adjoint and Adjoint+Riemannian.

Both methods share the MPS-adjoint backward pass; they differ only in the
V_mps optimizer:

    * "Adjoint"             = adjoint backward + Euclidean Adam
    * "Adjoint + Riemannian" = adjoint backward + Riemannian Adam (QR
                               retraction + Stiefel-tangent gradient)

Fig-2 analogue: W₂ and MMD per timepoint t ∈ {0, 0.25, 0.5, 0.75, 1.0} for
both methods at three training-iter budgets (analogue of the AM/MIOFlow
paper's "5, 10, 15 steps" inference axis).

Fig-3 analogue: petal-trajectory scatter at the best training budget for
each method, samples coloured by snapshot time, training data overlaid.

Usage::

    uv run python scripts/exploration/make_fig_radam_compare.py [--N 16]

Saves to ``results/figures/radam_compare_fig2.{pdf,png}`` and
``results/figures/radam_compare_fig3.{pdf,png}``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import forward_with_cache_mps  # noqa: E402
from riemannian_adam_mps import _RiemannianAdamMPS, init_v_mps_riemannian  # noqa: E402
from train_v_mps_end_to_end import (  # noqa: E402
    _AdamMPS,
    _init_V_mps_list,
    _train_step,
    bin_samples_to_density,
)
from v_mps_parameterization import V_mps_to_dense  # noqa: E402

from tnwf.data.petals import sample_petals_trajectory
from tnwf.data.embryoid_body import sample_eb_trajectory
from tnwf.metrics.mmd import mmd_rbf
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.mps.core import right_canonicalize, sample_mps_indices


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "petals_2d": dict(d=2, L=4.0,
                       sampler=lambda n, seed, K:
                           sample_petals_trajectory(n_per_step=n, seed=seed,
                                                     n_timepoints=K + 1)),
    "eb_5d":     dict(d=5, L=10.0,
                       sampler=lambda n, seed, K:
                           sample_eb_trajectory(n_per_step=n, seed=seed)),
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_adam(K_data, N, d, D_V_param, psi0_np, q_grids_np,
                 alpha, beta, L, D_max, D_V, n_iter, lr, seed):
    rng = np.random.default_rng(seed)
    V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param, rng=rng)
    flat = [c for cores in V_mps_list for c in cores]
    opt = _AdamMPS(flat, lr=lr)
    losses = []
    for _ in range(n_iter):
        losses.append(_train_step(V_mps_list, opt, psi0_np, q_grids_np,
                                   K_data, alpha, beta, N, d, L, D_max, D_V,
                                   optimizer_kind="adam"))
    return V_mps_list, losses


def _train_radam(K_data, N, d, D_V_param, psi0_np, q_grids_np,
                  alpha, beta, L, D_max, D_V, n_iter, lr, seed):
    rng = np.random.default_rng(seed)
    V_mps_list = init_v_mps_riemannian(K_data, N, d, D_V_param, rng=rng)
    opts = [_RiemannianAdamMPS(V_mps_list[k], lr=lr) for k in range(K_data)]
    losses = []
    for _ in range(n_iter):
        losses.append(_train_step(V_mps_list, opts, psi0_np, q_grids_np,
                                   K_data, alpha, beta, N, d, L, D_max, D_V,
                                   optimizer_kind="radam"))
    return V_mps_list, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _autoreg_sample(mps, n, N, L, rng):
    rc = right_canonicalize(mps)
    idx = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, idx.shape)
    return ((idx + jitter) * dx).astype(np.float64)


def _eval_per_timepoint(V_mps_list, psi0_np, target_world, N, d, L,
                         K_data, alpha, beta, D_max, D_V,
                         n_samples=400, seed=0):
    """Returns (snapshots, w2_per_t, mmd_per_t)."""
    V_grids = [V_mps_to_dense(V_mps_list[k], N, d) for k in range(K_data)]
    cache = forward_with_cache_mps(
        psi0_np, V_grids,
        alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V,
    )
    rng = np.random.default_rng(seed)
    snapshots, w2_list, mmd_list = [], [], []
    for k in range(K_data + 1):
        mps_k = cache[k * 8]
        samples = _autoreg_sample(mps_k, n=n_samples, N=N, L=L, rng=rng)
        target_k = target_world[k].astype(np.float64)
        # take same-size subsample of target for stable MMD/W2
        n_sub = min(200, samples.shape[0], target_k.shape[0])
        w2, _ = wasserstein_2_subsampled(
            samples, target_k, n_sub=n_sub, n_repeats=3, seed=seed)
        # MMD on the subsampled sets (matches sample size for fair compare)
        idx_s = rng.choice(samples.shape[0], n_sub, replace=False)
        idx_t = rng.choice(target_k.shape[0], n_sub, replace=False)
        mmd = mmd_rbf(samples[idx_s], target_k[idx_t])
        snapshots.append(samples)
        w2_list.append(float(w2))
        mmd_list.append(float(mmd))
    return snapshots, np.array(w2_list), np.array(mmd_list)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

ADAM_COLOR = "#1f77b4"      # blue
RADAM_COLOR = "#d62728"     # red

def _plot_fig2(ax_w2, ax_mmd, t_axis,
                w2_adam_by_iter, mmd_adam_by_iter,
                w2_radam_by_iter, mmd_radam_by_iter,
                iter_budgets,
                overlay_paper: bool = True):
    """Plot W₂ vs t and MMD vs t for the largest iter budget only.

    If ``overlay_paper`` is true, also overlay approximate W₂ and MMD
    values for entropic AM and MIOFlow on the petals dataset, read from
    Figure 2 of Neklyudov et al. (2023). These are eyeball estimates of
    the median (over the 5/10/15 step counts) at the paper's six
    timepoints ``t ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}``.
    """
    # Largest budget only.
    idx = int(np.argmax(iter_budgets))
    n_iter = iter_budgets[idx]
    ax_w2.plot(t_axis, w2_adam_by_iter[idx], color=ADAM_COLOR,
               linestyle="-", lw=1.8, marker="o", markersize=5,
               label=f"Adjoint ({n_iter} iters, ours)")
    ax_w2.plot(t_axis, w2_radam_by_iter[idx], color=RADAM_COLOR,
               linestyle="-", lw=1.8, marker="s", markersize=5,
               label=f"Adjoint + Riemannian ({n_iter} iters, ours)")
    ax_mmd.plot(t_axis, mmd_adam_by_iter[idx], color=ADAM_COLOR,
                linestyle="-", lw=1.8, marker="o", markersize=5)
    ax_mmd.plot(t_axis, mmd_radam_by_iter[idx], color=RADAM_COLOR,
                linestyle="-", lw=1.8, marker="s", markersize=5)

    if overlay_paper:
        # Eyeballed median values from Fig 2 of Neklyudov et al. 2023
        # (entropic Action Matching paper) on the petals dataset.
        # eAM is the BLUE/LOWER curves in the paper (their method,
        # better); MIOFlow is the ORANGE/UPPER curves (baseline).
        # x-axis is 6 points: t ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.
        #
        # Both panels overlaid; ABSOLUTE MMD numbers across
        # implementations carry a bandwidth-convention caveat (the
        # paper's RBF kernel σ is not necessarily ours), so the MMD
        # overlay is qualitative.
        #
        # Eyeballed at OUR timepoints (paper x-axis is continuous, we
        # sample at the same 5 t-values we report).
        t_paper = np.array(t_axis)                              # = our 5 timepoints
        w2_eam     = np.array([0.02, 0.06, 0.08, 0.10, 0.13])
        w2_mioflow = np.array([0.02, 0.16, 0.22, 0.27, 0.30])
        # MMD (from zoomed screenshot): entropic AM is essentially flat
        # at ~0.01–0.025 across all t; MIOFlow has three separate
        # lines (5/10/15 steps) spread roughly between 0.08 and 0.30,
        # with median around 0.13 → 0.17 over the trajectory.
        mmd_eam     = np.array([0.005, 0.020, 0.015, 0.013, 0.015])
        mmd_mioflow = np.array([0.005, 0.13,  0.14,  0.16,  0.17])

        ax_w2.plot(t_paper, w2_eam, color="#ff7f0e", linestyle="--",
                   lw=1.4, marker="^", markersize=5, markerfacecolor="white",
                   label="entropic AM (Neklyudov et al. 2023)")
        ax_w2.plot(t_paper, w2_mioflow, color="#17becf", linestyle="--",
                   lw=1.4, marker="v", markersize=5, markerfacecolor="white",
                   label="MIOFlow (Huguet et al. 2022)")
        ax_mmd.plot(t_paper, mmd_eam, color="#ff7f0e", linestyle="--",
                    lw=1.4, marker="^", markersize=5, markerfacecolor="white")
        ax_mmd.plot(t_paper, mmd_mioflow, color="#17becf", linestyle="--",
                    lw=1.4, marker="v", markersize=5, markerfacecolor="white")

    ax_w2.set_xlabel("time"); ax_w2.set_ylabel(r"$W_2$ distance")
    ax_mmd.set_xlabel("time"); ax_mmd.set_ylabel("MMD")
    ax_w2.set_xticks(t_axis); ax_mmd.set_xticks(t_axis)
    ax_w2.legend(loc="best", fontsize=8)
    ax_w2.grid(alpha=0.25); ax_mmd.grid(alpha=0.25)


def _grad_v_grid(V_grid: np.ndarray, N: int, d: int, L: float) -> np.ndarray:
    """Numerical gradient of V (shape (N^d,)) on the periodic grid.

    Returns shape ``(d, N^d)`` where component ``a`` is ∂V/∂x_a.
    """
    V = V_grid.reshape((N,) * d)
    dx = L / N
    grads = np.empty((d, *V.shape), dtype=np.float64)
    for a in range(d):
        grads[a] = (np.roll(V, -1, axis=a) - np.roll(V, 1, axis=a)) / (2 * dx)
    return grads.reshape(d, -1)


def _integrate_paths(V_grids, psi0_np, N, d, L, K_data,
                      n_paths=40, n_substeps=40, seed=0,
                      velocity_scale=1.0):
    """Integrate ẋ = velocity_scale · ∇V_t(x) from samples of |ψ_0|² through
    the K segments.

    The classical-limit velocity of the Trotter pipeline H_cons = i[K, V_t]
    is ∇V_t to lowest order, but the *quantum* density transport receives
    additional contribution from the K-substep momentum spread, so the
    plain classical trajectories typically undershoot the quantum pipeline's
    actual transport. ``velocity_scale`` is a phenomenological knob to
    visually match the data envelope; ``None`` activates automatic
    calibration that scales velocity so the median path radial displacement
    matches the median data radial displacement from t=0 to t=1.

    V_grids: list of K_data dense V grids ((N^d,) each), one per segment.
    Returns ``(K_data * n_substeps + 1, n_paths, d)`` array of positions in
    centred coordinates (will be shifted by L/2 by the caller for plotting).
    """
    rng = np.random.default_rng(seed)
    # Sample seed positions from |ψ_0|² via the histogram cells.
    probs = np.abs(psi0_np) ** 2
    probs = probs / probs.sum()
    cell_idx = rng.choice(probs.size, size=n_paths, p=probs)
    # Convert flat cell index to multi-index, add a jitter to land in the
    # cell, and shift to centred coordinates.
    mi = np.unravel_index(cell_idx, (N,) * d)
    pos = np.stack(mi, axis=1).astype(np.float64)              # (n_paths, d)
    pos += rng.random(pos.shape)                                 # cell jitter
    dx = L / N
    pos = pos * dx - L / 2.0                                     # centred

    # Pre-compute ∇V for each segment as gridded fields. Time-anchor
    # segment k's potential at its midpoint t_k_mid = (k + 0.5) / K_data,
    # so the velocity field varies smoothly in time via linear
    # interpolation between adjacent segment midpoints.
    grads = [_grad_v_grid(V_grids[k], N, d, L) for k in range(K_data)]
    t_mids = np.array([(k + 0.5) / K_data for k in range(K_data)])

    total_steps = K_data * n_substeps
    dt_step = 1.0 / total_steps
    traj = np.empty((total_steps + 1, n_paths, d), dtype=np.float64)
    traj[0] = pos
    for step in range(total_steps):
        t_now = (step + 0.5) / total_steps                       # cell-centre time
        # Find adjacent segment midpoints and blend the gradient.
        if t_now <= t_mids[0]:
            k_lo, k_hi, w = 0, 0, 0.0
        elif t_now >= t_mids[-1]:
            k_lo, k_hi, w = K_data - 1, K_data - 1, 0.0
        else:
            k_lo = int(np.searchsorted(t_mids, t_now) - 1)
            k_hi = k_lo + 1
            w = (t_now - t_mids[k_lo]) / (t_mids[k_hi] - t_mids[k_lo])

        # Sample the (time-blended) gradient at the current position.
        idx_each = []
        for a in range(d):
            ia = np.mod(np.round((pos[:, a] + L / 2.0) / dx).astype(np.int64),
                         N)
            idx_each.append(ia)
        flat = np.zeros(n_paths, dtype=np.int64)
        for a in range(d):
            flat = flat * N + idx_each[a]
        v_lo = np.stack([grads[k_lo][a][flat] for a in range(d)], axis=1)
        if k_hi == k_lo:
            v = v_lo
        else:
            v_hi = np.stack([grads[k_hi][a][flat] for a in range(d)], axis=1)
            v = (1.0 - w) * v_lo + w * v_hi
        pos = pos + dt_step * velocity_scale * v
        # Wrap to [-L/2, L/2)
        pos = ((pos + L / 2.0) % L) - L / 2.0
        traj[step + 1] = pos
    return traj


def _build_ot_chain_trajectories(V_grids, psi0_np, N, d, L, K_data,
                                   alpha, beta, D_max, D_V,
                                   n_paths=40, seed=0):
    """Build n_paths trajectories from |ψ_0|² to |ψ_K|² by:

    1. Running the forward MPS Trotter pipeline with the given V_grids to
       get cached MPS snapshots ψ_{t_k} for k = 0..K_data.
    2. Autoregressively sampling n_paths points from |ψ_{t_k}|² at each
       snapshot.
    3. OT-pairing samples between consecutive snapshots (Hungarian
       assignment on the n_paths × n_paths Euclidean cost matrix) so each
       path follows the same identity across snapshots.

    Returns ``(K_data + 1, n_paths, d)`` array in *world* coordinates
    (shifted by L/2).
    """
    from scipy.optimize import linear_sum_assignment

    cache = forward_with_cache_mps(
        psi0_np, V_grids, alpha=alpha, beta=beta, N=N, d=d, L=L,
        D_max=D_max, D_V=D_V,
    )
    rng = np.random.default_rng(seed)
    snaps = []
    for k in range(K_data + 1):
        mps_k = cache[k * 8]
        # _autoreg_sample already returns world coords in [0, L].
        s = _autoreg_sample(mps_k, n=n_paths, N=N, L=L, rng=rng)
        snaps.append(s)

    # Re-order snaps[k+1] so each path index follows OT-paired identity.
    paired = [snaps[0]]
    for k in range(K_data):
        prev = paired[-1]
        nxt = snaps[k + 1]
        cost = np.linalg.norm(prev[:, None, :] - nxt[None, :, :], axis=-1)
        _, col = linear_sum_assignment(cost)
        paired.append(nxt[col])
    return np.stack(paired, axis=0)                        # (K+1, n_paths, d)


def _calibrate_velocity_scale(V_grids, psi0_np, target_world, N, d, L,
                                K_data, n_paths=40, n_substeps=40,
                                seed=0):
    """Empirically find a velocity scale so that the median radial
    displacement of integrated trajectories at t=1 matches the median
    radial displacement of the data from t=0 to t=1.

    Uses a single bisection-like fit: integrate at scale=1, measure the
    ratio of median data radial displacement to median path radial
    displacement, and return that ratio.
    """
    # Sample paths at scale=1
    traj1 = _integrate_paths(V_grids, psi0_np, N, d, L, K_data,
                              n_paths=n_paths, n_substeps=n_substeps,
                              seed=seed, velocity_scale=1.0)
    start = traj1[0]                                            # (n_paths, d)
    end = traj1[-1]
    path_disp = np.linalg.norm(end - start, axis=1)             # (n_paths,)
    # Use 95th percentile: many seed positions near the centre have
    # near-zero V-gradient, so the median path barely moves and would
    # bias the scale too low. The longest paths are the ones flowing
    # toward the petal tips — calibrate against those.
    path_ref = float(np.percentile(path_disp, 95))

    # Reference data radial displacement from t=0 to t=1 (same percentile).
    centred_0 = target_world[0] - L / 2.0
    centred_K = target_world[-1] - L / 2.0
    r0 = np.linalg.norm(centred_0, axis=1)
    rK = np.linalg.norm(centred_K, axis=1)
    data_ref = float(np.percentile(rK, 95) - np.percentile(r0, 95))

    if path_ref <= 1e-8:
        return 1.0
    return data_ref / path_ref


def _plot_fig3(ax_left, ax_right, snaps_left, snaps_right,
                paths_left, paths_right,
                t_axis, L, w2_left, w2_right, title_left, title_right,
                proj_dims=(0, 1)):
    """Two scatter panels, one per method. Samples per snapshot, coloured
    by time; no overlay lines (model trajectories aren't well-defined for
    this quantum pipeline — see commit history for OT-paired and
    classical-limit experiments). For d > 2 (e.g. EB d=5) the panels
    project onto the two dims given by ``proj_dims`` (default the first
    two).
    """
    del paths_left, paths_right                     # unused, kept for API stability
    i0, i1 = proj_dims
    snaps_left  = [s[..., [i0, i1]] if s.shape[-1] > 2 else s for s in snaps_left]
    snaps_right = [s[..., [i0, i1]] if s.shape[-1] > 2 else s for s in snaps_right]
    cmap = plt.cm.viridis
    K_plus_1 = len(snaps_left)
    for k, samples in enumerate(snaps_left):
        c = cmap(k / max(K_plus_1 - 1, 1))
        ax_left.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.55,
                        color=c, edgecolors="none",
                        label=f"$t={t_axis[k]:.2f}$")
    for k, samples in enumerate(snaps_right):
        c = cmap(k / max(K_plus_1 - 1, 1))
        ax_right.scatter(samples[:, 0], samples[:, 1], s=8, alpha=0.55,
                         color=c, edgecolors="none")

    for ax in (ax_left, ax_right):
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)

    ax_left.set_title(f"{title_left}\n$W_2$@$t{{=}}1{{:}}{w2_left[-1]:.3f}$",
                      fontsize=10)
    ax_right.set_title(f"{title_right}\n$W_2$@$t{{=}}1{{:}}{w2_right[-1]:.3f}$",
                       fontsize=10)
    ax_left.legend(loc="upper right", fontsize=7, framealpha=0.85)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()),
                   default="petals_2d")
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--D_V_param", type=int, default=8)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=16)
    p.add_argument("--K_data", type=int, default=4)
    p.add_argument("--L", type=float, default=None,
                   help="Grid half-width override; default from dataset.")
    p.add_argument("--n_per_step", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr_adam", type=float, default=1e-2)
    p.add_argument("--lr_radam", type=float, default=1e-2)
    p.add_argument("--iter_budgets", type=str, default="200,500,2000")
    p.add_argument("--n_samples_eval", type=int, default=400)
    p.add_argument("--out_suffix", type=str, default="",
                   help="Appended to output filename, e.g. '_N32'.")
    p.add_argument("--from_cache", action="store_true",
                   help="If set, load cached V_grids/snapshots/W2/MMD from "
                        "results/figures/radam_compare_cache{suffix}.npz "
                        "and only re-render the figures.")
    args = p.parse_args()

    iter_budgets = [int(b) for b in args.iter_budgets.split(",")]
    iter_budgets.sort()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = DATASET_CONFIG[args.dataset]
    d = cfg["d"]
    L = args.L if args.L is not None else cfg["L"]
    N, K_data = args.N, args.K_data
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)
    D_max, D_V = args.D_max, args.D_V

    target_traj = cfg["sampler"](args.n_per_step, args.seed, K_data)
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)
    target_world = target_traj + L / 2.0
    t_axis = np.array([k / K_data for k in range(K_data + 1)])

    print(f"Comparing Adjoint vs Adjoint+Riemannian on petals_2d")
    print(f"  N={N} d={d} K={K_data}  D_max={D_max}  D_V_param={args.D_V_param}")
    print(f"  iter budgets: {iter_budgets}")
    print(f"  lr_adam={args.lr_adam}  lr_radam={args.lr_radam}")

    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"radam_compare_cache{args.out_suffix}.npz"

    w2_adam, mmd_adam, snaps_adam, losses_adam = [], [], [], []
    w2_radam, mmd_radam, snaps_radam, losses_radam = [], [], [], []
    V_grids_adam, V_grids_radam = [], []

    if args.from_cache and cache_path.exists():
        print(f"Loading cached training artefacts from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        w2_adam   = list(cache["w2_adam"])
        mmd_adam  = list(cache["mmd_adam"])
        snaps_adam = [[np.asarray(arr, dtype=np.float64) for arr in s]
                      for s in cache["snaps_adam"]]
        V_grids_adam = [[np.asarray(arr, dtype=np.float64) for arr in s]
                        for s in cache["V_grids_adam"]]
        w2_radam   = list(cache["w2_radam"])
        mmd_radam  = list(cache["mmd_radam"])
        snaps_radam = [[np.asarray(arr, dtype=np.float64) for arr in s]
                       for s in cache["snaps_radam"]]
        V_grids_radam = [[np.asarray(arr, dtype=np.float64) for arr in s]
                         for s in cache["V_grids_radam"]]
    else:
        for n_iter in iter_budgets:
            print(f"\n--- iter budget {n_iter} ---")
            t0 = time.time()
            V_a, L_a = _train_adam(K_data, N, d, args.D_V_param,
                                    psi0_np, q_grids_np, alpha, beta, L,
                                    D_max, D_V, n_iter, args.lr_adam, args.seed)
            wall_a = time.time() - t0
            s_a, w_a, m_a = _eval_per_timepoint(V_a, psi0_np, target_world,
                                                  N, d, L, K_data, alpha, beta,
                                                  D_max, D_V,
                                                  n_samples=args.n_samples_eval,
                                                  seed=args.seed)
            print(f"  Adjoint              loss={np.mean(L_a[-50:]):.4f}  "
                  f"W₂[t=1]={w_a[-1]:.4f}  ({wall_a:.1f}s)")

            t0 = time.time()
            V_r, L_r = _train_radam(K_data, N, d, args.D_V_param,
                                     psi0_np, q_grids_np, alpha, beta, L,
                                     D_max, D_V, n_iter, args.lr_radam, args.seed)
            wall_r = time.time() - t0
            s_r, w_r, m_r = _eval_per_timepoint(V_r, psi0_np, target_world,
                                                  N, d, L, K_data, alpha, beta,
                                                  D_max, D_V,
                                                  n_samples=args.n_samples_eval,
                                                  seed=args.seed)
            print(f"  Adjoint+Riemannian   loss={np.mean(L_r[-50:]):.4f}  "
                  f"W₂[t=1]={w_r[-1]:.4f}  ({wall_r:.1f}s)")

            w2_adam.append(w_a); mmd_adam.append(m_a); snaps_adam.append(s_a)
            losses_adam.append(L_a)
            w2_radam.append(w_r); mmd_radam.append(m_r); snaps_radam.append(s_r)
            losses_radam.append(L_r)
            V_grids_adam.append([V_mps_to_dense(V_a[k], N, d)
                                  for k in range(K_data)])
            V_grids_radam.append([V_mps_to_dense(V_r[k], N, d)
                                   for k in range(K_data)])

        # Cache for cheap re-rendering
        np.savez(
            cache_path,
            iter_budgets=np.array(iter_budgets),
            w2_adam=np.array(w2_adam),
            mmd_adam=np.array(mmd_adam),
            w2_radam=np.array(w2_radam),
            mmd_radam=np.array(mmd_radam),
            snaps_adam=np.array(snaps_adam, dtype=object),
            snaps_radam=np.array(snaps_radam, dtype=object),
            V_grids_adam=np.array(V_grids_adam, dtype=object),
            V_grids_radam=np.array(V_grids_radam, dtype=object),
        )
        print(f"\nwrote training-artefact cache to {cache_path}")

    # Pick the best budget (largest n_iter) for Fig 3 trajectories
    best_idx = -1
    snaps_a_best = snaps_adam[best_idx]
    snaps_r_best = snaps_radam[best_idx]
    w_a_best = w2_adam[best_idx]; w_r_best = w2_radam[best_idx]
    # V_grids for path integration (already collected during training)
    V_grids_a_best = V_grids_adam[best_idx]
    V_grids_r_best = V_grids_radam[best_idx]
    # Build "trajectory" polylines by OT-pairing samples across pipeline
    # snapshots (more honest than classical-limit integration for this
    # quantum pipeline — the per-segment V_t doesn't form a smooth
    # classical flow field, but the sequence of |ψ_{t_k}|² snapshots is
    # exactly what the pipeline produces).
    paths_a_world = _build_ot_chain_trajectories(
        V_grids_a_best, psi0_np, N, d, L, K_data,
        alpha, beta, D_max, D_V,
        n_paths=20, seed=args.seed)
    paths_r_world = _build_ot_chain_trajectories(
        V_grids_r_best, psi0_np, N, d, L, K_data,
        alpha, beta, D_max, D_V,
        n_paths=20, seed=args.seed)

    # --- Figure 2 (W₂ + MMD per timepoint, multiple budgets) -------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_fig2(axes2[0], axes2[1], t_axis,
                w2_adam, mmd_adam, w2_radam, mmd_radam, iter_budgets)
    fig2.suptitle(f"$W_2$ and MMD vs time on petals_2d "
                  f"(N={N}, $D_V$={args.D_V_param}, "
                  f"lr=$10^{{-3}}$, 2000 iters)  "
                  f"— ours vs eAM/MIOFlow (Neklyudov et al. 2023)",
                  fontsize=10)
    fig2.tight_layout()
    out_dir = Path("results/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    p2_pdf = out_dir / f"radam_compare_fig2{args.out_suffix}.pdf"
    p2_png = out_dir / f"radam_compare_fig2{args.out_suffix}.png"
    fig2.savefig(p2_pdf, bbox_inches="tight")
    fig2.savefig(p2_png, bbox_inches="tight", dpi=150)
    print(f"\nwrote {p2_pdf}")
    print(f"wrote {p2_png}")

    # --- Figure 3 (trajectory scatter + integrated model paths) ----------
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 5.4))
    _plot_fig3(axes3[0], axes3[1],
                snaps_a_best, snaps_r_best,
                paths_a_world, paths_r_world,
                t_axis, L,
                w_a_best, w_r_best,
                title_left=f"Adjoint  (Euclidean Adam, {iter_budgets[best_idx]} iters)",
                title_right=f"Adjoint + Riemannian  ({iter_budgets[best_idx]} iters)")
    fig3.suptitle("Generated petal samples — same MPS-adjoint pipeline, "
                  "Euclidean Adam vs Riemannian Adam on $V_{\\mathrm{mps}}$",
                  fontsize=10, y=1.00)
    fig3.tight_layout()
    p3_pdf = out_dir / f"radam_compare_fig3{args.out_suffix}.pdf"
    p3_png = out_dir / f"radam_compare_fig3{args.out_suffix}.png"
    fig3.savefig(p3_pdf, bbox_inches="tight")
    fig3.savefig(p3_png, bbox_inches="tight", dpi=150)
    print(f"wrote {p3_pdf}")
    print(f"wrote {p3_png}")


if __name__ == "__main__":
    main()
