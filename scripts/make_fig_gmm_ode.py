"""GMM marginal ODE visualization in the flow-matching style.

Two panels (1 row x 2 cols):
  (A) Samples from the marginal ODE at four times t in {0, 0.33, 0.67, 1.0}.
      Source samples are drawn from N(0, sigma_0^2 I) and integrated under
      x_dot = v_t(x) = grad V_t(x) for the analytic 2D orthogonal-mode GMM
      potential of theory.make_analytic_V_fn.
  (B) Trajectories of the same samples from t=0 to t=1, drawn as gray
      lines overlaid on a faded target-density background.

Usage:
    uv run python scripts/make_fig_gmm_ode.py \\
        --out figures/fig_gmm_ode.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from tnwf.data.gaussian_mixture import gm_mode_centers
from tnwf.jam.train import DATASET_DEFAULTS


def gmm_velocity(x, t, *, centers, component_var, sigma_0,
                  weights=None):
    """v_t(x) = x/t + (1-t) sigma_0^2 / t * grad log p_t(x), with mu_0 = 0
    in the centred frame.

    Closed-form score for an isotropic Gaussian mixture:
        p_t(x) = (1/K) sum_k N(x; t c_k, sigma_t^2 I).
    """
    n, d = x.shape
    K = centers.shape[0]
    if weights is None:
        weights = np.full(K, 1.0 / K)
    var_t = (1.0 - t) ** 2 * sigma_0 ** 2 + t ** 2 * component_var
    diffs = x[:, None, :] - t * centers[None, :, :]      # (n, K, d)
    sq = (diffs ** 2).sum(axis=-1)                        # (n, K)
    log_terms = (np.log(weights)[None, :]
                 - 0.5 * sq / var_t
                 - 0.5 * d * np.log(2 * np.pi * var_t))
    log_resp = log_terms - logsumexp(log_terms, axis=-1, keepdims=True)
    resp = np.exp(log_resp)
    # grad log p_t(x) = - sum_k resp_k (x - t c_k) / sigma_t^2
    score = -(resp[..., None] * diffs).sum(axis=1) / var_t
    return x / t + (1.0 - t) * sigma_0 ** 2 / t * score


def integrate_rk4(x0, *, n_steps, centers, component_var, sigma_0,
                   t_eps=1e-3):
    n, d = x0.shape
    ts = np.linspace(t_eps, 1.0, n_steps)
    dt = ts[1] - ts[0]
    xs = np.zeros((n_steps, n, d))
    xs[0] = x0
    for i in range(n_steps - 1):
        t = ts[i]
        x = xs[i]
        kw = dict(centers=centers, component_var=component_var,
                  sigma_0=sigma_0)
        k1 = gmm_velocity(x, t, **kw)
        k2 = gmm_velocity(x + 0.5 * dt * k1, t + 0.5 * dt, **kw)
        k3 = gmm_velocity(x + 0.5 * dt * k2, t + 0.5 * dt, **kw)
        k4 = gmm_velocity(x + dt * k3, t + dt, **kw)
        xs[i + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return ts, xs


def target_density(grid_x, grid_y, *, centers, component_var):
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    var = component_var
    diffs = pts[:, None, :] - centers[None, :, :]
    sq = (diffs ** 2).sum(axis=-1)
    log_p = (-0.5 * sq / var
             - np.log(centers.shape[0])
             - np.log(2 * np.pi * var))
    return np.exp(logsumexp(log_p, axis=-1)).reshape(grid_x.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig_gmm_ode.pdf")
    p.add_argument("--n_samples", type=int, default=800)
    p.add_argument("--n_traj", type=int, default=60)
    p.add_argument("--n_steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = DATASET_DEFAULTS["gmm_2d"]
    d, scale, std = cfg["d"], cfg["scale"], cfg["std"]
    sigma_0 = 1.0
    centers = gm_mode_centers(d=d, scale=scale, arrangement="orthogonal")
    component_var = float(std) ** 2

    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal(size=(args.n_samples, d)) * sigma_0

    print("Integrating ODE...")
    ts, xs = integrate_rk4(x0, n_steps=args.n_steps,
                            centers=centers,
                            component_var=component_var, sigma_0=sigma_0)

    snap_req = [0.0, 0.33, 0.67, 1.0]
    snap_idx = [int(np.argmin(np.abs(ts - tr))) for tr in snap_req]
    snap_t = ts[snap_idx]

    xy_max = max(float(np.abs(centers).max()) * 1.5, 4.0)
    g = np.linspace(-xy_max, xy_max, 240)
    xx, yy = np.meshgrid(g, g)
    p_target = target_density(xx, yy,
                               centers=centers, component_var=component_var)
    p_target /= p_target.max() if p_target.max() > 0 else 1.0

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300,
                                    constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.10)

    # Okabe-Ito colourblind-safe palette, ordered cool->warm by time.
    palette = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]

    for ax in (axA, axB):
        ax.imshow(p_target,
                  extent=(-xy_max, xy_max, -xy_max, xy_max),
                  origin="lower", cmap="Blues",
                  alpha=0.22, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-xy_max, xy_max); ax.set_ylim(-xy_max, xy_max)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_color("0.7"); spine.set_linewidth(0.6)

    # ── Panel A: sample scatter at four time points ──────────────────────
    for k, (idx, t_show) in enumerate(zip(snap_idx, snap_t)):
        x_k = xs[idx]
        axA.scatter(x_k[:, 0], x_k[:, 1], s=20, alpha=0.55,
                    color=palette[k], edgecolors="none",
                    label=f"t={t_show:.2f}", zorder=2 + k)
    axA.set_title("Samples from Marginal ODE", fontsize=17)
    leg = axA.legend(loc="upper right", fontsize=15, framealpha=0.95,
                      handlelength=0.9, borderpad=0.45,
                      labelspacing=0.45)
    leg.get_frame().set_edgecolor("0.7")
    for h in leg.legend_handles:
        h.set_sizes([60])

    # ── Panel B: trajectories ─────────────────────────────────────────────
    sel = rng.choice(args.n_samples,
                     size=min(args.n_traj, args.n_samples), replace=False)
    for i in sel:
        path = xs[:, i, :]
        axB.plot(path[:, 0], path[:, 1], color="0.25", lw=0.7, alpha=0.55,
                 zorder=2)
    axB.scatter(xs[-1, sel, 0], xs[-1, sel, 1], s=26, alpha=0.75,
                color=palette[-1], edgecolors="none", zorder=3)
    axB.set_title("Trajectories of Marginal ODE", fontsize=17)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
