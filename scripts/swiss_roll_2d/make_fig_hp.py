"""GP-interpolate the Sobol-sampled (N, K, D_max) sweep + validate vs grid anchors.

Pipeline:
  1. Load Sobol pilot from results/swiss_roll_hp_pilot/.
  2. Fit a Gaussian Process on (log2 N, log2 K, log2 D_max) → log SW.
     Predicted on the same axes; uncertainty quantified via GP posterior.
  3. Validate: for each (N, K) where the existing N×K grid has D=32 data,
     compare GP_pred(N, K, 32) vs measured SW. Compute residuals,
     coverage of GP 95% intervals, RMSE.
  4. Render: 2D heatmap slices at fixed D_max ∈ {16, 32, 64} with std overlay,
     side-by-side residual plot at the validation anchors.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

NK_D_PATTERN = re.compile(r"^N(\d+)_K(\d+)_D(\d+)$")
NK_PATTERN = re.compile(r"^N(\d+)_K(\d+)$")


def _safe_load(path: Path) -> dict | None:
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        return {"sw": float(z["sw"][-1]), "mmd": float(z["mmd"][-1])}
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        print(f"[skip {path}] {type(e).__name__}: {e}")
        return None


def collect_pilot(results_dir: Path, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_input, y_sw): X has rows (log2 N, log2 K, log2 D), y is SW seed-mean."""
    method_dir = results_dir / method
    rows = []
    for sub in method_dir.iterdir():
        m = NK_D_PATTERN.match(sub.name)
        if not m:
            continue
        N, K, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sw_seeds = []
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is not None:
                sw_seeds.append(rec["sw"])
        if sw_seeds:
            rows.append((np.log2(N), np.log2(K), np.log2(D), np.mean(sw_seeds), len(sw_seeds)))
    if not rows:
        raise SystemExit(f"No pilot data at {method_dir}/N*_K*_D*/seed*.npz")
    arr = np.array(rows, dtype=float)
    return arr[:, :3], arr[:, 3]


def collect_anchors(grid_dir: Path, method: str, D_anchor: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Anchor measurements at D=D_anchor from the existing N×K grid sweep."""
    method_dir = grid_dir / method
    if not method_dir.exists():
        return np.empty((0, 3)), np.empty(0)
    rows = []
    for sub in method_dir.iterdir():
        m = NK_PATTERN.match(sub.name)
        if not m:
            continue
        N, K = int(m.group(1)), int(m.group(2))
        sw_seeds = []
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is not None:
                sw_seeds.append(rec["sw"])
        if sw_seeds:
            rows.append((np.log2(N), np.log2(K), np.log2(D_anchor),
                         np.mean(sw_seeds), len(sw_seeds)))
    if not rows:
        return np.empty((0, 3)), np.empty(0)
    arr = np.array(rows, dtype=float)
    return arr[:, :3], arr[:, 3]


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    log_y = np.log(np.maximum(y, 1e-6))
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0, 1.0, 1.0],
              length_scale_bounds=(1e-1, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=5, random_state=0)
    gp.fit(X, log_y)
    return gp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_dir", default="data/swiss_roll_hp_pilot")
    p.add_argument("--grid_dir", default="data/swiss_roll_NK")
    p.add_argument("--method", default="tci_tdvp2")
    p.add_argument("--out", default="results/swiss_roll_hp_pilot/fig_hp_pilot.pdf")
    args = p.parse_args()

    print(f"loading pilot for {args.method}...")
    X, y = collect_pilot(Path(args.pilot_dir), args.method)
    print(f"  {len(y)} pilot cells: log2 N range "
          f"[{X[:, 0].min():.1f}, {X[:, 0].max():.1f}], "
          f"K [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}], "
          f"D [{X[:, 2].min():.1f}, {X[:, 2].max():.1f}]")

    print("fitting GP...")
    gp = fit_gp(X, y)
    print(f"  kernel: {gp.kernel_}")

    # Validate against existing N×K grid (which is at D=32)
    Xa, ya = collect_anchors(Path(args.grid_dir), args.method, D_anchor=32)
    if len(ya) == 0:
        print("[warn] no anchor data found — skipping validation panel")
        rmse = None
        cov95 = None
        residuals = None
    else:
        log_pred, log_std = gp.predict(Xa, return_std=True)
        pred = np.exp(log_pred)
        # 95% CI in log-space
        log_lo = log_pred - 1.96 * log_std
        log_hi = log_pred + 1.96 * log_std
        ci_lo = np.exp(log_lo)
        ci_hi = np.exp(log_hi)
        residuals = ya - pred
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        cov95 = float(np.mean((ya >= ci_lo) & (ya <= ci_hi)))
        print(f"validation @ D=32 anchors: RMSE = {rmse:.4f}  "
              f"95% CI coverage = {cov95:.0%}  (n_anchors = {len(ya)})")

    # ── Render: heatmap slices at D ∈ {16, 32, 64} + validation scatter ─────
    Ns = sorted({2 ** int(round(v)) for v in X[:, 0]})
    Ks = sorted({2 ** int(round(v)) for v in X[:, 1]})
    D_slices = [16, 32, 64]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=200)
    for ax, D in zip(axes[:3], D_slices):
        # Build dense (N, K) grid for prediction
        log_N = np.log2(np.array(Ns))
        log_K = np.log2(np.array(Ks))
        NN, KK = np.meshgrid(log_N, log_K, indexing="xy")
        pts = np.stack([NN.ravel(), KK.ravel(),
                        np.full(NN.size, np.log2(D))], axis=1)
        log_mu, log_std = gp.predict(pts, return_std=True)
        mu = np.exp(log_mu).reshape(NN.shape)
        std = log_std.reshape(NN.shape)
        im = ax.imshow(mu, origin="lower", cmap="plasma_r", aspect="auto")
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                ax.text(j, i, f"{mu[i, j]:.3f}\n±{std[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mu[i, j] > np.median(mu) else "black")
        ax.set_xticks(np.arange(len(Ns)))
        ax.set_xticklabels([f"N={n}" for n in Ns])
        ax.set_yticks(np.arange(len(Ks)))
        ax.set_yticklabels([f"K={k}" for k in Ks])
        ax.set_title(f"D_max = {D}\nGP prediction (mean ± log-σ)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.04)

    # Validation scatter
    ax_v = axes[3]
    if residuals is not None:
        log_pred, log_std = gp.predict(Xa, return_std=True)
        pred = np.exp(log_pred)
        ci_lo = np.exp(log_pred - 1.96 * log_std)
        ci_hi = np.exp(log_pred + 1.96 * log_std)
        ax_v.errorbar(pred, ya, yerr=[pred - ci_lo, ci_hi - pred],
                       fmt="o", ms=8, capsize=3, alpha=0.8, color="tab:blue")
        lim = [min(pred.min(), ya.min()) * 0.8,
               max(pred.max(), ya.max()) * 1.2]
        ax_v.plot(lim, lim, "k--", lw=1.0, alpha=0.6, label="y = x")
        ax_v.set_xlim(lim); ax_v.set_ylim(lim)
        ax_v.set_xlabel("GP-predicted SW (95% CI)", fontsize=10)
        ax_v.set_ylabel("Measured SW (grid anchors @ D=32)", fontsize=10)
        ax_v.set_xscale("log"); ax_v.set_yscale("log")
        ax_v.set_title(
            f"Validation: GP vs measured @ D=32\n"
            f"RMSE = {rmse:.4f}, 95% CI coverage = {cov95:.0%}, n = {len(ya)}",
            fontsize=10,
        )
        ax_v.grid(True, which="both", alpha=0.3)
        ax_v.legend(fontsize=9)
    else:
        ax_v.axis("off")

    fig.suptitle(
        f"Swiss roll GP pilot — {args.method.upper()} (50 Sobol points × 5 seeds)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
