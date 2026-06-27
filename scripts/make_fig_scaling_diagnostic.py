"""TDVP1 walltime scaling diagnostic.

Three diagnostic plots at fixed (N=32, K=64), varying D:

  (A) Semi-log fit:  log(t) vs d, linear fit. Residuals U-shape ⇒ NOT exponential.
  (B) Log-log fit:   log(t) vs log(d), linear fit. Residuals U-shape ⇒ NOT a single
                      monomial; polynomial with lower-order terms.
  (C) Rectification: t / d^p for p ∈ {2, 3, 4, 5}. Flat curve identifies poly degree.

Usage:
    uv run python scripts/make_fig_scaling_diagnostic.py \\
        --out results/scaling_diagnostic.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [
    (2, "data/gmm_2d_hp"),
    (3, "data/gmm_3d_hp_v2"),
    (4, "data/gmm_4d_hp"),
    (5, "data/gmm_5d_hp"),
    (6, "data/gmm_6d_hp"),
    (7, "data/gmm_7d_hp"),
    (8, "data/gmm_8d_hp"),
]

D_COLORS = {8: "tab:blue", 16: "tab:green", 32: "tab:red", 64: "tab:purple"}
P_COLORS = {2: "tab:cyan", 3: "tab:orange", 4: "tab:green",
            5: "tab:red", 6: "tab:purple"}


def load_walltime(method: str, N_fixed: int, K_fixed: int) -> dict[int, list[tuple[int, float]]]:
    """Returns {D_max: [(d, walltime), ...]} for TDVP runs at fixed (N, K)."""
    out: dict[int, list[tuple[int, float]]] = {}
    for d, dir_ in DATASETS:
        base = Path(dir_) / method
        if not base.exists():
            continue
        for sub in base.iterdir():
            name = sub.name
            if not name.startswith(f"N{N_fixed}_K{K_fixed}_D"):
                continue
            try:
                D = int(name.split("_D")[-1])
            except ValueError:
                continue
            seeds_t = []
            for f in sub.glob("seed*.npz"):
                z = np.load(f, allow_pickle=True)
                if "total_time" in z.files:
                    t = float(z["total_time"])
                    if t > 0:
                        seeds_t.append(t)
            if seeds_t:
                out.setdefault(D, []).append((d, float(np.mean(seeds_t))))
    return {D: sorted(pts) for D, pts in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scaling_diagnostic.pdf")
    p.add_argument("--method", default="tci_tdvp1",
                   choices=["tci_tdvp1", "tci_tdvp2"])
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--K", type=int, default=64)
    args = p.parse_args()

    data = load_walltime(args.method, args.N, args.K)
    print(f"Fixed (N={args.N}, K={args.K}) {args.method} walltime:")
    for D, pts in sorted(data.items()):
        ds = ", ".join(f"d={d}: {t:.0f}s" for d, t in pts)
        print(f"  D={D:>3}: {ds}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300,
                              constrained_layout=True)
    ax_semi, ax_loglog, ax_rect = axes

    # ── Panel A: Semi-log (linear x, log y).  Exponential ⇒ straight line ──
    for D, pts in sorted(data.items()):
        if len(pts) < 2:
            continue
        d = np.array([p[0] for p in pts])
        t = np.array([p[1] for p in pts])
        ax_semi.semilogy(d, t, "o-", color=D_COLORS.get(D, "gray"),
                         markersize=11, lw=1.5,
                         markerfacecolor=D_COLORS.get(D, "gray"),
                         markeredgecolor="black", markeredgewidth=0.5,
                         label=f"D={D}")
        # Linear fit on log(t) vs d.  Slope = log(B) for t = A · B^d.
        slope, intercept = np.polyfit(d, np.log(t), 1)
        d_fit = np.linspace(d.min(), d.max(), 50)
        ax_semi.plot(d_fit, np.exp(slope * d_fit + intercept),
                     "--", color=D_COLORS.get(D, "gray"), alpha=0.5, lw=1.0)
        # Annotate residual sign pattern (U-shape ⇒ not exponential).
        residuals = np.log(t) - (slope * d + intercept)
        sign_flip = (residuals[0] > 0) and (residuals[-1] > 0) and (residuals[len(residuals)//2] < 0)
        suffix = " (U-shape ⇒ ¬exp)" if sign_flip else ""
        print(f"  D={D} semi-log fit:  t ≈ {np.exp(intercept):.2g} · {np.exp(slope):.2f}^d"
              f"   residual std = {residuals.std():.3f}{suffix}")
    ax_semi.set_xlabel("d  (linear)", fontsize=12)
    ax_semi.set_ylabel("walltime (s, log)", fontsize=12)
    ax_semi.set_title("(A)  Semi-log:   exponential ⇒ straight line", fontsize=11)
    ax_semi.legend(fontsize=10, loc="best")
    ax_semi.grid(True, which="both", alpha=0.3)

    # ── Panel B: Log-log.  Single-monomial power law ⇒ straight line ───────
    for D, pts in sorted(data.items()):
        if len(pts) < 2:
            continue
        d = np.array([p[0] for p in pts], dtype=float)
        t = np.array([p[1] for p in pts])
        ax_loglog.loglog(d, t, "o-", color=D_COLORS.get(D, "gray"),
                         markersize=11, lw=1.5,
                         markerfacecolor=D_COLORS.get(D, "gray"),
                         markeredgecolor="black", markeredgewidth=0.5,
                         label=f"D={D}")
        slope, intercept = np.polyfit(np.log(d), np.log(t), 1)
        d_fit = np.linspace(d.min(), d.max(), 50)
        ax_loglog.plot(d_fit, np.exp(intercept) * d_fit ** slope,
                       "--", color=D_COLORS.get(D, "gray"), alpha=0.5, lw=1.0)
        residuals = np.log(t) - (slope * np.log(d) + intercept)
        sign_flip = (residuals[0] > 0) and (residuals[-1] > 0) and (residuals[len(residuals)//2] < 0)
        suffix = " (U-shape ⇒ ¬monomial)" if sign_flip else ""
        print(f"  D={D} log-log fit:  t ≈ {np.exp(intercept):.2g} · d^{slope:+.2f}"
              f"   residual std = {residuals.std():.3f}{suffix}")
    ax_loglog.set_xlabel("d  (log)", fontsize=12)
    ax_loglog.set_ylabel("walltime (s, log)", fontsize=12)
    ax_loglog.set_title("(B)  Log-log:   single monomial t ∝ d^p ⇒ straight line", fontsize=11)
    ax_loglog.legend(fontsize=10, loc="best")
    ax_loglog.grid(True, which="both", alpha=0.3)

    # ── Panel C: Rectification.  t / d^p for several p; flat ⇒ that's the degree ──
    # Use the D-bucket with the most data points (4 points needed to see flatness).
    best_D = max(data, key=lambda D: len(data[D]))
    pts = data[best_D]
    d = np.array([p[0] for p in pts], dtype=float)
    t = np.array([p[1] for p in pts])
    candidate_p = [2, 3, 4, 5, 6]
    for p_val in candidate_p:
        y = t / d ** p_val
        ax_rect.semilogy(d, y, "o-", color=P_COLORS.get(p_val, "gray"),
                         markersize=11, lw=1.5,
                         markerfacecolor=P_COLORS.get(p_val, "gray"),
                         markeredgecolor="black", markeredgewidth=0.5,
                         label=f"t / d^{p_val}")
    ax_rect.set_xlabel("d", fontsize=12)
    ax_rect.set_ylabel("walltime / d^p  (log)", fontsize=12)
    ax_rect.set_title(f"(C)  Rectification (D={best_D}):   correct p ⇒ flat curve",
                      fontsize=11)
    ax_rect.legend(fontsize=10, loc="best")
    ax_rect.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Walltime scaling diagnostic — {args.method.upper()}  "
        f"(N={args.N}, K={args.K})",
        fontsize=13, fontweight="bold",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
