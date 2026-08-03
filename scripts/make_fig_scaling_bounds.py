"""Emit paper Fig 4: error scaling of the dense wavefunction flow vs the bounds.

Three panels:
  (a) grid/representation error vs N   -- Laplace kink, observed N^-3/2
  (b) Trotter error vs K               -- observed K^-1
  (c) Trotter constant vs d            -- with the d^2 upper bound

Two error measures, defined in the figure caption:
  eps_grid = ||(1-P_N) psi_T|| / ||psi_T||   relative L2 truncation of the target
  eps_time = ||psi_K - psi_inf||             phase-aligned distance to the K->inf flow

NOTE ON SCOPE. This is the *plotting* half of the analytic scaling study. It reads
a precomputed CSV cache (`data/scaling_theory/panel{A_state,B_repr,D_state}.csv`)
that ships with the data archive. The *compute* half -- which integrates the dense
flow at high K to produce those CSVs -- depends on internal solver code that is not
part of this release, so the cache is shipped rather than regenerated. Everything
here is pure csv/numpy/matplotlib, so the figure is fully reproducible from the
cache; the numbers in it are not re-derived.

Usage:
    make fig-scaling-bounds
    uv run python scripts/make_fig_scaling_bounds.py [--cache DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def fit_pow(x, y):
    """Least-squares power-law fit in log-log; returns (slope, intercept, R^2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (y > 0) & np.isfinite(y)
    sl, ic = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    pred = sl * np.log(x[m]) + ic
    r2 = 1 - np.sum((np.log(y[m]) - pred) ** 2) / np.sum(
        (np.log(y[m]) - np.log(y[m]).mean()) ** 2)
    return sl, ic, r2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/scaling_theory",
                   help="directory holding panel{A_state,B_repr,D_state}.csv")
    p.add_argument("--out", default="figures/fig_scaling_bounds.pdf")
    args = p.parse_args()

    cache = Path(args.cache)
    need = ["panelA_state.csv", "panelB_repr.csv", "panelD_state.csv"]
    missing = [n for n in need if not (cache / n).exists()]
    if missing:
        print(f"error: missing {', '.join(missing)} under {cache}\n"
              "       the scaling_theory CSV cache ships with the data archive")
        return 1

    def load(name):
        with (cache / f"{name}.csv").open() as f:
            return list(csv.DictReader(f))

    C0, C1, C3 = plt.cm.magma(0.18), plt.cm.magma(0.40), plt.cm.magma(0.76)
    EGRID = r"$\varepsilon_{\mathrm{grid}}$"
    ETIME = r"$\varepsilon_{\mathrm{time}}$"
    rc = {"font.size": 17, "axes.labelsize": 23, "xtick.labelsize": 16,
          "ytick.labelsize": 16, "legend.fontsize": 16}

    with plt.rc_context(rc):
        fig, (axR, axK, axD) = plt.subplots(1, 3, figsize=(17.5, 5.2))
        fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.16, wspace=0.30)

        # (a) grid / representation error vs N (Laplace sqrt(p)=e^{-|x|}: kink -> N^{-3/2})
        rB = load("panelB_repr")
        Nr = np.array([float(r["N"]) for r in rB])
        yf = np.array([float(r["laplace"]) for r in rB])
        m = yf > 1e-8
        nn, yy = Nr[m], yf[m]
        slf, icf = np.polyfit(np.log(nn), np.log(yy), 1)
        nnf = np.logspace(np.log10(nn[0]), np.log10(nn[-1]), 60)
        axR.loglog(nn, yy, "o", color=C1, ms=9, mec="0.3", zorder=5, label="observed")
        axR.loglog(nnf, np.exp(icf) * nnf ** slf, "--", color=C1, lw=2.3,
                   label=f"fit $N^{{{slf:.2f}}}$")
        axR.set(xlabel="grid resolution $N$", ylabel=EGRID)
        axR.legend(loc="lower left")
        axR.grid(alpha=0.3, which="both")

        # (b) Trotter error vs K
        rA = load("panelA_state")
        K = np.array([float(r["K"]) for r in rA])
        yA = np.array([float(r["sdist"]) for r in rA])
        sl, ic, _ = fit_pow(K, yA)
        kk = np.logspace(np.log10(K[0]), np.log10(K[-1]), 80)
        axK.loglog(K, yA, "o", color=C0, ms=9, mec="0.3", zorder=5, label="observed")
        axK.loglog(kk, np.exp(ic) * kk ** sl, "--", color=C0, lw=2.3,
                   label=f"fit $K^{{{sl:.2f}}}$")
        axK.set(xlabel="Trotter steps $K$", ylabel=ETIME)
        axK.legend(loc="lower left")
        axK.grid(alpha=0.3, which="both")

        # (c) Trotter constant vs d, with the d^2 upper bound
        rD = load("panelD_state")
        dv = np.array([float(r["d"]) for r in rD])
        yD = np.array([float(r["sdist"]) for r in rD])
        dd = np.linspace(dv[0], dv[-1], 60)
        axD.loglog(dv, yD, "o", color=C3, ms=10, mec="0.3", zorder=5, label="observed")
        axD.loglog(dd, yD[0] * (dd / dv[0]) ** 2.0, "-", color="0.25", lw=2.1,
                   label=r"upper bound $\propto d^{2}$")
        axD.set(xlabel="dimensionality $d$", ylabel=ETIME)
        axD.legend(loc="upper left")
        axD.grid(alpha=0.3, which="both")
        axD.set_xticks([2, 3, 4, 5])
        axD.set_xticklabels(["2", "3", "4", "5"])
        axD.set_yticks([0.02, 0.03, 0.04, 0.06])
        axD.set_yticklabels(["0.02", "0.03", "0.04", "0.06"])
        axD.minorticks_off()

        for ax, lab in ((axR, "(a)"), (axK, "(b)"), (axD, "(c)")):
            ax.text(0.02, 1.02, lab, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=21, fontweight="bold")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")

    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
