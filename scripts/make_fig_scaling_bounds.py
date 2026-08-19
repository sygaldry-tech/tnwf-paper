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
# Computer Modern for math, as in the other figure generators. The default
# "dejavusans" fontset renders every symbol here -- N, K, d, varepsilon and the
# exponents inside the legend entries -- in sans, against a serif caption.
matplotlib.rcParams["mathtext.fontset"] = "cm"

# Figure width in inches, and the width it is rendered at in the paper
# (\textwidth = 452.97 pt = 6.27 in). Keep _FIG_W_IN in step with figsize below.
_FIG_W_IN = 17.5
_RENDERED_W_IN = 6.27


def _pt(rendered: float) -> float:
    """The matplotlib fontsize that renders at `rendered` points in the paper.

    Approximate on the safe side: savefig uses bbox_inches="tight", so the saved
    page is a little WIDER than figsize once the labels grow, and the true
    downscale is a little smaller than figsize implies (0.354 measured against
    0.358 predicted). Base targets are therefore quoted at 8.5 pt against an
    8 pt caption, so the floor still holds after that shrink.
    """
    return round(rendered * _FIG_W_IN / _RENDERED_W_IN, 1)

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
    # Sizes quoted at the size they RENDER in the paper. figsize is 17.5 in and
    # sn-article.tex includes this at \textwidth (452.97 pt = 6.27 in), a
    # downscale of 0.358, so hand-picked values landed far below the 8 pt caption:
    # ticks and legend at 16 rendered at 5.7 pt, titles at 20 at 7.2 pt, the panel
    # letters at 21 at 7.5 pt. Only axes.labelsize was already at parity.
    #
    # Math sub/superscripts sit at ~0.7x their base by design -- the legend's
    # "fit $N^{-1.47}$" exponent was the 4.0 pt run in the audit -- and the same
    # is true of subscripts in the caption text, so the floor is applied to the
    # base sizes rather than inflating everything to keep exponents above 8 pt.
    rc = {"font.size": _pt(8.5), "axes.labelsize": _pt(9.0),
          "xtick.labelsize": _pt(8.5), "ytick.labelsize": _pt(8.5),
          "legend.fontsize": _pt(8.5),
          # One step below axes.labelsize, as FS_TITLE/FS_GLOSS are in
          # make_fig_cost_scaling.py and fig_ksweep.py.
          "axes.titlesize": _pt(8.5)}

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
        axR.set(xlabel="$N$", ylabel=EGRID)
        axR.set_title("Grid Resolution", color="0.25", pad=10)
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
        # Widen the limits so the decade ticks fall inside the axis rather than at
        # its very edges: the data spans ~1.7 decades and matplotlib was labelling
        # only 10^-2, which read as a broken axis.
        axK.set_ylim(yA.min() / 2.2, yA.max() * 2.2)
        axK.set(xlabel="$K$", ylabel=ETIME)
        axK.set_title("Trotter Error", color="0.25", pad=10)
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
        axD.set(xlabel="$d$", ylabel=ETIME)
        axD.set_title("Dimensionality", color="0.25", pad=10)
        # This panel has no free corner: the data and its d^2 bound both run the
        # full diagonal, so at caption-parity fonts the legend covered the bound
        # line from either upper-left or lower-right. Headroom above the data gives
        # it a clear band instead, and the clearance is measured below rather than
        # assumed -- the same failure mode as Fig. 8's legend.
        legD = axD.legend(loc="upper left")
        axD.grid(alpha=0.3, which="both")
        axD.set_xticks([2, 3, 4, 5])
        axD.set_xticklabels(["2", "3", "4", "5"])
        # 0.01 added and the floor dropped below the smallest point: the lowest
        # datum (~0.011) used to sit under the lowest tick, leaving the bottom of
        # the axis unlabelled.
        axD.set_ylim(bottom=yD.min() / 1.5)
        axD.set_yticks([0.01, 0.02, 0.03, 0.04, 0.06])
        axD.set_yticklabels(["0.01", "0.02", "0.03", "0.04", "0.06"])
        axD.minorticks_off()
        # Grow the top limit until a measurement says the legend is clear. A fixed
        # multiplier is not trustworthy here: the legend is sized at draw time, and
        # 3.4x still left five points under the box.
        _series = ((dv, yD), (dd, yD[0] * (dd / dv[0]) ** 2.0))

        def _hits_under_legend():
            fig.canvas.draw()
            bb = legD.get_window_extent()
            n = 0
            for x, y in _series:
                dsp = axD.transData.transform(np.column_stack([x, y]))
                n += int(np.sum((dsp[:, 0] >= bb.x0) & (dsp[:, 0] <= bb.x1)
                                & (dsp[:, 1] >= bb.y0) & (dsp[:, 1] <= bb.y1)))
            return n

        _top = yD.max() * 1.6
        for _ in range(24):
            axD.set_ylim(top=_top)
            if _hits_under_legend() == 0:
                break
            _top *= 1.18
        print(f"[legend] panel (c) ymax {_top:.4f} "
              f"({_top / yD.max():.2f}x the largest point); "
              f"{_hits_under_legend()} points under the box", flush=True)

        for ax, lab in ((axR, "(a)"), (axK, "(b)"), (axD, "(c)")):
            ax.text(0.02, 1.02, lab, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=_pt(9.0), fontweight="bold")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")

    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
