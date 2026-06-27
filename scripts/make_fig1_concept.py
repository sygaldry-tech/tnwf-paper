"""Conceptual figure 1 of the paper.

Two panels:
  (A) Wavefunction-flow lift: source N(0, sigma_0^2 I) -> target via
      Trotterised unitary with H^c = i[K, V_t].
  (B) Bifurcation of the shared oracle velocity V_t (fit by action /
      flow matching) into a CLASSICAL path -- gradient flow realised by
      integrating the marginal ODE x' = v_t(x) -- and a QUANTUM path --
      wavefunction flow under H^c = i[K, V_t] realised with tensor
      networks (MPS / QTT; see Fig. 2).

The tensor-network architecture and V-step routes live in their own
figure (make_fig2_tensor_networks.py).

Usage:
    uv run python scripts/make_fig1_concept.py \\
        --out figures/fig1_concept.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# ── Visual constants ─────────────────────────────────────────────────
# Okabe-Ito colourblind-safe families (blobs also differ in shape/position).
CLR_SOURCE = "#7fb8e3"   # blue   (source psi_0)
CLR_MID    = "#c79ec0"   # reddish-purple (intermediate psi_t)
CLR_TARGET = "#ec9e74"   # vermillion/orange (target psi_T)
CLR_ARROW  = "#6a737d"
CLR_MPS    = "#dbe6f3"   # very light blue
CLR_MPO    = "#fde9d9"   # very light orange
CLR_TEXT   = "black"
# Bifurcation palette: soft fill + saturated accent per role. Accents are
# colourblind-safe; the oracle is neutral slate so the two CVD-distinct
# branch hues (vermillion / purple -- the orange-purple axis is preserved
# under CVD) carry the bifurcation.
CLR_ORACLE     = "#edeef1"   # slate fill (shared oracle)
CLR_ORACLE_ACC = "#5b6470"   # slate accent
CLR_ORACLE_TXT = "#2f3640"   # dark slate text
CLR_CLASS      = "#fae4d6"   # lane tint (classical / vermillion)
CLR_CLASS_ACC  = "#D55E00"   # Okabe-Ito vermillion
CLR_QUANT      = "#f0e1ee"   # lane tint (quantum / purple)
CLR_QUANT_ACC  = "#9C4F96"   # CB-safe magenta-purple


def _gauss_blob(ax, cx, cy, sigma, color, n_levels=4):
    """Filled-contour Gaussian centered at (cx, cy)."""
    xx, yy = np.meshgrid(
        np.linspace(cx - 4 * sigma, cx + 4 * sigma, 64),
        np.linspace(cy - 4 * sigma, cy + 4 * sigma, 64),
    )
    z = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    levels = np.linspace(0.05, 1.0, n_levels)
    ax.contourf(xx, yy, z, levels=levels, colors=[color] * len(levels),
                alpha=0.55)


def _gmm_blobs(ax, cx, cy, scale, color):
    """Four well-separated Gaussians at +/- e_j (orthogonal GMM target):
    the modes do not overlap and the centre is empty (clear of the label)."""
    s = scale * 0.27
    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
        _gauss_blob(ax, cx + dx, cy + dy, sigma=scale * 0.052, color=color)


def _gmm_intermediate(ax, cx, cy, scale, color):
    """An intermediate marginal distribution (not a superposition): mass
    spreading from the centre toward the four modes, drawn as diffuse,
    partially merged blobs that form a filled cross still joined in the
    middle, the modes not yet separated."""
    s = scale * 0.12
    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
        _gauss_blob(ax, cx + dx, cy + dy, sigma=scale * 0.095, color=color,
                    n_levels=3)


def _draw_panel_a(ax):
    """Source -> intermediate -> target wavefunction-flow illustration with
    a generation-time axis, styled to match the bifurcation panel."""
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 5.0)
    ax.axis("off")
    ax.set_aspect("equal")          # circular Gaussians, not ellipses
    ax.set_anchor("N")              # top-align with Panel B after letterboxing

    BLOB_Y = 3.75
    xs0, xm, xt = 1.55, 4.0, 6.35

    # ---- the three states ----
    _gauss_blob(ax, xs0, BLOB_Y, sigma=0.42, color=CLR_SOURCE)
    ax.text(xs0, BLOB_Y, r"$\psi_0$", ha="center", va="center",
            fontsize=22, color=CLR_TEXT, fontweight="bold", zorder=5)

    _gmm_intermediate(ax, xm, BLOB_Y, scale=2.2, color=CLR_MID)
    ax.text(xm, BLOB_Y, r"$\psi_t$", ha="center", va="center",
            fontsize=20, color=CLR_TEXT, fontweight="bold", zorder=5)

    _gmm_blobs(ax, xt, BLOB_Y, scale=2.2, color=CLR_TARGET)
    ax.text(xt, BLOB_Y, r"$\psi_T$", ha="center", va="center",
            fontsize=22, color=CLR_TEXT, fontweight="bold", zorder=5)

    # ---- flow arrows in the gaps ----
    for x0, x1 in ((xs0 + 1.15, xm - 0.90), (xm + 0.90, xt - 0.85)):
        ax.add_patch(FancyArrowPatch(
            (x0, BLOB_Y + 0.05), (x1, BLOB_Y + 0.05),
            connectionstyle="arc3,rad=-0.22", arrowstyle="-|>",
            mutation_scale=20, lw=2.2, color=CLR_ARROW, zorder=4))

    # ---- generation-time axis ----
    y_ax = 1.15
    ax.add_patch(FancyArrowPatch(
        (0.7, y_ax), (7.55, y_ax), arrowstyle="-|>", mutation_scale=16,
        lw=1.6, color="#9aa3ad", zorder=2))
    for x, lab in ((xs0, r"$t=0$"), (xm, r"$t=T/2$"), (xt, r"$t=T$")):
        ax.plot([x, x], [y_ax + 0.10, y_ax - 0.10], "-", color="#9aa3ad",
                lw=1.4, zorder=3)
        ax.text(x, y_ax - 0.42, lab, ha="center", va="center", fontsize=12,
                color="dimgray")
    ax.text(7.55, y_ax + 0.32, r"time $t$", ha="right", va="center",
            fontsize=11.5, color="#9aa3ad")

    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")


def _rounded_box(ax, cx, cy, w, h, lines, fc, fontsizes=None,
                 weights=None, colors=None, ec="black", lw=1.4, z=3):
    """A rounded box centred at (cx, cy) with stacked text lines."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        fc=fc, ec=ec, lw=lw, zorder=z))
    n = len(lines)
    fontsizes = fontsizes or [14] * n
    weights = weights or ["normal"] * n
    colors = colors or [CLR_TEXT] * n
    gap = h / (n + 1)
    for k, txt in enumerate(lines):
        yy = cy + h / 2 - gap * (k + 1)
        ax.text(cx, yy, txt, ha="center", va="center",
                fontsize=fontsizes[k], fontweight=weights[k],
                color=colors[k], zorder=z + 1)


def _lane(ax, cx, w, y0, y1, fc):
    """Soft rounded background lane for a branch."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, y0), w, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.22",
        fc=fc, ec="none", zorder=0))


def _chip(ax, cx, cy, w, h, text, fc, tc, ec, fontsize=14, weight="bold",
          z=4):
    """Pill-shaped label chip."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.30",
        fc=fc, ec=ec, lw=1.5, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=tc, zorder=z + 1)


def _classical_glyph(ax, cx, cy, acc):
    """Tiny sample-transport cartoon: dots flowing along a soft arc."""
    xs = np.linspace(cx - 0.55, cx + 0.55, 5)
    ys = cy + 0.18 * np.sin((xs - cx) * 2.4)
    ax.plot(xs, ys, "-", color=acc, lw=1.4, alpha=0.55, zorder=3)
    ax.scatter(xs, ys, s=26, color=acc, zorder=4, edgecolors="white",
               linewidths=0.6)


def _quantum_glyph(ax, cx, cy, acc):
    """Tiny |psi_T|^2 four-mode density cartoon."""
    s = 0.30
    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
        ax.scatter([cx + dx], [cy + dy], s=120, color=acc, alpha=0.30,
                   zorder=3)
        ax.scatter([cx + dx], [cy + dy], s=34, color=acc, alpha=0.85,
                   zorder=4)


def _draw_panel_b(ax):
    """Bifurcation of the shared oracle velocity V_t into a classical
    (gradient-flow ODE) and a quantum (wavefunction-flow tensor-network)
    realisation. Sized ~1:1 to sit beside Panel A in a single row."""
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    xL, xR = 2.0, 6.0     # classical / quantum lane centres
    x_or = 4.0            # oracle centre

    # ---- branch lanes (drawn first, behind everything) ----
    _lane(ax, xL, 3.35, 0.25, 3.85, CLR_CLASS)
    _lane(ax, xR, 3.35, 0.25, 3.85, CLR_QUANT)

    # ---- shared oracle (top, centre) ----
    _rounded_box(
        ax, x_or, 4.48, 5.6, 0.72,
        lines=[r"Oracle  $v_t(x) = \nabla V_t(x)$"],
        fc=CLR_ORACLE, ec=CLR_ORACLE_ACC, lw=1.8,
        fontsizes=[15], weights=["bold"],
        colors=[CLR_ORACLE_TXT])

    # ---- bifurcation arrows (curved, colour-matched) ----
    ax.add_patch(FancyArrowPatch(
        (x_or - 0.75, 3.98), (xL, 3.62),
        connectionstyle="arc3,rad=0.22", arrowstyle="-|>",
        mutation_scale=20, lw=2.4, color=CLR_CLASS_ACC, zorder=2))
    ax.add_patch(FancyArrowPatch(
        (x_or + 0.75, 3.98), (xR, 3.62),
        connectionstyle="arc3,rad=-0.22", arrowstyle="-|>",
        mutation_scale=20, lw=2.4, color=CLR_QUANT_ACC, zorder=2))

    branches = [
        dict(xb=xL, acc=CLR_CLASS_ACC, title="CLASSICAL",
             eq=r"$\dot{x}_t = v_t(x_t)$", method="integrate ODE",
             glyph=_classical_glyph),
        dict(xb=xR, acc=CLR_QUANT_ACC, title="QUANTUM",
             eq=r"$H^{\mathrm{c}} = i[K, V_t]$", method="evolve wavefunction",
             glyph=_quantum_glyph),
    ]
    for b in branches:
        xb, acc = b["xb"], b["acc"]
        # header chip (solid accent)
        _chip(ax, xb, 3.25, 2.5, 0.58, b["title"], fc=acc, tc="white",
              ec=acc, fontsize=14)
        # governing equation (accent colour)
        ax.text(xb, 2.42, b["eq"], ha="center", va="center", fontsize=15,
                color=acc)
        # action
        ax.text(xb, 1.62, b["method"], ha="center", va="center",
                fontsize=13.5, color="#444444")
        # glyph
        b["glyph"](ax, xb, 0.82, acc)

    ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig1_concept.pdf")
    args = p.parse_args()

    # Single-row 1×2 layout, with each panel sized roughly 1:1
    # (square). Total figsize 12×6 → at LaTeX width=0.95\textwidth
    # (~6.18in) renders at 6.18 × 3.09in. Per-panel internal data
    # range is 8×7 (~1.14:1) which leaves headroom for the panel
    # title and the t/Hamiltonian labels above the blobs.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=300,
                              constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.30, h_pad=0.10)
    _draw_panel_a(axes[0])
    _draw_panel_b(axes[1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
