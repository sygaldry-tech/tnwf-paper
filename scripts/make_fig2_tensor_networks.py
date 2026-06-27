"""Figure 2 of the paper: tensor-network realisation of the quantum path.

Two panels showing the single main-text pipeline -- an MPS state evolved by
a generator-TT / TDVP velocity step:
  (A) State encoding. The flow state psi is stored as an MPS: one core per
      spatial axis, local dimension N, bond dimension D.
  (B) The position-diagonal velocity step exp(i beta V_t) is applied from
      the real *generator* V_t -- never as a generic MPO. The generator is
      represented as a real tensor train (crossed by TCI at runtime, or
      supplied directly by a trained V), and a TDVP step evolves psi under
      it, growing the bond adaptively (TDVP-1/2).

The diagonal-operator (Hadamard) route and the quantics QTT encoding are
relegated to the supplement and are not drawn here.

Usage:
    uv run python scripts/make_fig2_tensor_networks.py \\
        --out figures/fig2_tensor_networks.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# ── Visual constants ─────────────────────────────────────────────────
# Okabe-Ito colourblind-safe tints (black borders + text labels add a
# second channel): psi = blue, generator = green.
CLR_PSI  = "#cfe0f0"   # blue tint   (wavefunction cores)
CLR_GEN  = "#cfe9df"   # green tint  (generator TT)
CLR_TEXT = "black"
CLR_ARROW = "dimgray"


def _chain(ax, x_left, y, n, fc, box=0.5, gap=0.42, leg=0.34,
           leg_dir="down", site_labels=None, bond_label=None,
           phys_label=None, fontsize=13, clip=True, zlift=0):
    """Horizontal chain of n tensor boxes with physical legs and bonds.

    ``clip=False`` lets the chain draw outside the axes data limits (so it is
    not cut off by an equal-aspect data window); ``zlift`` raises its zorder.
    Returns (list of box centres, right edge x)."""
    pitch = box + gap
    centers = [x_left + box / 2 + i * pitch for i in range(n)]
    for i, cx in enumerate(centers):
        ax.add_patch(Rectangle((cx - box / 2, y - box / 2), box, box,
                               fc=fc, ec="black", lw=1.1, zorder=3 + zlift,
                               clip_on=clip))
        if site_labels:
            ax.text(cx, y, site_labels[i], ha="center", va="center",
                    fontsize=fontsize, zorder=4 + zlift, clip_on=clip)
        if leg_dir in ("down", "both"):
            ax.plot([cx, cx], [y - box / 2, y - box / 2 - leg], "k-", lw=1.0,
                    zorder=2 + zlift, clip_on=clip)
        if leg_dir in ("up", "both"):
            ax.plot([cx, cx], [y + box / 2, y + box / 2 + leg], "k-", lw=1.0,
                    zorder=2 + zlift, clip_on=clip)
        if i < n - 1:
            ax.plot([cx + box / 2, centers[i + 1] - box / 2], [y, y], "k-",
                    lw=1.0, zorder=2 + zlift, clip_on=clip)
    if bond_label and n > 1:
        ax.text((centers[0] + centers[1]) / 2, y + box / 2 + 0.08,
                bond_label, ha="center", va="bottom", fontsize=11,
                color="dimgray")
    if phys_label:
        ax.text(centers[0] + 0.14, y - box / 2 - leg / 2, phys_label,
                ha="left", va="center", fontsize=11, color="dimgray")
    return centers, centers[-1] + box / 2


def _draw_panel_a(ax):
    """MPS encoding of the flow state psi: one core per spatial axis."""
    ax.set_xlim(0, 8)
    ax.set_ylim(1.5, 5.1)
    ax.set_aspect("equal", adjustable="datalim")  # render cores as true squares
    ax.axis("off")

    # ---- MPS: one rank-<=D core per spatial axis x_j ----
    y_mps = 3.5
    c, _ = _chain(ax, 2.55, y_mps, 4, CLR_PSI, box=0.70, gap=0.58, leg=0.46,
                  bond_label="$D$", phys_label="$N$",
                  site_labels=[r"$A_1$", r"$A_2$", r"$\cdots$", r"$A_d$"])
    ax.text(1.95, y_mps, r"$|\psi\rangle =$", ha="right", va="center",
            fontsize=16)

    ax.set_title("(a)  MPS Encoding of the Flow State", fontsize=17,
                 fontweight="bold", loc="left", pad=10)


def _draw_panel_b(ax):
    """The generator-TT / TDVP route to the velocity step exp(i beta V_t)."""
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0.3, 5.45)
    ax.set_aspect("equal", adjustable="datalim")  # render cores as true squares
    ax.axis("off")

    sb, sg = 0.42, 0.28          # chain box / gap
    leg = 0.32

    # ===== real generator TT, evolved by TDVP =====
    # Keep content centered within the equal-aspect (datalim) y-window so the
    # generator chain near the top is not clipped.
    y = 2.7
    cP, rP = _chain(ax, 0.85, y, 3, CLR_PSI, box=sb, gap=sg, leg=leg)
    ax.text(0.72, y, r"$\psi$", ha="right", va="center", fontsize=16)
    # TDVP step box.
    xb = rP + 1.75
    ax.add_patch(FancyBboxPatch((xb - 1.0, y - 0.58), 2.0, 1.16,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc="white", ec="black", lw=1.5, zorder=3))
    ax.text(xb, y, "TDVP\nstep", ha="center", va="center", fontsize=14,
            zorder=4)
    ax.add_patch(FancyArrowPatch((rP + 0.28, y), (xb - 1.04, y),
                 arrowstyle="-|>", mutation_scale=17, lw=1.7,
                 color=CLR_ARROW))
    # Generator TT feeding the TDVP step from above.
    yg = y + 1.45
    cG, rG = _chain(ax, xb - 1.0, yg, 3, CLR_GEN, box=sb, gap=sg, leg=0.24,
                    leg_dir="up", clip=False, zlift=5)
    ax.text(rG + 0.22, yg, r"generator $V_t$" "\n" r"(real TT)",
            ha="left", va="center", fontsize=12.5, color="dimgray")
    ax.add_patch(FancyArrowPatch((xb, yg - 0.30), (xb, y + 0.60),
                 arrowstyle="-|>", mutation_scale=15, lw=1.5,
                 color=CLR_ARROW))
    # Output state.
    ax.add_patch(FancyArrowPatch((xb + 1.04, y), (xb + 1.7, y),
                 arrowstyle="-|>", mutation_scale=17, lw=1.7,
                 color=CLR_ARROW))
    cO, rO = _chain(ax, xb + 1.95, y, 3, CLR_PSI, box=sb, gap=sg, leg=leg)
    ax.text(rO + 0.22, y, r"$\psi'$", ha="left", va="center", fontsize=16)

    ax.set_title(r"(b)  Generator-TT Velocity Step  $e^{i\beta V_t}$  via TDVP",
                 fontsize=17, fontweight="bold", loc="left", pad=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",
                   default="figures/fig2_tensor_networks.pdf")
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.30, h_pad=0.20)
    _draw_panel_a(axes[0])
    _draw_panel_b(axes[1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
