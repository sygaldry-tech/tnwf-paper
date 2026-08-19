"""Rare-event sampling on the d=8 GMM: cost, and what a fixed budget harvests.

Produces paper Fig. 8 as three square panels:

  (a) sampling cost -- state preparations per accepted rare sample against the
      rare-event probability p_rare, for gradient flow + rejection sampling against
      tensor-network transport + amplitude amplification. Both pipelines are built on
      the SAME learned potential and each is costed on the state its own pipeline
      produces, so this is a pipeline comparison.
  (b) the rejection-sampling harvest at a fixed budget, on the flow-ODE state.
  (c) the amplitude-amplification harvest at the same budget, on the
      wavefunction-flow state.

Panels (b,c) keep the SAME pairing as (a) -- each sampling method is shown on
the state its own pipeline prepares -- so all three panels report one
comparison. They share a single t-SNE fit taken over both clouds at once, which
is what makes their ring densities comparable.

Pipeline
--------
1. Wavefunction-flow pipeline: the paper's V-MPS 2TDVP flow on the d=8 orthogonal Gaussian
   mixture (16 modes at +-3.e_j, sigma=0.5), N=32, K=160, D_max=64, shipped as
   final MPS cores under `examples/rare_event/rep*.npz`. Born samples are drawn
   with `tnwf.pipelines.run_evolution.sample_from_mps` and pooled across the
   independent MPS-V initializations the paper averages over.

   Note on SW: Born samples are dithered about the grid node (`tnwf.coords`), so
   the coordinates need no half-cell correction.

2. Flow-ODE pipeline: the same learned potential, integrated as a gradient flow
   xdot = grad V. This is what makes the comparison fair -- the learning error is
   common to both, so what differs is the transport and the sampling method.
   Both pipelines are classical simulations; "quantum" would misname either.
   Requires the MPS-V checkpoint from the Zenodo data release.

3. Rare events: nearest-mode k-sigma rule. Samples closer to the domain center
   than to any mode are flagged residual (untransported source mass) and excluded
   from the tail amplitude.

4. Costs come from `tnwf.amp`, which chooses the Grover round count to minimize
   expected preparations. There is no amplitude *estimation* here: on this state
   the preparation bias exceeds the estimator's statistical error at every
   reachable budget, so its query scaling is not usable.

Deliverables (figures/)
    fig_rare_event_advantage.{png,pdf}  paper Fig. 8, as included by the manuscript
    rare_event_advantage.npz            threshold sweep, amplitudes, SW, fitted
                                        prefactors; backs the caption's numbers

Run:
    make fig-rare-event
    uv run python scripts/make_fig_rare_event.py [--tail-k 4.0] [--n-samples 40000]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
from scipy.stats import chi2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from tnwf import coords  # noqa: E402,I001
from tnwf.amp import AMP_PREFACTOR, amp_cost  # noqa: E402
from tnwf.data.gaussian_mixture import sample_gaussian_mixture  # noqa: E402
from tnwf.metrics.sw import sliced_wasserstein as _sliced_wasserstein  # noqa: E402
from tnwf.pipelines.run_evolution import sample_from_mps  # noqa: E402

# isort cannot merge these into the block above: `matplotlib.use("Agg")` must run
# after `import matplotlib` but before `pyplot` is imported, so the backend is set
# headlessly. Hence the I001 exemption rather than a reordering.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
# Computer Modern for math, as in every other figure generator. This was the last
# figure on the default "dejavusans" fontset: p, 1/p, 1/sqrt(p), 4sigma and the
# decade ticks all rendered sans, and the radical pulled in a STIXSizeOneSym
# fallback, so one expression drew from three families.
matplotlib.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402


def sample_wf_coords_from_mps(mps, n, N, d, L, rng):
    """Born samples in the CENTERED physical frame [-L/2, L/2]^d.

    `tnwf.pipelines.run_evolution.sample_from_mps` returns grid-frame coordinates
    spanning [-dx/2, L-dx/2)^d -- node-centered, see `tnwf.coords`. This module,
    like `gm_mode_centers` below, works in the centered world frame, so the L/2
    offset must be removed. It matters: without it every sample is displaced by
    L/2 and the tail labeling silently breaks.
    """
    return sample_from_mps(mps, N=N, d=d, L=L, n=n, rng=rng) - L / 2.0


# ============================================================================
# Config
# ============================================================================
OUT_DIR = os.path.join(_PROJECT_ROOT, "figures")

# ── Font sizes, quoted at the size they RENDER in the paper ──────────────
# figsize is 19.5 in and sn-article.tex includes this at \textwidth (~6.3 in),
# a downscale of ~0.32, so the hand-picked sizes landed far below the caption:
# the panel letters at 25 rendered at 8.1 pt, axis labels at 21 at 6.8 pt, the
# panel names at 19 at 6.1 pt, legends and the 4-sigma mark at 16-17 at ~5.4 pt,
# and the tick labels were never set at all, so they inherited the rcParams
# default of 10 and came out near 3.2 pt -- against a caption of roughly 9 pt.
_FIG_W_IN = 19.5
_RENDERED_W_IN = 6.3


def _pt(rendered: float) -> float:
    """The matplotlib fontsize that renders at `rendered` points in the paper."""
    return round(rendered * _FIG_W_IN / _RENDERED_W_IN, 1)


FS_LABEL = _pt(9.0)     # axis labels, at parity with the caption
FS_TICK = _pt(8.0)      # tick labels
FS_TITLE = _pt(8.0)     # panel titles, one step below the labels
FS_ANNOT = _pt(8.0)     # the 4-sigma marker
# The one element in the paper below the 8 pt caption floor, and it is a width
# problem rather than a height one. Panel (a) is pinned square and is ~2.1 in wide
# as rendered; "amplitude amplification" at 8.5 pt needs ~1.2 in, so a four-entry
# legend spans the curve it explains no matter how much ymax headroom it is given
# -- _fit_legend_clearance ran to 57x and still measured an overlap, in one column
# and in two. Raising this needs the legend moved out of the axes (e.g. merged into
# the figure-level row at the bottom), which is a layout change, not a font change.
FS_LEGEND = _pt(6.5)
FS_PANEL = _pt(9.0)     # (a)/(b)/(c) letters, bold at caption size

PAPER_DIR = os.path.join(_PROJECT_ROOT, "examples/rare_event")  # Table-2 K=160 MPS cores
CKPT_DIR = os.path.join(_PROJECT_ROOT, "data/mps_v_checkpoints")

D = 8
N = 32                 # spatial grid (paper config); dx = L/N
L = 8.0                # domain edge; centered physical frame is [-L/2, L/2]^d
SIGMA = 0.5            # GMM component std
SCALE = 3.0            # mode separation (centers at +-SCALE on each axis)
DEFAULT_TAIL_K = 4.0   # rare-event threshold in units of sigma (the paper's value)
N_WF = 40000           # pooled Born samples per pipeline; the >4sigma tail is ~9%
N_SW = 4000            # samples used for SW only -- see below
SEED = 0

# SW is evaluated on a fixed N_SW subset rather than on all N_WF samples. The
# estimator is sample-size dependent (more samples -> lower SW: 0.030 at 4k
# against 0.019 at 40k on this same state), so computing it at the tail-statistics
# sample count would silently change a number the manuscript compares against
# Table 2. Tail fractions want as many samples as possible; SW wants comparability.

THRESHOLDS = np.arange(2.5, 6.51, 0.25)   # k-sigma values swept in panel (a)
TAIL_FIT_CUT = 0.20    # fit the amplification power law on p <= this (see below)
BUDGET = 500           # state preparations, for the panel (b,c) harvest
# Starting headroom above the highest cost in panel (a). _fit_legend_clearance
# grows it until the four-entry legend measurably clears the data, so this is a
# lower bound and a speed hint, not the final limit.
LEGEND_HEADROOM = 2.0

# Validated categorical pair; the paper's older #3b0f70/#8c8c8c fails the
# lightness and chroma checks (the gray reads as absence of a series, not as one).
C_Q, C_C = "#4a3aa7", "#eb6834"


# ============================================================================
# Geometry
# ============================================================================
def _pct(x: float) -> str:
    """Format a fraction as a percentage with 0.01% resolution."""
    return f"{100.0 * x:.2f}%"


def gm_mode_centers() -> np.ndarray:
    """Orthogonal centers at +-SCALE.e_j (centered physical frame)."""
    C = np.zeros((2 * D, D))
    for j in range(D):
        C[2 * j, j] = SCALE
        C[2 * j + 1, j] = -SCALE
    return C


def sliced_wasserstein(a, b, n_proj=128, seed=0) -> float:
    """The paper's SW estimator: `tnwf.metrics.sw` at 128 projections."""
    return float(_sliced_wasserstein(a, b, n_projections=n_proj,
                                     rng=np.random.default_rng(seed)))


# ============================================================================
# Stage 1 -- Born samples from the paper's V-MPS 2TDVP state
# ============================================================================
def load_wf_samples(source: str, seed: int, n_wf: int = N_WF):
    """Return (wf, target, sw_stats) in the centered [-L/2, L/2]^d frame."""
    rng = np.random.default_rng(seed)
    target = sample_gaussian_mixture(
        n_wf, d=D, std=SIGMA, scale=SCALE, arrangement="orthogonal",
        seed=1).astype(np.float64)

    if source == "paper":
        reps = sorted(glob.glob(os.path.join(PAPER_DIR, "rep*.npz")))
        if not reps:
            raise FileNotFoundError(
                f"No Table-2 MPS cores under {PAPER_DIR}; expected rep*.npz.")
        per = int(np.ceil(n_wf / len(reps)))
        chunks, stored_unb = [], []
        for d_ in reps:
            z = np.load(d_)
            mps = [z[f"core{j}"] for j in range(int(z["n_cores"]))]
            chunks.append(sample_wf_coords_from_mps(mps, per, N, D, L, rng))
            stored_unb.append(float(z["sw_unbiased_final"]))
        wf = np.vstack(chunks)[:n_wf]
        rng.shuffle(wf)
        n_sw = min(N_SW, len(wf))
        sw = sliced_wasserstein(wf[:n_sw], target[:n_sw])
        print(f"[load] paper Table-2 state: pooled {wf.shape[0]} Born samples over "
              f"{len(reps)} MPS-V reps (K=160, N={N}, D_max=64)", flush=True)
        print(f"       SW={sw:.4f} on n={n_sw}  "
              f"(paper Table-2 per-rep mean={np.mean(stored_unb):.4f})", flush=True)
        sw_stats = dict(sw=sw,
                        paper_unbiased_mean=float(np.mean(stored_unb)),
                        paper_unbiased_ci=float(1.96 * np.std(stored_unb)
                                                / np.sqrt(len(stored_unb))),
                        n_reps=len(reps), K=160, D_max=64, n_sw=n_sw)
    else:
        raise ValueError(f"unknown source {source!r}")
    return wf, target, sw_stats


# ============================================================================
# Stage 2 -- flow-ODE pipeline: gradient flow under the SAME learned potential
# ============================================================================
def classical_gradient_flow(n, seed=0, kref=160, sub=4, chunk=5000):
    """Integrate xdot = grad V of the MPS-V oracle the tensor network consumes.

    This is the flow-ODE pipeline of the comparison. Using the same learned
    potential is the point: oracle error is then common to both, so panel
    (a) contrasts the transports and sampling methods rather than two different
    velocity fields.

    Chunked to bound memory: the autograd graph is over (chunk, d) at each of
    kref*sub substeps.
    """
    import torch  # local: the plotting path must import without torch

    from tnwf.mps_v.model import MPSScalarPotentialTimeSite
    from tnwf.pipelines.run_evolution import checkpoint_source_sigma

    cands = [f for f in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
             if "_d8_" in os.path.basename(f)]
    if not cands:
        raise FileNotFoundError(
            f"No d=8 MPS-V checkpoint under {CKPT_DIR}.\n"
            "The flow-ODE pipeline needs it. Fetch the Zenodo data "
            "release and place or symlink it at data/mps_v_checkpoints/.")
    ck = torch.load(cands[0], map_location="cpu", weights_only=True)
    a = ck["args"]
    model = MPSScalarPotentialTimeSite(d=D, N=int(a["N_grid"]), D=int(a["D_mps"]),
                                       L=float(a["L"]), N_t=int(a["N_t"]))
    model.load_state_dict(ck["model"])
    model.eval()
    s0 = checkpoint_source_sigma(a, L)
    dt = 1.0 / (kref * sub)
    print(f"[classical] {os.path.basename(cands[0])}  sigma_0={s0:.4f}  "
          f"{kref * sub} substeps on {n} samples", flush=True)

    out = []
    for c0 in range(0, n, chunk):
        nb = min(chunk, n - c0)
        g = torch.Generator().manual_seed(1000 + seed + c0)
        x = s0 * torch.randn(nb, D, generator=g)
        for k in range(kref):
            for s in range(sub):
                xin = x.detach().requires_grad_(True)
                t = torch.full((nb, 1), (k * sub + s) * dt)
                gx = torch.autograd.grad(model(xin, t).sum(), xin)[0]
                with torch.no_grad():
                    x = (x + dt * gx).clamp(-L / 2, L / 2)
        out.append(x.detach().numpy().astype(np.float64))
        print(f"           {c0 + nb}/{n}", flush=True)
    return np.vstack(out)


# ============================================================================
# Stage 3 -- rare-event (k-sigma) tail labeling, residual separation
# ============================================================================
def label_tails(samples, centers, tail_k):
    """Nearest-mode radial k-sigma rule with residual-mass separation.

    residual  = closer to the domain center (origin) than to any mode.
    rare tail = non-residual and nearest-mode distance > tail_k*sigma.
    Returns (nearest_mode, nearest_dist, tail_mask, residual_mask).
    """
    dist = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dist.argmin(axis=1)
    nearest_dist = dist.min(axis=1)
    dist_center = np.linalg.norm(samples, axis=1)   # centered frame: origin
    residual = dist_center < nearest_dist
    tail = (nearest_dist > tail_k * SIGMA) & (~residual)
    return nearest, nearest_dist, tail, residual


# ============================================================================
# Stage 4 -- cost of a rare sample, swept over the threshold
# ============================================================================
def sweep_thresholds(wf, cl, centers):
    """Tail masses and per-rare-sample costs for both pipelines, over THRESHOLDS.

    Rejection costs 1/p preparations per accepted rare sample -- an identity,
    not a model. Amplification costs `amp_cost(p)`; see `tnwf.amp` for why the
    round count minimizes expected preparations rather than maximizing success
    probability.
    """
    rows = []
    for k in THRESHOLDS:
        a_cl = float(label_tails(cl, centers, k)[2].mean())
        a_wf = float(label_tails(wf, centers, k)[2].mean())
        if a_cl <= 0.0 or a_wf <= 0.0:
            continue                                  # no samples that far out
        rows.append((k, float(chi2.sf(k ** 2, df=D)), a_cl, a_wf,
                     1.0 / a_cl, float(amp_cost(a_wf))))
    r = np.array(rows)
    return dict(k=r[:, 0], a_true=r[:, 1], a_cl=r[:, 2], a_wf=r[:, 3],
                preps_cl=r[:, 4], preps_q=r[:, 5])


def _fit_prefactor(p, cost, exponent):
    """Least squares in log space with the exponent held at its structural value."""
    return float(np.exp(np.mean(np.log(cost) - exponent * np.log(p))))


# ============================================================================
# Stage 5 -- the figure
# ============================================================================
# Margin the legend must keep from any plotted geometry, quoted in PRINTED
# points. get_window_extent works in display pixels at fig.dpi, and this
# figure is then downscaled ~0.32 into the page, so a margin written straight
# into display units is worth about a seventh of its face value on paper --
# 10 there bought 1.4 pt here, which still read as a collision.
_LEGEND_MARGIN_PT = 4.0


def _margin_px(fig):
    """`_LEGEND_MARGIN_PT` printed points, expressed in display pixels."""
    return _LEGEND_MARGIN_PT * (_FIG_W_IN / _RENDERED_W_IN) * fig.dpi / 72.0


def _seg_hits_box(p0, p1, bb):
    """Does the segment p0->p1 touch the rectangle bb? (Liang-Barsky clip.)"""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bb.x0), (dx, bb.x1 - x0),
                 (-dy, y0 - bb.y0), (dy, bb.y1 - y0)):
        if p == 0:
            if q < 0:
                return False          # parallel and outside this slab
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def _legend_hits(fig, ax, leg, series):
    """How much plotted geometry the legend covers, in display coords.

    Counts SEGMENTS, not just vertices. The model curves are polylines through
    the 17 swept thresholds, so testing vertices alone passed a legend whose
    lower edge cut between two of them -- the curve visibly ran under the box
    while the check reported it clear. Markers are counted as degenerate
    segments by the same test.
    """
    fig.canvas.draw()
    # Expanded by a margin: "not intersecting" is too weak a test. At zero overlap
    # the legend's lower-left corner still sat a couple of points off the top of
    # the rejection curve, which at the printed panel width (~2 in) reads as a
    # collision even though nothing technically crosses. Require visible daylight.
    m = _margin_px(fig)
    e = leg.get_window_extent()
    bb = Bbox.from_extents(e.x0 - m, e.y0 - m, e.x1 + m, e.y1 + m)
    hits = 0
    for x, y in series:
        d = ax.transData.transform(np.column_stack([x, y]))
        for i in range(len(d)):
            if (bb.x0 <= d[i, 0] <= bb.x1) and (bb.y0 <= d[i, 1] <= bb.y1):
                hits += 1
            if i + 1 < len(d) and _seg_hits_box(d[i], d[i + 1], bb):
                hits += 1
    return hits


def _fit_legend_clearance(fig, ax, leg, series, bottom, top):
    """Raise `top` until the legend stops covering data; return the limit used.

    constrained_layout sizes the legend at draw time, so the clearance needed
    cannot be derived from font sizes -- and a hand-tuned multiplier silently
    goes stale when the fonts, the entry count or the data range change. Growing
    the limit until a measurement comes back clean keeps it honest. Each step
    only re-draws; the sampling pipeline is not re-run.
    """
    for _ in range(24):
        ax.set_ylim(bottom, top)
        if _legend_hits(fig, ax, leg, series) == 0:
            return top, True
        top *= 1.15
    return top, False


def _axtag(ax, txt, inside=False):
    if inside:
        ax.text(0.015, 0.975, txt, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", va="top", ha="left")
    else:
        ax.text(-0.02, 1.03, txt, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", va="bottom", ha="right")


def make_master_figure(sw_data, wf, cl, centers, out_dir, seed=0,
                       tail_k=DEFAULT_TAIL_K):
    """Three square panels: sampling cost, then the harvest split by method."""
    from sklearn.manifold import TSNE

    k = sw_data["k"]
    a_cl, a_wf = sw_data["a_cl"], sw_data["a_wf"]
    preps_cl, preps_q = sw_data["preps_cl"], sw_data["preps_q"]

    fig = plt.figure(figsize=(19.5, 7.4), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_cost = fig.add_subplot(gs[0, 0])
    ax_rej = fig.add_subplot(gs[0, 1])
    ax_amp = fig.add_subplot(gs[0, 2])

    # ---- (a) sampling cost -------------------------------------------------
    p_at = lambda kk: float(np.interp(kk, k, a_wf))            # noqa: E731
    ax_cost.axvline(p_at(tail_k), color="0.4", lw=1.5, ls=(0, (1, 2.2)), zorder=2)
    ax_cost.text(p_at(tail_k) * 1.15, 0.04, rf"${tail_k:g}\sigma$",
                 transform=ax_cost.get_xaxis_transform(),
                 ha="left", va="bottom", fontsize=FS_ANNOT, color="0.3")

    # Rejection's 1/p is an identity holding at every p, so it is fitted and drawn
    # across all points. Amplification's 1/sqrt(p) is asymptotic and fails past
    # p ~ 0.2, where the optimal round count reaches zero and the cost flattens
    # onto the one-preparation floor, so it is fitted and drawn on the tail only.
    mq = a_wf <= TAIL_FIT_CUT
    c_cl = _fit_prefactor(a_cl, preps_cl, -1.0)
    c_q = _fit_prefactor(a_wf[mq], preps_q[mq], -0.5)
    xc, xq = np.sort(a_cl), np.sort(a_wf[mq])
    print(f"[fit] rejection {c_cl:.4f}/p over all points (exact 1 by construction); "
          f"amplification {c_q:.3f}/sqrt(p) on p<={TAIL_FIT_CUT} "
          f"(asymptote {AMP_PREFACTOR:.3f})", flush=True)

    ax_cost.loglog(xc, c_cl / xc, "-", color=C_C, lw=2.4, zorder=2,
                   label=r"$\propto 1/p_{\mathrm{rare}}$   (slope $-1$)")
    ax_cost.loglog(xq, c_q / np.sqrt(xq), "-", color=C_Q, lw=2.4, zorder=2,
                   label=r"$\propto 1/\sqrt{p_{\mathrm{rare}}}$   (slope $-1/2$)")
    ax_cost.loglog(a_cl, preps_cl, ls="none", marker="s", color=C_C, ms=9,
                   mfc="white", mew=2.0, zorder=3, label="rejection sampling")
    ax_cost.loglog(a_wf, preps_q, ls="none", marker="o", color=C_Q, ms=9,
                   zorder=4, label="amplitude amplification")
    # Subscripted to match the manuscript: bare p is the probability density
    # (Sec. 2.3) and the grid momentum (SI S1.2), so the rare-event mass is
    # written p_rare everywhere it appears -- legend and axis label included.
    ax_cost.set_xlabel("$p_{\\mathrm{rare}}$", fontsize=FS_LABEL)
    ax_cost.set_title("Sampling Cost", fontsize=FS_TITLE, color="0.25", pad=10)
    ax_cost.tick_params(labelsize=FS_TICK)
    # Wrapped: at caption parity this is the longest string in the figure and,
    # set on one line, it overran the axes at both ends and collided with the
    # (a) panel letter. "per" rather than "/" also matches the caption.
    ax_cost.set_ylabel("state preparations\nper rare sample", fontsize=FS_LABEL)
    ax_cost.grid(True, which="both", color="0.85", lw=0.7, alpha=0.8)
    ax_cost.set_axisbelow(True)
    leg_cost = ax_cost.legend(fontsize=FS_LEGEND, loc="upper right",
                              framealpha=0.95, borderpad=0.5)
    xs = np.concatenate([a_cl, a_wf])
    ys = np.concatenate([preps_cl, preps_q])
    ax_cost.set_xlim(xs.min() / 1.12, xs.max() * 1.12)
    # Both costs fall from upper-left to lower-right, so the legend's upper-right
    # corner is the emptiest region -- but at four entries the box still reached
    # far enough left to sit on top of the rejection markers. The y axis carries
    # the headroom rather than the x axis: box_aspect is pinned square, so
    # stretching ymax compresses the data downward and opens a clear band under
    # the legend without changing the panel's shape or the fitted slopes.
    ax_cost.set_ylim(ys.min() / 1.12, ys.max() * LEGEND_HEADROOM)
    ax_cost.set_box_aspect(1)
    _axtag(ax_cost, "(a)")

    # ---- (b,c) harvest at a fixed budget -----------------------------------
    # Each panel shows the state its OWN pipeline prepares: rejection sampling
    # runs on the flow-ODE state, amplification on the wavefunction-flow state.
    # Pairing both panels to the wavefunction-flow state instead would make (b,c)
    # a different comparison from (a) -- the sampling method at fixed state
    # quality, rather than the two pipelines -- and the two lifts would disagree
    # (2.31x against panel (a)'s 2.51x) for a reason no reader could see.
    rng = np.random.default_rng(seed)

    def _cloud(pts):
        _near, _, _tail, _resid = label_tails(pts, centers, tail_k)
        bulk = np.flatnonzero(~_tail & ~_resid)
        if len(bulk) > 1600:
            bulk = rng.choice(bulk, 1600, replace=False)
        return _near, bulk, np.flatnonzero(_tail)[:600]

    near_c, bulk_c, pool_c = _cloud(cl)   # flow ODE      -> panel (b)
    near_q, bulk_q, pool_q = _cloud(wf)   # wavefunction  -> panel (c)

    # One t-SNE fit over both clouds, so the panels share a frame and the ring
    # densities are directly comparable. Fitting each panel separately would put
    # them in unrelated coordinates.
    blocks = [cl[bulk_c], cl[pool_c], wf[bulk_q], wf[pool_q]]
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=seed).fit_transform(np.vstack(blocks))
    cuts = np.cumsum([0] + [len(b) for b in blocks])
    e_bulk_c, e_pool_c, e_bulk_q, e_pool_q = (emb[cuts[i]:cuts[i + 1]]
                                              for i in range(4))
    cmap = plt.get_cmap("tab20")

    i_k = int(np.argmin(np.abs(k - tail_k)))
    n_rej = min(int(round(BUDGET * a_cl[i_k])), len(pool_c))
    n_amp = min(int(round(BUDGET / preps_q[i_k])), len(pool_q))
    print(f"[harvest] Q={BUDGET} preparations at {tail_k:g} sigma:  "
          f"rejection={n_rej} on the flow-ODE state (p={a_cl[i_k]:.5f})  "
          f"amplification={n_amp} on the wavefunction-flow state "
          f"(p={a_wf[i_k]:.5f}, cost {preps_q[i_k]:.2f})  "
          f"({n_amp / max(n_rej, 1):.2f}x)", flush=True)

    def _panel(ax, e_core, near_all, bulk, e_pool, idx, title, ring):
        ax.scatter(e_core[:, 0], e_core[:, 1], s=30, c=cmap(near_all[bulk] % 20),
                   alpha=0.16, linewidths=0)
        # Ring carries the method, matching that pipeline's color in (a); white fill so
        # the marker reads even where a bulk cluster shares the ring's hue.
        ax.scatter(e_pool[idx, 0], e_pool[idx, 1], s=58, facecolors="white",
                   edgecolors=ring, linewidths=1.8, zorder=4)
        ax.set_title(title, fontsize=FS_TITLE, color="0.25", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    _panel(ax_rej, e_bulk_c, near_c, bulk_c, e_pool_c,
           rng.choice(len(pool_c), n_rej, replace=False),
           "Rejection Sampling", C_C)
    _panel(ax_amp, e_bulk_q, near_q, bulk_q, e_pool_q,
           rng.choice(len(pool_q), n_amp, replace=False),
           "Amplitude Amplification", C_Q)
    _axtag(ax_rej, "(b)")
    _axtag(ax_amp, "(c)")
    # Square window with equal data aspect: a t-SNE map is isotropic, so filling a
    # wide box would distort the geometry. Shared frame, so density compares.
    cx = 0.5 * (emb[:, 0].min() + emb[:, 0].max())
    cy = 0.5 * (emb[:, 1].min() + emb[:, 1].max())
    half = 0.5 * max(np.ptp(emb[:, 0]), np.ptp(emb[:, 1])) + 2.0
    for ax in (ax_rej, ax_amp):
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_box_aspect(1)

    fig.legend(handles=[
        Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="none", ms=12,
               label="modes (bulk)"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec=C_C, mew=1.8,
               ms=10, label="rare sample (rejection)"),
        Line2D([0], [0], marker="o", ls="", mfc="white", mec=C_Q, mew=1.8,
               ms=10, label="rare sample (amplification)")],
        loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=FS_LEGEND,
        framealpha=0.9)

    # Regression guard for exactly the defect LEGEND_HEADROOM exists to prevent.
    # constrained_layout sizes the legend only at draw time, so the clearance
    # cannot be reasoned about from font sizes alone -- measure it. Counts both
    # the measured markers and the fitted model lines; any hit means the
    # headroom is too small and the legend is sitting on the data again.
    _series = [(a_cl, preps_cl), (a_wf, preps_q),
               (xc, c_cl / xc), (xq, c_q / np.sqrt(xq))]
    _top, _clear = _fit_legend_clearance(
        fig, ax_cost, leg_cost, _series, ys.min() / 1.12,
        ys.max() * LEGEND_HEADROOM)
    print(f"[legend] panel (a) ymax {_top:.1f} "
          f"({_top / ys.max():.2f}x the highest cost); "
          f"{'clear of the data' if _clear else 'STILL OVERLAPPING'}", flush=True)

    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig_rare_event_advantage.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        paths.append(p)
        print(f"[save] {p}")
    plt.close(fig)
    return dict(c_cl=c_cl, c_q=c_q, n_rej=n_rej, n_amp=n_amp)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["paper"], default="paper",
                    help="WF state source: the paper's Table-2 K=160 MPS cores")
    ap.add_argument("--tail-k", type=float, default=DEFAULT_TAIL_K)
    ap.add_argument("--n-samples", type=int, default=N_WF,
                    help="pooled Born samples per pipeline (default 40000)")
    ap.add_argument("--seed", type=int, default=SEED)
    # Without this the figure could only be written into this repo's figures/,
    # so every render left modified tracked files behind. Its six sibling
    # generators all take an output path; this one now does too.
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="directory to write the figure into (default: repo figures/)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.perf_counter()

    wf, target, sw = load_wf_samples(args.source, args.seed, args.n_samples)
    centers = gm_mode_centers()
    cl = classical_gradient_flow(args.n_samples, seed=args.seed)

    tgt_lab = label_tails(target, centers, args.tail_k)
    wf_lab = label_tails(wf, centers, args.tail_k)
    a_star = float(chi2.sf(args.tail_k ** 2, df=D))
    a_wf, a_tgt = float(wf_lab[2].mean()), float(tgt_lab[2].mean())
    a_cl = float(label_tails(cl, centers, args.tail_k)[2].mean())

    print(f"\n=== rare-event amplitudes (tail_k={args.tail_k:.2f} sigma) ===")
    print(f"  a*    (analytic P(chi_{D}>{args.tail_k:g}))         = {_pct(a_star)}")
    print(f"  a_tgt (target GMM, genuine tail)             = {_pct(a_tgt)}")
    print(f"  a_wf  (V-MPS 2TDVP, genuine tail)            = {_pct(a_wf)}")
    print(f"  a_cl  (gradient flow, same potential)        = {_pct(a_cl)}")
    print(f"  residual (untransported): target={_pct(tgt_lab[3].mean())}  "
          f"WF={_pct(wf_lab[3].mean())}")
    print(f"  SW^wf_T: {sw['sw']:.4f} on n={sw['n_sw']}")
    print(f"  tail over-population: a_wf/a* = {a_wf / a_star:.2f}x "
          f"(target/a* = {a_tgt / a_star:.2f}x)")

    print("\n=== cost per accepted rare sample ===", flush=True)
    s = sweep_thresholds(wf, cl, centers)
    fitinfo = make_master_figure(s, wf, cl, centers, args.out_dir, args.seed,
                                 args.tail_k)

    i_k = int(np.argmin(np.abs(s["k"] - args.tail_k)))
    lift = s["preps_cl"][i_k] / s["preps_q"][i_k]
    print(f"\n  lift at {args.tail_k:g} sigma: {lift:.2f}x  "
          f"(rejection {s['preps_cl'][i_k]:.1f} vs amplification "
          f"{s['preps_q'][i_k]:.2f} preparations per rare sample)")
    for kk in (5.0, 6.0):
        j = int(np.argmin(np.abs(s["k"] - kk)))
        if abs(s["k"][j] - kk) < 1e-9:
            print(f"  lift at {kk:g} sigma: "
                  f"{s['preps_cl'][j] / s['preps_q'][j]:.2f}x")

    npz_path = os.path.join(args.out_dir, "rare_event_advantage.npz")
    np.savez(
        npz_path,
        tail_k=args.tail_k, sigma=SIGMA, scale=SCALE, d=D, L=L, source=args.source,
        n_samples=args.n_samples, budget=BUDGET,
        a_star=a_star, a_tgt=a_tgt, a_wf=a_wf, a_cl=a_cl,
        sw=sw["sw"], n_sw=sw["n_sw"],
        paper_unbiased_mean=sw["paper_unbiased_mean"],
        target=target.astype(np.float32), wf=wf.astype(np.float32),
        cl=cl.astype(np.float32),
        target_tail=tgt_lab[2], wf_tail=wf_lab[2],
        target_residual=tgt_lab[3], wf_residual=wf_lab[3],
        target_nearest=tgt_lab[0], wf_nearest=wf_lab[0],
        amp_prefactor_theory=AMP_PREFACTOR,
        fit_prefactor_rejection=fitinfo["c_cl"], fit_prefactor_amp=fitinfo["c_q"],
        harvest_rejection=fitinfo["n_rej"], harvest_amplification=fitinfo["n_amp"],
        **{f"sweep_{name}": s[name] for name in s},
    )
    print(f"\n[save] {npz_path}")
    print(f"[done] total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
