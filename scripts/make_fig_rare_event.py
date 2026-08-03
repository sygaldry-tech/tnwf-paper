"""End-to-end quadratic advantage of wavefunction flows on d=8 GMM rare-event tails.

Pipeline
--------
1. State prep = the paper's **V-MPS 2TDVP** flow on the d=8 orthogonal Gaussian mixture
   (16 modes at +-3.e_j, sigma=0.5). We use the paper's *actual* Table-2 runs
   (N=32, K=160, D_max=64, self-limiting bond chi*~=16; SW^wf_T ~= 0.036 unbiased),
   shipped under `examples/rare_event/rep*.npz` (the final MPS cores; converted from the
   original pickles so this repository ships no `pickle` payloads),
   and draw Born samples from those cores with `tnwf.pipelines.run_evolution.sample_from_mps`.
   Samples are
   pooled across the independent MPS-V initialisations (reps) the paper averages over.
   [--source npz falls back to the older, reduced K=40/D_max=16 trajectory npz.]

   Note on SW: the paper reports the **unbiased** sliced-Wasserstein, i.e. with the
   half-grid-cell (dx/2) sampling offset removed (`SW(x - dx/2, target)`); the raw/biased
   SW is ~0.11. We report both. K=40 vs K=160 barely changes the biased SW (the flow is
   Trotter-converged); the paper's low number is the unbiased estimator + D_max=64 headroom.

2. Rare-event tails: standard multivariate 3-sigma rule -- a sample is a tail event for its
   nearest mode k iff ||x - c_k|| > TAIL_K * sigma (Mahalanobis radius, isotropic). Samples
   closer to the domain centre than to any mode are flagged as **residual** (untransported
   source mass) and excluded from the tail amplitude.

3. Estimate the tail probability with the **MLQAE algorithm** (Suzuki 2020, `tnwf.qae`)
   vs a classical Monte-Carlo baseline: Heisenberg slope -1 (MLQAE) vs -1/2 (MC).

Deliverables (figures/)
    fig_rare_event_advantage.{png,pdf}  paper Fig. 8, exactly as included by the
                                        manuscript.
    rare_event_qae.npz                  raw arrays + amplitudes + SW + slopes;
                                        backs the caption's slopes and counts.

Run:
    make fig-rare-event
    uv run python scripts/make_fig_rare_event.py [--tail-k 4.0] [--source paper|npz]
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
from tnwf.data.gaussian_mixture import sample_gaussian_mixture  # noqa: E402,I001
from tnwf.pipelines.run_evolution import sample_from_mps  # noqa: E402
from tnwf.qae import classical_mc_estimate, mlqae_estimate  # noqa: E402

# isort cannot merge these into the block above: `matplotlib.use("Agg")` must run
# after `import matplotlib` but before `pyplot` is imported, so the backend is set
# headlessly. Hence the I001 exemption rather than a reordering.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def sample_wf_coords_from_mps(mps, n, N, d, L, rng):
    """Born samples in the CENTRED physical frame [-L/2, L/2]^d.

    `tnwf.pipelines.run_evolution.sample_from_mps` returns grid-frame coordinates
    in [0, L)^d -- see the docstrings of `tnwf.grid.make_grid` ("[0, L)^d") and
    `sample_from_psi_grid` ("in [0, L)^d"). This module, like `gm_mode_centers`
    below, works in the centred world frame, so the L/2 offset must be removed.
    That is the same conversion the `--source npz` branch applies to
    `samples_T` ("npz frame is [0, L); recentre"), and it matters: without it
    every sample is displaced by L/2 and the tail labelling silently breaks.
    """
    return sample_from_mps(mps, N=N, d=d, L=L, n=n, rng=rng) - L / 2.0


# ============================================================================
# Config
# ============================================================================
OUT_DIR = os.path.join(_PROJECT_ROOT, "figures")
PAPER_DIR = os.path.join(_PROJECT_ROOT, "examples/rare_event")  # Table-2 K=160 MPS cores
NPZ_FALLBACK = os.path.join(PAPER_DIR, "fallback_K40_D16.npz")

D = 8
N = 32                 # spatial grid (paper config); dx = L/N
L = 8.0                # domain edge; centred physical frame is [-L/2, L/2]^d
DX = L / N
SIGMA = 0.5            # GMM component std
SCALE = 3.0            # mode separation (centres at +-SCALE on each axis)
DEFAULT_TAIL_K = 4.0   # rare-event threshold in units of sigma (the paper's value)
N_WF = 4000            # total pooled WF Born samples
N_TARGET = 4000        # target GMM samples
SEED = 0

# MLQAE vs MC sweep. Fixed shots-per-k, sweep the max Grover power M = 0..7 so the
# MLQAE error keeps its clean 1/2^M (Heisenberg) scaling while the smallest budgets
# (M=0,1 -> Q ~ 400,900) reach down to the MLQAE/MC crossover -- measured, not fit.
N_SHOTS_PER_K = 100
BUDGETS = [(M, N_SHOTS_PER_K) for M in range(8)]   # M = 0..7
ASYMPTOTIC_Q = 1500.0   # fit the -1 / -1/2 slopes on points above this
N_TRIALS = 201          # odd; more trials -> smoother medians at small budgets


# ============================================================================
# Geometry
# ============================================================================
def _pct(x: float) -> str:
    """Format a fraction as a percentage with 0.01% resolution."""
    return f"{100.0 * x:.2f}%"


def gm_mode_centers() -> np.ndarray:
    """Orthogonal centres at +-SCALE.e_j (centred physical frame)."""
    C = np.zeros((2 * D, D))
    for j in range(D):
        C[2 * j, j] = SCALE
        C[2 * j + 1, j] = -SCALE
    return C


def sliced_wasserstein(a, b, n_proj=200, seed=0) -> float:
    """Paper's SW estimator (analysis/gmm_phase_rank_control.py)."""
    rng = np.random.default_rng(seed)
    projs = rng.normal(size=(n_proj, a.shape[1]))
    projs /= np.linalg.norm(projs, axis=1, keepdims=True)
    a_s = np.sort(a @ projs.T, axis=0)
    b_s = np.sort(b @ projs.T, axis=0)
    if a_s.shape[0] != b_s.shape[0]:
        idx = np.linspace(0, b_s.shape[0] - 1, a_s.shape[0]).astype(int)
        b_s = b_s[idx]
    return float(np.mean(np.abs(a_s - b_s)))


# ============================================================================
# Stage 1 -- WF Born samples from the paper's V-MPS 2TDVP state
# ============================================================================
def load_wf_samples(source: str, seed: int):
    """Return (wf, target, sw_stats) in the centred [-L/2, L/2]^d frame.

    wf is already half-grid-cell corrected (x - dx/2), matching the paper's unbiased SW.
    """
    rng = np.random.default_rng(seed)
    target = sample_gaussian_mixture(
        N_TARGET, d=D, std=SIGMA, scale=SCALE, arrangement="orthogonal",
        seed=1).astype(np.float64)

    if source == "paper":
        reps = sorted(glob.glob(os.path.join(PAPER_DIR, "rep*.npz")))
        if not reps:
            raise FileNotFoundError(
                f"No Table-2 MPS cores under {PAPER_DIR}; expected rep*.npz. "
                "Use --source npz for the reduced K=40/D_max=16 fallback.")
        per = int(np.ceil(N_WF / len(reps)))
        chunks, stored_unb = [], []
        for d_ in reps:
            z = np.load(d_)
            mps = [z[f"core{j}"] for j in range(int(z["n_cores"]))]
            x = sample_wf_coords_from_mps(mps, per, N, D, L, rng)  # centred
            chunks.append(x)
            stored_unb.append(float(z["sw_unbiased_final"]))
        wf_raw = np.vstack(chunks)[:N_WF]
        rng.shuffle(wf_raw)
        wf = wf_raw - DX / 2.0   # half-cell correction (matches paper unbiased SW)
        sw_biased = sliced_wasserstein(wf_raw, target)
        sw_unbiased = sliced_wasserstein(wf, target)
        print(f"[load] paper Table-2 state: pooled {wf.shape[0]} Born samples over "
              f"{len(reps)} MPS-V reps (K=160, N={N}, D_max=64)", flush=True)
        print(f"       SW biased={sw_biased:.4f}  SW unbiased={sw_unbiased:.4f}  "
              f"(paper Table-2 per-rep unbiased mean={np.mean(stored_unb):.4f})",
              flush=True)
        sw_stats = dict(sw_biased=sw_biased, sw_unbiased=sw_unbiased,
                        paper_unbiased_mean=float(np.mean(stored_unb)),
                        paper_unbiased_ci=float(1.96 * np.std(stored_unb)
                                                / np.sqrt(len(stored_unb))),
                        n_reps=len(reps), K=160, D_max=64)
    elif source == "npz":
        d = np.load(NPZ_FALLBACK)
        # npz frame is [0, L); recentre and half-cell correct
        wf_raw = d["samples_T"].astype(np.float64) - L / 2.0
        wf = wf_raw - DX / 2.0
        sw_biased = sliced_wasserstein(wf_raw, target)
        sw_unbiased = sliced_wasserstein(wf, target)
        print(f"[load] fallback npz (reduced K=40/D_max=16): {wf.shape[0]} samples; "
              f"SW biased={sw_biased:.4f} unbiased={sw_unbiased:.4f}", flush=True)
        sw_stats = dict(sw_biased=sw_biased, sw_unbiased=sw_unbiased,
                        paper_unbiased_mean=float("nan"),
                        paper_unbiased_ci=float("nan"), n_reps=1, K=40, D_max=16)
    else:
        raise ValueError(f"unknown source {source!r}")
    return wf, target, sw_stats


# ============================================================================
# Stage 2 -- rare-event (k-sigma) tail labelling, residual separation
# ============================================================================
def label_tails(samples, centers, tail_k):
    """Nearest-mode radial k-sigma rule with residual-mass separation.

    residual  = closer to the domain centre (origin) than to any mode.
    rare tail = non-residual and nearest-mode distance > tail_k*sigma.
    Returns (nearest_mode, nearest_dist, tail_mask, residual_mask).
    """
    dist = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=-1)
    nearest = dist.argmin(axis=1)
    nearest_dist = dist.min(axis=1)
    dist_center = np.linalg.norm(samples, axis=1)   # centred frame: origin
    residual = dist_center < nearest_dist
    tail = (nearest_dist > tail_k * SIGMA) & (~residual)
    return nearest, nearest_dist, tail, residual


# ============================================================================
# Stage 4/5 -- MLQAE vs classical MC sweep
# ============================================================================
def _slope(xs, ys) -> float:
    mask = (xs > 0) & (ys > 0)
    s, _ = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), deg=1)
    return float(s)


def _empirical_crossover(q, em, ec):
    """Interpolate (in log-log) the Q where measured MLQAE error drops below MC.

    Returns (Qx, ex) or (None, None) if MLQAE already wins at the smallest budget.
    """
    d = np.log(em) - np.log(ec)     # >0 : MLQAE worse than MC
    for i in range(len(q) - 1):
        if d[i] > 0.0 >= d[i + 1]:
            t = d[i] / (d[i] - d[i + 1])
            lq = np.log(q[i]) + t * (np.log(q[i + 1]) - np.log(q[i]))
            le = np.log(em[i]) + t * (np.log(em[i + 1]) - np.log(em[i]))
            return float(np.exp(lq)), float(np.exp(le))
    return None, None


def mlqae_vs_mc_sweep(a_true, master_rng) -> dict:
    qs, mlqae_err, mc_err = [], [], []
    ml_lo, ml_hi, mc_lo, mc_hi = [], [], [], []   # 25/75 percentiles per budget
    for (M, n_shots) in BUDGETS:
        q_trials, e_trials = [], []
        for _ in range(N_TRIALS):
            rng = np.random.default_rng(master_rng.integers(2 ** 31))
            res = mlqae_estimate(a_true=a_true, M=M,
                                 n_shots_per_k=n_shots, rng=rng)
            q_trials.append(res.total_grover_queries)
            e_trials.append(abs(res.a_hat - a_true))
        q_med = float(np.median(q_trials))
        qs.append(q_med)
        mlqae_err.append(float(np.median(e_trials)))
        elo, ehi = np.percentile(e_trials, [25, 75])
        ml_lo.append(float(elo)); ml_hi.append(float(ehi))
        mc_trials = []
        for _ in range(N_TRIALS):
            rng = np.random.default_rng(master_rng.integers(2 ** 31))
            ahat, _n = classical_mc_estimate(a_true, n_samples=max(int(q_med), 1),
                                             rng=rng)
            mc_trials.append(abs(ahat - a_true))
        mc_err.append(float(np.median(mc_trials)))
        clo, chih = np.percentile(mc_trials, [25, 75])
        mc_lo.append(float(clo)); mc_hi.append(float(chih))
    order = np.argsort(qs)
    _o = lambda x: np.array(x)[order]  # noqa: E731
    qs = np.array(qs)[order]
    mlqae_err, mc_err = _o(mlqae_err), _o(mc_err)
    ml_lo, ml_hi, mc_lo, mc_hi = _o(ml_lo), _o(ml_hi), _o(mc_lo), _o(mc_hi)
    # slopes on the asymptotic tail (large Q) only
    asy = qs >= ASYMPTOTIC_Q
    Qx, ex = _empirical_crossover(qs, mlqae_err, mc_err)
    return dict(a_true=a_true, queries=qs, mlqae_err=mlqae_err, mc_err=mc_err,
                mlqae_lo=ml_lo, mlqae_hi=ml_hi, mc_lo=mc_lo, mc_hi=mc_hi,
                mlqae_slope=_slope(qs[asy], mlqae_err[asy]),
                mc_slope=_slope(qs[asy], mc_err[asy]),
                crossover_Q=Qx, crossover_err=ex,
                final_speedup=float(mc_err[-1] / max(mlqae_err[-1], 1e-12)))


# ============================================================================
# Rare-event SAMPLING advantage: amplitude amplification vs rejection sampling
# ============================================================================
def _amp_params(a):
    """Optimal Grover power k*, its success prob P, for good-subspace amplitude a."""
    theta = float(np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
    if theta <= 0.0:
        return 0, 0.0
    k = max(int(round(np.pi / (4.0 * theta) - 0.5)), 0)
    P = float(np.sin((2 * k + 1) * theta) ** 2)
    return k, min(max(P, 1e-9), 1.0)


# ============================================================================
# Master 4-panel figure
# ============================================================================
def _axtag(ax, txt, inside=False):
    if inside:
        ax.text(0.015, 0.975, txt, transform=ax.transAxes, fontsize=25,
                fontweight="bold", va="top", ha="left")
    else:
        ax.text(-0.02, 1.03, txt, transform=ax.transAxes, fontsize=25,
                fontweight="bold", va="bottom", ha="right")


def _draw_estimation(ax, s):
    c_q, c_c = "#3b0f70", "#8c8c8c"
    q, em, ec = s["queries"], s["mlqae_err"], s["mc_err"]
    ax.fill_between(q, s["mlqae_lo"], s["mlqae_hi"], color=c_q, alpha=0.15,
                    linewidth=0, zorder=1)
    ax.fill_between(q, s["mc_lo"], s["mc_hi"], color=c_c, alpha=0.15,
                    linewidth=0, zorder=1)
    ax.loglog(q, em, "o-", color=c_q, lw=2.8, ms=9,
              label=f"MLQAE  ({s['mlqae_slope']:+.2f})")
    ax.loglog(q, ec, "s--", color=c_c, lw=2.4, ms=8, mfc="white",
              label=f"Monte Carlo  ({s['mc_slope']:+.2f})")
    qg = np.array([q[0], q[-1]])
    ax.loglog(qg, em[0] * (qg / q[0]) ** -1.0, "-", color=c_q, lw=1, alpha=0.3)
    ax.loglog(qg, ec[0] * (qg / q[0]) ** -0.5, "--", color=c_c, lw=1, alpha=0.4)
    ax.set_xlabel(r"oracle queries  $Q$", fontsize=21)
    ax.set_ylabel(r"error  $|\hat a - a|$", fontsize=21)
    ax.tick_params(labelsize=17)
    ax.grid(True, which="both", alpha=0.3)
    # tight limits around the data + IQR bands (guide lines may clip)
    ax.set_xlim(q.min() / 1.15, q.max() * 1.15)
    ylo = min(float(s["mlqae_lo"].min()), float(s["mc_lo"].min()))
    yhi = max(float(s["mlqae_hi"].max()), float(s["mc_hi"].max()))
    ax.set_ylim(ylo / 1.4, yhi * 1.4)
    ax.legend(fontsize=18, loc="lower left")


def _draw_amp_cost(ax, seed):
    rng = np.random.default_rng(seed); N_RARE, TRIALS = 200, 61
    sigmas = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    ps = chi2.sf(sigmas ** 2, df=D)
    cost_cl, cost_qu = [], []
    cl_lo, cl_hi, qu_lo, qu_hi = [], [], [], []   # 25/75 percentiles per rarity
    for a in ps:
        k, P = _amp_params(a); cost = 2 * k + 1
        cl = (rng.negative_binomial(N_RARE, min(max(float(a), 1e-12), 1.0),
                                    size=TRIALS) + N_RARE) / N_RARE
        qu = ((rng.negative_binomial(N_RARE, P, size=TRIALS) + N_RARE) * cost) / N_RARE
        cost_cl.append(np.median(cl)); cost_qu.append(np.median(qu))
        c1, c2 = np.percentile(cl, [25, 75]); q1, q2 = np.percentile(qu, [25, 75])
        cl_lo.append(c1); cl_hi.append(c2); qu_lo.append(q1); qu_hi.append(q2)
    cost_cl = np.array(cost_cl); cost_qu = np.array(cost_qu)
    cl_lo, cl_hi = np.array(cl_lo), np.array(cl_hi)
    qu_lo, qu_hi = np.array(qu_lo), np.array(qu_hi)
    c_q, c_c = "#3b0f70", "#8c8c8c"
    ax.fill_between(ps, cl_lo, cl_hi, color=c_c, alpha=0.15, linewidth=0, zorder=1)
    ax.fill_between(ps, qu_lo, qu_hi, color=c_q, alpha=0.15, linewidth=0, zorder=1)
    ax.loglog(ps, cost_cl, "s--", color=c_c, lw=2.4, ms=9, mfc="white",
              label="rejection sampling")
    ax.loglog(ps, cost_qu, "o-", color=c_q, lw=2.8, ms=9,
              label="amplitude amplification")
    ax.loglog(ps, cost_cl[0] * (ps / ps[0]) ** -1.0, "-", color=c_c, lw=1, alpha=0.35)
    ax.loglog(ps, cost_qu[0] * (ps / ps[0]) ** -0.5, "-", color=c_q, lw=1, alpha=0.35)
    ax.set_xlabel(r"rare-event probability  $p$", fontsize=21)
    ax.set_ylabel("oracle queries / rare sample", fontsize=21)
    ax.tick_params(labelsize=17)
    ax.grid(True, which="both", alpha=0.3)
    # tight limits around the data + IQR bands (standard axis: small p on the left)
    ymin = min(float(qu_lo.min()), float(cl_lo.min()))
    ymax = max(float(qu_hi.max()), float(cl_hi.max()))
    ax.set_ylim(ymin / 1.4, ymax * 1.4)
    ax.set_xlim(float(ps.min()) / 1.6, float(ps.max()) * 1.6)
    ax.legend(fontsize=17, loc="upper right")


def _draw_sampling_tsne(ax_l, ax_r, seed, tail_k, budget=1000):
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    C = gm_mode_centers()

    def nearest(x):
        d = np.linalg.norm(x[:, None, :] - C[None], axis=-1)
        return d.argmin(1), d.min(1)

    core = sample_gaussian_mixture(1600, d=D, std=SIGMA, scale=SCALE,
                                   arrangement="orthogonal", seed=7).astype(float)
    core_near, _ = nearest(core)
    big = sample_gaussian_mixture(60000, d=D, std=SIGMA, scale=SCALE,
                                  arrangement="orthogonal", seed=11).astype(float)
    bn, bd = nearest(big)
    tmask = bd > tail_k * SIGMA
    pool = big[tmask][:600]; pool_near = bn[tmask][:600]
    a = float(chi2.sf(tail_k ** 2, D)); k, P = _amp_params(a); cost = 2 * k + 1
    n_rej = min(int(round(budget * a)), len(pool))
    n_amp = min(int(round(budget / cost * P)), len(pool))
    combo = np.vstack([core, pool])
    print(f"[master] embedding {combo.shape[0]} points...", flush=True)
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=seed).fit_transform(combo)
    e_core, e_pool = emb[:len(core)], emb[len(core):]
    cmap = plt.get_cmap("tab20")
    idx_rej = rng.choice(len(pool), n_rej, replace=False)
    idx_amp = rng.choice(len(pool), n_amp, replace=False)

    def _panel(ax, idx):
        ax.scatter(e_core[:, 0], e_core[:, 1], s=30, c=cmap(core_near % 20),
                   alpha=0.16, linewidths=0)
        ax.scatter(e_pool[idx, 0], e_pool[idx, 1], s=52, c=cmap(pool_near[idx] % 20),
                   alpha=1.0, edgecolors="black", linewidths=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.02)

    _panel(ax_l, idx_rej)
    _panel(ax_r, idx_amp)
    return n_rej, n_amp


def make_master_figure(sweeps, out_dir, seed=0, tail_k=4.0):
    fig = plt.figure(figsize=(14.0, 13.4), dpi=160, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.12])
    ax_est = fig.add_subplot(gs[0, 0])
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_rej = fig.add_subplot(gs[1, 0])
    ax_aa = fig.add_subplot(gs[1, 1])

    _draw_estimation(ax_est, sweeps["analytic a*"]); _axtag(ax_est, "(a)")
    _draw_amp_cost(ax_amp, seed); _axtag(ax_amp, "(b)")
    _draw_sampling_tsne(ax_rej, ax_aa, seed, tail_k); _axtag(ax_rej, "(c)", inside=True)
    _axtag(ax_aa, "(d)", inside=True)

    legend = [
        Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="none", ms=12,
               label="modes (bulk)"),
        Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="black", ms=9,
               label=f"rare sample (>{tail_k:g}σ)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.09),
               ncol=2, fontsize=19, framealpha=0.9)
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig_rare_event_advantage.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"[master] saved {p}", flush=True)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["paper", "npz"], default="paper",
                    help="WF state source: paper Table-2 K=160 (default) or reduced K=40 npz")
    ap.add_argument("--tail-k", type=float, default=DEFAULT_TAIL_K)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    wf, target, sw = load_wf_samples(args.source, args.seed)
    centers = gm_mode_centers()

    tgt_lab = label_tails(target, centers, args.tail_k)
    wf_lab = label_tails(wf, centers, args.tail_k)
    tgt_tail, tgt_res = tgt_lab[2], tgt_lab[3]
    wf_tail, wf_res = wf_lab[2], wf_lab[3]

    a_star = float(chi2.sf(args.tail_k ** 2, df=D))
    a_wf = float(wf_tail.mean())
    a_tgt = float(tgt_tail.mean())
    print("\n=== rare-event amplitudes (tail_k=%.2f sigma) ===" % args.tail_k)
    print(f"  a*    (analytic P(chi_{D}>{args.tail_k:g}))         = {_pct(a_star)}")
    print(f"  a_tgt (target GMM, genuine tail)             = {_pct(a_tgt)}  (n={target.shape[0]})")
    print(f"  a_wf  (V-MPS 2TDVP, genuine tail)            = {_pct(a_wf)}  (n={wf.shape[0]})")
    print(f"  residual (untransported): target={_pct(tgt_res.mean())}  WF={_pct(wf_res.mean())}")
    print(f"  SW^wf_T: biased={sw['sw_biased']:.4f}  unbiased={sw['sw_unbiased']:.4f}"
          + (f"  (paper Table-2: {sw['paper_unbiased_mean']:.4f} ± "
             f"{sw['paper_unbiased_ci']:.4f}, n={sw['n_reps']})"
             if sw['n_reps'] > 1 else ""))

    K = centers.shape[0]
    print(f"  per-mode rare counts (target): {np.bincount(tgt_lab[0][tgt_tail], minlength=K).tolist()}")
    print(f"  per-mode rare counts (WF):     {np.bincount(wf_lab[0][wf_tail], minlength=K).tolist()}")

    master_rng = np.random.default_rng(args.seed)
    print("\n=== MLQAE vs classical MC sweep ===", flush=True)
    sweeps = {}
    for label, a in [("analytic a*", a_star), ("WF a_wf", a_wf)]:
        s = mlqae_vs_mc_sweep(a, master_rng)
        sweeps[label] = s
        print(f"  [{label}] a={a:.4f}  MLQAE slope={s['mlqae_slope']:+.3f}  "
              f"MC slope={s['mc_slope']:+.3f}  final speedup={s['final_speedup']:.1f}x",
              flush=True)

    make_master_figure(sweeps, OUT_DIR, args.seed, args.tail_k)

    npz_path = os.path.join(OUT_DIR, "rare_event_qae.npz")
    np.savez(
        npz_path,
        tail_k=args.tail_k, sigma=SIGMA, scale=SCALE, d=D, L=L, source=args.source,
        a_star=a_star, a_tgt=a_tgt, a_wf=a_wf,
        sw_biased=sw["sw_biased"], sw_unbiased=sw["sw_unbiased"],
        paper_unbiased_mean=sw["paper_unbiased_mean"],
        target=target.astype(np.float32), wf=wf.astype(np.float32),
        target_tail=tgt_tail, wf_tail=wf_tail,
        target_residual=tgt_res, wf_residual=wf_res,
        target_nearest=tgt_lab[0], wf_nearest=wf_lab[0],
        budgets=np.array(BUDGETS), N_TRIALS=N_TRIALS,
        astar_crossover_Q=sweeps["analytic a*"]["crossover_Q"] or np.nan,
        awf_crossover_Q=sweeps["WF a_wf"]["crossover_Q"] or np.nan,
        **{f"{'astar' if 'a*' in k else 'awf'}_{fld}": sweeps[k][fld]
           for k in sweeps for fld in ("queries", "mlqae_err", "mc_err",
                                       "mlqae_slope", "mc_slope")},
    )
    print(f"\n[save] {npz_path}")

    print("\n" + "=" * 72)
    print(f"{'amplitude':>14s}  {'a':>7s}  {'MLQAE slope':>12s}  {'MC slope':>9s}  "
          f"{'final speedup':>14s}")
    print("-" * 72)
    for k, s in sweeps.items():
        print(f"{k:>14s}  {s['a_true']:>7.4f}  {s['mlqae_slope']:>+12.3f}  "
              f"{s['mc_slope']:>+9.3f}  {s['final_speedup']:>13.1f}x")
    print("=" * 72)
    print(f"[done] total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
