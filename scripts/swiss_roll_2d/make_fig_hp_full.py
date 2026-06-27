"""HP sweep visualisations — 1D marginals + 2D pairwise heatmaps + Pareto.

Per method:
  1. Fit a GP on (log2 N, log2 K, log2 D_max) → log SW (and log time).
  2. Plot 1D marginal means (with 95% CI bands) — overlaid across methods.
  3. Plot 2D pairwise heatmaps at the median value of the held-out axis.
  4. Plot SW vs runtime Pareto frontier.

Reads from data/{dataset}_hp/{method}/N{N}_K{K}_D{D}/seed*.npz.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

METHODS = [
    "jam", "dense", "tci_als", "aci",
    "tci_tdvp1", "tci_tdvp2",
]
METHOD_LABEL = {
    "jam":        "JAM",
    "dense":      "Dense",
    "tci_als":    "TCI+ALS",
    "aci":        "ACI",
    "tci_tdvp1":  "TCI+TDVP1",
    "tci_tdvp2":  "TCI+TDVP2",
}
METHOD_COLORS = {
    "jam":        "tab:gray",
    "dense":      "black",
    "tci_als":    "tab:blue",
    "aci":        "tab:orange",
    "tci_tdvp1":  "tab:green",
    "tci_tdvp2":  "tab:red",
}
METHOD_MARKERS = {
    "jam":        "v",
    "dense":      "P",
    "tci_als":    "s",
    "aci":        "D",
    "tci_tdvp1":  "^",
    "tci_tdvp2":  "o",
}
NK_D_PATTERN = re.compile(r"^N(\d+)_K(\d+)_D(\d+)$")


def _safe_load(path: Path) -> dict | None:
    import zipfile
    try:
        z = np.load(path, allow_pickle=True)
        rec = {"sw": float(z["sw"][-1]), "mmd": float(z["mmd"][-1])}
        if "total_time" in z.files:
            rec["time"] = float(np.asarray(z["total_time"]))
        else:
            rec["time"] = float("nan")
        chi_arr = np.asarray(z["chi_max"]) if "chi_max" in z.files else np.array([0])
        rec["chi_max"] = int(chi_arr.max())
        rec["d"] = int(np.asarray(z["d"])) if "d" in z.files else 2
        return rec
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as e:
        print(f"[skip {path}] {type(e).__name__}: {e}")
        return None


def _mps_param_count(N: float, chi: float, d: int) -> float:
    """Approx parameter count of a length-d MPS with physical dim N and
    uniform bond χ. Edge cores have one trivial bond; interior cores
    have shape (χ, N, χ). Total ≈ 2·N·χ + (d-2)·N·χ² for d ≥ 2.
    """
    return 2.0 * N * chi + max(d - 2, 0) * N * chi ** 2


def _compression_factor(N: float, chi: float, d: int) -> float:
    """Dense state size / MPS state size. Ratio > 1 means MPS uses fewer
    params than dense (compressed); ratio < 1 means MPS is heavier than
    dense (only happens at d=2 with χ near N)."""
    return (N ** d) / max(_mps_param_count(N, chi, d), 1.0)


def collect(results_dir: Path, method: str) -> dict:
    """Return dict with X (M×3 log2 inputs), sw, mmd, time, D_max, chi_obs,
    N_obs, d. Multiple seeds per cell are aggregated to mean here.
    """
    method_dir = results_dir / method
    rows = []
    d_value = 2
    for sub in method_dir.iterdir():
        m = NK_D_PATTERN.match(sub.name)
        if not m:
            continue
        N, K, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sw_seeds, mmd_seeds, time_seeds, chi_seeds = [], [], [], []
        for f in sorted(sub.glob("seed*.npz")):
            rec = _safe_load(f)
            if rec is not None:
                sw_seeds.append(rec["sw"])
                mmd_seeds.append(rec["mmd"])
                time_seeds.append(rec["time"])
                chi_seeds.append(rec["chi_max"])
                d_value = rec["d"]
        if sw_seeds:
            rows.append((np.log2(N), np.log2(K), np.log2(D),
                         float(np.mean(sw_seeds)),
                         float(np.mean(mmd_seeds)),
                         float(np.nanmean(time_seeds)),
                         float(D),
                         float(np.mean(chi_seeds)),
                         float(N)))
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    return {
        "X": arr[:, :3], "sw": arr[:, 3], "mmd": arr[:, 4],
        "time": arr[:, 5], "D_max": arr[:, 6], "chi_obs": arr[:, 7],
        "N_obs": arr[:, 8], "d": d_value,
    }


def fit_gp(X: np.ndarray, y: np.ndarray, log_y: bool = True) -> GaussianProcessRegressor:
    if log_y:
        y_t = np.log(np.maximum(y, 1e-8))
    else:
        y_t = y
    kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=[1.0, 1.0, 1.0],
              length_scale_bounds=(1e-1, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=3, random_state=0)
    gp.fit(X, y_t)
    return gp


# ── Plotting helpers ───────────────────────────────────────────────────────

def plot_marginals(method_data: dict, gps: dict, ax_axes, axis_idx: int,
                   axis_name: str, log_axis: np.ndarray,
                   other_axis_grids: tuple):
    """Plot 1D marginal: vary `axis_idx`, take argmin over the other two axes.

    For each value of axis_idx, search a fine grid over the held-out axes via
    the GP and pick the minimum predicted SW. This shows each method at its
    best operating point along the swept axis — what users actually want to
    know ("how low can SW go at this N?").

    other_axis_grids: tuple (grid_axis_a, grid_axis_b) of log2 values to scan.
                      Order: skip axis_idx (so length 2).
    """
    other_idx_a, other_idx_b = sorted({0, 1, 2} - {axis_idx})
    ga, gb = other_axis_grids
    GA, GB = np.meshgrid(ga, gb, indexing="xy")
    n_inner = GA.size

    ax = ax_axes[axis_idx]
    for method, gp_sw in gps.items():
        if gp_sw is None:
            continue
        mus = np.empty(len(log_axis))
        stds = np.empty(len(log_axis))
        for i, log_v in enumerate(log_axis):
            pts = np.zeros((n_inner, 3))
            pts[:, axis_idx] = log_v
            pts[:, other_idx_a] = GA.ravel()
            pts[:, other_idx_b] = GB.ravel()
            log_mu, log_std = gp_sw.predict(pts, return_std=True)
            j = int(np.argmin(log_mu))
            mus[i] = np.exp(log_mu[j])
            stds[i] = log_std[j]                          # uncertainty at argmin
        x_ticks = 2 ** log_axis
        lo = mus * np.exp(-1.96 * stds)
        hi = mus * np.exp(+1.96 * stds)
        ax.plot(x_ticks, mus, lw=2.0, color=METHOD_COLORS[method],
                marker="o", ms=5, label=METHOD_LABEL[method])
        ax.fill_between(x_ticks, lo, hi, alpha=0.18, color=METHOD_COLORS[method])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("min SW achievable\n(GP argmin over other axes; 95% CI)",
                  fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_2d_heatmap(gp_sw, ax, axis_a: int, axis_b: int,
                    log_grid_a: np.ndarray, log_grid_b: np.ndarray,
                    fixed_axis: int, fixed_log: float,
                    method_label: str, vmin: float, vmax: float):
    """Predict GP on a 2D grid varying axes (a, b), hold third at fixed_log."""
    AA, BB = np.meshgrid(log_grid_a, log_grid_b, indexing="xy")
    pts = np.zeros((AA.size, 3))
    pts[:, axis_a] = AA.ravel()
    pts[:, axis_b] = BB.ravel()
    pts[:, fixed_axis] = fixed_log
    log_mu = gp_sw.predict(pts).reshape(AA.shape)
    mu = np.exp(log_mu)
    im = ax.imshow(mu, origin="lower", cmap="plasma_r", aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(log_grid_a)))
    ax.set_xticklabels([f"{int(2 ** v)}" for v in log_grid_a], fontsize=8)
    ax.set_yticks(range(len(log_grid_b)))
    ax.set_yticklabels([f"{int(2 ** v)}" for v in log_grid_b], fontsize=8)
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            v = mu[i, j]
            norm_v = (v - vmin) / max(vmax - vmin, 1e-12)
            text_c = "white" if norm_v > 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color=text_c)
    ax.set_title(method_label, fontsize=10, color=METHOD_COLORS.get(
        method_label.lower().replace("+", "_").replace("dense", "dense"), "black"))
    return im


def _pareto_front_min_x_min_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pareto front for: minimize x, minimize y.

    A point (x_i, y_i) is non-dominated iff no other point has both x ≤ x_i
    and y ≤ y_i (strict on at least one). Sort by x ascending; kept i iff
    y_i < min(y[:i]). This is the standard south-west Pareto front: as you
    pay more (x increases) you get better accuracy (y decreases).
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    if n == 0:
        return xs, ys
    keep = []
    cur_min = np.inf
    for xi, yi in zip(xs, ys):
        if yi < cur_min:
            keep.append((xi, yi))
            cur_min = yi
    arr = np.array(keep)
    return arr[:, 0], arr[:, 1]


def plot_pareto(method_data: dict, gps_sw: dict, ax,
                log_Ns: np.ndarray, log_Ks: np.ndarray, log_Ds: np.ndarray):
    """Per-method Pareto frontier: SW ratio (vs optimal) vs effective bond χ̄.

    Y-axis is `SW / SW_best` where `SW_best` is the lowest SW observed across
    *any* method/cell on this dataset — a practical proxy for the dataset's
    optimum. Y=1 means "you've matched the best known result"; Y>1 means
    "you're paying this multiplicative penalty to compress."

    X-axis is χ̄ (mean peak bond reached during evolution). For methods with
    no bond (Dense, JAM): plotted as horizontal dashed reference lines at
    their normalized SW.

    Note: a cleaner reference would be the *analytical* optimum on the dense
    grid (KL to ground-truth p_t). For now we use empirical SW_best.
    """
    # Reference: best SW achieved by Dense (the no-compression baseline).
    dense_d = method_data.get("dense")
    if dense_d is None or len(dense_d["sw"]) == 0:
        ax.text(0.5, 0.5, "no Dense baseline data", transform=ax.transAxes,
                ha="center")
        return
    sw_dense_best = float(np.min(dense_d["sw"]))
    print(f"  Pareto reference SW_dense_best = {sw_dense_best:.4f}")

    # x-axis: memory used by MPS, expressed as a fraction of dense's memory.
    # Smaller = more compressed; larger = MPS heavier than dense.
    d_val = next((m["d"] for m in method_data.values() if m is not None), 2)

    def memory_fraction(N, chi):
        return _mps_param_count(N, chi, d_val) / (N ** d_val)

    all_ratios = []
    for m_d in method_data.values():
        if m_d is None:
            continue
        chi = m_d["chi_obs"]
        N = m_d["N_obs"]
        mask = chi > 0
        if mask.any():
            mf = np.array([memory_fraction(N[i], chi[i])
                           for i in np.where(mask)[0]])
            all_ratios.extend(mf.tolist())
    if all_ratios:
        x_lo = min(all_ratios) * 0.7
        x_hi = max(all_ratios) * 1.3
    else:
        x_lo, x_hi = 0.01, 10

    for method in METHODS:
        m_d = method_data.get(method)
        if m_d is None:
            continue
        if method == "dense":
            continue
        chi = m_d["chi_obs"]
        N = m_d["N_obs"]
        sw_ratio = m_d["sw"] / sw_dense_best
        if np.all(chi == 0):
            best = float(np.min(sw_ratio))
            ax.axhline(best, color=METHOD_COLORS[method], lw=2.5, ls=":",
                       alpha=0.9,
                       label=f"{METHOD_LABEL[method]} (no bond) = {best:.2f}×")
            continue
        mask = chi > 0
        if not mask.any():
            continue
        mf = np.array([memory_fraction(N[i], chi[i])
                       for i in np.where(mask)[0]])
        xs, ys = _pareto_front_min_x_min_y(mf, sw_ratio[mask])
        ax.plot(xs, ys, lw=2.5, color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method], ms=12,
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor="white", markeredgewidth=1.5,
                alpha=0.55, label=METHOD_LABEL[method])

    # Dense: y=1 horizontal reference (matches accuracy)
    ax.axhline(1.0, color="black", lw=1.5, ls="--", alpha=0.7,
               label="Dense baseline (1.0×)")
    # x=1 vertical: MPS memory = dense memory
    ax.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("MPS memory / dense memory", fontsize=14)
    ax.set_ylabel("SW / SW$_{\\mathrm{Dense}}$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(
        f"Accuracy vs compression — Pareto frontier\n"
        f"(reference: best Dense SW = {sw_dense_best:.4f})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="data/swiss_roll_hp_pilot")
    p.add_argument("--out", default="results/swiss_roll_hp_pilot/fig_hp_full.pdf")
    p.add_argument("--dataset_label", default="Swiss Roll (2D)")
    args = p.parse_args()

    res_dir = Path(args.results_dir)

    method_data = {}
    gps_sw = {}
    for method in METHODS:
        d = collect(res_dir, method)
        method_data[method] = d
        if d is None or len(d["sw"]) < 5:
            print(f"[skip {method}] not enough data ({0 if d is None else len(d['sw'])} pts)")
            gps_sw[method] = None
            continue
        gps_sw[method] = fit_gp(d["X"], d["sw"], log_y=True)
        print(f"  {method}: {len(d['sw'])} cells; GP kernel = {gps_sw[method].kernel_}")

    if not any(g is not None for g in gps_sw.values()):
        raise SystemExit("No GP could be fit (insufficient data).")

    # Prediction grids — same as Sobol inputs
    Ns = [16, 32, 64, 128, 256]
    Ks = [4, 8, 16, 32, 64]
    Ds = [8, 16, 32, 64, 128]
    log_Ns = np.log2(Ns)
    log_Ks = np.log2(Ks)
    log_Ds = np.log2(Ds)
    median_log = (np.median(log_Ns), np.median(log_Ks), np.median(log_Ds))

    n_methods_visible = sum(g is not None for g in gps_sw.values())
    # Layout (top to bottom):
    #   row 0: 1D marginals — three panels spanning the full width
    #   row 1: title strip for the N×K heatmap row (centered across columns)
    #   row 2: N × K heatmaps + colorbar
    #   row 3: title strip for the N×D_max heatmap row
    #   row 4: N × D_max heatmaps + colorbar
    #   row 5: Pareto frontier (SW vs D_max)
    n_heat_cols = len(METHODS)
    fig_w = 5.0 * n_heat_cols + 1.5
    fig = plt.figure(figsize=(fig_w, 26), dpi=200)
    gs = GridSpec(
        6, n_heat_cols + 1, figure=fig, hspace=0.55, wspace=0.4,
        height_ratios=[1.1, 0.10, 1.15, 0.10, 1.15, 1.5],
        width_ratios=[1.0] * n_heat_cols + [0.07],
    )

    # ── Row 0: 1D marginals (SW vs N, K, D_max) — span the full width ───────
    third = max(1, (n_heat_cols + 1) // 3)
    ax_marg = [
        fig.add_subplot(gs[0, 0:third]),
        fig.add_subplot(gs[0, third:2 * third]),
        fig.add_subplot(gs[0, 2 * third:]),
    ]
    fine_K = np.linspace(log_Ks[0], log_Ks[-1], 25)
    fine_D = np.linspace(log_Ds[0], log_Ds[-1], 25)
    fine_N = np.linspace(log_Ns[0], log_Ns[-1], 25)
    plot_marginals(method_data, gps_sw, ax_marg, axis_idx=0,
                   axis_name="N (grid resolution)",
                   log_axis=np.linspace(log_Ns[0], log_Ns[-1], 40),
                   other_axis_grids=(fine_K, fine_D))
    plot_marginals(method_data, gps_sw, ax_marg, axis_idx=1,
                   axis_name="K (Trotter steps)",
                   log_axis=np.linspace(log_Ks[0], log_Ks[-1], 40),
                   other_axis_grids=(fine_N, fine_D))
    plot_marginals(method_data, gps_sw, ax_marg, axis_idx=2,
                   axis_name="D_max (MPS bond)",
                   log_axis=np.linspace(log_Ds[0], log_Ds[-1], 40),
                   other_axis_grids=(fine_N, fine_K))
    for ax, ax_name in zip(ax_marg, ("N", "K", "D_max")):
        ax.set_title(f"min SW vs {ax_name}  (best over other axes)",
                     fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(12)
        ax.legend(fontsize=11, loc="best", ncol=2)

    # ── Heatmap rows: each preceded by a centered title strip ───────────────
    # Title strips are full-width (cols 0..n_heat_cols+1) thin axes used only
    # for centered text. Heatmap rows below are n_heat_cols + colorbar slot.
    PAIRS = [
        (0, 1, 2, log_Ns, log_Ks, log_Ds, "N × K  (best SW over D_max)", "N", "K", 1, 2),
        (0, 2, 1, log_Ns, log_Ds, log_Ks, "N × D_max  (best SW over K)", "N", "D_max", 3, 4),
    ]
    for ax_a, ax_b, fixed_ax, ga, gb, fixed_grid, title, name_a, name_b, title_row, heat_row in PAIRS:
        # Title strip — centered across the FULL width including colorbar slot
        title_ax = fig.add_subplot(gs[title_row, :])
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, title, ha="center", va="center",
                      fontsize=16, fontweight="bold",
                      transform=title_ax.transAxes)
        row_idx = heat_row
        # Argmin over the held-out axis (sweep a fine grid via GP, take min).
        all_vals = []
        per_method_grids = {}
        fine_fixed = np.linspace(fixed_grid[0], fixed_grid[-1], 25)
        for method, gp_sw in gps_sw.items():
            if gp_sw is None:
                continue
            AA, BB = np.meshgrid(ga, gb, indexing="xy")
            mu = np.empty(AA.shape)
            for i in range(AA.shape[0]):
                for j in range(AA.shape[1]):
                    pts = np.zeros((len(fine_fixed), 3))
                    pts[:, ax_a] = AA[i, j]
                    pts[:, ax_b] = BB[i, j]
                    pts[:, fixed_ax] = fine_fixed
                    log_mu = gp_sw.predict(pts)
                    mu[i, j] = float(np.exp(np.min(log_mu)))
            per_method_grids[method] = mu
            all_vals.extend(mu.ravel().tolist())
        if not all_vals:
            continue
        vmin = max(float(np.min(all_vals)), 1e-4)        # log scale needs > 0
        vmax = float(np.max(all_vals))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        last_im = None
        first_method_ax = None
        for j, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[row_idx, j])
            if first_method_ax is None:
                first_method_ax = ax
            mu = per_method_grids.get(method)
            if mu is None:
                ax.set_title(METHOD_LABEL[method] + "\n(no data)", fontsize=10)
                continue
            im = ax.imshow(np.clip(mu, vmin, None), origin="lower",
                           cmap="plasma_r", aspect="auto", norm=norm)
            last_im = im
            ax.set_xticks(range(len(ga)))
            ax.set_xticklabels([f"{int(2 ** v)}" for v in ga], fontsize=11)
            ax.set_yticks(range(len(gb)))
            ax.set_yticklabels([f"{int(2 ** v)}" for v in gb], fontsize=11)
            ax.set_xlabel(name_a, fontsize=13)
            if j == 0:
                ax.set_ylabel(name_b, fontsize=13)
            log_vmin = np.log(vmin)
            log_vmax = np.log(vmax)
            for i in range(mu.shape[0]):
                for jj in range(mu.shape[1]):
                    v = max(mu[i, jj], vmin)
                    norm_v = (np.log(v) - log_vmin) / max(log_vmax - log_vmin, 1e-12)
                    text_c = "white" if norm_v > 0.55 else "black"
                    ax.text(jj, i, f"{mu[i, jj]:.3f}", ha="center", va="center",
                            fontsize=10, color=text_c, fontweight="bold")
            ax.set_title(METHOD_LABEL[method], fontsize=13, fontweight="bold",
                         color=METHOD_COLORS[method])
        # Colorbar — aligned to its row via the dedicated last column slot.
        if last_im is not None:
            cax = fig.add_subplot(gs[row_idx, n_heat_cols])
            cbar = fig.colorbar(last_im, cax=cax)
            cbar.set_label("SW", fontsize=12)
            cbar.ax.tick_params(labelsize=11)

    # ── Row 5: Pareto (SW vs D_max) + GP kernel summary ─────────────────────
    half = max(1, (n_heat_cols + 1) // 2)
    ax_pareto = fig.add_subplot(gs[5, 0:half])
    plot_pareto(method_data, gps_sw, ax_pareto, log_Ns, log_Ks, log_Ds)

    # GP kernel summary alongside Pareto
    ax_summary = fig.add_subplot(gs[5, half:])
    ax_summary.axis("off")
    rows_ks = []
    for method, gp in gps_sw.items():
        if gp is None:
            continue
        k = gp.kernel_
        try:
            ls = k.k1.k2.length_scale
            ls_str = f"ℓ_N={ls[0]:.2f}, ℓ_K={ls[1]:.2f}, ℓ_D={ls[2]:.2f}"
        except Exception:
            ls_str = str(k)
        rows_ks.append([METHOD_LABEL[method], ls_str, len(method_data[method]["sw"])])
    table_str = "GP length scales (log₂ units)\n\n"
    for r in rows_ks:
        table_str += f"  {r[0]:<11s}  {r[1]:<38s}  ({r[2]} cells)\n"
    table_str += "\nLarge ℓ ⇒ SW is insensitive to that axis."
    ax_summary.text(0.02, 0.95, table_str, fontfamily="monospace", fontsize=12,
                    va="top", transform=ax_summary.transAxes)

    fig.suptitle(
        f"{args.dataset_label} — hyperparameter sweep visualisations\n"
        f"GP-interpolated from Sobol-sampled (N, K, D_max) cells",
        fontsize=14, fontweight="bold", y=0.995,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
