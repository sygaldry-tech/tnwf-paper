"""Petals — per-snapshot SW for the six V-step methods sharing one JAM.

Mirrors the visual idiom of AM-paper Fig 2's method axis: SW + MMD vs
time with one line per V-step method, all using the same trained JAM
checkpoint as the source of V_t. The wave-function-based methods
(TDVP-1, TDVP-2) are the conservative analogue of "evolving the
density under H = K + V" rather than the gradient-flow `ẋ = ∇V_t(x)`.

Reads ``data/petals_2d/<method>{,_cfm_ot}/N{N}_K{K}/seed*.npz`` and
falls back gracefully if a method's sweep hasn't been run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tnwf.metrics import wasserstein_2_subsampled

METHOD_STYLE = {
    "jam":       dict(label="JAM grad-flow",  color="#7f7f7f", marker="s"),
    "dense":     dict(label="Dense",          color="#1f77b4", marker="D"),
    "tci_als":   dict(label="TCI + ALS",      color="#2ca02c", marker="^"),
    "aci":       dict(label="ACI",            color="#d62728", marker="v"),
    "tci_tdvp1": dict(label="TCI + TDVP-1",   color="#9467bd", marker="o"),
    "tci_tdvp2": dict(label="TCI + TDVP-2",   color="#ff7f0e", marker="x"),
}


def _load_w2(method: str, suffix: str, N: int, K: int) -> np.ndarray | None:
    """Return per-snapshot W_2, computed on-the-fly from samples_per_step
    + target_traj when not natively saved in the npz.
    """
    arrs = []
    for p in sorted(Path(f"data/petals_2d/{method}{suffix}/N{N}_K{K}").glob("seed*.npz")):
        d = np.load(p)
        if "w2" in d.files and d["w2"].size > 0:
            arrs.append(d["w2"])
            continue
        # Fallback: compute W_2 from samples_per_step vs target_traj
        if "target_traj" not in d.files:
            continue
        sps = d["samples_per_step"]                    # (K+1, ≤200, dim)
        tt = d["target_traj"]                          # (K+1, n_tgt, dim)
        seed = int(d["seed"]) if "seed" in d.files else 0
        w2_seq = []
        for k in range(sps.shape[0]):
            n_sub = min(sps.shape[1], tt.shape[1], 200)
            w2_mean, _ = wasserstein_2_subsampled(
                sps[k].astype(np.float64), tt[k].astype(np.float64),
                n_sub=n_sub, n_repeats=3, seed=seed,
            )
            w2_seq.append(w2_mean)
        arrs.append(np.array(w2_seq))
    return np.stack(arrs) if arrs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--suffix", default="_cfm_ot",
                   help="Method-dir suffix selecting which JAM-training "
                        "variant's sweep to read (default '_cfm_ot').")
    p.add_argument("--out_dir", default="results/figures")
    args = p.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))
    t = np.linspace(0.0, 1.0, args.K + 1)
    for method, style in METHOD_STYLE.items():
        arr = _load_w2(method, args.suffix, args.N, args.K)
        if arr is None:
            arr = _load_w2(method, "", args.N, args.K)  # fall back
        if arr is None or not np.all(np.isfinite(arr)):
            continue
        mu = arr.mean(axis=0)
        sd = arr.std(axis=0)
        ax.plot(t, mu, marker=style["marker"], color=style["color"],
                lw=1.7, ms=6.0, label=style["label"])
        ax.fill_between(t, mu - sd, mu + sd,
                        color=style["color"], alpha=0.13)
    ax.set_xlabel("time $t$", fontsize=11)
    ax.set_ylabel("$W_2$ distance (lower better)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2)
    ax.set_title(
        f"Petals (2D) — per-snapshot $W_2$ × V-step method  "
        f"(N={args.N}, K={args.K}, D=16, 3 seeds, ±1σ band)",
        fontsize=11, fontweight="bold")
    fig.tight_layout()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "petals_method_comparison.pdf"
    png = out_dir / "petals_method_comparison.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
