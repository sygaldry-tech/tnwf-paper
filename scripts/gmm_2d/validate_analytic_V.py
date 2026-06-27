"""Small validation: JAM-trained V vs analytic V_t on gmm_2d.

Hypothesis: analytic V_t (closed-form Tweedie expression) eliminates JAM
training noise. The remaining gap (JAM SW − analytic SW) at the same seed
isolates JAM-specific error.

Design:
    cells          = [(N=16, K=8), (N=32, K=16)]
    methods        = dense, tci_als, aci, tci_tdvp1, tci_tdvp2  (skip jam)
    seeds          = 0..4
    V_sources      = jam, analytic
    total          = 5 methods × 2 cells × 5 seeds × 2 sources = 100 runs

Outputs CSV summary + paired t-test for each (method, cell):
    JAM_mean - analytic_mean   →   if positive, JAM is worse (more error)
                               →   the JAM training noise component
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from scipy import stats

from tnwf.pipelines.run_evolution import run

CELLS = [(16, 8), (32, 16)]
METHODS = ["dense", "tci_als", "aci", "tci_tdvp1", "tci_tdvp2"]
SEEDS = list(range(5))
CKPT_DIR = Path("data/gmm_2d/jam")
OUT_CSV = Path("results/gmm_2d/analytic_V_validation.csv")
OUT_MD = Path("results/gmm_2d/analytic_V_validation.md")


def main():
    rows = []
    t0 = time.perf_counter()
    for N, K in CELLS:
        D_max = min(N, 32)
        method_kwargs = {"D_max": D_max, "D_V": D_max, "D_out": D_max,
                         "D_init": D_max}
        for method in METHODS:
            for seed in SEEDS:
                ckpt = CKPT_DIR / f"seed{seed}.pt"
                if not ckpt.exists():
                    print(f"[skip] no ckpt {ckpt}")
                    continue
                for V_source in ("jam", "analytic"):
                    print(f"  N={N} K={K} {method:>10s} seed={seed} V={V_source}", flush=True)
                    out = run(
                        method=method, dataset="gmm_2d", jam_ckpt=str(ckpt),
                        seed=seed, N=N, K=K, n_samples=2000,
                        save=False, method_kwargs=method_kwargs,
                        V_source=V_source,
                    )
                    rows.append({
                        "N": N, "K": K, "method": method,
                        "seed": seed, "V_source": V_source,
                        "sw": float(out["sw"][-1]),
                        "mmd": float(out["mmd"][-1]),
                        "time": float(out["total_time"]),
                    })
    print(f"\nTotal: {len(rows)} runs in {time.perf_counter() - t0:.0f} s")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")

    # ── Paired analysis ────────────────────────────────────────────────────
    md = ["# Validation: JAM-trained vs analytic V_t  (gmm_2d)", ""]
    md.append("Paired (same-seed) comparison.  μ_diff = SW_jam − SW_analytic.")
    md.append("Positive μ_diff ⇒ analytic is **better** (less error).")
    md.append("")
    md.append("| N | K | method | n_pairs | SW_jam | SW_analytic | μ_diff | paired p (1-sided JAM>analytic) | sig |")
    md.append("|---|---|--------|---------|--------|-------------|--------|---------------------------------|-----|")

    by_cell = {}
    for r in rows:
        key = (r["N"], r["K"], r["method"])
        by_cell.setdefault(key, []).append(r)

    for (N, K, method), entries in sorted(by_cell.items()):
        jam_by_seed = {e["seed"]: e["sw"] for e in entries if e["V_source"] == "jam"}
        ana_by_seed = {e["seed"]: e["sw"] for e in entries if e["V_source"] == "analytic"}
        common = sorted(set(jam_by_seed) & set(ana_by_seed))
        if len(common) < 3:
            continue
        a = np.array([jam_by_seed[s] for s in common])
        b = np.array([ana_by_seed[s] for s in common])
        diff = a - b
        if diff.std(ddof=1) < 1e-12:
            p_one = 0.5
        else:
            t, p_two = stats.ttest_rel(a, b)
            p_one = p_two / 2 if t > 0 else 1 - p_two / 2     # H1: a > b
        sig = ("***" if p_one < 0.001 else "**" if p_one < 0.01 else
               "*"   if p_one < 0.05  else "·"  if p_one < 0.10 else "ns")
        md.append(
            f"| {N} | {K} | {method} | {len(common)} | "
            f"{a.mean():.4f} ± {a.std(ddof=1):.4f} | "
            f"{b.mean():.4f} ± {b.std(ddof=1):.4f} | "
            f"{diff.mean():+.4f} | {p_one:.3f} | {sig} |"
        )

    md.append("")
    md.append("**Interpretation**:")
    md.append("- μ_diff ≈ 0  → JAM training already converged; nothing to gain from analytic V_t.")
    md.append("- μ_diff > 0  → analytic V_t reduces error; the gap is JAM training noise.")
    md.append("- μ_diff < 0  → analytic V_t is worse (perhaps numerical issue at low t).")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
