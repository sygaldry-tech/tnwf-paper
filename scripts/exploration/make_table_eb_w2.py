"""EB scRNA-seq W2 table — Adjoint vs Adjoint+Riemannian columns.

Reproduces the table layout of Koshizuka & Sato (2022) Table 1 / Neklyudov
et al. (2023, eAM Table 1): per-snapshot W2 distance between the test
marginal and the model's prediction of that marginal, using the
**one-step prediction protocol** (predict t_k from the observed t_{k-1}
marginal, not from t_0).

For each seed:

  1. Sample the EB PCA-5 trajectory once (5 snapshots × n_per_step
     samples), split into train + held-out test sets.
  2. Train V_mps on the train split using the K=4 Trotter chain.
  3. For each k in 1..K:
       * Reset ψ = √q̂_{t_{k-1}, test}  (from test marginal at t_{k-1}).
       * Run a single Trotter segment forward.
       * Sample N_eval points from |ψ_{t_k}|².
       * Compute W2 against the test marginal at t_k.

Two optimisers ("adjoint" = Euclidean Adam, "radam" = Riemannian Adam);
multiple seeds; report mean ± std per (method, t_k).

Run::

    uv run python scripts/exploration/make_table_eb_w2.py \
        --N 16 --n_iter 1000 --seeds 0,1,2

Output: prints a plain-text table and writes
``results/eb_5d_w2_table.npz`` for downstream use (e.g. LaTeX rendering).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adjoint_trotter import trotter_coefficients  # noqa: E402
from adjoint_trotter_mps import forward_with_cache_mps  # noqa: E402
from riemannian_adam_mps import _RiemannianAdamMPS, init_v_mps_riemannian  # noqa: E402
from train_v_mps_end_to_end import (  # noqa: E402
    _AdamMPS,
    _init_V_mps_list,
    _train_step,
    bin_samples_to_density,
)
from v_mps_parameterization import V_mps_to_dense  # noqa: E402

from tnwf.data.embryoid_body import sample_eb_trajectory
from tnwf.metrics.wasserstein import wasserstein_2_subsampled
from tnwf.mps.core import dense_to_mps, mps_to_dense, right_canonicalize, sample_mps_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _autoreg_sample(mps, n, N, L, rng):
    """Sample n points from |ψ|² via autoregressive draws on a centred grid.
    Returns ``(n, d) float64`` in centred coords."""
    rc = right_canonicalize(mps)
    idx = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, idx.shape)
    return ((idx + jitter) * dx - L / 2.0).astype(np.float64)


def _bin_centred(samples_centred, N, d, L, eps=1e-6):
    """Bin centred-coordinate samples on the [-L/2, L/2]^d grid; return a
    flat (N^d,) probability vector."""
    edges = [np.linspace(-L / 2.0, L / 2.0, N + 1)] * d
    counts, _ = np.histogramdd(samples_centred, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    return (counts / counts.sum()).reshape(-1)


def _train_one(optimizer_kind, K_data, N, d, D_V_param,
                psi0_np, q_grids_np, alpha, beta, L,
                D_max, D_V, n_iter, lr, seed,
                grad_clip_norm=0.0):
    """Train and return the trained V_mps_list."""
    rng = np.random.default_rng(seed)
    if optimizer_kind == "radam":
        V_mps_list = init_v_mps_riemannian(K_data, N, d, D_V_param, rng=rng)
        opt = [_RiemannianAdamMPS(V_mps_list[k], lr=lr) for k in range(K_data)]
    else:
        V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param, rng=rng)
        flat = [c for cores in V_mps_list for c in cores]
        opt = _AdamMPS(flat, lr=lr)

    for _ in range(n_iter):
        _train_step(V_mps_list, opt, psi0_np, q_grids_np,
                     K_data, alpha, beta, N, d, L, D_max, D_V,
                     grad_clip_norm=grad_clip_norm,
                     optimizer_kind=optimizer_kind)
    return V_mps_list


def _one_step_w2(V_mps_list, q_test_grids, test_marginals_centred,
                  N, d, L, alpha, beta, D_max, D_V,
                  n_eval=400, w2_subsample=200, w2_repeats=3,
                  rng_seed=0):
    """One-step prediction protocol:

    For each k = 1..K, reset ψ = √q̂_{test, k-1}, run a single Trotter
    segment with V_mps_list[k-1], sample, compute W2 against the
    held-out test marginal at t_k.

    Returns ``(W2_t1, ..., W2_tK)``.
    """
    K_data = len(V_mps_list)
    out = np.empty(K_data, dtype=np.float64)
    for k in range(1, K_data + 1):
        # Reset ψ to test-marginal amplitude at t_{k-1}
        psi_kminus1 = np.sqrt(q_test_grids[k - 1]).astype(np.complex128)
        nrm = np.linalg.norm(psi_kminus1)
        if nrm == 0:
            out[k - 1] = float("nan"); continue
        psi_kminus1 /= nrm
        # Run only segment k-1 (single Trotter segment) using the
        # corresponding learned V_mps_list[k-1]
        V_grid_seg = [V_mps_to_dense(V_mps_list[k - 1], N, d)]
        cache = forward_with_cache_mps(
            psi_kminus1, V_grid_seg,
            alpha=alpha, beta=beta, N=N, d=d, L=L,
            D_max=D_max, D_V=D_V,
        )
        mps_after = cache[8]                                  # last substep
        rng = np.random.default_rng(rng_seed + 1000 * k)
        samples_centred = _autoreg_sample(mps_after, n=n_eval, N=N, L=L,
                                            rng=rng)
        target_centred = test_marginals_centred[k]
        n_sub = min(w2_subsample, samples_centred.shape[0],
                     target_centred.shape[0])
        w2, _ = wasserstein_2_subsampled(
            samples_centred, target_centred,
            n_sub=n_sub, n_repeats=w2_repeats, seed=rng_seed)
        out[k - 1] = float(w2)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--D_V_param", type=int, default=8)
    p.add_argument("--D_max", type=int, default=16)
    p.add_argument("--D_V", type=int, default=16)
    p.add_argument("--K_data", type=int, default=4)
    p.add_argument("--L", type=float, default=10.0)
    p.add_argument("--lr_adam", type=float, default=1e-3)
    p.add_argument("--lr_radam", type=float, default=1e-3)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n_train", type=int, default=1500,
                   help="Train samples per snapshot.")
    p.add_argument("--n_test", type=int, default=500,
                   help="Held-out test samples per snapshot.")
    p.add_argument("--out_npz", type=str, default="results/eb_5d_w2_table.npz")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [("Adjoint", "adam", args.lr_adam),
               ("Adjoint + Riemannian", "radam", args.lr_radam)]

    d = 5
    N, L, K_data = args.N, args.L, args.K_data
    delta_t = 1.0 / K_data
    alpha, beta = trotter_coefficients(delta_t, N, d, L)

    # Storage: w2_results[method_name][seed_idx, k_idx]  shape (n_seeds, K_data)
    w2_results = {name: np.empty((len(seeds), K_data), dtype=np.float64)
                  for name, *_ in methods}

    print(f"Dataset: eb_5d (PCA-5 EB scRNA-seq)")
    print(f"  d={d} N={N} L={L}  K_data={K_data}  "
          f"D_V_param={args.D_V_param}  n_iter={args.n_iter}")
    print(f"  seeds={seeds}\n")

    for si, seed in enumerate(seeds):
        print(f"=== seed {seed} ===")
        # Draw a single trajectory with enough samples for train+test
        n_total = args.n_train + args.n_test
        full_traj = sample_eb_trajectory(n_per_step=n_total, seed=seed)
        # Split: first n_train for training, last n_test for held-out test
        train_traj = full_traj[:, :args.n_train, :]
        test_traj  = full_traj[:, args.n_train:args.n_train + args.n_test, :]

        # Build training-density grids and ψ_0
        q_train_grids = [_bin_centred(train_traj[k], N, d, L)
                         for k in range(K_data + 1)]
        psi0_train = np.sqrt(q_train_grids[0]).astype(np.complex128)
        psi0_train /= np.linalg.norm(psi0_train)
        # Test-density grids (used to reset ψ at each step) and the
        # raw centred test samples (used to compute W2).
        q_test_grids = [_bin_centred(test_traj[k], N, d, L)
                        for k in range(K_data + 1)]
        test_marginals_centred = [test_traj[k].astype(np.float64)
                                   for k in range(K_data + 1)]

        for name, kind, lr in methods:
            t0 = time.time()
            V_list = _train_one(kind, K_data, N, d, args.D_V_param,
                                 psi0_train, q_train_grids, alpha, beta, L,
                                 args.D_max, args.D_V,
                                 n_iter=args.n_iter, lr=lr, seed=seed)
            w2_per_k = _one_step_w2(
                V_list, q_test_grids, test_marginals_centred,
                N, d, L, alpha, beta, args.D_max, args.D_V,
                n_eval=400, rng_seed=seed)
            w2_results[name][si] = w2_per_k
            wall = time.time() - t0
            print(f"  {name:24s} W2={['%.3f' % v for v in w2_per_k]}  ({wall:.1f}s)")
        print()

    # Aggregate
    print("\n=== Aggregated Table (mean ± std across {} seeds) ===".format(len(seeds)))
    header = "{:24s}".format("Method") + "  ".join(
        f"W2(q_t{k}, q̂_t{k})".ljust(17) for k in range(1, K_data + 1))
    print(header)
    print("-" * len(header))
    for name, *_ in methods:
        arr = w2_results[name]                                # (n_seeds, K)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        row = "{:24s}".format(name) + "  ".join(
            f"{m:.3f} ± {s:.3f}".ljust(17) for m, s in zip(means, stds))
        print(row)

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             seeds=np.array(seeds),
             timepoints=np.arange(1, K_data + 1),
             **{f"w2_{name.replace(' ', '_').replace('+', 'plus')}":
                arr for name, arr in w2_results.items()})
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
