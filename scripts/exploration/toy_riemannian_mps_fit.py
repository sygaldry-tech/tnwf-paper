"""Toy validation: fit a known real MPS by minimizing ½‖V_dense - target‖².

This isolates the optimizer geometry from the Path-3 Trotter loss. The
question is: at high learning rate, does Riemannian Adam (QR retraction +
gauge-projected gradient) tolerate steps that Euclidean Adam can't?

Setup
-----
- Target: random rank-D MPS, contracted to dense V_target on N^d grid.
- Param: V_mps with the same bond structure, small random init.
- Loss: ½ ‖V_dense(V_mps) - V_target‖² (closed-form ∇L = V_dense - V_target).
- No Trotter, no adjoint — pure parameter-space test.

Compares
--------
1. Euclidean Adam  (existing _AdamMPS)
2. Euclidean Adam + periodic re-canonicalization (Phase 0 baseline)
3. Riemannian Adam (_RiemannianAdamMPS)

For each, sweep lr ∈ {1e-3, 1e-2, 1e-1} and report final loss, NaN status,
and parameter Frobenius-norm growth.

Output
------
- Prints a table to stdout.
- Saves ``results/figures/toy_riemannian_compare.png`` with loss curves and
  Stiefel-orthogonality drift over iterations.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from riemannian_adam_mps import _RiemannianAdamMPS, init_v_mps_riemannian  # noqa: E402
from riemannian_mps import (  # noqa: E402
    stiefel_orthogonality_error,
    to_mixed_canonical,
)
from v_mps_parameterization import (  # noqa: E402
    V_mps_to_dense,
    project_dense_grad_to_mps_cores,
)


def _random_target_mps(N: int, d: int, D: int, rng: np.random.Generator
                       ) -> list[np.ndarray]:
    cores = []
    D_prev = 1
    for j in range(d):
        D_next = 1 if j == d - 1 else min(D, N ** (j + 1), N ** (d - j - 1))
        cores.append(rng.standard_normal((D_prev, N, D_next)))
        D_prev = D_next
    return cores


def _init_random_v_mps(N: int, d: int, D: int, rng: np.random.Generator,
                        init_std: float = 0.1) -> list[np.ndarray]:
    cores = []
    D_prev = 1
    for j in range(d):
        D_next = 1 if j == d - 1 else min(D, N ** (j + 1), N ** (d - j - 1))
        cores.append(rng.standard_normal((D_prev, N, D_next)) * init_std)
        D_prev = D_next
    return cores


class _EuclideanAdamMPS:
    """Copy of the optimizer from train_v_mps_end_to_end.py."""
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _run_one(optimizer_name: str, lr: float, target_dense: np.ndarray,
              N: int, d: int, D: int, n_iter: int,
              recanonicalize_every: int = 0,
              rng_seed: int = 0):
    """Run one optimization trajectory and return diagnostics."""
    rng = np.random.default_rng(rng_seed)
    if optimizer_name == "radam":
        V_mps = init_v_mps_riemannian(K_data=1, N=N, d=d, D_V_param=D,
                                       init_std=0.1, rng=rng)[0]
        opt = _RiemannianAdamMPS(V_mps, lr=lr)
    else:
        V_mps = _init_random_v_mps(N, d, D, rng, init_std=0.1)
        opt = _EuclideanAdamMPS(V_mps, lr=lr)

    losses = []
    stiefel_errs = []
    param_norms = []
    for it in range(n_iter):
        V_dense = V_mps_to_dense(V_mps, N, d)
        grad_dense = (V_dense - target_dense).reshape(-1)
        loss = 0.5 * float(np.sum(grad_dense ** 2))
        if not np.isfinite(loss):
            # Pad remaining iters with NaN so plots stay aligned
            while len(losses) < n_iter:
                losses.append(float("nan"))
                stiefel_errs.append(float("nan"))
                param_norms.append(float("nan"))
            return losses, stiefel_errs, param_norms

        losses.append(loss)
        # diagnostics: stiefel error against the "intended" mixed-canonical
        # form (only meaningful for radam; for euclidean it's how far the
        # cores have drifted from any canonical form)
        try:
            stiefel_errs.append(stiefel_orthogonality_error(V_mps))
        except Exception:
            stiefel_errs.append(float("nan"))
        param_norms.append(float(np.sqrt(sum((c ** 2).sum() for c in V_mps))))

        grad_cores = project_dense_grad_to_mps_cores(grad_dense, V_mps, N, d)
        opt.step(grad_cores)

        if recanonicalize_every > 0 and (it + 1) % recanonicalize_every == 0:
            new_V = to_mixed_canonical(V_mps)
            for c, nc in zip(V_mps, new_V):
                c[...] = nc

    return losses, stiefel_errs, param_norms


def main():
    N, d, D = 8, 2, 4                                # toy size
    n_iter = 500
    rng = np.random.default_rng(42)
    target_mps = _random_target_mps(N, d, D, rng)
    target_dense = V_mps_to_dense(target_mps, N, d)
    target_norm = float(np.linalg.norm(target_dense))
    print(f"Target: N={N} d={d} D={D}  ‖V_target‖={target_norm:.3f}\n")

    configs = [
        # name,                  optimizer, lr,    recanonicalize_every
        ("Euclidean Adam, lr=1e-3",       "adam",  1e-3, 0),
        ("Euclidean Adam, lr=1e-2",       "adam",  1e-2, 0),
        ("Euclidean Adam, lr=1e-1",       "adam",  1e-1, 0),
        ("Eucl + recanon@50, lr=1e-2",    "adam",  1e-2, 50),
        ("Eucl + recanon@50, lr=1e-1",    "adam",  1e-1, 50),
        ("Riemannian Adam, lr=1e-3",      "radam", 1e-3, 0),
        ("Riemannian Adam, lr=1e-2",      "radam", 1e-2, 0),
        ("Riemannian Adam, lr=1e-1",      "radam", 1e-1, 0),
    ]

    results = {}
    print(f"{'config':<35} {'final loss':>14} {'final ‖θ‖':>12} {'finite':>8}")
    print("-" * 70)
    for name, optim, lr, recanon in configs:
        t0 = time.time()
        losses, stiefel, norms = _run_one(
            optim, lr, target_dense, N, d, D, n_iter,
            recanonicalize_every=recanon,
        )
        wall = time.time() - t0
        final_loss = losses[-1]
        final_norm = norms[-1]
        is_finite = np.isfinite(final_loss)
        results[name] = (losses, stiefel, norms)
        print(f"{name:<35} {final_loss:>14.4e} {final_norm:>12.3e} {str(is_finite):>8} "
              f"({wall:.1f}s)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = {
        "Euclidean Adam, lr=1e-3":    "#1f77b4",
        "Euclidean Adam, lr=1e-2":    "#1f77b4",
        "Euclidean Adam, lr=1e-1":    "#1f77b4",
        "Eucl + recanon@50, lr=1e-2": "#ff7f0e",
        "Eucl + recanon@50, lr=1e-1": "#ff7f0e",
        "Riemannian Adam, lr=1e-3":   "#2ca02c",
        "Riemannian Adam, lr=1e-2":   "#2ca02c",
        "Riemannian Adam, lr=1e-1":   "#2ca02c",
    }
    lr_to_style = {1e-3: ":", 1e-2: "-", 1e-1: "--"}
    for name, (losses, stiefel, norms) in results.items():
        # Determine lr from name (hacky but ok for toy)
        if "1e-3" in name:
            ls = lr_to_style[1e-3]
        elif "1e-2" in name:
            ls = lr_to_style[1e-2]
        else:
            ls = lr_to_style[1e-1]
        c = palette[name]
        axes[0].plot(losses, label=name, color=c, linestyle=ls, lw=1.4)
        axes[1].plot(stiefel, color=c, linestyle=ls, lw=1.4)
        axes[2].plot(norms, color=c, linestyle=ls, lw=1.4)

    axes[0].set_yscale("log"); axes[0].set_xlabel("iter"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss (target ‖V‖={target_norm:.2f})")
    axes[0].legend(loc="best", fontsize=7)
    axes[1].set_yscale("log"); axes[1].set_xlabel("iter"); axes[1].set_ylabel("Stiefel error")
    axes[1].set_title("Stiefel drift (constrained-core orthogonality)")
    axes[2].set_yscale("log"); axes[2].set_xlabel("iter"); axes[2].set_ylabel("‖θ‖_F")
    axes[2].set_title("Total parameter Frobenius norm")
    fig.suptitle(f"Toy MPS fit: N={N} d={d} D={D}  Riemannian vs Euclidean Adam",
                 y=1.02)
    fig.tight_layout()
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "toy_riemannian_compare.pdf"
    out_png = out_dir / "toy_riemannian_compare.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=140)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
