"""End-to-end JAM-driven wavefunction-flow pipeline.

run(method, dataset, seed, N, d, K, β-schedule, jam_ckpt) → snapshots .npz

For each method in {dense, tci_tdvp1, tci_tdvp2, jam}
this:
  1. Loads the JAM checkpoint to obtain V_fn(x, t).
  2. Initialises ψ_0 = √(N(0, I)) on the centred grid (positive real, ‖ψ_0‖=1).
  3. Evolves through K Trotter steps via the 8-step product formula.
  4. After each Trotter step, samples from |ψ|² and computes SW/MMD/NLL vs the
     target distribution at that time slice.
  5. Saves snapshots to results/{dataset}/{method}/seed{S}.npz.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from tnwf.data.gaussian_mixture import sample_gaussian_mixture
from tnwf.data.swiss_roll import sample_swiss_roll
from tnwf.dense.evolution import (
    apply_K_step,
    apply_V_step,
    trotter_coefficients,
)
from tnwf.grid import make_grid, make_kinetic_eigenvalues
from tnwf.jam.train import load_jam, make_V_fn, sample_gradient_flow
from tnwf.metrics import (density_from_psi, mmd_rbf, nll_from_density_grid,
                           sliced_wasserstein, wasserstein_2_subsampled)
from tnwf.mps.core import (
    apply_K_step_mps,
    chi_max,
    dense_to_mps,
    mps_to_dense,
    right_canonicalize,
    sample_mps_indices,
)
from tnwf.mps.vstep_tdvp import (
    apply_V_step_mps_tci_tdvp1,
    apply_V_step_mps_tci_tdvp2,
)

Method = Literal[
    "jam",          # classical baseline: dx/dt = ∇V_t(x), Euler integration
    "dense",
    "tci_tdvp1",
    "tci_tdvp2",
]


# ---------------------------------------------------------------------------
# Initial state + sampling
# ---------------------------------------------------------------------------

def initial_psi_dense(N: int, d: int, L: float, sigma: float = 1.0) -> np.ndarray:
    """Real-valued ψ_0(x) = √(p_source) where p_source = N(L/2, σ²·I) on [0,L)^d."""
    grid = make_grid(N=N, d=d, L=L)
    centred = grid - L / 2.0
    log_p = -0.5 * np.sum(centred ** 2, axis=-1) / (sigma ** 2)
    psi = np.exp(0.5 * log_p)                       # √p (positive real)
    psi = psi.astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def initial_psi_mps_from_samples(samples_world: np.ndarray, N: int, d: int,
                                  L: float, D_init: int = 16,
                                  eps: float = 1e-6) -> list[np.ndarray]:
    """Initial-state MPS approximating ``ψ_0(x) = √q_{t=0}(x)`` from samples.

    Replaces the Gaussian-on-grid default for trajectory datasets so the
    MPS pipeline starts at the same physical distribution as the JAM
    gradient-flow (which integrates from ``target_traj[0]``). Removes the
    initial-condition asymmetry called out in
    :doc:`results/RESULTS.md` (Table 1 caveat).

    Procedure:
      1. Bin world-frame samples on the ``N^d`` grid via histogramdd
      2. Add a uniform floor ``eps / N^d`` so every cell has positive mass
      3. Normalise → density ``ρ``, take ``√ρ`` → real-valued wave function
      4. Compress to a left-canonical MPS with bond cap ``D_init`` via
         :func:`tnwf.mps.core.dense_to_mps`

    Materialises an ``N^d``-cell dense vector internally. Feasible for
    ``N^d ≲ 10⁷`` (≈ 80 MB at complex128) — covers petals ``N=32 d=2``
    (1024 cells) and EB ``N=16 d=5`` (≈ 1M cells). For larger cells, use
    TCI from samples directly (TODO).
    """
    from tnwf.mps.core import dense_to_mps
    edges = [np.linspace(0.0, L, N + 1) for _ in range(d)]
    counts, _ = np.histogramdd(samples_world, bins=edges)
    counts = counts.astype(np.float64) + eps / (N ** d)
    rho = counts / counts.sum()
    psi_dense = np.sqrt(rho).ravel().astype(np.complex128)
    psi_dense /= np.linalg.norm(psi_dense)
    return dense_to_mps(psi_dense, N=N, d=d, D_max=D_init)


def initial_psi_mps(N: int, d: int, L: float, sigma: float = 1.0,
                    D_init: int = 16) -> list[np.ndarray]:
    """Initial-state MPS for ψ_0(x) = √(p_source), built directly without
    materializing the dense (N^d,) state. Equivalent to
    ``dense_to_mps(initial_psi_dense(...), D_max=D_init)`` but uses O(d·N·D²)
    memory instead of O(N^d) — required at d≥6 where the dense state is
    larger than container memory.

    Since the source Gaussian is separable in coordinates, the exact MPS
    representation is a rank-1 product state. We pad the bond dim to
    ``D_init`` (filling unused slices with zeros), matching what
    ``dense_to_mps`` produces — this is what TDVP needs to have headroom
    for the bond to grow during evolution.
    """
    grid_1d = np.linspace(0.0, L, N, endpoint=False) - L / 2.0
    psi_1d = np.exp(-(grid_1d ** 2) / (4.0 * sigma ** 2))     # √p_source 1D
    psi_1d = (psi_1d / np.linalg.norm(psi_1d)).astype(np.complex128)
    cores: list[np.ndarray] = []
    for j in range(d):
        D_L = 1 if j == 0 else D_init
        D_R = 1 if j == d - 1 else D_init
        core = np.zeros((D_L, N, D_R), dtype=np.complex128)
        core[0, :, 0] = psi_1d
        cores.append(core)
    return cores


def sample_target_distribution(
    dataset: str,
    n: int,
    seed: int,
    d: int,
    L: float,
    *,
    noise: float = 0.1,
    std: float = 0.5,
    scale: float = 3.0,
) -> np.ndarray:
    """Draw n samples in [0, L)^d (already shifted by L/2)."""
    if dataset == "swiss_roll_2d":
        x = sample_swiss_roll(n, noise=noise, seed=seed)
    elif dataset.startswith("gmm_") and dataset.endswith("d") and dataset[4:-1].isdigit():
        x = sample_gaussian_mixture(n, d=d, std=std, scale=scale,
                                    arrangement="orthogonal", seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # Shift to [0, L) frame
    return x.astype(np.float64) + L / 2.0


def sample_target_trajectory_world(
    dataset: str,
    n_per_step: int,
    seed: int,
    d: int,
    L: float,
    **kw,
) -> np.ndarray:
    """Trajectory datasets are not part of this minimal release (endpoint only)."""
    raise ValueError(
        f"{dataset!r}: trajectory datasets are not included in this release; "
        "only endpoint datasets (swiss_roll_2d, gmm_*d) are supported."
    )


def sample_from_psi_grid(
    psi: np.ndarray,
    N: int,
    d: int,
    L: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n samples in [0, L)^d from |ψ|² (binned on the N^d grid + uniform jitter)."""
    rho = density_from_psi(psi)
    cell_idx = rng.choice(len(rho), size=n, p=rho)
    nd_idx = np.array(np.unravel_index(cell_idx, (N,) * d)).T
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, nd_idx.shape)
    return ((nd_idx + jitter) * dx).astype(np.float64)


def sample_from_mps(
    mps: list[np.ndarray],
    N: int,
    d: int,
    L: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Autoregressive sampling. Right-canonicalises first so that the running
    left-context vector ‖v_k(n)‖² gives the correct conditional p(x_k | x_<k).
    """
    rc = right_canonicalize(mps)
    indices = sample_mps_indices(rc, n, rng)
    dx = L / N
    jitter = rng.uniform(0.0, 1.0, indices.shape)
    return ((indices + jitter) * dx).astype(np.float64)


# ---------------------------------------------------------------------------
# Evolution: 8-step product formula generalised to any V-step method
# ---------------------------------------------------------------------------

def _v_step_dense(psi, V_fn, beta, t_k, N, d, L, **_):
    grid_pts = make_grid(N=N, d=d, L=L)
    V_grid = np.asarray(V_fn(grid_pts, t_k)).ravel()
    return apply_V_step(psi, beta=beta, V_grid=V_grid)


def _make_v_step(method: Method, **method_kw):
    """Return a callable v_step(state, V_fn, beta, t_k, N, d, L) → new state.

    For 'dense': state is (N^d,) ψ array. For all MPS methods: state is mps list.
    """
    if method == "dense":
        return _v_step_dense
    if method == "tci_tdvp1":
        D_V = method_kw.get("D_V", 16)
        n_sw = method_kw.get("n_sweeps", 1)
        n_sw_cross = method_kw.get("n_sweeps_cross", 2)
        device = method_kw.get("device", "cpu")

        def f(state, V_fn, beta, t_k, N, d, L):
            return apply_V_step_mps_tci_tdvp1(
                state, V_fn=V_fn, beta=beta, t_k=t_k, N=N, d=d, L=L,
                D_V=D_V, n_sweeps=n_sw, n_sweeps_cross=n_sw_cross,
                device=device,
            )
        return f
    if method == "tci_tdvp2":
        D_max = method_kw.get("D_max", 16)
        D_V = method_kw.get("D_V", 16)
        n_sw = method_kw.get("n_sweeps", 1)
        n_sw_cross = method_kw.get("n_sweeps_cross", 2)
        tol = method_kw.get("tol", 1e-8)
        device = method_kw.get("device", "cpu")

        def f(state, V_fn, beta, t_k, N, d, L):
            return apply_V_step_mps_tci_tdvp2(
                state, V_fn=V_fn, beta=beta, t_k=t_k, N=N, d=d, L=L,
                D_max=D_max, D_V=D_V,
                n_sweeps=n_sw, n_sweeps_cross=n_sw_cross, tol=tol,
                device=device,
            )
        return f
    raise ValueError(f"Unknown method: {method}")


def _k_step(state, alpha, N, d, L, eigenvalues, *, is_dense: bool):
    if is_dense:
        return apply_K_step(state, alpha=alpha, N=N, d=d, L=L, eigenvalues=eigenvalues)
    return apply_K_step_mps(state, alpha=alpha, N=N, L=L)


def _product_formula_step(
    state,
    v_step: Callable,
    V_fn,
    alpha: float,
    beta: float,
    t_k: float,
    N: int,
    d: int,
    L: float,
    eigenvalues,
    *,
    is_dense: bool,
    n_substeps: int = 1,
):
    """8-step W ≈ exp(αβ · 2 [K, V]) approximation of exp(Δt · [K, V_t]).

    With αβ = Δt/2 (the natural Trotter coefficients), one call applies
    one 8-step BCH product formula. To reduce Trotter error, split the
    outer Δt into ``n_substeps`` equal sub-intervals, each running the
    8-step formula with rescaled coefficients α' = α/√n, β' = β/√n so
    α'β' = αβ/n = Δt/(2n). Per-substep BCH error is O((αβ/n)^{3/2}); the
    total error over n substeps is O(n · (αβ/n)^{3/2}) = O((αβ)^{3/2}/√n),
    so doubling n cuts the error by ~√2.
    """
    if n_substeps <= 1:
        a, b = alpha, beta
        state = _k_step(state, a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, b, t_k, N, d, L)
        state = _k_step(state, -a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, -b, t_k, N, d, L)
        state = _k_step(state, -a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, -b, t_k, N, d, L)
        state = _k_step(state, a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, b, t_k, N, d, L)
        return state
    # rescale: αβ_sub = αβ / n_substeps  ⟹  α' = α/√n, β' = β/√n
    scale = 1.0 / np.sqrt(n_substeps)
    a, b = alpha * scale, beta * scale
    for _ in range(n_substeps):
        state = _k_step(state, a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, b, t_k, N, d, L)
        state = _k_step(state, -a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, -b, t_k, N, d, L)
        state = _k_step(state, -a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, -b, t_k, N, d, L)
        state = _k_step(state, a, N, d, L, eigenvalues, is_dense=is_dense)
        state = v_step(state, V_fn, b, t_k, N, d, L)
    return state


# ---------------------------------------------------------------------------
# Top-level run() → npz of snapshots + per-step metrics
# ---------------------------------------------------------------------------

_DENSITY_GRID_MAX_CELLS = 1 << 24                      # 16M cells ≈ 128 MB float64

def _samples_to_grid_density(
    samples_centred: np.ndarray,
    N: int,
    d: int,
    L: float,
    smooth: bool = True,
) -> np.ndarray | None:
    """Bin centred-frame (n, d) samples into an M^d grid pmf normalised to sum=1.

    With ``smooth=True``, applies Laplace smoothing (+1 count per cell) so empty
    cells get a small floor — keeps log-density finite when target points fall
    in cells the method's samples never visit.

    Returns None when N^d would exceed _DENSITY_GRID_MAX_CELLS — at d≥7 N=32
    the bincount alone would allocate 34 GB and the binned density is
    meaningless anyway (most cells empty with only n_samples points).
    """
    if N ** d > _DENSITY_GRID_MAX_CELLS:
        return None
    grid_coords = np.asarray(samples_centred, dtype=np.float64) + L / 2.0
    grid_coords = np.clip(grid_coords, 0.0, L - 1e-9)
    idx = np.clip((grid_coords / L * N).astype(int), 0, N - 1)
    flat_idx = np.ravel_multi_index([idx[:, j] for j in range(d)], (N,) * d)
    counts = np.bincount(flat_idx, minlength=N**d).astype(np.float64)
    if smooth:
        counts = counts + 1.0
    return counts / counts.sum()


def _run_jam(
    model,
    dataset: str,
    seed: int,
    d: int,
    L: float,
    K: int,
    n_samples: int,
    out_dir: str | Path | None,
    save: bool,
    N_grid: int,
) -> dict:
    """Classical JAM baseline: integrate dx/dt = ∇V_t(x), bin samples for density."""
    import time as _time
    import torch

    rng = np.random.default_rng(seed + 1)
    # Resolve trajectory vs endpoint dataset semantics (mirrors run()).
    from tnwf.jam.train import DATASET_DEFAULTS as _DEFAULTS
    _ds_cfg = _DEFAULTS.get(dataset, {})
    kind = _ds_cfg.get("kind", "endpoint")
    trajectory_K = _ds_cfg.get("trajectory_K")
    if kind == "trajectory":
        if K != trajectory_K:
            raise ValueError(
                f"dataset {dataset!r} is trajectory with trajectory_K={trajectory_K}; "
                f"caller passed K={K}."
            )
        _traj_kw = {k: v for k, v in _ds_cfg.items()
                    if k not in ("d", "n_samples", "L", "kind", "trajectory_K")}
        target_traj = sample_target_trajectory_world(
            dataset, n_samples, seed=seed, d=d, L=L, **_traj_kw,
        )
        target = target_traj[-1]
    else:
        target_traj = None
        target = sample_target_distribution(dataset, n_samples, seed=seed, d=d, L=L)
    target_centred = target - L / 2.0                 # match model's centred frame

    # Initial particles for the gradient-flow ODE.
    #  - Endpoint datasets: source is N(0, I) (the JAM training source).
    #  - Trajectory datasets: source is q_{t=0} (the bio data's first
    #    snapshot). Sampling from N(0, I) instead would put the initial
    #    cloud at a much wider spread than the actual q_0 — this matches
    #    how the Action Matching paper (Neklyudov 2022) Euler-integrates
    #    from the data's first marginal forward, not from a Gaussian.
    if kind == "trajectory":
        target_traj_centred = target_traj - L / 2.0
        q0 = target_traj_centred[0]                    # (n_samples_traj, d) centred
        idx0 = rng.integers(0, q0.shape[0], size=n_samples)
        z_np = q0[idx0].astype(np.float32)
    else:
        z_np = rng.standard_normal((n_samples, d)).astype(np.float32)
    z = torch.from_numpy(z_np)

    t_run_start = _time.perf_counter()
    snaps = sample_gradient_flow(
        model, z, n_steps=K, save_every=1, L=L,
    )                                                  # (K+1, n_samples, d) centred
    total_time = _time.perf_counter() - t_run_start

    sw_list, mmd_list, nll_list, samples_snaps = [], [], [], []
    w2_list: list[float] = []                  # populated only for trajectory data
    for ti in range(K + 1):
        x_centred = snaps[ti].astype(np.float64)
        x_world = x_centred + L / 2.0                  # back to [0, L) frame for SW/MMD
        target_for_step = target_traj[ti] if target_traj is not None else target
        sw_list.append(sliced_wasserstein(x_world, target_for_step, n_projections=128, rng=rng))
        mmd_list.append(mmd_rbf(x_world, target_for_step))
        if target_traj is not None:
            w2_mean, _ = wasserstein_2_subsampled(
                x_world, target_for_step, n_sub=1000, n_repeats=3, seed=seed,
            )
            w2_list.append(w2_mean)
        rho = _samples_to_grid_density(x_centred, N=N_grid, d=d, L=L)
        if rho is None:
            nll_list.append(float("nan"))
        else:
            nll_list.append(nll_from_density_grid(target_for_step, rho, N=N_grid, d=d, L=L))
        samples_snaps.append(x_world[:200].astype(np.float32))

    samples_T = (snaps[-1] + L / 2.0).astype(np.float32)
    # JAM has uniform per-step cost (Euler), so split total over K steps
    step_times = np.full(K, total_time / max(K, 1), dtype=np.float64)
    out = {
        "method": "jam",
        "dataset": dataset,
        "seed": seed,
        "N": N_grid,
        "d": d,
        "K": K,
        "L": L,
        "sw": np.asarray(sw_list, dtype=np.float64),
        "mmd": np.asarray(mmd_list, dtype=np.float64),
        "nll": np.asarray(nll_list, dtype=np.float64),
        "chi_max": np.zeros(K + 1, dtype=np.int32),
        "w2": (np.asarray(w2_list, dtype=np.float64) if w2_list
               else np.zeros(0, dtype=np.float64)),
        "samples_T": samples_T,
        "target": target.astype(np.float32),
        "samples_per_step": np.stack(samples_snaps, axis=0),
        "step_times": step_times,
        "total_time": np.asarray(total_time, dtype=np.float64),
        "target_traj": (target_traj.astype(np.float32)
                        if target_traj is not None else None),
    }
    if save:
        if out_dir is None:
            out_dir_path = Path(f"data/{dataset}/jam")
        else:
            out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        savable = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
        for k in ("method", "dataset", "seed", "N", "d", "K", "L"):
            savable[k] = np.asarray(out[k])
        np.savez_compressed(out_dir_path / f"seed{seed}.npz", **savable)
    return out


def run(
    method: Method,
    dataset: str,
    jam_ckpt: str | None = None,
    *,
    seed: int = 0,
    N: int = 16,
    d: int | None = None,
    K: int = 8,
    L: float | None = None,
    n_samples: int = 2000,
    out_dir: str | Path | None = None,
    save: bool = True,
    method_kwargs: dict | None = None,
    snapshot_psi: bool = False,
    V_source: str = "jam",
    checkpoint_path: str | Path | None = None,
    checkpoint_callback=None,
    checkpoint_every: int = 1,
) -> dict:
    """Run a complete V-step pipeline and (optionally) save .npz snapshots.

    Args:
        method:    one of jam / dense / tci_tdvp1 / tci_tdvp2.
        dataset:   "swiss_roll_2d" / "gmm_2d" / "gmm_3d".
        jam_ckpt:  path to a trained JAM checkpoint. Required if V_source="jam"
                   or method="jam".
        V_source:  "jam"      — load V_t from the trained MLP (default).
                   "analytic" — closed-form V_t for Gaussian-source GMM
                                targets (see THEORY.md). No JAM needed.

    Returns dict with method, dataset, seed, N, d, K, L, sw, mmd, nll,
    chi_max, samples_T, target, samples_per_step, step_times, total_time
    (and optionally psi_T, psi_per_step).
    """
    method_kwargs = method_kwargs or {}
    if V_source == "analytic":
        from tnwf.theory import make_analytic_V_fn
        V_fn = make_analytic_V_fn(dataset)
        # Pull (d, L) from the dataset defaults rather than from a JAM ckpt
        from tnwf.jam.train import DATASET_DEFAULTS
        cfg = DATASET_DEFAULTS[dataset]
        if d is None:
            d = cfg["d"]
        if L is None:
            L = cfg["L"]
        model = None
    else:
        if jam_ckpt is None:
            raise ValueError("jam_ckpt is required when V_source='jam'")
        model, cfg = load_jam(jam_ckpt, device="cpu")
        if d is None:
            d = cfg["d"]
        if L is None:
            L = cfg["L"]
        V_fn = make_V_fn(model, device="cpu")

    # ── JAM gradient-flow baseline: skip wavefunction entirely ────────────
    if method == "jam":
        if model is None:
            raise ValueError(
                "method='jam' requires a JAM checkpoint (V_source='analytic' "
                "skips wavefunction methods, but JAM gradient-flow needs the "
                "trained MLP for ∇V via autograd)."
            )
        return _run_jam(
            model=model, dataset=dataset, seed=seed,
            d=d, L=L, K=K, n_samples=n_samples,
            out_dir=out_dir, save=save, N_grid=N,
        )

    delta_t = 1.0 / K
    alpha, beta = trotter_coefficients(delta_t, N=N, d=d, L=L)
    # MPS K-step builds 1-D eigenvalues per site; Dense needs the full N^d
    # tensor. At d≥7 N=32 the dense version alone is hundreds of GB and only
    # the MPS path is feasible — only allocate when the dense method needs it.
    eigenvalues = (make_kinetic_eigenvalues(N=N, d=d, L=L)
                   if method == "dense" else None)

    is_dense = method == "dense"
    D_init = method_kwargs.get("D_init", method_kwargs.get("D_max", 16))

    # Per-snapshot metric containers (snapshot index 0 = initial state)
    sw_list, mmd_list, nll_list, chi_list = [], [], [], []
    w2_list: list[float] = []                  # populated only for trajectory data
    samples_snapshots: list[np.ndarray] = []
    psi_snapshots: list[np.ndarray] = []
    rng = np.random.default_rng(seed + 1)

    # ── Resolve trajectory vs endpoint dataset semantics ──────────────────
    # DATASET_DEFAULTS is the canonical source of truth (JAM ckpts may be
    # older than the trajectory feature). Moved BEFORE state-init so the
    # MPS / dense initial state can be matched to q_{t=0} on trajectory data.
    from tnwf.jam.train import DATASET_DEFAULTS as _DEFAULTS
    _ds_cfg = _DEFAULTS.get(dataset, {})
    kind = _ds_cfg.get("kind", "endpoint")
    trajectory_K = _ds_cfg.get("trajectory_K")
    if kind == "trajectory":
        if trajectory_K is None:
            raise ValueError(
                f"dataset {dataset!r} has kind='trajectory' but no trajectory_K"
            )
        if K != trajectory_K:
            raise ValueError(
                f"dataset {dataset!r} is a trajectory dataset with "
                f"trajectory_K={trajectory_K}; caller passed K={K}. "
                f"They must match for per-snapshot SW alignment."
            )
        _traj_kw = {k: v for k, v in _ds_cfg.items()
                    if k not in ("d", "n_samples", "L", "kind", "trajectory_K")}
        target_traj = sample_target_trajectory_world(
            dataset, n_samples, seed=seed, d=d, L=L, **_traj_kw,
        )
        # Backward-compat scalar "target" used by checkpoint/output dict
        # consumers: pin it to the final timepoint.
        target = target_traj[-1]
    else:
        target_traj = None
        target = sample_target_distribution(dataset, n_samples, seed=seed, d=d, L=L)

    # ── State initialisation ─────────────────────────────────────────────
    # For trajectory data we match the initial wave-function density to
    # ``q_{t=0}`` (the bio data's first snapshot) — matches the JAM grad-flow
    # protocol (which starts at q_{t=0}) and removes the previous Gaussian-
    # vs-q_0 asymmetry between MPS pipeline and JAM baseline.
    if is_dense:
        if target_traj is not None and N**d <= 10_000_000:
            edges = [np.linspace(0.0, L, N + 1)] * d
            counts, _ = np.histogramdd(target_traj[0], bins=edges)
            counts = counts.astype(np.float64) + 1e-6 / (N ** d)
            rho = counts / counts.sum()
            state = np.sqrt(rho).ravel().astype(np.complex128)
            state /= np.linalg.norm(state)
        else:
            state = initial_psi_dense(N=N, d=d, L=L, sigma=1.0)
    else:
        if target_traj is not None and N**d <= 10_000_000:
            state = initial_psi_mps_from_samples(
                target_traj[0], N=N, d=d, L=L, D_init=D_init,
            )
        else:
            state = initial_psi_mps(N=N, d=d, L=L, sigma=1.0, D_init=D_init)

    # Build v_step dispatch
    v_step = _make_v_step(method, **method_kwargs)

    def _target_at(step_idx: int) -> np.ndarray:
        """Bio snapshot for step ``step_idx`` (0 = initial state, K = final)."""
        if target_traj is None:
            return target
        return target_traj[step_idx]

    def _record(state, target_for_step):
        """Compute per-snapshot metrics against ``target_for_step``.

        For endpoint datasets, callers pass the same ``target`` every step. For
        trajectory datasets (wired in a later commit), callers pass the bio
        snapshot corresponding to the current Trotter step.
        """
        if is_dense:
            psi_dense = state
            samples = sample_from_psi_grid(psi_dense, N=N, d=d, L=L,
                                           n=n_samples, rng=rng)
            chi_list.append(0)
        else:
            psi_dense = mps_to_dense(state, N=N, d=d) if N**d <= 4096 else None
            samples = sample_from_mps(state, N=N, d=d, L=L, n=n_samples, rng=rng)
            chi_list.append(int(chi_max(state)))
        sw_list.append(sliced_wasserstein(samples, target_for_step, n_projections=128, rng=rng))
        mmd_list.append(mmd_rbf(samples, target_for_step))
        # W₂ is the AM-paper metric (Neklyudov 2022 Table 1 + Fig 2). For
        # trajectory datasets we also report it. Subsampled n_sub=1000 over
        # 3 repeats — exact Hungarian on the full 2000×2000 cost matrix is
        # O(n³) = 8e9 ops, too slow inside the hot loop.
        if target_traj is not None:
            w2_mean, _ = wasserstein_2_subsampled(
                samples, target_for_step, n_sub=1000, n_repeats=3, seed=seed,
            )
            w2_list.append(w2_mean)
        # NLL = goodness-of-fit: evaluate the method's density at TARGET samples
        # (held-out, identical across methods). Lower NLL ⇒ method assigns higher
        # probability to true data. Self-NLL (= entropy) was misleading: a sharp
        # but mis-located density would score better than a correct but diffuse one.
        if psi_dense is not None:
            rho = density_from_psi(psi_dense)
            nll_list.append(nll_from_density_grid(target_for_step, rho, N=N, d=d, L=L))
        else:
            nll_list.append(float("nan"))
        # Lightweight snapshots — only first 200 samples per step + ψ if requested
        samples_snapshots.append(samples[:200].astype(np.float32))
        if snapshot_psi and psi_dense is not None:
            psi_snapshots.append(psi_dense.astype(np.complex64))

    import time as _time
    import pickle
    t_run_start = _time.perf_counter()
    step_times: list[float] = []
    start_k = 0

    # ── Resume from checkpoint if one exists ─────────────────────────────
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_path is not None and ckpt_path.exists():
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            # Validate the checkpoint matches our run config (N, d, K, method, seed)
            if (ckpt.get("method") == method and ckpt.get("dataset") == dataset
                    and ckpt.get("seed") == seed and ckpt.get("N") == N
                    and ckpt.get("d") == d and ckpt.get("K") == K):
                state = ckpt["state"]
                start_k = int(ckpt["next_k"])
                sw_list = list(ckpt["sw_list"])
                mmd_list = list(ckpt["mmd_list"])
                nll_list = list(ckpt["nll_list"])
                chi_list = list(ckpt["chi_list"])
                samples_snapshots = list(ckpt["samples_snapshots"])
                psi_snapshots = list(ckpt.get("psi_snapshots", []))
                step_times = list(ckpt["step_times"])
                rng = np.random.default_rng()
                rng.bit_generator.state = ckpt["rng_state"]
                target = ckpt["target"]
                if "target_traj" in ckpt and target_traj is not None:
                    target_traj = ckpt["target_traj"]
                print(f"[run] resumed from checkpoint at k={start_k}/{K}", flush=True)
            else:
                print(f"[run] checkpoint at {ckpt_path} mismatch — starting fresh",
                      flush=True)
        except Exception as e:
            print(f"[run] failed to load checkpoint ({e}) — starting fresh",
                  flush=True)

    if start_k == 0:
        _record(state, _target_at(0))

    def _save_checkpoint(next_k: int):
        """Atomically write checkpoint, then trigger external commit (volume, etc.)."""
        if ckpt_path is None:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({
                "method": method, "dataset": dataset, "seed": seed,
                "N": N, "d": d, "K": K,
                "next_k": next_k,
                "state": state,
                "sw_list": sw_list, "mmd_list": mmd_list,
                "nll_list": nll_list, "chi_list": chi_list,
                "samples_snapshots": samples_snapshots,
                "psi_snapshots": psi_snapshots,
                "step_times": step_times,
                "rng_state": rng.bit_generator.state,
                "target": target,
                "target_traj": target_traj,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(ckpt_path)
        if checkpoint_callback is not None:
            checkpoint_callback()

    n_substeps = int(method_kwargs.get("trotter_substeps", 1))
    progress_every = max(1, K // 20)               # ≤20 lines/run, always ≥1
    # Re-canonicalise the MPS after each Trotter step to clean up gauge drift
    # from bond-cap-saturated SVD truncation. Without this, at D=64 (chi cap)
    # the truncation residual compounds across 4–5 steps and SW jumps from
    # 0.27 → 1.16 (verified on the A100 g=2 N=64 K=8 diagnostic, 2026-05-18).
    # right_canonicalize is a pure-gauge transformation: preserves the state,
    # restores right-isometric form, ~one QR sweep — cheap vs the V-step.
    # Skipped for dense pipelines (no MPS).
    from tnwf.mps.core import right_canonicalize as _right_canonicalize
    for k in range(start_k, K):
        t_k = max((k + 0.5) * delta_t, 1e-5)
        t0 = _time.perf_counter()
        state = _product_formula_step(
            state, v_step, V_fn, alpha=alpha, beta=beta, t_k=t_k,
            N=N, d=d, L=L, eigenvalues=eigenvalues, is_dense=is_dense,
            n_substeps=n_substeps,
        )
        if not is_dense:
            state = _right_canonicalize(state)
        step_times.append(_time.perf_counter() - t0)
        _record(state, _target_at(k + 1))
        if checkpoint_every > 0 and (k + 1) % checkpoint_every == 0:
            _save_checkpoint(next_k=k + 1)
        # Per-step progress (flushed for live logs). Only every
        # `progress_every` step so we don't drown short cells with output.
        if (k + 1) % progress_every == 0 or (k + 1) == K:
            chi_last = chi_list[-1] if chi_list else 0
            sw_last = sw_list[-1] if sw_list else float("nan")
            print(f"  [run] {method} k={k+1}/{K}  "
                   f"t_step={step_times[-1]:.1f}s  chi={chi_last}  "
                   f"sw={sw_last:.4f}",
                   flush=True)

    total_time = _time.perf_counter() - t_run_start

    # Final samples
    if is_dense:
        samples_T = sample_from_psi_grid(state, N=N, d=d, L=L, n=n_samples, rng=rng)
        psi_T = state
    else:
        samples_T = sample_from_mps(state, N=N, d=d, L=L, n=n_samples, rng=rng)
        psi_T = mps_to_dense(state, N=N, d=d) if (snapshot_psi and N**d <= 4096) else None

    out = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "N": N,
        "d": d,
        "K": K,
        "L": L,
        "sw": np.asarray(sw_list, dtype=np.float64),
        "mmd": np.asarray(mmd_list, dtype=np.float64),
        "nll": np.asarray(nll_list, dtype=np.float64),
        "chi_max": np.asarray(chi_list, dtype=np.int32),
        # W₂ — populated only for trajectory datasets; AM-paper metric
        "w2": (np.asarray(w2_list, dtype=np.float64) if w2_list
               else np.zeros(0, dtype=np.float64)),
        "samples_T": samples_T.astype(np.float32),
        "target": target.astype(np.float32),
        # K+1 stacked (200, d) sample snapshots — useful for time-evolution panels
        "samples_per_step": np.stack(samples_snapshots, axis=0),
        # For trajectory datasets: (K+1, n, d) bio-data snapshots aligned with
        # the per-step SW comparisons. None for endpoint datasets (dropped from
        # npz by the isinstance-ndarray filter below).
        "target_traj": (target_traj.astype(np.float32)
                        if target_traj is not None else None),
        # Wall-clock per-Trotter-step (length K, in seconds) + total run time
        "step_times": np.asarray(step_times, dtype=np.float64),
        "total_time": np.asarray(total_time, dtype=np.float64),
    }
    if snapshot_psi:
        if psi_T is not None:
            out["psi_T"] = psi_T
        if psi_snapshots:
            out["psi_per_step"] = np.stack(psi_snapshots, axis=0)

    if save:
        if out_dir is None:
            out_dir = Path(f"data/{dataset}/{method}")
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        savable = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
        for k in ("method", "dataset", "seed", "N", "d", "K", "L"):
            savable[k] = np.asarray(out[k])
        np.savez_compressed(out_dir / f"seed{seed}.npz", **savable)
        # Successful save → drop the checkpoint to save volume space and avoid
        # confusing future runs.
        if ckpt_path is not None and ckpt_path.exists():
            try:
                ckpt_path.unlink()
            except OSError:
                pass

    return out
