"""Modal app: run the N=32 V_mps end-to-end training across configurations
(K_data, n_substeps, lr) in parallel.

Three variants, each a separate Modal container, all running in parallel:

  baseline    : K=4,  n_substeps=1,  lr=1e-4  (slow but stable — what we
                already ran locally; final W2≈0.61)
  option-1    : K=16, n_substeps=1,  lr=1e-2  (more dataset snapshots →
                smaller Δt → smaller β; allows original lr=1e-2)
  option-2    : K=4,  n_substeps=4,  lr=1e-2  (split each segment into
                4 Trotter substeps → β'=β/2; allows original lr=1e-2)

Both option-1 and option-2 should yield the same effective β = β(N=16),
but option-1 introduces extra dataset snapshots and changes the loss
function (more terms), while option-2 keeps the loss unchanged.

Usage:
    modal run scripts/exploration/modal_v_mps_n32_compare.py::sweep

Setup (one-time):
    pip install modal
    modal setup
"""
from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "tnwf-v-mps-n32-compare"
REMOTE_APP_DIR = "/app"
REMOTE_OUT_DIR = "/output"

_LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
_MLP_CKPT_N32 = "data/petals_2d/jam_trotter_loss_N32/seed0.pt"
_MLP_CKPT_N16 = "data/petals_2d/jam_trotter_loss/seed0.pt"
_MLP_CKPT_N64 = "data/petals_2d/jam_trotter_loss_N64/seed0.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "torch>=2.2",
        "matplotlib>=3.8",
        "scikit-learn>=1.4",
    )
    .add_local_dir(
        _LOCAL_ROOT,
        REMOTE_APP_DIR,
        ignore=[
            ".git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache/**",
            "results/figures/**",
            ".venv/**",
            "data/eb_5d/**",
        ],
    )
    # Checkpoints are .gitignored so add_local_dir skips them; pull explicitly.
    .add_local_file(_LOCAL_ROOT / _MLP_CKPT_N32,
                    f"{REMOTE_APP_DIR}/{_MLP_CKPT_N32}")
    .add_local_file(_LOCAL_ROOT / _MLP_CKPT_N16,
                    f"{REMOTE_APP_DIR}/{_MLP_CKPT_N16}")
    .add_local_file(_LOCAL_ROOT / _MLP_CKPT_N64,
                    f"{REMOTE_APP_DIR}/{_MLP_CKPT_N64}")
)

app = modal.App(APP_NAME, image=image)
output_vol = modal.Volume.from_name("tnwf-v-mps-out", create_if_missing=True)


@app.function(cpu=4, memory=8192, timeout=28800,
              volumes={REMOTE_OUT_DIR: output_vol})
def run_variant(variant: dict) -> dict:
    """Run one (K_data, n_substeps, lr) variant of N=32 V_mps training."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import sys
    sys.path.insert(0, f"{REMOTE_APP_DIR}/src")
    sys.path.insert(0, f"{REMOTE_APP_DIR}/scripts/exploration")

    import time
    import numpy as np
    import torch

    from adjoint_trotter import (
        make_grid_centred,
        trotter_coefficients,
    )
    from adjoint_trotter_mps import forward_with_cache_mps
    from train_v_mps_end_to_end import (
        _AdamMPS,
        _init_V_mps_list,
        _train_step,
        bin_samples_to_density,
    )
    from v_mps_parameterization import V_mps_to_dense
    from tnwf.data.petals import sample_petals_trajectory
    from tnwf.jam.scalar_potential import ScalarPotentialMLP
    from tnwf.metrics.wasserstein import wasserstein_2_subsampled
    from tnwf.mps.core import right_canonicalize, sample_mps_indices

    label = variant["label"]
    K_data = variant["K_data"]
    n_substeps = variant["n_substeps"]
    lr = variant["lr"]
    n_iter = variant.get("n_iter", 5000)
    N = variant.get("N", 32)
    L = 4.0
    d = 2
    D_max = variant.get("D_max", 16)
    D_V = variant.get("D_V", 16)
    D_V_param = variant.get("D_V_param", 16)
    mlp_ckpt_relpath = variant.get(
        "mlp_ckpt", "data/petals_2d/jam_trotter_loss_N32/seed0.pt"
    )

    np.random.seed(0); torch.manual_seed(0)
    delta_t_substep = 1.0 / (K_data * n_substeps)
    alpha, beta = trotter_coefficients(delta_t_substep, N, d, L)

    target_traj = sample_petals_trajectory(
        n_per_step=2000, seed=0, n_timepoints=K_data + 1
    )
    q_grids_np = [bin_samples_to_density(target_traj[k], N, d, L)
                  for k in range(K_data + 1)]
    psi0_np = np.sqrt(q_grids_np[0]).astype(np.complex128)
    psi0_np /= np.linalg.norm(psi0_np)
    target_world = target_traj + L / 2.0

    print(f"[{label}] N={N} K_data={K_data} n_substeps={n_substeps} "
          f"lr={lr} α={alpha:.4f} β={beta:.4f}")

    # MLP baseline through this pipeline
    mlp_ckpt = f"{REMOTE_APP_DIR}/{mlp_ckpt_relpath}"
    ckpt = torch.load(mlp_ckpt, weights_only=False)
    cfg = ckpt["config"]
    model = ScalarPotentialMLP(
        d=d, hidden=cfg.get("hidden", 256),
        time_embed_dim=cfg.get("time_embed_dim", 64),
        L=L, n_layers=cfg.get("n_layers", 5),
    )
    model.load_state_dict(ckpt["model_state"]); model.eval()
    grid_centred = make_grid_centred(N, d, L, "cpu").to(torch.float32)
    V_grids_mlp = []
    with torch.no_grad():
        for k in range(K_data):
            t_mid = float((k + 0.5) / K_data)
            t_tensor = torch.full((grid_centred.shape[0], 1), t_mid,
                                  dtype=torch.float32)
            V_grids_mlp.append(model(grid_centred, t_tensor).reshape(-1)
                                .to(torch.float64).numpy())

    def _sample_pipeline(V_grids):
        cache = forward_with_cache_mps(
            psi0_np, V_grids,
            alpha=alpha, beta=beta, N=N, d=d, L=L,
            D_max=D_max, D_V=D_V, n_substeps=n_substeps,
        )
        rng = np.random.default_rng(0)
        snaps = []
        for k in range(K_data + 1):
            mps_k = cache[k * n_substeps * 8]
            rc = right_canonicalize(mps_k)
            idx = sample_mps_indices(rc, 200, rng)
            dx = L / N
            jitter = rng.uniform(0.0, 1.0, idx.shape)
            snaps.append(((idx + jitter) * dx).astype(np.float64))
        return np.stack(snaps, axis=0)

    snaps_mlp = _sample_pipeline(V_grids_mlp)
    w2_mlp, _ = wasserstein_2_subsampled(
        snaps_mlp[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)
    print(f"[{label}] MLP-through-pipeline W2@t=1 = {w2_mlp:.4f}")

    # V_mps from-scratch
    rng_init = np.random.default_rng(0)
    V_mps_list = _init_V_mps_list(K_data, N, d, D_V_param=D_V_param,
                                   rng=rng_init)
    flat = [c for cores in V_mps_list for c in cores]
    opt = _AdamMPS(flat, lr=lr)

    t0 = time.time()
    losses = []
    for it in range(n_iter):
        loss = _train_step(
            V_mps_list, opt, psi0_np, q_grids_np,
            K_data, alpha, beta, N, d, L, D_max, D_V,
            n_substeps=n_substeps,
        )
        losses.append(loss)
        if (it + 1) % 500 == 0:
            recent = float(np.mean(losses[-500:]))
            print(f"[{label}] iter {it + 1}/{n_iter}  loss={recent:.5f}  "
                  f"({(it + 1) / (time.time() - t0):.1f} it/s)")
    wall = time.time() - t0

    V_grids_mps = [V_mps_to_dense(V_mps_list[k], N, d) for k in range(K_data)]
    snaps_mps = _sample_pipeline(V_grids_mps)
    w2_mps, _ = wasserstein_2_subsampled(
        snaps_mps[K_data].astype(np.float64),
        target_world[K_data].astype(np.float64),
        n_sub=200, n_repeats=3, seed=0)

    out = {
        "label": label,
        "config": variant,
        "loss_history": losses,
        "final_loss": float(np.mean(losses[-100:])),
        "w2_mlp_t1": float(w2_mlp),
        "w2_mps_t1": float(w2_mps),
        "wall_seconds": wall,
        "snaps_mlp": snaps_mlp.tolist(),
        "snaps_mps": snaps_mps.tolist(),
        "target_world_t1": target_world[K_data].tolist(),
    }

    # Save to volume for download
    import json
    out_path = f"{REMOTE_OUT_DIR}/{label}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    output_vol.commit()
    print(f"[{label}] DONE  W2 MLP={w2_mlp:.4f}  W2 V_mps={w2_mps:.4f}  "
          f"final loss={out['final_loss']:.4f}  wall={wall:.1f}s")
    return {
        "label": label,
        "final_loss": out["final_loss"],
        "w2_mlp_t1": out["w2_mlp_t1"],
        "w2_mps_t1": out["w2_mps_t1"],
        "wall_seconds": wall,
    }


@app.local_entrypoint()
def sweep():
    """Bracket the convergence range for options 1 and 2 at lower lr.

    Previous run diverged at lr=1e-2 because gradient magnitude scales
    with K_data (option 1) or n_substeps (option 2) — both ~4× larger
    than baseline. So both need lr smaller than baseline's 1e-4, not
    larger. This sweep covers lr ∈ {1e-3, 1e-4} for each option.
    """
    variants = [
        {"label": "option1_K16_M1_lr1e-3",
         "K_data": 16, "n_substeps": 1, "lr": 1e-3, "n_iter": 3000},
        {"label": "option1_K16_M1_lr1e-4",
         "K_data": 16, "n_substeps": 1, "lr": 1e-4, "n_iter": 3000},
        {"label": "option2_K4_M4_lr1e-3",
         "K_data": 4,  "n_substeps": 4, "lr": 1e-3, "n_iter": 3000},
        {"label": "option2_K4_M4_lr1e-4",
         "K_data": 4,  "n_substeps": 4, "lr": 1e-4, "n_iter": 3000},
    ]
    print(f"Launching {len(variants)} variants in parallel on Modal...")
    results = list(run_variant.map(variants))
    print("\n=== Summary ===")
    print(f"{'variant':35s} {'final loss':>10} {'W2 MLP':>8} {'W2 V_mps':>10} {'wall (s)':>10}")
    for r in results:
        print(f"{r['label']:35s} {r['final_loss']:>10.4f} "
              f"{r['w2_mlp_t1']:>8.4f} {r['w2_mps_t1']:>10.4f} "
              f"{r['wall_seconds']:>10.1f}")


@app.local_entrypoint()
def n64_d64():
    """N=64 full-bond (D=64) — the central-config completion at the largest N
    we test. ~3-5 hour wall time per container (SVD scales as D³).

    Uses ``.spawn()`` rather than ``.map()`` so the job survives local-client
    disconnects under ``--detach``. Output lands in the Modal volume; pull
    via ``modal volume get tnwf-v-mps-out N64_D64_fullbond_fromscratch.json``.
    """
    variant = {
        "label": "N64_D64_fullbond_fromscratch",
        "N": 64, "K_data": 4, "n_substeps": 1, "lr": 5e-5,
        "n_iter": 2000,
        "D_max": 64, "D_V": 64, "D_V_param": 32,
        "mlp_ckpt": "data/petals_2d/jam_trotter_loss_N64/seed0.pt",
    }
    fn_call = run_variant.spawn(variant)
    print(f"Launched N=64 D=64 on Modal (function call id: {fn_call.object_id})")
    print("Survives local-client disconnect. Result will appear at:")
    print(f"  modal volume get tnwf-v-mps-out N64_D64_fullbond_fromscratch.json")
    print(f"Or fetch the return value later: modal call get {fn_call.object_id}")


@app.local_entrypoint()
def download_results():
    """Download all JSON results from the Modal volume."""
    import subprocess
    local_dir = Path("data/modal_v_mps_out")
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "modal", "volume", "get", "tnwf-v-mps-out", "/", str(local_dir),
        "--force",
    ], check=True)
    print(f"Results downloaded to {local_dir}")
