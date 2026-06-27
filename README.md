# tnWF — Scalable Simulation of Wavefunction Flows via Tensor Networks

Reproduction code for the paper **"Scalable Simulation of Wavefunction Flows via
Tensor Networks"** (npj Quantum Information, in submission).

> 📄 Paper: _link to appear (arXiv / journal)_
> 🔖 If you use this code, please cite the paper — see [`CITATION.cff`](CITATION.cff).

## Overview

We compress the position-diagonal potential ("V-step") of a wavefunction-flow
evolution with tensor-network methods, evolving a wavefunction MPS under
`exp(iβ V_t)`. The pipeline is:

**JAM** (Joint Action Matching — learns a conservative scalar potential `V_t`)
→ **V-step** methods that apply `exp(iβ V_t)` to the MPS.

### V-step methods

1. **Dense** — exact `O(N^d)` reference
2. **TCI+ALS** — TT-cross initialization + variational ALS sweeps
3. **ACI** — adaptive cross interpolation (Hadamard), `O(χ³)`
4. **TCI+TDVP1** — TT-cross MPO + 1-site TDVP V-step
5. **TCI+TDVP2** — TT-cross MPO + 2-site TDVP V-step
6. **ACI+TDVP1** — ACI-built MPO + 1-site TDVP
7. **ACI+TDVP2** — ACI-built MPO + 2-site TDVP

Background notes: [`THEORY.md`](THEORY.md).

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
uv sync --all-extras
```

## Quickstart

```bash
# Sanity tests (fast)
uv run pytest -m needle              # < 30s
uv run pytest -m medium              # < 5min, full pipeline

# Train a JAM potential and run the methods on a 2D dataset
uv run python -m tnwf.jam.train --dataset swiss_roll_2d --seed 0
uv run python scripts/swiss_roll_2d/run_all_methods.py --seeds 0,1,2
uv run python scripts/swiss_roll_2d/make_fig1.py
```

Datasets are generated on the fly (`swiss_roll_2d`, `gmm_2d`, `gmm_3d`,
`gmm_d_scaling`, `gmm_N_scaling`, `petals_2d`, `eb_5d`); MNIST is downloaded
automatically by `torchvision` on first use.

## Reproducing figures

Per-dataset runners live under `scripts/{dataset}/`; cross-dataset figure
generators are the top-level `scripts/make_fig*.py`. Each writes to a relative
`figures/` path by default (override with `--out`). Example:

```bash
uv run python scripts/make_fig4.py            # cost-scaling figure
uv run python scripts/make_fig3_tsne_grid.py
```

> **Note:** `scripts/make_fig5*.py` (coarse-MNIST / BAS panels) consume image
> inputs produced by a separate experiment pipeline; point the `inputs/…` paths
> in those scripts at your own generated outputs.

## Layout

```
tnWF/
├── src/tnwf/             # package: jam/, mps/, mpo/, dense/, metrics/, data/, pipelines/
├── tests/                # pytest suite mirroring src/tnwf/
├── scripts/{dataset}/    # per-dataset runners + figure generators
├── modal/                # distributed d/N scaling sweeps (optional, needs Modal)
├── experiments/          # warm-start ablation harness
├── pyproject.toml        # package + dependency declarations
├── uv.lock               # fully pinned dependency lock
└── licenses/             # third-party dependency license manifest
```

Generated artifacts (`results/`, `data/`, `figures/`, checkpoints, `*.npz`) are
not version-controlled — regenerate them with the scripts above.

## Dependencies & licenses

Dependencies are pinned in [`uv.lock`](uv.lock). A best-effort third-party
license manifest is in
[`licenses/THIRD_PARTY_LICENSES.md`](licenses/THIRD_PARTY_LICENSES.md)
(auto-generated from PyPI metadata; not a legal determination).

## License

See [`LICENSE`](LICENSE). _(License pending review — see the file.)_
