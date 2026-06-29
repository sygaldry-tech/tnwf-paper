# tnWF — Exponentially-accelerated simulations of wavefunction flows via tensor networks

Reproduction code for the paper **"Exponentially-accelerated simulations of
wavefunction flows via tensor networks"** (npj Quantum Information, in submission).

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
2. **TCI+TDVP1** — TT-cross MPO + 1-site TDVP V-step
3. **TCI+TDVP2** — TT-cross MPO + 2-site TDVP V-step

Background notes: [`THEORY.md`](THEORY.md).

## Installation

This project uses [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
uv sync --extra dev
```

## Quickstart

```bash
# Sanity tests (fast)
uv run pytest -m needle              # < 30s
uv run pytest -m medium              # < 5min, full pipeline

# Train a JAM potential and run the methods on a 2D dataset
uv run python -m tnwf.jam.train --dataset swiss_roll_2d --seed 0
uv run python scripts/swiss_roll_2d/run_all_methods.py --seeds 0,1,2
uv run python scripts/make_fig2_dense_evolution.py   # writes figures/fig2_dense.pdf
```

Datasets are generated on the fly: `swiss_roll_2d` and `gmm_2d … gmm_16d`.

## Reproducing the paper figures

This repository reproduces the manuscript's Swiss-roll / Gaussian-mixture
results. Each generator writes to a relative `figures/` path by default
(override with `--out`):

| Paper float | Script |
|---|---|
| Fig 1 (overview)        | `scripts/make_fig1_combined.py` |
| Fig 2 (dense evolution) | `scripts/make_fig2_dense_evolution.py` |
| Fig 3 (t-SNE grid)      | `scripts/make_fig3_tsne_grid.py` |
| Fig 4 (cost scaling)    | `scripts/make_fig4.py` |
| Fig S1 (supp. Pareto)   | `scripts/make_fig_supp_pareto.py` |
| fig_gmm_ode             | `scripts/make_fig_gmm_ode.py` |
| Table 3 (Pareto configs)| `scripts/make_table3_pareto_configs.py` |

**Data provenance.** Figs 1–2 and `fig_gmm_ode` reproduce locally: the
per-dataset runners
(`scripts/{swiss_roll_2d,gmm_2d,gmm_3d}/{train_jam,run_all_methods}.py`) write
to `data/{swiss_roll_2d,gmm_2d,gmm_3d}/…`, which those generators read.

Figs 3–4, Fig S1, and Table 3 instead read the hyperparameter-sweep outputs
under `data/gmm_*_hp*/` (e.g. `data/gmm_5d_hp`, `data/gmm_3d_hp_v2`). Those
sweeps spanned dimensions up to `gmm_8d` and were run on a large parallel
cloud-compute backend; **the sweep outputs and their drivers are not included
in this repository.** The generator scripts are provided so the exact figures
can be reproduced from sweep outputs of the same layout.

## Layout

```
tnWF/
├── src/tnwf/             # package: jam/, mps/, mpo/, dense/, metrics/, data/, pipelines/
├── tests/                # pytest suite mirroring src/tnwf/
├── scripts/{dataset}/    # per-dataset runners + figure generators
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
