# tnwf — Scalable quantum simulation of continuous-time generative models via tensor networks

Reproduction code for the paper **"Scalable quantum simulation of continuous-time
generative models via tensor networks"** (npj Quantum Information,
in submission).

> 📄 Paper: _link to appear (arXiv / journal)_
> 🔖 If you use this code, please cite the paper — see [`CITATION.cff`](CITATION.cff).

![Dense wavefunction evolution on the Swiss-roll target](assets/fig2_dense.png)

*Paper Fig. 2 — dense wavefunction evolution (N=64) across generation time. Top:
the complex wavefunction ψ(t) (hue = arg ψ, brightness = |ψ|). Bottom: the
probability mass |ψ(t)|². The dynamics under Hᶜ = i[K, Vₜ] smoothly transport the
Gaussian source into the target distribution.*

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
4. **MPS-V + 2TDVP** — a velocity potential pre-trained as a tensor train
   (`tnwf.mps_v`) fed directly to the 2-site TDVP V-step, with **no runtime
   tensor-cross** (the trained-V "bypass"). Train one with
   `python -m tnwf.mps_v.train`; run via `run(method="mps_v_tdvp2", ...)`.

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

## Demo notebook

[`examples/tnwf_demo.ipynb`](examples/tnwf_demo.ipynb) is a self-contained tour of
the V-step methods (no training required — it uses the closed-form analytic GMM
potential). **It ships with its outputs rendered, so the figures are viewable
directly on GitHub without running anything:**

- **Part 1** — `Dense` vs `TCI+1TDVP` on a 3-D Gaussian mixture, compared by a
  t-SNE overlay against the target (reproduces paper **Fig. 5**, left / *d*=3).
- **Part 2** — the **V-MPS + 2TDVP** pipeline at *d*=8, σ=0.5: a trained
  tensor-train velocity potential (MPS-V) fed directly to 2-site TDVP with no
  runtime cross. Loads the paper's **Table 2** run (N=32, K=160) — grid-unbiased
  sliced-Wasserstein **0.036 ± 0.003** (at the sample-size floor 0.030),
  self-limiting bond χ\* ≈ 16 — and a t-SNE from the saved final state. Training
  and generation are standalone scripts under [`scripts/mps_v/`](scripts/mps_v);
  the shipped `.npz` ([`examples/checkpoints/`](examples/checkpoints)) is the
  paper run's per-step metrics plus samples of its final wavefunction, so no
  multi-hour re-run is needed. Reproduce from scratch with
  `scripts/mps_v/generate.py --K 160`.

```bash
make notebook        # or: uv run --with jupyter jupyter notebook examples/tnwf_demo.ipynb
```

(Jupyter is pulled in on demand via `uv run --with`, so it stays out of the
project's locked dependencies.)

## Reproducing the paper figures

This repository reproduces the manuscript's Swiss-roll / Gaussian-mixture
results. Each generator writes to a relative `figures/` path by default
(override with `--out`):

The manuscript has **8 figures + 2 tables**; the supplement adds **5 figures + 2
tables**. The table below is exhaustive, so what is and is not released here is
unambiguous.

**Filenames do not match float numbers.** `fig3_tsne.pdf` is Fig 5 and
`fig4_cost_scaling.pdf` is Fig 6 — those names date from an earlier draft ordering.
Float numbers below follow the manuscript.

### Main text

| Float | Asset | Generator | Status |
|---|---|---|---|
| Fig 1 | `fig1_overview_tikz.pdf` | — | TikZ, built in the LaTeX source |
| Fig 2 | `fig2_dense.pdf` | `make_fig2_dense_evolution.py` | ✅ needs `jam/seed0.pt` |
| Fig 3 | `fig_gmm_ode.pdf` | `make_fig_gmm_ode.py` | ✅ self-contained |
| Fig 4 | `fig_scaling_bounds.pdf` | `make_fig_scaling_bounds.py` | ✅ needs `scaling_theory` CSVs |
| Fig 5 | `fig3_tsne.pdf` | `make_fig3_tsne_grid.py` | ✅ needs sweeps |
| Table 1 | `table1_sw.tex` | `make_table1_sw.py` | ✅ needs sweeps |
| Fig 6 | `fig4_cost_scaling.pdf` | `make_fig4.py` | ✅ needs sweeps |
| Fig 7 | `fig_ksweep_trained_std0.5_noHyb_wt.pdf` | *not released* | cloud (Modal) timing driver |
| Table 2 | `table_scaling.tex` | `make_table2_scaling.py` | ✅ needs `timing_scaling` + MPS-V checkpoints |
| Fig 8 | `fig_rare_event_advantage.pdf` | `make_fig_rare_event.py` | ✅ needs MPS-V checkpoints |

### Supplement

| Float | Asset | Generator | Status |
|---|---|---|---|
| Table S1 | — | — | hand-written tabular in the LaTeX source |
| Fig S1 | `figS1_supp_pareto.pdf` | `make_fig_supp_pareto.py` | ✅ needs sweeps |
| Table S2 | `table_core4_audit.tex` | *not released* | cloud (Modal) audit driver |
| Fig S2 | `fig_panelA_sigma.pdf` | *not released* | intermediate run outputs not preserved |
| Fig S3 | `fig_panelN_resolution.pdf` | *not released* | intermediate run outputs not preserved |
| Fig S4 | `fig_panelD_ksigma.pdf` | *not released* | cloud (Modal) timing driver |
| Fig S5 | `fig_analytic_vs_trained_std0.5_wt.pdf` | *not released* | intermediate run outputs not preserved |

**Fig 2–6, Fig 8, Tables 1–2 and Fig S1 all reproduce exactly** — every float the
release covers is byte- or content-identical to the manuscript. Fig 1 and Table S1
are built by LaTeX. The remaining **3 float slots are not reproducible from this
release** (Fig 7, Table S2, Fig S2–S5 — 6 assets), all generated by cloud drivers
or from intermediate outputs that were not preserved.

Fig 4 ships only the *plotting* half of the analytic scaling study: the
`data/scaling_theory/` CSV cache is part of the data archive below, but the
compute pass that produces it depends on internal solver code that is not
released. The figure is exactly reproducible from the cache; its numbers are not
re-derived.

## Getting the data archive

Everything above marked "needs …" reads from `data/`, which is **not in this
repository** — the hyperparameter sweeps alone are ~155 MB. It is archived
separately with a DOI:

The archive is deposited with **restricted** access: the DOI and its metadata
are public and citable, but the files are released on request (see
[`zenodo.json`](zenodo.json)). Everything needed to regenerate every released
figure and table from it is in this repository.

<!-- TODO(author): replace ZENODO_RECORD once the deposit is published. -->

```bash
cd tnwf-paper
curl -L -o tnwf-paper-data.tar.gz "https://zenodo.org/records/ZENODO_RECORD/files/tnwf-paper-data.tar.gz"

# Verify before unpacking. This digest is for the v2 archive built by
# scripts/package_data_archive.py; the build is byte-reproducible, so
# rebuilding from the same inputs reproduces it exactly.
echo "c10a9173c6b3066e3c91b35eb6037faccc5ddd304244a22d38a4ad937ee949ba  tnwf-paper-data.tar.gz" | shasum -a 256 -c

# Unpack AT THE REPO ROOT — every member is rooted at data/, and several
# generators hardcode relative paths like data/gmm_2d_hp/ with no CLI override.
tar xzf tnwf-paper-data.tar.gz

make verify-data          # every file against the archive's own MANIFEST.tsv
```

That creates:

```
data/
├── gmm_2d_hp/  gmm_3d_hp_v2/  gmm_4d_hp/ … gmm_8d_hp/   sweeps  → Fig 5, 6, S1, Table 1
├── scaling_theory/                                      CSVs    → Fig 4
├── timing_scaling/  mps_v_checkpoints/                          → Table 2
└── {swiss_roll_2d,gmm_2d,gmm_3d}/jam/                   JAM ckpts → Fig 2
```

The archive also carries a `MANIFEST.tsv` recording every file's size and
sha256; `make verify-data` checks the unpacked tree against it. (The manifest
travels inside the archive, so that establishes internal consistency, not
provenance — for provenance use the tarball checksum above.) `data/exact_sw_floor.json` (Table 1's target–target row) is small enough
to live in this repository and is already present.

With the archive unpacked, every reproducible float renders:

```bash
make fig2 fig-scaling-bounds fig3 fig4 supp table1 table2
make fig-rare-event                                    # needs MPS-V checkpoints
uv run python scripts/make_fig_gmm_ode.py              # needs no archive data
```

### Notes on the reproducible floats

**Fig 2 needs the trained JAM checkpoint, not merely retraining.** `make jam-swiss`
will train one, but training is stochastic, so a retrained checkpoint yields a
slightly different `V_t` and Fig 2 then differs from the manuscript — verified at
517468 bytes retrained versus 517509 with the original. The original `jam/*.pt`
checkpoints are in the data archive at
`data/{swiss_roll_2d,gmm_2d,gmm_3d}/jam/`; unpacking it is enough.

**Fig 8 needs the MPS-V checkpoint**, unlike earlier releases where it was
self-contained. Its quantum arm reads `examples/rare_event/rep*.npz` — the final
MPS cores from the ten Table-2 replicates (N=32, K=160, D_max=64), converted from
the original pickles so this repository ships no `pickle` payloads. But the
classical arm integrates `grad V` of the *same* learned potential, which lives in
`data/mps_v_checkpoints/` in the data archive. Using one potential for both arms
is what makes the comparison fair: the learning error is then common to the two,
so the figure contrasts the transports and sampling methods rather than two
different velocity fields. `make fig-rare-event` fails with a pointer to the
archive if the checkpoint is absent.

`fallback_K40_D16.npz` is a reduced K=40/D_max=16 state for `--source npz`. The
script writes the manuscript's filename
(`figures/fig_rare_event_advantage.pdf`) directly and defaults to the paper's
`--tail-k 4.0`, so a bare `python scripts/make_fig_rare_event.py` reproduces the
figure. Takes ~1 min at the default 40,000 samples per arm.

**Fig 5 palette.** The color stops in `make_fig3_tsne_grid.py` are pinned as
explicit magma triples (LUT indices 38 / 128 / 199) rather than `plt.cm.magma(f)`
calls. magma is a 256-entry lookup table, so `magma(f)` is piecewise constant in
`f` and an approximate `f` lands on a neighboring stop. Leave them pinned.

## Layout

```
tnwf/
├── src/tnwf/             # package: jam/, mps_v/, mps/, mpo/, dense/, metrics/, data/, pipelines/, amp.py
├── tests/                # pytest suite mirroring src/tnwf/
├── scripts/make_*.py     # one generator per paper float
├── scripts/{dataset}/    # per-dataset runners (train_jam, run_all_methods)
├── examples/rare_event/  # Fig 8's MPS cores (the only bulk data tracked here)
├── pyproject.toml        # package + dependency declarations
└── uv.lock               # fully pinned dependency lock
```

Everything under `scripts/` either generates a manuscript float or produces the
sweep data one consumes; there are no draft-only generators. `figures/`,
`results/` and the bulk of `data/` are generated or downloaded, and are not
version-controlled — see the archive section above.

## Dependencies & licenses

Dependencies are pinned in [`uv.lock`](uv.lock), which records the exact
resolved version of every direct and transitive dependency.
[`licenses/THIRD_PARTY_LICENSES.md`](licenses/THIRD_PARTY_LICENSES.md) reports
each one's license, regenerated from the lock by `make licenses`
(`make licenses-check` exits non-zero if anything non-permissive appears).

As of the current lock, the 36 packages reachable on macOS and Windows are all
permissively licensed — no copyleft of any kind. The remaining 17 are the
NVIDIA CUDA runtime, pulled in transitively by `torch` behind
`sys_platform == 'linux'`; they are proprietary, are not installed on other
platforms, and are not fetched by the CPU-only reproduction path documented
above. This repository vendors no third-party code.

## License

MIT — see [`LICENSE`](LICENSE).
