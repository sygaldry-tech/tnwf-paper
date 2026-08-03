# Repository structure and metrics

A map of what is in this repository, what each part is for, and what it takes to
reproduce the manuscript from it. [`README.md`](README.md) is the user-facing
instructions; [`LOG.md`](LOG.md) is the historical record of the pre-publication
review. This file is the current-state reference.

Float numbers follow the **manuscript**, not the asset filenames, which disagree:
`fig3_tsne.pdf` is Fig 5 and `fig4_cost_scaling.pdf` is Fig 6.

---

## At a glance

| | |
|---|---|
| Tracked files | **96** |
| Python | **7,381 LOC** — 4,010 package / 927 tests / 2,444 generators & runners |
| Tests | **74**, all passing (69 `needle` + 5 `medium`), ~24 s |
| Tracked payload | **6.4 MB**, of which 5.6 MB is `examples/` (Fig 8's MPS cores + the demo notebook) |
| Direct dependencies | **5** (numpy, scipy, torch, matplotlib, scikit-learn); 54 packages locked |
| Python | ≥ 3.11 (verified on 3.14.3) |
| External data | **151.5 MiB** archive, downloaded separately |
| Manuscript floats reproduced | **9 of 9** reproducible ones, byte- or content-identical |

## Repository map

```
tnwf-paper/
├── README.md              installation, float table, data-archive download
├── LOG.md                 pre-publication review record
├── STRUCTURE.md           this file
├── LICENSE                MIT
├── CITATION.cff           software + article citation metadata
├── Makefile               one target per float, plus training/runner targets
├── pyproject.toml         5 direct dependencies
├── uv.lock                54 packages, fully pinned
├── conftest.py            pytest fixtures (--save-plots opt-in)
│
├── src/tnwf/              4,010 LOC — the package
├── tests/                 927 LOC, 74 tests, mirrors src/tnwf/
├── scripts/               2,444 LOC — 9 float generators + dataset runners
├── examples/              Fig 8's data, the demo notebook, an MPS-V checkpoint
├── assets/                README image
└── data/                  exact_sw_floor.json only; the rest is downloaded
```

## Package: `src/tnwf/`

`pipelines/run_evolution.py` is the hub — every method dispatches through it, and
everything else is either something it calls or something that consumes its
output.

| Module | LOC | Role |
|---|---:|---|
| `pipelines/run_evolution.py` | 841 | The pipeline: initial state → K Trotter steps → per-step metrics → `.npz` |
| `leaderboard.py` | 498 | Aggregates runner outputs into a results table |
| `jam/train.py` | 361 | JAM training; `load_jam`, `make_V_fn`, gradient-flow sampling |
| `mps/tdvp.py` | 311 | 1- and 2-site TDVP, plus the `_robust_svd` fallback chain |
| `mps/tt_cross.py` | 287 | TT-cross (SVD/MAXVOL pivots) — behind every TCI result |
| `qae.py` | 205 | MLQAE and classical MC estimators (Fig 8) |
| `jam/scalar_potential.py` | 200 | The JAM MLP: periodic encoding, time embedding |
| `mps/core.py` | 189 | MPS primitives: canonicalisation, truncation, sampling, K-step |
| `mps_v/model.py` | 185 | Trained tensor-train velocity potential |
| `mps_v/train.py` | 135 | MPS-V training loop |
| `theory.py` | 132 | Closed-form analytic GMM potential |
| `mpo/build_evolution_mpo.py` | 121 | Builds the V-MPO consumed by the TDVP V-steps |
| `dense/evolution.py` | 107 | Exact `O(N^d)` reference path |
| `mps_v/__init__.py` | 81 | `load_mps_v`, the trained-V bypass provider |
| `mps/vstep_tdvp.py` | 67 | TCI+1TDVP and TCI+2TDVP V-step wrappers |
| `metrics/` | 185 | `sw` (drives every headline number), `mmd`, `nll`, `wasserstein` |
| `data/` | 69 | Gaussian-mixture and Swiss-roll samplers |
| `grid.py` | 34 | Shared grid and kinetic eigenvalues |

### The four V-step methods

All four share the same 8-step commutator product formula in `run_evolution.py`
and differ only in how they apply `exp(iβV_t)`:

1. **`dense`** — exact `O(N^d)` reference. Infeasible beyond `N^d ≈ 10⁹`.
2. **`tci_tdvp1`** — TT-cross MPO + 1-site TDVP; bond fixed, no SVD.
3. **`tci_tdvp2`** — TT-cross MPO + 2-site TDVP; bond grows to `D_max` via SVD.
4. **`mps_v_tdvp2`** — a pre-trained tensor-train `V_t` fed straight to 2-site
   TDVP, with **no runtime tensor-cross** (the trained-V bypass).

Plus `jam`, the classical baseline: `dx/dt = ∇V_t(x)` by Euler integration.

## Floats → generators → data

Nine of the manuscript's 17 floats are reproducible from this repository. Two are
built by LaTeX. Six are not reproducible, by disclosure — their cloud drivers and
intermediate outputs were not preserved.

| Float | Asset | Generator | Needs |
|---|---|---|---|
| Fig 1 | `fig1_overview_tikz.pdf` | — | LaTeX TikZ |
| **Fig 2** | `fig2_dense.pdf` | `make_fig2_dense_evolution.py` | `data/swiss_roll_2d/jam/seed0.pt` |
| **Fig 3** | `fig_gmm_ode.pdf` | `make_fig_gmm_ode.py` | nothing — self-contained |
| **Fig 4** | `fig_scaling_bounds.pdf` | `make_fig_scaling_bounds.py` | `data/scaling_theory/*.csv` |
| **Fig 5** | `fig3_tsne.pdf` | `make_fig3_tsne_grid.py` | 6 cells in `gmm_{3,5,7}d_hp*` |
| **Table 1** | `table1_sw.tex` | `make_table1_sw.py` | 7 sweeps + `data/exact_sw_floor.json` |
| **Fig 6** | `fig4_cost_scaling.pdf` | `make_fig4.py` | 7 sweeps |
| Fig 7 | `fig_ksweep_*.pdf` | *not released* | cloud (Modal) driver |
| **Table 2** | `table_scaling.tex` | `make_table2_scaling.py` | `timing_scaling` + `mps_v_checkpoints` |
| **Fig 8** | `fig_rare_event_advantage.pdf` | `make_fig_rare_event.py` | nothing — `examples/rare_event/` |
| Table S1 | — | — | LaTeX tabular |
| **Fig S1** | `figS1_supp_pareto.pdf` | `make_fig_supp_pareto.py` | `gmm_{3,4,5}d_hp*` |
| Table S2, Fig S2–S5 | — | *not released* | cloud drivers / unpreserved |

`scripts/_hp_utils.py` (165 LOC) is the shared sweep loader behind Table 1,
Fig 6 and Fig S1: it globs `{method}/N*_K*_D*/seed*.npz`, reads `sw`, `mmd`,
`chi_max`, `total_time`, `d`, and averages over seeds. It also holds the memory
model (`mps_param_count`, `memory_fraction`) that Fig 6 plots.

Two generators (`make_fig4.py`, `make_fig_supp_pareto.py`) hardcode relative
`data/…` paths with no CLI override — which is why the archive must be unpacked
at the repository root. `make_table1_sw.py` and `make_table2_scaling.py` accept
`--out` and directory overrides.

## Dataset runners

`scripts/{swiss_roll_2d,gmm_2d,gmm_3d}/` each hold `train_jam.py` and
`run_all_methods.py`. These regenerate sweep data rather than a float directly;
`make jam-*` and `make run-*` drive them.

Fig 2 is the one place this matters: it needs the *original* JAM checkpoint, not
a retrained one. Training is stochastic, so retraining yields a different `V_t`
and a visibly different Fig 2 (517468 bytes retrained vs 517509 original). The
originals ship in the data archive.

## Tests

74 tests, 927 LOC, mirroring the package layout. `needle` (69) is the fast tier;
`medium` (5) exercises the full pipeline.

| File | Tests | | File | Tests |
|---|---:|---|---|---:|
| `data/test_loaders.py` | 11 | | `mps/test_tt_cross.py` | 5 |
| `metrics/test_metrics.py` | 10 | | `jam/test_scalar_potential.py` | 5 |
| `mps/test_core.py` | 10 | | `test_leaderboard.py` | 4 |
| `dense/test_evolution.py` | 9 | | `pipelines/test_run_evolution.py` | 3 |
| `test_grid.py` | 7 | | `mps/test_vsteps.py` | 2 |
| `test_theory.py` | 6 | | `jam/test_train.py` | 1 |

```sh
make test-needle    # < 30 s
make test-medium    # full pipeline
make test           # both
```

**Known gaps.** `src/tnwf/qae.py` and `src/tnwf/mps_v/` have no direct unit
tests; both are covered only end-to-end, by Fig 8 and Table 2 reproducing their
published numbers. Worth adding if this code gets reused beyond the paper.

## The data archive

Not in this repository — 155 MB of sweep outputs, archived separately with a DOI.

| | |
|---|---|
| File | `tnwf-paper-data.tar.gz` |
| Size | 158,855,577 B (151.5 MiB) |
| sha256 | `baa5744f184dbae19136f5f68260eaed783332990ca05e6b1554217ed41cef86` |
| Contents | 2,152 files, 155.1 MB unpacked, all rooted at `data/` |

```
data/
├── gmm_2d_hp/ gmm_3d_hp_v2/ … gmm_8d_hp/   2,069 sweep .npz → Fig 5, 6, S1, Table 1
├── scaling_theory/                          9 CSVs          → Fig 4
├── timing_scaling/wf_eval/                  40 traj.npz     → Table 2
├── mps_v_checkpoints/                       4 .pt oracles   → Table 2
└── {swiss_roll_2d,gmm_2d,gmm_3d}/jam/       30 .pt          → Fig 2
```

Each sweep `.npz` carries `sw`, `mmd`, `nll`, `chi_max`, `samples_T`, `target`,
`samples_per_step`, `step_times`, `total_time` plus scalar config — pure numeric
arrays and short strings, no pickled objects and no host paths. The archive
includes a `MANIFEST.tsv` with per-file size and sha256.

## Deliberately absent

Not oversights — each was checked against the manuscript and excluded:

- **The QTT / quantics workstream** (`src/tnwf/qtt/`, its test, and
  `scripts/exploration/`). A separate line of work that no float uses; the
  binary-folded MPS-V variant raises `NotImplementedError` to keep that true.
- **A torch/GPU TDVP backend.** Existed but was unreachable — nothing set
  `device`, and its test was unmarked so the suite skipped it.
- **Draft-era figure generators.** Every `scripts/make_*.py` now feeds a
  current float.
- **`samples_per_step` consumers.** The field is ~70% of the archive by size and
  is written by the pipeline, but no generator reads it. It is kept in the
  archive as part of the run record.
- **Pickle payloads.** Fig 8's MPS cores were converted to `.npz`; the archive
  staging filters `ck.pkl` out.

## What the review changed

| | Before | After |
|---|---:|---:|
| Tracked files | 85 | 96 |
| Package LOC | 4,480 | 4,010 |
| Test LOC | 1,046 | 927 |
| Generator/runner LOC | 2,091 | 2,444 |
| **Total Python** | **7,617** | **7,381** |
| Floats reproducible from a clone | **5 of 9** | **9 of 9** |
| Lint findings | 102 | 66 |

The package shrank while the reproducible surface grew: 470 lines came out of
`src/` (an unreachable GPU backend and three zero-caller functions) and 501 out
of `scripts/` (draft generators), while 981 lines of genuinely missing float
generators came in. `make_fig_rare_event.py` alone went 866 → 526 lines and
~5 min → 36 s by dropping five plotting passes that produced no manuscript float.

The 66 remaining lint findings are pre-existing and style-only — mostly `E7xx`
and the deliberate `E402` from the `sys.path.insert` idiom the figure scripts use
to import `_hp_utils`.

## Reproducing everything

```sh
git clone https://github.com/sygaldry-tech/tnwf-paper.git && cd tnwf-paper
# download + verify + unpack the archive (see README)
uv sync --extra dev
uv run pytest -m "needle or medium"

make fig2 fig-scaling-bounds fig3 fig4 supp table1 table2   # needs the archive
make fig-rare-event                                         # ~40 s, no archive
uv run python scripts/make_fig_gmm_ode.py                   # no archive
```

Verified end to end from a clean clone: 74 tests pass and all nine floats match
the manuscript — seven content-identical PDFs (equal after normalising
`/CreationDate`) and two byte-identical `.tex` tables. Per-float results are in
[`LOG.md`](LOG.md) §4.
