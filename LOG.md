# Release review log

A record of the pre-publication code and data review of this repository and its
companion data archive, conducted 2026-08-02 against the manuscript
`sn-article.tex` (npj Quantum Information, in submission).

The goal of the review: **everything in this repository should serve the
reproduction of a manuscript float, and everything needed to reproduce one should
be here.** Anything that met neither test was removed or flagged.

Float numbers throughout follow the **manuscript**, not the asset filenames,
which disagree: `fig3_tsne.pdf` is Fig 5 and `fig4_cost_scaling.pdf` is Fig 6.

---

## 1. Starting state

### 1.1 Git

- Remote: `https://github.com/sygaldry-tech/tnwf-paper.git`
- `HEAD` = `b523bba` ("Retitle: prepend 'Scalable'"), identical to `origin/main`
- **83 tracked files**

```
src/tnwf/     30    tests/       19    scripts/    18
(root)         9    examples/     3    assets/      1    licenses/  1
```

Source size at review time: `src/` 4,746 LOC, `tests/` 1,046 LOC,
`scripts/` 3,270 LOC (excluding the gitignored `scripts/exploration/`).

### 1.2 Working tree relative to `HEAD`

The working tree and the published repository had diverged substantially. The
working tree was the *correct* state; `origin/main` was stale.

| State | Files |
|---|---|
| Modified (7) | `.gitignore`, `LICENSE`, `Makefile`, `README.md`, `scripts/make_fig3_tsne_grid.py`, `src/tnwf/mps/core.py`, `src/tnwf/pipelines/run_evolution.py` |
| Deleted, staged (2) | `licenses/THIRD_PARTY_LICENSES.md`, `scripts/make_table3_pareto_configs.py` |
| Untracked (17) | `scripts/make_table1_sw.py`, `scripts/make_table2_scaling.py`, `scripts/make_fig_scaling_bounds.py`, `scripts/make_fig_rare_event.py`, `src/tnwf/qae.py`, `examples/rare_event/*.npz` (11), `data/exact_sw_floor.json` |

### 1.3 Package structure at review time

```
src/tnwf/
├── __init__.py            version stub
├── grid.py                shared grid / kinetic eigenvalues
├── theory.py              analytic GMM potential (closed form)
├── qae.py                 MLQAE + classical MC estimators      [UNTRACKED]
├── leaderboard.py         runner bookkeeping (no float uses it)
├── data/                  gaussian_mixture.py, swiss_roll.py
├── dense/evolution.py     exact O(N^d) reference
├── jam/                   scalar_potential.py, train.py
├── metrics/               sw.py, mmd.py, nll.py, wasserstein.py
├── mpo/build_evolution_mpo.py
├── mps/                   core.py, tdvp.py, tdvp_torch.py, tt_cross.py, vstep_tdvp.py
├── mps_v/                 model.py, train.py  (trained tensor-train potential)
├── pipelines/run_evolution.py   the hub: all methods dispatch here
└── qtt/                   [GITIGNORED — separate workstream, never shipped]
```

### 1.4 Float inventory (verified against the manuscript)

17 floats: 9 reproducible from this release, 2 built by LaTeX, 6 not reproducible
(cloud drivers or intermediates that were not preserved).

| Float | Asset | Generator | Data needed | At review |
|---|---|---|---|---|
| Fig 1 | `fig1_overview_tikz.pdf` | LaTeX TikZ | — | n/a |
| Fig 2 | `fig2_dense.pdf` | `make_fig2_dense_evolution.py` | `data/swiss_roll_2d/jam/seed0.pt` | tracked |
| Fig 3 | `fig_gmm_ode.pdf` | `make_fig_gmm_ode.py` | none | tracked |
| Fig 4 | `fig_scaling_bounds.pdf` | `make_fig_scaling_bounds.py` | `data/scaling_theory/*.csv` | **untracked** |
| Fig 5 | `fig3_tsne.pdf` | `make_fig3_tsne_grid.py` | 6 cells, `gmm_{3,5,7}d_hp*` | tracked |
| Table 1 | `table1_sw.tex` | `make_table1_sw.py` | 7 sweeps + `exact_sw_floor.json` | **untracked** |
| Fig 6 | `fig4_cost_scaling.pdf` | `make_fig4.py` | 7 sweeps | tracked |
| Fig 7 | `fig_ksweep_*.pdf` | *not released* (Modal driver) | — | n/a |
| Table 2 | `table_scaling.tex` | `make_table2_scaling.py` | `timing_scaling`, `mps_v_checkpoints` | **untracked** |
| Fig 8 | `fig_rare_event_advantage.pdf` | `make_fig_rare_event.py` | `examples/rare_event/` | **untracked** |
| Table S1 | — | LaTeX tabular | — | n/a |
| Fig S1 | `figS1_supp_pareto.pdf` | `make_fig_supp_pareto.py` | `gmm_{3,4,5}d_hp*` | tracked |
| Table S2 | `table_core4_audit.tex` | *not released* | — | n/a |
| Fig S2–S5 | — | *not released* | — | n/a |

### 1.5 Companion data archive

`lab-nathan/projects/accInf_WF/tnwf-data-release` — 2,152 files, 155.1 MB staged
under `data/`, plus a 151.5 MiB `tnwf-paper-data.tar.gz`. Nothing in the
directory is committed to git. Contents map onto the code's hardcoded paths:
`gmm_{2,3_v2,4,5,6,7,8}d_hp*`, `scaling_theory`, `timing_scaling`,
`mps_v_checkpoints`, and `{swiss_roll_2d,gmm_2d,gmm_3d}/jam`.

---

## 2. Findings

Ordered by severity. Each is addressed in §3 unless marked *flagged only*.

### F1 — Four of the nine reproducible floats had no generator on GitHub

`make_table1_sw.py` (Table 1), `make_table2_scaling.py` (Table 2),
`make_fig_scaling_bounds.py` (Fig 4) and `make_fig_rare_event.py` (Fig 8) were
untracked, along with `src/tnwf/qae.py` (Fig 8's only dependency),
`examples/rare_event/*.npz` (Fig 8's data) and `data/exact_sw_floor.json`
(Table 1's target–target row). A clone of `origin/main` reproduced **5** floats,
not 9.

Also uncommitted: the Fig 5 magma-palette correction in
`make_fig3_tsne_grid.py`. Without it Fig 5 renders in a stale Wong/Okabe-Ito
palette and does *not* match the manuscript.

### F2 — The working tree would have shipped a code path that cannot import

The uncommitted `run_evolution.py` diff added `method="qtt"`. Every one of its
branches imports `tnwf.qtt.*` (`_make_qtt_stepper`, and the initial-state,
snapshot and final-sample paths), but `.gitignore:64` excludes `src/tnwf/qtt/`
from the release by design. On a fresh clone `method="qtt"` raises
`ModuleNotFoundError`. No manuscript float uses it.

### F3 — The data archive has no public home, contradicting the manuscript

`sn-article.tex:1157-1158` (Data availability) states the data is "available on
GitHub at `https://github.com/sygaldry-tech/tnwf-paper`". It is not. The 155 MB
archive exists only on the author's disk, with no URL and no DOI, and the
generators that consume it hardcode relative `data/…` paths. As written, the
statement cannot be satisfied.

### F4 — `LICENSE` blocked publication

`LICENSE` read "LICENSE PENDING REVIEW … all rights are reserved by Sygaldry,
Inc. and no permission is granted to use, copy, modify, or distribute this
software." `CITATION.cff` carried `license: "TODO"`.

### F5 — Fig 8 could not be reproduced by running its own script

Two independent defects in `make_fig_rare_event.py`:

- It wrote `figures/rare_event_master.pdf`. The manuscript includes
  `figures/fig_rare_event_advantage.pdf` (`sn-article.tex:672`). The script's own
  docstring (`:29`) documented the equivalence, but no code performed the rename,
  so the file the manuscript needs was never produced.
- `--tail-k` defaulted to `5.0` (`:91`) while the paper's threshold is `4.0`.
  Only the `Makefile` target passed the correct value, so anyone running the
  script directly got a *different figure* with no warning.

### F6 — Unreachable and non-float code in the release

| Item | Why it does not serve reproduction |
|---|---|
| `src/tnwf/mps/tdvp_torch.py` (598 LOC) | Reachable only via `device != "cpu"` in `mps/vstep_tdvp.py:41,81`. Nothing in the repo ever sets `device`; `run_evolution.py:246,261` hardcode `"cpu"`. Its one test carries no `needle`/`medium` marker, so `make test` skipped it. |
| `scripts/make_fig1_combined.py`, `make_fig1_concept.py`, `make_fig2_tensor_networks.py` | Draft-era panels. Fig 1 is TikZ in the LaTeX source; the manuscript's Fig 2 is `fig2_dense` only. |
| `pyproject.toml:10` `torchvision` | Zero uses across `src/`, `scripts/`, `tests/`, `examples/`. |
| `data/gaussian_mixture.py:55 mode_coverage`, `dense/evolution.py:111 build_steps_from_V_fn`, `mps/core.py:193 eval_mps_at_indices`, `metrics/wasserstein.py:20 wasserstein_2` | Zero callers. |

### F7 — `make clean-results` cleaned the wrong directory

`Makefile:clean-results` globbed `results/*/*/*.npz` and `results/*/*/N*/seed*.npz`,
but `run_evolution.py:488,879` writes pipeline outputs under `data/`. Only
`results/leaderboard.{csv,md}` was ever produced where the target looked.

### F8 — Data archive: `.DS_Store` broke `make verify` and would ship

`make_manifest.py` inventories with an unfiltered `rglob("*")`. Three `.DS_Store`
files had appeared after `MANIFEST.tsv` was written, including
`data/.DS_Store` (10,244 B). Consequences: `make verify` fails with a spurious
mismatch (2,153 files found vs 2,152 recorded), and a re-run of `make archive`
would have shipped a macOS Finder artifact into the public data release.

### F9 — Data archive README inaccuracies

- Claimed "All 2078 staged files currently pass the integrity check";
  `MANIFEST.tsv` records **2,152**.
- Claimed "Tracked here: README.md, Makefile, make_manifest.py, MANIFEST.tsv,
  .gitignore". None of the five is tracked — `git ls-files .` is empty.
- Claimed `fetch/` holds "the `ck.pkl` files that are deliberately not staged".
  `fetch/wf_eval/` contains only `traj.npz` (40/40), so the Makefile's
  pickle-exclusion filter is currently a no-op.
- 24 empty sweep-config directories ship in the archive, undocumented.

### F10 — Flagged only: items that are not mine to close

- `CITATION.cff`: author list, ORCIDs, affiliations and DOI are all `TODO`.
- `sn-article.tex:11-12` author/affiliation block and `:1138` acknowledgements
  are `TODO`.
- Nothing in `tnwf-data-release` is committed to `lab-nathan`, contradicting its
  own README.

### F11 — Not a defect: `95%%` in `table_scaling.tex`

`make_table2_scaling.py:169` emits a literal `95%%` into the generated table's
header comment. This looks like an un-formatted `%`-escape, but the manuscript's
committed `figures/table_scaling.tex` contains the same `95%%`, and it sits
inside a LaTeX comment line, so it renders nothing. **Byte-identity with the
manuscript depends on reproducing it.** Left as-is with an explanatory comment.

---

## 3. Changes

### 3.1 `tnwf-paper` (this repository)

**Ship the four missing paper-float generators** — addresses F1.

Added `scripts/make_table1_sw.py`, `scripts/make_table2_scaling.py`,
`scripts/make_fig_scaling_bounds.py`, `scripts/make_fig_rare_event.py`,
`src/tnwf/qae.py`, `examples/rare_event/*.npz` (11 files) and
`data/exact_sw_floor.json`, together with the `.gitignore` negations and
`Makefile` targets they need. Committed the pending Fig 5 magma-palette fix and
the `mps/core.py` `_robust_svd` fallback. Dropped
`scripts/make_table3_pareto_configs.py` (its float is commented out at
`si.tex:1283-1290`) and `licenses/THIRD_PARTY_LICENSES.md` (`uv.lock` is the
authoritative dependency record; a hand-maintained restatement can only drift).

**Dropped the QTT path** — addresses F2. The `method="qtt"` additions were never
committed, so this required no revert: the working-tree diff was simply not
carried forward. `src/tnwf/qtt/`, `tests/mps/test_qtt_layer.py` and
`scripts/exploration/` remain gitignored on disk, untouched and unshipped.

**Prune** — addresses F6, F7.

| Removed | Lines | Reason |
|---|---|---|
| `src/tnwf/mps/tdvp_torch.py` + `tests/test_tdvp_torch_equiv.py` | 714 | Unreachable; the `device` plumbing in `vstep_tdvp.py` and `run_evolution.py` went with it |
| `scripts/make_fig1_combined.py`, `make_fig1_concept.py`, `make_fig2_tensor_networks.py` | 501 | No manuscript float; `make fig1`/`figs` removed |
| `mode_coverage`, `build_steps_from_V_fn`, `eval_mps_at_indices` | 63 | Zero callers |
| `torchvision` (`pyproject.toml`) | — | Zero uses repo-wide |

`metrics/__init__` stopped re-exporting `wasserstein_2` and `grid_pmf_indices`.
Both **remain in their modules** — contrary to the initial finding they are not
dead, being called internally by `wasserstein_2_subsampled` and
`nll_from_density_grid` — but nothing outside the package consumed them.

`make clean-results` was retargeted from `results/` to `data/` and explicitly
scoped to the three regenerable datasets. The obvious fix — globbing `data/*/` —
would have deleted the downloaded archive and the `jam/*.pt` checkpoints, neither
of which this repository can regenerate. The `Makefile` now says so.

Applied ruff's safe autofixes: the lint baseline drops from **102 findings to
66**, all pre-existing and style-only (`E4`/`E7` plus the deliberate `E402` from
the `sys.path.insert` idiom in the figure scripts).

**Fig 8** — addresses F5. Emits `fig_rare_event_advantage.{pdf,png}` (the
manuscript's filename) instead of `rare_event_master.*`, and `--tail-k` now
defaults to the paper's `4.0` instead of `5.0`. Also removed five plotting
passes that produce no manuscript float — three were called
(`make_combined_figure`, `make_amplification_figure`,
`make_sampling_tsne_figure`), two already had no callers at all
(`make_tsne_figure`, `make_advantage_figure`). The script goes **866 → 526
lines** and runs in **36 s instead of ~5 min**, because each removed pass fitted
its own t-SNE. Fig 8's own chain is untouched.

**License and documentation** — addresses F3, F4. `LICENSE` is MIT
(© 2026 Sygaldry, Inc.); `CITATION.cff` sets `license: MIT`. `README.md` gains a
concrete archive-download section (curl, sha256, `tar xzf` at the repo root) and
drops the incorrect claim that `data/scaling_theory/` ships in the repository.

`make_table2_scaling.py`'s literal `95%%` and stale generator name were **left
alone** and annotated — see F11.

### 3.2 `tnwf-data-release`

- Deleted the three `.DS_Store` files (F8) and taught both `make_manifest.py`
  (an `is_junk` filter) and the `archive` tar (`--exclude`) to skip
  `.DS_Store` / `._*` / `Thumbs.db`. The filter has to exist in *both*: the
  manifest is regenerated from `rglob`, so filtering only one of them
  re-desynchronises them. `make archive` now also prints a junk-member count.
- `make verify` passes again: **2,152 files re-hashed, payload matches**.
- Rebuilt the archive. `tnwf-paper-data.tar.gz`, **158,855,577 B (151.5 MiB)**,
  sha256 `baa5744f184dbae19136f5f68260eaed783332990ca05e6b1554217ed41cef86`,
  2,154 members all rooted at `data/`, 0 stray, 0 junk.
- Corrected the README (F9): file count 2078 → 2,152; removed the false "Tracked
  here: …" claim (nothing in the directory has ever been committed); corrected
  the `ck.pkl` description; documented the 24 empty sweep-config directories as
  the record of cells that diverged or ran out of container memory.
- **Kept `fetch/`** despite being byte-identical to `data/` (44 files, 0.9 MB).
  It is the record of what came off the Modal volumes and what `make stage`
  re-stages from; 0.9 MB is not worth losing that.

### 3.3 `sn-article.tex` (left uncommitted — the user's own checkout)

Rewrote the Data availability statement (F3). It previously claimed the data was
on GitHub, which was untrue for the 155 MB archive. It now cites a Zenodo DOI for
the archive and keeps GitHub for the code and the small in-repo inputs, with the
DOI as a marked `10.5281/zenodo.XXXXXXX` placeholder.

---

## 4. Verification

*(appended after the end-to-end reproduction run)*

---

## 5. Release blockers

Items only the authors can close.

- [ ] **Upload `tnwf-paper-data.tar.gz` to Zenodo** and replace the
      `10.5281/zenodo.XXXXXXX` placeholder in three places: `sn-article.tex`
      (Data availability), `README.md` ("Getting the data archive"), and
      `CITATION.cff` (the commented-out `identifiers:` block). Publish the
      sha256 above alongside it.
- [ ] **`CITATION.cff`**: author list, ORCIDs, affiliations, and the article DOI
      are all still `TODO`.
- [ ] **`sn-article.tex`**: the author/affiliation block (`:11-12`) and
      acknowledgements (`:1138`) are still `TODO`.
- [ ] **Confirm MIT** is the intended outbound license with whoever owns that
      call at Sygaldry. It was applied on request during this review; every
      dependency in `uv.lock` is BSD/Apache-family, so there is no inbound
      conflict, but the decision is the company's.
- [ ] **Decide whether `tnwf-data-release`'s docs get committed** to
      `lab-nathan`. Today nothing there is tracked, which contradicts its own
      README (now corrected to say so). The payload itself should stay out of
      git regardless.
