.PHONY: help sync test test-needle test-medium \
        jam-all jam-swiss jam-gmm2 jam-gmm3 \
        run-swiss run-gmm2 run-gmm3 \
        fig1 fig2 fig3 fig4 figs supp notebook leaderboard leaderboard-nll clean-results \
        fig-rare-event table1 table2 fig-scaling-bounds

help:
	@echo "tnwf — npj-QI reproduction Makefile"
	@echo ""
	@echo "  make sync          uv sync --extra dev"
	@echo "  make test-needle   pytest -m needle (<30s)"
	@echo "  make test-medium   pytest -m medium (<5min)"
	@echo "  make test          all needle + medium"
	@echo ""
	@echo "  make jam-all       train JAM checkpoints for all 3 datasets × 10 seeds"
	@echo "  make jam-swiss     swiss_roll_2d only"
	@echo "  make jam-gmm2      gmm_2d only"
	@echo "  make jam-gmm3      gmm_3d only"
	@echo ""
	@echo "  make run-swiss     all methods × 10 seeds on swiss_roll_2d"
	@echo "  make run-gmm2      all methods × 10 seeds on gmm_2d"
	@echo "  make run-gmm3      all methods × 10 seeds × N×K sweep on gmm_3d"
	@echo ""
	@echo "  make fig1          draft-only overview panels (paper Fig 1 is TikZ)"
	@echo "  make fig2          render paper Fig 2 (dense evolution) + a draft-only panel"
	@echo "  make figs          fig1 + fig2 (locally reproducible)"
	@echo "  make fig3          render paper Fig 5 (t-SNE grid)  — needs gmm_*_hp sweeps"
	@echo "  make fig4          render paper Fig 6 (cost scaling) — needs gmm_*_hp sweeps"
	@echo "  make supp          render paper Fig S1              — needs gmm_*_hp sweeps"
	@echo ""
	@echo "  make table1        render paper Table 1 (best-cell SW) — needs gmm_*_hp sweeps"
	@echo "  make table2        render paper Table 2 (MPS-V scaling) — needs timing_scaling + checkpoints"
	@echo "  make fig-scaling-bounds  render paper Fig 4 (error scaling) — needs scaling_theory CSVs"
	@echo "  make fig-rare-event  render the rare-event advantage figure (paper Fig 8)"
	@echo ""
	@echo "  make notebook      launch the examples/tnwf_demo.ipynb demo notebook"
	@echo ""
	@echo "  NB: fig3/fig4/supp read data/gmm_*_hp*/ produced by large cloud"
	@echo "      HP-sweeps not included in this repository (see README)."

sync:
	uv sync --extra dev

test-needle:
	uv run pytest -m needle

test-medium:
	uv run pytest -m medium

test:
	uv run pytest -m "needle or medium"

# ── JAM training ──────────────────────────────────────────────────────────
jam-swiss:
	uv run python scripts/swiss_roll_2d/train_jam.py --n_iter 8000

jam-gmm2:
	uv run python scripts/gmm_2d/train_jam.py --n_iter 8000

jam-gmm3:
	uv run python scripts/gmm_3d/train_jam.py --n_iter 12000

jam-all: jam-swiss jam-gmm2 jam-gmm3

# ── pipeline runs ─────────────────────────────────────────────────────────
run-swiss:
	uv run python scripts/swiss_roll_2d/run_all_methods.py

run-gmm2:
	uv run python scripts/gmm_2d/run_all_methods.py

run-gmm3:
	uv run python scripts/gmm_3d/run_all_methods.py

# ── figures ───────────────────────────────────────────────────────────────
# Locally reproducible from the runners above.
#
# NB: fig1_overview / fig1_concept / fig2_tensor_networks are NOT used by the
# submitted manuscript — Fig 1 is a TikZ figure built in the LaTeX source, and
# the manuscript's Fig 2 is fig2_dense only. These generators are retained
# because their output appears in earlier drafts; only `make_fig2_dense_evolution`
# below feeds a current paper float. See the README float table.
fig1:
	uv run python scripts/make_fig1_combined.py
	uv run python scripts/make_fig1_concept.py

fig2:
	uv run python scripts/make_fig2_dense_evolution.py
	uv run python scripts/make_fig2_tensor_networks.py

figs: fig1 fig2

# Require the gmm_*_hp* HP-sweep data (cloud sweeps, not shipped — see README).
fig3:
	uv run python scripts/make_fig3_tsne_grid.py

fig4:
	uv run python scripts/make_fig4.py

supp:
	uv run python scripts/make_fig_supp_pareto.py

# Table 1 (best-cell SW). Needs the gmm_*_hp sweeps unpacked at data/, plus the
# tracked data/exact_sw_floor.json for the target--target row.
table1:
	uv run python scripts/make_table1_sw.py

# Paper Table 2 (trained MPS-V scaling, d=8..32). Reads the 40 replicate traj.npz
# under data/timing_scaling/wf_eval/ plus the four MPS-V oracles in
# data/mps_v_checkpoints/. Recomputes the classical floor via torch (~70 s).
table2:
	uv run python scripts/make_table2_scaling.py

# Paper Fig 4 (error scaling vs the bounds). Plots from the shipped
# data/scaling_theory/ CSV cache; the compute half is not part of this release.
fig-scaling-bounds:
	uv run python scripts/make_fig_scaling_bounds.py

# Rare-event advantage (paper Fig 8). Self-contained: the Table-2 K=160 MPS cores
# ship in examples/rare_event/, so this needs no HP-sweep data. --tail-k 4.0 is the
# paper's threshold (the script default is 5.0). Takes ~5 min (t-SNE embeddings).
fig-rare-event:
	uv run python scripts/make_fig_rare_event.py --source paper --tail-k 4.0

# ── demo notebook ─────────────────────────────────────────────────────────
notebook:
	uv run --with jupyter jupyter notebook examples/tnwf_demo.ipynb

# ── leaderboard ───────────────────────────────────────────────────────────
leaderboard:
	uv run python -m tnwf.leaderboard --print

leaderboard-nll:
	uv run python -m tnwf.leaderboard --metric nll_mean --print

clean-results:
	rm -rf results/*/*/*.npz results/*/*/N*/seed*.npz results/leaderboard.{csv,md}
