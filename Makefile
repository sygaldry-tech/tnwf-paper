.PHONY: help sync test test-needle test-medium \
        jam-all jam-swiss jam-gmm2 jam-gmm3 \
        run-swiss run-gmm2 run-gmm3 \
        fig1 fig2 fig3 fig4 figs supp notebook leaderboard leaderboard-nll clean-results

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
	@echo "  make fig1          render Fig 1 (overview)"
	@echo "  make fig2          render Fig 2 (dense evolution + tensor networks)"
	@echo "  make figs          fig1 + fig2 (locally reproducible)"
	@echo "  make fig3          render Fig 3 (t-SNE grid)   — needs gmm_*_hp sweep data"
	@echo "  make fig4          render Fig 4 (cost scaling)  — needs gmm_*_hp sweep data"
	@echo "  make supp          render Fig S1 + Table 3      — needs gmm_*_hp sweep data"
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
	uv run python scripts/make_table3_pareto_configs.py

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
