.PHONY: help sync test test-needle test-medium \
        jam-all jam-swiss jam-gmm2 jam-gmm3 jam-petals jam-eb \
        run-swiss run-gmm2 run-gmm3 run-petals run-eb \
        fetch-eb \
        fig1 fig2 fig3 figs clean-results

help:
	@echo "tnWF — npj-QI reproduction Makefile"
	@echo ""
	@echo "  make sync          uv sync --all-extras"
	@echo "  make test-needle   pytest -m needle (<30s)"
	@echo "  make test-medium   pytest -m medium (<5min)"
	@echo "  make test          all needle + medium"
	@echo ""
	@echo "  make jam-all       train JAM checkpoints for all 3 datasets × 10 seeds"
	@echo "  make jam-swiss     swiss_roll_2d only"
	@echo "  make jam-gmm2      gmm_2d only"
	@echo "  make jam-gmm3      gmm_3d only"
	@echo ""
	@echo "  make run-swiss     all 7 methods × 10 seeds on swiss_roll_2d"
	@echo "  make run-gmm2      all 7 methods × 10 seeds on gmm_2d"
	@echo "  make run-gmm3      all 7 methods × 10 seeds × N×K sweep on gmm_3d"
	@echo ""
	@echo "  make fig1          render Fig 1 from results/{swiss_roll_2d,gmm_2d}"
	@echo "  make fig2          render Fig 2 from results/gmm_3d"
	@echo "  make fig3          render Fig 3 from results/gmm_{d,N}_scaling (after Modal)"
	@echo "  make figs          fig1 + fig2"

sync:
	uv sync --all-extras

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

# ── Biological datasets ───────────────────────────────────────────────────
# Petals: pure-numpy, no external download.
jam-petals:
	uv run python scripts/petals_2d/train_jam.py --n_iter 10000

run-petals:
	uv run python scripts/petals_2d/run_all_methods.py

# Embryoid body: requires a one-time Mendeley download (~hundreds of MB).
fetch-eb:
	uv run python scripts/eb_5d/fetch_raw.py

jam-eb: fetch-eb
	uv run python scripts/eb_5d/train_jam.py --n_iter 15000

run-eb:
	uv run python scripts/eb_5d/run_all_methods.py

# ── pipeline runs ─────────────────────────────────────────────────────────
run-swiss:
	uv run python scripts/swiss_roll_2d/run_all_methods.py

run-gmm2:
	uv run python scripts/gmm_2d/run_all_methods.py

run-gmm3:
	uv run python scripts/gmm_3d/run_all_methods.py

# ── figures ───────────────────────────────────────────────────────────────
fig1:
	uv run python scripts/swiss_roll_2d/make_fig1.py
	uv run python scripts/gmm_2d/make_fig1_panel.py

fig2:
	uv run python scripts/gmm_3d/make_fig2.py

fig3:
	uv run python scripts/gmm_d_scaling/make_fig3a.py
	uv run python scripts/gmm_N_scaling/make_fig3b.py

figs: fig1 fig2

leaderboard:
	uv run python -m tnwf.leaderboard --print

leaderboard-nll:
	uv run python -m tnwf.leaderboard --metric nll_mean --print

clean-results:
	rm -rf results/*/*/*.npz results/*/*/N*/seed*.npz results/leaderboard.{csv,md}
