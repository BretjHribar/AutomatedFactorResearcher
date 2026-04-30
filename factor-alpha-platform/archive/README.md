# archive/

One-off scripts moved out of the project root during the 2026-04-30 cleanup.
Everything here is preserved (git history intact) but no longer in the active
research path. Restore with `git mv archive/<file> .` if needed.

## Subdirs

- **eval_superseded/** — eval scripts replaced by the unified config-driven
  pipeline (`src/pipeline/runner.py` + `prod/config/research_*.json`).
  Includes `eval_smallcap_d0_*.py`, `eval_crypto_qp.py`,
  `eval_isichenko_proper.py`, `eval_alpha*.py`, etc. (26 files)

- **discovery/** — alpha-discovery one-off loops and selection sweeps. The
  alphas they produced live in `data/alpha_results.db`; the scripts
  themselves aren't reusable artifacts. Includes `autonomous_alpha_hunt.py`,
  `discover_alphas_*.py`, `select_chars_*.py`, `sweep_alpha_*.py`,
  `save_*.py`. (16 files)

- **voc/** — 21 numbered iteration snapshots of the VOC equity backtest plus
  4 dependent diagnostic/plot scripts (`diagnose_equities_break.py`,
  `diagnose_stale_fundamentals.py`, `plot_equities_best.py`,
  `plot_winning_curve.py`). (28 files)

- **aipt_binance/** — top-level scripts for the AIPT and Binance crypto
  systems, separate from the KuCoin futures pipeline. The live AIPT/Binance
  code under `prod/` (`prod/aipt_trader.py`, `prod/binance_trader.py`, etc.)
  is **untouched** and still active. (5 files)

## Canonical pipeline (active)

- Configs: `prod/config/research_equity.json`, `prod/config/research_crypto.json`
- Runner: `src/pipeline/runner.py`
- Fees:   `src/pipeline/fees.py`
- Preprocess: `src/portfolio/preprocessing.py`
- Combiners: `src/portfolio/combiners.py`  (incl. new `combiner_topn_train`)
- Risk model: `src/portfolio/risk_model.py`
- QP:     `src/portfolio/qp.py`
- Acceptance test: `tools/test_pipeline_acceptance.py`
- Drivers: `test_qp_combiners.py`, `test_qp_capacity.py`,
           `test_qp_factor_vs_diag.py`, `test_qp_crypto.py`
