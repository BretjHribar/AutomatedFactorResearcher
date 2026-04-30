# Unified Research Pipeline

One config-driven pipeline for both equity (daily, MCAP_100M_500M) and
crypto (4h, KuCoin futures). All knobs live in JSON config files. The same
Python code drives both markets — only the config differs.

This document is the authoritative reference for the **active research
pipeline**. The wider README describes the platform's BRAIN-replication
features (operator library, GP discovery, FastAPI server) and broader
context.

---

## TL;DR

```bash
# Run the equity research pipeline
python -c "from src.pipeline.runner import run; print(run('prod/config/research_equity.json').metrics['FULL'])"

# Run the crypto research pipeline
python -c "from src.pipeline.runner import run; print(run('prod/config/research_crypto.json').metrics['TRAIN'])"

# Acceptance test (must PASS before any pipeline change ships)
python tools/test_pipeline_acceptance.py
```

**Targets** asserted by `tools/test_pipeline_acceptance.py`:
- Equity: equal × diag → FULL net SR **+4.98 ± 0.10**
- Crypto: topn_train(50) → TRAIN gross SR **+5.30 ± 0.10**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CONFIG (JSON)                                │
│  prod/config/research_equity.json    prod/config/research_crypto.json│
└─────────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   src/pipeline/runner.py  ← single entry point       │
│                                                                      │
│   run(config) → PipelineResult                                       │
│   ─────────────────────────────────────────────────────────────────  │
│   Stage 0  load_universe_and_matrices                                │
│   Stage 1  load_alphas (incl. optional train_sharpes from DB)        │
│   Stage 2  apply_preprocess  (per-alpha, src/portfolio/preprocessing) │
│   Stage 3  combine           (src/portfolio/combiners.py)             │
│   Stage 4  post_combiner     (re-L1-norm, clip)                      │
│   Stage 5  QP walk-forward   (optional, src/portfolio/qp.py)          │
│            risk_model        (src/portfolio/risk_model.py)            │
│   Stage 6  realized cost     (src/pipeline/fees.py)                   │
│   Stage 7  metrics           (per split: TRAIN/VAL/TEST/V+T/FULL)    │
└──────────────────────────────────────────────────────────────────────┘
```

### Markets share the same code

```
   EQUITY (daily)                       CRYPTO (4h, KuCoin)
   ──────────────                       ────────────────────
   • FMP matrices + universes           • KuCoin matrices + universes
   • subindustry-demean                 • cross-section demean only
   • clip ±0.02                         • no clip (caller's choice)
   • combiner=equal (or any)            • combiner=topn_train (top-N
                                          by stored TRAIN Sharpe)
   • risk_model in {diagonal, pca,      • risk_model in {diagonal, pca}
       style, style+pca}                  (no fundamentals → no style)
   • cost = per-share IB MOC            • cost = bps_taker (2.5+1.0)
   • bars/yr = 252                      • bars/yr = 2190
                          ┌──────────────────────────┐
                          │   SAME CORE PIPELINE     │
                          │  src/pipeline/runner.py  │
                          │  src/pipeline/fees.py    │
                          │  src/portfolio/qp.py     │
                          │  src/portfolio/        │
                          │    risk_model.py         │
                          │    combiners.py          │
                          │    preprocessing.py      │
                          └──────────────────────────┘
```

---

## File map (canonical)

| Layer | Path | What it does |
|-------|------|--------------|
| Config | `prod/config/research_equity.json` | Equity research baseline |
| Config | `prod/config/research_crypto.json` | Crypto research baseline |
| Runner | `src/pipeline/runner.py` | `run(config) → PipelineResult` |
| Fees | `src/pipeline/fees.py` | `cost_per_share_ib` + `cost_bps_taker` |
| Preprocess | `src/portfolio/preprocessing.py` | `apply_preprocess(...)` per-alpha |
| Combiners | `src/portfolio/combiners.py` | `combiner_{equal,billions,risk_par,ic_wt,sharpe_wt,topn_sharpe,topn_train,adaptive}` |
| Risk model | `src/portfolio/risk_model.py` | `build_{diagonal,pca,style,style_pca}`, `build_style_factors` |
| QP | `src/portfolio/qp.py` | `solve_qp` + `run_walkforward` (ADV-cap-aware) |
| Expression engine | `src/operators/fastexpression.py` | Compiles alpha DSL → DataFrame ops |
| Acceptance | `tools/test_pipeline_acceptance.py` | Regression test — must PASS before merging |
| Drivers | `test_qp_combiners.py` | Sweep 7 combiners × 2 risk models (equity) |
| Drivers | `test_qp_capacity.py` | Sweep books × 2 risk models with ADV cap (equity) |
| Drivers | `test_qp_factor_vs_diag.py` | DIAG_60 / DIAG_126 / FACTOR_K5 risk-model A/B |
| Drivers | `test_qp_crypto.py` | Crypto sweep (top_n × {no QP, QP diag, QP pca}) |

All four `test_qp_*.py` drivers are thin loops that load a config and override
one knob per cell. They share zero setup code.

---

## Config schema

Both markets use the same JSON shape. Every field below is required unless
marked optional.

```jsonc
{
  "market": "equity",
  "interval": "1d",
  "annualization": { "bars_per_year": 252 },

  // STAGE 0: data
  "data": {
    "matrices_dir": "data/fmp_cache/matrices",
    "universe_path": "data/fmp_cache/universes/MCAP_100M_500M.parquet",
    "universe_filter": { "method": "coverage", "threshold": 0.5 },
    "returns_source": "compute_from_close"   // or "matrix:returns"
  },

  // STAGE 1: alpha source
  "alpha_source": {
    "db_path": "data/alpha_results.db",
    "table": "alphas",                       // crypto: "alphas_crypto"
    "filter_sql": "archived=0 AND ...",
    "train_sharpe_table": null,              // crypto: "evaluations_crypto"
    "train_sharpe_column": null              // crypto: "sharpe_is"
  },

  // STAGE 2: per-alpha preprocessing
  "preprocessing": {
    "universe_mask": true,                   // mask each alpha to uni per bar
    "demean_method": "subindustry",          // "cross_section"|"subindustry"|"none"
    "subindustry_field": "subindustry",      // matrix file name (no .parquet)
    "normalize": "l1",                       // "l1"|"zscore"|"none"
    "clip_max_w": 0.02                       // null = no clip
  },

  // STAGE 3: combine alphas
  "combiner": {
    "name": "equal",                         // see combiner table below
    "params": { "max_wt": 0.02 }
  },

  // STAGE 4: post-combiner re-normalization
  "post_combiner": {
    "renormalize_l1": true,
    "clip_max_w": 0.02
  },

  // STAGE 5a: risk model (only used if qp.enabled)
  "risk_model": {
    "name": "diagonal",                      // "diagonal"|"pca"|"style"|"style+pca"
    "params": {
      "factor_window": 126,                  // bars
      "n_pca_factors": 5,
      "vol_window": 60                       // bars
    }
  },

  // STAGE 5b: QP optimizer (optional)
  "qp": {
    "enabled": true,                         // false = combiner output IS portfolio
    "lambda_risk": 5.0,
    "kappa_tc": 30.0,
    "max_w": 0.02,
    "dollar_neutral": true,
    "max_gross_leverage": 1.0,
    "commission_per_share": 0.0045,          // baked into κ_i for t-cost penalty
    "impact_bps": 0.5,
    "adv_cap": null                          // or { "adv_field":"adv20", "moc_frac":0.10, "max_part":0.30 }
  },

  // STAGE 6: realized cost model
  "fees": {
    "model": "per_share_ib",                 // "per_share_ib" or "bps_taker"
    "params": { /* model-specific */ }
  },

  // STAGE 7: time splits
  "splits": { "train_end": "2024-01-01", "val_end": "2025-04-01" },
  "book": 500000
}
```

### Combiners

| `combiner.name` | Function | Params (in `combiner.params`) | Notes |
|---|---|---|---|
| `equal` | `combiner_equal` | `max_wt` | 1/N average of preprocessed signals |
| `adaptive` | `combiner_adaptive` | `lookback`, `max_wt` | Rolling-expected-return-weighted |
| `risk_par` | `combiner_risk_parity` | `lookback`, `max_wt` | Inverse-vol-weighted |
| `billions` | `combiner_billions` | `optim_lookback`, `max_wt` | Kakushadze regression |
| `ic_wt` | `combiner_ic_weighted` | `lookback`, `max_wt` | Rolling cross-sectional IC |
| `sharpe_wt` | `combiner_sharpe_weighted` | `lookback`, `max_wt` | Rolling standalone Sharpe |
| `topn_sharpe` | `combiner_topn_sharpe` | `lookback`, `top_n`, `max_wt` | Top-N by rolling Sharpe per bar |
| **`topn_train`** | **`combiner_topn_train`** | **`top_n`**, `max_wt` | Top-N by **stored TRAIN Sharpe** from `train_sharpes` (passed in by runner). Crypto canonical. |

### Risk models

| `risk_model.name` | Decomposition | Builder |
|---|---|---|
| `diagonal` | Σ ≈ diag(σ²)  (σ from rolling stddev) | `build_diagonal` |
| `pca` | Σ ≈ B_pca B_pca' + diag(s²)  (K-factor PCA on rolling cov) | `build_pca` |
| `style` | Σ ≈ B_style Σ_F B_style' + diag(s²)  (Barra-ish style factors) | `build_style` |
| `style+pca` | style + PCA on residuals | `build_style_pca` |

Style factors require fundamentals (`build_style_factors` in `risk_model.py`):
market_beta, size, value, momentum, profitability, low_vol, growth, leverage.
Crypto can't use style — no fundamentals.

### Fee models

| `fees.model` | Function | Params |
|---|---|---|
| `per_share_ib` | `cost_per_share_ib` | `commission_per_share, per_order_min, sec_fee_per_dollar, sell_fraction, impact_bps, borrow_bps_annual` |
| `bps_taker` | `cost_bps_taker` | `taker_bps, slippage_bps` |

---

## Stage details

### Stage 0 — data

- Loads `universe_path` parquet, filters tickers by `universe_filter`
  (currently only `coverage` method: tickers present in ≥ `threshold` fraction
  of bars).
- Loads every `*.parquet` in `matrices_dir` (top level only; nested dirs
  like `matrices/4h/prod/` are skipped intentionally).
- Returns:
  `compute_from_close` (uses `close.pct_change()`) or `matrix:<field>`
  (uses the named matrix file, e.g. `matrix:returns` for crypto's
  precomputed `returns.parquet`).

### Stage 1 — alpha source

- SQL: `SELECT id, expression FROM <table> WHERE <filter_sql>`
- If `train_sharpe_table` and `train_sharpe_column` are set, also pulls
  `{alpha_id: train_sharpe}` for combiners that use it (e.g. `topn_train`).
- Each expression is evaluated by `FastExpressionEngine.evaluate(expr)`.

### Stage 2 — apply_preprocess

`src/portfolio/preprocessing.py:apply_preprocess` is byte-equivalent to the
two legacy helpers it replaces:

| Legacy helper | Equivalent config |
|---|---|
| `proc_signal_subind` (equity) | `universe_mask=true, demean_method='subindustry', normalize='l1', clip_max_w=0.02` |
| `signal_to_portfolio` (crypto) | `universe_mask=false, demean_method='cross_section', normalize='l1', clip_max_w=null` |

Verified by `tools/test_preprocessing_byteeq.py` (max abs diff = 0.00 across
both markets).

### Stage 3 — combine

The runner passes the dict of preprocessed alpha signals to the chosen
combiner. `topn_train` additionally receives `train_sharpes`. All combiners
return a single `(date × ticker)` DataFrame.

### Stage 4 — post_combiner

- `renormalize_l1`: divide each row by Σ|w_i|, restoring `Σ|w_i| = 1`
- `clip_max_w`: clip to ±max_w per name. Required when `combiner=equal`
  averages many normalized alphas (the average has magnitude ~1/√N).

### Stage 5 — QP (optional)

Skipped entirely if `qp.enabled=false` (crypto canonical config). When enabled,
`run_walkforward` solves per-day:

```
maximize  α'w − ½λ Σ_k ‖L_k' w‖² − ½λ s²·w² − Σ_i κ_i |w_i − w_prev,i|
s.t.      ‖w‖₁ ≤ max_gross_leverage
          |w_i| ≤ max_w   (scalar OR per-name from ADV cap)
          Σ_i w_i = 0     (if dollar_neutral)
```

`L_k` and `s²` come from the configured risk-model builder. `κ_i` is per-name
(scaled by commission_per_share/price + impact_bps).

If `qp.adv_cap` is set, the per-name cap becomes
`min(max_w, moc_frac × max_part × ADV_i / book)`.

### Stage 6 — realized cost

Independent of the QP's t-cost penalty. Realistic per-bar cost based on the
final weights. Two models in `fees.py`:

- `per_share_ib`: equity IB MOC (commission + impact + SEC fee + borrow)
- `bps_taker`: linear bps (taker + slippage), one fee unit per |Δw|

### Stage 7 — metrics

Five splits computed for both gross and net PnL:

| Split | Range |
|---|---|
| `TRAIN`    | `[start, train_end]`   |
| `VAL`      | `[train_end, val_end]` |
| `TEST`     | `[val_end, end]`       |
| `VAL+TEST` | `[train_end, end]`     |
| `FULL`     | full window            |

Each cell: `n_bars`, `SR_gross`, `SR_net`, `ret_ann_gross`, `ret_ann_net`.

---

## Acceptance targets

`tools/test_pipeline_acceptance.py` is the regression gate:

| Test | Cell | Target | Tolerance |
|------|------|-------:|----------:|
| Equity reproduction | equal × diag, FULL net SR | +4.98 | ±0.10 |
| Crypto reproduction | topn_train(50), TRAIN gross SR | +5.30 | ±0.10 |

Both currently PASS at delta +0.001 / +0.048 (run on `5cf5073`).

---

## Databases

`data/alpha_results.db` (SQLite) holds **both** systems side-by-side after the
2026-04-30 merge. **Untracked in git** — local-only.

### Equity tables (original schema)

| Table | Columns |
|---|---|
| `alphas` | id, expression, notes, archived |
| `evaluations` | id, alpha_id, sharpe, fitness, ic, ... |

### Crypto tables (richer schema — copied from `data/alphas.db`)

| Table | Notable columns |
|---|---|
| `alphas_crypto` | id, expression, name, **asset_class**, **interval**, source, archived, notes, **universe** |
| `evaluations_crypto` | alpha_id, sharpe_is/oos/train/val/test, return_total/ann, max_drawdown, turnover, fitness, ic_mean, ic_ir, psr, train/val/test_start/end, n_bars, evaluated_at, interval, universe |
| `runs_crypto`, `selections_crypto`, `correlations_crypto`, `trial_log_crypto` | (currently empty / 37k log rows) |

The merge is performed by `tools/merge_crypto_db.py` (idempotent).

---

## Adding a new market

1. Build matrices + universe in `data/<exchange>_cache/`.
2. Stand up an alpha DB table (or alphas_crypto-style copy).
3. Write `prod/config/research_<market>.json` mirroring one of the existing
   configs.
4. Pick a `preprocessing` setup, a `combiner`, optionally enable QP.
5. Add an acceptance assertion in `tools/test_pipeline_acceptance.py`.
6. `python -c "from src.pipeline.runner import run; run('prod/config/research_<market>.json')"`

---

## Adding a new combiner / risk model / fee model

| Add to | Pattern |
|---|---|
| Combiner | New function in `src/portfolio/combiners.py` matching the existing signature; register in `runner._COMBINERS` |
| Risk model | New `build_<name>(R, ...)` in `src/portfolio/risk_model.py` returning `(L_list, s²)`; add a branch in `runner._build_risk_model_fn` |
| Fee model | New `cost_<name>(w, ...)` in `src/pipeline/fees.py`; add a branch in `make_cost_fn` |

The QP solver in `src/portfolio/qp.py` is risk-model-agnostic — it just consumes
`L_list, s²` from the builder, so new risk models drop in without touching
the QP.

---

## How the pieces converge

This pipeline replaced ~75 one-off eval / discovery / iteration scripts (now
under `archive/`). The mapping:

| Old | New |
|---|---|
| `eval_smallcap_d0_*.py` (12 files) | `test_qp_*.py` + `research_equity.json` |
| `eval_crypto_qp.py`, `eval_isichenko_proper.py` | `test_qp_crypto.py` + `research_crypto.json` |
| `proc_signal_subind` (in eval_smallcap_d0_final.py) | `apply_preprocess(demean_method='subindustry', ...)` |
| `signal_to_portfolio` (in update_wq_alphas_db.py) | `apply_preprocess(demean_method='cross_section', ...)` |
| Inline QP in 8 eval scripts + 2 prod runners | `src/portfolio/qp.py:run_walkforward` |
| Multiple `realistic_cost` copies | `src/pipeline/fees.py:cost_per_share_ib` |
| Crypto `qp_combiner` in eval_crypto_qp.py | `combiner=topn_train` config |

Active prod runners (`run_ib_portfolio.py`, `run_4h_portfolio.py`,
`prod/kucoin_trader.py`) still inline-implement the QP. **Migrating prod to
use this runner is deferred** — high-risk change for live trading; will
happen after research is fully locked.

---

## Recent changes log

- **2026-04-30** — built unified pipeline (`5cf5073` on `PROD2`). Archived 75
  superseded scripts. Acceptance: equity +4.98, crypto +5.35. See
  `archive/README.md` for the move map.
