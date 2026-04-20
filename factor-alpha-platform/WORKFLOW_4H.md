# 4H Crypto Alpha Research & Portfolio Construction -- Agent Workflow

## Overview

This workflow discovers orthogonal alpha factors on 4h Binance perpetual futures data and evaluates them through a comprehensive portfolio construction pipeline. It is designed to be run by an autonomous agent.

## Environment

- **Directory**: `c:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform`
- **Universe**: `BINANCE_TOP50` (4h bars, 6 bars/day)
- **Data range**: Jan 2020 - Mar 2026 (universe viable from Oct 2020+)
- **Database**: `data/alphas.db` (columns: `interval='4h'`, `universe='BINANCE_TOP50'`)

## Splits

| Split | Start | End | Duration |
|-------|-------|-----|----------|
| Train | 2021-01-01 | 2025-01-01 | 4 years |
| Val | 2025-01-01 | 2025-09-01 | 8 months |
| Test | 2025-09-01 | 2026-03-05 | ~6 months |

## Step 1: Alpha Discovery

Run the discovery script to find alphas with IS Sharpe >= 4.0:

```
python discover_alphas_4h.py
```

This script:
- Generates candidate composite alpha expressions from atom libraries
- Evaluates each on the TRAIN set (no fees)
- Applies quality gates: SR >= 4.0, TO < 0.40, sub-period stability, diversity check
- Saves passing alphas to `data/alphas.db` with `interval='4h'` and `universe='BINANCE_TOP50'`

**Goal**: Discover 16+ orthogonal alphas. If the script runs out of candidates, increase the seed or expand the atom library.

Check current alpha count:
```
python eval_alpha.py --list
```

## Step 2: Portfolio Construction Evaluation

Once you have alphas, run the unified evaluation:

```
python run_4h_portfolio.py
```

This script evaluates ALL of the following automatically:

### Combiners
1. **Equal Weight** -- simple average of all alpha signals
2. **Billion Alphas** -- Kakushadze regression (original paper, NO smoothing)
3. **Factor MAX (5d)** -- sort alphas by max single-bar PnL, 30-bar lookback
4. **Factor MAX (10d)** -- same, 60-bar lookback
5. **Adaptive** -- rolling expected-return weighted, 120-bar lookback
6. **Risk Parity** -- inverse-volatility weighted, 120-bar lookback

### Execution Modes
For each combiner:
- **Raw** -- signal processed through standard pipeline (demean, scale, clip)
- **+ QP** -- CVXPY convex optimizer on aggregate signal (PCA risk model + L1 tcost)

### Fee Levels
Each combiner x mode is evaluated at: 0, 2, 5, 7 bps

### Output
- Full results table: 12 methods x 3 splits x 4 fees = 144 evaluations
- Per-combiner detailed breakdown (Sharpe, Return%, ReturnAnn%, MaxDD%, Turnover, Fitness)
- Cumulative PnL charts per fee level (train+val+test stitched)
- Test Sharpe bar chart: Raw vs QP across fee levels
- Charts saved as `4h_portfolio_*.png`

## Step 3: Interpret Results

When reporting results, display ALL data. Include:
- The full summary table at each fee level
- The per-combiner breakdown
- Highlight the best method at each fee level
- Note any val/test Sharpe divergence (overfitting signals)
- Report total runtime

## Files

| File | Purpose |
|------|---------|
| `eval_alpha.py` | Alpha evaluation harness (train only, quality gates) |
| `discover_alphas_4h.py` | Batch alpha discovery (generates + tests candidates) |
| `run_4h_portfolio.py` | Unified portfolio evaluation (all combiners x fees x splits) |
| `data/alphas.db` | Alpha database (shared, filtered by interval+universe) |

## Notes

- The database differentiates alphas by `interval` and `universe` columns. 5m and 4h alphas coexist.
- Billion Alphas uses the ORIGINAL Kakushadze algorithm: daily return aggregation, OLS regression, residual weighting. No EMA smoothing, no signal_smooth.
- QP optimizer runs on the AGGREGATE signal from each combiner, not on individual alphas.
- 4h data has 6 bars/day, so turnover is naturally much lower than 5m (288 bars/day).
- Universe is empty before Oct 2020. Train starts 2021-01-01.
