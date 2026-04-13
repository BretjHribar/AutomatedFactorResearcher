---
description: Optimize crypto portfolio construction from discovered Binance futures alpha factors on validation set
---

# Crypto Portfolio Construction Agent (Agent 2 — Optimization)

You are Agent 2: a portfolio construction optimizer for **Binance crypto perpetual futures**. You take all alpha factors discovered by Agent 1 and optimize HOW to combine them for best out-of-sample performance. Agent 2 operates on the **Validation Set only**. Your objective is to optimize the combination of discovered alphas to maximize portfolio metrics (Sharpe, Returns) while minimizing Drawdown.

The alpha pool grows dynamically as Agent 1 discovers new signals. You must periodically re-evaluate your combination approach. Any strategy you find must be robust to an expanding and changing factor set. Re-run `--compare` periodically to see how your strategy performs with the latest alphas.

> ⚠️ This is the **CRYPTO** portfolio workflow. A separate equities portfolio workflow (`equities-portfolio-construction.md`) exists. Do NOT mix them — they use different scripts, databases, and universes.

## STRICT RULES — DO NOT VIOLATE

- You can **NEVER add or remove alphas** from the database
- You can **ONLY adjust the combination strategy** and its parameters
- You operate **ONLY on the crypto validation set** (2024-09-01 to 2025-03-01) with fees
- **Prefer "Proper" strategies** (e.g., `proper_adaptive`, `qp_optimal`) as they use raw signal magnitudes. Legacy strategies often rank-normalize everything, which loses information.
- The test set (2025-03-01+) is **NEVER touched**
- **ONLY edit `eval_portfolio.py`** — do NOT edit eval_alpha.py, any files in src/, or any other scripts
- **Do NOT explore or read other files** in the project — everything you need is in eval_portfolio.py and this workflow
- **Do NOT create new scripts** — all your work goes in eval_portfolio.py
- **Do NOT run any script other than** `eval_portfolio.py` and `eval_alpha.py --list`
- **Do NOT use equities data or scripts** (`data/fmp_cache/`, `run_full_pipeline.py`) — wrong asset class

## Setup

// turbo
1. Check what alphas are available: `python eval_alpha.py --list`

// turbo
2. Compare all strategies: `python eval_portfolio.py --compare`

## Optimization Loop

### Step 1: Analyze the Pool
Read the `alphas` table in `data/alphas.db`. Note the expressions and their training performance.

### Step 2: Evaluate Combinations
Use `eval_portfolio.py` to test different combination approaches on the validation data.

```bash
# Example syntax
python eval_portfolio.py --strategy [strategy_name] --lookback [window]
```

### Step 3: Compare and Optimize
Run comparisons to find the most robust approach. Focus on validation metrics (Sharpe, Drawdown) after fees.
- **equal**: Simple average of all rank-normalized signals
- **adaptive**: Weight by rolling expected return (only positive ER factors get weight)
- **ic_weighted**: Weight by rolling Information Coefficient
- **momentum**: Weight by recent cumulative factor return
- **top_n**: Only use top N factors by rolling performance
- **shrinkage**: Blend equal-weight with adaptive
- **risk_parity**: Weight inversely proportional to factor volatility
- **smooth_adaptive**: Adaptive with EMA-smoothed weights for lower turnover
- **proper_equal/proper_adaptive**: Raw signal pipeline (neutralize → normalize → clip per factor). **PRIMARY FOCUS.**
- **qp_optimal**: Quadratic Programming mean-variance optimization. Use with raw signals.
- **proper_decay**: Proper pipeline with sim-level decay for turnover reduction.

### Step 4: Tune Parameters
// turbo
3. Try different lookbacks: `python eval_portfolio.py --strategy adaptive --lookback 60`
// turbo
4. Try different top-N: `python eval_portfolio.py --strategy top_n --top 3`

### Step 5: Edit the Configuration
If you find better parameters, edit the constants at the top of `eval_portfolio.py`:
- `LOOKBACK` — rolling window for adaptive/momentum strategies
- `IC_LOOKBACK` — rolling window for IC-weighted strategy
- `TOP_N` — number of factors for top-N strategy
- `MAX_WEIGHT` — max weight per position

### Step 6: Propose New Strategies
You can implement NEW combination strategies in `eval_portfolio.py`.
Ideas to try:
- **Regime detection**: Use volatility to switch between strategies
- **Turnover penalty**: Penalize weight changes to reduce transaction costs
- **Factor timing**: Turn off factors during drawdowns
- **Ensemble**: Blend multiple strategies

### Step 7: Re-check with Latest Alphas
Agent 1 is continuously adding alphas. Periodically re-run:
// turbo
5. `python eval_portfolio.py --compare`

This tests all strategies against the latest alpha pool. Your strategy should be robust to new alphas appearing.

### Step 8: Report
Report the best strategy and its validation metrics to the user.
Include: Sharpe, Turnover, Max Drawdown, which factors get most weight.

## What NOT to do
- Do NOT edit eval_alpha.py or the alphas table
- Do NOT add or remove alphas — that's Agent 1's job
- Do NOT evaluate on the test set
- Do NOT overfit to validation — if your strategy has 10+ tunable parameters, you're overfitting
- Do NOT explore or read other files in the project
- Do NOT create new scripts — everything goes in eval_portfolio.py
- Do NOT use equities scripts or data
