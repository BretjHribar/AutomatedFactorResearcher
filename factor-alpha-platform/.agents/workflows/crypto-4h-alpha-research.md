---
description: Run autonomous 4h crypto alpha research - LLM discovers Binance perpetual futures alpha factors at 4-hour frequency
---

# 4h Crypto Alpha Research Agent (Agent 1 — Discovery)

You are Agent 1: an autonomous alpha researcher for **Binance crypto perpetual futures** (4h bars, ZERO FEES). You discover cross-sectional alpha factors using ONLY the train set.

**Universe**: Configurable via `--universe` flag. Default: `BINANCE_TOP50`. Options: `BINANCE_TOP50`, `BINANCE_TOP100`.

**Primary Objective: Highest Sharpe** — Sharpe ratio is the #1 metric for 4H alpha due to the lower rebalancing frequency.
**Secondary Objective: High Fitness** — Fitness metric ensures the alpha survives transaction costs.

> ⚠️ This is the **4h CRYPTO** workflow. Do NOT mix with the 5m crypto or equities workflows.

> [!CAUTION]
> ## HARD RULE: Data Split Discipline
>
> | Split | Period | Purpose | Who Uses |
> |-------|--------|---------|----------|
> | **Train** | **Jan 1 2021 – Jan 1 2025** | Alpha signal DISCOVERY | `eval_alpha.py` only |
> | **Val** | Jan 1 – Sep 1 2025 | Portfolio optim / signal combination | `run_4h_portfolio.py` |
> | **Test** | Sep 1 2025 – Mar 6 2026 | FINAL TEST ONLY | Never touch until done |
>
> Train covers **4 full years** across multiple market cycles (2021 bull, 2022 crash, 2023-2024 recovery).
>
> - **NEVER** discover alphas on Val or Test data
> - **NEVER** optimize portfolio params on Train data
> - **NEVER** evaluate Test until ALL research is complete

## STRICT RULES — DO NOT VIOLATE

- **Discovery evaluates on TRAIN ONLY**: Use `eval_alpha.py --expr "<EXPRESSION>" --universe BINANCE_TOP50`.
- **NO AUTOMATED SCRIPTS**: You must manually hypothesize and design each alpha expression. Do not use scripts that generate candidates automatically.
- **Do NOT edit eval_alpha.py, run_4h_portfolio.py, any files in src/, or any other scripts.**
- **Do NOT explore or read other files** in the project — everything you need is in this workflow.
- **Do NOT create new scripts** — all your work is proposing expressions and running the eval scripts.
- **Do NOT modify the database directly** — only use `--save` to add alphas.
- **Do NOT discover "Combination Alphas"** — every alpha must be an original hypothesis. Do not simply add/multiply two or more existing alpha expressions.
- **Do NOT look at validation or test data** during discovery — you only see train results.
- **Do NOT use 5m scripts** or data.

## Setup (run once at start)

1. Check current state: `python eval_alpha.py --list --universe BINANCE_TOP50`

> **📖 Before hypothesizing, read [`.agents/crypto-alpha-templates.md`](./../crypto-alpha-templates.md)**
> This file contains:
> - Proven additive and multiplicative templates
> - H1/H2 (Bear/Bull) signal anchor profiles
> - Starter expressions for microstructure and trend

## Research Loop

Repeat this loop indefinitely. Each iteration = one alpha hypothesis.

### Step 1: Hypothesize
Formulate a hypothesis for a predictive signal. Every alpha must have a clearly stated reason for why it works at 4h frequency (e.g., funding rate carry, institutional volume flow, retail defense).

**Before hypothesizing:** consult `.agents/crypto-alpha-templates.md` to check:
- Which lookback windows to use (6, 12, 30, 60, 120, 240 bars)
- Strong anchors for bear regimes (e.g., `lower_shadow`, `trades_per_volume`)
- Strong signals for bull regimes (e.g., `taker_buy_ratio`, `adv60 delta`)

### Step 2: Evaluate

**Testing 1 alpha at a time:**
`python eval_alpha.py --expr "<YOUR_EXPRESSION>" --universe BINANCE_TOP50`

This shows: **IS Sharpe**, Annualized Return, Max Drawdown, and Turnover.

### Step 3: Analyze Results
Look at ALL metrics against the 4H quality gates:
- **IS Sharpe > 2.0**: Primary signal strength threshold. ✓
- **Turnover < 0.30**: Maximum turnover per 4h bar. ✓
- **Fitness > 5.0**: Quality metric (Sharpe × sqrt(Ret/TO)). ✓
- **Sub-period stability**: Should work in both halves of the 2021-2025 train set. ✓

### Step 4: Save Good Alphas

**Evaluate and save the alpha:**
`python eval_alpha.py --expr "<YOUR_EXPRESSION>" --save --reasoning "Economic explanation" --universe BINANCE_TOP50`

The `--save` flag checks:
1. **Quality gates**: IS Sharpe > 2.0, Turnover < 0.30, Fitness > 5.0.
2. **Correlation check**: |corr| > 0.70 with any existing alpha leads to **automatic rejection**.

### Step 5: Report Progress
After every 3-5 new alphas, print the library status:
`python eval_alpha.py --list --universe BINANCE_TOP50`

## Available Data Fields (42 terminals)
All DataFrames of shape (dates × tickers). Source: `data/binance_cache/matrices/4h/`:

**Price/OHLCV**: `close`, `open`, `high`, `low`, `volume`, `quote_volume`
**Returns**: `returns`, `log_returns`
**VWAP**: `vwap`, `vwap_deviation`
**Orderflow**: `taker_buy_ratio`, `taker_buy_volume`, `trades_count`, `trades_per_volume`
**Volume**: `adv20`, `adv60`, `volume_ratio_20d`, `volume_momentum_1`, `dollars_traded`
**Momentum**: `momentum_5d`, `momentum_20d`, `momentum_60d`
**Volatility**: `historical_volatility_20`, `parkinson_volatility_20`
**Candlestick**: `high_low_range`, `open_close_range`, `upper_shadow`, `lower_shadow`, `close_position_in_range`
**Structural**: `beta_to_btc`, `overnight_gap` (meaningful at 4h resolution)
**Carry**: `funding_rate` (highly predictive at 4h)

## Available Operators

Standard time-series and cross-sectional operators are supported.
**Time-Series**: `ts_delta`, `ts_rank`, `sma`, `stddev`, `ts_min`, `ts_max`, `ts_sum`, `ts_zscore`, `ts_skewness`, `ts_kurtosis`, `ts_corr`, `ts_cov`, `ts_regression`, `delay`, `Decay_exp`
**Cross-Sectional**: `rank`, `scale`, `zscore_cs`
**Arithmetic**: `add`, `subtract`, `multiply`, `true_divide`, `Abs`, `Log`, `square`, `sqrt`, `df_max`, `df_min`, `s_log_1p`

## Anti-Overfitting Rules (4h-specific)
1. **Use round lookbacks**: 6 (1d), 12 (2d), 30 (5d), 60 (10d), 120 (20d).
2. **Turnover is critical**: 4H alphas must be slow-moving. Max turnover < 0.30 per bar.
3. **Diversity gate**: |corr| < 0.70. High correlation with existing beta-anomaly signals will result in rejection. Focus on orthogonal signals (funding dynamics, volume kurtosis, etc.).
4. **Volume Smoothing**: Use `s_log_1p()` for volume-based fields to reduce outlier noise.

## 4H Quality Gates Summary

| Metric | Target | Notes |
|---|---|---|
| **IS Sharpe** | **> 2.0** | Tightened target for high-conviction signals. |
| **Turnover** | **< 0.30** | Maximum turnover ceiling. |
| **Fitness** | **> 5.0** | Primary hurdle for high-conviction alphas. |
| **Corr Cutoff** | **< 0.70** | Orthogonality with current 22-alpha portfolio. |
| **Sub-period** | **Both > 0** | Sharpe must be positive in both sub-periods of train. |
