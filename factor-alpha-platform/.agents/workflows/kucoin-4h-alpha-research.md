---
description: Run autonomous 4h crypto alpha research on KuCoin - LLM discovers KuCoin perpetual futures alpha factors at 4-hour frequency
---

# 4h KuCoin Alpha Research Agent (Agent 1 — Discovery)

You are Agent 1: an autonomous alpha researcher for **KuCoin crypto perpetual futures** (4h bars, ZERO FEES). You discover cross-sectional alpha factors using ONLY the train set.

**Universe**: Default: `KUCOIN_TOP100`. Options: `KUCOIN_TOP50`, `KUCOIN_TOP100`.

**Primary Objective: Highest Sharpe** — Sharpe ratio is the #1 metric for 4H alpha.
**Secondary Objective: High Fitness** — Fitness metric ensures the alpha survives transaction costs.

> ⚠️ This is the **4h KuCoin** workflow. Do NOT mix with the Binance or equities workflows.

> [!CAUTION]
> ## HARD RULE: Data Split Discipline
>
> | Split | Period | Purpose | Who Uses |
> |-------|--------|---------|----------|
> | **Train** | **Sep 1 2023 – Sep 1 2025** | Alpha signal DISCOVERY | `eval_alpha.py` only |
> | **Val** | Sep 1 – Dec 1 2025 | Portfolio optim / signal combination | Portfolio scripts |
> | **Test** | Dec 1 2025 – Apr 2026 | FINAL TEST ONLY | Never touch until done |
>
> Train covers **2 years** across bear/recovery/bull cycles.
>
> - **NEVER** discover alphas on Val or Test data
> - **NEVER** optimize portfolio params on Train data
> - **NEVER** evaluate Test until ALL research is complete

## STRICT RULES — DO NOT VIOLATE

- **Discovery evaluates on TRAIN ONLY**: Use `python eval_alpha.py --expr "<EXPRESSION>" --universe KUCOIN_TOP50`.
- **Set exchange before evaluating**: The eval script detects exchange from the universe prefix automatically.
- **NO AUTOMATED SCRIPTS**: Manually hypothesize each alpha expression.
- **Do NOT edit eval_alpha.py, any files in src/, or any other scripts.**
- **Do NOT explore or read other files** in the project — everything you need is in this workflow.
- **Do NOT create new scripts** — propose expressions and run the eval scripts.
- **Do NOT modify the database directly** — only use `--save` to add alphas.
- **Do NOT discover "Combination Alphas"** — every alpha must be an original hypothesis.
- **Do NOT look at validation or test data** during discovery.

## Setup (run once at start)

1. Check current state: `python eval_alpha.py --list --universe KUCOIN_TOP50`

> **📖 Before hypothesizing, read [`.agents/kucoin-alpha-templates.md`](./../kucoin-alpha-templates.md)**
> This file contains proven templates adapted for KuCoin's available fields.

## Research Loop

Repeat this loop indefinitely. Each iteration = one alpha hypothesis.

### Step 1: Hypothesize
Formulate a hypothesis for a predictive signal using **only KuCoin-available fields**.
Every alpha must have a clearly stated reason for why it works at 4h frequency.

### Step 2: Evaluate

**Testing 1 alpha at a time:**
`python eval_alpha.py --expr "<YOUR_EXPRESSION>" --universe KUCOIN_TOP50`

This shows: **IS Sharpe**, Annualized Return, Max Drawdown, and Turnover.

### Step 3: Analyze Results
Look at ALL metrics against the 4H quality gates:
- **IS Sharpe > 3.0**: Primary signal strength threshold. ✓
- **Turnover < 0.30**: Maximum turnover per 4h bar. ✓
- **Fitness > 5.0**: Quality metric (Sharpe × sqrt(Ret/TO)). ✓
- **Sub-period stability**: Should work in both halves of the train set. ✓

### Step 4: Save Good Alphas

`python eval_alpha.py --expr "<YOUR_EXPRESSION>" --save --reasoning "Economic explanation" --universe KUCOIN_TOP50`

### Step 5: Report Progress
`python eval_alpha.py --list --universe KUCOIN_TOP50`

## Available Data Fields (31 terminals)
All DataFrames of shape (dates x tickers). Source: `data/kucoin_cache/matrices/4h/`:

**Price/OHLCV**: `close`, `open`, `high`, `low`, `volume`, `quote_volume`, `turnover`
**Returns**: `returns`, `log_returns`
**VWAP**: `vwap`, `vwap_deviation`
**Volume**: `adv20`, `adv60`, `volume_ratio_20d`, `volume_momentum_1`, `volume_momentum_5_20`, `dollars_traded`
**Momentum**: `momentum_5d`, `momentum_20d`, `momentum_60d`
**Volatility**: `historical_volatility_10`, `historical_volatility_20`, `historical_volatility_60`, `historical_volatility_120`
**Parkinson Vol**: `parkinson_volatility_10`, `parkinson_volatility_20`, `parkinson_volatility_60`
**Candlestick**: `high_low_range`, `open_close_range`, `close_position_in_range`
**Structural**: `beta_to_btc`

> [!NOTE]
> **Why no `overnight_gap`, `upper_shadow`, `lower_shadow`?**
> Crypto trades 24/7 so `overnight_gap` is always ~0 (meaningless).
> KuCoin's kline API does not return reliable wick data (high/low are proxied from open/close),
> so `upper_shadow` and `lower_shadow` are always exactly 0 and provide no signal.

> [!WARNING]
> ## Fields NOT available on KuCoin (DO NOT USE)
> - `funding_rate`, `funding_rate_avg_7d`, `funding_rate_cumsum_3`, `funding_rate_zscore`
> - `taker_buy_ratio`, `taker_buy_volume`, `taker_buy_quote_volume`
> - `trades_count`, `trades_per_volume`
>
> Using these fields will cause expression evaluation to fail.

## Available Operators

**Time-Series**: `ts_delta`, `ts_rank`, `sma`, `stddev`, `ts_min`, `ts_max`, `ts_sum`, `ts_zscore`, `ts_skewness`, `ts_kurtosis`, `ts_corr`, `ts_cov`, `ts_regression`, `delay`, `Decay_exp`
**Cross-Sectional**: `rank`, `scale`, `zscore_cs`
**Arithmetic**: `add`, `subtract`, `multiply`, `true_divide`, `Abs`, `Log`, `square`, `sqrt`, `df_max`, `df_min`, `s_log_1p`

## Anti-Overfitting Rules (4h-specific)
1. **Use round lookbacks**: 6 (1d), 12 (2d), 30 (5d), 60 (10d), 120 (20d).
2. **Turnover is critical**: 4H alphas must be slow-moving. Max turnover < 0.30 per bar.
3. **Diversity gate**: |corr| < 0.70.
4. **Volume Smoothing**: Use `s_log_1p()` for volume-based fields to reduce outlier noise.

## 4H Quality Gates Summary

| Metric | Target | Notes |
|---|---|---|
| **IS Sharpe** | **> 3.0** | Tightened target for high-conviction signals. |
| **Turnover** | **< 0.30** | Maximum turnover ceiling. |
| **Fitness** | **> 5.0** | Primary hurdle for high-conviction alphas. |
| **Corr Cutoff** | **< 0.70** | Orthogonality with existing alpha portfolio. |
| **Sub-period** | **Both > 0** | Sharpe must be positive in both sub-periods of train. |
