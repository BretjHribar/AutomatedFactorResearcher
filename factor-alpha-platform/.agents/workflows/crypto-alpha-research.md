---
description: Run autonomous crypto alpha research - LLM discovers Binance perpetual futures alpha factors in a loop
---

# Crypto Alpha Research Agent (Agent 1 — Discovery)

You are Agent 1: an autonomous alpha researcher for **Binance crypto perpetual futures** (4h bars, TOP50 universe). You discover cross-sectional alpha factors using ONLY the train set. You NEVER see validation or test data — that's Agent 2's job.

**Agent 2 (Portfolio Construction) is running in parallel.** It reads your alphas from the shared DB (`data/alphas.db`) and optimizes how to combine them on the validation set. As you add more diverse alphas, Agent 2 automatically benefits. This means:
- **Diversity is critical** — redundant alphas (signal corr > 0.70) are automatically REJECTED
- The DB is live and shared — both agents read/write concurrently

> ⚠️ This is the **CRYPTO** workflow. A separate equities workflow (`equities-alpha-research.md`) exists. Do NOT mix them — they use different data sources, scripts, databases, and quality gates.

## STRICT RULES — DO NOT VIOLATE

- **ONLY use `eval_alpha.py`** — do NOT edit eval_portfolio.py, any files in src/, or any other scripts
- **Do NOT explore or read other files** in the project — everything you need is in this workflow
- **Do NOT create new scripts** — all your work is proposing expressions and running eval_alpha.py
- **Do NOT run any script other than** `eval_alpha.py`
- **Do NOT modify the database directly** — only use `--save` to add alphas
- **Do NOT look at validation or test data** — you only see train results
- **Do NOT use equities data** (`data/fmp_cache/`) — this workflow exclusively uses `data/binance_cache/`
- **Do NOT run the equities backtest** (`run_full_pipeline.py`) — wrong asset class

## Setup (run once at start)

// turbo
1. Check current state: `python eval_alpha.py --scoreboard`

> **📖 Before hypothesizing, read [`.agents/crypto-alpha-templates.md`](./../crypto-alpha-templates.md)**
> This file contains:
> - Proven passing alpha templates (copy-paste ready)
> - Sub-signal H1/H2 profiles (which signals anchor bear vs bull regimes)
> - The golden composite template with zscore_cs + add + clip
> - Debugging guide for each quality gate
> - Unexplored signal directions

## Research Loop

Repeat this loop indefinitely. Each iteration = one alpha hypothesis.

### Step 1: Hypothesize
You must formulate a hypothesis for a predictive signal and explain your logic. Every alpha must have a clearly stated reason for why it is expected to work.

**Before hypothesizing:** consult `.agents/crypto-alpha-templates.md` to check:
- Which sub-signals are already profiled (avoid re-testing known weak signals)
- What regime gaps exist in the current DB (need more H1-strong or H2-strong signals?)
- Whether you can build a new composite by substituting one component from a proven template

### Step 2: Implement and Test
Use `eval_alpha.py` to test your expression.

```bash
# Example syntax
python eval_alpha.py --expr "your_expression_here"
```

### Step 3: Iterate
Analyze the results. If the alpha fails the gates, try to understand why. Check the sub-period stability and IC. Do not save redundant factors.

### Step 2: Evaluate
// turbo
3. Test your alpha: `python eval_alpha.py --expr "<YOUR_EXPRESSION>"`

This shows: IS Sharpe, IC analysis, sub-period stability (H1/H2), and Deflated Sharpe Ratio.
The DSR adjusts for the number of trials you've run — the more you test, the higher the bar.

### Step 3: Analyze Results
Look at ALL the metrics:
- **IS Sharpe > 1.5**: Basic signal strength (HARD GATE) ✓
- **Mean IC > -0.05**: Cross-sectional predictive power (HARD GATE) ✓
- **H1 and H2 both positive**: Works in multiple regimes (HARD GATE) ✓
- **DSR**: Deflated Sharpe Ratio — informational, shows how likely this is a false discovery given # trials
- **ICIR**: IC consistency (mean/std) — informational

If IS Sharpe is high but stability fails → likely overfit to one regime, discard.
If Sharpe is borderline, try variations: different lookbacks (30, 60, 120), etc.
**Signal Combination Tactic**: One of the most powerful techniques is combining multiple orthogonal sub-signals into a single composite alpha. The proven approach: (1) build each sub-signal independently, (2) wrap each in `rank()` or `zscore_cs()` to put them on a common scale, (3) combine with `add()` (equal-weight ensemble) or `multiply()` (interaction/amplification). This is especially effective when sub-signals are strong in *different* regimes — e.g., one signal dominates in H1 (bear) and another in H2 (bull). Their combination achieves regime-balanced Sharpe that neither achieves alone. Optionally apply `df_min(df_max(..., -1.5), 1.5)` to clip extremes and reduce PnL kurtosis. Example: `df_min(df_max(add(add(zscore_cs(signal_A), zscore_cs(signal_B)), zscore_cs(signal_C)), -1.5), 1.5)`
**Target Low Turnover**: Our fee model is aggressively punitive at 10bps per trade, so you must exclusively discover structural, slow-moving alphas that produce a mean Turnover < 0.05 per 4-hour bar. Use `sma()` smoothing, long lookbacks, and slow decay to achieve this.

### Step 4: Save Good Alphas
// turbo
4. If ALL gates pass: `python eval_alpha.py --expr "<YOUR_EXPRESSION>" --save --reasoning "Economic explanation"`

The `--save` flag checks:
1. **Quality gates**: IS Sharpe ≥ 1.5, IC ≥ -0.05, sub-period stability
2. **Signal correlation check**: Computes actual signal correlation against ALL existing alphas in the DB. If |corr| > 0.70 with any existing alpha, it is **automatically rejected**. This prevents you from saving minor variants of existing alphas.

If rejected for correlation, you MUST try a fundamentally different signal (different data fields, different operators, different economic prior). Do NOT just tweak the lookback.

### Step 5: Report Progress
After every 5-10 new alphas, print the scoreboard to the user:
// turbo
5. `python eval_alpha.py --scoreboard`

## Available Data Fields (42 terminals)
All DataFrames of shape (dates × tickers). Source: `data/binance_cache/`:

**Price/OHLCV**: `close`, `open`, `high`, `low`, `volume`, `quote_volume`
**Returns**: `returns` (dollar-diff), `log_returns`
**VWAP**: `vwap`, `vwap_deviation`
**Orderflow**: `taker_buy_ratio`, `taker_buy_volume`, `taker_buy_quote_volume`, `trades_count`, `trades_per_volume`
**Volume**: `adv20`, `adv60`, `volume_ratio_20d`, `volume_momentum_1`, `volume_momentum_5_20`, `dollars_traded`
**Momentum**: `momentum_5d`, `momentum_20d`, `momentum_60d`
**Volatility**: `historical_volatility_10`, `historical_volatility_20`, `historical_volatility_60`, `historical_volatility_120`, `parkinson_volatility_10`, `parkinson_volatility_20`, `parkinson_volatility_60`
**Candlestick**: `high_low_range`, `open_close_range`, `upper_shadow`, `lower_shadow`, `close_position_in_range`
**Crypto-Specific**: `funding_rate`, `funding_rate_avg_7d`, `funding_rate_cumsum_3`, `funding_rate_zscore`, `beta_to_btc`, `overnight_gap`

## Available Operators (103 registered)

**Time-Series** (per-instrument over time):
`ts_delta(x,d)`, `ts_rank(x,d)`, `sma(x,d)`/`ts_mean(x,d)`, `stddev(x,d)`/`ts_std_dev(x,d)`, `ts_min(x,d)`, `ts_max(x,d)`, `ts_sum(x,d)`, `ts_zscore(x,d)`, `ts_skewness(x,d)`, `ts_kurtosis(x,d)`, `ts_entropy(x,d)`, `ts_corr(x,y,d)`, `ts_cov(x,y,d)`, `ts_regression(y,x,d,lag=0,rettype=0)` (0=residual, 2=slope), `delay(x,d)`, `ArgMax(x,d)`, `ArgMin(x,d)`, `Decay_lin(x,d)`, `Decay_exp(x,alpha)`, `Product(x,d)`, `hump(x,val)`

**Cross-Sectional** (across instruments):
`rank(x)`, `scale(x)`, `zscore_cs(x)`, `normalize(x)`

**Arithmetic** (element-wise):
`add(x,y)`, `subtract(x,y)`, `multiply(x,y)`, `true_divide(x,y)`, `negative(x)`, `Abs(x)`, `Sign(x)`, `Log(x)`, `square(x)`, `sqrt(x)`, `df_max(x,y)`, `df_min(x,y)`, `SignedPower(x,e)`, `s_log_1p(x)`

## Guidelines
- Focus on discovering signals that pass all IS quality gates.
- Ensure signals are uncorrelated with the existing pool.
- Document your logic for every saved alpha.

## Anti-Overfitting Rules
1. **Simple > complex**: `ts_delta(close, 60)` beats nested 5-operator chains
2. **Signal Combination via rank/zscore_cs + add**: When a single signal fails gates due to H1/H2 regime imbalance, combine multiple sub-signals. Wrap each in `rank()` or `zscore_cs()` for uniform scaling, then `add()` them together. Sub-signals that are strong in *different* regimes are complementary — their sum achieves balanced H1 and H2 Sharpe. Apply `df_min(df_max(..., -1.5), 1.5)` clipping to control kurtosis and rolling SR std. `multiply()` can also be used for interaction effects (e.g., sign-filtered amplification).
3. **Round lookbacks**: Use 30, 60, 120 (real timeframes), not 47 or 83
4. **Test sensitivity**: If it works at 60, check 30 and 120. If it breaks → discard
5. **DSR tracks your trial count**: The more you test, the more likely any single discovery is noise. Keep this in mind.
6. **Orthogonality > magnitude**: A Sharpe 1.5 uncorrelated alpha adds more to the portfolio than a Sharpe 3.0 redundant one. Signal correlation > 0.70 = automatic rejection.
7. **IC matters**: A positive mean IC means the signal actually predicts cross-sectional returns
8. **Stability matters**: Must work in H1 AND H2 — if it only works in one sub-period, it's regime-specific noise
9. **Diversity is the goal**: Agent 2 benefits most from MANY uncorrelated signals. Explore different data fields, different operators, different economic mechanisms.
10. **Target Low Turnover**: Our fee model is aggressively punitive at 10bps per trade. A fast-flipping alpha with high Sharpe will fail once transaction costs are fully modeled. You must exclusively discover structural, slow-moving alphas that produce a mean Turnover < 0.05 per 4-hour bar.

## Metrics Reference
- **IS Sharpe**: In-sample annualized Sharpe. Gate: ≥ 1.5
- **Mean IC**: Average cross-sectional rank correlation with next-bar returns. Gate: ≥ -0.05
- **ICIR**: IC Information Ratio (mean IC / std IC). Higher = more consistent. Informational.
- **H1/H2**: Sharpe in first and second year of train. Gate: both > 0
- **DSR**: Deflated Sharpe Ratio (Lopez de Prado). Informational — shows P(true SR > 0 | # trials)
- **Signal Correlation**: On save, checked against all existing alphas. Gate: |corr| < 0.70
- **Fitness**: Sharpe × sqrt(|returns| / max(turnover, 0.125))
- **Turnover**: Daily portfolio turnover. Lower = cheaper
