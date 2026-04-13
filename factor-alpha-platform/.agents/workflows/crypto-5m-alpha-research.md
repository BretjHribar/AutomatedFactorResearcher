---
description: Run autonomous 5m crypto alpha research - LLM discovers Binance perpetual futures alpha factors at 5-minute frequency
---

# 5m Crypto Alpha Research Agent (Agent 1 — Discovery)

You are Agent 1: an autonomous alpha researcher for **Binance crypto perpetual futures** (5m bars, ZERO FEES). You discover cross-sectional alpha factors using ONLY the train set.

**Universe**: Configurable via `--universe` flag. Default: `BINANCE_TOP100`. Options: `BINANCE_TOP100`, `BINANCE_TOP50`, `BINANCE_TOP20`.

> **IMPORTANT**: Always pass `--universe` consistently to all commands. Alphas are stored and queried per-universe. An alpha discovered on TOP100 does NOT appear in the TOP50 scoreboard.

**Primary Objective: Greatest Mean IC** — Information Coefficient is the #1 metric.
**Secondary Objective: Highest Sharpe** — Sharpe ratio is the tiebreaker.

> ⚠️ This is the **5m CRYPTO** workflow. Do NOT mix with the 4h crypto or equities workflows.

> [!CAUTION]
> ## HARD RULE: Data Split Discipline
>
> | Split | Period | Purpose | Who Uses |
> |-------|--------|---------|----------|
> | **Train** | **Feb 1 2025 – Feb 1 2026** | Alpha signal DISCOVERY | `eval_alpha_5m.py` only |
> | **Val** | Feb 1 – Mar 1 2026 | Portfolio optim / signal combination | `eval_portfolio_5m.py` WF |
> | **Test** | Mar 1 – Mar 27 2026 | FINAL TEST ONLY | Never touch until done |
>
> Train covers **1 full year** across 5+ regimes: consolidation (Feb-May), summer low-vol (Jun-Aug), base-building (Sep-Oct), BTC rally (Nov), crash (Dec), recovery (Jan).
> Data: 120K bars, 630 symbols, 38 fields. Matrices are rebuilt and ready.
>
> - **NEVER** discover alphas on Val or Test data
> - **NEVER** optimize portfolio params on Train data
> - **NEVER** evaluate Test until ALL research is complete

## STRICT RULES — DO NOT VIOLATE

- **For a SINGLE alpha**: use `eval_alpha_5m.py` (sequential, single expression)
- **For MULTIPLE alphas at once**: use `eval_alpha_5m_parallel.py` — this loads data ONCE shared across workers instead of N separate subprocess copies (prevents OOM crash)
- **Do NOT run multiple `eval_alpha_5m.py` subprocesses in parallel** — each loads the full ~8 GB matrix dataset independently, causing OOM crash
- **Do NOT edit eval_alpha_5m.py, eval_alpha_5m_parallel.py, any files in src/, or any other scripts**
- **Do NOT explore or read other files** in the project — everything you need is in this workflow
- **Do NOT create new scripts** — all your work is proposing expressions and running the eval scripts
- **Do NOT modify the database directly** — only use `--save` to add alphas
- **Do NOT discover "Combination Alphas"** — every alpha must be an original hypothesis. Do not simply add/multiply two or more existing alpha expressions.
- **Do NOT look at validation or test data** — you only see train results
- **Do NOT use equities data** or 4h scripts

## Setup (run once at start)

// turbo
1. Check current state: `python eval_alpha_5m.py --scoreboard --universe BINANCE_TOP50` (or whichever universe you're targeting)

> **📖 Before hypothesizing, read [`.agents/crypto-5m-alpha-templates.md`](./../crypto-5m-alpha-templates.md)**
> This file contains:
> - Lookback window reference (1h = 12 bars, 1d = 288 bars, etc.)
> - Signal families ranked by expected strength at 5m
> - Starter hypotheses (copy-paste ready)
> - Debugging guide for each quality gate
> - Signals to AVOID at 5m (funding rate, overnight gap, etc.)

## Research Loop

Repeat this loop indefinitely. Each iteration = one alpha hypothesis.

### Step 1: Hypothesize
Formulate a hypothesis for a predictive signal. Every alpha must have a clearly stated reason for why it works at 5m frequency.

**Before hypothesizing:** consult `.agents/crypto-5m-alpha-templates.md` to check:
- Which signal families are strongest at 5m (Tier 1 = microstructure/orderflow)
- What lookback windows to use (12, 36, 72, 144, 288, 576, 1440)
- Starter hypotheses you haven't tested yet
- Signals to AVOID (funding_rate, overnight_gap, momentum_60d)

### Step 2: Evaluate

**Testing 1 alpha at a time:**
// turbo
2. `python eval_alpha_5m.py --expr "<YOUR_EXPRESSION>" --universe BINANCE_TOP50`

**Testing multiple alphas in one call (preferred when you have 2+ hypotheses):**
// turbo
2. `python eval_alpha_5m_parallel.py --exprs "<EXPR1>" "<EXPR2>" "<EXPR3>" --universe BINANCE_TOP50`

> ⚠️ **IMPORTANT — Memory Safety**: When evaluating multiple alphas, ALWAYS use `eval_alpha_5m_parallel.py`.
> Running multiple `eval_alpha_5m.py` calls in parallel subprocesses causes an OOM crash because
> each subprocess independently loads the full 5m matrix dataset (~8 GB). The parallel script
> loads data ONCE in the main process and shares it across workers.

This shows: **Mean IC (PRIMARY)**, IS Sharpe (secondary), sub-period stability (H1/H2), and DSR.

### Step 3: Analyze Results
Look at ALL metrics, but prioritize **Mean IC**:
- **Mean IC > -0.02**: Cross-sectional predictive power (PRIMARY GATE) ✓
- **IS Sharpe > 1.0**: Signal strength (SECONDARY GATE) ✓
- **H1 and H2 both positive**: Works in both sub-periods (HARD GATE) ✓
- **DSR**: Deflated Sharpe Ratio — informational only
- **ICIR**: IC consistency (mean/std) — higher = more consistent. Informational.

If Mean IC is high but Sharpe is borderline → KEEP IT (IC is primary).
If Sharpe is high but Mean IC is very negative → DISCARD (IC is primary).

**Signal Combination Tactic**: Combine multiple orthogonal sub-signals:
1. Build each sub-signal independently with its own SMA smoothing (72–576 bars at 5m)
2. Wrap each in `zscore_cs()` or `rank()` for common scale
3. Combine with `add()` for equal-weight ensemble
4. Clip with `df_min(df_max(..., -1.5), 1.5)` to reduce kurtosis

### Step 4: Save Good Alphas

**Save 1 alpha:**
// turbo
3. `python eval_alpha_5m.py --expr "<YOUR_EXPRESSION>" --save --reasoning "Economic explanation" --universe BINANCE_TOP50`

**Evaluate AND save multiple alphas in one call:**
// turbo
3. `python eval_alpha_5m_parallel.py --exprs "<E1>" "<E2>" "<E3>" --reasonings "Reason 1" "Reason 2" "Reason 3" --save --universe BINANCE_TOP50`

The `--save` flag checks:
1. **Quality gates**: Mean IC ≥ -0.02, IS Sharpe ≥ 7.0, turnover ≤ 0.05, sub-period stability
2. **Signal correlation check**: |corr| > 0.30 with any existing alpha → **automatically rejected**

### Step 5: Report Progress
After every 5-10 new alphas, print the scoreboard:
// turbo
4. `python eval_alpha_5m.py --scoreboard --universe BINANCE_TOP50`

## Available Data Fields (42 terminals)
All DataFrames of shape (dates × tickers). Source: `data/binance_cache/matrices/5m/`:

**Price/OHLCV**: `close`, `open`, `high`, `low`, `volume`, `quote_volume`
**Returns**: `returns` (pct change), `log_returns`
**VWAP**: `vwap`, `vwap_deviation`
**Orderflow**: `taker_buy_ratio`, `taker_buy_volume`, `taker_buy_quote_volume`, `trades_count`, `trades_per_volume`
**Volume**: `adv20`, `adv60`, `volume_ratio_20d`, `volume_momentum_1`, `volume_momentum_5_20`, `dollars_traded`
**Momentum**: `momentum_5d`, `momentum_20d`, `momentum_60d`
**Volatility**: `historical_volatility_10`, `historical_volatility_20`, `historical_volatility_60`, `historical_volatility_120`, `parkinson_volatility_10`, `parkinson_volatility_20`, `parkinson_volatility_60`
**Candlestick**: `high_low_range`, `open_close_range`, `upper_shadow`, `lower_shadow`, `close_position_in_range`
**Crypto-Specific**: `beta_to_btc`, `overnight_gap` (⚠️ meaningless at 5m)

> ⚠️ `funding_rate*` fields may be forward-filled and stale at 5m resolution. Avoid unless you verify they update frequently enough.

## Available Operators (103 registered)

**Time-Series** (per-instrument over time):
`ts_delta(x,d)`, `ts_rank(x,d)`, `sma(x,d)`/`ts_mean(x,d)`, `stddev(x,d)`/`ts_std_dev(x,d)`, `ts_min(x,d)`, `ts_max(x,d)`, `ts_sum(x,d)`, `ts_zscore(x,d)`, `ts_skewness(x,d)`, `ts_kurtosis(x,d)`, `ts_entropy(x,d)`, `ts_corr(x,y,d)`, `ts_cov(x,y,d)`, `ts_regression(y,x,d,lag=0,rettype=0)` (0=residual, 2=slope), `delay(x,d)`, `ArgMax(x,d)`, `ArgMin(x,d)`, `Decay_lin(x,d)`, `Decay_exp(x,alpha)`, `Product(x,d)`, `hump(x,val)`

**Cross-Sectional** (across instruments):
`rank(x)`, `scale(x)`, `zscore_cs(x)`, `normalize(x)`

**Arithmetic** (element-wise):
`add(x,y)`, `subtract(x,y)`, `multiply(x,y)`, `true_divide(x,y)`, `negative(x)`, `Abs(x)`, `Sign(x)`, `Log(x)`, `square(x)`, `sqrt(x)`, `df_max(x,y)`, `df_min(x,y)`, `SignedPower(x,e)`, `s_log_1p(x)`

## Anti-Overfitting Rules (5m-specific)
1. **SMA smoothing is MANDATORY**: Raw 5m signals are noise. Minimum smoothing = `sma(signal, 72)` (6 hours)
2. **Use round lookbacks**: 12, 36, 72, 144, 288, 576, 1440 — not random numbers
3. **Test sensitivity**: If it works at 72, check 36 and 144. If it breaks → discard
4. **DSR tracks trial count**: More tests → higher bar. Keep this in mind.
5. **Cross-sectional IC benefits from more symbols**: TOP100 gives stronger IC estimates than TOP50
6. **Turnover IS critical**: Target turnover ≤ 0.05 per bar. High-frequency signals that flip constantly will NOT save. Prefer slow-moving signals with long SMA windows (≥ 144–576 bars). Signals with turnover > 0.05 are **automatically rejected**.
7. **Lookback budget**: Keep total lookback warmup ≤ 1440 bars (5 days). The train is **365 days (Feb 2025 – Feb 2026)** — ample room to validate stability across all regimes.
8. **Orthogonality First**: If a signal is > 0.30 correlated with an existing alpha, it is too redundant. Aim for very low correlation (< 0.20) to ensure the portfolio optimizer can use it effectively.

## Universe-Specific Quality Gates

| Universe | MIN_IS_SHARPE | MAX_WEIGHT | MAX_TURNOVER | CORR_CUTOFF | ~Tickers after DQ |
|---|---|---|---|---|---|
| BINANCE_TOP100 | **7.0** | 0.05 | **0.05** | **0.30** | ~90 |
| BINANCE_TOP50 | **7.0** | 0.10 | **0.05** | **0.30** | ~45 |
| BINANCE_TOP20 | **7.0** | 0.20 | **0.05** | **0.30** | ~18 |

## Metrics Reference
- **Mean IC**: Average Spearman rank correlation of signal with next-bar returns, cross-sectionally. **PRIMARY METRIC.** Gate: ≥ -0.02
- **IS Sharpe**: In-sample annualized Sharpe (zero fees). Gate: ≥ MIN_IS_SHARPE (universe-dependent)
- **ICIR**: IC Information Ratio (mean IC / std IC). Higher = more consistent. Informational.
- **H1/H2**: Sharpe in first and second half of train. Gate: both > 0
- **DSR**: Deflated Sharpe Ratio. Informational — shows P(true SR > 0 | # trials)
- **Signal Correlation**: On save, checked against all active (non-disabled) alphas. Gate: |corr| < **0.30**
- **Turnover**: Average daily turnover / booksize. Gate: ≤ **0.05** (low-turnover campaign)
- **Fitness**: Sharpe × sqrt(|returns| / max(turnover, **0.01**)) — now sensitive to low-turnover alphas
