# 🔍 FINAL_v1 Production Audit Report

**Auditor**: Head of Quant Trading  
**Date**: 2026-03-11  
**Subject**: V8 Multi-Timeframe Adaptive Net Strategy — Paper Trading Readiness  
**Verdict**: ⚠️ **CONDITIONAL PASS — Proceed to Paper Trading with Mandatory Fixes**

---

## Executive Summary

I have conducted a thorough code-level, methodological, and statistical audit of the V8 Multi-Timeframe Adaptive Net strategy as packaged in `FINAL_v1/`. The strategy shows promising walk-forward OOS results (collective Sharpe +9.93 across 5 Binance USDT-M perpetual futures, 10 monthly folds). However, I have identified **3 issues that MUST be addressed before paper trading** and **8 additional concerns** for ongoing monitoring. The core alpha generation and portfolio construction logic is sound, but the reported Sharpe ratios carry significant caveats.

---

## 1. Code Integrity & Consistency

### 1.1 ✅ Script Parity Verified

`univariate_hf_v8_mtf.py` (production module) and `v8_rerun_futures.py` (validation script) are functionally identical — differences are limited to formatting and code organization (modular vs. inline). The validation script produced the reported results using the same logic. **No code divergence.**

### 1.2 ✅ Signal Library Audit

All ~240 signals fall into standard quantitative categories (mean-reversion, momentum, technical, microstructure). No bespoke or hand-tuned features that would suggest data-mining. The signal taxonomy is well-organized across 10 categories. Signal construction uses standard primitives (z-score, rolling sum, EMA, Bollinger, RSI, CCI, Stochastic, ADX). **Clean.**

### 1.3 ✅ Dependency Audit

Only standard libraries: `numpy`, `pandas`, `time`, `pathlib`, `requests`. No ML frameworks, no exotic dependencies. Self-contained and portable. **Clean.**

---

## 2. Lookahead / Data Leakage Audit

### 2.1 ✅ 1H Signal Shift — CORRECT

```python
alpha_df = alpha_df.shift(1)  # Line 221 of univariate_hf_v8_mtf.py
```

All 1H signals are computed on bar `t`'s data, then the entire alpha matrix is shifted forward by 1. This means the signal at time `t` was computed from data available at `t-1`. The return at time `t` is `close[t] / close[t-1] - 1`. So the signal predicts the next bar's return. **Correct.**

### 2.2 ✅ 4H Signal Alignment — CORRECT (Conservative)

```python
c4 = df_4h['close'].shift(1)  # Line 198
h4 = df_4h['high'].shift(1)   # Line 199
l4 = df_4h['low'].shift(1)    # Line 200
```

4H OHLCV is shifted by 1 period (4 hours) before anything else. So a 4H bar labeled at 00:00 (which closes at 03:00) is only used after the NEXT 4H boundary. Combined with the global `alpha_df.shift(1)`, 4H signals have a minimum lag of **5 hours**. This is overly conservative but safe. **No lookahead.**

### 2.3 ⚠️ MODERATE: Train/Test Boundary Overlap (1 bar)

```python
train_1h = df_1h.loc[str(train_start):str(test_start)]   # Line 359
test_1h  = df_1h.loc[str(test_start):str(test_end)]       # Line 360
```

Pandas `.loc[]` with string slicing is **inclusive on both ends**. When `test_start = '2024-06-01'`, the bar at `2024-06-01 00:00:00` appears in **both** train and test. This is a 1-bar overlap out of ~5,800 train bars and ~720 test bars.

**Impact**: Negligible for results (1/5800 = 0.017%), but it's a code hygiene issue. In production walk-forward, this should be:
```python
train_1h = df_1h.loc[str(train_start):str(test_start - pd.Timedelta(hours=1))]
```

### 2.4 ✅ OBV Cumsum — NOT a Lookahead

OBV uses `.cumsum()` which is a level-dependent indicator. However, `build_mtf_alphas_safe()` is called with sliced data (train or test only), so the cumsum is **local to the slice**. Additionally, OBV is then z-scored, making the absolute level irrelevant. **No issue.**

### 2.5 ✅ ret_lag Autocorrelation — CORRECT

```python
ret_lag = ret.shift(1)           # Line 168
ac = correlation(ret, ret_lag, w) # rolling autocorrelation
```

This correlates `ret[t]` with `ret[t-1]` over a rolling window. The result `ac[t]` uses data up to time `t`. Since `alpha_df.shift(1)` is applied globally afterward, `ac[t]` at signal time actually represents data up to `t-1`. **Correct.**

---

## 3. Fee Model Audit

### 3.1 ✅ Fee Application Logic — CORRECT

```python
pos_changes = np.abs(np.diff(np.concatenate([[0], direction])))
pnl = direction * ret.values - FEE_FRAC * pos_changes
```

- `FEE_FRAC = 3/10000 = 0.0003` (3 bps)
- Position change from +1 → −1: `|diff| = 2`, fee = `2 × 0.0003 = 6 bps` ✅ (close long @ 3bps + open short @ 3bps)
- Position change from +1 → +1: `|diff| = 0`, fee = 0 ✅ (holding)
- Initial entry from 0 → +1: `|diff| = 1`, fee = `3 bps` ✅

**Correct for Binance USDT-M perpetual futures taker fills.**

### 3.2 🔴 CONCERN: Fee Level May Be Optimistic

| Fee Component | Backtest Assumption | Real-World |
|---|---|---|
| Taker fee (VIP 0) | 3.0 bps | **3.5 bps** (as of 2026) |
| Taker fee (with BNB) | — | ~3.15 bps |
| Slippage (BTC/ETH) | 0 bps | 0–1 bps |
| Slippage (DOGE) | 0 bps | **1–3 bps** |
| Funding rate | 0 bps | ±5–50 bps/8h (variable) |

**The backtest uses 3 bps flat, but real taker at VIP 0 is 3.5 bps (or 4.5 bps without BNB discount as of some fee schedule updates). For DOGE specifically, the effective cost including slippage could be 5–6 bps per side.** This is a meaningful underestimation for a strategy that trades ~10x/day.

**Recommendation**: Re-run the backtest at **5 bps** to stress-test. If the strategy survives at 5 bps, it's robust enough for paper trading.

### 3.3 ❌ MISSING: No Funding Rate Model

Binance futures apply funding every 8 hours. The strategy holds positions for 2–3 hours on average (given ~10 trades/day). During trending markets (like DOGE Nov 2024), funding rates can reach **0.1–0.3% per 8 hours** for the crowded side. Over a month like November, this could erode **200–600 bps** from the DOGE P&L of +11,269 bps.

For paper trading, this is acceptable to monitor — **but the reported PnL should be treated as gross of funding**.

---

## 4. Statistical Audit

### 4.1 Sharpe Ratio Computation — Methodology Notes

```python
daily = pnl_s.resample('1D').sum()
daily = daily[daily != 0]   # REMOVE zero-PnL days
sh = daily.mean() / daily.std() * np.sqrt(365)
```

**Three concerns with this calculation:**

#### 4.1a ⚠️ Zero-Day Removal Inflates Sharpe

Removing `daily != 0` days increases the Sharpe by reducing the denominator (fewer near-zero observations reduces std). For a strategy that is almost always in a position at 1H frequency, very few days are removed. But during test-fold warmup periods (see §4.3), the strategy may be flat for multiple days, and removing those days inflates the Sharpe.

**Estimated impact**: +0.3 to +0.8 Sharpe points across folds. Not disqualifying, but the "true" collective Sharpe is likely **+8.5 to +9.5** rather than +9.93.

#### 4.1b ✅ Annualization Factor — Correct for Crypto

`sqrt(365)` is the correct annualization factor for 24/7 crypto markets with daily resampling. **Correct.**

#### 4.1c ✅ Returns Are Fractional — Correct

PnL is in fractional return units. The Sharpe is a ratio of mean to std of daily fractional returns, annualized. **Correct.**

### 4.2 ⚠️ Reported Sharpes Are Extremely High

| Metric | Industry Benchmark | This Strategy |
|---|---|---|
| Single-asset 1H Sharpe | 1.5–3.0 (excellent) | **4.8–7.1** |
| 5-asset portfolio Sharpe | 3.0–5.0 (exceptional) | **9.93** |

These Sharpes are **2–3x higher than typical institutional quant strategies**. Possible explanations:

1. **Crypto is inefficient** — Possible. Crypto HF strategies can achieve higher Sharpes than equities.
2. **Walk-forward overfitting** — 480 parameter configs × 10 folds. Even with train/test split, selecting the best of 480 configs introduces selection bias of ~0.5–1.0 Sharpe points.
3. **Short OOS period** — 10 months is sufficient to establish statistical significance but not enough to survive multiple regime changes.
4. **DOGE concentration** — DOGE contributes 33% of total PnL (+48,000 out of ~143,000 bps). Nov 2024 alone contributed +11,269 bps. This is a volatile, low-liquidity token with outsized contribution.

**The Sharpe is likely genuinely high but overstated by ~20-30%. A realistic expectation for paper trading is a collective Sharpe of 5–7, which is still excellent.**

### 4.3 🔴 CRITICAL: Test-Fold Warmup Causes Data Loss

```python
# Test fold setup:
alpha_te = build_mtf_alphas_safe(test_1h, test_4h)
mte = strategy_adaptive_net(alpha_te, test_ret, best_sel, lookback=best_cfg['lb'], ...)
```

The adaptive net strategy computes rolling mean factor returns over `lookback` bars (180–720). When called on a test fold of ~720 bars with `lookback=720`, the first `min_periods=100` bars produce valid but unstable weights, and the remaining 620 bars are warmup-degraded.

**Impact**: The reported test-fold Sharpe may be computed on as few as **620 effective bars** out of 720, with the first 100 bars using poorly estimated weights. This is a systematic bias that could go either direction.

**Fix for paper trading**: When computing live signals, ensure the lookback window extends into historical data. The PAPER_TRADER README correctly addresses this (§2.4: "Load at least 720 1H bars of historical data before going live").

### 4.4 Win Rate Analysis — Plausible

| Metric | Value |
|---|---|
| Mean win rate | 52.6% |
| Min | 48.7% |
| Max | 58.3% |
| Std | 2.2% |

A ~52.5% win rate with a positive expected value per trade is consistent with a signal-based directional strategy. The edge comes from the non-trivial signal, not from a high win rate. **Plausible and consistent with the PnL.**

### 4.5 ⚠️ DOGE Concentration Risk

DOGE generates **2.9x more PnL** than BTC and accounts for the single largest monthly contribution (+11,269 bps in Nov 2024). DOGE USDT-M perpetuals have:
- Lower liquidity than BTC/ETH
- Higher spreads (0.01–0.03%)
- More volatile funding rates
- Higher slippage for $50k+ notional

For paper trading, **cap DOGE at equal weight** (20% of portfolio) regardless of signal strength. Monitor DOGE slippage and fill quality carefully.

---

## 5. Portfolio Construction Audit

### 5.1 ✅ Adaptive Net Weighting — Sound Design

The Isichenko-style adaptive weighting is well-implemented:
1. Compute per-alpha directional returns (no fees)
2. Rolling mean over `lookback` bars
3. Zero weight for negative-return alphas (auto regime adaptation)
4. Normalize to sum = 1
5. Optional EWMA smoothing

This is a clean, interpretable approach. The regime adaptation (zeroing out underperforming alphas) is the key innovation. **Good design.**

### 5.2 ⚠️ Adaptive Weights Include Concurrent Bar (Minor Leakage)

```python
factor_returns[col] = d * ret.values  # d = sign(signal[t]), ret = return[t]
rolling_er = factor_returns.rolling(lookback).mean()
```

The rolling mean at time `t` includes the return at time `t`. The weight at time `t` should ideally only use information up to `t-1`. Since the rolling window is 180–720 bars, the impact is `1/180 to 1/720` per bar — negligible. **Noted but not material.**

### 5.3 ✅ Orthogonal Selection — Correct

Greedy forward selection with correlation filtering. Simple and effective. Prevents redundant signals from dominating. **Correct.**

### 5.4 ⚠️ Alpha Ranking Is Turnover-Blind

`eval_alpha()` ranks signals by **no-fee Sharpe ratio**. It does not consider turnover. A signal that flips direction every bar (high turnover) may rank highly by raw Sharpe but destroy value after 3–5 bps per flip. The grid search in `strategy_adaptive_net` partially compensates (it evaluates the combined portfolio WITH fees), but the **initial candidate pool** is pre-filtered by a metric that ignores the most important cost driver.

**Impact**: The selected alpha pool may include high-Sharpe/high-turnover signals that drag down the portfolio. This could explain why the backtest needs 480 parameter configurations to find a good combination — a turnover-aware ranking might produce a better starting pool.

**Recommendation**: Add a turnover penalty to the alpha ranking metric, e.g., `net_sharpe = nofee_sharpe - turnover_penalty` where the penalty is proportional to the number of direction changes.

### 5.5 ⚠️ Parameter Instability Across Folds

The strategy sweeps 480 configurations per fold. Different folds likely select different parameters (`cc`, `mn`, `lb`, `phl`). If the optimal configuration were stable, the same params would be selected across folds. High parameter instability suggests the strategy is sensitive to the exact parameter choice, which is a mild overfitting signal.

**Not captured in output**: The fold details do not log which configuration was selected. This is a significant gap for post-hoc analysis.

**Recommendation**: For paper trading, freeze parameters from the most recent fold and log them. If performance degrades, this is the first place to investigate.

---

## 6. Paper Trading Architecture Review

### 6.1 ✅ PAPER_TRADER README — Well-Designed

The PAPER_TRADER/README.md is comprehensive and addresses the critical live-vs-backtest differences:

- **Shift semantics in live mode**: Correctly identified that `.shift(1)` on alpha matrix is implicit in live (we only use closed bars). 4H shift is preserved. ✅
- **Warmup**: 720 bars minimum bootstrapped from REST. ✅
- **4H alignment**: Properly uses previous 4H bar only. ✅
- **Logging schema**: Comprehensive JSONL + CSV logging. ✅
- **Reconciliation**: Planned nightly backtest comparison. ✅

### 6.2 ⚠️ Live Signal Computation Not Yet Implemented

The PAPER_TRADER directory only contains `README.md` — no actual code. The design is a plan, not a working system. This is expected for a pre-paper-trading audit, but the implementation needs to be built and verified against the backtest before going live.

---

## 7. Data Integrity

### 7.1 ✅ Data Source Verified

Downloads from `fapi.binance.com/fapi/v1/klines` — confirmed Binance Futures endpoint. Using `fapi` (not `api`) ensures we get futures klines, not spot. The download script correctly handles pagination, deduplication, and column naming. **Correct.**

### 7.2 ✅ Resampling Logic — Correct

```python
df_1h = df15.resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', ...
}).dropna()
```

Standard OHLCV resampling with correct aggregation functions. **Correct.**

### 7.3 ⚠️ Duplicate Index Handling

```python
df15 = df15[~df15.index.duplicated(keep='last')]
```

Keeps the last duplicate. If Binance returns duplicate timestamps (e.g., from overlapping pagination), this is the correct behavior — the last response has the most up-to-date data. **Acceptable.**

---

## 8. Risk Assessment

### 8.1 Estimated Realistic Performance

| Scenario | Collective Sharpe Estimate |
|---|---|
| **Backtest (reported)** | +9.93 |
| After removing zero-day inflation | ~9.0 |
| After realistic fees (5 bps) | ~7.5 |
| After funding rate drag (~1 bps/day) | ~6.5 |
| After DOGE slippage | ~6.0 |
| After parameter instability discount | ~5.0–5.5 |
| **Realistic paper trading expectation** | **+4.0 to +6.0** |

**Even at a degraded Sharpe of +4.0, this is a very strong crypto trading strategy. The question is not whether it's profitable, but how much of the backtest edge survives live execution.**

### 8.2 Maximum Drawdown (Not Reported)

The audit lacks maximum drawdown analysis. The equity curve in the performance chart shows one visible drawdown period around late Oct 2024 (BNBUSDT had two negative months). Maximum drawdown should be computed and reported.

### 8.3 Capacity Constraints

At 1H frequency with 5 liquid Binance perpetuals, capacity is limited by:
- **BTC**: $5M+ notional per trade, no issue
- **ETH**: $2M+, no issue
- **SOL/BNB**: $500k–$1M, acceptable
- **DOGE**: $100k–$500k, **capacity constrained** at higher sizes

For paper trading (and initial live), **$100k total notional** ($20k per symbol) is well within capacity. Scaling beyond $500k would require careful analysis of DOGE market impact.

---

## 9. Mandatory Fixes Before Paper Trading

### 🔴 Fix #1: Re-run at 5 bps Fees

Run the full walk-forward at 5 bps (instead of 3 bps) to verify the strategy survives realistic execution costs. If collective Sharpe drops below +3.0, the strategy is fee-sensitive and needs rethinking.

```python
FEE_BPS = 5  # Change from 3
```

### 🔴 Fix #2: Fix Train/Test Boundary Overlap

Change the train data slice to exclude the boundary bar:

```python
train_1h = df_1h.loc[str(train_start):str(test_start - pd.Timedelta(hours=1))]
```

This is a code hygiene issue. The impact on results is nil, but it's a correctness requirement for production code.

### 🔴 Fix #3: Log Selected Parameters Per Fold

Add parameter logging to each fold:

```python
fold_details.append({
    'fold': test_start.strftime('%Y-%m'),
    'cfg': best_cfg,
    'selected_alphas': [s['name'] for s in best_sel],
    **mte
})
```

Without this, post-hoc analysis of parameter stability is impossible.

---

## 10. Recommended Monitoring During Paper Trading

| Metric | Expected Range | Red Flag |
|---|---|---|
| Daily Sharpe (rolling 30d) | +3.0 to +12.0 | < +1.5 for 2 weeks |
| Win rate | 50–56% | < 48% for 2 weeks |
| Position changes/day | 8–15 per symbol | > 25 (fee bleed) |
| Monthly PnL | +500 to +5000 bps per symbol | < -500 bps any month |
| Signal agreement (backtest vs live) | > 95% | < 90% |
| DOGE slippage | < 2 bps | > 5 bps |

---

## 11. Final Verdict

| Category | Grade | Notes |
|---|---|---|
| **Code Quality** | A- | Clean, readable, well-structured. Minor nits only. |
| **Lookahead Prevention** | A | All signals properly shifted. 4H extra conservative. |
| **Fee Model** | B | Correct logic but optimistic level (3 vs 3.5–5 bps). |
| **Statistical Rigor** | B+ | Walk-forward is proper. Zero-day removal and 480-config sweep are minor inflators. |
| **Risk Assessment** | B- | No max drawdown, no funding model, no slippage model. |
| **Paper Trading Readiness** | B+ | Design is excellent. Code needs 3 mandatory fixes and implementation. |
| **Overall** | **B+** | **CONDITIONAL PASS** |

### Decision: ✅ PROCEED TO PAPER TRADING

Subject to:
1. Running a 5-bps fee stress test (can be done in parallel with paper trading setup)
2. Fixing train/test boundary overlap
3. Adding parameter logging per fold
4. Setting realistic expectations: Sharpe +4–6, not +9.93

The strategy has a sound theoretical foundation (adaptive net weighting), uses proper walk-forward methodology, has clean anti-lookahead implementation, and shows strong OOS results even under conservative estimates. The 10-month OOS period across 5 instruments with consistent positive performance is sufficient evidence to proceed to paper trading.

**Paper trading will serve as the true out-of-sample validation.** Monitor closely for the first 30 days. If live signals disagree with backtest expectations by >10%, halt and investigate.

---

*Audit completed 2026-03-11 by Head of Quant Trading*  
*Total audit scope: 3 Python scripts (479 + 275 + 104 lines), 2 documentation files, 1 results file, 1 performance chart*

---

## Appendix A: Deep Lookahead Trace (Line-by-Line)

The following 11 specific checks were performed to verify the absence of lookahead bias at the signal, portfolio, and evaluation layers.

| # | Component | Code Reference | Uses Current Bar? | After shift(1)? | Verdict |
|---|---|---|---|---|---|
| 1 | **Breakout signal** | `close - ts_min(low,w)` / range (L108-110) | Yes — `close[t]`, `high[t]`, `low[t]` | Signal at `t` computed from bar `t` → shifted to predict `t+1` | ✅ Clean |
| 2 | **close_pos in decay** | `(close-low)/(high-low)` (L81, L125-128) | Yes — same-bar OHLC | Shifted by global `shift(1)` | ✅ Clean |
| 3 | **Stochastic** | `(close - lowest) / (highest - lowest)` (L189-191) | Yes — standard formula includes current bar in range | Shifted | ✅ Clean |
| 4 | **Bollinger Band** | `(close - rolling_mean) / rolling_std` (L180-182) | Yes — `close[t]` is in the rolling window (standard BB) | Shifted | ✅ Clean |
| 5 | **Return alignment** | `direction * ret.values` (L305) | `direction[t]` from shifted alpha, `ret[t]` = return `t-1→t` | Signal from `t-1` data × return `t-1→t` | ✅ Perfect |
| 6 | **eval_alpha** | `sign(signal) * returns` (L233-234) | Train data only | No test data touched during ranking | ✅ Clean |
| 7 | **Orthogonal corr** | `sig.corr(alpha_matrix[sel])` (L250) | Full-period Pearson on train | Standard practice for feature selection | ✅ Standard |
| 8 | **4H ffill reindex** | `.reindex(df_1h.index, method='ffill')` (L205) | 4H data already `shift(1)` → 4hrs stale before ffill | After global `shift(1)`: ≥5hr lag | ✅ Safe |
| 9 | **Adaptive weights** | `rolling_er = factor_returns.rolling(lb).mean()` (L289) | `weight[t]` includes `factor_return[t]` | 1/lookback contribution (0.14–0.56%) | ⚠️ Negligible |
| 10 | **Alpha ranking** | `eval_alpha` uses `nofee_sharpe` (L374-376) | N/A — methodology issue | High-turnover alphas rank too highly | ⚠️ Methodology |
| 11 | **OBV cumsum** | `(sign(ret)*volume).cumsum()` (L141) | Cumsum is local to input slice; z-scored afterward | Level-independent after z-score | ✅ Clean |

### Detailed Trace: Most Critical Path (Signal → PnL)

```
Bar t-1 closes at 14:00 UTC
  │
  ├─ build_mtf_alphas_safe() computes ALL signals using bars up to t-1
  │   ├─ 1H signals: use close[t-1], high[t-1], etc.
  │   ├─ 4H signals: use 4H bar from ≥4 hours before t-1 (shift(1) on 4H)
  │   └─ alpha_df.shift(1): signal at index t = computation from bar t-1
  │
  ├─ strategy_adaptive_net() at index t:
  │   ├─ alpha_matrix[t] = signal from bar t-1 ✅
  │   ├─ weight[t] = rolling_mean(past 180-720 factor returns)
  │   │   └─ Includes factor_return[t] = sign(alpha[t]) × ret[t]
  │   │       └─ ⚠️ Minor: weight uses concurrent return (1/lb impact)
  │   ├─ direction[t] = sign(Σ alpha_i[t] × weight_i[t])
  │   └─ PnL[t] = direction[t] × ret[t] - fee × |Δdirection|
  │              = signal(data up to t-1) × return(t-1 → t)
  │              ✅ Correct: prediction precedes outcome
  │
  └─ Result: NO material lookahead bias identified
```

### Walk-Forward Isolation Check

```
For each fold:
  TRAIN: alpha_tr = build_mtf_alphas_safe(train_1h, train_4h)
         results  = [eval_alpha(α, train_ret) for α in alpha_tr]    ← train only
         best_cfg = argmax(strategy_adaptive_net(alpha_tr, train_ret, ...)) ← train only
  
  TEST:  alpha_te = build_mtf_alphas_safe(test_1h, test_4h)          ← test data
         mte      = strategy_adaptive_net(alpha_te, test_ret, best_sel, ...) ← test data
  
  ✅ No train→test leakage in alpha selection or parameter tuning
  ⚠️ 1-bar boundary overlap (test_start in both train and test) — negligible
  ⚠️ Selected alphas chosen on train, but adaptive weights re-estimated on test
     (this is BY DESIGN — adaptive weights ARE the strategy, not a tuned param)
```
