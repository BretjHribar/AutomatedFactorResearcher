# 5m Alpha Construction Templates & Signal Profiles

This file documents signal construction patterns optimized for **5-minute crypto perpetual futures** on the BINANCE_TOP100 universe. **Read this before hypothesizing.** Update it whenever a new passing alpha or key insight is found.

> **Key difference from 4h**: At 5m frequency (288 bars/day), microstructure and orderflow signals dominate. Macro momentum and funding-rate signals are less useful because they update too slowly relative to the bar frequency.

---

## Lookback Window Reference (5m bars)

| Bars | Real Time | Use Case |
|---|---|---|
| 12 | 1 hour | Ultra-short microstructure |
| 36 | 3 hours | Intraday momentum |
| 72 | 6 hours | Half-day patterns |
| 144 | 12 hours | Session-level signals |
| 288 | 1 day | Daily patterns |
| 576 | 2 days | Multi-day smoothing |
| 1440 | 5 days | Weekly patterns |

> ⚠️ **Max practical lookback: 1440 bars (5 days)**. The train period is only ~18 days, so lookbacks above 1440 eat too much warmup. Keep inner lookbacks ≤ 576 and outer smoothing (sma) ≤ 1440.

---

## Template 1 — Additive Composite (Standard)

The most reliable structure:

```python
df_min(df_max(
  add(
    zscore_cs(sub_signal_A),
    add(
      zscore_cs(sub_signal_B),
      zscore_cs(sub_signal_C)
    )
  ),
  -1.5), 1.5)
```

**Rules:**
1. Each `sub_signal` uses SMA smoothing appropriate for 5m: `sma(raw_signal, 72)` to `sma(raw_signal, 576)`
2. Wrap each in `zscore_cs()` or `rank()` before adding
3. Clip with `df_min(df_max(..., -1.5), 1.5)` to reduce kurtosis
4. At 5m, signals can be noisier — use **wider SMA** to stabilize

---

## Signal Families for 5m

### Tier 1 — High-Frequency Microstructure (likely strongest at 5m)

| Expression | Description | Notes |
|---|---|---|
| `sma(taker_buy_ratio, 72)` | Aggressive buyer demand (6h smooth) | Primary orderflow signal |
| `sma(vwap_deviation, 72)` | Close vs VWAP deviation | Mean-reverts faster at 5m |
| `sma(ts_delta(trades_per_volume, 12), 72)` | Trade fragmentation momentum | Retail activity spikes |
| `sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 288)` | Buyer defense (candle wicking) | Lower shadow ratio, smoothed 1d |
| `sma(close_position_in_range, 144)` | Where close falls in H-L range | Bar positioning signal |
| `sma(ts_zscore(close_position_in_range, 72), 144)` | Normalized bar position | Z-scored version for stability |

### Tier 2 — Volume & Flow Dynamics

| Expression | Description | Notes |
|---|---|---|
| `sma(ts_delta(s_log_1p(adv60), 36), 288)` | Dollar volume flow momentum | Institutional interest |
| `sma(ts_delta(s_log_1p(quote_volume), 12), 144)` | USD flow momentum (3h delta, 12h sma) | |
| `sma(volume_momentum_1, 72)` | Bar-over-bar volume ratio | Volume spikes |
| `sma(volume_momentum_5_20, 288)` | Short/long volume ratio | Volume regime |
| `sma(ts_delta(volume_ratio_20d, 36), 144)` | Volume ratio changes | Attention shifts |

### Tier 3 — Short-Term Momentum & Volatility

| Expression | Description | Notes |
|---|---|---|
| `sma(ts_zscore(log_returns, 288), 288)` | 1-day return z-score, smoothed 1d | Daily momentum |
| `sma(ts_zscore(log_returns, 72), 144)` | 6h return z-score, smoothed 12h | Intraday momentum |
| `negative(sma(parkinson_volatility_10, 288))` | Low-vol accumulation | Quiet = accumulation |
| `sma(ts_skewness(log_returns, 72), 144)` | Return distribution shape | Lottery demand |
| `sma(open_close_range, 288)` | Candle body size | Conviction indicator |

### Tier 4 — Cross-Field Correlations (ts_corr)

| Expression | Description | Notes |
|---|---|---|
| `sma(ts_corr(log_returns, taker_buy_ratio, 72), 144)` | Return-flow alignment | When buying → returns |
| `sma(ts_corr(close_position_in_range, taker_buy_ratio, 72), 144)` | Microstructure confirmation | Buyers closing high |
| `sma(ts_corr(log_returns, trades_per_volume, 72), 144)` | Retail momentum quality | Small trades → returns |
| `sma(ts_corr(log_returns, delay(log_returns, 1), 72), 144)` | Return autocorrelation | Momentum regime |

### Tier 5 — Regression Slopes (ts_regression)

| Expression | Description | Notes |
|---|---|---|
| `sma(ts_regression(log_returns, historical_volatility_20, 72, 0, 2), 144)` | Vol-return slope | Risk-return tradeoff |
| `sma(ts_regression(log_returns, taker_buy_ratio, 72, 0, 2), 144)` | Flow impact coefficient | Market impact |

### ⚠️ Signals to AVOID at 5m

| Signal | Why |
|---|---|
| `funding_rate*` | Updates every 8 hours — stale at 5m. Forward-filled data is flat. |
| `overnight_gap` | Meaningless in 24/7 crypto |
| `momentum_60d` | Way too slow for 5m bars (60 day = 17,280 bars) |
| `beta_to_btc` (long window) | 60-bar beta at 5m = only 5 hours. Use 288+ bars. |

---

## Starter Hypotheses (Try These First)

### Hypothesis 1 — Orderflow + Microstructure
```python
df_min(df_max(add(add(
  zscore_cs(sma(taker_buy_ratio, 72)),
  zscore_cs(sma(vwap_deviation, 72))),
  zscore_cs(sma(ts_delta(trades_per_volume, 12), 72))
), -1.5), 1.5)
```
**Why**: Three microstructure signals capturing aggressive buying, VWAP positioning, and trade fragmentation. All update meaningfully at 5m.

### Hypothesis 2 — Candle + Volume + Momentum
```python
df_min(df_max(add(add(
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 288)),
  zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 12), 144))),
  zscore_cs(sma(ts_zscore(log_returns, 288), 288))
), -1.5), 1.5)
```
**Why**: Buyer defense (lower shadow), volume flow momentum, and daily return z-score. Mix of timeframes for regime balance.

### Hypothesis 3 — Cross-Field Correlation Composite
```python
df_min(df_max(add(add(
  zscore_cs(sma(ts_corr(log_returns, taker_buy_ratio, 72), 144)),
  zscore_cs(sma(ts_zscore(close_position_in_range, 72), 144))),
  zscore_cs(sma(ts_delta(s_log_1p(adv60), 36), 288))
), -1.5), 1.5)
```
**Why**: Flow-return alignment quality, normalized bar position, and institutional flow momentum.

---

## Debugging Common Gate Failures (5m-specific)

| Gate Failing | Typical Cause | Fix |
|---|---|---|
| **IS Sharpe < 1.0** | Signal too weak | Combine 3-4 signals; increase SMA smoothing |
| **Mean IC < -0.02** | Signal doesn't predict returns cross-sectionally | Try different data fields; check if signal is actually inverted (try `negative()`) |
| **Min sub-period Sharpe < 0.5** | One 9-day half is negative | Add regime-balanced signals; try different lookback mix |
| **Rolling SR std > 0.10** | Inconsistent daily Sharpe | More smoothing (longer SMA); more signal components |
| **PnL Kurtosis > 30** | Extreme outlier days | Tighter clip (±1.0); use zscore_cs not raw |

---

## Key Principles for 5m

1. **SMA smoothing is MANDATORY** at 5m. Raw signals are too noisy. Minimum `sma(signal, 72)` (6 hours).
2. **Volume/flow signals dominate** at 5m. Price momentum works best when combined with orderflow confirmation.
3. **Cross-sectional IC over 100 symbols** gives good signal — even with 18 days of data.
4. **Turnover ≤ 0.05 is a HARD GATE** — signals are auto-rejected above this even at zero fees. Use long SMA windows (≥ 576 bars = 2 days) to keep signals slow-moving.
5. **Lookbacks: 12, 36, 72, 144, 288, 576, 1440**. No odd numbers.
6. **The clip `df_min(df_max(..., -1.5), 1.5)` matters even more** at 5m because microstructure signals have fat tails.
7. **`s_log_1p()` for volume fields** — essential to compress outliers from massive volume spikes.
