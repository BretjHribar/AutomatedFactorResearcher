# KuCoin 4H Alpha Templates

Proven alpha expression patterns adapted for KuCoin's available data fields.
KuCoin does NOT have: `funding_rate`, `taker_buy_ratio`, `trades_count`, `trades_per_volume`.

**Default universe**: `KUCOIN_TOP100` (larger N improves cross-sectional breadth for multiplicative signals).

## Key Differences from Binance Templates
- No funding carry signals → focus on volume flow and volatility-adjusted momentum
- No order flow (taker buy) → substitute with `volume_ratio_20d` and `volume_momentum_5_20`
- No trade count → use candle structure as microstructure proxy

## Template A: Beta-Anomaly Momentum (Strongest Family)

Core pattern: `Decay_exp(multiply(ts_delta(close, W), BETA_SIGNAL), RATE)`

```
# Short-term beta deviation × price momentum
Decay_exp(multiply(ts_delta(close, 12), subtract(beta_to_btc, sma(beta_to_btc, 60))), 0.05)

# Longer window variants
Decay_exp(multiply(ts_delta(close, 24), subtract(beta_to_btc, sma(beta_to_btc, 120))), 0.1)
sma(multiply(ts_delta(close, 6), ts_delta(beta_to_btc, 12)), 3)
```

## Template B: Sign × Volume-Adjusted Momentum

Core pattern: `multiply(Sign(MOMENTUM), VOLUME_RATIO / VOLATILITY)`

```
# Momentum direction × volume surprise / volatility
multiply(Sign(momentum_60d), true_divide(volume_momentum_5_20, df_max(parkinson_volatility_60, 0.0001)))

# Trend confirmation with volume
sma(multiply(Sign(sma(returns, 120)), true_divide(volume_ratio_20d, df_max(historical_volatility_20, 0.0001))), 12)
```

## Template C: CS Composite (Additive)

Core pattern: `df_min(df_max(add(ATOM1, add(ATOM2, ATOM3)), -1.5), 1.5)`

### Strong KuCoin-compatible atoms:
```
zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))
zscore_cs(sma(vwap_deviation, 120))
zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))
zscore_cs(sma(ts_zscore(log_returns, 120), 120))
zscore_cs(sma(ts_skewness(log_returns, 60), 60))
zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 30), 60))
zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 30), 120))
zscore_cs(sma(volume_momentum_5_20, 60))
zscore_cs(sma(ts_kurtosis(log_returns, 60), 60))
zscore_cs(sma(ts_corr(high_low_range, volume_ratio_20d, 60), 60))
zscore_cs(sma(ts_delta(parkinson_volatility_20, 30), 60))
zscore_cs(sma(ts_corr(close_position_in_range, volume_ratio_20d, 60), 60))
zscore_cs(sma(true_divide(open_close_range, df_max(high_low_range, 0.001)), 120))
```

### Example 3-component composite:
```
df_min(df_max(add(add(zscore_cs(sma(vwap_deviation, 120)), zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))), zscore_cs(sma(ts_skewness(log_returns, 60), 60))), -1.5), 1.5)
```

## Template D: Candle Structure (Microstructure Proxy)

KuCoin lacks trade-level and wick data, so use candle geometry with available fields:

```
# Body ratio momentum: large open→close moves signal conviction
sma(multiply(ts_delta(close, 6), true_divide(open_close_range, df_max(high_low_range, 0.0001))), 2)

# Close-in-range momentum: bars closing near top = bullish conviction
Decay_exp(multiply(ts_delta(close, 12), subtract(close_position_in_range, sma(close_position_in_range, 60))), 0.05)

# Volatility compression breakout
Decay_exp(multiply(ts_delta(close, 12), subtract(historical_volatility_20, sma(historical_volatility_20, 60))), 0.05)

# Range position momentum
zscore_cs(sma(ts_corr(close_position_in_range, volume_ratio_20d, 60), 60))
```

## Template E: Volatility-Adjusted Return Spread

```
# Returns normalized by vol — Sharpe-like signal
true_divide(ts_sum(log_returns, 60), df_max(historical_volatility_60, 0.0001))

# Vol of vol as regime indicator
zscore_cs(sma(ts_delta(parkinson_volatility_20, 30), 60))
```

## Lookback Windows (4h bars)
| Bars | Calendar | 
|------|----------|
| 6 | 1 day |
| 12 | 2 days |
| 30 | 5 days |
| 60 | 10 days |
| 120 | 20 days |
| 240 | 40 days |
| 360 | 60 days |

## Template F: Multiplicative Combos (Amplified Signal)

Multiplying two z-scored atoms creates a **non-linear signal** that is strong only when BOTH atoms agree directionally. This can push Sharpe above the additive ceiling.

Core pattern: `zscore_cs(multiply(ATOM1, ATOM2))`

The atoms should be pre-smoothed so the product is stable:

```
# VWAP position × vol-adjusted momentum (agree on direction → amplify)
zscore_cs(multiply(sma(vwap_deviation, 120), true_divide(ts_sum(log_returns, 120), df_max(historical_volatility_120, 0.0001))))

# Body Ratio Modulation (Dominant Pattern in SOTA Alphas)
# This pattern uses a decayed body ratio as a probability filter for a modulating signal (Z-score).
zscore_cs(multiply(ts_rank(ts_zscore(close_position_in_range, 360), 240), ts_rank(Decay_exp(true_divide(open_close_range, df_max(high_low_range, 0.001)), 0.02), 240)))

# Volumetric Body Ratio (High Stability Discovery)
zscore_cs(multiply(ts_rank(sma(ts_delta(s_log_1p(adv20), 30), 120), 240), ts_rank(Decay_exp(true_divide(open_close_range, df_max(high_low_range, 0.001)), 0.05), 240)))

# Body ratio × volume momentum (intrabar conviction × volume surge)
zscore_cs(multiply(sma(true_divide(open_close_range, df_max(high_low_range, 0.001)), 120), sma(volume_momentum_5_20, 60)))

# Range-vol correlation × vol-adj momentum
zscore_cs(multiply(sma(ts_corr(high_low_range, volume_ratio_20d, 60), 60), true_divide(ts_sum(log_returns, 120), df_max(historical_volatility_120, 0.0001))))

# Skewness × VWAP (skewed up AND above VWAP = double bullish)
zscore_cs(multiply(sma(ts_skewness(log_returns, 60), 60), sma(vwap_deviation, 120)))

# Vol-adj momentum × volume quote growth (momentum confirmed by capital inflow)
zscore_cs(multiply(true_divide(ts_sum(log_returns, 120), df_max(historical_volatility_120, 0.0001)), sma(ts_delta(s_log_1p(quote_volume), 30), 120)))
```

### Hybrid: Add best additive composite + multiply two atoms
```
# Additive base + multiplicative booster
df_min(df_max(add(add(zscore_cs(sma(vwap_deviation, 120)), zscore_cs(true_divide(ts_sum(log_returns, 120), df_max(historical_volatility_120, 0.0001)))), zscore_cs(multiply(sma(true_divide(open_close_range, df_max(high_low_range, 0.001)), 120), sma(volume_momentum_5_20, 60)))), -1.5), 1.5)
```

> [!TIP]
> Multiplication is most powerful when atoms are **pre-smoothed** (SMA or Decay_exp) so the product doesn't spike. Always wrap the product in `zscore_cs()` before using in a composite.
