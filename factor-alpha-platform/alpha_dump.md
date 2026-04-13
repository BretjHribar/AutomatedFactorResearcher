### Alpha #1
- **Name**: df_min(df_max(add(add(zscore_cs(sma(taker_buy_ratio, 72)), zscore_cs(sma(vwap_de
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:04:24
- **Notes**: Microstructure composite: aggressive buyer ratio + VWAP positioning + trade fragmentation momentum. All three signals update meaningfully at 5m. Captures orderflow imbalances.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(taker_buy_ratio, 72)), zscore_cs(sma(vwap_deviation, 72))), zscore_cs(sma(ts_delta(trades_per_volume, 12), 72))), -1.5), 1.5)`

---

### Alpha #2
- **Name**: sma(lower_shadow, 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:13:12
- **Notes**: Buyer defense signal: coins with consistent lower shadows show buying pressure at lows, indicating demand support at 5m frequency
- **Expression**: `sma(lower_shadow, 144)`

---

### Alpha #3
- **Name**: sma(ts_rank(taker_buy_ratio, 288), 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:14:39
- **Notes**: Time-series rank of buyer aggression over 1 day, smoothed 12h. Captures cross-sectional relative positioning of orderflow regime.
- **Expression**: `sma(ts_rank(taker_buy_ratio, 288), 144)`

---

### Alpha #4
- **Name**: Decay_lin(taker_buy_ratio, 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:16:44
- **Notes**: Linear-decay weighted taker buy ratio over 12h. More recent orderflow weighed more heavily, capturing momentum in aggressive buying.
- **Expression**: `Decay_lin(taker_buy_ratio, 144)`

---

### Alpha #5
- **Name**: sma(ts_skewness(log_returns, 144), 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:18:19
- **Notes**: Return distribution skewness: coins with positively skewed 12h returns continue to outperform. Captures asymmetric upside momentum at 5m.
- **Expression**: `sma(ts_skewness(log_returns, 144), 144)`

---

### Alpha #6
- **Name**: sma(ts_zscore(taker_buy_ratio, 72), 288)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:19:09
- **Notes**: Z-scored taker buy ratio captures how unusual current orderflow aggression is relative to recent 6h history, smoothed 1d. Captures sudden demand surges.
- **Expression**: `sma(ts_zscore(taker_buy_ratio, 72), 288)`

---

### Alpha #8
- **Name**: sma(volume_momentum_5_20, 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 03:23:01
- **Notes**: Short vs long volume ratio: coins where recent 5-bar volume exceeds 20-bar average are seeing attention/interest surges that persist at 5m frequency.
- **Expression**: `sma(volume_momentum_5_20, 144)`

---

### Alpha #9
- **Name**: sma(ts_zscore(beta_to_btc, 288), 144)
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-28 19:05:57
- **Notes**: Beta anomaly: high-beta alts are overpriced. Smoothed z-score of BTC beta captures mean-reversion in beta loadings.
- **Expression**: `sma(ts_zscore(beta_to_btc, 288), 144)`

---

### Alpha #11
- **Name**: df_min(df_max(add(add(zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_ra
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:38:26
- **Notes**: Candle buyer defense + volume flow + daily momentum composite. Lower shadow ratio captures buyer absorption at lows, quote volume delta captures institutional interest spikes, 1d return z-score captures daily momentum regime. Mixed timeframes (288/144/288) for multi-scale balance.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 288)), zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 12), 144))), zscore_cs(sma(ts_zscore(log_returns, 288), 288))), -1.5), 1.5)`

---

### Alpha #12
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_corr(log_returns, taker_buy_ratio, 72), 1
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:39:18
- **Notes**: Cross-field correlation composite. Return-orderflow alignment quality captures when taker buying actually drives prices. Normalized bar position identifies accumulation vs distribution. ADV60 delta captures institutional flow shifts. Orthogonal to price-based signals.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_corr(log_returns, taker_buy_ratio, 72), 144)), zscore_cs(sma(ts_zscore(close_position_in_range, 72), 144))), zscore_cs(sma(ts_delta(s_log_1p(adv60), 36), 288))), -1.5), 1.5)`

---

### Alpha #13
- **Name**: df_min(df_max(add(add(zscore_cs(negative(sma(parkinson_volatility_10, 288))), zs
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:40:05
- **Notes**: Volatility compression + accumulation composite. Low Parkinson vol signals quiet accumulation. High close_position_in_range means buying into the close. Volume momentum 5/20 ratio identifies volume regime shifts. Orthogonal to orderflow and momentum signals.
- **Expression**: `df_min(df_max(add(add(zscore_cs(negative(sma(parkinson_volatility_10, 288))), zscore_cs(sma(close_position_in_range, 144))), zscore_cs(sma(volume_momentum_5_20, 288))), -1.5), 1.5)`

---

### Alpha #14
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:43:06
- **Notes**: Momentum regime quality composite. Return autocorrelation identifies trending vs mean-reverting regimes. Return skewness captures lottery demand/distribution asymmetry. Vol-return regression slope measures risk-return tradeoff efficiency. All second-order statistics orthogonal to direct price/flow signals.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 72), 144)), zscore_cs(sma(ts_skewness(log_returns, 72), 144))), zscore_cs(sma(ts_regression(log_returns, historical_volatility_20, 72, 0, 2), 144))), -1.5), 1.5)`

---

### Alpha #15
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_corr(close_position_in_range, taker_buy_r
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:46:47
- **Notes**: Microstructure confirmation composite. Close-position vs taker-buy correlation captures when aggressive buying coincides with strong closes. Decay-weighted VWAP deviation gives recency-biased fair value drift. Open-close range (body size) measures conviction. Different signal family from prior alphas.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_corr(close_position_in_range, taker_buy_ratio, 72), 144)), zscore_cs(Decay_lin(vwap_deviation, 72))), zscore_cs(sma(open_close_range, 288))), -1.5), 1.5)`

---

### Alpha #16
- **Name**: df_min(df_max(add(add(zscore_cs(negative(sma(true_divide(ArgMin(close, 288), 288
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:50:14
- **Notes**: Oversold timing + flow confirmation composite. ArgMin recency signals how recently the low was made — recent lows signal oversold conditions with rebound potential. Taker buy ratio z-score confirms that aggressive buying is returning. Lower shadow ratio shows buyer absorption at lows. All three signal oversold-recovery dynamics.
- **Expression**: `df_min(df_max(add(add(zscore_cs(negative(sma(true_divide(ArgMin(close, 288), 288), 144))), zscore_cs(sma(ts_zscore(taker_buy_ratio, 144), 288))), zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 144))), -1.5), 1.5)`

---

### Alpha #17
- **Name**: df_min(df_max(add(add(zscore_cs(sma(multiply(hump(volume_momentum_1, 1.5), Sign(
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:54:13
- **Notes**: Volume-gated momentum + support detection + flow decay composite. Hump-gated volume momentum captures high-conviction directional moves (vol surge with positive return). ts_min of close identifies support level recency. Decay-weighted taker buy ratio captures recent buyer intent with temporal decay. Novel use of hump operator for conditional signaling.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(multiply(hump(volume_momentum_1, 1.5), Sign(log_returns)), 144)), zscore_cs(sma(ts_min(close, 144), 288))), zscore_cs(sma(Decay_lin(taker_buy_ratio, 72), 288))), -1.5), 1.5)`

---

### Alpha #18
- **Name**: df_min(df_max(add(add(add(zscore_cs(sma(taker_buy_ratio, 144)), zscore_cs(sma(cl
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:56:15
- **Notes**: 4-signal multi-timeframe ensemble. Taker buy ratio (12h SMA) captures sustained buyer demand. Close position in range (1d SMA) identifies accumulation on closes near highs. Volume momentum (12h SMA) detects attention surges. Return skewness (2d SMA of 12h window) captures lottery demand asymmetry. Each signal uses different timeframe for multi-scale coverage. Very low kurtosis (2.8) and excellent sub-period stability.
- **Expression**: `df_min(df_max(add(add(add(zscore_cs(sma(taker_buy_ratio, 144)), zscore_cs(sma(close_position_in_range, 288))), zscore_cs(sma(volume_momentum_1, 144))), zscore_cs(sma(ts_skewness(log_returns, 144), 288))), -1.5), 1.5)`

---

### Alpha #19
- **Name**: df_min(df_max(add(add(zscore_cs(sma(multiply(close_position_in_range, s_log_1p(v
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 01:58:35
- **Notes**: Volume-weighted microstructure composite. Close position weighted by log-volume captures where heavy volume closes happen in the range (accumulation at highs = bullish). Trade count z-score detects unusual trading activity relative to 12h history. Signed high-low range captures directional conviction — wide range + positive return = strong trend bar. Novel multiplicative interaction terms.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(multiply(close_position_in_range, s_log_1p(volume)), 144)), zscore_cs(sma(ts_zscore(s_log_1p(trades_count), 144), 288))), zscore_cs(sma(multiply(Sign(log_returns), high_low_range), 288))), -1.5), 1.5)`

---

### Alpha #20
- **Name**: df_min(df_max(add(add(zscore_cs(negative(sma(historical_volatility_120, 576))), 
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 02:01:11
- **Notes**: Low-vol anomaly + persistent buying + signed volume surge composite. Low historical_volatility_120 (via 2-day SMA) captures the well-known low-vol anomaly: quiet instruments outperform. Decay_lin taker_buy_ratio gives persistent/recency-weighted orderflow. Signed volume z-score captures high-volume directional conviction bars. Three distinct economic mechanisms for diversification.
- **Expression**: `df_min(df_max(add(add(zscore_cs(negative(sma(historical_volatility_120, 576))), zscore_cs(sma(Decay_lin(taker_buy_ratio, 144), 288))), zscore_cs(sma(multiply(Sign(log_returns), ts_zscore(s_log_1p(volume), 72)), 288))), -1.5), 1.5)`

---

### Alpha #21
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_rank(s_log_1p(quote_volume), 288), 288)),
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 02:02:56
- **Notes**: Institutional attention + candle positioning + momentum regime composite. Quote volume rank (1d rank smoothed 1d) captures cross-sectional attention — top volume names tend to continue. Close_position_in_range centered at 0.5 captures accumulation/distribution bias. Return autocorrelation (12h window, 1d smooth) identifies trending vs mean-reverting regime. Three distinct economic mechanisms with long smoothing for stability.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_rank(s_log_1p(quote_volume), 288), 288)), zscore_cs(sma(subtract(close_position_in_range, 0.5), 288))), zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 144), 288))), -1.5), 1.5)`

---

### Alpha #22
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_rank(trades_per_volume, 144), 288)), zsco
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 02:04:37
- **Notes**: Trade fragmentation + declining vol + candle conviction composite. High trades_per_volume rank signals retail/small-batch accumulation interest. Declining vol (negative delta of HV20 smoothed 1d) captures the vol compression that precedes breakouts. Open-close range (body size) measures conviction. Three signals capturing the 'accumulation before breakout' pattern.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_rank(trades_per_volume, 144), 288)), zscore_cs(negative(sma(ts_delta(historical_volatility_20, 72), 288)))), zscore_cs(sma(open_close_range, 144))), -1.5), 1.5)`

---

### Alpha #23
- **Name**: df_min(df_max(add(add(zscore_cs(sma(true_divide(ArgMax(volume, 144), 144), 288))
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 02:06:21
- **Notes**: Volume peak recency + buyer defense trend + multi-day flow composite. ArgMax(volume, 144)/144 signals when peak volume occurred — recent peak = active interest. Lower shadow delta captures increasing buyer defense at lows. Taker buy ratio z-score over 1d with 1d smoothing captures sustained flow regime. Ultra-low kurtosis (1.9) suggests robust, well-diversified signal.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(true_divide(ArgMax(volume, 144), 144), 288)), zscore_cs(sma(ts_delta(lower_shadow, 72), 288))), zscore_cs(sma(ts_zscore(taker_buy_ratio, 288), 288))), -1.5), 1.5)`

---

### Alpha #24
- **Name**: df_min(df_max(add(add(zscore_cs(sma(ts_delta(beta_to_btc, 288), 288)), zscore_cs
- **Category**: 
- **Source**: agent1_research
- **Created**: 2026-03-29 02:08:05
- **Notes**: Dynamic beta + wick asymmetry + intraday momentum composite. Rising beta to BTC signals increasing market sensitivity. Lower shadow minus upper shadow captures net buyer vs seller rejection asymmetry. Return z-score (6h lookback, 1d SMA) captures intraday momentum regime. Beta change is a unique signal not used by other alphas.
- **Expression**: `df_min(df_max(add(add(zscore_cs(sma(ts_delta(beta_to_btc, 288), 288)), zscore_cs(sma(subtract(lower_shadow, upper_shadow), 288))), zscore_cs(sma(ts_zscore(log_returns, 72), 288))), -1.5), 1.5)`

---

