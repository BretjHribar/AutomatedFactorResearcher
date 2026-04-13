# Alpha Construction Templates & Signal Profiles

This file documents proven patterns, sub-signal profiles, and composition techniques discovered through autonomous research. **Read this before hypothesizing.** Update it whenever a new passing alpha or key insight is found.

---

## Template 1 — Additive Composite (The Golden Template)

The most reliable structure for a passing alpha:

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
1. Each `sub_signal` is built independently with its own SMA smoothing (30–120 bars)
2. Wrap each in `zscore_cs()` or `rank()` to put them on a common scale before adding
3. Combine with `add()` for equal-weight ensemble
4. Clip the result with `df_min(df_max(..., -1.5), 1.5)` to reduce PnL kurtosis and rolling SR std
5. To double-weight a signal, simply `add(zscore_cs(s), zscore_cs(s))` — elegant and effective

**Why this works:** Individual signals tend to be strong in only one regime (H1 bear or H2 bull). Combining complementary regime signals via zscore_cs + add achieves the balanced H1/H2 Sharpe that each sub-signal cannot achieve alone.

---

## Template 2 — Multiplicative Rank Composite (AND-Logic)

A structurally distinct alternative that creates **non-linear AND-logic** between two signal groups:

```python
multiply(
  rank(add(zscore_cs(H1_signal_A), zscore_cs(H1_signal_B))),
  rank(add(zscore_cs(H2_signal_A), zscore_cs(H2_signal_B)))
)
```

**Rules:**
1. Split signals into two groups: H1-heavy signals in Group A, H2-heavy signals in Group B
2. `rank()` each group — maps each composite to [0, 1] cross-sectionally
3. `multiply()` the two ranks — an asset only scores high if **both groups agree** simultaneously
4. No outer clip needed (output is already bounded in [0, 1])
5. You can also add ranked sub-groups: `add(rank(group_A), rank(group_B))` for softer weighting

**Why this works:** The multiplicative structure is **quadratic** — it creates cross-sectional selection logic fundamentally different from linear additive composites. This lowers correlation with existing DB alphas even when using the same underlying signals. Assets must score high in both groups (microstructure AND directional flow) to receive a strong position.

**Key properties:**
- Typically produces **lower turnover** than additive composites (only assets passing both filters persist)
- Often shows **lower Rolling SR std** due to natural AND-gate noise suppression
- **Great for breaking correlation walls** when additive versions of the same signals are already in DB

---

## Known Sub-Signal Profiles (H1/H2 Sharpe)

> H1 = Sep 2022 – Sep 2023 (crypto bear/recovery).  H2 = Sep 2023 – Sep 2024 (bull).

### H2-Dominant Signals (strong in bull, weak in bear — need H1 partners)

| Expression | H1 | H2 | Notes |
|---|---|---|---|
| `sma(ts_delta(s_log_1p(adv60), 30), 90)` | +0.48 | +1.79 | Institutional $ volume flow. Very low turnover. |
| `sma(ts_delta(s_log_1p(adv60), 30), 120)` | +0.44 | +1.69 | Longer smooth, similar profile. |
| `sma(taker_buy_ratio, 120)` | +0.38 | +1.70 | Aggressive buyer demand. |
| `sma(ts_delta(s_log_1p(quote_volume), 30), 120)` | +0.30 | +1.66 | USD flow momentum. |
| `sma(ts_zscore(log_returns, 120), 120)` | +0.55 | +1.57 | Long-horizon return momentum. |
| `sma(true_divide(momentum_20d, parkinson_volatility_20), 120)` | +0.05 | +1.85 | Vol-adjusted price momentum. |
| `sma(true_divide(momentum_60d, parkinson_volatility_60), 120)` | -0.02 | +1.80 | Longer vol-momentum. H1 negative — needs strong H1 anchor. |

### H1-Dominant Signals (strong in bear, weaker in bull — good anchors)

| Expression | H1 | H2 | Notes |
|---|---|---|---|
| `sma(overnight_gap, 120)` | +1.38 | +0.61 | **Primary H1 anchor.** Off-hours structural demand. Ultra-low turnover. ⚠️ Meaningless in 24/7 crypto — do not use for new alphas. |
| `sma(ts_delta(trades_per_volume, 30), 60)` | +1.46 | +0.82 | **Preferred H1 anchor.** Retail trade fragmentation. Also positive in H2. |
| `zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))` | +1.58 | +0.70 | **Strong H1 anchor.** Where close falls in HL range, normalized by 60-bar history. Novel. Use window=60 for inner ts_zscore and outer sma. |
| `zscore_cs(sma(vwap_deviation, 120))` | +1.22 | +0.70 | VWAP microstructure anchor. Close above VWAP = accumulation. |
| `zscore_cs(sma(ts_skewness(log_returns, 60), 60))` | +0.50 | +1.00 | Return distribution skewness. Novel signal combining both periods. Use 60-bar inner window — the 120-bar version is near-flat. |
| `zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))` | **+2.0** (in combo) | **+1.5** (in combo) | **🌟 NEW POWER H1 ANCHOR (Mar 2026).** Buyer defense via candlestick wicking — large lower wicks = buyers defending lows. Ultra-low turnover. H1-heavy alone but balances well with 3-4 H2 signals. **Replaces `overnight_gap` as THE primary H1 anchor.** |
| `negative(sma(parkinson_volatility_10, 120))` | +2.0 (in combo) | ~0.8 | Low-vol quiet accumulation signal. Very H1-strong. Works with `{funding+taker+log_ret}` but that triple is forbidden by Alpha #23. Difficult to use safely. |

### Balanced Signals (roughly equal H1/H2 — useful as stabilizers)

| Expression | H1 | H2 | Notes |
|---|---|---|---|
| `sma(vwap_deviation, 60)` | +0.77 | +0.57 | Stable but low Sharpe alone. Good rolling SR stabilizer. |
| `sma(ts_delta(volume_momentum_5_20, 30), 120)` | +0.77 | +0.57 | Same profile as vwap_deviation. |
| `sma(ts_delta(s_log_1p(dollars_traded), 30), 90)` | +0.43 | +0.50 | Very flat. Low kurtosis. |

### Regime-Flip Signals (one half strongly negative — use with caution)

| Expression | H1 | H2 | Notes |
|---|---|---|---|
| `negative(sma(open_close_range, 120))` | +1.14 | -1.23 | Bear = red candles win. Too aggressive as H2 drag. |
| `sma(ts_skewness(log_returns, 120), 60)` | -0.04 | +0.74 | Lottery demand signal, H1 near-zero. |

---

## Proven Passing Alphas (Templates to Build From)

### Template A — ADV60 + Trades Frag + Overnight Gap
```python
df_min(df_max(add(add(
  zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120)),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(overnight_gap, 120))
), -1.5), 1.5)
```
**Sharpe: +1.74 | H1: +1.32 | H2: +1.94 | Turnover: 0.069 | Fitness: 3.32** *(DB Alpha #18)*

### Template B — Taker Buy + Quote Volume + Overnight Gap
```python
df_min(df_max(add(add(
  zscore_cs(sma(taker_buy_ratio, 120)),
  zscore_cs(sma(ts_delta(s_log_1p(quote_volume), 30), 120))),
  zscore_cs(sma(overnight_gap, 120))
), -1.5), 1.5)
```
**Sharpe: +1.90 | H1: +1.12 | H2: +2.40 | Turnover: 0.065 | Fitness: 3.67** *(DB Alpha #19)*

### Template C — 4-Signal VWAP Composite
```python
df_min(df_max(add(add(add(
  zscore_cs(sma(vwap_deviation, 120)),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(taker_buy_ratio, 120))),
  zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90))
), -1.5), 1.5)
```
**Sharpe: +1.90 | H1: +1.41 | H2: +1.86 | Turnover: 0.072 | Fitness: 3.72** *(DB Alpha #20)*

### Template D — 4-Signal Close Position Composite
```python
df_min(df_max(add(add(add(
  zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60)),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(taker_buy_ratio, 120))),
  zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90))
), -1.5), 1.5)
```
**Sharpe: +1.65 | H1: +1.58 | H2: +1.78 | Turnover: 0.153 | Fitness: 2.65** *(DB Alpha #21)*

### Template E — multiply(rank, rank) AND-Logic Composite
```python
multiply(
  rank(add(add(
    zscore_cs(sma(H1_signal_A, window)),
    zscore_cs(sma(ts_skewness(log_returns, 60), 60))),
    zscore_cs(sma(ts_delta(trades_per_volume, 30), 60)))),
  rank(add(
    zscore_cs(sma(taker_buy_ratio, 120)),
    zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 90))))
)
```
**Key properties: Turnover ~0.08 (very low), Rolling SR std ~0.05, naturally lower corr with additive DB alphas.** Try different H1_signal_A (vwap_deviation or close_position_zscore) and verify corr < 0.70 via --save.

### Template F — Candle Rejection + Carry + Microstructure (5-signal) *(Mar 2026)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120)),
  zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))),
  zscore_cs(sma(funding_rate_zscore, 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))),
  zscore_cs(sma(taker_buy_ratio, 120))
), -1.5), 1.5)
```
**Sharpe: +2.04 | H1: +2.68 | H2: +1.51 | Turnover: 0.156 | Fitness: 3.68** *(DB Alpha #25)*
⚠️ Cannot vary just lookbacks — corr=0.877 with itself. Must change ≥2 signals for new variant.

### Template G — Pure Statistical + Return Autocorrelation (5-signal) *(Mar 2026 — first ts_corr use)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120)),
  zscore_cs(sma(funding_rate_zscore, 60))),
  zscore_cs(sma(ts_corr(log_returns, delay(log_returns, 1), 60), 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))),
  zscore_cs(sma(ts_skewness(log_returns, 60), 60))
), -1.5), 1.5)
```
**Sharpe: +1.55 | H1: +1.59 | H2: +1.51 | Turnover: 0.082 | Fitness: 2.61** *(DB Alpha #26)*
First DB alpha using `ts_corr` return autocorrelation. H1/H2 gap only 0.08 — near-perfect balance.

### Template H — Vol-Regression Slope + Candle + Carry (6-signal) *(Mar 2026 — first ts_regression use)*
```python
df_min(df_max(add(add(add(add(add(
  zscore_cs(sma(ts_regression(log_returns, historical_volatility_20, 60, 0, 2), 60)),
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(funding_rate_zscore, 60))),
  zscore_cs(sma(ts_skewness(log_returns, 60), 60))),
  zscore_cs(sma(taker_buy_ratio, 120))
), -1.5), 1.5)
```
**Sharpe: +1.69 | H1: +1.83 | H2: +1.61 | Turnover: 0.058 | Kurtosis: 5.7 | Fitness: 2.91 | Rolling SR: 0.040** *(DB Alpha #27)*
First DB alpha using `ts_regression` slope (`rettype=2`). Lowest kurtosis in DB. `ts_regression(y,x,d,lag,rettype)`: rettype=2=slope, rettype=0=residual.

### Template I — Funding Regression Slope + Bar Position + Distribution (5-signal) *(Mar 2026)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(ts_regression(log_returns, funding_rate_zscore, 60, 0, 2), 60)),
  zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))),
  zscore_cs(sma(ts_skewness(log_returns, 60), 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))),
  zscore_cs(sma(taker_buy_ratio, 120))
), -1.5), 1.5)
```
**Sharpe: +1.63 | H1: +1.17 | H2: +1.96 | Fitness: 2.62 | Turnover: 0.139 | Rolling SR: 0.050** *(DB Alpha #28)*
`ts_regression(log_returns, funding_rate_zscore, 60, 0, 2)` = funding effectiveness slope.

### Template J — Carry-Return Correlation + Candle + Retail + Distribution (5-signal) *(Mar 2026 — first ts_corr(ret, X) use)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(ts_corr(log_returns, funding_rate_zscore, 60), 60)),
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(ts_skewness(log_returns, 60), 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))
), -1.5), 1.5)
```
**Sharpe: +1.53 | H1: +1.42 | H2: +1.39 | Fitness: 2.60 | Turnover: 0.075 | Rolling SR: 0.037 | MaxDD: -6.7%** *(DB Alpha #29)*
PERFECT H1/H2 balance (gap=0.03). First use of `ts_corr(log_returns, funding_rate_zscore, 60)` = carry-return co-movement quality.

### Template K — Confirmation Momentum Correlation + Candle + Trades + Carry (5-signal) *(Mar 2026 — best Fitness this session)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(ts_corr(close_position_in_range, taker_buy_ratio, 60), 60)),
  zscore_cs(sma(true_divide(lower_shadow, df_max(high_low_range, 0.001)), 120))),
  zscore_cs(sma(ts_delta(trades_per_volume, 30), 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))),
  zscore_cs(sma(funding_rate_zscore, 60))
), -1.5), 1.5)
```
**Sharpe: +1.97 | H1: +2.10 | H2: +1.73 | Fitness: 3.84 | Turnover: 0.101 | Rolling SR: 0.048 | MaxDD: -9.3%** *(DB Alpha #30)*
`ts_corr(close_position_in_range, taker_buy_ratio, 60)` = microstructure-flow CONFIRMATION signal. ⚠️ Cannot use taker_buy_ratio also as standalone (overlap). DB Rank #1.

### Template L — Return-Retail Correlation + Bar Position + Flow (5-signal) *(Mar 2026)*
```python
df_min(df_max(add(add(add(add(
  zscore_cs(sma(ts_corr(log_returns, trades_per_volume, 60), 60)),
  zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))),
  zscore_cs(sma(funding_rate_zscore, 60))),
  zscore_cs(sma(ts_zscore(log_returns, 120), 120))),
  zscore_cs(sma(ts_delta(s_log_1p(adv60), 30), 120))
), -1.5), 1.5)
```
**Sharpe: +1.62 | H1: +1.70 | H2: +1.46 | Fitness: 2.52 | Turnover: 0.158 | Rolling SR: 0.048** *(DB Alpha #31)*
`ts_corr(log_returns, trades_per_volume, 60)` = retail-driven momentum quality. No lower_shadow, no taker_buy_ratio.

---

## Debugging Common Gate Failures

| Gate Failing | Typical Cause | Fix |
|---|---|---|
| **IS Sharpe < 1.5** | Signal too weak overall | Combine with 1–2 complementary signals via zscore_cs + add |
| **Min sub-period Sharpe < 1.0 (H1)** | H2-dominant signals | Add `zscore_cs(sma(ts_zscore(close_position_in_range, 60), 60))` or `zscore_cs(sma(vwap_deviation, 120))` as H1 anchor; double-weight trades_per_volume delta |
| **Min sub-period Sharpe < 1.0 (H2)** | H1-dominant signals | Add taker_buy_ratio, log_return_zscore, or adv60 delta |
| **Rolling SR std > 0.05** | Large H1/H2 Sharpe gap | Tighten clip (try ±1.5 → ±1.2); add more balanced signals; or double-weight the H1 anchor |
| **PnL Kurtosis > 20** | Extreme position outliers | Use `zscore_cs` (not raw) before adding; tighten clip; avoid `rank()` on high-magnitude fields |
| **PnL Skew < -0.5** | Left-tail crashes in signal | Avoid negating H2-dominant signals as H1 anchors; prefer natural H1-positive signals |
| **Turnover > 0.05 target** | Short lookback or noisy signal | Extend SMA window (60→90→120); use longer delta lookback (30→60) |

---

## Key Construction Principles

1. **Primary H1 anchors (in priority order):** `lower_shadow/high_low_range sma(120)` (H1=~2.0 in combo, NEW TOP ANCHOR), `ts_zscore(close_position_in_range, 60) sma(60)` (H1=1.58), `ts_delta(trades_per_volume, 30) sma(60)` (H1=1.46), `vwap_deviation sma(120)` (H1=1.22). `overnight_gap` is meaningless in 24/7 crypto — do not use.

2. **🌟 NEW: ts_corr(X, Y, d) cross-field signals (Mar 2026).** The most productive new alpha family. Instead of using X or Y as standalone signals, compute their rolling CORRELATION. This captures *regime quality* — when two unrelated signals co-move, it signals a productive regime for their interaction:
   - `ts_corr(close_position_in_range, taker_buy_ratio, 60)` = Confirmation momentum (microstructure + flow alignment). **Best alpha: SR=+1.97, Fitness=3.84**
   - `ts_corr(log_returns, funding_rate_zscore, 60)` = Carry-return quality (carry working efficiently). **Rolling SR=0.037 (lowest in DB)**
   - `ts_corr(log_returns, trades_per_volume, 60)` = Retail-momentum quality (retail buying rewarded)
   - `ts_corr(log_returns, delay(log_returns, 1), 60)` = Return autocorrelation (momentum regime quality, used in Alpha #26)
   - **Key insight: ts_corr with X inside means X is NOT a standalone zscore, avoiding {X+...} forbidden triples!**

3. **Double-weighting via add(s, s).** To give a signal 2× weight in a composite without changing the expression structure, write `add(zscore_cs(signal), zscore_cs(signal))`. This is cleaner than multiplying by 2.

4. **Clip tightness controls rolling SR std.** Wrap the entire composite in `df_min(df_max(<signal>, -1.5), 1.5)` to truncate extreme cross-sectional outliers **before** the pipeline normalizes the signal. This is a pre-normalization step that caps raw alpha scores in ±1.5 sigma — it reduces PnL kurtosis and rolling SR std by preventing any single asset's extreme score from dominating the cross-section. The `±1.5` value is the sweet spot; try `±1.2` or `±1.0` if rolling SR std is stubborn. This clipping will eventually be built into the pipeline globally, but for now include it explicitly in every expression.

4. **Use s_log_1p() for volume-type fields.** Raw volume/ADV fields have extreme outliers. `s_log_1p()` compresses the distribution and dramatically improves kurtosis and signal stability.

5. **Round lookbacks: 30, 60, 90, 120 bars.** These correspond to real timeframes (~5d, 10d, 15d, 20d at 4h bars). Always test sensitivity at adjacent round lookbacks before finalizing.

6. **zscore_cs vs rank().** Both normalize cross-sectionally. `zscore_cs` preserves relative distances (better kurtosis in combo); `rank()` maps to [0,1] (more robust to outliers). Either works — test both when close to gates.

---

## Correlation Gate Notes (what gets rejected when)

| Signal Combination | Rejected With | Notes |
|---|---|---|
| `{vwap + trades_per_vol + taker + adv60}` | Alpha #20 | This IS Alpha #20. Any combo sharing all 4 signals hits corr ≥ 0.70. |
| `{close_position + trades_per_vol + taker + adv60}` | Alpha #21 | This IS Alpha #21. Any combo sharing all 4 hits corr ≥ 0.70. |
| `{close_position + taker + adv60}` | Alpha #21 | Even 3 of 4 signals shared is enough for rejection. |
| `{vwap + trades_per_vol + taker}` | Alpha #20 | 3 of 4 signals sufficient for rejection. |
| `multiply(rank(vwap+skew+trades), rank(taker+adv60))` | Alpha #20 | multiply doesn't help if underlying signals are identical to existing alpha. Must change ≥ 1 signal per group. |
| `{funding_rate_zscore + taker_buy_ratio + adv60}` | **Alpha #22** | This IS Alpha #22. Cannot use all three together. |
| `{funding_rate_zscore + taker_buy_ratio + log_ret_zscore}` | **Alpha #23** | This triple (+ anything) is correlated with Alpha #23. Avoid. |
| `{lower_shadow + close_position(60) + funding + log_ret + taker}` | Alpha #25 | This IS Alpha #25. Even changing lookbacks → corr=0.877. Must use different fields. |
| `{lower_shadow + funding + autocorr + log_ret + skewness}` | Alpha #26 | This IS Alpha #26. |
| `{lower_shadow + vwap + taker + adv60 + ...}` | Alpha #20 | `{vwap + taker + adv60}` triple sufficient for rejection even when lower_shadow added. |
| `{negative(parkinson_vol) + funding + log_ret + taker}` | Alpha #23 | corr=0.75 with Alpha #23. |

**Updated rules (Mar 2026):**
- The pair `{funding_rate_zscore + taker_buy_ratio}` is high-risk: adding adv60 OR log_ret_zscore creates rejection (Alphas #22/#23).
- `lower_shadow` is safe to pair with EITHER `taker_buy_ratio` OR `close_position_in_range`, but NOT both (that's Alpha #25).
- `ts_regression(log_returns, vol20, 60, 0, 2)` is currently SAFE — no existing alpha uses it.
- `ts_corr(log_returns, delay(log_returns, 1), 60)` is currently SAFE (only in Alpha #26 with very specific co-signals).
- Adding 6 signals can pass the corr gate even when sub-triplets overlap with existing alphas.

---

## Unexplored / Promising Directions

- `taker_buy_quote_volume` — ❌ TESTED: weak signal at 30-bar delta. Not worth exploring further.
- `ts_corr(log_returns, delay(log_returns, 1), 60)` — ✅ USED in Alpha #26. Return autocorrelation. Currently safe for new combinations with different co-signals.
- `ts_regression(log_returns, vol, 60, 0, 2)` — ✅ USED in Alpha #27. The slope is a strong, low-kurtosis signal. `ts_regression(y,x,d,lag,rettype)`: rettype=2=slope, rettype=0=residual.
- `ts_regression(log_returns, taker_buy_ratio, 60, 0, 2)` slope — market impact coefficient. Untested. Captures how much taker buying translates to returns (informed flow quality).
- `ts_regression(vwap_deviation, momentum_20d, 60, 0, 0)` residual — pure VWAP microstructure stripped of price momentum. Tested at H1=+0.36/H2=+1.86 — needs more H1 support.
- `upper_shadow / high_low_range` negated — H2-dominant (sellers capitulate in bull market). NOT yet used in a passing composite. Needs strong H1 anchors (lower_shadow or close_position).
- `volume_momentum_5_20` (5d ADV / 20d ADV) — H1-dominant in isolation. Could pair with strong H2 signals.
- `ts_kurtosis(log_returns, 60)` — TESTED: H2-dominant signal alone. Needs H1 anchors.
- `ts_regression` slope with different (x,y) pairs — broad unexplored space. Try: `(log_returns, funding_rate, 60)`, `(close_position_in_range, taker_buy_ratio, 60)` for novel regime interaction signals.
- `multiply(rank(group_A), rank(group_B))` with `{ts_regression_slope + lower_shadow + trades_per_vol}` × `{log_ret_zscore + skewness}` — novel AND-logic on new signal groups.
