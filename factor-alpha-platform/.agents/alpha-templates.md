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

1. **Primary H1 anchors (in priority order):** `ts_zscore(close_position_in_range, 60) sma(60)` (H1=1.58), `vwap_deviation sma(120)` (H1=1.22), `ts_delta(trades_per_volume, 30) sma(60)` (H1=1.46). `overnight_gap` is meaningless in 24/7 crypto — do not use for new alphas.

2. **Double-weighting via add(s, s).** To give a signal 2× weight in a composite without changing the expression structure, write `add(zscore_cs(signal), zscore_cs(signal))`. This is cleaner than multiplying by 2.

3. **Clip tightness controls rolling SR std.** The `±1.5` clip is generally the sweet spot — tighter reduces kurtosis but raises skew and may reduce overall Sharpe. Try `±1.2` or `±1.0` if rolling SR std is stubborn. Alternatively, switch to `multiply(rank(A), rank(B))` architecture which naturally handles rolling SR via its [0,1] bounded output.

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

**Rule of thumb:** You must change at least 2 of 4 core signals from an existing passing alpha to pass the correlation gate. Changing only the operator (add → multiply) is NOT sufficient.

---

## Unexplored / Promising Directions

- `taker_buy_quote_volume` — dollar-denominated taker buy flow (not yet tested as composite component)
- `ts_corr(log_returns, delay(log_returns, 1), 60)` — return autocorrelation (serial correlation) as a momentum quality filter
- `ts_skewness(log_returns, 60) sma(60)` as H1 anchor in multiply(rank, rank) — novel statistical signal, contributes ~H1=0.5 alone but synergizes well
- `multiply(rank(group_A), rank(group_B))` where Group A uses `{close_position_zscore + ts_skewness + trades_per_vol}` and Group B uses signals NOT in #20 or #21 (e.g., `quote_volume_delta + log_return_zscore`)
- Beta-to-BTC deviation forms: e.g., `ts_delta(beta_to_btc, 60)` smoothed (already in DB as alphas 12/14/16 — check correlation gate carefully)
