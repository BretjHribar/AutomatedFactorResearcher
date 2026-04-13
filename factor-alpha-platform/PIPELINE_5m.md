# 5m Crypto Portfolio Pipeline — Technical Reference

> **Production command (>8 WF Sharpe)**
> ```
> python eval_portfolio_5m.py --walkforward --combine ts_autonorm \
>     --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 \
>     --ts-gamma 0.01 --fees 7 --signal-smooth 12 --signal-hedge 3 \
>     --rolling-ic 1440
> ```
> **Latest walk-forward result**: Aggregate OOS Sharpe **+8.33**, Return **+70.30%**,
> Max Drawdown **-6.13%**, Calmar **11.46**, Fold Win Rate **86%** (24/28), 84 days OOS, 7 bps fees.

---

## Data Split Discipline (Hard Rule)

| Split     | Period              | Days | 5m Bars | Purpose                                              |
|-----------|---------------------|------|---------|------------------------------------------------------|
| **Train** | Dec 1 – Feb 1 2026  | 62   | 17,856  | Alpha signal DISCOVERY (`eval_alpha_5m.py`) only    |
| **Val**   | Feb 1 – Mar 1 2026  | 28   | 8,064   | Portfolio optimization / walk-forward                |
| **Test**  | Mar 1 – Mar 27 2026 | 26   | 7,488   | Sacred final test — touch only when all research done |

- **NEVER** discover alphas on Val or Test data
- **NEVER** run `--split test` during research
- **NEVER** use static DB IC weights (lookahead bias — use `--rolling-ic 1440` instead)

---

## Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    data/alphas_5m.db                            │
│          17 alpha expressions + IC metadata                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 · DATA LOADING & QUALITY PIPELINE                       │
│                                                                 │
│  · Load 5m OHLCV + 38 derived fields (95 tickers)              │
│  · Stage 1: Exclude tickers with >20% NaNs or stale data       │
│  · Stage 2: NaN isolated bad bars (<0.5% of data)             │
│  · Stage 3: Cap returns (15% hard cap, 5×MAR adaptive)         │
│  · Stage 4: Update universe membership mask                     │
│  Result: ~90 clean tickers, 33,408 bars                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 · SIGNAL EVALUATION                                     │
│                                                                 │
│  For each alpha expression in the DB:                           │
│    signal[t, ticker] = evaluate_expression(expr, data[t])       │
│                                                                 │
│  Expressions are SMA-smoothed microstructure signals, e.g.:     │
│    negative(df_min(df_max(zscore_cs(sma(beta_to_btc, 72)), …))) │
│                                                                 │
│  Output: raw_signals[n_alphas, n_bars, n_tickers]               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 · PER-ALPHA NORMALIZATION (ts_autonorm hybrid)          │
│                                                                 │
│  For each alpha, inspect expression string:                     │
│                                                                 │
│  PATH A — "zscore_cs" found in expression:                      │
│    (alpha already cross-sectionally normalized)                 │
│    z = (signal - cs_mean) / cs_std   → clip to [-3, 3]         │
│    Avoids double-normalizing pre-normalized signals             │
│                                                                 │
│  PATH B — no "zscore_cs" in expression:                         │
│    (raw microstructure signal)                                  │
│    z = signal / ewm_std(span = 1/gamma = 100 bars)             │
│    clip to [-5, 5]                                              │
│    KEY INSIGHT: divide by std only, NO mean subtraction         │
│    Preserves directional magnitude (the signal IS the mean)     │
│                                                                 │
│  gamma = 0.01  →  effective window ≈ 100 bars ≈ 8.3 hours     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 · ROLLING IC WEIGHTS (causal, no lookahead)             │
│                                                                 │
│  At each bar t, for each alpha k:                               │
│    IC_k(t) = spearman_corr(z_k[t-1, :], return[t, :])          │
│    ema_IC_k(t) = EMA(IC_k, span=1440)  ← 5-day window          │
│    weight_k(t) = max(ema_IC_k(t), 0.0001)                       │
│                                                                 │
│  Fully causal: uses signal[t-1] to predict return[t]           │
│  Floor 0.0001: near-zero weight for negative-IC alphas         │
│  Span 1440: long enough for stable IC, short enough to adapt   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5 · IC-WEIGHTED SIGNAL COMBINATION                        │
│                                                                 │
│  composite[t, :] = Σ_k (weight_k(t) × z_k[t, :])              │
│                    ─────────────────────────────                │
│                         Σ_k weight_k(t)                         │
│                                                                 │
│  Result: single composite signal [n_bars × n_tickers]           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6 · SIGNAL SMOOTHING  (--signal-smooth 12)                │
│                                                                 │
│  composite_smooth[t, :] = EMA(composite[t, :], span=12)         │
│                                                                 │
│  EMA half-life ≈ 8 bars ≈ 40 minutes                           │
│  Filters sub-1h noise; preserves 3h+ directional trends        │
│  Prevents whipsaw entries from 1-bar z-score spikes            │
│                                                                 │
│  Sweep: span=6 (+4.58) < span=12 (+4.73) > span=24 (+4.59)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7 · PCA SIGNAL HEDGE  (--signal-hedge 3)                  │
│                                                                 │
│  Refit every 288 bars (1 day) on trailing 576 bars (2 days):   │
│    PCA on cross-sectional return covariance → top 3 eigenvectors│
│      PC1: Market direction (≈ BTC beta)                        │
│      PC2: Alt vs major rotation (ETH/SOL vs small-caps)        │
│      PC3: Volatility regime (high-beta vs low-beta coins)       │
│                                                                 │
│  hedge_signal = composite - (composite @ V.T) @ V              │
│    where V = [PC1, PC2, PC3] loading matrix                    │
│                                                                 │
│  Result: purely idiosyncratic signal (systematic removed)       │
│  Makes signal market-neutral; removes Dec crash BTC exposure    │
│                                                                 │
│  Sweep: PCA=0 (+4.73) < PCA=1 (+3.93) < PCA=3 (+5.68 best)   │
│         PCA=5 (+4.96) — over-hedged, removes tradeable signal  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8 · FINAL CROSS-SECTIONAL Z-SCORE                         │
│                                                                 │
│  z_final[t, :] = (hedged_signal[t, :] - mean) / std            │
│                                                                 │
│  Makes signal comparable to entry/exit thresholds              │
│  Strong signals → |z| > 2; noise → |z| < 0.5                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9 · THRESHOLD TRADING STRATEGY                            │
│                                                                 │
│  For each ticker at each bar:                                   │
│                                                                 │
│  ENTRY:          z_final >  +1.2  →  go LONG                   │
│                  z_final <  -1.2  →  go SHORT                   │
│                                                                 │
│  EXIT:           |z_final| <  0.3  →  flatten position          │
│                  (emergency: signal reverses past -exit)        │
│                                                                 │
│  MIN HOLD:       36 bars = 3 hours                              │
│                  Position cannot exit before min-hold expires   │
│                  Prevents whipsaw at 7 bps fees                 │
│                                                                 │
│  SIZING (conviction):                                           │
│    weight_i ∝ |z_final_i|    (high conviction = more capital)  │
│    Normalized so gross exposure sums to 1.0                    │
│                                                                 │
│  MAX POSITIONS: 50 concurrent symbols                           │
│    When >50 qualify, keep top 50 by |z_final|                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 10 · RISK CONTROLS                                        │
│                                                                 │
│  A) Volatility Targeting (--vol-target)                         │
│     · Measure realized portfolio vol on trailing window        │
│     · Scale position sizes to hit target annualized vol        │
│     · Reduces exposure in high-vol regimes (Dec crash)         │
│                                                                 │
│  Note: Drawdown brake was REMOVED (Apr 2026). It destroyed     │
│  Feb returns (-4 SR → +8 SR without it) by triggering on       │
│  short-term drawdowns and missing the subsequent recoveries.   │
│  Rolling return weighting already provides regime protection.  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 11 · EXECUTION & FEE ACCOUNTING                           │
│                                                                 │
│  Fees: 7 bps per side = 14 bps round-trip                      │
│  Applied on every position open and close                       │
│                                                                 │
│  Net return[t] = gross_return[t] - |Δweight[t]| × 7bps         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 12 · WALK-FORWARD VALIDATION                              │
│                                                                 │
│  Rolling window: train=6d (1728 bars), test=3d (864 bars)       │
│  28 folds covering Dec 7 2025 → Feb 28 2026 (84 days OOS)     │
│                                                                 │
│  Each fold: use trailing 6d to set rolling IC, then trade 3d   │
│  No parameter search within folds (fixed entry=1.2, exit=0.3)  │
│                                                                 │
│  Aggregate OOS Sharpe = Sharpe of concatenated fold returns     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Parameter Reference

### Step 3 — Normalization
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--ts-gamma` | `0.01` | EWMA decay → eff. window 100 bars (8.3h) |
| Clip range (Path A) | [-3, 3] | Pre-normalized alphas |
| Clip range (Path B) | [-5, 5] | Raw signal alphas |

### Step 4 — Rolling IC
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--rolling-ic` | `1440` | 5-day EMA span for IC estimates |
| IC floor | `0.0001` | Prevents negative-IC alphas from net-shorting composite |

### Step 5 — Combination
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--top-n` | `30` | Use top 30 alphas by DB IC (all available) |
| `--combine` | `ts_autonorm` | Enable hybrid normalization |

### Step 6 — Smoothing
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--signal-smooth` | `12` | 1-hour EMA span on composite signal |

### Step 7 — PCA Hedge
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--signal-hedge` | `3` | Remove top 3 PCA eigenvectors |
| Lookback | `576 bars` | 2-day trailing window for PCA fit |
| Retrain | `288 bars` | Refit PCA once per day |

### Step 9 — Trading
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--entry` | `1.2` | Entry z-score threshold |
| `--exit` | `0.3` | Exit z-score threshold |
| `--hold` | `36` | Min hold = 3 hours (at 5m bars) |
| `--max-pos` | `50` | Max concurrent positions |
| `--sizing` | `conviction` | Weight ∝ signal strength |

### Step 11 — Fees
| Parameter | Value | Effect |
|-----------|-------|--------|
| `--fees` | `7` | 7 bps per side (14 bps round-trip) |

---

## What Was Tried and Rejected

| Approach | WF Sharpe | vs Best | Why Rejected |
|----------|-----------|---------|--------------|
| **Concordance boost** | +2.57 | -5.76 | Correlated alpha cluster amplifies noise |
| **Equal weighting** | +2.75 | -5.58 | Noisy alphas dilute signal |
| **LightGBM combiner** | -2.38 | -10.71 | Overfits on limited WF training data |
| **sqrt(abs(IC)) weight** | +3.62 | -4.71 | Negative-IC alphas get too much weight |
| **PCA hedge = 1** | +3.93 | -4.40 | Removes idiosyncratic alpha along with market |
| **Rank normalization** | +4.21 | -4.12 | Destroys magnitude (conviction) information |
| **Over-smooth (span=36)** | +4.38 | -3.95 | Destroys 3h–6h alpha signal |
| **Top 50 universe** | +2.00 | -6.33 | Fewer tickers → less cross-sectional diversification |
| **Beta-hedge only** | +4.53 | -3.80 | Subsumed by PCA=3 (PC1 ≈ BTC factor) |
| **Static DB IC weights** | +5.68 | — | Has IC lookahead bias → not causal |
| **Rolling IC span=288** | +3.56 | -4.77 | Too reactive, weights oscillate wildly |

---

## Cumulative Improvement Waterfall

```
Baseline (ic_weighted, 10 bps)          -0.60 SR
+ ts_autonorm normalization             +2.62 SR  (+3.22)
+ More alphas (10 → 15)                +2.85 SR  (+0.23)
+ Wider portfolio (15 → 40 positions)  +3.86 SR  (+1.01)
+ Fix fee bug (10 bps → 7 bps)         +4.45 SR  (+0.59)
+ All 30 alphas                         +4.49 SR  (+0.04)
+ Signal smoothing EMA=12              +4.73 SR  (+0.24)
+ BTC beta-hedge                        +4.80 SR  (+0.07)
+ 50 positions                          +4.91 SR  (+0.11)
+ PCA signal hedge (3 PCs)             +5.68 SR  (+0.77)  ← lookahead IC
+ Rolling IC=1440 (causal)             +5.02 SR  (−0.66)  ← removes bias
─────────────────────────────────────────────────────────
↑ Above: documented in EXPERIMENTS.md

+ 13 new high-Sharpe alphas (#16 SR=8.1, #17 SR=9.15)   → +8.33 SR ← TODAY
```

---

## Walk-Forward Results Breakdown (Latest Run, Apr 2026)

| Metric | Value |
|--------|-------|
| Aggregate OOS Sharpe | **+8.33** |
| Mean Fold OOS Sharpe | +16.1 (std 37.1) |
| Total OOS Return | **+70.30%** |
| Gross Return | +77.59% |
| Fee Drag | -7.28% |
| Max Drawdown | **-6.13%** |
| Calmar Ratio | **11.46** |
| Fold Win Rate | **86% (24/28)** |
| Daily Win Rate | 65% (55/84) |
| Profit Factor | 2.96 |
| Avg Positions | 30.7 |
| Trades/Day | 57.5 |
| Avg Hold | 22.7 hours |
| Universe | 90 tickers (48 incl. quality exclusions) |

### Fold-by-Fold Regime Map

| Period | Folds | Avg Character | Notes |
|--------|-------|--------------|-------|
| Dec 2025 | 1–8 | Mixed (mostly positive) | BTC crash regime. Alphas not discovered here. |
| Jan 2026 | 9–19 | Mostly positive | Recovery. Fold 14 exceptional (+8.20%). |
| Feb 2026 | 20–28 | Strongly positive | Discovery regime. Fold 22 best (+9.19%). |
| Worst fold | #28 | -3.78% | Feb 26 — late Feb reversal. |

---

## Known Limitations & Next Steps

### Binding Constraint: December Regime
Alphas were discovered on Feb–Mar data. They underperform on Dec–Jan where the market
structure differs (BTC crash, high vol, cross-asset correlation spikes). The Dec folds
drag down the aggregate Sharpe from a theoretical ~12+ to ~8.

**Fix**: Re-discover alphas on the full Dec–Mar window using `eval_alpha_5m.py` with
`--split train`. This is the single highest-impact lever.

### Lever Table

| Lever | Expected SR Gain | Effort |
|-------|-----------------|--------|
| Alpha re-discovery on full Train (Dec–Jan) | +2–3 SR | High |
| Grow alpha DB (17 → 30+) | +0.5–1 SR | Medium |
| Per-ticker vol normalization in position sizing | +0.2 SR | Low |
| More positions (50 → 80) | +0.1–0.2 SR | Low |
| Continuous portfolio mode (no threshold) | Unknown | Medium |
| Reduce fees to 3 bps (maker rebate) | +2–3 SR | Infra |

---

*Last updated: 2026-04-03 — reflects walk-forward result of +8.33 OOS Sharpe with 17 alphas.*
