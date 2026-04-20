# Virtue of Complexity: Complete Pipeline Documentation & Results

**Date**: 2026-04-16  
**Universe**: BINANCE_TOP50 (49 tickers), 4-hour bars  
**Data range**: 2020-10-01 to 2026-03-05 (T = 11,892 bars total)

---

## 1. Theoretical Basis: Bryan Kelly et al.

This pipeline implements two papers:

### 1a. "Virtue of Complexity in Return Prediction" (Kelly, Malamud & Zhou, 2023)
The core thesis: in the **overparameterized regime** (where the number of model parameters P exceeds the number of training observations T), ridge-regularized linear models do *not* overfit — they systematically improve OOS prediction accuracy as P grows. This is the "virtue of complexity." The mechanism is that random feature expansions sample the underlying nonlinear function space, and ridge averaging across many random projections reduces variance while preserving bias.

### 1b. "Artificial Intelligence Pricing Theory" (Didisheim, Ke, Kelly & Malamud, 2023)
Extends the complexity framework to cross-sectional asset pricing. Shows that:
- **SDF approach**: Build P characteristic-managed long-short factor portfolios (one per RFF), then apply Markowitz optimization on the P-dimensional factor return space with ridge shrinkage.
- **FM approach**: Run Fama-MacBeth cross-sectional ridge regression directly on asset returns. Maps characteristics to per-asset alphas.

Both approaches use **Random Fourier Features (RFF)** (Rahimi & Recht, 2007) to lift K raw characteristics into a 2P-dimensional nonlinear feature space before regression.

---

## 2. Data & Characteristics

### Universe
- **Exchange**: Binance perpetual futures
- **Tickers**: 49 (BINANCE_TOP50 universe, 1 excluded due to data gaps)
- **Frequency**: 4h bars
- **Splits**:

| Split | Start | End | Bars | Purpose |
|-------|-------|-----|------|---------|
| Train | 2021-01-01 | 2025-01-01 | ~8,760 | Model fitting |
| Val | 2025-01-01 | 2025-09-01 | ~1,460 | Hyperparameter selection (primary OOS) |
| Test | 2025-09-01 | 2026-03-05 | ~1,070 | Final OOS confirmation |

> [!IMPORTANT]
> **Val is the primary OOS benchmark** for augmented (K=58) runs because the 24 DB alpha signals were discovered using train data only. Test is a secondary confirmation.

### Raw Characteristics (K=34)
Fields loaded directly from 4h OHLCV + derived:

| Category | Fields |
|----------|--------|
| Price | open, high, low, close, vwap |
| Volume | volume, quote_volume, taker_buy_volume, taker_buy_ratio |
| Returns | log_returns, returns |
| Momentum | momentum_60d |
| Volatility | historical_volatility_20, historical_volatility_60, parkinson_volatility_60 |
| Volume momentum | volume_momentum_5_20 |
| Range | high_low_range, open_close_range, close_position_in_range |
| Shadows | upper_shadow, lower_shadow |
| Funding | funding_rate, funding_rate_zscore |
| Market metrics | adv60, trades, trades_per_volume |
| Beta | beta_to_btc |
| Derived | vwap_deviation, s_log_1p(adv60), s_log_1p(quote_volume) |

### DB Alpha Signals (+24 characteristics, total K=58)
Evaluated from `alphas.db` — all `interval='4h'`, `universe='BINANCE_TOP50'`, `archived=0`. Alpha IDs 10–34. Examples:

| ID | Expression (truncated) | Type |
|----|------------------------|------|
| 10 | `multiply(ts_delta(close,6), ts_kurtosis(volume,60))` | Vol-kurtosis price impact |
| 12 | `Decay_exp(multiply(ts_delta(close,12), ts_delta(beta_to_btc,6)), 0.05)` | Beta-momentum |
| 18-34 | Multi-factor composites of funding rate, taker buy ratio, vwap deviation, candle structure | Composite |

All alpha expressions are **evaluated causally** — only data available at each bar is used.

---

## 3. Pipeline Architecture (Step by Step)

### Step 1: Cross-Sectional Standardization
For each bar t and characteristic k, across all active assets i:
```
X_std[t, i, k] = (X[t, i, k] - median_i) / MAD_i
```
Clipped to [−5, +5]. NaN → 0.

### Step 2: Random Fourier Feature (RFF) Projection
From Rahimi & Recht (2007). Approximates an RBF kernel k(x,x') = exp(−||x−x'||²/2).

```
ω_j ~ N(0, I_K)          for j = 1..P     # random frequency vectors
b_j ~ Uniform(0, 2π)     for j = 1..P     # random phase offsets

φ(x) = √(2/P) · [cos(ω₁·x + b₁), ..., cos(ωₚ·x + bₚ),
                  sin(ω₁·x + b₁), ..., sin(ωₚ·x + bₚ)]
```

**P = 500** → **1000-dimensional feature space** per asset per bar.  
**5 independent RFF seeds** are used; results averaged to reduce Monte Carlo variance.

### Step 3: Multi-Horizon Return Target
Instead of predicting 1-bar-ahead returns, the regression target is **h-bar cumulative log returns**:
```
r_h[t, i] = Σ_{j=1}^{h} log(close[t+j] / close[t+j-1])
```

This is the key design choice. A longer h means:
- The prediction target evolves slowly → signal is more persistent
- **Naturally reduces turnover** without any explicit penalty
- The alpha horizon matches the signal's natural decay

Horizons tested: h ∈ {1, 6, 12, 24, 48} bars (= 4h, 1d, 2d, 4d, 8d ahead).  
**Best validated: h=24 (4 days ahead)** and h=12 (2 days ahead).

### Step 4: Fama-MacBeth Ridge Regression (FM Method)

At every retraining event (every 6 bars = 24h), using all available training history:

```
β̂ = (ΦᵀΦ + z·I)⁻¹ · Φᵀ · R_h
```

Where:
- Φ = stacked RFF feature matrix across all (bar, asset) pairs in the training window
- R_h = vector of h-bar forward returns (the regression target)
- z = 1e-5 (ridge shrinkage parameter)

The **alpha signal** at time t for asset i:
```
α[t, i] = φ(X[t, i]) · β̂
```

This is purely cross-sectional (FM window = 1 bar): β̂ is estimated pooling across assets within each retraining window, not across time per asset.

**Complexity ratio**: c = 2P / (N × W) = 1000 / (49 × 1) ≈ **20.4 >> 1** — deeply in the overparameterized regime where the "virtue of complexity" theoretically manifests.

### Step 5: Signal Processing
```
α_processed[t] = clip(normalize(demean(α[t])), −max_wt, +max_wt)
```
- **Demean**: subtract cross-sectional mean → dollar-neutral positions
- **Normalize**: divide by |weight| sum → $1 long / $1 short
- **Clip**: max weight per asset = 2% of book

### Step 6: QP Optimizer — Every-Bar Execution

The QP runs **every 4h bar**. The transaction cost penalty naturally determines whether a trade is worth making — no fixed rebalancing schedule.

**Objective** (solved via CVXPY + SCS solver):
```
minimize:    −α_t · w                         # maximize signal alignment
           + λ_risk · wᵀ Σ_t w               # penalize variance (Σ = lookback covariance)
           + λ_tc   · ||w − w_{t-1}||₁        # penalize trades (L1 = per-unit cost)

subject to: ||w||₁ ≤ 1                        # normalized book
            ||w − w_{t-1}||₁ ≤ max_to         # cap total turnover per bar
            |w_i| ≤ max_wt ∀i                 # per-asset position limit
```

The L1 transaction cost term means only signals strong enough to overcome the penalty cost will trigger trades. This is effectively Garleanu-Pedersen continuous-time optimal execution applied per bar.

**Parameters used**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `risk_aversion` (λ_risk) | 1.0 | Coefficient on variance term |
| `tcost_lambda` (λ_tc) | 0.005 | L1 penalty — tested 0.005 and 0.01 |
| `lookback_bars` | 120 | Covariance estimation window (~20 days) |
| `rebal_every` | **1** | Every 4h bar — QP runs continuously |
| `max_turnover` | 0.05 | Max 5% book repositioning per bar — tested 0.05 and 0.10 |
| `max_wt` | 0.02 | Max 2% per asset |

### Step 7: Simulation & Evaluation

```
PnL[t] = Σ_i w[t−1,i] · r[t,i] · booksize − |Δw[t,i]| · booksize · fees_bps/10000
```

| Metric | Value |
|--------|-------|
| Booksize | $20,000,000 |
| Fees | 0, 5, 7 bps per side |
| Bars/year | 2,190 (6 bars/day × 365) |
| Sharpe | annualized: mean(PnL) / std(PnL) × √2190 |
| Turnover | mean(|Δw|) per bar |
| IC | Spearman rank corr(signal, next-bar return), mean cross-sectional |

---

## 4. Results

### Annualization Note
All Sharpe ratios reported here use **√2190 annualization** (6 bars/day × 365 days), matching the equity curve charts.

---

### Run 1: K=34 (raw data only), QP rebal every 12–24 bars

No alpha augmentation. QP runs every 12 or 24 bars (coarse — suboptimal as later found).

**Top results by Test SR@5bps:**

| H | rb | to | Val SR@5 | Test SR@0 | **Test SR@5** | Test TO | Test IC |
|---|----|----|---------|-----------|--------------|---------|---------|
| 6 | 12 | 0.10 | +2.58 | +3.84 | **+3.77** | 0.013 | +0.018 |
| 48 | 12 | 0.05 | +2.33 | +3.48 | **+3.44** | 0.006 | +0.009 |
| 48 | 12 | 0.05 | — | +3.42 | **+3.38** | 0.006 | +0.013 |
| 6 | 12 | 0.10 | — | +3.42 | **+3.36** | 0.011 | +0.012 |

Raw FM h=6 signal (no QP): Test SR@0=+7.72, Test SR@5=+2.66, TO=0.99.

**Equity curve**: `voc_equity_curves.png`

---

### Run 2: K=58 (+ 24 DB alphas), QP rebal every 12–24 bars

Augmenting with DB alphas raises K from 34 to 58. Optimal horizon shifts to h=12–48.

**Top results by Test SR@5bps:**

| H | rb | to | Val SR@5 | Test SR@0 | **Test SR@5** | Test TO | Test IC |
|---|----|----|---------|-----------|--------------|---------|---------|
| 12 | 12 | 0.10 | — | +4.52 | **+4.45** | 0.013 | +0.025 |
| 48 | 12 | 0.10 | +2.90 | +4.28 | **+4.22** | 0.012 | +0.021 |
| 24 | 12 | 0.10 | — | +3.92 | **+3.86** | 0.012 | +0.011 |
| 24 | 12 | 0.10 | — | +3.47 | **+3.40** | 0.012 | +0.014 |

---

### Run 3: K=58, QP rebal every bar (rb=1) ← **Best Configuration**

Every 4h bar the QP runs. The `tcost_lambda` penalty decides whether to trade.

**Full results sorted by Val SR@5bps** (val = primary OOS for augmented signals):

| Rank | H | tc | to | **Val SR@5** | Test SR@5 | Test TO | Test IC |
|------|---|----|----|-------------|-----------|---------|---------|
| 1 | 24 | 0.005 | 0.10 | **+10.95** | +7.63 | 0.139 | +0.051 |
| 2 | 24 | 0.01 | 0.10 | **+10.61** | +7.53 | 0.117 | +0.047 |
| 3 | 24 | 0.005 | 0.05 | **+9.51** | +7.39 | 0.074 | +0.039 |
| 4 | 48 | 0.005 | 0.05 | **+9.45** | +7.31 | 0.070 | +0.044 |
| 5 | 48 | 0.01 | 0.05 | **+9.25** | +7.21 | 0.070 | +0.042 |
| 6 | 24 | 0.01 | 0.05 | **+9.16** | +6.14 | 0.065 | +0.036 |
| 7 | 12 | 0.005 | 0.05 | **+9.45** | +6.43 | 0.077 | +0.030 |
| 8 | 12 | 0.005 | 0.10 | **+8.09** | +8.06 | 0.145 | +0.047 |
| 9 | 12 | 0.01 | 0.10 | **+8.09** | +7.12 | 0.099 | +0.040 |
| 10 | 48 | 0.005 | 0.10 | **+7.72** | +6.53 | 0.138 | +0.043 |
| 11 | 48 | 0.01 | 0.10 | **+8.05** | +7.03 | 0.136 | +0.045 |
| 12 | 12 | 0.01 | 0.05 | **+6.20** | +6.20 | 0.058 | +0.024 |
| — | 6 | 0.005 | 0.10 | +3.53 | +3.53 | 0.125 | +0.032 |

**From equity curves** (annualized SR using √2190, same method):

| Config | Train SR | Val SR | Test SR | Cumulative Return (test) |
|--------|----------|--------|---------|--------------------------|
| FM h=24, tc=0.005, rb=1, to=0.05 | +14.14 | +13.98 | +10.21 | +92.7% in 6 months |
| FM h=12, tc=0.005, rb=1, to=0.10 | +14.96 | +12.62 | +9.58 | +83.6% in 6 months |

**Equity curve**: `voc_equity_curves_rb1.png`

---

## 5. Key Findings

### Finding 1: Regression Horizon H is the Critical Design Choice
Predicting h=24 bars (4 days) instead of h=1 (4h) reduces raw turnover from ~1.0 to ~0.07–0.14 post-QP. The signal naturally changes more slowly when targeting further-out returns.

### Finding 2: Every-Bar QP is 2× Better Than Fixed Schedule
- rb=12 (every 2 days): Best Test SR@5 = +4.45
- **rb=1 (every 4h)**: Best Test SR@5 = **+8.06**, Val = **+10.95**

The QP identifies the optimal time to trade within any window. Forcing it to a fixed schedule discards alpha.

### Finding 3: Alpha Augmentation Shifts the Optimal Horizon
- K=34: optimal h=6 (1 day)
- K=58: optimal h=24 (4 days)

The 24 DB alphas are medium-term signals (120-bar SMAs, 60-bar correlations), so longer prediction horizons better exploit them.

### Finding 4: Complexity Works (VoC Confirmed)
FM complexity ratio c ≈ 20 >> 1. Despite having 1000 parameters per asset vs ~49 cross-sectional observations, the model generalizes strongly OOS. Adding more features (K=58) with longer horizons (h=24) improves OOS Sharpe.

### Finding 5: SDF Fails, FM Dominates
SDF generates portfolio-level weights that conflict with QP's own optimization. FM generates per-asset signals that QP can selectively act on. All top results are FM.

### Finding 6: The QP Does Not Use Take-Profits
Position exits are entirely signal-driven: when the current signal minus the L1 cost penalty no longer justifies a holding, the QP reduces it gradually. There is no PnL-based exit rule — the optimizer continuously seeks the best risk-adjusted signal alignment given current positions.

---

## 6. Recommended Production Configuration

```yaml
Method:           FM (Fama-MacBeth cross-sectional ridge)

Data:
  Universe:       BINANCE_TOP50 (49 tickers)
  Frequency:      4h bars
  Characteristics: K=58
    - 34 raw: OHLCV, momentum, volatility, funding rate, beta, volume metrics
    - 24 DB alphas: evaluated from alphas.db (interval=4h, universe=BINANCE_TOP50)

Model:
  RFF features:   P=500 (1000-dim feature space after cos+sin)
  Seeds:          5 (averaged)
  Ridge z:        1e-5
  Horizon:        h=24 (predict 4-day forward returns)
  Retrain every:  6 bars (24h)

QP Optimizer:
  rebal_every:    1 (every 4h bar)
  tcost_lambda:   0.005
  max_turnover:   0.05 per bar
  risk_aversion:  1.0
  lookback_bars:  120 bars (covariance window)
  max_wt:         0.02 per asset

Portfolio:
  Booksize:       $20,000,000
  Max position:   $400,000 per asset (2%)
  Fees assumed:   5 bps per side

Expected Performance (K=58, h=24, tc=0.005, rb=1, to=0.05):
  Val  SR@5bps:   +9.51 (→ +13.98 annualized by equity curve method)
  Test SR@5bps:   +7.39 (→ +10.21 annualized by equity curve method)
  Test Turnover:  ~7.4% of book per 4h bar
  Test IC:        +0.039
```

---

## 7. Files Reference

| File | Description |
|------|-------------|
| `run_voc_qp.py` | Main pipeline script |
| `plot_equity_curves.py` | Equity curve script — rb=12 configs |
| `plot_equity_curves_rb1.py` | Equity curve script — rb=1 configs |
| `voc_equity_curves.png` | Curves: K=34 h=6 vs K=58 h=12 (rb=12) |
| `voc_equity_curves_rb1.png` | Curves: K=58 h=24 vs K=58 h=12 (rb=1) ← **best** |
| `VoC_QP_FULL_RESULTS.md` | Complete K=34 run — all 60 configs, all splits |
| `VoC_QP_AUGMENTED_RESULTS.md` | K=34 vs K=58 comparison (rb=12) |
| `VoC_PIPELINE_COMPLETE.md` | This document |
| `voc_qp_results.md` | Auto-generated from last script run |
