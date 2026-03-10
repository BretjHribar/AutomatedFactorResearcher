# Isichenko Stat-Arb Pipeline — Detailed Walkthrough

> Implementation of *Quantitative Portfolio Management* (Michael Isichenko, 2021)
> Codebase: `src/pipeline/isichenko.py` + `run_isichenko_pipeline.py`

---

## Table of Contents

1. [Overview & Two-Stage Architecture](#1-overview--two-stage-architecture)
2. [What The Book Actually Says](#2-what-the-book-actually-says)
3. [Stage 1: Alpha Combination via Ridge Regression](#3-stage-1-alpha-combination-via-ridge-regression)
4. [Stage 2: Portfolio Optimization via QP](#4-stage-2-portfolio-optimization-via-qp)
5. [Complete Daily Loop](#5-complete-daily-loop)
6. [Risk Model](#6-risk-model)
7. [Fee & Transaction Cost Model](#7-fee--transaction-cost-model)
8. [Configuration Reference](#8-configuration-reference)
9. [Current Implementation vs Book](#9-current-implementation-vs-book)
10. [Known Issues & Future Work](#10-known-issues--future-work)

---

## 1. Overview & Two-Stage Architecture

The pipeline is a **two-stage optimization** that separates alpha combination from portfolio construction:

```
                    STAGE 1: Alpha Combination                STAGE 2: Portfolio Optimization
                    ─────────────────────────                 ──────────────────────────────
                                                              
  α₁(t) ──┐                                                  
  α₂(t) ──┤    Rolling Ridge           Combined               QP Optimizer        Optimal
  α₃(t) ──┼──▶ Regression     ──▶  Return Forecast  ──▶  min(risk - α'w + tcost) ──▶ Weights w*
  ...     ─┤    on fwd returns         μ̂(t)                  s.t. constraints  
  αₖ(t) ──┘    (Eq 2.32-2.38)                                (Eq 6.6)
                                                              ▲
                                                              │
                                                    Factor Risk Model C = BFB' + Dσ²
                                                         (Ch 4)
```

**Key principle**: Stage 1 finds the BEST LINEAR COMBINATION of alpha signals to predict forward 
returns (minimizing MSE). Stage 2 takes that combined forecast and finds optimal portfolio 
weights subject to risk, costs, and constraints. These are fundamentally different problems 
with different objectives.

### What Goes Wrong If You Skip Stage 1

If you skip the ridge regression and just rank-average the signals, you must manually scale 
the combined signal into return units. This introduces arbitrary scaling that breaks the 
QP's risk-return tradeoff:
- Too large → QP concentrates into max-weight positions (all positions hit caps)
- Too small → QP gives near-zero positions (everything subsumed by risk)
- No scaling works well because the proper scaling IS the ridge regression

---

## 2. What The Book Actually Says

### Isichenko Eq 2.32: Single Alpha OLS Scaling

For a single alpha signal f, the optimal linear predictor of forward return R is:

```
R̂ = k × f    where    k = Cov(f, R) / Var(f)
```

This is just the OLS slope coefficient. k is estimated with an EMA (halflife ~120 days).
The resulting R̂ is in **return units** (e.g., daily bps) — not rank units.

### Isichenko Eq 2.38: Multiple Alpha Combination 

For K alpha signals f₁...fₖ, the optimal linear combination is:

```
R̂ = Σᵢ wᵢ × fᵢ    where    w = (F'F + λI)⁻¹ F'R
```

This is **ridge regression** (OLS with L2 regularization):
- **F**: (T × K) matrix of alpha signal values across time (cross-sectionally stacked)
- **R**: (T,) vector of forward returns
- **λ**: ridge penalty controlling overfitting (set via cross-validation or the textbook recommendation)
- **w**: (K,) combination weights in return units

The output R̂ᵢ is automatically in return units — no manual Grinold scaling needed.

### Why Ridge, Not OLS?

With many correlated alpha signals:
- OLS gives unstable weights (near-singular F'F, large ±weights)
- Ridge shrinks all weights toward zero, preventing blowup
- Ridge penalty λ controls bias-variance tradeoff
- Cross-validate λ, or use λ ≈ trace(F'F) / K as starting point

### Isichenko Eq 6.6: Portfolio Optimization

Given combined forecast μ̂ (from Stage 1) and risk model C = BFB' + Dσ²:

```
minimize   ½κ w'Cw  - μ̂'w  + tcost(w - w_prev)
subject to Σ|wᵢ| ≤ 1           (GMV constraint in weight space)
           Σwᵢ = 0              (dollar neutrality)
           |wᵢ| ≤ w_max         (position limits)
```

Where:
- **κ**: risk aversion (how much you penalize variance relative to expected return)
- **w'Cw**: portfolio variance (factor risk + idiosyncratic risk)
- **μ̂'w**: expected portfolio return (from Stage 1)
- **tcost**: linear (slippage) + non-linear (market impact) trading costs

> **Critical**: μ̂ must be in the SAME UNITS as the diagonal of C (daily return variance).
> The ridge regression ensures this automatically because it regresses signals on actual returns.

---

## 3. Stage 1: Alpha Combination via Ridge Regression

### Rolling Ridge Regression (Walk-Forward)

On each day t, we have:
- K alpha signals: fₐ(t) for a = 1..K, each an N-vector across stocks
- Forward returns: R(t+1), an N-vector of next-day returns

We pool cross-sectional observations over a rolling window (e.g., 252 trading days):

```
For each day t:
  1. Collect training data from [t-window, t-1]:
     X[i,a] = fₐ at stock i on day d          (shape: window×N × K)
     y[i]   = R at stock i on day d+1          (shape: window×N)
     
  2. Solve ridge:  w* = (X'X + λI)⁻¹ X'y       (shape: K)
     
  3. Predict: μ̂(t) = Σₐ wₐ* × fₐ(t)           (shape: N — one per stock)
```

### Key Implementation Details

1. **Cross-sectional stacking**: Each day contributes N observations (one per stock in universe).
   Over a 252-day window with N=50 stocks, we have 12,500 training samples for K~30 signals.

2. **Ridge penalty λ**: 
   - Start with λ = trace(X'X) / K (auto-calibrated)
   - Or cross-validate with rolling train/validate split

3. **Rank-normalize signals first**: Before regression, rank-normalize each signal cross-sectionally 
   to [-0.5, +0.5]. This makes all signals comparable and avoids scale issues in the regression.

4. **The regression output is already in return units**: No Grinold scaling hack needed.

5. **Negative weights are fine**: Ridge can assign negative weights to signals if they're 
   anti-predictive after controlling for correlations with other signals.

### What Stage 1 Gives You

- A single N-vector μ̂(t) of expected returns per stock per day
- Typical magnitude: a few basis points (e.g., [-30 bps, +30 bps])
- Automatically decorrelated: correlated signals don't double-count
- Noise-robust: ridge shrinks zero-IC signals to zero weight

---

## 4. Stage 2: Portfolio Optimization via QP

### The QP Problem (Isichenko Eq 6.6)

Given μ̂(t) from Stage 1, solve:

```
minimize   ½κ ||Qw||²                    # Factor risk
         + ½κ Σᵢ σᵢ² wᵢ²                # Idiosyncratic risk  
         - μ̂'w                           # Expected return
         + c_linear |Δw|₁               # Linear trading cost (slippage)
         + c_impact Σᵢ (σᵢ/√ADVᵢ) Δwᵢ²  # Market impact (Almgren-Chriss)
         + λ_trade ||Δw||²              # Turnover aversion

subject to  |wᵢ| ≤ w_max_i             # Per-stock position limit
            ||w||₁ ≤ 1                   # GMV ≤ booksize
            Σw = 0                       # Dollar neutral
```

Where:
- w: (N,) weight vector (fraction of booksize)
- Q: (K×N) factor risk sqrt matrix (Q'Q = BFB')
- σ²: (N,) idiosyncratic variance
- Δw = w - w_prev
- κ: risk aversion parameter

### Risk Aversion Calibration

κ controls the risk-return tradeoff. The optimal κ depends on:
- Signal quality (higher IC → lower κ to capture more alpha)
- Universe size (more stocks → lower κ because diversification helps)  
- Position limits (tighter limits → κ matters less)

**Calibration method**: Set κ such that the optimal portfolio has target volatility ~15-20% 
annualized, or equivalently, the average position size matches your target (e.g., ~2%).

### Solver Chain

```
1. OSQP (fastest, warm-started from previous solution)
2. SCS (robust fallback if OSQP fails)  
3. Analytical fallback: w* = μ̂ / (κσ²) clipped to constraints
```

---

## 5. Complete Daily Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│ DAY t                                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. REALIZE PnL                                                         │
│     gross_pnl = w(t-1) · r(t) × booksize                              │
│     (yesterday's weights × today's returns)                             │
│                                                                         │
│  2. UPDATE RISK MODEL                                                   │
│     Feed r(t) into EMA update of factor covariance F and spec var σ²   │
│                                                                         │
│  3. REBUILD FACTOR LOADINGS (every 20 days)                             │
│     Recompute B with fresh market cap, momentum, etc.                   │
│                                                                         │
│  4. STAGE 1: COMBINE ALPHAS (Ridge Regression)                          │
│     a. Get each alpha signal fₐ(t-1) (delay=1: yesterday's signal)    │
│     b. Update rolling X,y matrices with (fₐ(t-1), r(t)) observation   │
│     c. Solve ridge: w* = (X'X + λI)⁻¹ X'y                            │
│     d. Predict: μ̂(t) = Σₐ wₐ* × fₐ(t)                               │
│     e. Apply universe mask: μ̂ᵢ = 0 for stocks not in universe          │
│                                                                         │
│  5. STAGE 2: OPTIMIZE PORTFOLIO (QP)                                    │
│     a. Get risk matrices: Q, σ² from risk model                        │
│     b. Get ADV for position limits                                      │
│     c. Solve QP: w*(t) = argmin [risk - μ̂'w + tcost]                  │
│        s.t. ||w||₁ ≤ 1, Σw = 0, |wᵢ| ≤ w_max                        │
│                                                                         │
│  6. COMPUTE TRANSACTION COSTS                                           │
│     trades = w*(t) - w(t-1)                                            │
│     tcost = slippage + impact + borrow                                  │
│                                                                         │
│  7. RECORD & ADVANCE                                                    │
│     net_pnl = gross_pnl - tcost                                        │
│     w(t-1) ← w*(t)                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Risk Model

### For Equities (Isichenko Ch 4)

Factor model: C = BFB' + diag(σ²)

16 factors:
- 11 GICS sector dummies
- Size: z-score of log(market_cap)
- Value: z-score of book_to_market
- Momentum: z-score of 12-month return
- Volatility: z-score of 60-day historical vol  
- Leverage: z-score of debt_to_equity

Factor covariance F and specific variance σ² updated via EMA (halflife=60 days).

### For Crypto (Simplified)

No GICS sectors. Two options:
1. **Rolling PCA**: Extract K statistical factors from returns covariance (e.g., K=5)
2. **L2 regularization**: Skip the factor model entirely, use ½κ||w||² as risk (equivalent to 
   assuming all stocks have equal variance and zero correlation → forces diversification)

Dollar neutrality constraint substitutes for sector neutrality.

---

## 7. Fee & Transaction Cost Model

| Cost | Formula | In Optimizer? | In Backtest? |
|------|---------|:---:|:---:|
| Linear slippage | c × |Δw|₁ | ✅ | ✅ |
| Market impact (Almgren-Chriss) | η × σ × √(|Δw|/ADV) × |Δw| | ✅ | ✅ |
| Short borrow | β × |w_short|₁ | ❌ | ✅ |
| Turnover penalty | λ × ||Δw||² | ✅ | ❌ (implicit) |

For crypto: use linear fee only (e.g., 5 bps taker fee including slippage). No borrow cost 
for perpetual futures. Impact negligible for TOP50 at $2M book.

---

## 8. Configuration Reference

| Parameter | Equities Default | Crypto Default | Description |
|-----------|:---:|:---:|---|
| booksize | $20M | $2M | Gross market value |
| max_position_pct_gmv | 1-2% | 5% | Max single position as % of book |
| max_position_pct_adv | 5% | N/A | Max single position as % of ADV |
| risk_aversion (κ) | 500 (wt space) | TBD | Risk penalty coefficient |
| dollar_neutral | True | True | Σw = 0 |
| sector_neutral | True (soft) | False | Crypto has no sectors |
| slippage_bps | 3 | 5 | Linear cost per trade |
| ridge_window | 252 | 252 | Rolling window for ridge regression |
| ridge_lambda | auto | auto | Ridge penalty (cross-validated or auto) |
| ema_halflife_risk | 60 | 60 | Risk model EMA decay |
| warmup_days | 120 | 120 | Days to calibrate before trading |

---

## 9. Current Implementation vs Book

### What's Correct ✅

| Component | Status |
|-----------|--------|
| QP formulation (weight space) | ✅ Matches Eq 6.6 |
| Factor risk model (equities) | ✅ BFB' + diag(σ²) with EMA |
| Dollar neutrality constraint | ✅ Hard constraint in QP |
| Position limits | ✅ Min of GMV% and ADV% |
| Linear transaction cost in QP | ✅ Correct |
| OSQP solver with warm start | ✅ Fast sequential solves |

### What's Wrong ❌

| Component | Current (Broken) | Should Be (Book) |
|-----------|-----------------|-------------------|
| **Alpha combination** | Rank-average + Grinold scaling hack | **Ridge regression on forward returns** |
| **Alpha scaling** | rank × cs_vol (arbitrary, ~20x too large) | **Regression coefficients (return units)** |
| **Risk aversion (crypto)** | κ=2 (arbitrary) | **Calibrate to target vol or grid search** |
| **Alpha decorrelation** | None (signals double-count) | **Ridge naturally decorrelates** |
| **MSE weighting** | Not implemented | **Ridge 1/MSE weighting (Eq 2.38)** |

### Root Cause of QP Failure

**The entire problem traces to Stage 1 being wrong.** Without proper ridge regression:
1. The combined signal scale is arbitrary → κ can't be calibrated
2. Correlated signals double-count → concentrated positions on popular themes
3. Anti-predictive signals aren't removed → noise added to forecast
4. The QP can't balance risk and return because the return forecast is meaningless

**Fix**: Implement rolling ridge regression. The output will be a proper return forecast,
and the QP will work as designed.

---

## 10. Known Issues & Future Work

### Critical (Blocking QP Performance)
1. **Implement rolling ridge regression** — Stage 1 alpha combination (Eq 2.38)
2. **Calibrate κ for crypto** — after ridge is correct, sweep κ to find optimal
3. **Auto-calibrate ridge λ** — cross-validation or trace heuristic

### Important (Production Readiness)
4. **PCA risk model for crypto** — statistical factors from returns covariance
5. **Decay sweep** — auto-wrap high-turnover alphas in EMA
6. **Turnover penalty tuning** — match IS and OOS turnover levels

### Nice to Have
7. **Bayesian shrinkage** (Black-Litterman) instead of ridge
8. **Alpha correlation management** — DCC or LASSO for sparse combination
9. **Intraday execution model** — partial fills, VWAP/TWAP

---

## 11. Empirical Results: Crypto (N=50, $2M Book)

> Tested on 209 GP-discovered alphas, Binance TOP50 universe, 5 bps fees.
> Train: 2020-2024, OOS: 2024-present.

### Method Comparison (OOS, delay=1)

| Method | OOS Sharpe | OOS PnL | Turnover |
|:---|:---:|:---:|:---:|
| **Equal-Weight Rank Avg** | **+0.92** | **+$1.81M** | 19.7% |
| Ridge Forecast → Simple Sim | +0.31 | +$620K | 62.5% |
| Ridge → QP+PCA (κ=1) | +0.01 | +$17K | — |
| Ridge → QP+PCA (κ=5) | -0.10 | -$207K | — |
| Ridge → QP+PCA (κ=10-100) | -0.12 to -0.30 | -$233K to -$449K | — |

### Key Finding

**For N=50 crypto, equal-weight rank averaging is the correct approach.**
The QP optimizer destroys value at every kappa level because:
1. N=50 with 5% max weight → only 10 positions per side → no diversification room
2. Ridge adds 3× turnover without proportional forecast improvement
3. The PCA risk model doesn't capture crypto's BTC-dominated structure well enough

This aligns with Isichenko Sec 6.1: "hedged allocation" (`P = A·f`) is sufficient
for small universes where the QP can't diversify beyond the position limits.

