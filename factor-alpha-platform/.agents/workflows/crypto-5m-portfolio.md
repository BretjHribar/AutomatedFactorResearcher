---
description: Run 5m crypto portfolio construction - threshold-based trading with fee optimization
---

# 5m Crypto Portfolio Construction (Agent 2)

You are Agent 2: a portfolio constructor for **5m Binance crypto** alpha signals. You take the alphas discovered by Agent 1 and optimize HOW to trade them with real fees.

**Primary Objective: Maximize Sharpe ratio** after fees.
**Secondary Objective: Maximize net returns** while controlling drawdown.

> [!CAUTION]
> ## HARD RULE: Data Split Discipline
>
> | Split | Period | Purpose | Who Uses |
> |-------|--------|---------|----------|
> | **Train** | Dec 1 – Feb 1 | Alpha signal DISCOVERY | `eval_alpha_5m.py` only |
> | **Val** | Feb 1 – Mar 1 | Portfolio optim / signal combination | `eval_portfolio_5m.py` WF |
> | **Test** | Mar 1 – Mar 27 | FINAL TEST ONLY | Never touch until done |
>
> - **NEVER** discover alphas on Val or Test data
> - **NEVER** optimize portfolio params on Train data
> - **NEVER** evaluate Test until ALL research is complete

## Data Configuration

### Splits (116-day window, Dec 2025 – Mar 2026)
| Split       | Start      | End        | Days | Bars   |
|-------------|------------|------------|------|--------|
| **Train**   | 2025-12-01 | 2026-02-01 | 62   | 17,856 |
| **Val**     | 2026-02-01 | 2026-03-01 | 28   | 8,064  |
| **Test**    | 2026-03-01 | 2026-03-27 | 26   | 7,488  |

> **WARNING**: Agent 1 alphas were discovered on the Val+Test window (Feb 1 – Mar 26).
> This means Val performance is IN-SAMPLE for the alphas — focus on walk-forward OOS.

### Universe & Fees
- **Universe**: Binance TOP100 USDT-perp (101 tickers)
- **Interval**: 5-minute bars (288 bars/day)
- **Fees**: 7 bps per side (configurable via `--fees`). Round-trip = 14 bps.
- **Alpha DB**: `data/alphas_5m.db` — currently **30 alphas**, 298 agent trials

---

## STRICT RULES — DO NOT VIOLATE

1. **ONLY use `eval_portfolio_5m.py`** — do NOT modify eval_alpha_5m.py or any src/ files.
2. **Do NOT discover new alphas** — you only optimize combination and thresholds.
3. **NEVER run on `--split test`** — the test set is sacred. Only run it when the USER explicitly says it's OK.
4. **Do NOT look at test data** — only use train and val splits (or --walkforward).
5. **Alphas are fixed** — you combine and threshold, never modify expressions.
6. **Avoid grid searches** — they overfit massively. Prefer principled parameter choices.

---

## Architecture Overview

```
Alpha DB (30 signals)
    |
    v
Signal Evaluation (evaluate_expression per alpha)
    |
    v
Per-Alpha Normalization (ts_autonorm hybrid: CS z-score or EWM std-only)
    |
    v
Signal Combination (IC-weighted average or equal-weight)
    |
    v
Creative Enhancements (signal smoothing, beta-hedge, concordance, rank-norm)
    |
    v
Final Cross-Sectional Z-Score (for threshold compatibility)
    |
    v
Threshold Strategy (entry/exit/hold logic with conviction sizing)
    |
    v
Portfolio with Vol Targeting & Drawdown Brake
```

---

## Signal Combination Methods

### ts_autonorm (BEST METHOD — use this)

The **Time-Series Auto-Normalization** method is the production standard. It handles
two types of alpha signals differently:

#### Hybrid Normalization Logic

For each alpha expression, the code inspects whether `zscore_cs` appears in the string:

**Path A — Pre-normalized alphas** (`zscore_cs` in expression):
```
z = (signal - cs_mean) / cs_std    # standard cross-sectional z-score
clip to [-3, 3]
```
These alphas already contain cross-sectional normalization in their expression
(e.g. `zscore_cs(sma(taker_buy_ratio, 72))`), so we just do a vanilla CS z-score
to avoid double-normalizing.

**Path B — Raw alphas** (no `zscore_cs` in expression):
```
z = signal / ewm_std(span=1/gamma)    # divide by EWMA std only
clip to [-5, 5]
```
This is the key insight: **dividing by EWM std WITHOUT subtracting EWM mean** preserves
the directional trend of the signal. Standard z-scoring `(x - mean) / std` destroys
the alpha by detrending.

**Why it matters**: If ALL coins have elevated taker_buy_ratio, CS z-scoring sees no
signal (everything cancels). But std-only normalization preserves the absolute magnitude
while standardizing scale across assets.

#### IC Weighting

Signals are combined with IC (information coefficient) weights from the discovery DB:
```python
weight[alpha_id] = max(ic_mean, 0.0001)
```
This gives near-zero weight to negative-IC alphas while still including them in the
composite. The `0.0001` floor means they contribute negligibly but don't sabotage.

**Why not equal weight**: Equal weighting (+2.75 SR) is much worse than IC-weighted
(+4.49 SR) because it gives too much influence to noisy/zero-IC signals.

**Why not sqrt(abs(IC))**: Tested at +3.62 SR. Giving more weight to negative-IC
alphas adds noise via high-variance folds.

#### Control Parameter: `--ts-gamma`

Controls the EWMA decay rate for raw signal normalization:
- `0.01` (default) → 100-bar window (~8.3 hours). **Best tested value.**
- `0.005` → 200-bar window, smoother but slower to adapt.
- `0.02` → 50-bar window, noisier but more reactive.

### Other Combination Methods (for reference)

| Method         | Description                                         | Status   |
|----------------|-----------------------------------------------------|----------|
| `equal`        | Simple average of CS z-scores                       | Baseline |
| `ic_weighted`  | Weight by train-set IC from DB                      | Superseded by ts_autonorm |
| `rank`         | Average of CS percentile ranks                      | Worse |
| `lgbm`         | LightGBM tree-based combiner                        | OVERFITS OOS |
| `ml_ridge`     | Ridge regression with target engineering             | Legacy |
| `concordance`  | Vote-counting across alphas                         | Don't use |
| Others         | `ml_gbt`, `ml_gbt_class`, `hurdle`, `asym_loss`, `quantile` | Experimental |

---

## Creative Signal Enhancements

These are post-combination enhancements applied to the composite signal before trading.

### Signal Smoothing (`--signal-smooth N`)

Applies an EMA (exponential moving average) to the composite signal to reduce bar-to-bar noise.

**Why it works**: The composite z-score is noisy on a 5-minute basis. A 12-bar EMA
(1 hour) smooths out high-frequency noise while preserving the directional trend.
This reduces whipsaw entries/exits which are the primary source of fee drag.

| Smooth Span | Approximate Duration | WF Sharpe | Effect |
|-------------|---------------------|-----------|--------|
| 0 (off)     | —                   | +4.49     | Baseline |
| 6           | 30 min              | +4.58     | Mild improvement |
| **12**      | **1 hour**          | **+4.73** | **Optimal** |
| 24          | 2 hours             | +4.59     | Slightly worse |
| 36          | 3 hours             | +4.38     | Too much — destroys alpha |

**Recommendation**: Use `--signal-smooth 12` (default off).

### Beta Hedge (`--beta-hedge`)

Regresses out BTC market beta from each ticker's signal using a rolling window:
```
For each non-BTC ticker:
    beta = rolling_cov(signal, btc_returns, 576) / rolling_var(btc_returns, 576)
    hedged_signal = signal - beta * btc_returns
```

**Why it works**: Many alphas are correlated with BTC direction. During violent BTC
moves (e.g., Dec crash), the portfolio takes large directional bets. Beta-hedging
makes the signal market-neutral, reducing drawdowns in crash regimes.

- Lookback: 576 bars (2 days)
- Standalone effect: +0.04 SR
- Combined with smoothing: +0.07 SR additional
- **Note**: Superseded by `--signal-hedge 3` which captures BTC plus more factors.

### PCA Signal Hedge (`--signal-hedge N`) — **BEST HEDGE**

Generalizes beta-hedge from 1 factor (BTC) to N factors (PCA eigenvectors of the
return covariance matrix). Projects the combined signal onto the top N PCs and
subtracts, leaving only idiosyncratic (alpha-specific) signal.

The 3 eigenvectors capture:
- **PC1**: Market direction (≈ BTC beta)
- **PC2**: Alt-vs-major rotation (ETH/SOL vs small-caps)
- **PC3**: Volatility regime (high-beta vs low-beta coins)

| PCA Components | WF Sharpe | Notes |
|----------------|-----------|-------|
| 0 (none) | +4.73 | Smooth only |
| 1 | +3.93 | Too aggressive |
| **3** | **+5.68** | **Optimal** |
| 5 | +4.96 | Over-hedged |

- **Recommendation**: Use `--signal-hedge 3`.
- PCA is refit every 288 bars (1 day) on trailing 576 bars (2 days).
- Makes `--beta-hedge` redundant (PC1 ≈ BTC factor).

### Concordance Boost (`--concordance`)

Multiplies signal by alpha agreement score: `signal * (0.5 + |sum_of_signs| / n_alphas)`.

**DO NOT USE**: Hurts performance dramatically (+4.49 -> +2.57 SR). Root cause: many
alphas are correlated, so "agreement" just amplifies the dominant cluster's noise
instead of providing genuine conviction.

### Rank Normalization (`--rank-norm`)

Uses cross-sectional rank percentile -> inverse-normal CDF instead of z-score for
final normalization.

**DO NOT USE**: Hurts performance (+4.49 -> +4.21 SR). Destroys magnitudethe ranking loses
how *strong* a signal is — only preserves the order.

---

## Entry/Exit Logic

### Threshold Strategy (default)

```
ENTRY:   composite z-score > entry_threshold  → go LONG
         composite z-score < -entry_threshold → go SHORT

EXIT:    |composite z-score| < exit_threshold → flatten position

EMERGENCY EXIT: Signal reverses past -exit_threshold → force exit
                (overrides hold timer to prevent holding losing positions)

MIN HOLD: --hold N bars (prevents whipsawing)
          Default: 36 bars = 3 hours at 7 bps fees
```

### Position Sizing
- **conviction** (default): Weight proportional to `|signal_strength|`. Higher conviction = more capital.
- **equal**: Traditional 1/N equal weight across all active positions.

### Position Limits
- `--max-pos N`: Cap concurrent positions. When more symbols qualify, keep top N by signal strength.
  - **Best tested: 50** (more diversification = higher Sharpe)
- `--top-n N`: Only use top N alphas by IC from DB.
  - **Best tested: 30** (use all alphas; negative-IC ones get near-zero weight anyway)

### Risk Controls
- **Vol Targeting**: Scales portfolio size to target annualized volatility. Enabled by default.
- **Drawdown Brake**: Reduces position sizes during drawdowns. Enabled by default.

---

## Walk-Forward Validation

**Always run walk-forward first** to get a realistic OOS estimate. Never trust static
val optimization alone.

### Current Best Configuration (billion_alphas — DEFAULT)

**`billion_alphas`** (Kakushadze & Yu 2016, "How to Combine a Billion Alphas") is the production default.
It uses regression-based combination: projects expected alpha returns onto a low-rank factor space,
residualizes, and computes L1-normalized weights causally using only prior daily alpha returns.
Requires N (alphas) >> M (lookback days) — use `--ba-lookback 15` with ~27+ alphas.

> **IMPORTANT**: When switching to a new universe, ALWAYS run val optimization first to find the
> correct entry/exit thresholds. Do NOT reuse TOP100-tuned params (1.2/0.3) on TOP50.

#### Step 1 — Optimize thresholds on val (NEW UNIVERSE ONLY)
// turbo
0. `python eval_portfolio_5m.py --split val --optimize --combine billion_alphas --hold 36 --top-n 30 --max-pos 50 --ts-gamma 0.01 --fees 7 --signal-smooth 12 --ba-lookback 15 --universe TOP100`

This grid searches entry/exit on the val split. Use the resulting best params for steps 1-3.

#### Step 1 — Walk-Forward (primary evaluation)
// turbo
1. `python eval_portfolio_5m.py --walkforward --combine billion_alphas --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 --ts-gamma 0.01 --fees 7 --signal-smooth 12 --ba-lookback 15 --universe TOP100`

Expected result: **+10.17 Aggregate OOS Sharpe, +117.43% OOS Return** (28 folds, 84 days, fully causal)
Fold win rate: **89% (25/28)** | Profit Factor: **4.18** | Max DD: -11.30%
Test OOS (Mar 1-27): **+5.99 Sharpe, +19.50% Return** (true holdout, never touched during research)

#### Step 2 — Val Quick-Check
// turbo
2. `python eval_portfolio_5m.py --split val --combine billion_alphas --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 --ts-gamma 0.01 --fees 7 --signal-smooth 12 --ba-lookback 15 --universe TOP100`

Fast (~1 min) check on val split. Not definitive but useful for quick iteration.

#### Step 3 — Train Consistency Check
// turbo
3. `python eval_portfolio_5m.py --split train --combine billion_alphas --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 --ts-gamma 0.01 --fees 7 --signal-smooth 12 --ba-lookback 15 --universe TOP100`

Check strategy works on train (Dec-Jan). Low/negative Sharpe here expected due to
Dec regime, but should not be catastrophic (< -5 SR is a red flag).

#### Legacy: ts_autonorm (superseded)
// turbo
1b. `python eval_portfolio_5m.py --walkforward --combine ts_autonorm --entry 1.2 --exit 0.3 --hold 36 --top-n 30 --max-pos 50 --ts-gamma 0.01 --fees 7 --signal-smooth 12 --rolling-return 1440 --universe TOP100`

Superseded by billion_alphas. Previous best: +9.57 WF SR, +109.83% return, +6.07 test SR.

---

## CLI Reference

### Core Arguments
```
--split {train,val,test,trainval}      Data split to run on (default: val)
--combine {ts_autonorm,equal,...}      Signal combination method (default: ic_weighted)
--entry FLOAT                          Entry threshold in z-score units (default: auto)
--exit  FLOAT                          Exit threshold (default: auto)
--hold  INT                            Min hold period in bars (default: 36 = 3h)
--fees  FLOAT                          Fees in bps per side (default: 10)
```

### Signal Enhancement Arguments
```
--ts-gamma FLOAT        EWMA decay for ts_autonorm normalization (default: 0.01)
--signal-smooth INT     EMA span for composite smoothing. 0=off, 12=1h (BEST)
--rolling-return INT    Rolling return EMA span for causal P&L weighting. 0=off, 1440=BEST
                        Alphas with negative rolling return get ZERO weight.
--rolling-ic INT        Rolling IC EMA span for causal IC weighting. 0=off (SUPERSEDED by rolling-return)
--signal-hedge INT      PCA signal hedge: remove top N eigenvectors. 0=off (DON'T USE with rolling-return)
--beta-hedge            Regress out BTC beta from signal (SUPERSEDED by signal-hedge)
--concordance           Boost signal when alphas agree on direction (DON'T USE)
--rank-norm             Use rank percentile instead of z-score (DON'T USE)
--equal-weight          Equal weighting instead of IC weighting (DON'T USE)
```

### Portfolio Arguments
```
--top-n INT             Use only top N alphas by IC (default: all)
--max-pos INT           Max concurrent positions, 0=unlimited (best: 50)
--sizing {equal,conviction}  Position sizing mode (default: conviction)
--top-liquid INT        Only use top N tickers by quote volume (0=all)
--no-vol-target         Disable volatility targeting
--no-dd-brake           Disable drawdown brake
```

### Validation Arguments
```
--walkforward           Run walk-forward cross-validation
--wf-train-days INT     WF train window in days (default: 6)
--wf-test-days INT      WF test window in days (default: 3)
--optimize              Grid search thresholds on val
--objective {sharpe,return,ic}  Optimization objective (default: sharpe)
--compare               Compare all combination methods
```

### Advanced Arguments
```
--strategy {threshold,continuous}  Portfolio strategy type (default: threshold)
--trade-buffer FLOAT    Min weight change to trade (continuous mode)
--pos-smooth FLOAT      Position smoothing EMA (continuous mode)
--exchange {binance,kucoin}  Data source (default: binance)
--alpha-ids IDS         Comma-separated alpha IDs (e.g. '1,6,9,11')
--pca INT               PCA neutralization of signal (0=off)
--pca-hedge FLOAT       PCA risk hedge strength
--target-horizon INT    ML Ridge forward return horizon (default: 36)
--enriched              Enriched features for ML Ridge
```

---

## Current Production Performance

### Walk-Forward Summary — billion_alphas (28 folds, Dec 7 - Feb 28, 7 bps, TOP100)

| Metric | Value |
|--------|-------|
| Aggregate OOS Sharpe | **+10.17** |
| Total OOS Return | **+117.43%** |
| Mean Fold OOS Sharpe | +28.2 |
| Fold Win Rate | **89% (25/28)** |
| Daily Win Rate | 68% (57/84) |
| Profit Factor | **4.18** |
| Max Drawdown | -11.30% |
| Calmar Ratio | 10.39 |
| Avg Hold | 24.7 hours |
| Lookahead Bias | **None** (causal regression, retrain daily) |

### Test Holdout (Mar 1-27, true OOS, never touched during research)

| Metric | Value |
|--------|-------|
| Sharpe | **+5.99** |
| Return | **+19.50%** (26 days) |
| Max Drawdown | -8.04% |
| Trades | 977 (28.8h avg hold) |
| Avg Positions | 24.3 symbols |

### Configuration (billion_alphas — PRODUCTION DEFAULT)
```
--combine billion_alphas --entry 1.2 --exit 0.3 --hold 36 --ts-gamma 0.01
--top-n 30 --max-pos 50 --fees 7 --signal-smooth 12 --ba-lookback 15
--universe TOP100
```

### Combiner Comparison (TOP100, 27 alphas, same params)

| Combiner | WF SR | WF Return | Fold Win% | Test SR | Test Return |
|----------|-------|-----------|-----------|---------|-------------|
| **billion_alphas** | **+10.17** | **+117.4%** | **89%** | **+5.99** | **+19.5%** |
| ts_autonorm | +9.57 | +109.8% | 79% | +6.07 | +25.0% |

### Universe Comparison (billion_alphas, same params)

| Universe | N Alphas | WF SR | Test SR | Note |
|----------|----------|-------|---------|------|
| **TOP100** | 27 | **+10.17** | **+5.99** | **Production** |
| TOP50 | 32 | +3.26\* | -6.03\* | Entry/exit NOT optimized for TOP50 |

\* TOP50 used TOP100-tuned entry/exit (1.2/0.3) without re-optimization — invalid comparison.

---

## Tips & Best Practices

1. **Use walk-forward as ground truth** — val Sharpe can be misleading.
2. **billion_alphas is the default** — Kakushadze & Yu 2016 regression-based combination. Use `--ba-lookback 15`.
3. **Always optimize thresholds on val for new universes** — Entry/exit tuned for TOP100 do NOT transfer to TOP50/TOP20.
4. **Signal smoothing = free lung** — `--signal-smooth 12` improves SR with no downside.
5. **billion_alphas is self-weighting** — no `--rolling-return` needed; the regression causally weights by recent alpha returns.
6. **More positions = higher Sharpe** — 50pos > 40pos > 15pos.
7. **TOP100 > TOP50 > TOP20** — bigger universe = better diversification = higher risk-adjusted returns.
8. **LightGBM doesn't work here** — overfits badly with limited WF training data.
9. **Concordance hurts** — don't use it. Correlated alphas create false conviction.
10. **Rank normalization hurts** — destroys magnitude information.
11. **ts_autonorm + rolling-return** — legacy method, superseded by billion_alphas for WF but comparable on test.
12. **Fee sensitivity**: At 7 bps, fee drag is ~8% of gross returns with billion_alphas config.
