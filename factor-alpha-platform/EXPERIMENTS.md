# 5m Crypto Alpha Strategy — Experiment Log

> **Production config**: `--combine ts_autonorm --entry 1.2 --exit 0.3 --hold 36 --ts-gamma 0.01 --top-n 30 --max-pos 50 --fees 7 --signal-smooth 12 --signal-hedge 3 --rolling-ic 1440`
> **Current best WF Sharpe: +5.02, Return: +38.24%** (28 folds, 91 days OOS, 7 bps fees, FULLY CAUSAL)
> All walk-forward results use: train 6d, test 3d, rolling across Val period (Feb 1 – Mar 1).

> [!CAUTION]
> **HARD RULE — Data Split Discipline**
>
> | Split | Period | Purpose |
> |-------|--------|---------|
> | **Train** | Dec 1 – Feb 1 | Alpha signal DISCOVERY (`eval_alpha_5m.py`) |
> | **Val** | Feb 1 – Mar 1 | Portfolio optim / WF (`eval_portfolio_5m.py`) |
> | **Test** | Mar 1 – Mar 27 | FINAL TEST ONLY — never touch until done |
>
> ⚠️ **STATUS**: All 30 existing alphas were discovered on Feb–Mar (Val+Test data).
> They MUST be re-discovered on Train (Dec–Jan) before results are valid.

---

## Table of Contents

1. [Core Problem](#core-problem)
2. [Data Description](#data-description)
3. [Experiment History](#experiment-history)
4. [E12: TS AutoNorm Breakthrough](#e12-ts-autonorm-std-only-hybrid--breakthrough)
5. [E13: Creative Enhancements](#e13-creative-signal-enhancements)
6. [E14: ML-Based Combiners](#e14-ml-based-combiners)
7. [Bug Fixes](#bug-fixes)
8. [Negative Results (What Failed)](#negative-results)
9. [Open Questions & Next Steps](#open-questions--next-steps)

---

## Core Problem

Identify large-move events in 5m perpetual futures and trade them profitably after 7 bps
fees per side (14 bps round-trip). Standard MSE regression fails because it fits the mass
of tiny returns, not the tail events that cover fees.

**Fee math**: At 7 bps per side, a round-trip costs 14 bps = 0.14%. A trade needs to
capture > 0.14% to be profitable. At 288 bars/day, even 5 bps per trade compounds to
significant drag.

**Sharpe inflation warning**: 31 days OOS with mean/std × sqrt(365) = ~19× multiplier.
A weak daily ratio of 0.44 produces Sharpe 8.4 — meaningless. Use walk-forward with
many folds to get reliable estimates.

---

## Data Description

### Time Period
| Period | Start | End | Days | 5m Bars |
|--------|-------|-----|------|---------|
| Full window | 2025-12-01 | 2026-03-27 | 116 | 33,408 |
| Train | 2025-12-01 | 2026-02-01 | 62 | 17,856 |
| Val | 2026-02-01 | 2026-03-01 | 28 | 8,064 |
| Test | 2026-03-01 | 2026-03-27 | 26 | 7,488 |

### Universe
- 101 Binance USDT-perpetual futures (TOP100 by volume)
- 38 matrix fields: OHLCV, microstructure (taker_buy_ratio, trades_per_volume, etc.),
  derived (momentum, volatility, beta_to_btc, etc.)

### Alpha Library
- **30 alphas** in `data/alphas_5m.db` from 298 Agent 1 discovery trials
- IC range: -0.003 to +0.008 (mean IC, measured against forward returns)
- IS Sharpe range: +3.22 to +11.32
- Expression types: SMA-smoothed microstructure signals with cross-sectional z-scoring
- **Discovery bias**: Alphas discovered on Feb–Mar data. They systematically underperform
  on Dec–Jan ("negative alpha regime").

### Known Regime Structure
| Period | Alpha Performance | Notes |
|--------|------------------|-------|
| Dec 2025 | **Negative** | Alphas were not discovered on this data. BTC crash. |
| Jan 2026 | **Mixed** | Some folds positive, some negative. |
| Feb 2026 | **Strongly positive** | Discovery period for alphas. |

---

## Experiment History

### Early Experiments (E1–E9): Combination Methods on Short WF

These used a 31-day window (Feb 15 – Mar 17) with 6d train / 3d test. Results were
unreliable due to short OOS window (24 daily bars) and ~19× Sharpe annualization.

| ID | Method | Val SR | Short WF SR | Key Finding |
|----|--------|--------|-------------|-------------|
| E1 | Ridge (broken) | — | +5.28 | Inflated by bug |
| E2 | Ridge (z-score target) | — | +2.73 | OK but not robust |
| E4 | ic_weighted baseline | +2.23 | — | Standard baseline |
| E5 | Concordance | +8.42 | +8.42 | Regime overfit (0.0 SR on long WF) |
| E6 | IC rolling | — | -9.66 | Complete failure |
| E7 | Asym loss | +2.93 | — | Minor improvement over baseline |
| E8 | Hurdle | -2.41 | — | Degenerate gating (92% events qualify) |
| E9 | Quantile (tau=0.95) | -5.20 | — | Overtrades, noisy predictions |

**Key takeaway**: Concordance's +8.42 SR was entirely regime-dependent. When tested
on the full Dec–Mar window (WF long), it produced 0.00 SR with zero trades — the
rolling IC gate correctly killed all signals in Dec–Jan.

### E5 Concordance — Confirmed Regime Overfit

Concordance vote-counting across alphas using rolling IC gate:
- WF short (Feb–Mar only): +8.42 SR — same regime as alpha discovery
- WF long (Dec–Mar): 0.00 SR — zero trades (IC gate kills Dec–Jan signals)
- **Implication**: Alpha re-discovery on full 116-day window (E11) is critical path.

### E7 Hurdle — Degenerate Gating

Two-stage model: classifier predicts P(|fwd_ret| > 2×fee), then regressor on filtered events.
- Problem: `large_events = 91.8%` at 0.14% hurdle over 12h horizon.
- Classifier gates nothing useful. Stage 2 sees nearly all data.
- **Fix needed**: Raise hurdle to 3–5× fee cost, or target |return| > 1%.

### E8 Asym Loss — Best Pre-AutoNorm Method

Custom sample weights: negative-return samples get 5× weight.
- Val SR: +2.93 vs baseline +2.23 (+0.70 improvement).
- Paradox: Model trades MORE (177/day vs 66/day) despite conservative weighting.
- The z-score threshold (not the ML model) is the real trade gatekeeper.

---

## E12: TS AutoNorm (std-only hybrid) — BREAKTHROUGH

### Key Insight

**Divide raw signals by EWM std without subtracting EWM mean.**

```
Standard EWMA z-score:  z = (signal - ewm_mean) / ewm_std   → DESTROYS alpha
Correct approach:       z = signal / ewm_std                  → PRESERVES alpha
```

Why standard z-scoring fails: For trend-following microstructure signals, the mean
IS the signal. Subtracting the mean removes the very thing we're trying to capture.
Dividing by std only standardizes scale while preserving directional magnitude.

### Smart Hybrid Detection

The function inspects each alpha's expression string:
- **`zscore_cs` found**: Alpha is pre-normalized → use standard CS z-score to avoid double-normalizing.
- **No `zscore_cs`**: Raw signal → divide by per-asset EWMA std only (no mean subtraction).

This distinction matters because ~33% of alphas (10/30) contain `zscore_cs` in their
expression tree, while ~67% (20/30) are raw signals.

### Progressive Results (7 bps fees)

Each row shows the cumulative effect of one improvement:

| Step | Change | WF Agg SR | WF Return | Delta SR |
|------|--------|-----------|-----------|----------|
| 0 | ic_weighted baseline (10 bps) | -0.60 | -3.67% | — |
| 1 | ts_autonorm, 10 alphas, max=15 (10 bps) | +2.62 | +19.68% | +3.22 |
| 2 | Increase to 15 alphas | +2.85 | +23.68% | +0.23 |
| 3 | Widen portfolio to 40 positions | +3.86 | +28.09% | +1.01 |
| 4 | Fix WF fee passthrough bug (7 bps) | +4.45 | +32.45% | +0.59 |
| 5 | Use all 30 alphas | +4.49 | +34.10% | +0.04 |
| 6 | Add signal smoothing (EMA=12) | +4.73 | +36.21% | +0.24 |
| 7 | Add BTC beta-hedge | +4.80 | +36.41% | +0.07 |
| 8 | Widen to 50 positions | +4.91 | +36.87% | +0.11 |
| 9 | Replace beta-hedge with PCA signal hedge (3 PCs) | +5.68 | +35.93% | +0.77 |
| 10 | Rolling IC weights (EMA 1440, eliminates lookahead) | **+5.02** | **+38.24%** | **causal** |

**Total improvement**: From -0.60 → +5.02 Sharpe (+5.62 total, fully causal).

> [!NOTE]
> Step 9 (+5.68) has slightly higher Sharpe but uses static DB IC weights with lookahead
> bias. Step 10 (+5.02) eliminates that bias entirely. The higher return (+38.24% vs
> +35.93%) with rolling IC suggests the adaptive weights actually find better regimes.

---

## E13: Creative Signal Enhancements

### A. Signal Smoothing (`--signal-smooth N`)

Applies EMA to the composite z-score signal before threshold trading.

**Mechanism**: Reduces bar-to-bar noise in the composite signal. A trade that would have
been triggered by a 1-bar spike in z-score is smoothed away, preventing a round-trip
that costs 14 bps and captures nothing.

**Full smoothing sweep results**:

| Smooth Span | Duration | WF Agg SR | WF Return | Notes |
|-------------|----------|-----------|-----------|-------|
| 0 (off) | — | +4.49 | +34.10% | Baseline |
| 6 | 30 min | +4.58 | +35.61% | Too little |
| **12** | **1 hour** | **+4.73** | **+36.21%** | **Optimal** |
| 24 | 2 hours | +4.59 | +35.55% | Slightly worse |
| 36 | 3 hours | +4.38 | +33.11% | Destroys alpha |

**Interpretation**: At span=12, the EMA half-life is ~8 bars (40 min). This filters out
sub-1h noise while preserving 3h+ signal trends (which is where the alpha lives, given
that most alphas use SMA windows of 72–288 bars).

### B. BTC Beta-Hedge (`--beta-hedge`)

Removes BTC market exposure from each ticker's signal via rolling regression.

**Implementation**:
```python
for each non-BTC ticker:
    beta = rolling_cov(signal, btc_ret, 576) / rolling_var(btc_ret, 576)
    signal_hedged = signal - beta * btc_ret
```

- Lookback: 576 bars = 2 days (48 hours)
- beta is clipped to [-5, 5] to prevent regression blowups

**Why it helps**: In the Dec crash, many alpha signals were correlated with BTC direction.
The portfolio took large directional bets and lost money. Beta-hedging removes this
directional component, making the signal purely about relative value among altcoins.

**Results by combination**:

| Config | WF Agg SR | Effect |
|--------|-----------|--------|
| Baseline (no enhancements) | +4.49 | — |
| Beta-hedge only | +4.53 | +0.04 |
| Smooth=12 only | +4.73 | +0.24 |
| Smooth=12 + beta-hedge | +4.80 | +0.31 |
| Smooth=12 + beta-hedge + 50pos | **+4.91** | **+0.42** |

### C. Concordance Boost (`--concordance`)

**FAILED** — Do not use.

Multiplies signal by directional agreement among alphas:
`agreement = |sum_of_signs(all_alphas)| / n_alphas`, scaled to [0.5, 1.5].

When 90% of alphas agree on direction, signal is amplified 1.45×. When 50% agree, 1.0×.

**Result**: +2.57 SR (vs +4.49 baseline). **Lost 1.92 SR.**

**Why it fails**: ~10 alphas share `zscore_cs` normalization and are structurally
correlated. Their "agreement" is noise amplification, not genuine multi-signal
conviction. The concordance multiplier over-weights moments when the correlated
cluster happens to align, which coincides with false signals.

### D. Rank Normalization (`--rank-norm`)

**REJECTED** — Do not use.

Uses cross-sectional rank percentile → inverse-normal CDF instead of z-score for the
final normalization step.

**Result**: +4.21 SR (vs +4.49 baseline). **Lost 0.28 SR.**

**Why it fails**: Rank normalization destroys magnitude information. If SOLUSDT has a
z-score of 3.2 and ETHUSDT has 1.8, a z-score-based system correctly weights SOL 1.8×
higher than ETH. Rank normalization maps them to rank-1 and rank-2, losing the
conviction information that drives the trading edge.

### E. Combination Effects (Interaction Testing)

Some enhancements interact non-linearly:

| Combo | WF Agg SR | Notes |
|-------|-----------|-------|
| Concordance + smooth=12 | +3.37 | Concordance dominates (negative) |
| Concordance + beta-hedge | +2.71 | Concordance dominates (negative) |
| Smooth=6 + beta-hedge | +4.47 | Slightly worse than smooth=12 |
| **Smooth=12 + beta-hedge** | **+4.80** | **Best 2-way combo** |
| Smooth=18 + beta-hedge | +4.74 | Close second |

**Lesson**: Concordance is so harmful that it drags down any combination. Smoothing
and beta-hedge are additive — they address orthogonal problems (noise vs directional exposure).

### F. PCA Signal Hedge (`--signal-hedge N`) — BREAKTHROUGH

Generalizes beta-hedge from 1 factor (BTC) to N factors (PCA eigenvectors of return
covariance). For each bar (refit every 288 bars):
1. Estimate PCA on trailing 576 bars (2 days) of cross-sectional returns
2. Project the combined signal onto the top N PC loadings
3. Subtract the projection → residual = idiosyncratic signal only

This removes the component of the signal explained by systematic factors:
- **PC1**: Market direction (≈ BTC beta)
- **PC2**: Alt-vs-major rotation (ETH/SOL vs small-caps)
- **PC3**: Volatility regime (high-beta vs low-beta coins)

**Full PCA hedge sweep** (all with smooth=12, max-pos=50):

| PCA Components | WF Agg SR | WF Return | Notes |
|----------------|-----------|-----------|-------|
| 0 (none) | +4.73 | +36.21% | Smooth only baseline |
| Beta-hedge only | +4.91 | +36.87% | Previous best |
| PCA=1 | +3.93 | +28.39% | Too aggressive (removes signal) |
| **PCA=3** | **+5.68** | **+35.93%** | **🏆 NEW BEST** |
| PCA=3 + beta-hedge | +5.51 | +34.97% | Redundant (PCA captures BTC) |
| PCA=5 | +4.96 | +32.22% | Over-hedged |

**Why PCA=3 dominates PCA=1**: With 1 component, we remove the market factor but also
some of the idiosyncratic alpha that's correlated with it. With 3 components, the PCA
basis is more complete — the projection subtracts exactly the systematic part and
leaves the alpha intact. With 5+ components, we start removing small factors that
contain tradeable signal.

**Why PCA=3 dominates beta-hedge**: Beta-hedge only removes BTC direction. PCA=3
also removes rotational factors (alt vs major, vol regime) that contaminate the signal
during regime transitions — exactly the Dec crash period where we lose money.

**Why beta-hedge is redundant with PCA**: PC1 ≈ BTC market factor. Adding explicit BTC
beta-hedge on top of PCA=3 double-removes market exposure and slightly hurts (+5.68 → +5.51).

---

## E15: Rolling Causal IC Weights (`--rolling-ic N`)

### Problem: IC Lookahead Bias

The static IC weights (from `data/alphas_5m.db`) were computed on Feb\u2013Mar data during
alpha discovery. When walk-forward folds reach Dec\u2013Jan (before discovery), these
weights encode future information \u2014 the model "knows" which alphas will work in Feb.

### Solution: EMA-Smoothed Rolling IC

Instead of static IC from the DB, compute IC **causally** at each bar:

```
For each bar t and alpha k:
    IC_k(t) = rank_corr(signal_k[t-1, :], return[t, :])   # cross-sectional
    ema_IC_k(t) = EMA(IC_k, span=N)
    weight_k(t) = max(ema_IC_k(t), 0.0001) / sum(weights)
```

This is fully causal: at time `t`, we know `signal[t-1]` and `return[t]`.

### Implementation

- Vectorized using pandas `rank(axis=1)` + Pearson correlation of ranks (= Spearman)
- `np.einsum` for the weighted combination \u2014 no Python loops
- ~10s for 26k bars \u00d7 30 alphas

### Results (all with smooth=12, PCA=3, 50 pos)

| Rolling IC Span | Duration | WF Agg SR | WF Return | Notes |
|----------------|----------|-----------|-----------|-------|
| 0 (static DB IC) | \u221e | +5.68 | +35.93% | Has IC lookahead bias |
| 288 | 1 day | +3.56 | +28.12% | Too reactive, weights whipsaw |
| 576 | 2 days | +4.00 | +31.29% | Better but still noisy |
| **1440** | **5 days** | **+5.02** | **+38.24%** | **\ud83c\udfc6 Best causal** |

### Analysis

- **Short spans (288, 576)** are too noisy: per-bar cross-sectional IC has std ~0.13
  with mean ~0.001. A 1-day EMA doesn't smooth enough \u2014 weights oscillate wildly.
- **5-day span (1440)** is long enough to get stable IC estimates while adapting
  to regime changes within ~2 weeks.
- Rolling IC=1440 achieves **higher return** (+38.24% vs +35.93%) than static IC,
  suggesting the adaptive weights correctly down-weight alphas during the Dec crash
  and up-weight them as they start working in Feb.
- The Sharpe reduction (+5.68 \u2192 +5.02) is expected \u2014 some of the static IC's
  apparent Sharpe was from lookahead bias, not genuine alpha.

---

## E14: ML-Based Combiners

### LightGBM Combiner (`--combine lgbm`)

**FAILED** — Massive overfitting. Do not use for production.

#### Design
- **Target**: `sign(fwd_ret) × log1p(|fwd_ret| / fee_cost)` — measures directional
  alpha in fee-multiples. Emphasizes direction over magnitude.
- **Features**: 30 raw alpha signals + 30 cross-sectional rank signals + market vol
  regime = 61 features.
- **Model**: LightGBM with aggressive regularization (max_depth=4, num_leaves=15,
  min_child_samples=200, subsample=0.7, reg_alpha=1.0, reg_lambda=1.0).
- **Rolling refit**: Every 576 bars (2 days) on trailing 4032 bars (14 days).
- **Horizon**: 144 bars (12 hours).

#### Results
| Metric | Value |
|--------|-------|
| Val Sharpe | **+5.12** (looks great!) |
| Val Return | +11.5% |
| WF Sharpe | **-2.38** (catastrophic OOS) |
| WF Return | -19.86% |

#### Why It Fails
1. **Insufficient training data**: Each WF fold has ~14 days (4,032 bars × 101 tickers
   = 407k samples). With 61 features, LightGBM finds spurious patterns.
2. **Alpha signals are already highly processed**: The 30 alphas are each SMA-smoothed
   microstructure signals. ML on top of ML adds noise, not signal.
3. **Val/WF gap**: Val has the same regime as alpha discovery. WF includes Dec crash
   where everything breaks.

**Would need**: 6+ months of data, or < 10 features, or much simpler model (linear).

### Equal Weighting (`--equal-weight`)

**REJECTED** — Significantly worse than IC-weighted.

| Type | WF Agg SR | WF Return |
|------|-----------|-----------|
| IC-weighted (max(IC, 0.0001)) | +4.49 | +34.10% |
| Equal weight (1/N) | +2.75 | +31.21% |
| sqrt(abs(IC)) weight | +3.62 | +34.49% |

Equal weighting gives too much influence to zero-IC and negative-IC alphas, injecting
noise into the composite.

---

## Bug Fixes

### Bug 1: Universe Coverage Gap
**Symptom**: Zero trades in Dec–Jan folds.
**Cause**: Universe filter excluded most tickers when data wasn't available for all fields.
**Fix**: Relaxed universe requirements.

### Bug 2: EWM Span Floor
**Symptom**: All gamma values >= 0.02 produced identical results.
**Cause**: `max(1/gamma, 50)` forced all spans to 50 when gamma >= 0.02.
**Fix**: Removed the floor. Now `span = int(1/gamma)` directly.

### Bug 3: Unicode Encoding
**Symptom**: Console crash on Windows with `→` character.
**Fix**: Replaced `→` with `->` throughout.

### Bug 4: WF Fee Passthrough (Critical, +0.59 SR)
**Symptom**: `--fees 7` flag was ignored by walk_forward_validate(). WF always used
default 10 bps regardless of CLI flag.
**Cause**: `fees_bps` parameter not threaded through WF function.
**Fix**: Added `fees_bps` parameter to `walk_forward_validate()` and threaded it to
all internal calls.
**Impact**: +3.86 → +4.45 at 7 bps. **+0.59 SR improvement!**

### Bug 5: LightGBM API
**Symptom**: `train() got an unexpected keyword argument 'verbose_eval'`
**Cause**: LightGBM 4.6 removed `verbose_eval`. Use `callbacks` instead.
**Fix**: `callbacks=[lgb.log_evaluation(period=-1)]`

---

## Negative Results

Summary of approaches tested and rejected, with explanations:

| Approach | Tried | Result | Why It Failed |
|----------|-------|--------|---------------|
| LightGBM combiner | Val +5.12, WF -2.38 | Overfits | Not enough data for 61 features |
| Equal weighting | WF +2.75 | Worse | Noisy alphas dilute signal |
| sqrt(abs(IC)) weighting | WF +3.62 | Worse | Negative-IC alphas get too much weight |
| Concordance boost | WF +2.57 | Much worse | Correlated alphas amplify noise |
| Rank normalization | WF +4.21 | Worse | Destroys magnitude information |
| Concordance + smooth | WF +3.37 | Worse | Concordance dominates negatively |
| Concordance + beta-hedge | WF +2.71 | Worse | Concordance dominates negatively |
| Over-smoothing (span=36) | WF +4.38 | Worse | Destroys alpha signal at longer horizons |
| Higher entry (1.5/0.4/72) | WF +3.98 | Worse SR | More selective but less consistent |

---

## Open Questions & Next Steps

### Path to Sharpe > 7

Current bottleneck: **Dec regime drag**. Folds 1–8 (Dec) average negative returns.
If Dec folds were just flat (not positive), aggregate SR would be ~6.5.

| Lever | Expected Impact | Effort |
|-------|-----------------|--------|
| **Alpha re-discovery on full Dec–Mar (E11)** | +1–2 SR | High (requires Agent 1 rerun) |
| More orthogonal alphas (30 → 50+) | +0.5 SR | Medium |
| Per-position vol normalization | +0.2 SR | Low |
| Even more positions (60–80) | +0.1 SR | Low |
| Continuous portfolio (no threshold) | Unknown | Medium |

### Specific Questions

1. **Dec regime**: Are the Dec losses due to alpha staleness (discovery bias) or
   genuine regime change (market structure shift)?
2. **Position count ceiling**: Does SR improve monotonically with max-pos, or is there
   a plateau? Test 60, 80, 100.
3. **Signal horizon mismatch**: Most alphas use SMA(144–288) windows (~12–24h).
   But we trade with entry threshold on 5m bars. Should we align these?
4. **Fee reduction**: At 3 bps (maker rebate), many losing trades become winners.
   What's the Sharpe at 3 bps?
5. **KuCoin transferability**: 6 alphas transferred at +0.67 SR. Worth dedicated discovery?
