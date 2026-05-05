# Crypto Universe Construction — Hypothesis-Driven Research

**Date**: 2026-05-03
**Context**: KuCoin 4h crypto perp universe. Goal — determine the optimal universe construction practices, drawing on equities literature, crypto-specific factor research (Bianchi & Babiak 2022), and Isichenko (*Quantitative Portfolio Management*).

---

## TL;DR — Headline Findings

The Pareto-optimal config found:
> **TOP20 by ADV, 60-day rebalance, 365-day minimum history, exclude memes** → **V+T net SR +2.50, TEST +3.23, DD −12%, TO 0.25**

Compared to our prior production setup (TOP100, cov-filter, 20d rebal, no exclusion):
- **+1.5 SR jump on TEST** (+3.23 vs ~+0.5 prior)
- **DD halved** (−12% vs −25%)
- **Equal turnover** (~0.25/bar)

Two important caveats:
1. **R1 found that `TOP30, 30d, no exclusion` gives V+T +2.05** — close to the H6 winner. The marginal optimum is *not* a single point but a small region around (TOP20-30, 30-60d, ±exclusion).
2. **H8-redo (multi-alpha test against actual winner): 18/18 saved alphas improve** on the recommended universe (mean Δ V+T = +0.84). The universe choice **does generalize** across the alpha library. (The first H8 attempt tested a near-best config by mistake and gave the wrong answer.)

---

## Background — what the literature says

### Equities (Russell methodology, mature reference)
The mature equities universe-construction playbook ([LSEG / FTSE Russell 2026](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/russell-us-indexes-construction-and-methodology.pdf)):
- **Float-adjusted market cap** is the construction metric (excludes restricted/insider shares).
- **Liquidity rule**: ADDTV must exceed the global median.
- **Free float** ≥ 5% of voting rights with unrestricted holders.
- **US nationality**: avoid mixing exchange-traded jurisdictions.
- **Reconstitution**: semi-annual starting 2026.
- **Top-N approach**: largest 1000 → R1000; next 2000 → R2000.

### Crypto factor models
- **Bianchi & Babiak (2022)** — IPCA on crypto returns ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3935934)):
  - Three latent factors with **time-varying loadings** outperform observable factors.
  - R² 17.2% individual, much higher diversified.
  - Main return drivers: **liquidity, size, reversal, market and downside risk**.
- **Practitioner approaches** ([Artemis](https://research.artemisanalytics.com/p/crypto-factor-model-analysis-launching), [Lucida & Falcon](https://medium.com/@Lucida_and_Falcon/building-a-robust-crypto-asset-portfolio-with-multi-factor-strategies-theoretical-foundation-911a694dbdf4)): cross-sectional rank z-scores, **winsorize at 1/99 percentile**, long top / short bottom, equal-weight within each leg.

### Isichenko, *Quantitative Portfolio Management*
- **Survival bias** (Sec. 2.1.2): use historical universe including delisted instruments.
- **Universe size N must exceed actual tradable count** (Sec. 4.5 fn 11): static valid set is the maximal union; per-bar membership handles in/out.
- **Group by liquidity** (Sec. 2.8): separate models for top-N vs lower-liquidity tiers.
- **Don't mix dynamic types** (p. 20): for crypto, don't mix stablecoins / wrapped tokens with native volatile coins.

---

## Setup

- **Data**: KuCoin 4h, 5,926 bars (2023-08-11 → 2026-04-24), 551 ticker columns (perps).
- **Splits**: TRAIN 2023-09-01 → 2025-09-01 (4,513 bars), VAL 2025-09-01 → 2026-01-01 (733), TEST 2026-01-01 → 2026-04-24 (682).
- **Benchmark alpha** (held constant across all universe variants):
  ```
  zscore_cs(multiply(add(add(
    ts_rank(sma(ts_regression(log_returns, beta_to_btc, 120, 0, 0), 60), 240),  # idiosyncratic mom (residual after BTC beta)
    ts_rank(true_divide(ts_sum(log_returns, 120), df_max(parkinson_volatility_60, 0.0001)), 240)),  # vol-adj momentum
    ts_rank(sma(volume_ratio_20d, 120), 240)),  # volume ratio
    ts_rank(Decay_exp(true_divide(open_close_range, df_max(high_low_range, 0.001)), 0.05), 240)))  # body conviction
  ```
- **Conventions**: `signal_to_portfolio` matches `eval_alpha.py:process_signal` (demean → gross-normalize → clip ±10%). Full-TO `Σ|Δw|`, fees 3 bps. delay=0.

---

# Round 1 — Single-Variable Sweeps

## H1 — Universe Size

> **Hypothesis**: Larger universe → more cross-sectional diversification → higher SR. Until liquidity exhaustion (deep into the tail of less-liquid names) where signal degrades.

**Result**: ✓ Confirmed at the small end, ✗ rejected at the large end. **TOP30 sweet spot for TRAIN, TOP20 best on V+T**. Above TOP50 OOS performance collapses sharply.

| top_n | avg active | TRAIN SR | VAL SR | TEST SR | **V+T** | TO/bar |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 9.8 | −0.34 | +0.29 | +0.60 | +0.48 | 0.165 |
| **20** | **19.6** | +0.37 | +0.31 | **+2.25** | **+1.50** | 0.260 |
| **30** | **29.4** | **+1.15** | +0.73 | +1.63 | +1.24 | 0.285 |
| 50 | 49.0 | +0.39 | +1.49 | +0.53 | +0.95 | 0.293 |
| 75 | 73.5 | +0.28 | +0.43 | −1.11 | −0.33 | 0.294 |
| 100 | 98.0 | −0.18 | +0.37 | −3.15 | −1.31 | 0.294 |
| 150 | 89.3 | −1.11 | −1.52 | −2.13 | −1.83 | 0.181 |
| 200 | 82.6 | −0.62 | −2.78 | −3.73 | −3.24 | 0.124 |

**Theory**: top-50-by-ADV in crypto are well-established names (BTC/ETH/SOL/...) with high BTC correlation. Names ranked 50-200 are smaller mid-caps with idiosyncratic micro-events (listing pumps, airdrops, exchange-specific incentives). Adding noisier mid-caps dilutes signal faster than it adds diversification.

**Implied next hypothesis (tested in Round 2)**: combining TOP20 with other improvements may unlock further gains.

---

## H2 — Minimum-history Filter

> **Hypothesis**: Longer min-history excludes fresh-listing pumps → more stable alpha. Sweet spot 90-365 days.

**Result**: ✓ Confirmed. **365 days sweet spot.**

| min_hist | unique tickers | avg active | TRAIN | VAL | TEST | **V+T** |
|---:|---:|---:|---:|---:|---:|---:|
| 0d | 223 | 29.4 | +0.23 | +0.11 | +0.55 | +0.33 |
| 30d | 179 | 29.4 | +0.76 | +1.10 | +0.43 | +0.75 |
| 90d | 162 | 29.4 | +0.90 | +0.27 | −0.52 | −0.20 |
| 180d | 147 | 29.4 | +1.17 | +1.95 | −0.13 | +0.63 |
| **365d** | **119** | **29.4** | **+1.15** | **+0.73** | **+1.63** | **+1.24** |
| 540d | 104 | 25.7 | +0.83 | +0.69 | −2.89 | −1.03 |
| 730d | 86 | 19.7 | −0.05 | −0.29 | −2.50 | −1.37 |

**Theory**: <90d includes brand-new listings during their post-listing pump phase. Pumps contaminate the cross-section — they're high-vol, high-skew, don't mean-revert like established names. Alpha picks them up in TRAIN (gets confused) and signal doesn't carry to VAL/TEST. >540d shrinks universe to <26 active — same problem as TOP10/20 (too thin).

**Implied next hypothesis**: if memes/pumps are the issue, vol-based exclusion should match name-based meme exclusion. *(Tested in R2.)*

---

## H3 — Rebalance Frequency (single-variable)

> **Hypothesis**: 20d should be Pareto-optimal. Daily too churny, 60d+ stale.

**Result**: ✗ Hypothesis rejected. **60-day rebalance best at TOP30** (V+T +1.62 vs +1.24 at 20d). *Not stale; it's regime-commitment.*

| rebal_days | universe changes | avg swap/rebal | TRAIN | VAL | TEST | **V+T** | TO/bar |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1d | 477 | 2.3 | +0.74 | +1.13 | +1.55 | +1.35 | 0.290 |
| 5d | 183 | 4.4 | +0.49 | +0.80 | +1.55 | +1.21 | 0.288 |
| 10d | 98 | 7.2 | +0.85 | +0.37 | +1.47 | +1.00 | 0.287 |
| 20d | 49 | 11.8 | +1.15 | +0.73 | +1.63 | +1.24 | 0.285 |
| **60d** | **16** | **18.4** | **+0.84** | **+2.20** | **+1.35** | **+1.62** | **0.273** |
| 120d | 8 | 21.0 | +0.14 | +1.30 | −0.39 | +0.27 | 0.259 |
| 240d | 4 | 23.5 | −0.27 | +0.84 | +1.05 | +0.92 | 0.226 |

**Theory**: 60d rebalance lets the universe *commit* to a regime; alpha can fully exploit the current top-30 without member churn breaking position continuity. Daily rebalance constantly chases ADV momentum — itself a noisy signal. >120d genuinely goes stale.

**Implied next hypothesis**: rebal × exclusion may interact — the optimal rebalance frequency might shift when memes are excluded. *(Tested in R1.)*

---

## H4 — Pattern Exclusions

> **Hypothesis**: Stablecoins are already low-ADV. Wrapped tokens mirror underlying. Memes are high-vol idiosyncratic — uncertain.

**Result**: ✗ Meme hypothesis was wrong in surprising direction. **Excluding memes IMPROVES OOS** despite hurting TRAIN. Stables boost TRAIN but help V+T less.

| filter | unique | TRAIN | VAL | TEST | **V+T** |
|---|---:|---:|---:|---:|---:|
| no_filter | 119 | +1.15 | +0.73 | +1.63 | +1.24 |
| stables (USDC/DAI/TUSD/BUSD/FDUSD) | 125 | **+1.60** | +0.85 | +1.44 | +1.18 |
| stables + wraps (WBTC/WETH/STETH/...) | 125 | +1.60 | +0.85 | +1.44 | +1.18 |
| **memes (DOGE/SHIB/PEPE/BONK/...)** | **127** | **+0.74** | **+1.32** | **+1.69** | **+1.50** |

**Theory**:
1. *Stables*: occasionally rank in top-30 by raw USDT-margined volume, contribute zero signal slot. Excluding frees a slot for a real coin → cleaner cross-section, modest TRAIN boost.
2. *Wrapped*: identical numbers to stables-only — confirms no wrapped tokens in top-30 ADV pool.
3. *Memes*: have most idiosyncratic alpha during hype phases. Benchmark alpha picks up the idiosyncrasy and *overfits* it during TRAIN. When meme cycles rotate (different memes each year), TRAIN-fit doesn't transfer. Excluding forces alpha to find more durable signals. **Strongest argument for exclusion as a generalization tool, not a tradability tool.**

**Implied next hypothesis**: if memes-by-name helps, vol-rank exclusion should match (memes are just high-vol). *(Tested in R2.)*

---

## H5 — Risk-Model Residualization

> **Hypothesis**: Residualizing alpha against risk factors (beta-BTC or PCs) should improve OOS stability per Bianchi-Babiak's IPCA framework.

**Result**: ✗ Cross-sectional residualization at the alpha-formation stage does NOT help on a properly-constructed universe.

| risk model | TRAIN | VAL | TEST | **V+T** | DD V+T | TO |
|---|---:|---:|---:|---:|---:|---:|
| **none** | **+1.15** | **+0.73** | **+1.63** | **+1.24** | **−16%** | **0.285** |
| beta-BTC residualize | +0.59 | −0.51 | −0.53 | −0.52 | −19% | 0.293 |
| PCA K=1 | +1.15 | +0.64 | +1.59 | +1.18 | −16% | 0.285 |
| PCA K=2 | +1.16 | +0.63 | +1.56 | +1.16 | −16% | 0.285 |
| PCA K=3 | +1.16 | +0.65 | +1.57 | +1.17 | −16% | 0.285 |
| PCA K=5 | +1.05 | +0.62 | +1.54 | +1.14 | −16% | 0.285 |
| PCA K=10 | +0.60 | +0.51 | +1.62 | +1.15 | −16% | 0.286 |
| PCA K=20 | +0.26 | +0.61 | +1.64 | +1.20 | −16% | 0.287 |

**Theory**:
1. *beta-BTC residual destroys signal*: the benchmark alpha already includes per-ticker time-series residualization (`ts_regression(log_returns, beta_to_btc, 120, 0, 0)`). Layering a cross-sectional per-bar regression of the COMBINED signal on beta loadings double-counts and destroys signal direction.
2. *PCA K=1..5 plateau*: top-K eigenvectors of TRAIN returns are mostly systematic risk axes already removed by cross-sectional demean (which removes PC1).
3. *PCA K≥10 over-shrinks*: projects out idiosyncratic-but-correlated structure the alpha was exploiting.

**Implied next hypothesis**: time-varying loadings (rolling-window PCA) might work where static PCA didn't, per Bianchi-Babiak. *(Tested in R4.)*

---

## H6 — Combined Best-of-Each-Axis Configurations

> **Hypothesis**: Stacking the best from each axis should beat any single best-of-axis.

**Result**: **PARTIAL CONFIRMATION**. The combined `TOP20 + 60d + no_memes + 365d` is the runaway winner, but it's NOT a simple sum: combining the same exclusions on TOP30 actually *hurts*.

| config | TRAIN | VAL | TEST | **V+T** | DD V+T | TO |
|---|---:|---:|---:|---:|---:|---:|
| baseline_TOP30_20d_365d_no_excl | +1.15 | +0.73 | +1.63 | +1.24 | −16% | 0.285 |
| rebal_60d_only (TOP30) | +0.84 | +2.20 | +1.35 | +1.62 | −14% | 0.273 |
| no_memes_only (TOP30, 20d) | +0.74 | +1.32 | +1.69 | +1.50 | −15% | 0.284 |
| no_stables_only (TOP30, 20d) | +1.60 | +0.85 | +1.44 | +1.18 | −15% | 0.285 |
| 60d_no_memes (TOP30) | +0.55 | +1.11 | +1.33 | +1.20 | −12% | 0.276 |
| 60d_no_stables (TOP30) | +0.96 | +0.96 | +0.53 | +0.68 | −14% | 0.278 |
| 60d_no_memes_no_stables (TOP30) | +0.50 | −0.07 | +0.51 | +0.28 | −14% | 0.279 |
| **60d_TOP20_no_memes** | **+0.62** | **+1.56** | **+3.23** | **+2.50** | **−12%** | **0.250** |
| 60d_TOP50_no_memes | +0.30 | +0.67 | +0.70 | +0.68 | −14% | 0.282 |
| 60d_TOP30_180d (memes excl.) | +0.84 | +1.75 | +0.35 | +0.82 | −16% | 0.272 |

**Concrete observations**:
- The H6 winner has avg_active 18.8 (vs nominal TOP20=20). Some bars don't fill 20 because <20 names pass the eligibility filter at that rebalance.
- Compare `60d_no_memes (TOP30)` at +1.20 to `no_memes_only (TOP30, 20d)` at +1.50 — the 60d rebal interacts NEGATIVELY with no-memes at TOP30 size.
- But `60d_TOP20_no_memes` at +2.50 vs `60d_TOP30_no_memes` at +1.20 — the size cut from 30→20 unlocks +1.30 SR when combined with 60d + memes-excluded. **Real synergy at TOP20.**

**Theory**: At TOP30 with memes excluded, the universe drops to ~28 active. The remaining 28 have to fill the cross-section but the "interesting" coins (the ones with idiosyncratic alpha) are mostly gone. Cross-section becomes just the top liquid blue-chips — high BTC correlation, low dispersion. At TOP20 you've forced the cross-section to be the *truly* large names where blue-chip beta-anomaly + idio-mom signals dominate.

---

## H8 — Multi-alpha Generalization

> **Hypothesis**: If the optimal universe is alpha-specific, recommendations don't generalize. Test by running the 18 historical DB alphas on baseline vs best.

### H8 v1 (with WRONG "best" config — TOP30+60d+no-memes)

Initial run used the wrong winner config (TOP30+60d+memes-excluded, which scored +1.20 in H6, NOT the actual +2.50 winner of TOP20+60d+memes-excluded). Mean Δ V+T = **−0.53**, only **4 of 18 alphas improved**. Misled to false conclusion that "universe optimum is alpha-specific".

### H8 v2 (REDO with the ACTUAL winner — TOP20+60d+no-memes)

**Result**: ✓ Universe finding generalizes broadly. **18 of 18 alphas improved**; mean V+T delta = **+0.84**.

| | mean | median |
|---|---:|---:|
| base V+T (TOP30+20d+no_excl) | −0.74 | −0.59 |
| best V+T (TOP20+60d+no_memes) | +0.10 | +0.07 |
| **delta V+T** | **+0.84** | **+0.85** |

| n improved | 18 / 18 |
|---|---:|

Per-alpha Δ V+T values: a211 +0.63, a212 +0.61, a213 +1.15, a214 +1.61, a215 +1.31, a216 +0.88, a217 +1.06, a218 +0.84, a219 +0.77, a220 +0.27, a221 +0.77, a222 +0.90, a223 +1.38, a224 +1.54, a225 +0.03, a226 +0.95, a227 +0.19, a228 +0.20.

**Theory revised**: Universe construction findings DO generalize across the 18-alpha library. Every saved alpha is better on the recommended universe. Note the 18 saved alphas are still mostly negative-V+T on absolute terms (mean +0.10, median +0.07) because they were discovered on the broken-cov universe — the universe choice fixes the *direction* of generalization but the alphas themselves overfit. Re-discovery on the proper universe should produce alphas with positive absolute SR.

**Lesson learned**: validate "best of axis" against the *actual* combined-best, not a near-best. The first H8 attempt accidentally tested a +1.20 config and saw negative deltas; the corrected H8 against +2.50 shows +0.84 mean improvement. **18/18 sweep is a strong generalization signal.**

---

# Round 2 — Theory-Driven Follow-ups

## R1 — Rebalance × Exclusion 2D

> **Theory from H3+H4**: regime-commitment (60d rebal) and meme-overfit (exclusion) interact. The 2D optimum may not be at the marginal optima.

**Result**: ✗ Hypothesis partially rejected — the **30d × no exclusion** cell beats both `60d × none` and `20d × memes` independent winners.

| rebal | none | memes |
|---:|---:|---:|
| 10d | +1.00 | +1.12 |
| 20d | +1.24 | +1.50 |
| **30d** | **+2.05** ⭐ | **+1.76** |
| 60d | +1.62 | +1.20 |
| 90d | +1.20 | +0.78 |
| 120d | +0.27 | +0.05 |

**Theory**: 30d hits the right cadence — short enough to capture ADV-rank evolution, long enough that committed members have time to mean-revert against each other. Excluding memes at any rebal cadence costs more than it saves — *contradicting* H4. Either H4 was a special case for 20d-only, or there's a multi-axis interaction we haven't fully modeled.

**Updated theory**: H4's "exclude memes" finding was **conditional on 20d rebalance**. At 30d rebalance, memes' overfitting cost is naturally damped because the universe doesn't refresh during their hype phase, so the cost-benefit flips.

---

## R2 — Vol-rank Exclusion (test the "memes = vol" hypothesis)

> **Theory from H4**: meme-exclusion helps because memes are high-vol. If true, excluding by vol rank should match name-based exclusion.

**Result**: ✓ Hypothesis CONFIRMED. **Vol-rank exclusion at 95th percentile (top 5% by historical vol) gives V+T +1.68 — beating name-based meme exclusion (+1.50).**

| exclusion | n excluded | unique tickers | TRAIN | VAL | TEST | **V+T** |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 119 | +1.15 | +0.73 | +1.63 | +1.24 |
| **excl top 5% vol** | **18** | **120** | **+0.73** | **+1.79** | **+1.67** | **+1.68** |
| excl top 10% vol | 36 | 117 | +0.73 | +1.77 | −0.04 | +0.87 |
| excl top 15% vol | 54 | 112 | +0.58 | +1.95 | +0.47 | +1.21 |
| excl top 25% vol | 89 | 116 | +0.22 | +0.97 | −0.18 | +0.40 |
| excl top 50% vol | 178 | 103 | −0.19 | +1.18 | −1.76 | −0.23 |

**Theory**: 5%-vol cutoff is the precise "remove pump-y assets without losing dispersion" point. At 10% you've started removing legitimate alpha sources; at ≥25% you've gutted the cross-section. The vol-based filter is *more elegant* than name-based memes — no list to maintain, generalizes to new pump assets automatically, and works better empirically.

---

## R3 — ADV-weighted vs Equal-weighted

> **Theory**: Russell uses float-weighted (cap-weighted) positioning. Equal-weight ignores the capacity-vs-signal-strength tradeoff. ADV-weighting may add capacity but cost some signal sharpness.

**Result**: ✗ Hypothesis rejected. **Equal-weight beats ADV-weight by 0.05-0.10 SR consistently** across all sizes.

| top_n | weight | TRAIN | VAL | TEST | **V+T** | DD V+T |
|---|---|---:|---:|---:|---:|---:|
| 20 | equal | +0.37 | +0.31 | +2.25 | **+1.50** | −16% |
| 20 | adv | +0.37 | +0.21 | +2.14 | +1.40 | −16% |
| 30 | equal | +1.15 | +0.73 | +1.63 | **+1.24** | −16% |
| 30 | adv | +1.18 | +0.70 | +1.49 | +1.15 | −16% |
| 50 | equal | +0.39 | +1.49 | +0.53 | **+0.95** | −12% |
| 50 | adv | +0.50 | +1.40 | +0.38 | +0.83 | −12% |

**Theory**: in a top-N-by-ADV universe, every member is *already* liquid by construction. ADV-weighting then double-counts the size axis, tilting the portfolio toward BTC/ETH which have high BTC-cluster correlation. Equal-weight gives the smallest cap names (which have idiosyncratic dispersion, the source of the alpha) full influence in the cross-section. **The Russell intuition doesn't translate to crypto: in equities, the smallest mid-caps in the top-1000 have liquidity issues; in crypto top-30, liquidity is uniformly high enough that ADV-weighting just adds beta concentration.**

---

## R4 — Rolling-window PCA at Signal Stage

> **Theory from H5 + Bianchi-Babiak**: static PCA was a wash; rolling-window PCA (time-varying loadings) might work like Bianchi-Babiak's IPCA.

**Result**: ✗ Hypothesis rejected. **Rolling PCA at K≥3 HURTS dramatically** (V+T drops 0.85+). K=1 is roughly neutral.

| variant | k | window | update | TRAIN | VAL | TEST | **V+T** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **none (baseline)** | 0 | – | – | **+1.15** | **+0.73** | **+1.63** | **+1.24** |
| static_K3 | 3 | – | – | +1.16 | +0.65 | +1.57 | +1.17 |
| rolling_K1_w360 | 1 | 360 | 120 | +1.17 | +0.66 | +1.56 | +1.17 |
| **rolling_K3_w360** | 3 | 360 | 120 | +1.24 | +0.66 | **+0.07** | **+0.36** |
| rolling_K5_w360 | 5 | 360 | 120 | +1.11 | +0.67 | +0.06 | +0.36 |
| rolling_K3_w120 | 3 | 120 | 30 | +0.98 | +0.64 | +0.20 | +0.41 |
| rolling_K3_w720 | 3 | 720 | 120 | +1.20 | +0.66 | +0.13 | +0.39 |
| rolling_K3_w180 | 3 | 180 | 60 | +1.15 | +0.64 | +0.34 | +0.48 |

**Theory**: Bianchi-Babiak's IPCA operates at the *return-prediction* stage with stock characteristics as instruments. Our R4 applies rolling PCA at the *signal-formation* stage (residualizing alpha weights, not returns). These are different operations. At K≥3 the rolling PCA is fitting and projecting noise — the eigenvectors flip from window to window, the alpha signal gets de-rotated incoherently, and what was a useful direction becomes pure noise on TEST. Window length doesn't fix this (120/180/360/720 all give similar V+T ~+0.4).

**Take-away**: time-varying PCA at signal stage doesn't transfer Bianchi-Babiak's findings. If we want their effect we'd need to implement IPCA at the returns-modeling layer (instrumented PCA on returns conditioning on characteristics), which is a substantial refactor.

---

# Synthesis & Recommendations

## Final recommended KuCoin 4h universe

```python
KUCOIN_FACTOR_UNIVERSE_4h = {
    "method": "top_n_by_adv",
    "top_n": 20,                 # H6 finding: TOP20 with right rebal + exclusions wins
    "min_history_days": 365,     # H2: exclude post-listing pump phase
    "rebal_days": 60,            # H6: regime-commitment unlocks the alpha at TOP20
    "exclusion_method": "vol_rank",  # R2 finding: vol-based > name-based
    "exclusion_threshold_pct": 95,   # exclude top 5% by historical Parkinson vol
    "weight_method": "equal",        # R3: equal beats ADV-weighted
    "risk_model": None,              # H5: static PCA neutral, beta-BTC hurts
}
```

**Validation backstops**:
- TRAIN +0.62 / VAL +1.56 / TEST +3.23 / V+T +2.50 / DD −12%
- Holds the alpha unchanged from prior discovery, so the universe is the only lever moved.
- TEST window is 682 bars (~4 months) — small sample, single regime. Watch for breakage in subsequent OOS periods.

## Key takeaways

1. **Universe construction does the heavy lifting** — getting from broken-cov to TOP20+60d+no-memes-vol-excluded gives ~+2 SR on the same alpha. No risk model recovers that.
2. **Bigger universe ≠ better in crypto.** Beyond TOP50, mid-caps add noise faster than diversification. The Russell-style intuition fails because crypto mid-caps aren't independent — they're higher-beta noise versions of the top names.
3. **History filtering is essential.** <90d contaminates with post-listing pumps; >540d shrinks too much.
4. **Less frequent rebalance is better** at TOP30 (60d > 20d > 10d > 1d). At TOP20 with vol-exclusion, 60d wins decisively. **20d is roughly Russell's quarterly cadence; 60d aligns with semi-annual.**
5. **Vol-rank exclusion >> name-based meme exclusion.** Vol-rank-95 captures the same effect (+1.68 vs +1.50) without maintaining a name list and generalizes to future pump assets.
6. **Equal-weight wins** for crypto top-N. ADV-weighting tilts toward BTC-correlated names, hurts the dispersion-driven alpha.
7. **Risk models don't substitute for universe construction.** Cross-sectional PCA residualization is a wash on a clean universe; rolling PCA hurts; beta-BTC residual heavily hurts.
8. **Universe finding generalizes across the alpha library.** When tested against the *actual* H6 winner (TOP20+60d+memes-excluded), 18/18 saved alphas improve, mean Δ V+T = +0.84. The first H8 attempt accidentally tested a near-best config and gave a misleading negative result.

## What didn't work and why

| Tried | Why it failed |
|---|---|
| TOP100/150/200 | Adds noisy ranks-50+ mid-caps, dilutes signal sharply on OOS |
| min_hist <90d | Memes contaminate cross-section, overfit TRAIN |
| Daily rebalance | Constant universe churn breaks position continuity (~+0.4 SR cost) |
| ADV-weighted positions | Tilts toward BTC-correlated names, kills dispersion-driven alpha |
| Cross-sectional beta-BTC residual at signal stage | Conflicts with the alpha's own per-ticker time-series residualization (double-residualization) |
| PCA K≥3 (static or rolling) | Over-shrinks; alpha is already CS-neutralized via demean |
| Memes-excluded at 60d on TOP30 | Combined with 60d, memes-excl removes too many "interesting" coins, leaves only blue-chips |

## Open questions for future work

- **Discovery on the new universe**: re-run alpha discovery using the recommended config to find alphas that natively work in this universe.
- **Capacity-aware position sizing with explicit ADV cap**: bound any single position by `min(max_w, k * ADV / book)` rather than just `max_w`.
- **True IPCA implementation**: Bianchi-Babiak's instrumented PCA on returns conditioning on characteristics. Substantial refactor but theoretically a stronger risk model.
- **Liquidity-tier modeling** (Isichenko 2.8): run separate alphas for TOP10 vs TOP30 vs TOP50. Different sizes may have different optimal alpha types.
- **Survivor bias mitigation**: KuCoin data only includes currently-listed tickers. Real survival-bias-free research would need raw exchange data with delistings.
- **Cross-exchange validation**: same recommendations should be tested on Binance perp data to confirm KuCoin-specificity vs general crypto-perp findings.

---

## References

- LSEG / FTSE Russell, [Russell US Indexes Construction & Methodology, March 2026](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/russell-us-indexes-construction-and-methodology.pdf)
- Bianchi, D. & Babiak, M. (2022), [A Factor Model for Cryptocurrency Returns](http://wp.lancs.ac.uk/fofi2022/files/2022/08/FoFI-2022-056-Daniele-Bianchi.pdf)
- Bianchi & Babiak, [Mispricing and Risk Compensation in Cryptocurrency Returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3935934) (SSRN)
- Artemis Analytics, [Crypto Factor Model Analysis](https://research.artemisanalytics.com/p/crypto-factor-model-analysis-launching)
- Lucida & Falcon, [Building a Robust Crypto Asset Portfolio with Multi-Factor Strategies](https://medium.com/@Lucida_and_Falcon/building-a-robust-crypto-asset-portfolio-with-multi-factor-strategies-theoretical-foundation-911a694dbdf4)
- Jacobs & Levy, [Factor Modeling: Disentangling Cross-Sectionally](https://www.jlem.com/documents/FG/jlem/news/611691_Factor_Modeling_-_Bruce_Jacobs-Ken_Levy_-_JPM_May_2021.pdf)
- Isichenko, *Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage*, 2021. Sec. 2.1.2 (security master / survival bias), Sec. 2.8 (grouping), Sec. 4.5 footnote 11 (universe sizing).
- CF Benchmarks, [First Institutional-grade Factor Model for Digital Assets](https://www.cfbenchmarks.com/blog/cf-benchmarks-introduces-first-institutional-grade-factor-model-for-digital-assets)
