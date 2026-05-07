---
name: alpha-research
description: Discover or query cross-sectional alpha factors against any (asset class × venue × frequency × universe) lane in this repo. Use when the user asks to research/discover/save/find/list alphas, propose alpha expressions, search the alpha library by strategy family (momentum/value/quality/carry/microstructure/...), or audit what's already in the DB. Replaces the per-asset workflow files in `.agents/workflows/` with one parameterized loop. Lane-specific quality gates, data fields, and templates still live in `.agents/`.
---

# Alpha Research — Universal Loop

This skill is the unified entry point for two activities:

1. **Discovery loop** — propose, evaluate, and save a cross-sectional alpha for one specific (lane, universe) — same pattern as `.agents/workflows/{equities,kucoin-4h,binance-4h,crypto-4h}-alpha-research.md`, parameterized.
2. **Library search** — query the alpha DB by strategy family, dataset, universe, lane — used to audit coverage, find candidates to combine, or seed hypothesis generation.

The skill is **lane-agnostic**. It does not pick the universe or eval script for you — the user (or the conversation context) tells it which lane, and the skill enforces the discipline that applies to every lane.

---

## Step 0 — Pin the lane

Every alpha lives in exactly one lane. Confirm all six axes before proposing the first expression. WorldQuant BRAIN, Trexquant, and most factor shops use this same decomposition:

| Axis | What it controls | Example values |
|------|------------------|----------------|
| **Region / Venue** | Tradeable instrument set + fee schedule | USA equities (FMP), Binance USDT-M perps, KuCoin USDT-M perps |
| **Instrument** | Asset class — picks the data table | `equity`, `crypto` |
| **Frequency / Bar** | Sets √(bars/year) annualization, cost sensitivity, rebalance cadence | `1d` (252), `4h` (2190), `5m` (105k) |
| **Universe** | Cross-sectional breadth + per-name capacity | `TOP1000`, `MCAP_100M_500M`, `MCAP_500M_2B`, `KUCOIN_TOP30`, `BINANCE_TOP100` |
| **Delay** | Lookahead discipline | `1` (yesterday → today, default for daily), `0` (closing-auction MOC) |
| **Neutralization** | Which exposures we strip before the bet | `subindustry` (GICS-8), `sector`, `market`, `none` |

The lane fully determines: which `eval_*.py` script to call, which `data/<venue>_cache/matrices/` directory to read from, which DB table to read/write (`alphas` for equities, `alphas_crypto` for crypto), which corr-cutoff scope applies, and which quality-gate row in the table below.

Strict rule: **the corr cutoff is scoped to the alpha's own (universe, instrument) lane, never across lanes.** A KuCoin alpha is never compared to a Binance alpha; a TOP1000-d1 alpha is never compared to a TOP1500_2500-d0 alpha. Cross-lane novelty is meaningless because the lanes have different return drivers.

If the user has not pinned the lane yet, ask exactly once which lane to research. Do not guess.

---

## Step 1 — Strategy family (the WHY)

Every alpha must be filed under one strategy family before evaluation. The family is the economic mechanism, not the formula. Pick from this taxonomy (mirrors WorldQuant's "alpha families" and Trexquant's signal-source split):

### Price / volume (technical)
- **Momentum** — past winners continue (12-1 month for equities; 1-7 day for crypto). Skip-most-recent is mandatory for cross-sectional momentum.
- **Reversal** — short-term mean reversion (1-5 days equities; 1-3 bars crypto). Often the negative of momentum at a different horizon.
- **Low Volatility** — low-vol stocks earn risk-adjusted premium; sign is negative.
- **Trend strength** — efficiency ratio, ts_corr(close, time, N), momentum stability.
- **Range / candlestick** — `upper_shadow`, `lower_shadow`, `close_position_in_range`. Strong for crypto.

### Microstructure / orderflow
- **Volume momentum / surge** — `volume_ratio_*`, `s_log_1p(adv*)`, ADV regime shift.
- **Taker flow** — `taker_buy_ratio`, taker-maker imbalance (crypto only — Binance/KuCoin).
- **VWAP deviation** — `vwap_deviation`, close-vs-VWAP.
- **Trade intensity** — `trades_per_volume`, trade-size distribution.

### Carry / term structure (crypto only)
- **Funding** — `funding_rate`, funding-rate Z-score, funding-momentum. Highly predictive at 4h.
- **Basis / contango** — perpetual-vs-spot premium (when both available).

### Fundamentals (equities only)
- **Value** — `earnings_yield`, `book_to_market`, `ev_to_ebitda`, `free_cashflow_yield`.
- **Quality / profitability** — `roe`, `roa`, `gross_margin`, `operating_margin`.
- **Accruals / earnings quality** — Sloan accrual: `(net_income - cashflow_op) / assets`.
- **Investment / asset growth** — `ts_delta(assets, 252)`, capex/revenue.
- **Growth** — revenue/EPS growth.
- **Leverage** — `debt_to_equity`, `interest_coverage`.

### Liquidity / size
- Equities: `s_log_1p(adv60)`, `sharesout`, `market_cap`.
- Crypto: `adv60`, `quote_volume`.

### Risk / vol-of-vol
- `parkinson_volatility_*`, `historical_volatility_*`, vol-of-vol, beta-to-market (`beta_to_btc` for crypto), beta dispersion.

### Cross-asset / regime (when multi-asset matrices exist)
- BTC-conditional crypto signals; sector-conditional equity signals.

### Composite / multi-family
- Two or more orthogonal families combined via the **multiplicative rank composite** template (see `.agents/{crypto,equities}-alpha-templates.md`). Only register a composite if both base signals are themselves saved alphas — otherwise just save the base signals.

If a hypothesis spans multiple families, file under the *dominant* family and note the secondary in the reasoning.

---

## Step 2 — Discovery loop

Repeat per hypothesis. Each iteration is one alpha. **One alpha at a time** — never sweep parameters in a batch as a substitute for thinking.

```
HYPOTHESIZE  →  EVALUATE  →  ANALYZE  →  SAVE (if all gates pass)  →  REPORT
```

### 2a. Hypothesize
State, in order:
1. **Strategy family** (from §1).
2. **Economic mechanism** in one sentence. Why does this signal predict cross-sectional returns? What inefficiency, behavior, or compensation creates the edge?
3. **Why orthogonal to the existing set** — what observable, operator, time horizon, normalization, or aggregation does this use that no saved alpha in the same lane uses?
4. **Direction** — sign of the bet, with reasoning (don't blindly try both signs; that's overfitting).

If you can't write all four in one paragraph, the hypothesis is data-mining. Stop and pick a different mechanism.

### 2b. Evaluate (TRAIN only)
Run the lane's eval script with `--expr "..."` plus the lane's universe flag. Look at IS Sharpe, IC, Fitness, Turnover, sub-period stability — every metric the script reports.

Never look at VAL or TEST during discovery. The Train/Val/Test split for the lane is documented in the per-lane workflow file.

### 2c. Analyze
For each metric, compare to the lane's quality gate (§4). Decisions:

- **IS SR strong, IC ≈ 0** → outlier-driven, discard.
- **IS SR strong, sub-period unstable** → regime-specific, discard.
- **IS SR weak, corr-to-existing low** → real but noisy; refine normalization or aggregation. Do NOT just add decay (raises corr).
- **IS SR strong, corr-to-existing high** → in an existing family; change the *observable*, not the formula.
- **Sign opposite to expectation** → flip and reinterpret. The mechanism is opposite of expected — that's information, but write it up before saving.

### 2d. Save (if every gate passes)
Use the lane's eval script with `--save --reasoning "..."`. The script enforces:
1. All quality gates (§4).
2. **|corr| ≤ lane corr-cutoff** against every existing alpha in the same lane. This is non-negotiable; do not lower the cutoff.

Reasoning text must include:
- The strategy family tag (e.g. `[MOMENTUM]`, `[FUNDING_CARRY]`, `[ACCRUALS]`).
- The lane tag if non-default (e.g. `[SMALLCAP_D0]`, `[KUCOIN_TOP100]`).
- The economic mechanism in one sentence.

### 2e. Report
Every 5–10 saved alphas, run the lane's `--list` or `--scoreboard` command. Track family coverage, not just count.

---

## Step 3 — Library search by strategy type

The skill's other half: query existing alphas in `data/alpha_results.db` by strategy family, dataset, universe, or lane. Use this to audit what we have, find candidates to combine, or pick orthogonal areas to research next.

### 3a. Schema cheat-sheet

```sql
-- Equities
alphas(id, expression, name, category, asset_class, interval, source, notes, archived, created_at)
evaluations(alpha_id, sharpe_is, sharpe_train, return_ann, max_drawdown, turnover,
            fitness, ic_mean, ic_ir, psr, delay, decay, universe, max_weight, neutralization, ...)

-- Crypto
alphas_crypto(id, expression, name, category, asset_class, interval, source, archived, notes, universe, ...)
evaluations_crypto(alpha_id, run_id, sharpe_is, sharpe_oos, sharpe_train, sharpe_val,
                   sharpe_test, return_total, return_ann, max_drawdown, turnover,
                   fitness, ic_mean, ic_ir, psr, universe, ...)
```

The `category` and `notes` columns are where strategy family lives. The two columns are imperfect — `category` is sparse on older rows, and `notes` is free-text — so always combine `category` matching with **expression keyword grep** for completeness.

### 3b. Strategy-family keyword index

Use these keyword bundles when grepping `expression` to fill in for missing `category` tags:

| Family | Expression keyword pattern (case-insensitive) |
|--------|-----------------------------------------------|
| momentum | `momentum_`, `ts_delta(close,`, `ts_rank(close,`, `ts_rank(returns,` |
| reversal | `negative(rank(returns))`, `ts_zscore(returns,`, `Decay_exp(returns,` |
| low_vol | `historical_volatility_`, `parkinson_volatility_`, `negative(rank(historical_volatility` |
| value | `earnings_yield`, `book_to_market`, `ev_to_`, `free_cashflow_yield`, `pe_ratio`, `pb_ratio` |
| quality | `roe`, `roa`, `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `asset_turnover` |
| accruals | `net_income`, `cashflow_op`, `subtract(true_divide(net_income`, `Sloan` |
| growth | `eps_diluted`, `revenue`, `ts_delta(revenue,` , `ts_delta(eps,` |
| leverage | `debt_to_equity`, `debt_to_assets`, `interest_coverage`, `net_debt` |
| volume_microstructure | `adv20`, `adv60`, `volume_ratio`, `volume_momentum`, `s_log_1p(volume`, `dollars_traded` |
| taker_flow | `taker_buy_ratio`, `taker_buy_volume` (crypto only) |
| vwap | `vwap_deviation`, `vwap` |
| range_candlestick | `upper_shadow`, `lower_shadow`, `close_position_in_range`, `high_low_range`, `open_close_range` |
| funding_carry | `funding_rate` (crypto only) |
| trade_intensity | `trades_per_volume`, `trades_count` |
| size | `sharesout`, `market_cap`, `mcap` |
| beta_regime | `beta_to_btc`, `beta_to_market` |

Note: a single expression often spans two families (e.g. `multiply(zscore_cs(taker_buy_ratio), zscore_cs(funding_rate))` is `taker_flow` ∩ `funding_carry`). For audits, count an alpha under each family it touches.

### 3c. Search query patterns

**Coverage by family (one lane):**
```sql
-- Crypto KUCOIN_TOP30 alphas grouped by funding presence
SELECT
  CASE WHEN expression LIKE '%funding_rate%' THEN 'funding_carry' ELSE 'other' END AS family,
  COUNT(*) AS n,
  AVG(e.sharpe_is) AS avg_is_sr
FROM alphas_crypto a
JOIN evaluations_crypto e ON e.alpha_id = a.id
WHERE a.archived = 0
  AND a.universe = 'KUCOIN_TOP30'
  AND a.interval = '4h'
GROUP BY family;
```

**Top alphas in a family (equities):**
```sql
SELECT a.id, a.expression, e.sharpe_is, e.fitness, e.universe, e.neutralization
FROM alphas a
JOIN evaluations e ON e.alpha_id = a.id
WHERE a.archived = 0
  AND (a.expression LIKE '%earnings_yield%' OR a.expression LIKE '%book_to_market%'
       OR a.expression LIKE '%ev_to_%'        OR a.expression LIKE '%free_cashflow_yield%')
  AND e.universe = 'TOP1000'
  AND e.delay = 1
ORDER BY e.sharpe_is DESC
LIMIT 20;
```

**Find orthogonality gaps (which families are *under-represented* in a lane):**
List the family taxonomy (§1), grep the alpha set with each family's keywords (§3b), tally counts. Families with zero or low coverage are the priority research targets — they have the most orthogonality budget left.

**Cross-lane comparison (audit only — never to seed a save):**
```sql
SELECT a.universe, COUNT(*) AS n, AVG(e.sharpe_train) AS avg_train_sr
FROM alphas_crypto a
JOIN evaluations_crypto e ON e.alpha_id = a.id
WHERE a.archived = 0 AND a.interval = '4h'
GROUP BY a.universe;
```

When the user asks "what momentum alphas do we have on the small-cap MCAP_100M_500M lane?" — combine the `evaluations.universe` filter with the family keyword grep, return id + expression + sharpe_is + neutralization, sorted by sharpe.

---

## Step 4 — Quality gates (lookup table)

The numbers below come from the per-lane workflow files. Validate against the lane's workflow when in doubt — these are the published defaults, not constants.

| Lane | IS SR gate | IC gate | Fitness gate | Turnover ceiling | Corr cutoff | Sub-period rule |
|------|-----------|---------|--------------|------------------|-------------|------------------|
| Equities TOP1000 d=1 (default) | ≥ 1.0 | > 0 | informational | n/a (low fee sensitivity) | 0.65 | both halves > 0 |
| Equities small-cap MCAP_100M_500M d=0 | ≥ 5.0 (lane override) | > 0 | informational | n/a | 0.65 | both halves > 0 |
| Crypto KUCOIN_TOP30 4h | > 2.0 | (informational) | > 5.0 | < 0.30 / bar | 0.70 | H1 & H2 both SR > 0 |
| Crypto BINANCE_TOP30 4h | > 2.0 | (informational) | > 5.0 | < 0.30 / bar | 0.70 | H1 & H2 both SR > 0 |
| Crypto BINANCE_TOP50 / TOP100 4h | > 2.0 | (informational) | > 5.0 | < 0.30 / bar | 0.70 | both halves > 0 |

Every cell in this table is a hard gate at save time, enforced by the lane's eval script. None of these are ceilings — beating them by 30%+ is the goal, not the requirement to save.

**Never lower a gate to make a candidate fit.** If nothing passes, expand the strategy-family search (§3b shows which families are under-represented), expand the universe, or accept saturation. Lowering gates without explicit user permission is silent overfitting.

---

## Step 5 — Anti-overfitting rules (cross-cutting)

These apply to every lane. They are the difference between a real signal and a Train-period artifact.

1. **One hypothesis at a time.** Sweeping `lookback ∈ {30, 60, 90, 120}` on the same base signal is parameter search on a single hypothesis — it tells you nothing about whether the hypothesis is right and burns the test-set budget through the multiple-comparisons door.
2. **Round lookbacks only.** Daily: 5, 21, 63, 252. 4h crypto: 6, 12, 30, 60, 120, 240. Decimal lookbacks (e.g. `ts_zscore(close, 73)`) are a tell that you're optimizing.
3. **Simple beats complex.** `rank(earnings_yield)` is a real signal; a 6-deep nested operator chain is a coin flip.
4. **Test lookback sensitivity.** If a signal works at 60d, it should also work at 30d and 120d in the same direction. If it breaks under that perturbation, discard.
5. **Orthogonality > magnitude.** An uncorrelated SR 1.0 is worth more than a redundant SR 2.0 because the combiner can use it.
6. **DSR / multiple-comparisons awareness.** The more variants you've tested, the higher the bar for calling a discovery real. The eval script's DSR field surfaces this directly.
7. **Cross-sectional normalization is required.** Always wrap raw observables in `rank()` or `zscore_cs()` before combining. Raw fundamentals and raw volume have extreme skew.
8. **Use `s_log_1p(...)`** for any field with a heavy right tail (volume, dollars_traded, ADV).
9. **Saturation is a real answer.** When every SR-passing candidate fails the corr gate and family coverage is full, the lane is saturated. Report it. The correct response is to expand the universe, add a new dataset, or accept the cap — not to lower the threshold.

---

## Step 6 — When you save a new alpha to a lane that's also live

If the lane being researched is also a live signal lane (`research_equity.json`, `research_crypto.json` — i.e. consumed by Dagster's `research_*_signal_snapshot` assets), **verify byte-equivalence of the bounded incremental compute** before declaring the work done:

```bash
python tools/verify_incremental_signal.py prod/config/research_equity.json --bars 400
python tools/verify_incremental_signal.py prod/config/research_crypto.json --bars 1500
```

This re-runs the full pipeline twice (full history vs bounded slice) and asserts the latest weights row matches within fp64 tolerance. If a new alpha uses an unusually long lookback (`Decay_exp(α<0.02)`, `ts_*` window > 360 bars, etc.), bump `EQUITY_SIGNAL_MAX_LOOKBACK_BARS` / `CRYPTO_SIGNAL_MAX_LOOKBACK_BARS` and re-verify before merging. The truncation formula is documented in `src/pipeline/signal_service.py`.

A new alpha that fails verification at the current bound is a regression — fix the bound before saving, or save under a config that's not in the live signal path.

---

## Pointers

- Per-lane discovery details (data fields, alpha templates, narrative): `.agents/workflows/{equities,kucoin-4h,binance-4h,crypto-4h}-alpha-research.md`.
- Per-lane signal templates and proven composites: `.agents/{equities,crypto,kucoin}-alpha-templates.md`.
- Universe construction: `prod/config/{kucoin,binance}_universe.json`, `tools/build_kucoin_universe_20d.py`.
- Pipeline plumbing (preprocess, combine, QP, fees): `PIPELINE.md`.
- Live signal compute (incremental verification): `src/pipeline/signal_service.py`, `tools/verify_incremental_signal.py`.