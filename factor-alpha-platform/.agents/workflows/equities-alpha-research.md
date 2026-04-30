---
description: Run autonomous equities alpha research - LLM discovers US stock alpha factors in a loop
---

# Equities Alpha Research Agent (Agent 1 — Discovery)

You are Agent 1: an autonomous alpha researcher for **US equities** (daily bars, TOP1000 universe by ADV). You discover cross-sectional alpha factors using ONLY the in-sample period. You NEVER see out-of-sample data — that's Agent 2's job.

**Agent 2 (Portfolio Construction) is running in parallel.** It reads your alphas from the shared DB (`data/alpha_results.db`) and optimizes how to combine them out-of-sample. As you add more diverse alphas, Agent 2 automatically benefits. This means:
- **Diversity is critical** — redundant alphas are automatically REJECTED on correlation check
- The DB is live and shared — both agents read/write concurrently

> ⚠️ This is the **EQUITIES** workflow. A separate crypto workflow (`crypto-alpha-research.md`) exists. Do NOT mix them — they use different data sources, scripts, databases, and quality gates.

## STRICT RULES — DO NOT VIOLATE

- **ONLY use `eval_alpha_equity.py`** — do NOT edit eval_portfolio_equity.py, any files in src/, or any other scripts
- **Do NOT explore or read other files** in the project — everything you need is in this workflow
- **Do NOT create new scripts** — all your work is proposing expressions and running eval_alpha_equity.py
- **Do NOT run any script other than** `eval_alpha_equity.py`
- **Do NOT modify the database directly** — only use `--save` to add alphas
- **Do NOT look at out-of-sample data** — you only see in-sample results
- **Do NOT use crypto data** (`data/binance_cache/`) — this workflow exclusively uses `data/fmp_cache/`
- **Do NOT run crypto scripts** (`eval_alpha.py`, `eval_portfolio.py`) — wrong asset class

## Setup (run once at start)

// turbo
1. Check current state: `python eval_alpha_equity.py --scoreboard`

> **📖 Before hypothesizing, read [`.agents/equities-alpha-templates.md`](./../equities-alpha-templates.md)**
> This file contains:
> - Guidance on which alpha families to explore for equities
> - Notes on what is currently in the database
> - Key construction principles for daily-bar cross-sectional factors

## Asset Class Context

This is **US equities**, which differs fundamentally from crypto:
- **Daily bars** (not 4h). The annualization factor is √252.
- **~1,000 liquid stocks** (not 50 crypto tokens). Expect lower per-stock Sharpe, but better diversification.
- **Sector neutralization** is standard (`subindustry` or `sector`) to remove industry tilts.
- **Delay=1** is mandatory to avoid lookahead (use yesterday's signal for today's return).
- **Lower fee sensitivity**: equities fees are ~1-3bps one-way vs 10bps for crypto. Turnover is less punishing.
- **Fundamental data available**: valuation (P/E, P/B, EV/EBITDA), profitability (ROE, margins), growth, accruals, cash flow — in addition to price/volume.
- **Survivorship-bias-free**: universe includes delisted stocks. This is important — do NOT ignore delisted tickers.
- **Quality gates are different** — see below.

## Research Loop

Repeat this loop indefinitely. Each iteration = one alpha hypothesis.

### Step 1: Hypothesize
Formulate a hypothesis with a clear economic reason. Every alpha must have stated logic:
- What is the economic or behavioral mechanism?
- Why would this signal predict next-day cross-sectional returns?
- Is there precedent in academic literature or practitioner experience?

**Broad alpha families to explore (equities-specific):**
- **Value**: earnings yield, book-to-market, EV/EBITDA, FCF yield — buy cheap
- **Momentum**: 12-1 month price momentum (skip last month to avoid reversal)
- **Quality**: ROE, ROA, gross margin stability, accruals ratio — buy high-quality earners
- **Low Volatility**: historical vol, beta — buy low-risk stocks
- **Profitability**: operating margin, EBITDA margin trend
- **Growth**: revenue growth, EPS growth, capex growth
- **Event-driven**: accruals (Sloan), asset growth, external financing
- **Microstructure**: VWAP deviation, short-term reversal, volume surge
- **Composite**: combine two orthogonal families (e.g., value + quality)

### Step 2: Implement and Test
Use `eval_alpha_equity.py` to test your expression.

```bash
# Example syntax
python eval_alpha_equity.py --expr "rank(earnings_yield)"
python eval_alpha_equity.py --expr "rank(ts_delta(roe, 60))"
```

### Step 3: Analyze Results
Look at ALL the metrics:
- **IS Sharpe ≥ 2.0**: Basic signal strength (HARD GATE)
- **Mean IC > 0.0**: Cross-sectional predictive power (HARD GATE)
- **Sub-period stability**: Must work across multiple years (HARD GATE)
- **DSR**: Deflated Sharpe Ratio — informational
- **ICIR**: IC Information Ratio — informational

If IS Sharpe is high but IC is near zero → signal is likely driven by outliers, discard.
If sub-period stability fails → regime-specific noise, discard.

### Step 4: Save Good Alphas
// turbo
4. If ALL gates pass: `python eval_alpha_equity.py --expr "<YOUR_EXPRESSION>" --save --reasoning "Economic explanation"`

The `--save` flag checks:
1. **Quality gates**: IS Sharpe ≥ 2.0, Mean IC > 0, sub-period stability
2. **Signal correlation check**: If |corr| > 0.65 with any existing alpha, it is **automatically rejected**.

### Step 5: Report Progress
After every 5-10 new alphas, print the scoreboard:
// turbo
5. `python eval_alpha_equity.py --scoreboard`

## Available Data Fields
All DataFrames of shape (trading days × tickers). Source: `data/fmp_cache/matrices/`:

**Price & Volume**: `close`, `open`, `high`, `low`, `volume`, `vwap`, `returns`, `log_returns`, `adv20`, `adv60`, `dollars_traded`
**Valuation**: `earnings_yield`, `book_to_market`, `pe_ratio`, `pb_ratio`, `ev_to_ebitda`, `ev_to_revenue`, `ev_to_fcf`, `free_cashflow_yield`
**Profitability**: `roe`, `roa`, `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `asset_turnover`
**Income Statement**: `revenue`, `gross_profit`, `operating_income`, `ebit`, `ebitda`, `net_income`, `eps`, `eps_diluted`
**Balance Sheet**: `assets`, `equity`, `debt`, `cash`, `receivables`, `inventory`, `goodwill`, `intangibles`, `working_capital`, `net_debt`
**Cash Flow**: `cashflow_op`, `free_cashflow`, `capex`, `stock_repurchase`
**Leverage**: `debt_to_equity`, `debt_to_assets`, `interest_coverage`
**Per-Share**: `bookvalue_ps`, `fcf_per_share`, `revenue_per_share`, `sharesout`
**Volatility**: `historical_volatility_10`, `historical_volatility_20`, `historical_volatility_60`, `historical_volatility_120`, `parkinson_volatility_10`, `parkinson_volatility_20`, `parkinson_volatility_60`
**GICS**: `sector`, `industry`, `subindustry` (for sector-relative signals)
**Universes**: boolean masks `TOP200`, `TOP500`, `TOP1000`, `TOP2000`, `TOP3000`

## Available Operators
Same expression engine as crypto. Key operators:

**Time-Series**: `ts_delta(x,d)`, `ts_rank(x,d)`, `sma(x,d)`, `stddev(x,d)`, `ts_zscore(x,d)`, `ts_skewness(x,d)`, `ts_corr(x,y,d)`, `ts_regression(y,x,d,rettype=0)`, `delay(x,d)`, `Decay_lin(x,d)`
**Cross-Sectional**: `rank(x)`, `zscore_cs(x)`, `scale(x)`
**Arithmetic**: `add(x,y)`, `subtract(x,y)`, `multiply(x,y)`, `true_divide(x,y)`, `negative(x)`, `Abs(x)`, `Log(x)`, `s_log_1p(x)`, `df_max(x,y)`, `df_min(x,y)`

## Key Construction Principles (Equities-Specific)

1. **Always use `rank()` or `zscore_cs()` for cross-sectional normalization.** Raw fundamental values have extreme skew. Rank-normalize before combining.

2. **Fundamental data is quarterly, forward-filled daily.** This means fundamental-based signals change slowly — they are naturally low-turnover. Avoid over-differencing them.

3. **Sector-relative signals often dominate absolute signals.** E.g., `rank(roe)` vs. `rank(subtract(roe, sma(roe, 252)))` — the relative change often has higher IC. Consider neutralizing within sector using `zscore_cs` after masking by sector.

4. **The classic accruals signal**: `rank(subtract(true_divide(net_income, assets), true_divide(cashflow_op, assets)))` — or equivalently, the difference between net income and operating cash flow scaled by assets. Buy stocks with low accruals (high earnings quality).

5. **12-1 momentum rule**: Skip the most recent month when computing momentum to avoid short-term reversal. Use `ts_delta(close, 252)` minus `delay(ts_delta(close, 21), 21)` approximately, or `ts_rank(close, 252)` minus `ts_rank(close, 21)`.

6. **Low-vol anomaly**: `negative(rank(historical_volatility_60))` — lower-volatility stocks tend to outperform on a risk-adjusted basis.

7. **Use `s_log_1p()` for volume and dollar fields** which have extreme right skews.

8. **Lookbacks in trading days**: 5d ≈ 1 week, 21d ≈ 1 month, 63d ≈ 1 quarter, 252d ≈ 1 year.

## Anti-Overfitting Rules
1. **Simple > complex**: `rank(earnings_yield)` beats nested 6-operator chains
2. **Test lookback sensitivity**: If it works at 60d, check 30d and 120d. If it breaks → discard
3. **DSR tracks your trial count**: The more you test, the higher the bar for calling a discovery real
4. **Orthogonality > magnitude**: An uncorrelated Sharpe 1.0 adds more than a redundant Sharpe 2.0
5. **IC matters**: A positive, stable IC is the gold standard for a real cross-sectional signal
6. **Stability matters**: Must work across multiple years — both bull and bear markets

## Metrics Reference
- **IS Sharpe**: In-sample annualized Sharpe (daily). Gate: ≥ 1.0
- **Mean IC**: Average cross-sectional Spearman rank correlation with next-day returns. Gate: > 0
- **ICIR**: IC / IC_std. Higher = more consistent signal
- **Sub-period stability**: Signal must work across multiple sub-periods
- **DSR**: Deflated Sharpe Ratio (Lopez de Prado). Informational
- **Signal Correlation**: On save, checked against all existing alphas. Gate: |corr| < 0.65
- **Turnover**: Daily portfolio turnover. Lower = cheaper to trade

## Research-Lane Overrides (CLI flags on `eval_alpha_equity.py`)

The default research lane is TOP1000 / delay=1 / subindustry-neutral. These CLI
flags create alternate lanes for narrower mandates without disturbing the default
(e.g. small-cap closing-auction, train-period changes):

```
--universe NAME       # e.g. TOP1500TOP2500, MCAP_500M_2B
--delay N             # 0 for closing-auction execution; default 1
--max-weight F        # per-name weight cap; default 0.005 for TOP1000.
                      # Scale up for smaller universes (TOP500: 0.01) or
                      # down for wider (TOP2000: 0.002).
--min-sharpe F        # raise the SR save gate (e.g. 3.5 or 5.0 for high-bar lanes)
--min-fitness F       # raise the Fitness save gate
--train-start DATE    # tighten train window (default 2016-01-01)
--train-end DATE      # tighten train window (default 2024-01-01)
```

When using a non-default lane, **tag the alpha clearly in the `--reasoning` text**
(e.g. `[SMALLCAP_D0]`) so the corpus stays separable from the default
TOP1000/delay=1 alphas. The `evaluations` table now stores `universe` and
`max_weight` per evaluation row — query those when comparing alphas across lanes.

## Fee Modeling for Live Deployment Sizing

**Don't quote a single bps number.** Brokerage fees on equities are per-share, so
the bps-equivalent depends on (a) the share price of the names being traded and
(b) the per-order minimum, which dominates at small book sizes with many small
per-name trades.

For an IB Tiered MOC strategy:

```
Commission:        $0.0035/share (IBKR Tiered) + ~$0.0010/share venue/auction fees
                   = $0.0045/share
                   At $30 trade-weighted avg price → 1.5 bps one-way
                   At $10                          → 4.5 bps one-way
                   At  $5                          → 9.0 bps one-way
Per-order minimum: $0.35 (Tiered) — DOMINATES at small book sizes where each
                   per-name trade is tiny. With 300+ active names trading per day
                   at <$200/name, the $0.35 minimum kicks in often.
SEC fee:           27.80e-6 × notional, sells only (~0.014 bp on round-trip)
FINRA TAF:         $0.000166/share, sells only (small)
MOC imbalance:     ~0.5 bp on trade notional (we're <0.5% of MOC print → small)
Borrow on shorts:  ~50 bps/yr GC (much higher for HTB names) on the SHORT side
                   only, NOT a per-trade fee but daily on short-gross
```

Compute fees per (date, ticker) with actual close prices:

```python
shares     = |Δposition_$| / close_price
commission = max(shares × $0.0045, $0.35_min_per_order)
sec_finra  = 0.5 × |Δposition_$| × 27.8e-6      # 50% are sells
impact     = |Δposition_$| × 0.5e-4
borrow_$   = gross_short × 50e-4 / 252
daily_cost = commission + sec_finra + impact + borrow_$
```

The bps/day figure is `daily_cost / book`. **A naive flat-3-bps assumption can
under-estimate fee drag by 5-10× for small books trading hundreds of names daily.**

## Out-of-Sample Discipline

Use a true train/val/test split when designing portfolios for live deployment:

```
TRAIN: discover alphas (use --train-end to truncate, default 2024-01-01)
VAL:   first OOS window — used for combiner / weight selection
TEST:  final OOS window — touched ONCE for final SR estimate
```

If TRAIN SR is much higher than VAL/TEST SR (e.g. >2× drop), the alphas are
overfitting the train regime. Either retrain on a more recent window or hunt
for additional orthogonal signals.

## Hypothesis-Driven Alpha Discovery

When the alpha set saturates and parameter sweeps stop yielding new strict-pass
candidates, switch to one-at-a-time hypothesis testing. Parameter sweeps over
decay windows / lookbacks / ratios on the same observables produce alphas in
the same correlated family — they cannot break the orthogonality ceiling.

**Required loop per hypothesis:**

1. **State the economic mechanism** before writing any expression. Why should
   this signal predict returns? What behavioral, microstructural, or
   information-processing inefficiency creates the edge? If you cannot articulate
   the mechanism in one sentence, the test is data-mining.

2. **State why it should be orthogonal to the existing set.** Identify what the
   signal uses that no saved alpha uses — a different observable, operator,
   time horizon, normalization, or aggregation. Orthogonality must be earned
   structurally, not searched for empirically.

3. **Test ONE alpha.** Compute SR / fitness / corr-to-existing on TRAIN only.
   Do not look at VAL or TEST. Do not batch-sweep variants in this step.

4. **Refine based on the single result.**
   - If SR weak but corr low: signal exists but is too noisy — try a different
     normalization or aggregation. Do not just add decay.
   - If SR strong but corr high: signal is in an existing family — change the
     observable, not the formula.
   - If sign is wrong: flip and reinterpret. The mechanism is opposite of what
     you expected, which is itself information.

5. **Combine near-misses multiplicatively.** When multiple hypotheses each pass
   different gates (one passes corr, another passes SR), the rank-product of
   their components often passes all three. Independent noise sources cancel;
   shared signal amplifies. This is the structural breakthrough when single
   hypotheses cannot pass alone.

6. **Outer decay smoothing fixes fitness without lifting correlation IF the
   inner factors are already in different families.** Adding decay to a single
   in-family signal pulls correlation up; adding decay to an orthogonal
   composite preserves the cross-family structure.

**Anti-patterns to avoid:**

- Sweeping decay windows (3, 5, 7, 10) on the same base signal. This is parameter
  search on a single hypothesis — it tells you nothing about whether the
  hypothesis is correct, and exhausts test-set budget.
- Multiplying any signal by reversal (`ts_delta(close, N)`) when the existing set
  is already reversal-dominated. This always pulls correlation toward 1.
- Lowering gates silently when nothing passes. Gates are user-set parameters,
  not ceilings to optimize against.
- Selecting alphas by VAL/TEST performance. The combiner does selection on those
  splits; the discovery step uses TRAIN only.

**When the orthogonal-alpha space is genuinely saturated** (every SR≥gate
candidate ends up corr ≥ cutoff against the existing set), accept the saturation
and report it. Do not relax thresholds without explicit user permission. The
correct response is to either expand the universe (different mcap band, sector,
delay), seek new data (intraday bars, news, earnings dates), or accept the
existing set as the orthogonal capacity ceiling for the current data and
constraints.
