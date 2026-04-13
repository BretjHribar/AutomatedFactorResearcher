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
- **IS Sharpe > 1.0**: Basic signal strength (HARD GATE)
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
1. **Quality gates**: IS Sharpe ≥ 1.0, Mean IC > 0, sub-period stability
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
