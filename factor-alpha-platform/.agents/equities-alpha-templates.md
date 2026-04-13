# Equities Alpha Templates & Signal Guidance

This file provides **high-level guidance** for the equities alpha research agent. It documents known alpha families, key construction principles, and what is currently in the database. It is intentionally less prescriptive than the crypto templates — the agent should explore broadly without being anchored to specific expressions.

**Read this before hypothesizing.** Update it when meaningful new insights are discovered.

---

## Asset Class Context

- **Universe**: TOP1000 US stocks by 20-day average dollar volume (ADV20), rebalanced every 20 trading days
- **Frequency**: Daily bars
- **Data span**: 2016–2026 (~2,552 trading days)
- **Neutralization**: `subindustry` (GICS 8-digit) is the gold standard — removes sector bets
- **Delay**: Always use `delay=1` — yesterday's signal predicts today's return
- **Fee model**: ~1-3 bps one-way. Turnover is less punishing than crypto.
- **Data source**: `data/fmp_cache/matrices/` (126 fields, see DATA_CATALOG.md for full list)

---

## Alpha Families (Equities-Specific)

### Value
Buy cheap relative to fundamentals. Works because valuation spreads revert and value stocks earn a risk premium.

**Key fields**: `earnings_yield`, `book_to_market`, `ev_to_ebitda`, `ev_to_revenue`, `free_cashflow_yield`
**Direction**: Higher yield / lower multiple = long signal

### Momentum
Stocks that outperformed recently tend to continue (6-12 month horizon). Short-term reversal works in the opposite direction (1-5 days).

**Key fields**: `returns`, `log_returns`, `close`, `momentum_20d`, `momentum_60d`
**Key rule**: Skip the most recent month for classic cross-sectional momentum (avoids reversal contamination)

### Quality / Profitability
High-ROE, high-margin, stable-earnings companies tend to earn a systematic premium.

**Key fields**: `roe`, `roa`, `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `asset_turnover`
**Direction**: Higher values = long signal

### Accruals / Earnings Quality
Stocks with higher accruals (earnings not backed by cash) tend to disappoint. The Sloan (1996) accrual anomaly.

**Construction principle**: `rank(subtract(true_divide(net_income, assets), true_divide(cashflow_op, assets)))` — buy low accruals (high cash-backed earnings)

### Investment / Asset Growth
Firms that aggressively invest (grow assets or capex) tend to underperform. Capital is often misallocated.

**Key fields**: `assets`, `capex`, `ts_delta(assets, 252)`, `capex_to_revenue`
**Direction**: Lower asset growth = long signal (negative of asset growth)

### Low Volatility
The low-volatility anomaly: low-risk stocks systematically outperform on a risk-adjusted basis.

**Key fields**: `historical_volatility_20`, `historical_volatility_60`, `parkinson_volatility_20`
**Direction**: Lower vol = long signal (negate before ranking)

### Liquidity / Size
Less liquid / smaller stocks earn a premium as compensation for illiquidity risk.

**Key fields**: `adv20`, `adv60`, `market_cap`, `dollars_traded`
**Direction**: Lower ADV / smaller = long signal (negate before ranking)

### Microstructure / Short-Term
VWAP deviations, intraday momentum, volume confirmed moves.

**Key fields**: `vwap`, `volume`, `returns` (short-lag), `adv20`
**Note**: Higher turnover — ensure fee-adjusted alpha is positive

---

## Key Construction Principles

1. **Rank-normalize before combining.** Raw fundamental values (revenue, assets) have extreme skew. Always wrap in `rank()` or `zscore_cs()` before adding signals together.

2. **Fundamental data is quarterly, forward-filled daily.** Fundamental ratios (P/E, ROE, margins) change once per quarter. They are naturally low-turnover. No need to smooth aggressively.

3. **Sector-relative signals often dominate.** `rank(roe)` across the whole universe may include sector effects. Consider: does this signal apply within sectors or across sectors? Sector-neutralized evaluation is the gold standard.

4. **The 12-1 momentum convention.** For price momentum, compute over the past 12 months but exclude the most recent month:
   - Standard: `ts_rank(close, 252)` minus recent reversal is one approximation
   - Or simply test `ts_delta(close, 252)` with delay=21 and compare

5. **Use `s_log_1p()` for volume and dollar fields.** Volume and ADV have extreme right skew. Log-transform before ranking.

6. **Lookbacks in trading days**: ~5d=1 week, ~21d=1 month, ~63d=1 quarter, ~126d=6 months, ~252d=1 year

7. **Test sector-relative vs absolute.** For fundamentals, test both `rank(signal)` (market-wide) and `rank(zscore within sector)` (sector-relative). The sector-relative version often has higher IC.

8. **Use `--decay N` to trade Sharpe for Fitness.** The global `DECAY` parameter (default 0) applies N-day linear decay to the signal before simulation. This smooths daily positions, reducing turnover. The Fitness gate is `SR * sqrt(|ret_ann| / max(turnover, 0.125))`. Since the denominator is floored at 0.125, decay helps most when signal turnover is **above** 0.125 (e.g. momentum signals at 0.25-0.40). For fundamental signals already below the floor (turnover ~0.06), decay has minimal effect on Fitness. Example: `python eval_alpha_equity.py --expr "..." --decay 5`

---

## Common Failure Modes (Equities-Specific)

| Failure Mode | Symptom | Fix |
|---|---|---|
| **Sector contamination** | IC positive but Sharpe low after sector-neut | Signal is a sector tilt, not stock selection. Use sector-relative version. |
| **Low IC despite high Sharpe** | Driven by outliers, not cross-sectional rank | Signal dominated by a few extreme stocks. Rank-normalize and clip. |
| **Fundamental lag** | Signal appears to predict but is actually lagged | Ensure forward-fill is applied properly — use only data known at time T |
| **Survivorship bias** | Signal looks strong but suspect | Universe is survivorship-bias-free, but verify signal doesn't cherry-pick large surviving firms |
| **Turnover drag** | Looks good pre-fee, bad post-fee | Smooth with longer window or increase lookback to reduce churn |

---

## Debugging Quality Gate Failures

| Gate Failing | Typical Cause | Fix |
|---|---|---|
| **IS Sharpe < 1.0** | Signal too weak overall | Combine with 1-2 complementary orthogonal signals |
| **Mean IC ≈ 0** | Signal is outlier-driven, not cross-sectional | Rank-normalize; check for data issues |
| **Sub-period instability** | Regime-specific signal | Combine with signal that works in the failing sub-period |
| **High correlation with existing alpha** | Variant of something already in DB | Need a structurally different signal family |
| **Negative PnL post-fee** | Turnover too high for 1-3bps fee | Extend lookback, add smoothing, or use fundamental (naturally slow) signals |

---

## Unexplored / Promising Directions

This section should be updated by the agent as research progresses. Start with:

- Analyst estimate revisions (if data becomes available): EPS revision momentum is one of the most powerful equity signals
- Composite quality + value: combine `rank(roe)` with `rank(earnings_yield)` — the "Magic Formula" concept
- Accruals + momentum interaction: low-accrual stocks with positive momentum
- Asset growth reversal: firms that recently cut capex often subsequent outperform
- Earnings quality via cash conversion: `rank(true_divide(cashflow_op, net_income))` — high = better quality
- SG&A efficiency: `rank(negative(ts_delta(true_divide(sga_expense, revenue), 252)))` — improving operating discipline

---

## What Is Currently in the Database

*(Update this section as the agent adds alphas. At the start of each session, run `python eval_alpha_equity.py --scoreboard` and update the notes below.)*

### Active Alphas (as of Trial #31)

| # | Expression (truncated) | SR | Fitness | IC | TO | Family |
|---|---|---|---|---|---|---|
| 1 | `rank(add(negative(rank(subtract(net_income/assets, cashflow_op/assets))), rank(ts_delta(operating_margin,63))))` | +1.621 | 1.09 | +0.0046 | 0.0635 | Accruals + Op Margin |

### Key Research Findings
- **Fundamental quality/accruals signals** work best: low turnover, stable H1/H2, low kurtosis, positive skew
- **Low-vol anomaly is REVERSED** when subindustry-neutralized (sector-level effect only)
- **Momentum (12-1)**: SR ~1.0 but consistently fails PnL Skew gate (~-0.6) due to crash risk
- **Fitness bottleneck for slow fundamental signals**: turnover floor 0.125 means need SR >= ~1.6 for Fitness >= 1.0
- **Decay trick**: For high-turnover signals (>0.125), `--decay N` can lower turnover and boost Fitness at mild SR cost
- **Rolling SR std**: ~0.12-0.17 is typical for daily equity fundamentals (well within the 0.25 gate)

### Unexplored / Promising Directions
- **ROE + EBITDA margin + asset turnover improvement**: 3-factor SR=1.41, Fitness=0.88 -- just below Fitness gate; try with decay or additional component
- **12-1 Momentum with decay**: SR=0.998, but skew=-0.6 fails. Try `--decay 5` or `--decay 10` to see if skew improves
- **FCF yield composite**: `rank(add(rank(free_cashflow_yield), rank(ts_delta(roa, 63))))` - untested
- **EPS growth**: `rank(ts_delta(eps_diluted, 252))` - untested
- **Leverage reduction**: `negative(rank(ts_delta(debt_to_equity, 252)))` - untested
- **ROIC improvement**: `rank(ts_delta(roic, 63))` - untested
