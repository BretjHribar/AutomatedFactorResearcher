# Data Catalog

> Last updated: 2026-02-28
>
> **126 matrices** across **14 categories** · **2.2 GB** on disk · **2,552 trading days** (2016-01-04 → 2026-02-26)

---

## Summary

| Category | Fields | Tickers | Coverage | Disk Size |
|----------|-------:|--------:|---------:|----------:|
| [Price & Volume](#price--volume) | 6 | 7,835 | 57.2% | 322 MB |
| [Returns & Liquidity](#returns--liquidity) | 5 | 7,835 | 56.8–57.4% | 473 MB |
| [Income Statement](#income-statement) | 20 | 2,512 | 88.5–88.7% | 42 MB |
| [Balance Sheet](#balance-sheet) | 24 | 2,512 | 88.0% | 51 MB |
| [Cash Flow](#cash-flow) | 6 | 2,512 | 88.7% | 13 MB |
| [Key Metrics & Valuation](#key-metrics--valuation) | 15 | 2,512 | 51.0–87.1% | 121 MB |
| [Profitability Ratios](#profitability-ratios) | 9 | 2,512 | 84.7–87.7% | 21 MB |
| [Leverage & Solvency](#leverage--solvency) | 5 | 2,512 | 74.1–87.9% | 11 MB |
| [Per-Share Data](#per-share-data) | 6 | 7,835 | 26.1–26.2% | 296 MB |
| [Efficiency & Activity](#efficiency--activity) | 4 | 2,512 | 84.6–88.4% | 8 MB |
| [Historical Volatility](#historical-volatility) | 8 | 2,512 | 78.0–84.7% | 407 MB |
| [Parkinson Volatility](#parkinson-volatility) | 8 | 2,512 | 77.8–84.6% | 407 MB |
| [GICS Classification](#gics-classification) | 5 | 2,512 | 100% | 4 MB |
| [Universe Masks](#universe-masks) | 5 | 7,835 | — | 18 MB |
| **Total** | **126** | — | — | **2,193 MB** |

> **Ticker coverage**: 7,835 total unique tickers (2,512 actively traded + 5,323 delisted). Fundamental data is available for the 2,512 active tickers; price data spans all 7,835.

---

## Price & Volume

6 fields · 7,835 tickers · 322 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `close` | 2552 × 7835 | 57.2% | 47.9 MB | Adjusted closing price |
| `open` | 2552 × 7835 | 57.2% | 48.0 MB | Opening price |
| `high` | 2552 × 7835 | 57.2% | 49.7 MB | Daily high |
| `low` | 2552 × 7835 | 57.2% | 49.4 MB | Daily low |
| `volume` | 2552 × 7835 | 57.2% | 64.6 MB | Daily share volume |
| `vwap` | 2552 × 7835 | 57.2% | 62.2 MB | Volume-weighted average price |

> Coverage is ~57% because delisted stocks have NaN after their final trading day.

---

## Returns & Liquidity

5 fields · 7,835 tickers · 473 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `returns` | 2552 × 7835 | 56.8% | 95.3 MB | Daily simple return `close[t]/close[t-1] - 1` |
| `log_returns` | 2552 × 7835 | 56.8% | 95.9 MB | Daily log return `ln(close[t]/close[t-1])` |
| `adv20` | 2552 × 7835 | 57.4% | 97.8 MB | 20-day average dollar volume |
| `adv60` | 2552 × 7835 | 57.1% | 99.9 MB | 60-day average dollar volume |
| `dollars_traded` | 2552 × 7835 | 57.2% | 83.9 MB | Daily dollar volume `close × volume` |

---

## Income Statement

20 fields · 2,512 tickers · 42 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `revenue` | 2552 × 2512 | 88.7% | 2.2 MB | Total revenue |
| `sales` | 2552 × 2512 | 88.7% | 2.2 MB | Total sales (= revenue) |
| `gross_profit` | 2552 × 2512 | 88.7% | 2.2 MB | Revenue minus COGS |
| `gross_profit_field` | 2552 × 2512 | 88.7% | 2.2 MB | Gross profit (alias) |
| `operating_income` | 2552 × 2512 | 88.7% | 2.2 MB | Operating income (EBIT proxy) |
| `ebit` | 2552 × 2512 | 88.7% | 2.2 MB | Earnings before interest & taxes |
| `ebitda` | 2552 × 2512 | 88.7% | 2.2 MB | Earnings before interest, taxes, depreciation & amortization |
| `income` | 2552 × 2512 | 88.7% | 2.2 MB | Net income |
| `net_income` | 2552 × 2512 | 88.7% | 2.2 MB | Net income (alias) |
| `eps` | 2552 × 2512 | 88.7% | 2.1 MB | Earnings per share (basic) |
| `eps_diluted` | 2552 × 2512 | 88.7% | 2.1 MB | Earnings per share (diluted) |
| `cost_of_revenue` | 2552 × 2512 | 88.7% | 2.1 MB | Cost of goods sold |
| `cogs` | 2552 × 2512 | 88.7% | 2.1 MB | COGS (alias) |
| `operating_expense` | 2552 × 2512 | 88.7% | 2.1 MB | Total operating expenses |
| `sga_expense` | 2552 × 2512 | 88.7% | 2.1 MB | Selling, general & administrative |
| `rd_expense` | 2552 × 2512 | 88.7% | 1.6 MB | Research & development expense |
| `interest_expense` | 2552 × 2512 | 88.5% | 2.0 MB | Interest expense |
| `income_tax` | 2552 × 2512 | 88.7% | 2.1 MB | Income tax expense |
| `depreciation` | 2552 × 2512 | 88.7% | 2.1 MB | Depreciation expense |
| `depre_amort` | 2552 × 2512 | 88.7% | 2.1 MB | Depreciation & amortization |

---

## Balance Sheet

24 fields · 2,512 tickers · 51 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `assets` | 2552 × 2512 | 88.0% | 2.2 MB | Total assets |
| `total_assets` | 2552 × 2512 | 88.0% | 2.2 MB | Total assets (alias) |
| `assets_curr` | 2552 × 2512 | 88.0% | 2.2 MB | Current assets |
| `cash` | 2552 × 2512 | 88.0% | 2.2 MB | Cash & cash equivalents |
| `receivables` | 2552 × 2512 | 88.0% | 2.1 MB | Accounts receivable |
| `receivable` | 2552 × 2512 | 88.0% | 2.1 MB | Accounts receivable (alias) |
| `inventory` | 2552 × 2512 | 88.0% | 1.8 MB | Inventory |
| `goodwill` | 2552 × 2512 | 88.0% | 1.7 MB | Goodwill |
| `intangibles` | 2552 × 2512 | 88.0% | 1.9 MB | Intangible assets |
| `ppe_net` | 2552 × 2512 | 88.0% | 2.1 MB | Property, plant & equipment (net) |
| `ppent` | 2552 × 2512 | 88.0% | 2.1 MB | PP&E net (alias) |
| `liabilities` | 2552 × 2512 | 88.0% | 2.2 MB | Total liabilities |
| `total_liabilities` | 2552 × 2512 | 88.0% | 2.2 MB | Total liabilities (alias) |
| `liabilities_curr` | 2552 × 2512 | 88.0% | 2.2 MB | Current liabilities |
| `payables` | 2552 × 2512 | 88.0% | 2.1 MB | Accounts payable |
| `debt` | 2552 × 2512 | 88.0% | 2.1 MB | Total debt |
| `total_debt` | 2552 × 2512 | 88.0% | 2.1 MB | Total debt (alias) |
| `debt_lt` | 2552 × 2512 | 88.0% | 2.1 MB | Long-term debt |
| `equity` | 2552 × 2512 | 88.0% | 2.2 MB | Total stockholders' equity |
| `total_equity` | 2552 × 2512 | 88.0% | 2.2 MB | Total equity (alias) |
| `retained_earnings` | 2552 × 2512 | 88.0% | 2.2 MB | Retained earnings |
| `working_capital` | 2552 × 2512 | 88.0% | 2.2 MB | Current assets − current liabilities |
| `net_debt` | 2552 × 2512 | 88.0% | 2.2 MB | Total debt − cash (from key metrics) |
| `net_debt_calc` | 2552 × 2512 | 88.0% | 2.2 MB | Total debt − cash (calculated) |

---

## Cash Flow

6 fields · 2,512 tickers · 13 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `cashflow` | 2552 × 2512 | 88.7% | 2.2 MB | Net cash from all activities |
| `cashflow_op` | 2552 × 2512 | 88.7% | 2.2 MB | Cash from operations |
| `cashflow_invst` | 2552 × 2512 | 88.7% | 2.1 MB | Cash from investing |
| `free_cashflow` | 2552 × 2512 | 88.7% | 2.2 MB | Free cash flow (operating − capex) |
| `capex` | 2552 × 2512 | 88.7% | 2.1 MB | Capital expenditures |
| `stock_repurchase` | 2552 × 2512 | 88.7% | 1.8 MB | Share buybacks |

---

## Key Metrics & Valuation

15 fields · 2,512 tickers · 121 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `market_cap` | 2552 × 2512 | 87.1% | 2.2 MB | Market capitalization |
| `cap` | 2552 × 2512 | 87.1% | 2.2 MB | Market cap (alias) |
| `market_cap_metric` | 2552 × 2512 | 87.1% | 2.2 MB | Market cap (from key metrics) |
| `enterprise_value` | 2552 × 2512 | 87.1% | 2.2 MB | Enterprise value |
| `invested_capital` | 2552 × 2512 | 87.1% | 2.1 MB | Invested capital |
| `current_ratio` | 2552 × 2512 | 87.1% | 2.2 MB | Current assets / current liabilities |
| `inventory_turnover` | 2552 × 2512 | 51.0% | 1.9 MB | COGS / average inventory |
| `pe_ratio` | 2552 × 7835 | 26.4% | 38.8 MB | Price / EPS |
| `pb_ratio` | 2552 × 2512 | 86.4% | 3.1 MB | Price / book value |
| `ev_to_ebitda` | 2552 × 2512 | 86.2% | 3.0 MB | Enterprise value / EBITDA |
| `ev_to_revenue` | 2552 × 2512 | 83.2% | 3.0 MB | Enterprise value / revenue |
| `ev_to_fcf` | 2552 × 2512 | 86.5% | 2.9 MB | Enterprise value / free cash flow |
| `earnings_yield` | 2552 × 7835 | 26.5% | 48.9 MB | EPS / price (inverse P/E) |
| `free_cashflow_yield` | 2552 × 2512 | 86.7% | 3.1 MB | FCF / market cap |
| `book_to_market` | 2552 × 2512 | 86.2% | 3.1 MB | Book value / market cap |

---

## Profitability Ratios

9 fields · 2,512 tickers · 21 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `roe` | 2552 × 2512 | 87.7% | 2.3 MB | Return on equity |
| `return_equity` | 2552 × 2512 | 87.7% | 2.3 MB | Return on equity (alias) |
| `roa` | 2552 × 2512 | 87.0% | 2.3 MB | Return on assets |
| `return_assets` | 2552 × 2512 | 87.0% | 2.3 MB | Return on assets (alias) |
| `gross_margin` | 2552 × 2512 | 84.7% | 2.2 MB | Gross profit / revenue |
| `operating_margin` | 2552 × 2512 | 84.7% | 2.3 MB | Operating income / revenue |
| `net_margin` | 2552 × 2512 | 84.7% | 2.3 MB | Net income / revenue |
| `ebitda_margin` | 2552 × 2512 | 84.7% | 2.3 MB | EBITDA / revenue |
| `asset_turnover` | 2552 × 2512 | 87.0% | 2.3 MB | Revenue / total assets |

---

## Leverage & Solvency

5 fields · 2,512 tickers · 11 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `debt_to_equity` | 2552 × 2512 | 87.9% | 2.3 MB | Total debt / equity |
| `debt_to_assets` | 2552 × 2512 | 87.1% | 2.3 MB | Total debt / total assets |
| `cash_ratio` | 2552 × 2512 | 84.9% | 2.3 MB | Cash / current liabilities |
| `quick_ratio` | 2552 × 2512 | 84.9% | 2.3 MB | (Cash + receivables) / current liabilities |
| `interest_coverage` | 2552 × 2512 | 74.1% | 2.1 MB | EBIT / interest expense |

---

## Per-Share Data

6 fields · 7,835 tickers · 296 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `sharesout` | 2552 × 7835 | 26.2% | 49.2 MB | Shares outstanding |
| `bookvalue_ps` | 2552 × 7835 | 26.1% | 50.0 MB | Book value per share |
| `fcf_per_share` | 2552 × 7835 | 26.1% | 49.8 MB | Free cash flow per share |
| `revenue_per_share` | 2552 × 7835 | 26.1% | 48.4 MB | Revenue per share |
| `sales_ps` | 2552 × 7835 | 26.1% | 48.4 MB | Sales per share (= revenue_per_share) |
| `tangible_book_per_share` | 2552 × 7835 | 26.1% | 50.0 MB | Tangible book value per share |

---

## Efficiency & Activity

4 fields · 2,512 tickers · 8 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `capex_to_revenue` | 2552 × 2512 | 84.6% | 2.2 MB | Capital expenditure / revenue |
| `rd_to_revenue` | 2552 × 2512 | 84.7% | 1.7 MB | R&D expense / revenue |
| `sga_to_revenue` | 2552 × 2512 | 84.7% | 2.2 MB | SG&A expense / revenue |
| `cash_conversion_ratio` | 2552 × 2512 | 88.4% | 2.3 MB | Operating cash flow / EBITDA |

---

## Historical Volatility

8 fields · 2,512 tickers · 407 MB

Annualized historical volatility computed from daily returns at multiple lookback windows.

| Field | Shape | Coverage | Size | Window |
|-------|-------|----------|------|--------|
| `historical_volatility_10` | 2552 × 2512 | 84.7% | 52.6 MB | 10-day |
| `historical_volatility_20` | 2552 × 2512 | 84.3% | 52.4 MB | 20-day |
| `historical_volatility_30` | 2552 × 2512 | 83.9% | 52.2 MB | 30-day |
| `historical_volatility_60` | 2552 × 2512 | 82.7% | 51.4 MB | 60-day |
| `historical_volatility_90` | 2552 × 2512 | 81.5% | 50.7 MB | 90-day |
| `historical_volatility_120` | 2552 × 2512 | 80.4% | 50.0 MB | 120-day |
| `historical_volatility_150` | 2552 × 2512 | 79.2% | 49.4 MB | 150-day |
| `historical_volatility_180` | 2552 × 2512 | 78.0% | 48.6 MB | 180-day |

---

## Parkinson Volatility

8 fields · 2,512 tickers · 407 MB

Parkinson volatility estimator using high-low range, more efficient than close-to-close.

| Field | Shape | Coverage | Size | Window |
|-------|-------|----------|------|--------|
| `parkinson_volatility_10` | 2552 × 2512 | 84.6% | 52.5 MB | 10-day |
| `parkinson_volatility_20` | 2552 × 2512 | 84.2% | 52.3 MB | 20-day |
| `parkinson_volatility_30` | 2552 × 2512 | 83.8% | 52.1 MB | 30-day |
| `parkinson_volatility_60` | 2552 × 2512 | 82.6% | 51.4 MB | 60-day |
| `parkinson_volatility_90` | 2552 × 2512 | 81.4% | 50.7 MB | 90-day |
| `parkinson_volatility_120` | 2552 × 2512 | 80.2% | 50.0 MB | 120-day |
| `parkinson_volatility_150` | 2552 × 2512 | 79.0% | 49.3 MB | 150-day |
| `parkinson_volatility_180` | 2552 × 2512 | 77.8% | 48.5 MB | 180-day |

---

## GICS Classification

5 fields · 2,512 tickers · 4 MB

| Field | Shape | Coverage | Size | Description |
|-------|-------|----------|------|-------------|
| `sector` | 2552 × 2512 | 100% | 1.4 MB | 2-digit GICS sector code |
| `industry` | 2552 × 2512 | 100% | 1.4 MB | 6-digit GICS industry code |
| `subindustry` | 2552 × 2512 | 100% | 1.4 MB | 8-digit GICS sub-industry code |
| `_sector_groups` | 2512 × 1 | 100% | 0.0 MB | Sector group lookup |
| `_industry_groups` | 2512 × 1 | 100% | 0.0 MB | Industry group lookup |

> Full GICS hierarchy stored in `classifications.json` with 11 sectors, ~25 industry groups, ~69 industries, and 106 sub-industries.

---

## Universe Masks

5 fields · 7,835 tickers · 18 MB

Boolean masks indicating daily membership. Universes are ranked by **ADV20** (20-day average dollar volume) and rebalanced every 20 trading days.

| Universe | Shape | Avg/Day | Unique (10yr) | Size | Description |
|----------|-------|--------:|-------------:|------|-------------|
| `TOP200` | 2552 × 7835 | 198 | 745 | 3.5 MB | Top 200 by liquidity |
| `TOP500` | 2552 × 7835 | 496 | 1,529 | 3.5 MB | Top 500 by liquidity |
| `TOP1000` | 2552 × 7835 | 992 | 2,717 | 3.5 MB | Top 1000 by liquidity |
| `TOP2000` | 2552 × 7835 | 1,984 | 4,424 | 3.6 MB | Top 2000 by liquidity |
| `TOP3000` | 2552 × 7835 | 2,973 | 6,576 | 3.6 MB | Top 3000 by liquidity |

> **Survivorship-bias free**: Universes include delisted stocks in the periods when they were actively traded. The "Unique" column shows total distinct tickers that appeared in each universe over the full 10-year history.

---

## Split & Dividend Handling

### Stock Splits

FMP's historical price endpoint returns **retroactively split-adjusted prices**. All OHLC prices are adjusted backward in time when a split occurs. This has been verified:

| Stock | Split | Date | Pre-Split Close | Post-Split Close | Adjusted? |
|-------|-------|------|----------------:|----------------:|-----------|
| AAPL | 4:1 | 2020-08-28 | $125.01 | $124.81 | ✅ Continuous |
| TSLA | 3:1 | 2022-08-25 | $296.07 | $288.09 | ✅ Continuous |

No manual split adjustment is needed — the data is clean out of the box.

### Dividends

> [!WARNING]
> **Known gap vs. WorldQuant BRAIN:** WQ BRAIN's `close` field is *"carefully adjusted for corporate actions such as dividends, splits, and reverse splits"* ([source](https://platform.worldquantbrain.com/)). Our FMP price data is **split-adjusted but NOT dividend-adjusted**. This means:

- Our `returns` = price return only (`close[t]/close[t-1] - 1`), excluding dividend income
- WQ BRAIN's `close` includes retroactive dividend adjustments, so their returns capture total return
- For most cross-sectional ranking alphas, this gap is small (dividends are ~1.5% annualized across the universe and relatively uniform within sectors)
- The impact on Sharpe replication is likely <0.1 — consistent with our observed gap of 0.06

**To fix this properly**, we would need to:
1. Download dividend history from FMP (`/historical-price-full/stock_dividend`)
2. Compute adjustment factors for each ex-dividend date
3. Apply retroactive adjustments to all OHLC prices before that date
4. Rebuild all derived matrices (`returns`, `log_returns`, `adv20`, etc.)

---

## Already Downloaded but Not Extracted into Matrices

We cache the full FMP response for each fundamental type, but only extract a subset into Parquet matrices. Below is the complete gap — **all of these fields are already on disk** in per-ticker Parquet files under `data/fmp_cache/{income,balance,cashflow,metrics}/` and just need to be added to `build_matrices()`.

### Income Statement — 38 available, 13 extracted (25 unused)

| FMP Field | Status | Priority | Description |
|-----------|--------|----------|-------------|
| `revenue` | ✅ Extracted | — | |
| `costOfRevenue` | ✅ Extracted | — | |
| `grossProfit` | ✅ Extracted | — | |
| `operatingIncome` | ✅ Extracted | — | |
| `ebitda` | ✅ Extracted | — | |
| `netIncome` | ✅ Extracted | — | |
| `eps` | ✅ Extracted | — | |
| `epsDiluted` | ✅ Extracted | — | |
| `researchAndDevelopmentExpenses` | ✅ Extracted | — | |
| `sellingGeneralAndAdministrativeExpenses` | ✅ Extracted | — | |
| `interestExpense` | ✅ Extracted | — | |
| `incomeTaxExpense` | ✅ Extracted | — | |
| `incomeBeforeTax` | ❌ Missing | 🔴 High | Pre-tax income — needed for effective tax rate |
| `interestIncome` | ❌ Missing | 🔴 High | Net interest calc, financial sector alphas |
| `ebit` | ❌ Missing | 🟡 Med | Direct EBIT (we compute from operatingIncome) |
| `depreciationAndAmortization` | ❌ Missing | 🟡 Med | Direct D&A from income stmt |
| `generalAndAdministrativeExpenses` | ❌ Missing | 🟡 Med | G&A component of SGA |
| `sellingAndMarketingExpenses` | ❌ Missing | 🟡 Med | S&M component of SGA |
| `operatingExpenses` | ❌ Missing | 🟡 Med | Total opex |
| `costAndExpenses` | ❌ Missing | 🟡 Med | Total cost + expenses |
| `netIncomeFromContinuingOperations` | ❌ Missing | 🟡 Med | Core earnings quality |
| `netIncomeFromDiscontinuedOperations` | ❌ Missing | 🟡 Med | One-off items signal |
| `totalOtherIncomeExpensesNet` | ❌ Missing | 🟡 Med | Non-operating quality |
| `nonOperatingIncomeExcludingInterest` | ❌ Missing | 🟡 Med | Non-core income |
| `netInterestIncome` | ❌ Missing | 🟢 Low | Net interest income/expense |
| `otherExpenses` | ❌ Missing | 🟢 Low | Other expenses |
| `weightedAverageShsOut` | ❌ Missing | 🟢 Low | Avg shares (already from balance) |
| `weightedAverageShsOutDil` | ❌ Missing | 🟡 Med | Diluted shares — dilution signal |
| `otherAdjustmentsToNetIncome` | ❌ Missing | 🟢 Low | Adjustments |
| `netIncomeDeductions` | ❌ Missing | 🟢 Low | Deductions |
| `bottomLineNetIncome` | ❌ Missing | 🟢 Low | Bottom line net income |

### Balance Sheet — 58 available, 16 extracted (42 unused)

| FMP Field | Status | Priority | Description |
|-----------|--------|----------|-------------|
| `totalAssets` | ✅ Extracted | — | |
| `totalCurrentAssets` | ✅ Extracted | — | |
| `cashAndCashEquivalents` | ✅ Extracted | — | |
| `netReceivables` | ✅ Extracted | — | |
| `inventory` | ✅ Extracted | — | |
| `goodwill` | ✅ Extracted | — | |
| `intangibleAssets` | ✅ Extracted | — | |
| `propertyPlantEquipmentNet` | ✅ Extracted | — | |
| `totalLiabilities` | ✅ Extracted | — | |
| `totalCurrentLiabilities` | ✅ Extracted | — | |
| `accountPayables` | ✅ Extracted | — | |
| `totalDebt` | ✅ Extracted | — | |
| `totalStockholdersEquity` | ✅ Extracted | — | |
| `retainedEarnings` | ✅ Extracted | — | |
| `netDebt` | ✅ Extracted | — | |
| `weightedAverageShsOut` | ✅ Extracted | — | |
| `shortTermInvestments` | ❌ Missing | 🔴 High | Liquid investments |
| `cashAndShortTermInvestments` | ❌ Missing | 🔴 High | Broader cash measure |
| `shortTermDebt` | ❌ Missing | 🔴 High | ST vs LT debt decomposition |
| `longTermDebt` | ❌ Missing | 🔴 High | Already have totalDebt but not LT split |
| `longTermInvestments` | ❌ Missing | 🔴 High | LT investment assets |
| `accruedExpenses` | ❌ Missing | 🔴 High | Accrual-based earnings quality |
| `deferredRevenue` | ❌ Missing | 🔴 High | Revenue backlog signal |
| `capitalLeaseObligations` | ❌ Missing | 🟡 Med | Lease liabilities |
| `capitalLeaseObligationsCurrent` | ❌ Missing | 🟡 Med | Current lease portion |
| `capitalLeaseObligationsNonCurrent` | ❌ Missing | 🟡 Med | Non-current lease portion |
| `commonStock` | ❌ Missing | 🟡 Med | Common stock value |
| `additionalPaidInCapital` | ❌ Missing | 🟡 Med | APIC — equity structure |
| `treasuryStock` | ❌ Missing | 🔴 High | Buyback intensity signal |
| `accumulatedOtherComprehensiveIncomeLoss` | ❌ Missing | 🟡 Med | AOCI — unrealized gains |
| `minorityInterest` | ❌ Missing | 🟡 Med | Non-controlling interest |
| `preferredStock` | ❌ Missing | 🟢 Low | Preferred equity |
| `accountsReceivables` | ❌ Missing | 🟡 Med | Alt. to netReceivables |
| `otherReceivables` | ❌ Missing | 🟢 Low | Other receivables |
| `prepaids` | ❌ Missing | 🟢 Low | Prepaid expenses |
| `otherCurrentAssets` | ❌ Missing | 🟢 Low | Other current |
| `totalNonCurrentAssets` | ❌ Missing | 🟡 Med | Non-current asset total |
| `taxAssets` | ❌ Missing | 🟡 Med | Deferred tax assets |
| `otherNonCurrentAssets` | ❌ Missing | 🟢 Low | Other non-current |
| `goodwillAndIntangibleAssets` | ❌ Missing | 🟢 Low | Combined (have separately) |
| `totalPayables` | ❌ Missing | 🟡 Med | All payables combined |
| `otherPayables` | ❌ Missing | 🟢 Low | Other payables |
| `taxPayables` | ❌ Missing | 🟡 Med | Tax payables |
| `deferredRevenueNonCurrent` | ❌ Missing | 🟢 Low | LT deferred revenue |
| `deferredTaxLiabilitiesNonCurrent` | ❌ Missing | 🟡 Med | Deferred tax liabilities |
| `otherNonCurrentLiabilities` | ❌ Missing | 🟢 Low | Other non-current liab |
| `totalNonCurrentLiabilities` | ❌ Missing | 🟡 Med | Non-current liab total |
| `otherLiabilities` | ❌ Missing | 🟢 Low | Other liabilities |
| `otherTotalStockholdersEquity` | ❌ Missing | 🟢 Low | Other equity |
| `totalEquity` | ❌ Missing | 🟢 Low | Total equity (incl. minority) |
| `totalLiabilitiesAndTotalEquity` | ❌ Missing | 🟢 Low | Accounting identity check |
| `totalInvestments` | ❌ Missing | 🟡 Med | All investments |
| `otherAssets` | ❌ Missing | 🟢 Low | Other assets |

### Cash Flow — 44 available, 7 extracted (37 unused)

| FMP Field | Status | Priority | Description |
|-----------|--------|----------|-------------|
| `operatingCashFlow` | ✅ Extracted | — | |
| `capitalExpenditure` | ✅ Extracted | — | |
| `freeCashFlow` | ✅ Extracted | — | |
| `depreciationAndAmortization` | ✅ Extracted | — | |
| `commonStockRepurchased` | ✅ Extracted | — | |
| `debtRepayment` | ✅ Extracted | — | |
| `dividendsPaid` | ✅ Extracted | — | |
| `stockBasedCompensation` | ❌ Missing | 🔴 High | SBC — major earnings quality signal |
| `changeInWorkingCapital` | ❌ Missing | 🔴 High | Accrual quality, earnings manipulation |
| `acquisitionsNet` | ❌ Missing | 🔴 High | M&A activity signal |
| `deferredIncomeTax` | ❌ Missing | 🔴 High | Tax aggressiveness signal |
| `accountsReceivables` | ❌ Missing | 🟡 Med | CF from AR changes |
| `inventory` | ❌ Missing | 🟡 Med | CF from inventory changes |
| `accountsPayables` | ❌ Missing | 🟡 Med | CF from AP changes |
| `otherWorkingCapital` | ❌ Missing | 🟢 Low | Other WC changes |
| `otherNonCashItems` | ❌ Missing | 🟡 Med | Non-cash adjustments |
| `netCashProvidedByOperatingActivities` | ❌ Missing | 🟢 Low | = operatingCashFlow |
| `investmentsInPropertyPlantAndEquipment` | ❌ Missing | 🟢 Low | = capitalExpenditure |
| `purchasesOfInvestments` | ❌ Missing | 🟡 Med | Investment purchases |
| `salesMaturitiesOfInvestments` | ❌ Missing | 🟡 Med | Investment sales |
| `otherInvestingActivities` | ❌ Missing | 🟢 Low | Other investing |
| `netCashProvidedByInvestingActivities` | ❌ Missing | 🟡 Med | Total investing CF |
| `netDebtIssuance` | ❌ Missing | 🔴 High | Net debt issuance — leverage signal |
| `longTermNetDebtIssuance` | ❌ Missing | 🟡 Med | LT debt issuance |
| `shortTermNetDebtIssuance` | ❌ Missing | 🟡 Med | ST debt issuance |
| `netStockIssuance` | ❌ Missing | 🔴 High | Net equity issuance — dilution signal |
| `netCommonStockIssuance` | ❌ Missing | 🟡 Med | Common stock issuance |
| `commonStockIssuance` | ❌ Missing | 🟡 Med | Gross issuance |
| `netPreferredStockIssuance` | ❌ Missing | 🟢 Low | Preferred stock |
| `commonDividendsPaid` | ❌ Missing | 🟡 Med | Common dividends |
| `preferredDividendsPaid` | ❌ Missing | 🟢 Low | Preferred dividends |
| `netDividendsPaid` | ❌ Missing | 🟡 Med | Net dividends |
| `otherFinancingActivities` | ❌ Missing | 🟢 Low | Other financing |
| `netCashProvidedByFinancingActivities` | ❌ Missing | 🟡 Med | Total financing CF |
| `effectOfForexChangesOnCash` | ❌ Missing | 🟡 Med | FX impact on cash |
| `netChangeInCash` | ❌ Missing | 🟡 Med | Net cash change |
| `cashAtEndOfPeriod` | ❌ Missing | 🟡 Med | Period end cash |
| `cashAtBeginningOfPeriod` | ❌ Missing | 🟡 Med | Period start cash |
| `interestPaid` | ❌ Missing | 🔴 High | Interest paid (cash basis) |
| `incomeTaxesPaid` | ❌ Missing | 🔴 High | Taxes paid (cash basis) |

### Key Metrics — 44 available, ~12 extracted (32 unused)

| FMP Field | Status | Priority | Description |
|-----------|--------|----------|-------------|
| `marketCap` | ✅ Extracted | — | |
| `enterpriseValue` | ✅ Extracted | — | |
| `currentRatio` | ✅ Extracted | — | |
| `investedCapital` | ✅ Extracted | — | |
| `workingCapital` | ✅ Extracted | — | |
| `returnOnAssets` | ✅ Extracted | — | |
| `returnOnEquity` | ✅ Extracted | — | |
| `earningsYield` | ✅ Extracted | — | |
| `freeCashFlowYield` | ✅ Extracted | — | |
| `evToEBITDA` | ✅ Extracted | — | |
| `evToSales` | ✅ Extracted | — | |
| `evToFreeCashFlow` | ✅ Extracted | — | |
| `returnOnInvestedCapital` | ❌ Missing | 🔴 High | ROIC — best profitability measure |
| `returnOnCapitalEmployed` | ❌ Missing | 🔴 High | ROCE — capital efficiency |
| `returnOnTangibleAssets` | ❌ Missing | 🟡 Med | Return on tangible assets |
| `operatingReturnOnAssets` | ❌ Missing | 🟡 Med | Operating ROA |
| `incomeQuality` | ❌ Missing | 🔴 High | OCF / Net Income — accrual quality |
| `grahamNumber` | ❌ Missing | 🟡 Med | Graham value metric |
| `grahamNetNet` | ❌ Missing | 🟡 Med | Graham net-net value |
| `netDebtToEBITDA` | ❌ Missing | 🔴 High | Leverage metric |
| `taxBurden` | ❌ Missing | 🟡 Med | DuPont decomposition |
| `interestBurden` | ❌ Missing | 🟡 Med | DuPont decomposition |
| `capexToOperatingCashFlow` | ❌ Missing | 🟡 Med | Investment intensity |
| `capexToDepreciation` | ❌ Missing | 🔴 High | Maintenance vs growth capex |
| `stockBasedCompensationToRevenue` | ❌ Missing | 🔴 High | SBC dilution intensity |
| `intangiblesToTotalAssets` | ❌ Missing | 🟡 Med | Asset quality |
| `daysOfSalesOutstanding` | ❌ Missing | 🔴 High | DSO — receivables efficiency |
| `daysOfPayablesOutstanding` | ❌ Missing | 🔴 High | DPO — payables management |
| `daysOfInventoryOutstanding` | ❌ Missing | 🔴 High | DIO — inventory management |
| `operatingCycle` | ❌ Missing | 🔴 High | DSO + DIO |
| `cashConversionCycle` | ❌ Missing | 🔴 High | DSO + DIO - DPO |
| `freeCashFlowToEquity` | ❌ Missing | 🟡 Med | FCFE |
| `freeCashFlowToFirm` | ❌ Missing | 🟡 Med | FCFF |
| `tangibleAssetValue` | ❌ Missing | 🟡 Med | Tangible asset value |
| `netCurrentAssetValue` | ❌ Missing | 🟡 Med | NCAV (net-net) |
| `evToOperatingCashFlow` | ❌ Missing | 🟡 Med | EV/OCF valuation |
| `averageReceivables` | ❌ Missing | 🟢 Low | Avg AR |
| `averagePayables` | ❌ Missing | 🟢 Low | Avg AP |
| `averageInventory` | ❌ Missing | 🟢 Low | Avg inventory |
| `researchAndDevelopementToRevenue` | ❌ Missing | 🟢 Low | R&D/Rev (already derived) |
| `salesGeneralAndAdministrativeToRevenue` | ❌ Missing | 🟢 Low | SGA/Rev (already derived) |
| `capexToRevenue` | ❌ Missing | 🟢 Low | Capex/Rev (already derived) |

### Summary of Extraction Gap

| Statement | FMP Fields | Extracted | Missing | Missing 🔴 High |
|-----------|-----------|-----------|---------|-----------------|
| Income | 38 | 13 | 25 | 2 |
| Balance Sheet | 58 | 16 | 42 | 7 |
| Cash Flow | 44 | 7 | 37 | 8 |
| Key Metrics | 44 | ~12 | 32 | 11 |
| **Total** | **184** | **~48** | **136** | **28** |

> [!TIP]
> **Quick win**: The 28 🔴 High priority fields are already cached on disk — they just need to be added to the `build_matrices()` function in `bulk_download.py` and rerun. No new API calls required.

---

## Missing Data — Available from FMP but Not Yet Downloaded

The following datasets are available from our data source (Financial Modeling Prep) and would be **highly relevant** for alpha research. They are listed in order of expected impact on alpha discovery:

### High Priority (directly used in WQ BRAIN alphas)

| Dataset | FMP Endpoint | WQ Field Equivalent | Impact |
|---------|-------------|-------------------|--------|
| **Analyst Estimates** (EPS, Revenue) | `/analyst-estimates` | `est_eps`, `est_revenue`, `est_ebitda` | Earnings surprise, revision momentum |
| **Financial Growth Rates** | `/financial-growth` | `revenue_growth`, `eps_growth` | Growth factor, acceleration |
| **Dividends** (amount, yield, dates) | `/historical-price-full/stock_dividend` | `dividend_yield`, `div_amount` | Dividend yield factor, ex-date effects |
| **Stock Splits** (dates, ratios) | `/stock-split-calendar` | — | Correct for split artifacts |
| **Analyst Recommendations** | `/analyst-stock-recommendations` | `analyst_rating` | Consensus change signals |

### Medium Priority (useful for alternative alphas)

| Dataset | FMP Endpoint | WQ Field Equivalent | Impact |
|---------|-------------|-------------------|--------|
| **Insider Trading** (buy/sell transactions) | `/insider-trading` | `insider_buy`, `insider_sell` | Insider sentiment signal |
| **Institutional Ownership** (13F filings) | `/institutional-ownership` | `inst_ownership_pct` | Ownership concentration |
| **ESG Scores** | `/esg-environmental-social-governance-data` | `esg_score` | ESG factor tilt |
| **Social Sentiment** | `/social-sentiments` | — | Sentiment alpha |
| **Price Targets** | `/price-target` | `price_target_consensus` | Target vs. current price ratio |

### Lower Priority (niche but available)

| Dataset | FMP Endpoint | Description |
|---------|-------------|-------------|
| **SEC Filings** (10-K, 10-Q dates) | `/sec_filings` | Filing date event signals |
| **Earnings Calendar** | `/earning_calendar` | Event-driven pre/post earnings |
| **IPO Calendar** | `/ipo_calendar` | IPO pipeline effects |
| **Economic Calendar** | `/economic_calendar` | Macro regime conditioning |
| **Sector Performance** | `/sector-performance` | Sector momentum benchmarks |
| **Treasury Rates** | `/treasury` | Risk-free rate, yield curve |
| **Commodities** | `/commodities` | Factor conditioning |

> [!TIP]
> **Analyst Estimates** and **Financial Growth Rates** would be the highest-impact additions. Earnings surprise (`actual_eps - est_eps`) and estimate revision momentum (`delta(est_eps, 30)`) are among the most profitable alpha families in systematic equity trading.

---

## Data Source

All data sourced from [Financial Modeling Prep (FMP)](https://financialmodelingprep.com/).

- **Prices**: Daily OHLCV, adjusted for splits and dividends
- **Fundamentals**: Quarterly income statement, balance sheet, cash flow, key metrics — forward-filled daily
- **Classifications**: FMP industry → GICS mapping via `build_gics_classifications.py`
- **Delisted Stocks**: Downloaded from FMP's delisted companies endpoint via `expand_universe.py`
- **Update frequency**: Manual re-run of `bulk_download.py`

### Storage Format

All matrices are stored as **Apache Parquet** files with:
- **Index**: `DatetimeIndex` (trading days only)
- **Columns**: Ticker symbols (string)
- **Values**: `float64`, `NaN` for missing
- **Compression**: Snappy (default Parquet)

Path: `data/fmp_cache/matrices/<field_name>.parquet`
