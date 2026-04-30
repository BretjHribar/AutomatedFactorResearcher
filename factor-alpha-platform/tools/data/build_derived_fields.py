"""
Compute derived financial ratio matrices from raw FMP fundamentals.
Creates all the WorldQuant-standard ratio/per-share fields that were
previously empty placeholders.

Raw fields we have (88% coverage):
  - close, market_cap/cap, volume, vwap, etc.
  - eps, revenue, income, equity, assets, debt, ebitda, etc.
  - cashflow_op, free_cashflow, enterprise_value, etc.

Derived fields to create:
  - sharesout = market_cap / close (millions of shares)
  - bookvalue_ps = equity / sharesout
  - eps_diluted (copy from eps if not available)
  - pe_ratio = close / eps
  - pb_ratio = close / bookvalue_ps  (price-to-book)
  - ev_to_ebitda = enterprise_value / ebitda
  - roe / return_equity = income / equity
  - roa / return_assets = income / assets
  - debt_to_equity = total_debt / equity
  - fcf_per_share = free_cashflow / sharesout
  - revenue_per_share / sales_ps = revenue / sharesout
  - dividend_yield (from cashflow dividends if available)
  - tangible_book_per_share = (equity - goodwill - intangibles) / sharesout
  - current_ratio (already exists at 87%, verify)
  - inventory_turnover (already exists at 51%)
  - gross_margin = gross_profit / revenue
  - operating_margin = operating_income / revenue
  - net_margin = income / revenue
  - debt_to_assets = total_debt / assets
  - cash_ratio = cash / liabilities_curr
  - quick_ratio = (assets_curr - inventory) / liabilities_curr
  - asset_turnover = revenue / assets
  - capex_to_revenue = capex / revenue
  - rd_to_revenue = rd_expense / revenue
  - interest_coverage = ebit / interest_expense
  - free_cashflow_yield = free_cashflow / market_cap
  - earnings_yield = eps / close (inverse PE)
"""
import os
import numpy as np
import pandas as pd

MATRICES_DIR = "data/fmp_cache/matrices"

def load(name):
    path = os.path.join(MATRICES_DIR, f"{name}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def save(name, df):
    path = os.path.join(MATRICES_DIR, f"{name}.parquet")
    df.to_parquet(path)
    nn = df.notna().sum().sum()
    total = df.shape[0] * df.shape[1]
    pct = nn / total * 100 if total > 0 else 0
    print(f"  ✅ {name:35s} {pct:5.1f}% coverage  ({df.shape[0]}×{df.shape[1]})")

def safe_divide(num, den, replace_inf=True):
    """Division that handles zeros and infinities cleanly."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / den
    if replace_inf:
        result = result.replace([np.inf, -np.inf], np.nan)
    return result

def load_any(*names):
    """Load the first available field from a list of names."""
    for n in names:
        df = load(n)
        if df is not None:
            return df
    return None

# ── Load raw fields ──────────────────────────────────────────────────
print("Loading raw matrices...")
close = load("close")
market_cap = load_any("market_cap", "cap")
eps = load("eps")
revenue = load_any("revenue", "sales")
income = load_any("income", "net_income")
equity = load_any("equity", "total_equity")
assets = load_any("assets", "total_assets")
debt = load_any("debt", "total_debt")
ebitda = load("ebitda")
ebit = load("ebit")
enterprise_value = load("enterprise_value")
cashflow_op = load("cashflow_op")
free_cashflow = load("free_cashflow")
gross_profit = load_any("gross_profit", "gross_profit_field")
operating_income = load("operating_income")
goodwill = load("goodwill")
intangibles = load("intangibles")
assets_curr = load("assets_curr")
liabilities_curr = load("liabilities_curr")
inventory = load("inventory")
cash = load("cash")
interest_expense = load("interest_expense")
capex = load("capex")
rd_expense = load("rd_expense")
volume = load("volume")
depre_amort = load_any("depre_amort", "depreciation")
sga = load("sga_expense")
retained_earnings = load("retained_earnings")
debt_lt = load("debt_lt")

print(f"  Loaded {sum(1 for x in [close, market_cap, eps, revenue, income, equity, assets, debt, ebitda, ebit, enterprise_value, cashflow_op, free_cashflow, gross_profit, operating_income, goodwill, intangibles, assets_curr, liabilities_curr, inventory, cash, interest_expense, capex, rd_expense] if x is not None)} / 24 raw fields")

# ── Compute derived fields ───────────────────────────────────────────
print("\nComputing derived ratio matrices...")

# Shares outstanding (millions) from market_cap (millions) / close
if market_cap is not None and close is not None:
    sharesout = safe_divide(market_cap, close)
    # Clamp extreme values
    sharesout = sharesout.clip(lower=0.001)
    save("sharesout", sharesout)
else:
    sharesout = None
    print("  ⚠️  Cannot compute sharesout (need market_cap + close)")

# Book value per share
if equity is not None and sharesout is not None:
    bookvalue_ps = safe_divide(equity, sharesout)
    save("bookvalue_ps", bookvalue_ps)

# EPS diluted (use eps as proxy if no separate diluted)
if eps is not None:
    save("eps_diluted", eps.copy())

# PE ratio = close / eps
if close is not None and eps is not None:
    pe_ratio = safe_divide(close, eps)
    # Clamp PE to reasonable range
    pe_ratio = pe_ratio.clip(lower=-500, upper=500)
    save("pe_ratio", pe_ratio)

# PB ratio = close / bookvalue_ps = market_cap / equity
if market_cap is not None and equity is not None:
    pb_ratio = safe_divide(market_cap, equity)
    pb_ratio = pb_ratio.clip(lower=-100, upper=100)
    save("pb_ratio", pb_ratio)

# EV/EBITDA
if enterprise_value is not None and ebitda is not None:
    ev_to_ebitda = safe_divide(enterprise_value, ebitda)
    ev_to_ebitda = ev_to_ebitda.clip(lower=-200, upper=200)
    save("ev_to_ebitda", ev_to_ebitda)

# ROE = income / equity
if income is not None and equity is not None:
    roe = safe_divide(income, equity)
    roe = roe.clip(lower=-5, upper=5)
    save("roe", roe)
    save("return_equity", roe)  # alias

# ROA = income / assets
if income is not None and assets is not None:
    roa = safe_divide(income, assets)
    roa = roa.clip(lower=-2, upper=2)
    save("roa", roa)
    save("return_assets", roa)  # alias

# Debt to equity
if debt is not None and equity is not None:
    debt_to_equity = safe_divide(debt, equity)
    debt_to_equity = debt_to_equity.clip(lower=-20, upper=20)
    save("debt_to_equity", debt_to_equity)

# FCF per share
if free_cashflow is not None and sharesout is not None:
    fcf_per_share = safe_divide(free_cashflow, sharesout)
    save("fcf_per_share", fcf_per_share)

# Revenue per share / Sales per share
if revenue is not None and sharesout is not None:
    revenue_per_share = safe_divide(revenue, sharesout)
    save("revenue_per_share", revenue_per_share)
    save("sales_ps", revenue_per_share)  # alias

# Tangible book value per share = (equity - goodwill - intangibles) / sharesout
if equity is not None and sharesout is not None:
    tangible_equity = equity.copy()
    if goodwill is not None:
        tangible_equity = tangible_equity - goodwill.reindex_like(tangible_equity).fillna(0)
    if intangibles is not None:
        tangible_equity = tangible_equity - intangibles.reindex_like(tangible_equity).fillna(0)
    tangible_book_ps = safe_divide(tangible_equity, sharesout)
    save("tangible_book_per_share", tangible_book_ps)

# ── Additional profitability / efficiency ratios ─────────────────────
print("\nComputing additional ratios...")

# Gross margin = gross_profit / revenue
if gross_profit is not None and revenue is not None:
    gross_margin = safe_divide(gross_profit, revenue)
    gross_margin = gross_margin.clip(lower=-2, upper=2)
    save("gross_margin", gross_margin)

# Operating margin = operating_income / revenue
if operating_income is not None and revenue is not None:
    operating_margin = safe_divide(operating_income, revenue)
    operating_margin = operating_margin.clip(lower=-5, upper=5)
    save("operating_margin", operating_margin)

# Net margin = income / revenue
if income is not None and revenue is not None:
    net_margin = safe_divide(income, revenue)
    net_margin = net_margin.clip(lower=-5, upper=5)
    save("net_margin", net_margin)

# Debt-to-assets
if debt is not None and assets is not None:
    debt_to_assets = safe_divide(debt, assets)
    debt_to_assets = debt_to_assets.clip(lower=0, upper=5)
    save("debt_to_assets", debt_to_assets)

# Cash ratio = cash / current liabilities
if cash is not None and liabilities_curr is not None:
    cash_ratio = safe_divide(cash, liabilities_curr)
    cash_ratio = cash_ratio.clip(lower=0, upper=50)
    save("cash_ratio", cash_ratio)

# Quick ratio = (current_assets - inventory) / current_liabilities
if assets_curr is not None and liabilities_curr is not None:
    quick_assets = assets_curr.copy()
    if inventory is not None:
        quick_assets = quick_assets - inventory.reindex_like(quick_assets).fillna(0)
    quick_ratio = safe_divide(quick_assets, liabilities_curr)
    quick_ratio = quick_ratio.clip(lower=0, upper=50)
    save("quick_ratio", quick_ratio)

# Asset turnover = revenue / assets
if revenue is not None and assets is not None:
    asset_turnover = safe_divide(revenue, assets)
    asset_turnover = asset_turnover.clip(lower=0, upper=10)
    save("asset_turnover", asset_turnover)

# Capex-to-revenue
if capex is not None and revenue is not None:
    capex_to_revenue = safe_divide(capex.abs(), revenue)
    capex_to_revenue = capex_to_revenue.clip(lower=0, upper=2)
    save("capex_to_revenue", capex_to_revenue)

# R&D to revenue
if rd_expense is not None and revenue is not None:
    rd_to_revenue = safe_divide(rd_expense.abs(), revenue)
    rd_to_revenue = rd_to_revenue.clip(lower=0, upper=2)
    save("rd_to_revenue", rd_to_revenue)

# Interest coverage = EBIT / interest_expense
if ebit is not None and interest_expense is not None:
    interest_coverage = safe_divide(ebit, interest_expense)
    interest_coverage = interest_coverage.clip(lower=-100, upper=100)
    save("interest_coverage", interest_coverage)

# Free cashflow yield = FCF / market_cap
if free_cashflow is not None and market_cap is not None:
    fcf_yield = safe_divide(free_cashflow, market_cap)
    fcf_yield = fcf_yield.clip(lower=-1, upper=1)
    save("free_cashflow_yield", fcf_yield)

# Earnings yield = eps / close (inverse PE)
if eps is not None and close is not None:
    earnings_yield = safe_divide(eps, close)
    earnings_yield = earnings_yield.clip(lower=-1, upper=1)
    save("earnings_yield", earnings_yield)

# EBITDA margin = EBITDA / revenue
if ebitda is not None and revenue is not None:
    ebitda_margin = safe_divide(ebitda, revenue)
    ebitda_margin = ebitda_margin.clip(lower=-5, upper=5)
    save("ebitda_margin", ebitda_margin)

# Payout ratio (approximate from cashflow)
if cashflow_op is not None and income is not None:
    # Cash conversion ratio
    cash_conversion = safe_divide(cashflow_op, income)
    cash_conversion = cash_conversion.clip(lower=-10, upper=10)
    save("cash_conversion_ratio", cash_conversion)

# SGA-to-revenue
if sga is not None and revenue is not None:
    sga_to_revenue = safe_divide(sga, revenue)
    sga_to_revenue = sga_to_revenue.clip(lower=0, upper=2)
    save("sga_to_revenue", sga_to_revenue)

# Book-to-market = equity / market_cap (value factor)
if equity is not None and market_cap is not None:
    book_to_market = safe_divide(equity, market_cap)
    book_to_market = book_to_market.clip(lower=-5, upper=5)
    save("book_to_market", book_to_market)

# Enterprise value to revenue
if enterprise_value is not None and revenue is not None:
    ev_to_revenue = safe_divide(enterprise_value, revenue)
    ev_to_revenue = ev_to_revenue.clip(lower=-100, upper=100)
    save("ev_to_revenue", ev_to_revenue)

# Enterprise value to FCF
if enterprise_value is not None and free_cashflow is not None:
    ev_to_fcf = safe_divide(enterprise_value, free_cashflow)
    ev_to_fcf = ev_to_fcf.clip(lower=-200, upper=200)
    save("ev_to_fcf", ev_to_fcf)

# ── Summary ──────────────────────────────────────────────────────────
print("\n── Final field count ──")
all_fields = sorted([f.replace('.parquet', '') for f in os.listdir(MATRICES_DIR) 
                     if f.endswith('.parquet') and not f.startswith('_')])
print(f"Total matrices: {len(all_fields)}")

# Update metadata
import json
meta_path = "data/fmp_cache/metadata.json"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_fields"] = len(all_fields)
    meta["fields"] = all_fields
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated metadata.json: {len(all_fields)} fields")
