"""
WorldQuant BRAIN Data Field Catalog.

Comprehensive registry of all data fields available in BRAIN / WebSim.
Used by:
  - The LLM agent to know what fields exist and their descriptions
  - The FastExpressionEngine to validate field references
  - The data loader to map field names → source tables

Each field has:
  - name: the identifier used in fastexpression
  - group: which data category it belongs to
  - description: human-readable description
  - dtype: 'matrix' (date × ticker) or 'vector' (ticker only)
  - coverage: approximate coverage percentage (for analyst estimates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class FieldDefinition:
    name: str
    group: str
    description: str
    dtype: str = "matrix"       # "matrix" or "vector"
    coverage: float = 100.0     # percent of universe with data
    is_grouping: bool = False   # True for sector, industry, etc.


# ---------------------------------------------------------------------------
# Field Groups
# ---------------------------------------------------------------------------

MARKET_DATA_FIELDS = [
    FieldDefinition("close", "MARKET DATA", "Daily close price"),
    FieldDefinition("open", "MARKET DATA", "Daily open price"),
    FieldDefinition("high", "MARKET DATA", "Daily high price"),
    FieldDefinition("low", "MARKET DATA", "Daily low price"),
    FieldDefinition("volume", "MARKET DATA", "Daily trading volume"),
    FieldDefinition("vwap", "MARKET DATA", "Volume-weighted average price"),
    FieldDefinition("adv20", "MARKET DATA", "20-day average daily volume"),
    FieldDefinition("returns", "MARKET DATA", "Daily returns"),
    FieldDefinition("cap", "MARKET DATA", "Market capitalization (millions)"),
    FieldDefinition("sharesout", "MARKET DATA", "Outstanding shares (millions)"),
    FieldDefinition("dividend", "MARKET DATA", "Dividend"),
    FieldDefinition("split", "MARKET DATA", "Stock split ratio"),
    FieldDefinition("currency", "MARKET DATA", "Currency", dtype="vector"),
]

GROUPING_FIELDS = [
    FieldDefinition("industry", "GROUPING FIELDS", "GICS industry", dtype="vector", is_grouping=True),
    FieldDefinition("sector", "GROUPING FIELDS", "GICS sector", dtype="vector", is_grouping=True),
    FieldDefinition("subindustry", "GROUPING FIELDS", "GICS sub-industry", dtype="vector", is_grouping=True),
    FieldDefinition("country", "GROUPING FIELDS", "Country", dtype="vector", is_grouping=True),
    FieldDefinition("market", "GROUPING FIELDS", "Market", dtype="vector", is_grouping=True),
    FieldDefinition("exchange", "GROUPING FIELDS", "Exchange", dtype="vector", is_grouping=True),
]

SYMBOL_FIELDS = [
    FieldDefinition("ticker", "SYMBOL FIELDS", "Ticker symbol", dtype="vector"),
    FieldDefinition("sedol", "SYMBOL FIELDS", "SEDOL identifier", dtype="vector"),
]

FSCORE_FIELDS = [
    FieldDefinition("fscore_growth", "FSCORE METRICS", "Expected MT growth potential"),
    FieldDefinition("fscore_momentum", "FSCORE METRICS", "Analyst revision momentum"),
    FieldDefinition("fscore_profitability", "FSCORE METRICS", "Cash flow generation ability"),
    FieldDefinition("fscore_quality", "FSCORE METRICS", "Earnings sustainability and certainty"),
    FieldDefinition("fscore_surface", "FSCORE METRICS", "Pentagon surface score (0-100)"),
    FieldDefinition("fscore_surface_accel", "FSCORE METRICS", "Pentagon acceleration score"),
    FieldDefinition("fscore_total", "FSCORE METRICS", "Weighted average M-Score"),
    FieldDefinition("fscore_value", "FSCORE METRICS", "Valuation assessment"),
]

FINANCIAL_STATEMENT_FIELDS = [
    FieldDefinition("assets", "FINANCIAL STATEMENT DATA", "Total assets"),
    FieldDefinition("liabilities", "FINANCIAL STATEMENT DATA", "Total liabilities"),
    FieldDefinition("operating_income", "FINANCIAL STATEMENT DATA", "Operating income after depreciation (quarterly)"),
    FieldDefinition("sales", "FINANCIAL STATEMENT DATA", "Sales/turnover (net)"),
    FieldDefinition("enterprise_value", "FINANCIAL STATEMENT DATA", "Enterprise value"),
    FieldDefinition("capex", "FINANCIAL STATEMENT DATA", "Capital expenditures"),
    FieldDefinition("debt", "FINANCIAL STATEMENT DATA", "Total debt"),
    FieldDefinition("equity", "FINANCIAL STATEMENT DATA", "Common equity"),
    FieldDefinition("ebit", "FINANCIAL STATEMENT DATA", "EBIT"),
    FieldDefinition("ebitda", "FINANCIAL STATEMENT DATA", "EBITDA"),
    FieldDefinition("eps", "FINANCIAL STATEMENT DATA", "EPS (basic)"),
    FieldDefinition("debt_lt", "FINANCIAL STATEMENT DATA", "Long-term debt"),
    FieldDefinition("assets_curr", "FINANCIAL STATEMENT DATA", "Current assets"),
    FieldDefinition("goodwill", "FINANCIAL STATEMENT DATA", "Goodwill (net)"),
    FieldDefinition("income", "FINANCIAL STATEMENT DATA", "Net income"),
    FieldDefinition("cash", "FINANCIAL STATEMENT DATA", "Cash"),
    FieldDefinition("revenue", "FINANCIAL STATEMENT DATA", "Revenue"),
    FieldDefinition("cashflow_op", "FINANCIAL STATEMENT DATA", "Operating cash flow"),
    FieldDefinition("cogs", "FINANCIAL STATEMENT DATA", "Cost of goods sold"),
    FieldDefinition("bookvalue_ps", "FINANCIAL STATEMENT DATA", "Book value per share"),
    FieldDefinition("ppent", "FINANCIAL STATEMENT DATA", "PP&E (net)"),
    FieldDefinition("operating_expense", "FINANCIAL STATEMENT DATA", "Total operating expense"),
    FieldDefinition("debt_st", "FINANCIAL STATEMENT DATA", "Short-term debt"),
    FieldDefinition("cashflow", "FINANCIAL STATEMENT DATA", "Cash flow (annual)"),
    FieldDefinition("inventory", "FINANCIAL STATEMENT DATA", "Total inventories"),
    FieldDefinition("liabilities_curr", "FINANCIAL STATEMENT DATA", "Current liabilities"),
    FieldDefinition("cash_st", "FINANCIAL STATEMENT DATA", "Cash and short-term investments"),
    FieldDefinition("receivable", "FINANCIAL STATEMENT DATA", "Total receivables"),
    FieldDefinition("sga_expense", "FINANCIAL STATEMENT DATA", "SG&A expense"),
    FieldDefinition("return_equity", "FINANCIAL STATEMENT DATA", "Return on equity"),
    FieldDefinition("retained_earnings", "FINANCIAL STATEMENT DATA", "Retained earnings"),
    FieldDefinition("income_tax", "FINANCIAL STATEMENT DATA", "Total income taxes"),
    FieldDefinition("pretax_income", "FINANCIAL STATEMENT DATA", "Pretax income"),
    FieldDefinition("cashflow_fin", "FINANCIAL STATEMENT DATA", "Financing cash flow"),
    FieldDefinition("income_beforeextra", "FINANCIAL STATEMENT DATA", "Income before extraordinary items"),
    FieldDefinition("sales_growth", "FINANCIAL STATEMENT DATA", "Quarterly sales growth"),
    FieldDefinition("current_ratio", "FINANCIAL STATEMENT DATA", "Current ratio"),
    FieldDefinition("return_assets", "FINANCIAL STATEMENT DATA", "Return on assets"),
    FieldDefinition("inventory_turnover", "FINANCIAL STATEMENT DATA", "Inventory turnover"),
    FieldDefinition("sales_ps", "FINANCIAL STATEMENT DATA", "Sales per share (quarterly)"),
    FieldDefinition("invested_capital", "FINANCIAL STATEMENT DATA", "Invested capital (quarterly)"),
    FieldDefinition("cashflow_dividends", "FINANCIAL STATEMENT DATA", "Cash dividends"),
    FieldDefinition("working_capital", "FINANCIAL STATEMENT DATA", "Working capital"),
    FieldDefinition("employee", "FINANCIAL STATEMENT DATA", "Number of employees"),
    FieldDefinition("cashflow_invst", "FINANCIAL STATEMENT DATA", "Investing cash flow"),
    FieldDefinition("depre_amort", "FINANCIAL STATEMENT DATA", "D&A total"),
    FieldDefinition("interest_expense", "FINANCIAL STATEMENT DATA", "Interest expense"),
    FieldDefinition("rd_expense", "FINANCIAL STATEMENT DATA", "R&D expense (quarterly)"),
]

# fnd6_* fields — extended Compustat via BRAIN (partial list of most useful)
ADDITIONAL_FINANCIAL_FIELDS = [
    FieldDefinition("fnd6_newa1v1300_gp", "ADDITIONAL FINANCIAL DATA", "Gross profit (loss)"),
    FieldDefinition("fnd6_newa1v1300_ceq", "ADDITIONAL FINANCIAL DATA", "Common equity"),
    FieldDefinition("fnd6_cptnewqv1300_oibdpq", "ADDITIONAL FINANCIAL DATA", "Operating income before depreciation (quarterly)"),
    FieldDefinition("fnd6_newa2v1300_rdipa", "ADDITIONAL FINANCIAL DATA", "In-process R&D expense after-tax"),
    FieldDefinition("fnd6_cptmfmq_atq", "ADDITIONAL FINANCIAL DATA", "Assets total (quarterly)"),
    FieldDefinition("fnd6_newa2v1300_oiadp", "ADDITIONAL FINANCIAL DATA", "Operating income after depreciation"),
    FieldDefinition("fnd6_cptnewqv1300_atq", "ADDITIONAL FINANCIAL DATA", "Assets total (quarterly)"),
    FieldDefinition("fnd6_cptmfmq_saleq", "ADDITIONAL FINANCIAL DATA", "Sales (quarterly)"),
    FieldDefinition("fnd6_cptmfmq_ceqq", "ADDITIONAL FINANCIAL DATA", "Common equity (quarterly)"),
    FieldDefinition("fnd6_beta", "ADDITIONAL FINANCIAL DATA", "Stock beta"),
    FieldDefinition("fnd6_mfma2_revt", "ADDITIONAL FINANCIAL DATA", "Revenue total"),
    FieldDefinition("fnd6_newa2v1300_seq", "ADDITIONAL FINANCIAL DATA", "Stockholders equity - parent"),
    FieldDefinition("fnd6_cptnewqv1300_dlttq", "ADDITIONAL FINANCIAL DATA", "Long-term debt (quarterly)"),
    FieldDefinition("fnd6_newa2v1300_oibdp", "ADDITIONAL FINANCIAL DATA", "Operating income before depreciation"),
    FieldDefinition("fnd6_teq", "ADDITIONAL FINANCIAL DATA", "Stockholders equity total"),
    FieldDefinition("fnd6_intan", "ADDITIONAL FINANCIAL DATA", "Intangible assets total"),
]

ANALYST_ESTIMATES_FIELDS = [
    FieldDefinition("est_eps", "ANALYST ESTIMATES DATA", "EPS mean estimate", coverage=78),
    FieldDefinition("est_sales", "ANALYST ESTIMATES DATA", "Sales mean estimate", coverage=75),
    FieldDefinition("est_ebit", "ANALYST ESTIMATES DATA", "EBIT mean estimate", coverage=73),
    FieldDefinition("est_ebitda", "ANALYST ESTIMATES DATA", "EBITDA mean estimate", coverage=60),
    FieldDefinition("est_netprofit", "ANALYST ESTIMATES DATA", "Net profit mean estimate", coverage=79),
    FieldDefinition("est_capex", "ANALYST ESTIMATES DATA", "Capex mean estimate", coverage=64),
    FieldDefinition("est_cashflow_op", "ANALYST ESTIMATES DATA", "Operating cash flow mean estimate", coverage=65),
    FieldDefinition("est_epsr", "ANALYST ESTIMATES DATA", "GAAP EPS mean estimate", coverage=74),
    FieldDefinition("est_rd_expense", "ANALYST ESTIMATES DATA", "R&D expense mean estimate", coverage=31),
    FieldDefinition("anl4_ebit_value", "ANALYST ESTIMATES DATA", "EBIT announced value", coverage=49),
    FieldDefinition("anl4_ebitda_value", "ANALYST ESTIMATES DATA", "EBITDA announced value", coverage=45),
    FieldDefinition("anl4_total_rec", "ANALYST ESTIMATES DATA", "Total recommendations", dtype="vector", coverage=62),
    FieldDefinition("anl4_eaz2lrec_ratingvalue", "ANALYST ESTIMATES DATA", "Instrument rating score", dtype="vector", coverage=53),
]

SENTIMENT_FIELDS = [
    FieldDefinition("scl12_buzz", "SENTIMENT DATA", "Relative sentiment volume", coverage=95),
    FieldDefinition("scl12_sentiment", "SENTIMENT DATA", "Sentiment score", coverage=94),
    FieldDefinition("snt_buzz", "SENTIMENT DATA", "Negative relative sentiment volume (NaN→0)", coverage=95),
    FieldDefinition("snt_buzz_bfl", "SENTIMENT DATA", "Negative relative sentiment volume (NaN→1)", coverage=100),
    FieldDefinition("snt_buzz_ret", "SENTIMENT DATA", "Negative return of relative sentiment volume", coverage=95),
    FieldDefinition("snt_value", "SENTIMENT DATA", "Negative sentiment (NaN→0)", coverage=94),
    FieldDefinition("snt_social_value", "SENTIMENT DATA", "Z-score of sentiment", coverage=86),
    FieldDefinition("snt_social_volume", "SENTIMENT DATA", "Normalized tweet volume", coverage=86),
]

OPTIONS_VOLATILITY_FIELDS: List[FieldDefinition] = []
for _tenor in [10, 20, 30, 60, 90, 120, 150, 180]:
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"historical_volatility_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"Historical volatility ({_tenor}d)")
    )
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"parkinson_volatility_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"Parkinson volatility ({_tenor}d)")
    )
for _tenor in [10, 20, 30, 60, 90, 120, 150, 180, 270, 360, 720, 1080]:
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"implied_volatility_call_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"ATM implied vol call ({_tenor}d)")
    )
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"implied_volatility_put_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"ATM implied vol put ({_tenor}d)")
    )
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"implied_volatility_mean_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"ATM implied vol mean ({_tenor}d)")
    )
    OPTIONS_VOLATILITY_FIELDS.append(
        FieldDefinition(f"implied_volatility_mean_skew_{_tenor}", "OPTIONS & VOLATILITY DATA",
                        f"ATM implied vol mean skew ({_tenor}d)")
    )


# ---------------------------------------------------------------------------
# Master catalog
# ---------------------------------------------------------------------------

ALL_FIELDS: List[FieldDefinition] = (
    MARKET_DATA_FIELDS
    + GROUPING_FIELDS
    + SYMBOL_FIELDS
    + FSCORE_FIELDS
    + FINANCIAL_STATEMENT_FIELDS
    + ADDITIONAL_FINANCIAL_FIELDS
    + ANALYST_ESTIMATES_FIELDS
    + SENTIMENT_FIELDS
    + OPTIONS_VOLATILITY_FIELDS
)

FIELD_BY_NAME: Dict[str, FieldDefinition] = {f.name: f for f in ALL_FIELDS}

ALL_GROUPS: Set[str] = {f.group for f in ALL_FIELDS}

GROUPING_FIELD_NAMES: Set[str] = {f.name for f in ALL_FIELDS if f.is_grouping}


def get_fields_by_group(group: str) -> List[FieldDefinition]:
    """Get all fields in a given group."""
    return [f for f in ALL_FIELDS if f.group.upper() == group.upper()]


def get_field_names_for_groups(groups: List[str] | None = None) -> List[str]:
    """Get field names for the given groups (or all if None)."""
    if groups is None:
        return [f.name for f in ALL_FIELDS if not f.is_grouping]
    allowed = {g.upper() for g in groups}
    return [f.name for f in ALL_FIELDS if f.group.upper() in allowed and not f.is_grouping]


def format_fields_for_prompt(groups: List[str] | None = None,
                              max_per_group: int | None = None) -> str:
    """Format field catalog as text suitable for LLM prompts.

    Args:
        groups: Optional list of group names to include. None = all.
        max_per_group: Optional limit per group for prompt length control.
    """
    if groups is not None:
        allowed = {g.upper() for g in groups}
    else:
        allowed = None

    lines: List[str] = []
    current_group = ""
    count = 0
    for f in ALL_FIELDS:
        if allowed is not None and f.group.upper() not in allowed:
            continue
        if f.group != current_group:
            current_group = f.group
            count = 0
            lines.append(f"\n{current_group}:")
        if max_per_group is not None and count >= max_per_group:
            continue
        cov = f" (coverage {f.coverage:.0f}%)" if f.coverage < 100 else ""
        lines.append(f"  {f.name} — {f.description}{cov}")
        count += 1
    return "\n".join(lines)
