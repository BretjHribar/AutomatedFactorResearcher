"""
Extract ALL available FMP fields into Parquet matrices.

The raw per-ticker data is already cached in data/fmp_cache/{income,balance,cashflow,metrics}/.
This script reads those cached files and builds aligned (dates × tickers) matrices for every
field that FMP provides — not just the subset we currently extract.

No API calls needed — everything is already on disk.
"""
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("data/fmp_cache")
MATRICES_DIR = CACHE_DIR / "matrices"

# ── Complete field maps: our_name → FMP column name ──────────────────────────

INCOME_FIELDS = {
    # Already extracted
    "revenue": "revenue",
    "sales": "revenue",
    "cost_of_revenue": "costOfRevenue",
    "cogs": "costOfRevenue",
    "gross_profit": "grossProfit",
    "operating_income": "operatingIncome",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "net_income": "netIncome",
    "eps": "eps",
    "eps_diluted": "epsDiluted",
    "rd_expense": "researchAndDevelopmentExpenses",
    "sga_expense": "sellingGeneralAndAdministrativeExpenses",
    "interest_expense": "interestExpense",
    "income_tax": "incomeTaxExpense",
    # NEW fields
    "income_before_tax": "incomeBeforeTax",
    "interest_income": "interestIncome",
    "net_interest_income": "netInterestIncome",
    "depre_amort_income": "depreciationAndAmortization",
    "ga_expense": "generalAndAdministrativeExpenses",
    "selling_marketing_expense": "sellingAndMarketingExpenses",
    "operating_expenses_total": "operatingExpenses",
    "cost_and_expenses": "costAndExpenses",
    "net_income_continuing": "netIncomeFromContinuingOperations",
    "net_income_discontinued": "netIncomeFromDiscontinuedOperations",
    "other_income_expense_net": "totalOtherIncomeExpensesNet",
    "non_operating_income": "nonOperatingIncomeExcludingInterest",
    "other_expenses": "otherExpenses",
    "weighted_avg_shares": "weightedAverageShsOut",
    "weighted_avg_shares_diluted": "weightedAverageShsOutDil",
    "bottom_line_net_income": "bottomLineNetIncome",
    "net_income_deductions": "netIncomeDeductions",
    "other_adjustments_ni": "otherAdjustmentsToNetIncome",
}

BALANCE_FIELDS = {
    # Already extracted
    "total_assets": "totalAssets",
    "assets_curr": "totalCurrentAssets",
    "cash": "cashAndCashEquivalents",
    "receivables": "netReceivables",
    "inventory": "inventory",
    "goodwill": "goodwill",
    "intangibles": "intangibleAssets",
    "ppe_net": "propertyPlantEquipmentNet",
    "total_liabilities": "totalLiabilities",
    "liabilities_curr": "totalCurrentLiabilities",
    "payables": "accountPayables",
    "total_debt": "totalDebt",
    "total_equity": "totalStockholdersEquity",
    "retained_earnings": "retainedEarnings",
    "net_debt": "netDebt",
    # NEW fields
    "short_term_investments": "shortTermInvestments",
    "cash_and_st_investments": "cashAndShortTermInvestments",
    "accounts_receivables": "accountsReceivables",
    "other_receivables": "otherReceivables",
    "prepaids": "prepaids",
    "other_current_assets": "otherCurrentAssets",
    "long_term_investments": "longTermInvestments",
    "tax_assets": "taxAssets",
    "other_noncurrent_assets": "otherNonCurrentAssets",
    "total_noncurrent_assets": "totalNonCurrentAssets",
    "goodwill_and_intangibles": "goodwillAndIntangibleAssets",
    "other_assets": "otherAssets",
    "total_payables": "totalPayables",
    "other_payables": "otherPayables",
    "accrued_expenses": "accruedExpenses",
    "short_term_debt": "shortTermDebt",
    "long_term_debt": "longTermDebt",
    "capital_lease_obligations": "capitalLeaseObligations",
    "capital_lease_current": "capitalLeaseObligationsCurrent",
    "capital_lease_noncurrent": "capitalLeaseObligationsNonCurrent",
    "tax_payables": "taxPayables",
    "deferred_revenue": "deferredRevenue",
    "deferred_revenue_noncurrent": "deferredRevenueNonCurrent",
    "deferred_tax_liabilities": "deferredTaxLiabilitiesNonCurrent",
    "other_current_liabilities": "otherCurrentLiabilities",
    "other_noncurrent_liabilities": "otherNonCurrentLiabilities",
    "total_noncurrent_liabilities": "totalNonCurrentLiabilities",
    "other_liabilities": "otherLiabilities",
    "common_stock": "commonStock",
    "additional_paid_in_capital": "additionalPaidInCapital",
    "treasury_stock": "treasuryStock",
    "accumulated_oci": "accumulatedOtherComprehensiveIncomeLoss",
    "minority_interest": "minorityInterest",
    "preferred_stock": "preferredStock",
    "other_stockholders_equity": "otherTotalStockholdersEquity",
    "total_equity_incl_minority": "totalEquity",
    "total_liab_and_equity": "totalLiabilitiesAndTotalEquity",
    "total_investments": "totalInvestments",
}

CASHFLOW_FIELDS = {
    # Already extracted
    "cashflow_op": "operatingCashFlow",
    "capex": "capitalExpenditure",
    "free_cashflow": "freeCashFlow",
    "depreciation": "depreciationAndAmortization",
    "stock_repurchase": "commonStockRepurchased",
    "debt_repayment": "debtRepayment",
    "dividends_paid": "dividendsPaid",
    # NEW fields
    "stock_based_compensation": "stockBasedCompensation",
    "change_in_working_capital": "changeInWorkingCapital",
    "deferred_income_tax": "deferredIncomeTax",
    "cf_accounts_receivables": "accountsReceivables",
    "cf_inventory": "inventory",
    "cf_accounts_payables": "accountsPayables",
    "other_working_capital": "otherWorkingCapital",
    "other_non_cash_items": "otherNonCashItems",
    "acquisitions_net": "acquisitionsNet",
    "purchases_of_investments": "purchasesOfInvestments",
    "sales_of_investments": "salesMaturitiesOfInvestments",
    "other_investing": "otherInvestingActivities",
    "net_investing_cf": "netCashProvidedByInvestingActivities",
    "net_debt_issuance": "netDebtIssuance",
    "lt_net_debt_issuance": "longTermNetDebtIssuance",
    "st_net_debt_issuance": "shortTermNetDebtIssuance",
    "net_stock_issuance": "netStockIssuance",
    "net_common_stock_issuance": "netCommonStockIssuance",
    "common_stock_issuance": "commonStockIssuance",
    "net_preferred_stock_issuance": "netPreferredStockIssuance",
    "common_dividends_paid": "commonDividendsPaid",
    "preferred_dividends_paid": "preferredDividendsPaid",
    "net_dividends_paid": "netDividendsPaid",
    "other_financing": "otherFinancingActivities",
    "net_financing_cf": "netCashProvidedByFinancingActivities",
    "forex_effect_on_cash": "effectOfForexChangesOnCash",
    "net_change_in_cash": "netChangeInCash",
    "cash_at_end": "cashAtEndOfPeriod",
    "cash_at_beginning": "cashAtBeginningOfPeriod",
    "interest_paid": "interestPaid",
    "income_taxes_paid": "incomeTaxesPaid",
    "net_income_cf": "netIncome",
}

METRICS_FIELDS = {
    # Already extracted
    "market_cap_metric": "marketCap",
    "enterprise_value": "enterpriseValue",
    "current_ratio": "currentRatio",
    "invested_capital": "investedCapital",
    "working_capital": "workingCapital",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "earnings_yield": "earningsYield",
    "free_cashflow_yield": "freeCashFlowYield",
    "ev_to_ebitda": "evToEBITDA",
    "ev_to_revenue": "evToSales",
    "ev_to_fcf": "evToFreeCashFlow",
    # NEW fields
    "roic": "returnOnInvestedCapital",
    "roce": "returnOnCapitalEmployed",
    "rota": "returnOnTangibleAssets",
    "operating_roa": "operatingReturnOnAssets",
    "income_quality": "incomeQuality",
    "graham_number": "grahamNumber",
    "graham_net_net": "grahamNetNet",
    "net_debt_to_ebitda": "netDebtToEBITDA",
    "tax_burden": "taxBurden",
    "interest_burden": "interestBurden",
    "capex_to_ocf": "capexToOperatingCashFlow",
    "capex_to_depreciation": "capexToDepreciation",
    "sbc_to_revenue": "stockBasedCompensationToRevenue",
    "intangibles_to_assets": "intangiblesToTotalAssets",
    "days_sales_outstanding": "daysOfSalesOutstanding",
    "days_payables_outstanding": "daysOfPayablesOutstanding",
    "days_inventory_outstanding": "daysOfInventoryOutstanding",
    "operating_cycle": "operatingCycle",
    "cash_conversion_cycle": "cashConversionCycle",
    "fcf_to_equity": "freeCashFlowToEquity",
    "fcf_to_firm": "freeCashFlowToFirm",
    "tangible_asset_value": "tangibleAssetValue",
    "net_current_asset_value": "netCurrentAssetValue",
    "ev_to_ocf": "evToOperatingCashFlow",
    "avg_receivables": "averageReceivables",
    "avg_payables": "averagePayables",
    "avg_inventory": "averageInventory",
}


def extract_all_fields():
    """Extract all FMP fields into aligned matrices."""
    t0 = time.time()
    
    # Get existing date index and ticker list from close matrix
    close = pd.read_parquet(MATRICES_DIR / "close.parquet")
    all_dates = close.index.tolist()
    # Use fundamental tickers (2512) — the ones that have cached data
    
    fund_configs = [
        ("income", INCOME_FIELDS),
        ("balance", BALANCE_FIELDS),
        ("cashflow", CASHFLOW_FIELDS),
        ("metrics", METRICS_FIELDS),
    ]
    
    total_new = 0
    total_existing = 0
    
    for ftype, field_map in fund_configs:
        fund_dir = CACHE_DIR / ftype
        if not fund_dir.exists():
            print(f"  {ftype}: directory not found, skipping")
            continue
        
        # Get list of tickers with cached data
        cached_files = sorted(fund_dir.glob("*.parquet"))
        tickers = [f.stem for f in cached_files]
        print(f"\n{'='*70}")
        print(f"  {ftype.upper()}: {len(tickers)} tickers, {len(field_map)} fields to extract")
        print(f"{'='*70}")
        
        # Identify which fields are new vs already exist
        new_fields = {}
        existing_fields = {}
        for our_name, fmp_name in field_map.items():
            mat_path = MATRICES_DIR / f"{our_name}.parquet"
            if mat_path.exists():
                existing_fields[our_name] = fmp_name
            else:
                new_fields[our_name] = fmp_name
        
        print(f"  Already exist: {len(existing_fields)}")
        print(f"  New to build:  {len(new_fields)}")
        
        if not new_fields:
            print(f"  Nothing new to extract.")
            continue
        
        # Initialize new matrices
        matrices = {}
        for our_name in new_fields:
            matrices[our_name] = pd.DataFrame(
                index=all_dates, columns=tickers, dtype=float
            )
        
        # Read each ticker's cached data
        done = 0
        for sym in tickers:
            fpath = fund_dir / f"{sym}.parquet"
            try:
                fdf = pd.read_parquet(fpath)
                if fdf.index.duplicated().any():
                    fdf = fdf[~fdf.index.duplicated(keep='last')]
            except Exception:
                continue
            
            # Point-in-Time (PIT) alignment using acceptedDate
            pit_index = fdf.index
            if "acceptedDate" in fdf.columns:
                try:
                    accepted = pd.to_datetime(fdf["acceptedDate"])
                    pit_index = accepted.dt.normalize()
                except Exception:
                    pass
            elif "filingDate" in fdf.columns:
                try:
                    filed = pd.to_datetime(fdf["filingDate"])
                    pit_index = filed + pd.Timedelta(days=1)
                except Exception:
                    pass
            else:
                pit_index = fdf.index + pd.Timedelta(days=90)
            
            fdf_pit = fdf.copy()
            fdf_pit.index = pit_index
            fdf_pit = fdf_pit.sort_index()
            if fdf_pit.index.duplicated().any():
                fdf_pit = fdf_pit[~fdf_pit.index.duplicated(keep='last')]
            
            # Extract each new field
            for our_name, fmp_name in new_fields.items():
                if fmp_name in fdf_pit.columns:
                    series = fdf_pit[fmp_name]
                    # Convert to numeric, coerce errors
                    series = pd.to_numeric(series, errors='coerce')
                    daily = series.reindex(all_dates, method="ffill")
                    matrices[our_name][sym] = daily
            
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(tickers)} tickers processed...")
        
        # Save new matrices
        saved = 0
        for our_name, mat in matrices.items():
            # Check if matrix has any data
            non_null = mat.notna().sum().sum()
            total_cells = mat.shape[0] * mat.shape[1]
            coverage = non_null / total_cells * 100 if total_cells > 0 else 0
            
            if non_null > 0:
                mat_path = MATRICES_DIR / f"{our_name}.parquet"
                mat.to_parquet(mat_path)
                saved += 1
                print(f"    ✅ {our_name:40s} {mat.shape[1]:>5} tickers  cov={coverage:.1f}%")
            else:
                print(f"    ⚠️  {our_name:40s} ALL NULL - skipped")
        
        total_new += saved
        print(f"  Saved {saved} new matrices for {ftype}")
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE: {total_new} new matrices created in {elapsed:.1f}s")
    print(f"{'='*70}")
    
    # Final count
    all_parquets = list(MATRICES_DIR.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in all_parquets) / 1e6
    print(f"  Total matrices on disk: {len(all_parquets)}")
    print(f"  Total disk usage: {total_mb:.0f} MB")


if __name__ == "__main__":
    extract_all_fields()
