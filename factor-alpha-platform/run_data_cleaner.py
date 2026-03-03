"""
Data Cleaning Pipeline
======================
Scans all matrix parquet files, detects quality issues, 
cleans data, and produces an integrity report.

Cleaning Rules:
  1. Prices: Replace values > $1M/share or < $0.001 with NaN
  2. Returns: Clip to ±100% (no single stock moves >100% in a day realistically)  
  3. Volumes: Cap at reasonable maximums, replace negatives with NaN
  4. Ratios: Winsorize at 1st/99th percentile
  5. Per-share: Winsorize at 1st/99th percentile  
  6. General: Replace Inf with NaN
  7. Forward-fill NaN up to 5 days, then leave as NaN

Produces:
  - Cleaned parquet files in data/fmp_cache/matrices_clean/
  - Integrity report in data/data_integrity_report.json
  - Human-readable report printed to stdout
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

# ═════════════════════════════════════════════════════
# FIELD CATEGORIES
# ═════════════════════════════════════════════════════
PRICE_FIELDS = {"close", "open", "high", "low", "vwap"}
RETURN_FIELDS = {"returns", "log_returns"}
VOLUME_FIELDS = {"volume", "dollars_traded", "adv20", "adv60"}
MARKET_CAP_FIELDS = {"market_cap", "market_cap_metric", "enterprise_value"}
PER_SHARE_FIELDS = {
    "bookvalue_ps", "tangible_book_per_share", "revenue_per_share",
    "sales_ps", "eps", "eps_diluted", "fcf_per_share", "graham_number",
}
RATIO_FIELDS = {
    "pe_ratio", "pb_ratio", "earnings_yield", "ev_to_ebitda", "ev_to_fcf",
    "ev_to_ocf", "ev_to_revenue", "free_cashflow_yield", "book_to_market",
    "current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity",
    "debt_to_assets", "interest_coverage", "asset_turnover", 
    "inventory_turnover", "operating_margin", "gross_margin", "net_margin",
    "ebitda_margin", "return_equity", "return_assets", "roa", "roe", "roic",
    "roce", "rota", "operating_roa", "income_quality", "interest_burden",
    "tax_burden", "capex_to_revenue", "capex_to_depreciation", "capex_to_ocf",
    "cash_conversion_ratio", "sga_to_revenue", "rd_to_revenue",
    "sbc_to_revenue", "intangibles_to_assets", "days_inventory_outstanding",
    "days_payables_outstanding", "days_sales_outstanding", "cash_conversion_cycle",
    "operating_cycle", "net_debt_to_ebitda",
}
VOLATILITY_FIELDS = {
    f"{prefix}_{w}" 
    for prefix in ["historical_volatility", "parkinson_volatility"]
    for w in [10, 20, 30, 60, 90, 120, 150, 180]
}
CLASSIFICATION_FIELDS = {"sector", "industry", "subindustry"}


def classify_field(name: str) -> str:
    """Classify a field into its cleaning category."""
    if name in PRICE_FIELDS:
        return "price"
    if name in RETURN_FIELDS:
        return "return"
    if name in VOLUME_FIELDS:
        return "volume"
    if name in MARKET_CAP_FIELDS:
        return "market_cap"
    if name in PER_SHARE_FIELDS:
        return "per_share"
    if name in RATIO_FIELDS:
        return "ratio"
    if name in VOLATILITY_FIELDS:
        return "volatility"
    if name in CLASSIFICATION_FIELDS:
        return "classification"
    return "general"


# ═════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═════════════════════════════════════════════════════

def clean_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean price data: remove values > $1M/share or <= 0."""
    stats = {"replaced_high": 0, "replaced_low": 0, "replaced_neg": 0}
    
    mask_high = df > 1_000_000
    mask_low = (df <= 0) & df.notna()
    
    stats["replaced_high"] = int(mask_high.sum().sum())
    stats["replaced_low"] = int(mask_low.sum().sum())
    
    df = df.where(~mask_high)
    df = df.where(~mask_low)
    
    return df, stats


def clean_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean returns: clip to ±100%, replace Inf."""
    stats = {"clipped": 0, "inf_replaced": 0}
    
    # Replace Inf
    inf_mask = np.isinf(df.values.astype(float))
    stats["inf_replaced"] = int(inf_mask.sum())
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Clip to ±100%
    clip_mask = (df.abs() > 1.0) & df.notna()
    stats["clipped"] = int(clip_mask.sum().sum())
    df = df.clip(-1.0, 1.0)
    
    return df, stats


def clean_volumes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean volume data: replace negatives and extreme outliers."""
    stats = {"replaced_neg": 0, "replaced_extreme": 0}
    
    # Remove negatives
    neg_mask = (df < 0) & df.notna()
    stats["replaced_neg"] = int(neg_mask.sum().sum())
    df = df.where(~neg_mask)
    
    # Cap at 99.9th percentile per column (handles data errors)
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 100:
            cap = vals.quantile(0.999)
            extreme = df[col] > cap * 10  # 10x the 99.9th percentile is surely an error
            stats["replaced_extreme"] += int(extreme.sum())
            df.loc[extreme, col] = np.nan
    
    return df, stats


def clean_market_cap(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean market cap: no negatives, cap at reasonable maximums."""
    stats = {"replaced_neg": 0, "replaced_extreme": 0}
    
    neg_mask = (df < 0) & df.notna()
    stats["replaced_neg"] = int(neg_mask.sum().sum())
    df = df.where(~neg_mask)
    
    # Cap at $10T (no company is worth > $10T)
    extreme = df > 10e12
    stats["replaced_extreme"] = int(extreme.sum().sum())
    df = df.where(~extreme)
    
    return df, stats


def clean_ratios(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean ratio data: winsorize at 1st/99th percentile."""
    stats = {"winsorized": 0}
    
    # Replace Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 50:
            q01 = vals.quantile(0.01)
            q99 = vals.quantile(0.99)
            clip_mask = ((df[col] < q01) | (df[col] > q99)) & df[col].notna()
            stats["winsorized"] += int(clip_mask.sum())
            df[col] = df[col].clip(q01, q99)
    
    return df, stats


def clean_per_share(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean per-share data: winsorize and remove extreme values."""
    stats = {"winsorized": 0, "extreme_removed": 0}
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove values where |value| > 1e6 (no stock has $1M EPS)
    extreme = df.abs() > 1_000_000
    stats["extreme_removed"] = int(extreme.sum().sum())
    df = df.where(~extreme)
    
    # Winsorize at 1st/99th
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 50:
            q01 = vals.quantile(0.01)
            q99 = vals.quantile(0.99)
            clip_mask = ((df[col] < q01) | (df[col] > q99)) & df[col].notna()
            stats["winsorized"] += int(clip_mask.sum())
            df[col] = df[col].clip(q01, q99)
    
    return df, stats


def clean_volatility(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean volatility: must be non-negative, cap at reasonable values."""
    stats = {"replaced_neg": 0, "capped": 0}
    
    neg_mask = (df < 0) & df.notna()
    stats["replaced_neg"] = int(neg_mask.sum().sum())
    df = df.where(~neg_mask)
    
    # Cap at 500% annualized vol
    cap_mask = (df > 5.0) & df.notna()
    stats["capped"] = int(cap_mask.sum().sum())
    df = df.clip(upper=5.0)
    
    return df, stats


def clean_general(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Clean general numeric data: replace Inf, winsorize extreme outliers."""
    stats = {"inf_replaced": 0, "extreme_removed": 0}
    
    # Check if numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df, stats
    
    # Replace Inf
    for col in numeric_cols:
        inf_mask = np.isinf(df[col].values.astype(float))
        stats["inf_replaced"] += int(inf_mask.sum())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Remove extreme outliers (> 1e12 absolute value)
    for col in numeric_cols:
        extreme = df[col].abs() > 1e12
        if extreme.any():
            stats["extreme_removed"] += int(extreme.sum())
            df.loc[extreme, col] = np.nan
    
    return df, stats


def forward_fill(df: pd.DataFrame, limit: int = 5) -> Tuple[pd.DataFrame, int]:
    """Forward-fill NaN values up to `limit` days."""
    before_nan = int(df.isna().sum().sum())
    df = df.ffill(limit=limit)
    after_nan = int(df.isna().sum().sum())
    filled = before_nan - after_nan
    return df, filled


# ═════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════

def run_data_cleaning():
    t0 = time.time()
    
    print("=" * 70)
    print("  DATA CLEANING PIPELINE")
    print("=" * 70)
    
    mdir = "data/fmp_cache/matrices"
    outdir = "data/fmp_cache/matrices_clean"
    os.makedirs(outdir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(mdir) if f.endswith(".parquet") and not f.startswith("_")])
    print(f"\n  Scanning {len(files)} data fields...\n")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_fields": len(files),
        "fields": {},
        "summary": {},
    }
    
    total_issues = 0
    total_cleaned = 0
    total_ffilled = 0
    
    category_counts = defaultdict(int)
    issue_counts = defaultdict(int)
    
    for i, fn in enumerate(files):
        name = fn.replace(".parquet", "")
        category = classify_field(name)
        category_counts[category] += 1
        
        df = pd.read_parquet(f"{mdir}/{fn}")
        
        # Pre-cleaning stats
        pre_shape = df.shape
        pre_nan = int(df.select_dtypes(include=[np.number]).isna().sum().sum()) if len(df.select_dtypes(include=[np.number]).columns) > 0 else 0
        pre_total = df.select_dtypes(include=[np.number]).size if len(df.select_dtypes(include=[np.number]).columns) > 0 else 0
        
        # Skip non-numeric (classifications)
        if category == "classification":
            df.to_parquet(f"{outdir}/{fn}")
            report["fields"][name] = {
                "category": category,
                "shape": list(pre_shape),
                "issues": 0,
                "status": "skipped (non-numeric)",
            }
            continue
        
        # Apply cleaning
        stats = {}
        if category == "price":
            df, stats = clean_prices(df)
        elif category == "return":
            df, stats = clean_returns(df)
        elif category == "volume":
            df, stats = clean_volumes(df)
        elif category == "market_cap":
            df, stats = clean_market_cap(df)
        elif category == "ratio":
            df, stats = clean_ratios(df)
        elif category == "per_share":
            df, stats = clean_per_share(df)
        elif category == "volatility":
            df, stats = clean_volatility(df)
        else:
            df, stats = clean_general(df)
        
        # Forward-fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols], ffilled = forward_fill(df[numeric_cols], limit=5)
        else:
            ffilled = 0
        
        # Count issues
        issues_in_field = sum(v for v in stats.values() if isinstance(v, (int, float)))
        total_issues += issues_in_field
        total_cleaned += issues_in_field
        total_ffilled += ffilled
        
        for k, v in stats.items():
            if v > 0:
                issue_counts[k] += v
        
        # Post-cleaning stats
        post_nan = int(df.select_dtypes(include=[np.number]).isna().sum().sum()) if len(numeric_cols) > 0 else 0
        nan_pct = post_nan / pre_total * 100 if pre_total > 0 else 0
        
        # Save
        df.to_parquet(f"{outdir}/{fn}")
        
        # Status indicator
        if issues_in_field > 100:
            status = "🔴 MAJOR"
        elif issues_in_field > 0:
            status = "🟡 MINOR"
        else:
            status = "🟢 CLEAN"
        
        report["fields"][name] = {
            "category": category,
            "shape": list(pre_shape),
            "issues": issues_in_field,
            "ffilled": ffilled,
            "nan_pct_after": round(nan_pct, 1),
            "cleaning_stats": stats,
            "status": status,
        }
        
        if issues_in_field > 0 or (i % 30 == 0):
            print(f"  [{i+1:3d}/{len(files)}] {status} {name:45s} cat={category:12s} issues={issues_in_field:>8,d} ffill={ffilled:>8,d}")
    
    # ── Recompute derived fields ──
    print("\n  Recomputing derived fields...")
    
    # Recompute returns from clean close
    close = pd.read_parquet(f"{outdir}/close.parquet")
    clean_returns_df = close.pct_change().clip(-1.0, 1.0)
    clean_returns_df.to_parquet(f"{outdir}/returns.parquet")
    print("    ✅ returns recomputed from clean close")
    
    # Recompute log_returns
    clean_log_returns = np.log(close / close.shift(1))
    clean_log_returns = clean_log_returns.replace([np.inf, -np.inf], np.nan)
    clean_log_returns = clean_log_returns.clip(-1.0, 1.0)
    clean_log_returns.to_parquet(f"{outdir}/log_returns.parquet")
    print("    ✅ log_returns recomputed from clean close")
    
    # ── Summary ──
    elapsed = time.time() - t0
    
    # Count by severity
    major = sum(1 for f in report["fields"].values() if "MAJOR" in f.get("status", ""))
    minor = sum(1 for f in report["fields"].values() if "MINOR" in f.get("status", ""))
    clean = sum(1 for f in report["fields"].values() if "CLEAN" in f.get("status", ""))
    
    report["summary"] = {
        "total_fields": len(files),
        "major_issues": major,
        "minor_issues": minor,
        "clean_fields": clean,
        "total_data_points_fixed": total_cleaned,
        "total_ffilled": total_ffilled,
        "issue_breakdown": dict(issue_counts),
        "category_counts": dict(category_counts),
        "runtime_seconds": round(elapsed, 1),
    }
    
    # Save report
    with open("data/data_integrity_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print report
    print(f"\n{'='*70}")
    print(f"  DATA INTEGRITY REPORT")
    print(f"{'='*70}")
    print(f"\n  Fields scanned:   {len(files)}")
    print(f"  🔴 Major issues:  {major}")
    print(f"  🟡 Minor issues:  {minor}")
    print(f"  🟢 Clean:         {clean}")
    print(f"\n  Total data points fixed: {total_cleaned:,d}")
    print(f"  Total NaN forward-filled: {total_ffilled:,d}")
    
    print(f"\n  Issue breakdown:")
    for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:30s}: {v:>10,d}")
    
    print(f"\n  Field categories:")
    for k, v in sorted(category_counts.items()):
        print(f"    {k:15s}: {v}")
    
    # Top 10 worst fields
    worst = sorted(report["fields"].items(), key=lambda x: -x[1].get("issues", 0))[:15]
    print(f"\n  Top 15 worst fields:")
    for name, info in worst:
        if info["issues"] > 0:
            print(f"    {name:45s} {info['status']} issues={info['issues']:>8,d} ({info['category']})")
    
    print(f"\n  Output: data/fmp_cache/matrices_clean/")
    print(f"  Report: data/data_integrity_report.json")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_data_cleaning()
