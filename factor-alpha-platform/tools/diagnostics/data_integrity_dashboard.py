"""
Data Integrity Dashboard — Comprehensive quality report for Binance 4h matrices.

Generates an HTML dashboard with PASS/FAIL checks:
  1. Kline file completeness (raw downloads — expected bars vs actual)
  2. Per-matrix NaN / Inf / coverage analysis
  3. Price sanity (negative, zero, extreme prices)
  4. Return sanity (extreme returns, suspiciously flat)
  5. Volume sanity (negative, zero volume)
  6. Cross-matrix consistency (close vs returns alignment)
  7. Duplicate bar detection
  8. Gap detection (missing bars within a symbol's active range)
  9. Data freshness (staleness check)
  10. Universe coverage depth

Exit code 0 = ALL GREEN, exit code 1 = FAILURES.

Usage:
    python data_integrity_dashboard.py [--interval 4h]
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path("data/binance_cache")

# ── Thresholds ──
PRICE_MAX = 200_000        # BTC ~$100K, allow 2x headroom
PRICE_MIN_ABS = 1e-10      # basically zero
RETURN_MAX_ABS = 5.0        # 500% single-bar return (4h) is extreme
RETURN_WARN_ABS = 1.0       # 100% single-bar is suspicious
MIN_EXPECTED_BARS_4H = {    # rough minimum bars from listing to now
    "BTCUSDT": 13000,
    "ETHUSDT": 13000,
    "BNBUSDT": 10000,
}
FRESHNESS_HOURS = 12        # data older than this is "stale"
GAP_MULTIPLE = 3            # gap > 3× interval is flagged
INTERVAL_HOURS = {"4h": 4, "1d": 24, "12h": 12}


def log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════
#  CHECK 1: Kline file completeness
# ══════════════════════════════════════════════════════════════════════

def check_kline_files(interval: str = "4h") -> dict:
    """Check raw kline parquet files for completeness and sanity."""
    kline_dir = DATA_DIR / "klines" / interval
    if not kline_dir.exists():
        return {}
    
    results = {}
    for fpath in sorted(kline_dir.glob("*.parquet")):
        sym = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                results[sym] = {"n_bars": 0, "status": "EMPTY", "error": "Empty file"}
                continue
            
            dt_col = df["datetime"]
            n_bars = len(df)
            dt_min = dt_col.min()
            dt_max = dt_col.max()
            
            # Check for duplicates
            n_dupes = n_bars - dt_col.nunique()
            
            # Check freshness
            now = pd.Timestamp.now(tz=None)
            hours_stale = (now - dt_max).total_seconds() / 3600
            
            # Expected bars from listing to now
            span_hours = (dt_max - dt_min).total_seconds() / 3600
            int_hours = INTERVAL_HOURS.get(interval, 4)
            expected_bars = int(span_hours / int_hours) + 1
            coverage_ratio = n_bars / max(expected_bars, 1)
            
            # Detect gaps
            diffs = dt_col.sort_values().diff().dropna()
            expected_delta = pd.Timedelta(hours=int_hours)
            gap_count = int((diffs > expected_delta * GAP_MULTIPLE).sum())
            max_gap_hours = diffs.max().total_seconds() / 3600 if len(diffs) > 0 else 0
            
            results[sym] = {
                "n_bars": n_bars,
                "date_min": str(dt_min),
                "date_max": str(dt_max),
                "hours_stale": round(hours_stale, 1),
                "n_dupes": n_dupes,
                "expected_bars": expected_bars,
                "coverage_ratio": round(coverage_ratio, 4),
                "gap_count": gap_count,
                "max_gap_hours": round(max_gap_hours, 1),
                "file_size_kb": round(fpath.stat().st_size / 1024, 1),
                "status": "OK",
            }
        except Exception as e:
            results[sym] = {"n_bars": 0, "status": "ERROR", "error": str(e)}
    
    return results


# ══════════════════════════════════════════════════════════════════════
#  CHECK 2-5: Matrix quality checks
# ══════════════════════════════════════════════════════════════════════

def analyze_matrix(name: str, df: pd.DataFrame) -> dict:
    """Compute quality metrics for a single matrix DataFrame."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    
    numeric_df = df.select_dtypes(include=[np.number])
    n_nan = int(df.isna().sum().sum())
    n_inf = int(np.isinf(numeric_df.values).sum()) if numeric_df.shape[1] > 0 else 0
    
    nan_pct = 100.0 * n_nan / max(total_cells, 1)
    
    # Per-symbol coverage
    per_sym_valid = df.notna().mean()
    per_time_valid = df.notna().mean(axis=1)
    
    # All-NaN analysis
    all_nan_rows = int(df.isna().all(axis=1).sum())
    all_nan_cols = int(df.isna().all(axis=0).sum())
    
    result = {
        "shape": f"{n_rows}×{n_cols}",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_cells": total_cells,
        "nan_count": n_nan,
        "nan_pct": round(nan_pct, 2),
        "inf_count": n_inf,
        "all_nan_rows": all_nan_rows,
        "all_nan_cols": all_nan_cols,
        "per_sym_coverage_min": round(per_sym_valid.min(), 4),
        "per_sym_coverage_max": round(per_sym_valid.max(), 4),
        "per_sym_coverage_mean": round(per_sym_valid.mean(), 4),
        "per_time_coverage_min": round(per_time_valid.min(), 4),
        "per_time_coverage_mean": round(per_time_valid.mean(), 4),
        "symbols_with_full_data": int((per_sym_valid > 0.99).sum()),
        "symbols_with_no_data": int((per_sym_valid < 0.01).sum()),
        "date_range": f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "empty",
        "issues": [],
    }
    
    # Exact-match names for price/volume checks (exclude derived ratios/deviations)
    PRICE_MATRICES = {"close", "open", "high", "low", "vwap"}
    VOLUME_MATRICES = {"volume", "quote_volume", "taker_buy_volume", 
                       "taker_buy_quote_volume", "dollars_traded",
                       "adv20", "adv60", "trades_count"}
    
    # ── Price checks (exact name match only) ──
    if name in PRICE_MATRICES:
        vals = numeric_df.values.ravel()
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            neg = int(np.sum(valid < 0))
            zeros = int(np.sum(valid == 0))
            extreme = int(np.sum(valid > PRICE_MAX))
            result["negative_prices"] = neg
            result["zero_prices"] = zeros
            result["extreme_prices"] = extreme
            result["price_min"] = float(np.nanmin(valid))
            result["price_max"] = float(np.nanmax(valid))
            if neg > 0:
                result["issues"].append(("FAIL", f"{neg} negative prices"))
            if zeros > 0 and name == "close":
                result["issues"].append(("FAIL", f"{zeros} zero close prices"))
            if extreme > 0:
                result["issues"].append(("WARN", f"{extreme} prices > ${PRICE_MAX:,}"))
    
    # ── Return checks ──
    elif name in {"returns", "log_returns"}:
        vals = numeric_df.values.ravel()
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            extreme_count = int(np.sum(np.abs(valid) > RETURN_MAX_ABS))
            extreme_warn = int(np.sum(np.abs(valid) > RETURN_WARN_ABS))
            result["return_min"] = float(np.nanmin(valid))
            result["return_max"] = float(np.nanmax(valid))
            result["extreme_returns_500pct"] = extreme_count
            result["extreme_returns_100pct"] = extreme_warn
            result["mean_return"] = float(np.nanmean(valid))
            if extreme_count > 0:
                result["issues"].append(("WARN", f"{extreme_count} returns > ±500%"))
    
    # ── Volume checks (exact name match only) ──
    elif name in VOLUME_MATRICES:
        vals = numeric_df.values.ravel()
        valid = vals[np.isfinite(vals)]
        if len(valid) > 0:
            neg = int(np.sum(valid < 0))
            zeros = int(np.sum(valid == 0))
            result["negative_volumes"] = neg
            result["zero_volumes"] = zeros
            result["vol_min"] = float(np.nanmin(valid))
            result["vol_max"] = float(np.nanmax(valid))
            if neg > 0:
                result["issues"].append(("FAIL", f"{neg} negative volumes"))
    
    # ── Universal checks ──
    if n_inf > 0:
        result["issues"].append(("FAIL", f"{n_inf} Inf values"))
    if all_nan_cols > 0:
        result["issues"].append(("WARN", f"{all_nan_cols} entirely-NaN columns"))
    if nan_pct > 95:
        result["issues"].append(("FAIL", f"NaN% = {nan_pct:.1f}% — matrix mostly empty"))
    elif nan_pct > 80:
        result["issues"].append(("WARN", f"NaN% = {nan_pct:.1f}%"))
    
    return result


# ══════════════════════════════════════════════════════════════════════
#  CHECK 6: Cross-matrix consistency
# ══════════════════════════════════════════════════════════════════════

def check_cross_consistency(matrices: dict) -> List[Tuple[str, str, str]]:
    """Check relationships between matrices. Returns (status, check_name, detail)."""
    checks = []
    
    close = matrices.get("close")
    returns = matrices.get("returns")
    high = matrices.get("high")
    low = matrices.get("low")
    open_ = matrices.get("open")
    volume = matrices.get("volume")
    
    # 1. close & returns should share indices/columns
    if close is not None and returns is not None:
        if close.shape != returns.shape:
            checks.append(("FAIL", "Shape match (close vs returns)", 
                          f"close={close.shape}, returns={returns.shape}"))
        else:
            checks.append(("PASS", "Shape match (close vs returns)", 
                          f"Both {close.shape}"))
        
        # Verify returns ≈ close.pct_change()
        recomputed = close.pct_change()
        common = close.columns.intersection(returns.columns)
        if len(common) > 0:
            diff = (returns[common] - recomputed[common]).abs()
            max_diff = diff.max().max()
            if np.isfinite(max_diff) and max_diff < 1e-8:
                checks.append(("PASS", "Returns = close.pct_change()", 
                              f"Max diff: {max_diff:.2e}"))
            elif np.isfinite(max_diff):
                checks.append(("WARN", "Returns ≈ close.pct_change()", 
                              f"Max diff: {max_diff:.2e}"))
            else:
                checks.append(("WARN", "Returns vs close.pct_change()", 
                              "Contains Inf diffs (likely divide-by-zero)"))
    
    # 2. High >= Close >= Low (where all three exist)
    if close is not None and high is not None and low is not None:
        common = close.columns.intersection(high.columns).intersection(low.columns)
        if len(common) > 0:
            h = high[common].values
            l = low[common].values
            c = close[common].values
            mask = np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
            violations_high = int(np.sum((c > h) & mask))
            violations_low = int(np.sum((c < l) & mask))
            total_valid = int(mask.sum())
            if violations_high == 0 and violations_low == 0:
                checks.append(("PASS", "Price ordering (L ≤ C ≤ H)", 
                              f"0 violations in {total_valid:,} cells"))
            else:
                checks.append(("WARN", "Price ordering (L ≤ C ≤ H)", 
                              f"{violations_high} high violations, {violations_low} low violations"))
    
    # 3. Volume ≥ 0
    if volume is not None:
        vals = volume.values.ravel()
        valid = vals[np.isfinite(vals)]
        neg_count = int(np.sum(valid < 0))
        if neg_count == 0:
            checks.append(("PASS", "Volume ≥ 0", f"No negative volumes"))
        else:
            checks.append(("FAIL", "Volume ≥ 0", f"{neg_count} negative values"))
    
    # 4. All matrices have same index
    if close is not None:
        ref_index = close.index
        for name, mat in matrices.items():
            if name == "close":
                continue
            if not mat.index.equals(ref_index):
                # Check if at least first/last match
                if len(mat.index) == len(ref_index):
                    if mat.index[0] == ref_index[0] and mat.index[-1] == ref_index[-1]:
                        checks.append(("PASS", f"Index alignment ({name})", 
                                      f"Same length & endpoints"))
                    else:
                        checks.append(("WARN", f"Index alignment ({name})", 
                                      f"Same length but different dates"))
                else:
                    checks.append(("FAIL", f"Index alignment ({name})", 
                                  f"close has {len(ref_index)} rows, {name} has {len(mat.index)}"))
    
    return checks


# ══════════════════════════════════════════════════════════════════════
#  CHECK 7-8: Duplicate & gap detection (per-symbol)
# ══════════════════════════════════════════════════════════════════════

def check_kline_gaps_and_dupes(interval: str = "4h") -> dict:
    """Detailed gap/duplicate check on raw kline files."""
    kline_dir = DATA_DIR / "klines" / interval
    if not kline_dir.exists():
        return {"total_dupes": 0, "total_gaps": 0, "symbols_with_gaps": 0}
    
    total_dupes = 0
    total_gaps = 0
    symbols_with_gaps = 0
    symbols_with_dupes = 0
    gap_details = []
    int_hours = INTERVAL_HOURS.get(interval, 4)
    expected_delta = pd.Timedelta(hours=int_hours)
    
    for fpath in sorted(kline_dir.glob("*.parquet")):
        sym = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if df.empty or len(df) < 2:
                continue
            
            dts = df["datetime"].sort_values()
            
            # Duplicates
            dupes = dts.duplicated().sum()
            if dupes > 0:
                total_dupes += dupes
                symbols_with_dupes += 1
            
            # Gaps
            diffs = dts.diff().dropna()
            gaps = diffs[diffs > expected_delta * GAP_MULTIPLE]
            if len(gaps) > 0:
                total_gaps += len(gaps)
                symbols_with_gaps += 1
                max_gap = gaps.max()
                gap_details.append((sym, len(gaps), max_gap.total_seconds() / 3600))
        except Exception:
            continue
    
    return {
        "total_dupes": total_dupes,
        "symbols_with_dupes": symbols_with_dupes,
        "total_gaps": total_gaps,
        "symbols_with_gaps": symbols_with_gaps,
        "worst_gap_symbols": sorted(gap_details, key=lambda x: -x[2])[:10],
    }


# ══════════════════════════════════════════════════════════════════════
#  CHECK 9: Data freshness
# ══════════════════════════════════════════════════════════════════════

def check_freshness(kline_info: dict, interval: str) -> List[Tuple[str, str, str]]:
    """Check how fresh the data is."""
    checks = []
    
    n_stale = sum(1 for v in kline_info.values() 
                  if v.get("hours_stale", 999) > FRESHNESS_HOURS)
    n_total = len(kline_info)
    stale_pct = 100.0 * n_stale / max(n_total, 1)
    
    if stale_pct < 5:
        checks.append(("PASS", f"Data freshness (<{FRESHNESS_HOURS}h old)", 
                       f"{n_total - n_stale}/{n_total} symbols up to date"))
    elif stale_pct < 20:
        checks.append(("WARN", f"Data freshness (<{FRESHNESS_HOURS}h old)", 
                       f"{n_stale}/{n_total} symbols are stale ({stale_pct:.0f}%)"))
    else:
        checks.append(("FAIL", f"Data freshness (<{FRESHNESS_HOURS}h old)", 
                       f"{n_stale}/{n_total} symbols are stale ({stale_pct:.0f}%)"))
    
    return checks


# ══════════════════════════════════════════════════════════════════════
#  CHECK 10: Universe coverage
# ══════════════════════════════════════════════════════════════════════

def check_universe_coverage(interval: str) -> List[Tuple[str, str, str]]:
    """Check universe mask files exist and have reasonable coverage."""
    checks = []
    universe_dir = DATA_DIR / "universes"
    
    suffix = f"_{interval}" if interval != "1d" else ""
    for tier in ["TOP20", "TOP50", "TOP100"]:
        fname = f"BINANCE_{tier}{suffix}.parquet"
        fpath = universe_dir / fname
        if not fpath.exists():
            checks.append(("FAIL", f"Universe {tier} exists", f"{fname} not found"))
            continue
        
        try:
            mask = pd.read_parquet(fpath)
            avg_count = mask.sum(axis=1).mean()
            target = int(tier.replace("TOP", ""))
            fill_pct = 100.0 * avg_count / target
            
            # Check recent fill (last year) — early periods have fewer eligible symbols
            recent_mask = mask.loc[mask.index > mask.index[-1] - pd.Timedelta(days=365)]
            recent_avg = recent_mask.sum(axis=1).mean() if len(recent_mask) > 0 else avg_count
            recent_fill_pct = 100.0 * recent_avg / target
            
            if recent_fill_pct > 80:
                checks.append(("PASS", f"Universe {tier} fill", 
                              f"Recent: {recent_avg:.0f}/{target} ({recent_fill_pct:.0f}%), overall avg: {avg_count:.0f}"))
            elif recent_fill_pct > 50:
                checks.append(("WARN", f"Universe {tier} fill", 
                              f"Recent: {recent_avg:.0f}/{target} ({recent_fill_pct:.0f}%)"))
            else:
                checks.append(("FAIL", f"Universe {tier} fill", 
                              f"Recent: {recent_avg:.0f}/{target} — likely data issue"))
        except Exception as e:
            checks.append(("FAIL", f"Universe {tier} readable", str(e)))
    
    return checks


# ══════════════════════════════════════════════════════════════════════
#  HTML Report Generation
# ══════════════════════════════════════════════════════════════════════

def status_icon(status: str) -> str:
    if status == "PASS":
        return '🟢'
    elif status == "WARN":
        return '🟡'
    else:
        return '🔴'

def status_class(status: str) -> str:
    return {"PASS": "ok", "WARN": "warn", "FAIL": "error"}.get(status, "error")


def generate_html_report(
    matrices: dict,
    kline_info: dict,
    interval: str,
    all_checks: List[Tuple[str, str, str]],
    analyses: dict,
    gap_info: dict,
) -> str:
    """Generate comprehensive HTML dashboard."""
    
    n_pass = sum(1 for s, _, _ in all_checks if s == "PASS")
    n_warn = sum(1 for s, _, _ in all_checks if s == "WARN")
    n_fail = sum(1 for s, _, _ in all_checks if s == "FAIL")
    all_green = n_fail == 0 and n_warn == 0
    
    close_a = analyses.get("close", {})
    n_symbols = close_a.get("n_cols", 0)
    n_bars = close_a.get("n_rows", 0)
    nan_pct = close_a.get("nan_pct", 0)
    
    # Kline stats
    full_klines = sum(1 for v in kline_info.values() if v.get("n_bars", 0) > 5000)
    partial_klines = sum(1 for v in kline_info.values() if 1000 < v.get("n_bars", 0) <= 5000)
    sparse_klines = sum(1 for v in kline_info.values() if 0 < v.get("n_bars", 0) <= 1000)
    
    overall_status = "ALL GREEN ✅" if all_green else f"{n_fail} FAILURES ❌" if n_fail > 0 else f"{n_warn} WARNINGS ⚠️"
    overall_color = "#4ade80" if all_green else "#ef4444" if n_fail > 0 else "#fbbf24"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Integrity Dashboard — {interval}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Inter', -apple-system, sans-serif;
            background: #07070d;
            color: #e0e0e8;
            padding: 24px;
            min-height: 100vh;
        }}
        
        .header {{
            background: linear-gradient(135deg, #10102a 0%, #0f1b36 100%);
            border: 1px solid rgba(58, 123, 213, 0.3);
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }}
        .header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, {overall_color}, transparent);
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header .meta {{ color: #6b7280; font-size: 13px; }}
        .header .overall-status {{
            position: absolute;
            right: 32px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 22px;
            font-weight: 700;
            color: {overall_color};
            text-align: right;
        }}
        .header .overall-status .sub {{
            font-size: 12px;
            color: #6b7280;
            font-weight: 400;
        }}
        
        /* Summary stats */
        .summary-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 24px; }}
        .summary-stat {{
            background: linear-gradient(180deg, #111120 0%, #0d0d18 100%);
            border: 1px solid #1e1e36;
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            transition: border-color 0.2s, transform 0.2s;
        }}
        .summary-stat:hover {{ border-color: #3a7bd5; transform: translateY(-2px); }}
        .summary-stat .number {{
            font-size: 30px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.2;
        }}
        .summary-stat .label {{
            font-size: 10px;
            color: #6b7280;
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        /* Cards */
        .card {{
            background: linear-gradient(180deg, #111120 0%, #0d0d18 100%);
            border: 1px solid #1e1e36;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: border-color 0.2s;
        }}
        .card:hover {{ border-color: #2a3a5a; }}
        .card h3 {{
            font-size: 13px;
            color: #3a7bd5;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }}
        
        /* Grid */
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 18px; margin-bottom: 20px; }}
        .full-width {{ grid-column: 1 / -1; }}
        
        /* Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        th {{
            text-align: left;
            padding: 10px 8px;
            color: #3a7bd5;
            border-bottom: 2px solid #1e1e36;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #141428;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }}
        tr:hover {{ background: rgba(58, 123, 213, 0.04); }}
        
        /* Checks table */
        .check-row {{ display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #141428; }}
        .check-row:last-child {{ border-bottom: none; }}
        .check-icon {{ font-size: 16px; margin-right: 12px; flex-shrink: 0; }}
        .check-name {{ flex: 1; font-size: 13px; font-weight: 500; }}
        .check-detail {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8b8fa3; text-align: right; }}
        
        /* Status colors */
        .ok {{ color: #4ade80; }}
        .warn {{ color: #fbbf24; }}
        .error {{ color: #ef4444; }}
        .muted {{ color: #6b7280; }}
        
        /* Coverage bars */
        .bar-container {{
            width: 100px; height: 6px; background: #1a1a2e;
            border-radius: 3px; display: inline-block; vertical-align: middle;
        }}
        .bar {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
        .bar-ok {{ background: linear-gradient(90deg, #4ade80, #22c55e); }}
        .bar-warn {{ background: linear-gradient(90deg, #fbbf24, #f59e0b); }}
        .bar-error {{ background: linear-gradient(90deg, #ef4444, #dc2626); }}
        
        @media (max-width: 768px) {{
            .summary-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .grid {{ grid-template-columns: 1fr; }}
            .header .overall-status {{ position: static; transform: none; margin-top: 12px; text-align: left; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Data Integrity Dashboard</h1>
        <div class="meta">
            Interval: {interval} &nbsp;|&nbsp;
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
            Matrices: {len(matrices)} &nbsp;|&nbsp;
            Kline files: {len(kline_info)}
        </div>
        <div class="overall-status">
            {overall_status}
            <div class="sub">{n_pass} pass · {n_warn} warn · {n_fail} fail</div>
        </div>
    </div>
"""
    
    # ── Summary stats ──
    nan_class = "ok" if nan_pct < 30 else ("warn" if nan_pct < 60 else "error")
    html += f"""
    <div class="summary-grid">
        <div class="summary-stat">
            <div class="number">{len(matrices)}</div>
            <div class="label">Matrices</div>
        </div>
        <div class="summary-stat">
            <div class="number">{n_symbols}</div>
            <div class="label">Symbols</div>
        </div>
        <div class="summary-stat">
            <div class="number">{n_bars:,}</div>
            <div class="label">Total Bars</div>
        </div>
        <div class="summary-stat">
            <div class="number {nan_class}">{nan_pct:.1f}%</div>
            <div class="label">Close NaN %</div>
        </div>
        <div class="summary-stat">
            <div class="number ok">{full_klines}</div>
            <div class="label">Full Klines (&gt;5K)</div>
        </div>
        <div class="summary-stat">
            <div class="number error">{sparse_klines}</div>
            <div class="label">Sparse (&lt;1K)</div>
        </div>
    </div>
"""
    
    # ── All integrity checks ──
    html += """
    <div class="card full-width">
        <h3>🔍 Integrity Checks</h3>
"""
    for status, check_name, detail in all_checks:
        icon = status_icon(status)
        cls = status_class(status)
        html += f"""        <div class="check-row">
            <span class="check-icon">{icon}</span>
            <span class="check-name">{check_name}</span>
            <span class="check-detail {cls}">{detail}</span>
        </div>
"""
    html += "    </div>\n"
    
    # ── Matrix quality table ──
    html += """
    <div class="card full-width">
        <h3>📋 Matrix Quality Summary</h3>
        <table>
            <tr>
                <th>Matrix</th>
                <th>Shape</th>
                <th>NaN %</th>
                <th>Coverage</th>
                <th>Full Syms</th>
                <th>Empty Syms</th>
                <th>Issues</th>
            </tr>
"""
    for name, a in sorted(analyses.items()):
        nan_p = a["nan_pct"]
        bar_class = "bar-ok" if nan_p < 30 else ("bar-warn" if nan_p < 60 else "bar-error")
        nan_cls = "ok" if nan_p < 30 else ("warn" if nan_p < 60 else "error")
        coverage_width = max(0, 100 - nan_p)
        
        issues_list = a.get("issues", [])
        if not issues_list:
            issues_str = '<span class="ok">✓</span>'
        else:
            parts = []
            for sev, msg in issues_list:
                cls = status_class(sev)
                parts.append(f'<span class="{cls}">{msg}</span>')
            issues_str = " · ".join(parts)
        
        html += f"""            <tr>
                <td><strong>{name}</strong></td>
                <td>{a['shape']}</td>
                <td class="{nan_cls}">{nan_p:.1f}%</td>
                <td>
                    <div class="bar-container">
                        <div class="bar {bar_class}" style="width:{coverage_width}%"></div>
                    </div>
                    {a['per_sym_coverage_mean']:.1%}
                </td>
                <td>{a.get('symbols_with_full_data', '?')}</td>
                <td>{a.get('symbols_with_no_data', '?')}</td>
                <td>{issues_str}</td>
            </tr>
"""
    html += "        </table>\n    </div>\n"
    
    # ── Kline file details ──
    if kline_info:
        sorted_klines = sorted(kline_info.items(), 
                               key=lambda x: x[1].get("n_bars", 0), reverse=True)
        html += """
    <div class="card full-width">
        <h3>📦 Kline File Completeness</h3>
        <table>
            <tr><th>Symbol</th><th>Bars</th><th>Start</th><th>End</th>
            <th>Coverage</th><th>Gaps</th><th>Stale (hrs)</th><th>Status</th></tr>
"""
        for sym, info in sorted_klines[:40]:
            n = info.get("n_bars", 0)
            cov = info.get("coverage_ratio", 0)
            gaps = info.get("gap_count", 0)
            stale_h = info.get("hours_stale", 999)
            
            if n > 5000 and stale_h < FRESHNESS_HOURS and cov > 0.95:
                status_text = "FULL"
                st_class = "ok"
            elif n > 1000:
                status_text = "PARTIAL"
                st_class = "warn"
            elif n > 0:
                status_text = "SPARSE"
                st_class = "error"
            else:
                status_text = "EMPTY"
                st_class = "error"
            
            stale_cls = "ok" if stale_h < FRESHNESS_HOURS else ("warn" if stale_h < 48 else "error")
            gap_cls = "ok" if gaps == 0 else ("warn" if gaps < 5 else "error")
            
            html += f"""            <tr>
                <td>{sym}</td>
                <td class="{st_class}">{n:,}</td>
                <td class="muted">{info.get('date_min', '?')[:10]}</td>
                <td class="muted">{info.get('date_max', '?')[:10]}</td>
                <td>{cov:.0%}</td>
                <td class="{gap_cls}">{gaps}</td>
                <td class="{stale_cls}">{stale_h:.0f}h</td>
                <td class="{st_class}">{status_text}</td>
            </tr>
"""
        if len(sorted_klines) > 40:
            html += f'            <tr><td colspan="8" class="muted">... and {len(sorted_klines)-40} more symbols</td></tr>\n'
        html += "        </table>\n    </div>\n"
    
    # ── Gap details ──
    if gap_info.get("worst_gap_symbols"):
        html += """
    <div class="card full-width">
        <h3>⚠️ Symbols with Largest Gaps</h3>
        <table>
            <tr><th>Symbol</th><th># Gaps</th><th>Max Gap (hours)</th></tr>
"""
        for sym, n_gaps, max_gap_h in gap_info["worst_gap_symbols"]:
            gap_cls = "warn" if max_gap_h < 48 else "error"
            html += f'            <tr><td>{sym}</td><td>{n_gaps}</td><td class="{gap_cls}">{max_gap_h:.0f}h</td></tr>\n'
        html += "        </table>\n    </div>\n"
    
    # ── Symbol coverage (close matrix) ──
    close = matrices.get("close")
    if close is not None:
        sym_coverage = close.notna().mean().sort_values(ascending=False)
        
        html += """
    <div class="grid">
        <div class="card">
            <h3>🟢 Best Coverage (Close)</h3>
            <table>
                <tr><th>Symbol</th><th>Coverage</th><th></th></tr>
"""
        for sym, cov in sym_coverage.head(15).items():
            cls = "ok" if cov > 0.9 else "warn"
            bar_cls = "bar-ok" if cov > 0.9 else "bar-warn"
            html += f'                <tr><td>{sym}</td><td class="{cls}">{cov:.1%}</td><td><div class="bar-container"><div class="bar {bar_cls}" style="width:{cov*100:.0f}%"></div></div></td></tr>\n'
        html += """            </table>
        </div>
        <div class="card">
            <h3>🔴 Worst Coverage (Close)</h3>
            <table>
                <tr><th>Symbol</th><th>Coverage</th><th></th></tr>
"""
        for sym, cov in sym_coverage.tail(15).items():
            cls = "ok" if cov > 0.9 else ("warn" if cov > 0.3 else "error")
            bar_cls = "bar-ok" if cov > 0.9 else ("bar-warn" if cov > 0.3 else "bar-error")
            w = max(cov * 100, 1)
            html += f'                <tr><td>{sym}</td><td class="{cls}">{cov:.1%}</td><td><div class="bar-container"><div class="bar {bar_cls}" style="width:{w:.0f}%"></div></div></td></tr>\n'
        html += """            </table>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    return html


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Data Integrity Dashboard")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--output", default="data_integrity_report.html")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("DATA INTEGRITY DASHBOARD", flush=True)
    print(f"  Interval: {args.interval}", flush=True)
    print("=" * 60, flush=True)

    # ── Load data ──
    log("Loading matrices...")
    matrices_dir = DATA_DIR / "matrices" / args.interval if args.interval != "1d" else DATA_DIR / "matrices"
    matrices = {}
    for fpath in sorted(matrices_dir.glob("*.parquet")):
        matrices[fpath.stem] = pd.read_parquet(fpath)
    log(f"  {len(matrices)} matrices loaded")

    log("Checking kline files...")
    kline_info = check_kline_files(args.interval)
    log(f"  {len(kline_info)} kline files checked")

    # ── Run all checks ──
    all_checks = []
    
    # Analyze matrices
    log("Analyzing matrices...")
    analyses = {}
    for name, df in sorted(matrices.items()):
        analyses[name] = analyze_matrix(name, df)
    
    # Check 1: Kline completeness
    log("Check 1: Kline completeness...")
    n_klines = len(kline_info)
    n_full = sum(1 for v in kline_info.values() if v.get("coverage_ratio", 0) > 0.95 and v.get("n_bars", 0) > 100)
    kline_pct = 100.0 * n_full / max(n_klines, 1)
    if kline_pct > 90:
        all_checks.append(("PASS", "Kline coverage (>95% bars present)", 
                          f"{n_full}/{n_klines} files complete ({kline_pct:.0f}%)"))
    elif kline_pct > 50:
        all_checks.append(("WARN", "Kline coverage (>95% bars present)", 
                          f"{n_full}/{n_klines} files complete ({kline_pct:.0f}%)"))
    else:
        all_checks.append(("FAIL", "Kline coverage (>95% bars present)", 
                          f"{n_full}/{n_klines} files complete ({kline_pct:.0f}%)"))
    
    # Check 2: Active-range coverage (smart NaN check for panel data)
    # In a panel dataset with staggered crypto listings, high overall NaN% is expected.
    # The real check: within each symbol's active range, is data complete?
    log("Check 2: Active-range coverage...")
    close_a = analyses.get("close", {})
    close_nan = close_a.get("nan_pct", 100)
    close_mat = matrices.get("close")
    if close_mat is not None:
        # For each symbol, compute coverage within its active range (first valid -> last valid)
        active_coverages = []
        for col in close_mat.columns:
            ser = close_mat[col].dropna()
            if len(ser) < 2:
                continue
            first_valid = ser.index[0]
            last_valid = ser.index[-1]
            active_range = close_mat.loc[first_valid:last_valid, col]
            active_cov = active_range.notna().mean()
            active_coverages.append(active_cov)
        
        if active_coverages:
            mean_active_cov = np.mean(active_coverages)
            min_active_cov = np.min(active_coverages)
            poor_symbols = sum(1 for c in active_coverages if c < 0.95)
            
            if mean_active_cov > 0.98 and poor_symbols == 0:
                all_checks.append(("PASS", "Active-range coverage", 
                                  f"Mean {mean_active_cov:.1%} (raw NaN%={close_nan:.1f}% due to staggered listings)"))
            elif mean_active_cov > 0.90:
                all_checks.append(("WARN", "Active-range coverage", 
                                  f"Mean {mean_active_cov:.1%}, {poor_symbols} symbols <95%"))
            else:
                all_checks.append(("FAIL", "Active-range coverage", 
                                  f"Mean {mean_active_cov:.1%}, {poor_symbols} symbols <95% — gaps in active data"))
        else:
            all_checks.append(("FAIL", "Active-range coverage", "No symbols have valid data"))
    else:
        all_checks.append(("FAIL", "Active-range coverage", "Close matrix not found"))
    
    # Check 3: No Inf values in any matrix
    log("Check 3: Inf values...")
    total_inf = sum(a.get("inf_count", 0) for a in analyses.values())
    if total_inf == 0:
        all_checks.append(("PASS", "No Inf values in any matrix", "0 Inf values"))
    else:
        all_checks.append(("FAIL", "No Inf values in any matrix", f"{total_inf} Inf values found"))
    
    # Check 4: No negative prices
    log("Check 4: Negative prices...")
    neg_prices = sum(a.get("negative_prices", 0) for a in analyses.values())
    if neg_prices == 0:
        all_checks.append(("PASS", "No negative prices", "0 negative prices"))
    else:
        all_checks.append(("FAIL", "No negative prices", f"{neg_prices} negative price values"))
    
    # Check 5: No zero close prices
    log("Check 5: Zero prices...")
    zero_close = analyses.get("close", {}).get("zero_prices", 0)
    if zero_close == 0:
        all_checks.append(("PASS", "No zero close prices", "0 zero close values"))
    else:
        all_checks.append(("FAIL", "No zero close prices", f"{zero_close} zero close values"))
    
    # Check 6: No negative volumes
    log("Check 6: Negative volumes...")
    neg_vols = sum(a.get("negative_volumes", 0) for a in analyses.values())
    if neg_vols == 0:
        all_checks.append(("PASS", "No negative volumes", "0 negative volumes"))
    else:
        all_checks.append(("FAIL", "No negative volumes", f"{neg_vols} negative volumes"))
    
    # Check 7: Cross-matrix consistency
    log("Check 7: Cross-matrix consistency...")
    cross_checks = check_cross_consistency(matrices)
    all_checks.extend(cross_checks)
    
    # Check 8: Gaps and duplicates
    log("Check 8: Gaps and duplicates...")
    gap_info = check_kline_gaps_and_dupes(args.interval)
    if gap_info["total_dupes"] == 0:
        all_checks.append(("PASS", "No duplicate bars", "0 duplicates"))
    else:
        all_checks.append(("WARN", "Duplicate bars", 
                          f"{gap_info['total_dupes']} dupes in {gap_info['symbols_with_dupes']} symbols"))
    
    if gap_info["symbols_with_gaps"] == 0:
        all_checks.append(("PASS", f"No large gaps (>{GAP_MULTIPLE}× interval)", "0 gaps"))
    elif gap_info["symbols_with_gaps"] < len(kline_info) * 0.1:
        all_checks.append(("WARN", f"Large gaps (>{GAP_MULTIPLE}× interval)", 
                          f"{gap_info['total_gaps']} gaps in {gap_info['symbols_with_gaps']} symbols"))
    else:
        all_checks.append(("FAIL", f"Large gaps (>{GAP_MULTIPLE}× interval)", 
                          f"{gap_info['total_gaps']} gaps in {gap_info['symbols_with_gaps']} symbols"))
    
    # Check 9: Freshness
    log("Check 9: Freshness...")
    freshness_checks = check_freshness(kline_info, args.interval)
    all_checks.extend(freshness_checks)
    
    # Check 10: Universe coverage
    log("Check 10: Universe coverage...")
    universe_checks = check_universe_coverage(args.interval)
    all_checks.extend(universe_checks)
    
    # Check 11: Matrix shape consistency
    log("Check 11: Matrix shapes...")
    shapes = set(a["shape"] for a in analyses.values())
    base_matrices = [n for n in ["close", "open", "high", "low", "volume", "returns"] if n in analyses]
    base_shapes = set(analyses[n]["shape"] for n in base_matrices)
    if len(base_shapes) == 1:
        all_checks.append(("PASS", "Core matrix shapes consistent", 
                          f"All core matrices: {list(base_shapes)[0]}"))
    else:
        all_checks.append(("FAIL", "Core matrix shapes consistent", 
                          f"Multiple shapes: {base_shapes}"))
    
    # ── Generate report ──
    log("Generating HTML report...")
    html = generate_html_report(matrices, kline_info, args.interval, 
                               all_checks, analyses, gap_info)

    output_path = Path(args.output)
    output_path.write_text(html, encoding="utf-8")
    log(f"  Report saved to {output_path}")

    # ── Console summary ──
    print("\n" + "=" * 60, flush=True)
    n_pass = sum(1 for s, _, _ in all_checks if s == "PASS")
    n_warn = sum(1 for s, _, _ in all_checks if s == "WARN")
    n_fail = sum(1 for s, _, _ in all_checks if s == "FAIL")
    
    for status, check_name, detail in all_checks:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(status, "?")
        print(f"  {icon} {check_name}: {detail}", flush=True)
    
    print("\n" + "-" * 60, flush=True)
    if n_fail > 0:
        print(f"  RESULT: {n_fail} FAILURES, {n_warn} warnings, {n_pass} passed", flush=True)
        print("=" * 60, flush=True)
        sys.exit(1)
    elif n_warn > 0:
        print(f"  RESULT: ALL PASSED ({n_warn} warnings), {n_pass} clean", flush=True)
        print("=" * 60, flush=True)
        sys.exit(0)
    else:
        print(f"  RESULT: ALL GREEN ✅ — {n_pass} checks passed", flush=True)
        print("=" * 60, flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
