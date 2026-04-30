"""
Rebuild equity matrices with PIT (point-in-time) correctness.

Key fix: forward-fill fundamentals from filingDate (when SEC filing was made
public), not from period-end date. Original fmp_loader bug introduced ~30-90
days of look-ahead bias on every fundamental.

Outputs to data/fmp_cache/matrices_pit/ — drop-in replacement for matrices_clean.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CACHE_DIR = PROJECT_ROOT / "data/fmp_cache"
PRICES_DIR = CACHE_DIR / "prices"
INCOME_DIR = CACHE_DIR / "income"
BALANCE_DIR = CACHE_DIR / "balance"
CASHFLOW_DIR = CACHE_DIR / "cashflow"
METRICS_DIR = CACHE_DIR / "metrics"
OUT_DIR = CACHE_DIR / "matrices_pit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = CACHE_DIR / "universes" / "TOP3000.parquet"   # superset
PIT_LAG_DAYS = 1   # next-day-tradeable convention


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRICE MATRICES (source of truth — fresh through 2026-04-20)
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(tickers):
    """Load OHLCV from per-ticker parquets in prices/. Returns dict of (T,N) DataFrames."""
    print(f"\n[1/3] Loading prices for {len(tickers)} tickers...", flush=True)
    t0 = time.time()
    fields = ["open", "high", "low", "close", "volume", "vwap"]
    series_per_field = {f: {} for f in fields}
    bad = 0
    for sym in tickers:
        fp = PRICES_DIR / f"{sym}.parquet"
        if not fp.exists():
            bad += 1
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            bad += 1; continue
        if df.empty:
            bad += 1; continue
        for f in fields:
            if f in df.columns:
                series_per_field[f][sym] = df[f]

    # Stitch into matrices
    all_dates = sorted(set().union(*(s.index for s in series_per_field["close"].values())))
    if not all_dates:
        raise RuntimeError("No price data found")
    all_dates = pd.DatetimeIndex(all_dates)
    actual_tickers = sorted(series_per_field["close"].keys())

    matrices = {}
    for f in fields:
        mat = pd.DataFrame(index=all_dates, columns=actual_tickers, dtype=float)
        for sym, ser in series_per_field[f].items():
            mat[sym] = ser.reindex(all_dates)
        matrices[f] = mat
    print(f"  prices: T={len(all_dates)} N={len(actual_tickers)} ({time.time()-t0:.1f}s)", flush=True)
    return matrices, actual_tickers, all_dates


def derive_timeseries_chars(matrices, dates):
    """Compute log_returns, vol, adv etc from prices. Independent of fundamentals."""
    print(f"\n[2/3] Deriving time-series chars from prices...", flush=True)
    t0 = time.time()
    close = matrices["close"]
    high = matrices["high"]
    low = matrices["low"]
    volume = matrices["volume"]

    out = dict(matrices)
    out["returns"] = close.pct_change(fill_method=None)
    out["log_returns"] = np.log(close / close.shift(1))
    out["dollars_traded"] = close * volume
    out["adv20"] = out["dollars_traded"].rolling(20, min_periods=10).mean()
    out["adv60"] = out["dollars_traded"].rolling(60, min_periods=20).mean()
    out["high_low_range"] = (high - low) / close
    out["open_close_range"] = (close - matrices["open"]) / matrices["open"]

    # Realized vol (annualized fraction)
    log_ret = out["log_returns"]
    for w in [10, 20, 30, 60, 90, 120]:
        out[f"historical_volatility_{w}"] = log_ret.rolling(w, min_periods=max(5, w//4)).std() * np.sqrt(252)

    # Parkinson vol
    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    pk = (log_hl ** 2) / (4 * np.log(2))
    for w in [10, 20, 30, 60, 90, 120]:
        out[f"parkinson_volatility_{w}"] = np.sqrt(pk.rolling(w, min_periods=max(5, w//4)).mean()) * np.sqrt(252)

    # Momentum
    out["momentum_5d"] = close.pct_change(5, fill_method=None)
    out["momentum_20d"] = close.pct_change(20, fill_method=None)
    out["momentum_60d"] = close.pct_change(60, fill_method=None)
    out["momentum_120d"] = close.pct_change(120, fill_method=None)
    out["momentum_252d"] = close.pct_change(252, fill_method=None)

    print(f"  derived {sum(1 for k in out if k not in ['open','high','low','close','volume','vwap'])} time-series chars  ({time.time()-t0:.1f}s)", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. PIT FUNDAMENTALS (forward-fill from filingDate, not period-end date)
# ─────────────────────────────────────────────────────────────────────────────

def build_pit_field(stmt_dir: Path, fmp_field: str, tickers: list[str],
                    dates: pd.DatetimeIndex, lag_days: int = PIT_LAG_DAYS) -> pd.DataFrame:
    """Build a (T,N) matrix where mat[t,sym] = latest known value of fmp_field for sym
    where 'known' = filingDate + lag_days <= t."""
    mat = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    miss_field = miss_filing = 0
    for sym in tickers:
        fp = stmt_dir / f"{sym}.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if fmp_field not in df.columns or "filingDate" not in df.columns:
            miss_field += 1; continue
        eff = pd.to_datetime(df["filingDate"], errors="coerce") + pd.Timedelta(days=lag_days)
        valid = eff.notna()
        if valid.sum() == 0:
            miss_filing += 1; continue
        s = pd.Series(df[fmp_field].values[valid], index=eff[valid]).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        # ffill onto trading dates
        mat[sym] = s.reindex(dates, method="ffill")
    return mat


def build_pit_fundamentals(tickers, dates):
    """Build all PIT fundamental matrices needed for ratios."""
    print(f"\n[3/3] Building PIT fundamentals from raw filings...", flush=True)
    print(f"  Using filingDate + {PIT_LAG_DAYS}d lag (next-day-tradeable)")
    t0 = time.time()

    # (statement_dir, fmp_field, our_name)
    fields = [
        # Income statement
        (INCOME_DIR, "revenue", "revenue"),
        (INCOME_DIR, "grossProfit", "gross_profit"),
        (INCOME_DIR, "operatingIncome", "operating_income"),
        (INCOME_DIR, "netIncome", "net_income"),
        (INCOME_DIR, "ebitda", "ebitda"),
        (INCOME_DIR, "ebit", "ebit"),
        (INCOME_DIR, "eps", "eps"),
        (INCOME_DIR, "epsDiluted", "eps_diluted"),
        (INCOME_DIR, "weightedAverageShsOut", "shares_out"),
        (INCOME_DIR, "weightedAverageShsOutDil", "shares_out_diluted"),
        (INCOME_DIR, "interestExpense", "interest_expense"),
        (INCOME_DIR, "depreciationAndAmortization", "depreciation_amortization"),
        # Balance sheet
        (BALANCE_DIR, "totalAssets", "total_assets"),
        (BALANCE_DIR, "totalEquity", "total_equity"),
        (BALANCE_DIR, "totalDebt", "total_debt"),
        (BALANCE_DIR, "netDebt", "net_debt"),
        (BALANCE_DIR, "totalCurrentAssets", "total_current_assets"),
        (BALANCE_DIR, "totalCurrentLiabilities", "total_current_liabilities"),
        (BALANCE_DIR, "cashAndCashEquivalents", "cash"),
        (BALANCE_DIR, "inventory", "inventory"),
        (BALANCE_DIR, "totalLiabilities", "total_liabilities"),
        (BALANCE_DIR, "goodwill", "goodwill"),
        (BALANCE_DIR, "intangibleAssets", "intangibles"),
        (BALANCE_DIR, "totalStockholdersEquity", "total_stockholders_equity"),
        (BALANCE_DIR, "longTermDebt", "long_term_debt"),
        (BALANCE_DIR, "shortTermDebt", "short_term_debt"),
        # Cashflow
        (CASHFLOW_DIR, "operatingCashFlow", "operating_cashflow"),
        (CASHFLOW_DIR, "capitalExpenditure", "capex"),
        (CASHFLOW_DIR, "freeCashFlow", "free_cashflow"),
        (CASHFLOW_DIR, "commonDividendsPaid", "dividends_paid"),
    ]

    out = {}
    for stmt_dir, fmp_field, our_name in fields:
        ts = time.time()
        mat = build_pit_field(stmt_dir, fmp_field, tickers, dates)
        out[our_name] = mat
        coverage = mat.notna().sum().sum() / (mat.shape[0] * mat.shape[1]) * 100
        print(f"  {our_name:<32} cov={coverage:5.1f}%  ({time.time()-ts:.1f}s)", flush=True)
    print(f"  built {len(out)} PIT fundamental fields  ({(time.time()-t0)/60:.1f}min)", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. DERIVED RATIOS (computed from PIT fundamentals + daily prices)
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(a, b):
    """Element-wise division producing NaN on zero/inf."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.DataFrame) else out


def derive_ratios(prices_chars, fund):
    """Compute all derived ratios from PIT fundamentals + daily prices."""
    print(f"\n[4/4] Deriving ratios from PIT fundamentals + daily prices...", flush=True)
    t0 = time.time()
    out = {}
    close = prices_chars["close"]

    # Market cap = close * shares_out (PIT shares_out updates on each filing)
    cap = close * fund["shares_out"]
    out["cap"] = cap
    out["market_cap"] = cap

    # Enterprise value = cap + total_debt - cash (PIT)
    ev = cap.add(fund["total_debt"], fill_value=np.nan).sub(fund["cash"], fill_value=np.nan)
    out["enterprise_value"] = ev

    # Yield ratios (use close directly → daily updates of denominator)
    out["earnings_yield"] = safe_div(fund["eps"], close)
    out["book_to_market"] = safe_div(fund["total_equity"], cap)
    out["free_cashflow_yield"] = safe_div(fund["free_cashflow"], cap)
    out["fcf_per_share"] = safe_div(fund["free_cashflow"], fund["shares_out"])
    out["ev_to_ebitda"] = safe_div(ev, fund["ebitda"])
    out["ev_to_revenue"] = safe_div(ev, fund["revenue"])

    # Profitability
    out["roe"] = safe_div(fund["net_income"], fund["total_equity"])
    out["roa"] = safe_div(fund["net_income"], fund["total_assets"])
    out["return_equity"] = out["roe"]
    out["return_assets"] = out["roa"]
    out["gross_margin"] = safe_div(fund["gross_profit"], fund["revenue"])
    out["operating_margin"] = safe_div(fund["operating_income"], fund["revenue"])
    out["net_margin"] = safe_div(fund["net_income"], fund["revenue"])
    out["asset_turnover"] = safe_div(fund["revenue"], fund["total_assets"])

    # Leverage
    out["debt_to_equity"] = safe_div(fund["total_debt"], fund["total_equity"])
    out["debt_to_assets"] = safe_div(fund["total_debt"], fund["total_assets"])
    out["current_ratio"] = safe_div(fund["total_current_assets"], fund["total_current_liabilities"])

    # Tangible book
    tangible_equity = fund["total_equity"].sub(fund["goodwill"], fill_value=0).sub(fund["intangibles"], fill_value=0)
    out["tangible_book_per_share"] = safe_div(tangible_equity, fund["shares_out"])
    out["bookvalue_ps"] = safe_div(fund["total_equity"], fund["shares_out"])

    # Other useful
    out["turnover"] = safe_div(prices_chars["dollars_traded"], cap)   # daily turnover ratio
    out["volume_ratio_20d"] = safe_div(prices_chars["volume"], prices_chars["volume"].rolling(20, min_periods=10).mean())
    out["volume_momentum_1"] = prices_chars["volume"].pct_change(fill_method=None)
    out["volume_momentum_5_20"] = safe_div(prices_chars["volume"].rolling(5, min_periods=2).mean(),
                                           prices_chars["volume"].rolling(20, min_periods=10).mean()) - 1.0

    # Beta to SPY proxy (= equal-weighted basket here; cheap)
    market_ret = prices_chars["returns"].mean(axis=1)
    # Rolling 60-bar regression: beta_i = cov(r_i, r_m) / var(r_m)
    # Cheap approximation: rolling correlation × ratio of stds
    rs = prices_chars["returns"]
    cov_mt = rs.rolling(60, min_periods=20).corr(market_ret) * rs.rolling(60, min_periods=20).std()
    market_std = market_ret.rolling(60, min_periods=20).std()
    out["beta_to_btc"] = cov_mt.div(market_std, axis=0)   # name kept for backward compat

    # vwap_deviation
    out["vwap_deviation"] = safe_div(close - prices_chars["vwap"], close)

    # close_position_in_range
    out["close_position_in_range"] = safe_div(close - prices_chars["low"],
                                              prices_chars["high"] - prices_chars["low"])

    print(f"  derived {len(out)} ratio fields  ({time.time()-t0:.1f}s)", flush=True)
    return out


def main():
    overall_t0 = time.time()
    print("=" * 90)
    print("PIT EQUITY MATRIX REBUILD")
    print("=" * 90)

    # Universe: pick TOP3000 (superset for backtest universes)
    if not UNIVERSE_PATH.exists():
        # fall back to sniff: just take all tickers for which we have an income filing
        tickers = sorted({fp.stem for fp in INCOME_DIR.glob("*.parquet")})
        print(f"  No universe parquet — using all {len(tickers)} tickers with income filings")
    else:
        uni = pd.read_parquet(UNIVERSE_PATH)
        all_uni = sorted(uni.columns.tolist())
        # Intersect with tickers that have income filings
        income_have = {fp.stem for fp in INCOME_DIR.glob("*.parquet")}
        tickers = sorted(set(all_uni) & income_have)
        print(f"  TOP3000 ∩ income cache = {len(tickers)} tickers")

    # 1. Prices
    prices, tickers_with_prices, dates = load_prices(tickers)
    tickers = tickers_with_prices

    # 2. Time-series derivations
    prices_chars = derive_timeseries_chars(prices, dates)

    # 3. PIT fundamentals
    fund = build_pit_fundamentals(tickers, dates)

    # 4. Ratios
    ratios = derive_ratios(prices_chars, fund)

    # 5. Save everything
    print(f"\n[save] Writing matrices to {OUT_DIR}/", flush=True)
    t0 = time.time()
    all_out = {**prices_chars, **fund, **ratios}
    for name, df in all_out.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet")
    print(f"  wrote {len(all_out)} parquets  ({time.time()-t0:.1f}s)", flush=True)

    # Manifest
    manifest = {
        "build_timestamp": datetime.now().isoformat(),
        "n_tickers": len(tickers),
        "n_dates": len(dates),
        "date_range": [str(dates.min()), str(dates.max())],
        "pit_lag_days": PIT_LAG_DAYS,
        "fields": sorted(all_out.keys()),
        "elapsed_min": round((time.time() - overall_t0) / 60, 2),
    }
    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nManifest: {OUT_DIR / 'manifest.json'}")
    print(f"DONE in {(time.time()-overall_t0)/60:.1f} min  →  {OUT_DIR}/")


if __name__ == "__main__":
    main()
