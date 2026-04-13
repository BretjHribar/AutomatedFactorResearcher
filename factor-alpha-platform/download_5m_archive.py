"""
Download 5m klines from Binance public data archive (data.binance.vision).

The fapi (REST API) is geo-restricted, but the public data archive works.
Downloads daily ZIP files containing 5m klines for all USDT perpetual futures.

Usage:
    python download_5m_archive.py [--start 2026-02-01] [--end 2026-03-27]
    python download_5m_archive.py --rebuild-only  # Skip download, rebuild matrices
"""

import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Config ──
DATA_DIR = Path("data/binance_cache")
KLINES_DIR = DATA_DIR / "klines" / "5m"
MATRICES_DIR = DATA_DIR / "matrices" / "5m"
UNIVERSE_DIR = DATA_DIR / "universes"
ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

MAX_WORKERS = 8       # Concurrent downloads
PAUSE_SECONDS = 0.05  # Small pause between downloads

# Get symbols from existing 4h klines (already downloaded)
KLINES_4H_DIR = DATA_DIR / "klines" / "4h"


def log(msg: str):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_symbols_from_4h():
    """Get list of symbols from existing 4h kline data."""
    if not KLINES_4H_DIR.exists():
        raise FileNotFoundError(f"No 4h klines found at {KLINES_4H_DIR}")
    symbols = sorted([f.stem for f in KLINES_4H_DIR.glob("*.parquet")])
    log(f"Found {len(symbols)} symbols from 4h klines")
    return symbols


def download_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    """Download one day of 5m klines for one symbol from the archive."""
    url = f"{ARCHIVE_BASE}/{symbol}/5m/{symbol}-5m-{date_str}.zip"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None  # No data for this date (not listed yet or delisted)
        resp.raise_for_status()
    except Exception:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Check if first line is a header
                first_line = f.readline().decode().strip()
                f.seek(0)
                
                col_names = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades_count",
                    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
                ]
                
                if first_line.startswith("open_time"):
                    # Has header — read with header=0, then rename
                    df = pd.read_csv(f, header=0)
                    df.columns = col_names[:len(df.columns)]
                else:
                    # No header
                    df = pd.read_csv(f, header=None, names=col_names)
    except Exception as e:
        return None

    if df.empty:
        return None

    # Convert types
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df.dropna(subset=["open_time"])
    df["datetime"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")
    for col in ["open", "high", "low", "close", "volume",
                 "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype(int)

    df = df[["datetime", "open", "high", "low", "close", "volume",
             "quote_volume", "trades_count", "taker_buy_volume",
             "taker_buy_quote_volume"]].copy()
    return df


def download_symbol(symbol: str, dates: list[str]) -> int:
    """Download all dates for one symbol, append to parquet."""
    fpath = KLINES_DIR / f"{symbol}.parquet"

    # Load existing data if present
    existing = pd.DataFrame()
    existing_dates = set()
    if fpath.exists():
        try:
            existing = pd.read_parquet(fpath)
            existing_dates = set(existing["datetime"].dt.strftime("%Y-%m-%d").unique())
        except Exception:
            pass

    # Filter to dates we don't have yet
    dates_needed = [d for d in dates if d not in existing_dates]
    if not dates_needed:
        return 0

    all_dfs = []
    for date_str in dates_needed:
        df = download_day(symbol, date_str)
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(PAUSE_SECONDS)

    if not all_dfs:
        return 0

    new_data = pd.concat(all_dfs, ignore_index=True)

    if not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    else:
        combined = new_data.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    combined.to_parquet(fpath, index=False)
    return len(new_data)


def generate_dates(start_date: str, end_date: str) -> list[str]:
    """Generate list of date strings between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def build_matrices():
    """Build derived field matrices from 5m klines — same logic as ingest_binance.py."""
    log("Building 5m matrices...")
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for fpath in sorted(KLINES_DIR.glob("*.parquet")):
        sym = fpath.stem
        df = pd.read_parquet(fpath)
        if df.empty:
            continue
        df = df.set_index("datetime")
        all_data[sym] = df

    if not all_data:
        log("  No data found!")
        return

    log(f"  {len(all_data)} symbols loaded")

    # Build each field
    fields = ["open", "high", "low", "close", "volume", "quote_volume",
              "trades_count", "taker_buy_volume", "taker_buy_quote_volume"]

    for field in fields:
        series_dict = {}
        for sym, df in all_data.items():
            if field in df.columns:
                series_dict[sym] = df[field]
        if series_dict:
            mat = pd.DataFrame(series_dict)
            mat.index.name = "date"
            mat.to_parquet(MATRICES_DIR / f"{field}.parquet")
            log(f"    {field}: {mat.shape}")

    # ── Derived fields ──
    close = pd.DataFrame({s: d["close"] for s, d in all_data.items()})
    open_ = pd.DataFrame({s: d["open"] for s, d in all_data.items()})
    high = pd.DataFrame({s: d["high"] for s, d in all_data.items()})
    low = pd.DataFrame({s: d["low"] for s, d in all_data.items()})
    volume = pd.DataFrame({s: d["volume"] for s, d in all_data.items()})
    qv = pd.DataFrame({s: d["quote_volume"] for s, d in all_data.items()})
    tbv = pd.DataFrame({s: d.get("taker_buy_volume", pd.Series(dtype=float)) for s, d in all_data.items()})

    # Returns
    returns = close.pct_change()
    returns.to_parquet(MATRICES_DIR / "returns.parquet")
    log(f"    returns: {returns.shape}")

    log_returns = np.log(close / close.shift(1))
    log_returns.to_parquet(MATRICES_DIR / "log_returns.parquet")

    # VWAP proxy
    safe_vol = volume.replace(0, np.nan)
    vwap = qv / safe_vol
    vwap.to_parquet(MATRICES_DIR / "vwap.parquet")

    vwap_dev = (close - vwap) / vwap
    vwap_dev.to_parquet(MATRICES_DIR / "vwap_deviation.parquet")

    # Taker buy ratio
    tbr = tbv / safe_vol
    tbr.to_parquet(MATRICES_DIR / "taker_buy_ratio.parquet")

    # ADV
    for window in [20, 60]:
        adv = qv.rolling(window, min_periods=max(window // 2, 1)).mean()
        adv.to_parquet(MATRICES_DIR / f"adv{window}.parquet")
        log(f"    adv{window}: {adv.shape}")

    # Volume ratio
    adv20 = qv.rolling(20, min_periods=10).mean()
    vol_ratio = qv / adv20
    vol_ratio.to_parquet(MATRICES_DIR / "volume_ratio_20d.parquet")

    # Volatility
    for window in [10, 20, 60, 120]:
        hvol = returns.rolling(window, min_periods=max(window // 2, 1)).std() * np.sqrt(252 * 288)
        hvol.to_parquet(MATRICES_DIR / f"historical_volatility_{window}.parquet")

    # Parkinson vol
    hl_ratio = np.log(high / low)
    for window in [10, 20, 60]:
        pvol = hl_ratio.rolling(window, min_periods=max(window // 2, 1)).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))) * np.sqrt(252 * 288),
            raw=True,
        )
        pvol.to_parquet(MATRICES_DIR / f"parkinson_volatility_{window}.parquet")

    # High-low range
    hlr = (high - low) / close
    hlr.to_parquet(MATRICES_DIR / "high_low_range.parquet")

    # Open-close range
    ocr = (close - open_).abs() / close
    ocr.to_parquet(MATRICES_DIR / "open_close_range.parquet")

    # Momentum (in bars, not days — 5d=1440, 20d=5760, 60d=17280)
    for window_d, label in [(5, "5d"), (20, "20d"), (60, "60d")]:
        bars = window_d * 288
        mom = close / close.shift(bars) - 1
        mom.to_parquet(MATRICES_DIR / f"momentum_{label}.parquet")

    # Overnight gap (open / prev_close - 1)
    gap = open_ / close.shift(1) - 1
    gap.to_parquet(MATRICES_DIR / "overnight_gap.parquet")

    # Trades per volume
    tc = pd.DataFrame({s: d["trades_count"] for s, d in all_data.items()})
    tpv = tc / safe_vol
    tpv.to_parquet(MATRICES_DIR / "trades_per_volume.parquet")

    # Upper/lower shadow
    max_oc = pd.DataFrame(np.maximum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    min_oc = pd.DataFrame(np.minimum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    hl_range = high - low
    safe_hl = hl_range.replace(0, np.nan)
    upper_shadow = (high - max_oc) / safe_hl
    lower_shadow = (min_oc - low) / safe_hl
    upper_shadow.to_parquet(MATRICES_DIR / "upper_shadow.parquet")
    lower_shadow.to_parquet(MATRICES_DIR / "lower_shadow.parquet")

    # Close position in range
    cpr = (close - low) / safe_hl
    cpr.to_parquet(MATRICES_DIR / "close_position_in_range.parquet")

    # Dollar volume
    qv.to_parquet(MATRICES_DIR / "dollars_traded.parquet")

    # Volume momentum
    dv_mom = qv / qv.shift(1).replace(0, np.nan)
    dv_mom.to_parquet(MATRICES_DIR / "volume_momentum_1.parquet")

    dv_mom5 = qv.rolling(5, min_periods=2).mean() / qv.rolling(20, min_periods=5).mean()
    dv_mom5.to_parquet(MATRICES_DIR / "volume_momentum_5_20.parquet")

    # Taker buy quote volume
    tbqv = pd.DataFrame({s: d.get("taker_buy_quote_volume", pd.Series(dtype=float))
                          for s, d in all_data.items()})
    if not tbqv.empty:
        tbqv.to_parquet(MATRICES_DIR / "taker_buy_quote_volume.parquet")

    # BTC beta
    if "BTCUSDT" in returns.columns:
        btc_ret = returns["BTCUSDT"]
        btc_var = btc_ret.rolling(288, min_periods=144).var()  # 1-day window
        betas = {}
        for col in returns.columns:
            if col == "BTCUSDT":
                betas[col] = pd.Series(1.0, index=returns.index)
                continue
            cov = returns[col].rolling(288, min_periods=144).cov(btc_ret)
            betas[col] = cov / btc_var
        beta_df = pd.DataFrame(betas)
        beta_df.to_parquet(MATRICES_DIR / "beta_to_btc.parquet")
        log(f"    beta_to_btc: {beta_df.shape}")

    log("  [OK] All 5m matrices built")


def build_universes():
    """Build TOP100 universe from 5m adv20 data.
    
    Uses shorter rebalance period (1440 bars = 5 days) since our data
    window is only ~54 days. No listing age restriction for 5m since
    we only have recent data.
    """
    log("Building 5m universes...")
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    adv_path = MATRICES_DIR / "adv20.parquet"
    if not adv_path.exists():
        log("  ⚠ adv20 not found!")
        return

    adv20 = pd.read_parquet(adv_path)
    all_dates = adv20.index

    # Rebalance every 1440 bars (~5 days) for better coverage with short data
    rebal_period = 1440
    # Start from bar 288 (1 day) to give ADV20 some warmup, but not too much
    start_bar = min(288, len(all_dates) - 1)
    rebal_indices = list(range(start_bar, len(all_dates), rebal_period))
    rebal_dates = all_dates[rebal_indices]
    log(f"  {len(rebal_dates)} rebalance dates (every {rebal_period} bars, start bar {start_bar})")

    for tier_name, tier_size in [("TOP100", 100), ("TOP50", 50), ("TOP20", 20)]:
        mask = pd.DataFrame(False, index=all_dates, columns=adv20.columns)

        for i, reb_date in enumerate(rebal_dates):
            adv_row = adv20.loc[reb_date].dropna()
            adv_row = adv_row[adv_row > 0]
            top_n = adv_row.nlargest(min(tier_size, len(adv_row))).index.tolist()

            # Window extends backward to cover gap before first rebalance
            if i == 0:
                period = all_dates[all_dates < (rebal_dates[1] if len(rebal_dates) > 1 else all_dates[-1])]
            elif i + 1 < len(rebal_dates):
                end_date = rebal_dates[i + 1]
                period = all_dates[(all_dates >= reb_date) & (all_dates < end_date)]
            else:
                period = all_dates[all_dates >= reb_date]

            mask.loc[period, top_n] = True

        out_path = UNIVERSE_DIR / f"BINANCE_{tier_name}_5m.parquet"
        mask.to_parquet(out_path)
        avg_count = mask.sum(axis=1).mean()
        log(f"  {tier_name}: avg {avg_count:.0f} symbols per bar -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Download 5m Binance klines from public archive")
    parser.add_argument("--start", default="2026-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-27", help="End date (YYYY-MM-DD)")
    parser.add_argument("--rebuild-only", action="store_true", help="Skip download, rebuild matrices/universes")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols to download (0=all)")
    args = parser.parse_args()

    KLINES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_only:
        # Get symbols from existing 4h data
        symbols = get_symbols_from_4h()

        if args.max_symbols > 0:
            symbols = symbols[:args.max_symbols]

        dates = generate_dates(args.start, args.end)

        print("=" * 60)
        print("5m KLINES DOWNLOAD (data.binance.vision)")
        print(f"Symbols: {len(symbols)}")
        print(f"Dates: {args.start} to {args.end} ({len(dates)} days)")
        print(f"Expected: ~{len(symbols) * len(dates)} symbol-day downloads")
        print("=" * 60)

        completed = 0
        total_bars = 0

        # Download with thread pool (per-symbol)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_symbol, sym, dates): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    bars = future.result()
                    total_bars += bars
                except Exception as e:
                    log(f"  [!] {sym}: {e}")
                completed += 1
                if completed % 10 == 0 or completed == len(symbols):
                    log(f"  {completed}/{len(symbols)} symbols done ({total_bars:,} new bars)")

        log(f"  Download complete: {completed} symbols, {total_bars:,} total new bars")

    # Build matrices
    build_matrices()

    # Build universes
    build_universes()

    print("\n" + "=" * 60)
    print("[OK] 5m DATA PIPELINE COMPLETE")
    print(f"  Klines: {KLINES_DIR}")
    print(f"  Matrices: {MATRICES_DIR}")
    print(f"  Universes: {UNIVERSE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
