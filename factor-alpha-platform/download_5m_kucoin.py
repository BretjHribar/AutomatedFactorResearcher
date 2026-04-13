"""
Download 5m klines from KuCoin Futures public REST API.

Fetches all USDT-margined perpetual contracts, downloads 5m candles,
builds matrices and universes in the same format as the Binance pipeline.

Usage:
    python download_5m_kucoin.py [--start 2026-02-01] [--end 2026-03-27]
    python download_5m_kucoin.py --rebuild-only  # Skip download, rebuild matrices
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Config ──
DATA_DIR = Path("data/kucoin_cache")
KLINES_DIR = DATA_DIR / "klines" / "5m"
MATRICES_DIR = DATA_DIR / "matrices" / "5m"
UNIVERSE_DIR = DATA_DIR / "universes"

KUCOIN_BASE = "https://api-futures.kucoin.com"
SYMBOLS_URL = f"{KUCOIN_BASE}/api/v1/contracts/active"
KLINES_URL = f"{KUCOIN_BASE}/api/v1/kline/query"

MAX_PER_REQUEST = 500  # KuCoin returns max 500 data points per call
PAUSE_SECONDS = 0.12   # Rate limit per request
MAX_WORKERS = 4        # Concurrent download threads


def log(msg: str):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_kucoin_symbols() -> list[dict]:
    """Get all active USDT-margined perpetual contracts from KuCoin Futures."""
    resp = requests.get(SYMBOLS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    if data.get("code") != "200000":
        raise ValueError(f"KuCoin API error: {data}")
    
    contracts = data["data"]
    # Filter to USDT-margined perpetual contracts only
    usdt_perps = [
        c for c in contracts
        if c.get("quoteCurrency") == "USDT"
        and c.get("isInverse") == False
        and c.get("status") == "Open"
    ]
    
    log(f"Found {len(usdt_perps)} active USDT perpetual contracts on KuCoin")
    return usdt_perps


def kucoin_symbol_to_binance(kucoin_sym: str) -> str:
    """Convert KuCoin symbol format (XBTUSDTM) to Binance format (BTCUSDT).
    
    KuCoin uses XBT for Bitcoin, and appends M for margin.
    """
    # Remove trailing M
    sym = kucoin_sym.rstrip("M")
    # Replace XBT with BTC
    sym = sym.replace("XBT", "BTC")
    return sym


def download_klines(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Download all 5m klines for a symbol between start and end timestamps.
    
    Handles pagination (max 500 per request) by walking forward in time.
    """
    all_rows = []
    current_start = start_ts
    
    while current_start < end_ts:
        # KuCoin expects timestamps in milliseconds
        params = {
            "symbol": symbol,
            "granularity": 5,  # 5 minutes (in minutes, not string)
            "from": current_start * 1000,
            "to": min(current_start + MAX_PER_REQUEST * 5 * 60, end_ts) * 1000,
        }
        
        try:
            resp = requests.get(KLINES_URL, params=params, timeout=30)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("code") != "200000":
                break
            
            rows = data.get("data", [])
            if not rows:
                # No more data, advance past this window
                current_start += MAX_PER_REQUEST * 5 * 60
                time.sleep(PAUSE_SECONDS)
                continue
            
            all_rows.extend(rows)
            
            # KuCoin returns [time, open, close, high, low, volume, turnover]
            # Advance past the last candle we received
            last_time = max(r[0] for r in rows)
            # last_time might be in ms or seconds — handle both
            if last_time > 1e12:
                last_time = last_time / 1000
            current_start = int(last_time) + 300  # +5 min
            
        except Exception as e:
            log(f"    Error fetching {symbol}: {e}")
            current_start += MAX_PER_REQUEST * 5 * 60
        
        time.sleep(PAUSE_SECONDS)
    
    if not all_rows:
        return pd.DataFrame()
    
    # Parse KuCoin Futures kline response.
    # KuCoin Futures API returns: [time, open, close, high, low, volume, turnover]
    # Note: KuCoin order has close before high/low (different from Binance).
    # IMPORTANT: The high and low values are SWAPPED relative to convention —
    #   field index 3 (labeled 'high' in API docs) contains the bar's LOW price
    #   field index 4 (labeled 'low'  in API docs) contains the bar's HIGH price
    # This is a known KuCoin Futures API quirk. We fix it by swapping here.
    df = pd.DataFrame(all_rows, columns=[
        "time", "open", "close", "low", "high", "volume", "turnover"
    ])  # Note: low and high swapped to match actual values
    
    # Convert time — KuCoin returns seconds or milliseconds
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if df["time"].median() > 1e12:
        df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    else:
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
    
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Rename to match Binance format
    df = df.rename(columns={"turnover": "quote_volume"})
    
    # KuCoin doesn't provide taker buy volume or trade count in klines
    # Set them to NaN — our alphas that use these will be masked out
    df["trades_count"] = np.nan
    df["taker_buy_volume"] = np.nan
    df["taker_buy_quote_volume"] = np.nan
    
    df = df[["datetime", "open", "high", "low", "close", "volume",
             "quote_volume", "trades_count", "taker_buy_volume",
             "taker_buy_quote_volume"]].copy()
    
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df



def download_symbol(symbol: str, binance_name: str, start_ts: int, end_ts: int) -> int:
    """Download all data for one symbol, save to parquet.
    
    Handles incremental updates: if file already exists, only downloads
    the date range that is missing from the existing data.
    """
    fpath = KLINES_DIR / f"{binance_name}.parquet"
    
    # Check what date range we already have
    existing_start_ts = None
    existing_end_ts = None
    if fpath.exists():
        try:
            existing = pd.read_parquet(fpath)
            if not existing.empty:
                existing["datetime"] = pd.to_datetime(existing["datetime"])
                existing_start_ts = int(existing["datetime"].min().timestamp())
                existing_end_ts = int(existing["datetime"].max().timestamp())
        except Exception:
            pass
    
    # Determine what range to download
    # If we have data, only download the missing prefix (before existing start)
    # or suffix (after existing end). For backfills, this is the prefix.
    download_start = start_ts
    download_end = end_ts
    
    if existing_start_ts is not None:
        if existing_start_ts <= start_ts + 300 and existing_end_ts >= end_ts - 300:
            return 0  # Full coverage, skip
        if existing_start_ts > start_ts + 300:
            # Need to backfill: download from start_ts up to existing_start_ts
            download_end = existing_start_ts - 300  # stop just before existing data
        elif existing_end_ts < end_ts - 300:
            # Need forward fill: download from existing_end_ts
            download_start = existing_end_ts + 300
    
    if download_start >= download_end:
        return 0
    
    df_new = download_klines(symbol, download_start, download_end)
    if df_new.empty:
        return 0
    
    # Merge with existing data if present
    if fpath.exists():
        try:
            existing = pd.read_parquet(fpath)
            if not existing.empty:
                combined = pd.concat([existing, df_new], ignore_index=True)
                combined = combined.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
                combined.to_parquet(fpath, index=False)
                return len(df_new)
        except Exception:
            pass
    
    df_new.to_parquet(fpath, index=False)
    return len(df_new)


def build_matrices():
    """Build derived field matrices from 5m klines — same logic as Binance pipeline."""
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
    fields = ["open", "high", "low", "close", "volume", "quote_volume"]

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

    # Taker buy ratio — KuCoin doesn't provide this in klines
    # Create NaN placeholder so alpha expressions don't crash
    tbr = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    tbr.to_parquet(MATRICES_DIR / "taker_buy_ratio.parquet")
    
    taker_buy_volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    taker_buy_volume.to_parquet(MATRICES_DIR / "taker_buy_volume.parquet")
    
    taker_buy_quote_volume = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    taker_buy_quote_volume.to_parquet(MATRICES_DIR / "taker_buy_quote_volume.parquet")
    
    trades_count = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    trades_count.to_parquet(MATRICES_DIR / "trades_count.parquet")

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

    # Momentum (in bars, not days)
    for window_d, label in [(5, "5d"), (20, "20d"), (60, "60d")]:
        bars = window_d * 288
        mom = close / close.shift(bars) - 1
        mom.to_parquet(MATRICES_DIR / f"momentum_{label}.parquet")

    # Overnight gap
    gap = open_ / close.shift(1) - 1
    gap.to_parquet(MATRICES_DIR / "overnight_gap.parquet")

    # Trades per volume — KuCoin placeholder
    tpv = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
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

    # BTC beta
    btc_col = None
    for candidate in ["BTCUSDT", "XBTUSDT"]:
        if candidate in returns.columns:
            btc_col = candidate
            break
    
    if btc_col:
        btc_ret = returns[btc_col]
        btc_var = btc_ret.rolling(288, min_periods=144).var()
        betas = {}
        for col in returns.columns:
            if col == btc_col:
                betas[col] = pd.Series(1.0, index=returns.index)
                continue
            cov = returns[col].rolling(288, min_periods=144).cov(btc_ret)
            betas[col] = cov / btc_var
        beta_df = pd.DataFrame(betas)
        beta_df.to_parquet(MATRICES_DIR / "beta_to_btc.parquet")
        log(f"    beta_to_btc: {beta_df.shape}")
    else:
        log("    WARNING: No BTC column found, skipping beta_to_btc")

    log("  All 5m KuCoin matrices built")


def build_universes():
    """Build universe from 5m adv20 data.
    
    Uses same approach as Binance: ADV20-ranked, rebalance every 1440 bars,
    survivorship-bias-free (only adds coins at rebalance dates).
    """
    log("Building 5m KuCoin universes...")
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)

    adv_path = MATRICES_DIR / "adv20.parquet"
    if not adv_path.exists():
        log("  WARNING: adv20 not found!")
        return

    adv20 = pd.read_parquet(adv_path)
    all_dates = adv20.index
    n_symbols = len(adv20.columns)

    # Rebalance every 1440 bars (~5 days) 
    rebal_period = 1440
    start_bar = min(288, len(all_dates) - 1)
    rebal_indices = list(range(start_bar, len(all_dates), rebal_period))
    rebal_dates = all_dates[rebal_indices]
    log(f"  {len(rebal_dates)} rebalance dates (every {rebal_period} bars, start bar {start_bar})")

    # KuCoin has fewer contracts than Binance, adjust tier sizes
    for tier_name, tier_size in [("TOP100", min(100, n_symbols)), 
                                  ("TOP50", min(50, n_symbols)), 
                                  ("TOP20", min(20, n_symbols))]:
        mask = pd.DataFrame(False, index=all_dates, columns=adv20.columns)

        for i, reb_date in enumerate(rebal_dates):
            adv_row = adv20.loc[reb_date].dropna()
            adv_row = adv_row[adv_row > 0]
            top_n = adv_row.nlargest(min(tier_size, len(adv_row))).index.tolist()

            if i == 0:
                period = all_dates[all_dates < (rebal_dates[1] if len(rebal_dates) > 1 else all_dates[-1])]
            elif i + 1 < len(rebal_dates):
                end_date = rebal_dates[i + 1]
                period = all_dates[(all_dates >= reb_date) & (all_dates < end_date)]
            else:
                period = all_dates[all_dates >= reb_date]

            mask.loc[period, top_n] = True

        out_path = UNIVERSE_DIR / f"KUCOIN_{tier_name}_5m.parquet"
        mask.to_parquet(out_path)
        avg_count = mask.sum(axis=1).mean()
        log(f"  {tier_name}: avg {avg_count:.0f} symbols per bar -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Download 5m KuCoin Futures klines")
    parser.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-27", help="End date (YYYY-MM-DD)")
    parser.add_argument("--rebuild-only", action="store_true", help="Skip download, rebuild matrices/universes")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols to download (0=all)")
    args = parser.parse_args()

    KLINES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_only:
        # Get all active KuCoin perpetual contracts
        contracts = get_kucoin_symbols()
        
        if args.max_symbols > 0:
            contracts = contracts[:args.max_symbols]

        start_ts = int(datetime.strptime(args.start, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(args.end, "%Y-%m-%d").timestamp())

        print("=" * 60)
        print("5m KLINES DOWNLOAD (KuCoin Futures)")
        print(f"Symbols: {len(contracts)}")
        print(f"Dates: {args.start} to {args.end}")
        print("=" * 60)

        completed = 0
        total_bars = 0
        skipped = []

        def _dl(contract):
            kucoin_sym = contract["symbol"]
            binance_name = kucoin_symbol_to_binance(kucoin_sym)
            try:
                bars = download_symbol(kucoin_sym, binance_name, start_ts, end_ts)
                return kucoin_sym, binance_name, bars
            except Exception as e:
                return kucoin_sym, binance_name, -1

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_dl, c): c for c in contracts}
            for future in as_completed(futures):
                ksym, bname, bars = future.result()
                completed += 1
                if bars > 0:
                    total_bars += bars
                    if completed <= 5 or completed % 25 == 0:
                        log(f"  {ksym} -> {bname}: {bars} bars")
                elif bars <= 0:
                    skipped.append(ksym)
                if completed % 50 == 0:
                    log(f"  {completed}/{len(contracts)} symbols done ({total_bars:,} bars)")

        log(f"  Download complete: {completed - len(skipped)} symbols with data, "
            f"{len(skipped)} skipped, {total_bars:,} total bars")

    # Build matrices
    build_matrices()

    # Build universes
    build_universes()

    print("\n" + "=" * 60)
    print("KuCoin 5m DATA PIPELINE COMPLETE")
    print(f"  Klines: {KLINES_DIR}")
    print(f"  Matrices: {MATRICES_DIR}")
    print(f"  Universes: {UNIVERSE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
