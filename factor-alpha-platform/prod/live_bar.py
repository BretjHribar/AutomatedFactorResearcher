"""
prod/live_bar.py — Live Bar Construction for Delay-0 MOC Trading

In the backtest (delay=0), on day T the alpha expression evaluates using
day T's OHLCV. In production, we must construct today's estimated bar at
~3:40 PM and append it to the historical matrices before computing signals.

Pipeline:
  1. Fetch today's live OHLCV from FMP batch-quote API for all universe tickers
  2. Sanity-check vs historical: detect splits/dividends via previousClose mismatch
  3. Construct a single-row "live bar" for open, high, low, close, volume, vwap
  4. Append to each matrix so the last row IS today's estimated data
  5. Returns become computable from the extended close series

Split/dividend detection:
  - FMP returns previousClose (yesterday's adjusted close)
  - Our historical close.iloc[-1] is yesterday's close
  - If |fmp_prevClose - hist_close| / hist_close > 2%, flag as split/dividend
  - Exclude flagged tickers from today's signal (data mismatch → garbage signal)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("moc_prod")

# FMP API config
FMP_API_KEY = os.environ.get(
    "FMP_API_KEY", "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy"
)
FMP_BATCH_QUOTE_URL = "https://financialmodelingprep.com/stable/batch-quote"
BATCH_SIZE = 100  # FMP supports comma-separated batch


def fetch_fmp_live_quotes(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch today's live OHLCV from FMP for all tickers.

    Returns: {ticker: {open, high, low, price, volume, previousClose, ...}}
    """
    quotes = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        symbols = ",".join(batch)
        try:
            resp = requests.get(
                FMP_BATCH_QUOTE_URL,
                params={"symbols": symbols, "apikey": FMP_API_KEY},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for q in data:
                    sym = q.get("symbol")
                    if sym:
                        quotes[sym] = q
        except Exception as e:
            log.warning(f"  FMP batch-quote failed for batch {i//BATCH_SIZE + 1}: {e}")

    log.info(f"  FMP live quotes: {len(quotes)}/{len(tickers)} tickers received")
    return quotes


def detect_corporate_actions(
    quotes: dict[str, dict],
    hist_close: pd.Series,
    tolerance: float = 0.02,
) -> tuple[list[str], list[str]]:
    """
    Detect splits/dividends by comparing FMP previousClose to our historical close.

    If |fmp_prevClose - hist_close| / hist_close > tolerance, the ticker likely
    had a split or ex-dividend that our historical data reflects differently.

    Returns: (clean_tickers, flagged_tickers)
    """
    clean = []
    flagged = []

    for sym, q in quotes.items():
        fmp_prev = q.get("previousClose")
        hist_val = hist_close.get(sym)

        if fmp_prev is None or hist_val is None or hist_val == 0:
            # Can't compare — include but note it
            clean.append(sym)
            continue

        pct_diff = abs(fmp_prev - hist_val) / abs(hist_val)
        if pct_diff > tolerance:
            flagged.append(sym)
            log.warning(
                f"  CORP ACTION: {sym} FMP_prevClose=${fmp_prev:.2f} vs "
                f"hist_close=${hist_val:.2f} ({pct_diff:.1%} diff)"
            )
        else:
            clean.append(sym)

    if flagged:
        log.warning(f"  {len(flagged)} tickers flagged for corporate actions: {flagged[:10]}")
    else:
        log.info("  No corporate action mismatches detected")

    return clean, flagged


def construct_live_bar(
    matrices: dict[str, pd.DataFrame],
    quotes: dict[str, dict],
    flagged_tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Append today's estimated bar to each matrix.

    For each field, construct a single-row DataFrame for today and concat:
      - close  = current price (~3:40 PM, best estimate of today's close)
      - open   = today's actual open
      - high   = today's intraday high
      - low    = today's intraday low
      - volume = today's volume so far (partial day)
      - vwap   = estimated as (high + low + close) / 3  (typical price proxy)
      - returns = (live_close - yesterday_close) / yesterday_close

    Flagged tickers (splits/dividends) get NaN to exclude them from signals.
    """
    today = pd.Timestamp(dt.date.today())
    ref_cols = matrices["close"].columns.tolist()
    hist_close = matrices["close"].iloc[-1]

    # Build live row for each field
    live_data = {
        "close": {},
        "open": {},
        "high": {},
        "low": {},
        "volume": {},
    }

    for sym in ref_cols:
        if sym in flagged_tickers:
            # Exclude — NaN will propagate through alpha expressions
            for field in live_data:
                live_data[field][sym] = np.nan
            continue

        q = quotes.get(sym)
        if q is None:
            # No quote available — use NaN
            for field in live_data:
                live_data[field][sym] = np.nan
            continue

        price = q.get("price")
        if price is None or price <= 0:
            for field in live_data:
                live_data[field][sym] = np.nan
            continue

        live_data["close"][sym] = price
        live_data["open"][sym] = q.get("open", price)
        live_data["high"][sym] = q.get("dayHigh", price)
        live_data["low"][sym] = q.get("dayLow", price)
        live_data["volume"][sym] = q.get("volume", 0)

    # Convert to single-row DataFrames
    extended = {}
    for name, df in matrices.items():
        if name in live_data:
            live_row = pd.DataFrame(
                {col: [live_data[name].get(col, np.nan)] for col in ref_cols},
                index=[today],
            )
            extended[name] = pd.concat([df, live_row])
        else:
            extended[name] = df

    # Compute vwap as typical price = (H + L + C) / 3
    if "vwap" in matrices:
        vwap_row = {}
        for sym in ref_cols:
            h = live_data["high"].get(sym)
            l = live_data["low"].get(sym)
            c = live_data["close"].get(sym)
            if h and l and c and not (np.isnan(h) or np.isnan(l) or np.isnan(c)):
                vwap_row[sym] = (h + l + c) / 3.0
            else:
                vwap_row[sym] = np.nan
        vwap_live = pd.DataFrame(
            {col: [vwap_row.get(col, np.nan)] for col in ref_cols},
            index=[today],
        )
        extended["vwap"] = pd.concat([matrices["vwap"], vwap_live])

    # Recompute returns from extended close
    if "close" in extended:
        raw_ret = extended["close"].pct_change(fill_method=None)
        MAX_DAILY_RETURN = 0.5
        raw_ret = raw_ret.where(raw_ret.abs() <= MAX_DAILY_RETURN, np.nan)
        extended["returns"] = raw_ret
        extended["log_returns"] = np.log1p(raw_ret.fillna(0))

    n_live = sum(1 for v in live_data["close"].values()
                 if v is not None and not (isinstance(v, float) and np.isnan(v)))
    log.info(f"  Live bar appended: {n_live}/{len(ref_cols)} tickers with prices")
    log.info(f"  Matrices now end at: {extended['close'].index[-1].date()}")

    return extended


def append_live_bar(matrices: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    """
    Full pipeline: fetch → detect corp actions → construct → return extended matrices.

    Returns: (extended_matrices, live_quotes, flagged_tickers)
    """
    tickers = matrices["close"].columns.tolist()
    hist_close = matrices["close"].iloc[-1]
    today = dt.date.today()

    # Check if matrices already include today
    last_date = matrices["close"].index[-1].date()
    if last_date >= today:
        log.info(f"  Matrices already include today ({last_date}), skipping live bar")
        return matrices, {}, []

    log.info(f"  Historical data ends: {last_date}")
    log.info(f"  Fetching live quotes for {len(tickers)} tickers...")

    # Step 1: Fetch live quotes
    t0 = time.time()
    quotes = fetch_fmp_live_quotes(tickers)
    log.info(f"  Quotes fetched in {time.time()-t0:.1f}s")

    if len(quotes) < 50:
        log.error(f"  Only {len(quotes)} quotes received — aborting live bar")
        return matrices, quotes, []

    # Step 2: Detect corporate actions
    log.info("  Checking for splits/dividends...")
    clean, flagged = detect_corporate_actions(quotes, hist_close)

    # Step 3: Construct and append live bar
    log.info("  Constructing live bar...")
    extended = construct_live_bar(matrices, quotes, flagged)

    return extended, quotes, flagged


if __name__ == "__main__":
    """Quick test: load data, append live bar, print summary."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import eval_alpha_ib
    eval_alpha_ib.UNIVERSE = "TOP2000TOP3000"
    eval_alpha_ib.NEUTRALIZE = "market"
    matrices, universe, classifications = eval_alpha_ib.load_data("full")

    print(f"\nBefore: close matrix shape = {matrices['close'].shape}")
    print(f"        last date = {matrices['close'].index[-1].date()}")

    extended, quotes, flagged = append_live_bar(matrices)

    print(f"\nAfter:  close matrix shape = {extended['close'].shape}")
    print(f"        last date = {extended['close'].index[-1].date()}")

    if flagged:
        print(f"\nFlagged tickers (corp actions): {flagged}")

    # Show some live values
    live_close = extended["close"].iloc[-1]
    live_open = extended["open"].iloc[-1]
    live_high = extended["high"].iloc[-1]
    live_low = extended["low"].iloc[-1]
    live_vol = extended["volume"].iloc[-1]

    print(f"\nSample live bar values:")
    for sym in list(live_close.dropna().index[:5]):
        print(f"  {sym}: O={live_open[sym]:.2f} H={live_high[sym]:.2f} "
              f"L={live_low[sym]:.2f} C={live_close[sym]:.2f} V={live_vol[sym]:,.0f}")

    # Show today's returns
    live_ret = extended["returns"].iloc[-1]
    print(f"\n  Today's returns: mean={live_ret.mean():.4f}, "
          f"median={live_ret.median():.4f}, "
          f"range=[{live_ret.min():.3f}, {live_ret.max():.3f}]")
