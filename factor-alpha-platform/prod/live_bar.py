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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("moc_prod")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FMP API config
FMP_API_KEY = os.environ.get(
    "FMP_API_KEY", "C6T2KGmSbbsDL3sM7gjx680hmUTiEXfy"
)
FMP_BATCH_QUOTE_URL = "https://financialmodelingprep.com/stable/batch-quote"
FMP_INTRADAY_CHART_URL = "https://financialmodelingprep.com/stable/historical-chart/5min"
BATCH_SIZE = 100  # FMP supports comma-separated batch
_quote_tape_dir = Path(os.environ.get("LIVE_QUOTE_TAPE_DIR", "prod/logs/live_quotes"))
LIVE_QUOTE_TAPE_DIR = _quote_tape_dir if _quote_tape_dir.is_absolute() else PROJECT_ROOT / _quote_tape_dir
ENABLE_IB_LIVE_VWAP = os.environ.get("ENABLE_IB_LIVE_VWAP", "0") == "1"
ENABLE_FMP_INTRADAY_VWAP = os.environ.get("ENABLE_FMP_INTRADAY_VWAP", "1") != "0"
FMP_INTRADAY_MAX_WORKERS = int(os.environ.get("FMP_INTRADAY_MAX_WORKERS", "12"))
FMP_INTRADAY_TIMEOUT_SEC = float(os.environ.get("FMP_INTRADAY_TIMEOUT_SEC", "12"))
QUOTE_TAPE_MIN_SNAPSHOTS = int(os.environ.get("QUOTE_TAPE_MIN_SNAPSHOTS", "8"))
QUOTE_TAPE_MIN_SPAN_MINUTES = float(os.environ.get("QUOTE_TAPE_MIN_SPAN_MINUTES", "120"))


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


def save_fmp_quote_snapshot(quotes: dict[str, dict],
                            captured_at: Optional[pd.Timestamp] = None) -> Path | None:
    """Persist the FMP batch quote snapshot for intraday VWAP reconstruction."""
    if not quotes:
        return None
    captured_at = captured_at or pd.Timestamp.utcnow()
    if captured_at.tzinfo is None:
        captured_at = captured_at.tz_localize("UTC")
    LIVE_QUOTE_TAPE_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_QUOTE_TAPE_DIR / f"fmp_quotes_{captured_at.date().isoformat()}.csv"

    rows = []
    for sym, q in quotes.items():
        rows.append({
            "captured_at_utc": captured_at.isoformat(),
            "symbol": sym,
            "price": q.get("price"),
            "volume": q.get("volume"),
            "open": q.get("open"),
            "dayHigh": q.get("dayHigh"),
            "dayLow": q.get("dayLow"),
            "previousClose": q.get("previousClose"),
            "source_timestamp": q.get("timestamp"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    log.info(f"  Quote snapshot appended: {path} ({len(rows)} rows)")
    return path


def _quote_tape_vwap_for_symbol(df: pd.DataFrame) -> float | None:
    """Compute cumulative-volume VWAP from repeated quote snapshots."""
    if len(df) < QUOTE_TAPE_MIN_SNAPSHOTS:
        return None
    work = df.copy()
    work["captured_at_utc"] = pd.to_datetime(work["captured_at_utc"], utc=True, errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    work = work.dropna(subset=["captured_at_utc", "price", "volume"])
    work = work[(work["price"] > 0) & (work["volume"] >= 0)]
    if len(work) < QUOTE_TAPE_MIN_SNAPSHOTS:
        return None
    work = work.sort_values("captured_at_utc")
    span_min = (work["captured_at_utc"].iloc[-1] - work["captured_at_utc"].iloc[0]).total_seconds() / 60
    if span_min < QUOTE_TAPE_MIN_SPAN_MINUTES:
        return None

    vol_delta = work["volume"].diff()
    first_volume = work["volume"].iloc[0]
    # If the collector was already running near the open, include first
    # cumulative volume. Otherwise use only observed increments.
    first_et = work["captured_at_utc"].iloc[0].tz_convert("America/New_York").time()
    if first_et <= dt.time(10, 0):
        vol_delta.iloc[0] = first_volume
    else:
        vol_delta.iloc[0] = np.nan

    vol_delta = vol_delta.clip(lower=0)
    valid = (vol_delta > 0) & work["price"].notna()
    if not valid.any():
        return None
    denom = float(vol_delta[valid].sum())
    if denom <= 0:
        return None
    return float((work.loc[valid, "price"] * vol_delta[valid]).sum() / denom)


def load_quote_tape_vwap(tickers: list[str],
                         today: Optional[dt.date] = None) -> tuple[dict[str, float], dict]:
    """Compute VWAP from the local FMP quote snapshot tape if enough history exists."""
    today = today or dt.date.today()
    path = LIVE_QUOTE_TAPE_DIR / f"fmp_quotes_{today.isoformat()}.csv"
    if not path.exists():
        return {}, {"path": str(path), "reason": "missing_tape"}

    try:
        tape = pd.read_csv(path)
    except Exception as e:
        log.warning(f"  Quote tape read failed: {e}")
        return {}, {"path": str(path), "reason": "read_failed"}

    tape = tape[tape["symbol"].isin(tickers)]
    out = {}
    for sym, sdf in tape.groupby("symbol", sort=False):
        v = _quote_tape_vwap_for_symbol(sdf)
        if v is not None and v > 0:
            out[sym] = v

    meta = {
        "path": str(path),
        "symbols": len(out),
        "rows": int(len(tape)),
        "min_snapshots": QUOTE_TAPE_MIN_SNAPSHOTS,
        "min_span_minutes": QUOTE_TAPE_MIN_SPAN_MINUTES,
    }
    log.info(f"  Quote-tape VWAP: {len(out)}/{len(tickers)} tickers from {path.name}")
    return out, meta


def _intraday_rows_to_vwap(rows: list[dict],
                           today: Optional[dt.date] = None) -> float | None:
    """Compute today's VWAP from FMP intraday OHLCV bars."""
    if not rows:
        return None
    today = today or dt.date.today()
    df = pd.DataFrame(rows)
    if df.empty or "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].dt.date == today]
    if df.empty:
        return None

    for col in ["high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["high", "low", "close", "volume"])
    df = df[df["volume"] > 0]
    if df.empty:
        return None

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    denom = float(df["volume"].sum())
    if denom <= 0:
        return None
    return float((typical * df["volume"]).sum() / denom)


def fetch_fmp_intraday_vwap(tickers: list[str],
                            today: Optional[dt.date] = None,
                            max_workers: int = FMP_INTRADAY_MAX_WORKERS) -> tuple[dict[str, float], dict]:
    """
    Fetch FMP 5-minute bars and compute true intraday VWAP from bar volume.

    This is much slower than batch quote, but far more faithful than a
    one-shot (H+L+C)/3 daily proxy. It is intended as the reliable live-bar
    fallback when we do not yet have a full-day local quote tape.
    """
    today = today or dt.date.today()
    tickers = sorted(set(tickers))
    out: dict[str, float] = {}
    errors: dict[str, str] = {}

    def fetch_one(sym: str) -> tuple[str, float | None, str | None]:
        try:
            resp = requests.get(
                FMP_INTRADAY_CHART_URL,
                params={"symbol": sym, "apikey": FMP_API_KEY},
                timeout=FMP_INTRADAY_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return sym, None, "bad_response"
            vwap = _intraday_rows_to_vwap(data, today=today)
            if vwap is None or vwap <= 0:
                return sym, None, "no_vwap"
            return sym, vwap, None
        except Exception as e:
            return sym, None, str(e)[:120]

    if not tickers:
        return {}, {"errors": errors}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_one, sym) for sym in tickers]
        for fut in as_completed(futures):
            sym, vwap, err = fut.result()
            if vwap is not None:
                out[sym] = vwap
            elif err:
                errors[sym] = err

    meta = {
        "requested": len(tickers),
        "received": len(out),
        "errors": len(errors),
        "elapsed_sec": round(time.time() - t0, 2),
        "interval": "5min",
    }
    log.info(f"  FMP intraday VWAP: {len(out)}/{len(tickers)} tickers "
             f"in {meta['elapsed_sec']:.1f}s")
    if errors:
        sample = list(errors.items())[:5]
        log.warning(f"  FMP intraday VWAP missing {len(errors)} tickers; sample={sample}")
    return out, meta


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
    ib_vwap: dict[str, float] | None = None,
    vwap_sources: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Append today's estimated bar to each matrix.

    For each field, construct a single-row DataFrame for today and concat:
      - close  = current price (~3:40 PM snapshot — last continuous trade pre-auction;
                 paper IB MOC fills land here, NOT at the auction match)
      - open   = today's actual open
      - high   = today's intraday high
      - low    = today's intraday low
      - volume = YESTERDAY'S full-day volume carried forward.
                 Mid-day partial volume mis-states volume_ratio / volume_momentum
                 / dollars_traded by ~5-15% with non-stationary intraday curve, so
                 1-bar carry-forward is the lesser evil. Volume-sensitive alphas
                 are now structurally 1 bar stale; tag them as such if used.
      - vwap   = live intraday VWAP when available, sourced from quote tape,
                 FMP 5-minute bars, or IB stream. Falls back to (H+L+C)/3
                 only when no intraday volume-weighted estimate exists.
      - returns = (live_close - yesterday_close) / yesterday_close

    Flagged tickers (splits/dividends) get NaN to exclude them from signals.
    """
    today = pd.Timestamp(dt.date.today())
    ref_cols = matrices["close"].columns.tolist()
    hist_close = matrices["close"].iloc[-1]
    hist_volume = matrices["volume"].iloc[-1] if "volume" in matrices else None

    # Build live row for each field
    live_data = {
        "close": {},
        "open": {},
        "high": {},
        "low": {},
        "volume": {},
    }

    n_vwap_tape = 0
    n_vwap_fmp_intraday = 0
    n_vwap_ib = 0
    n_vwap_fallback = 0

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
        # Carry-forward yesterday's volume — partial-day volume is an unreliable
        # input for volume-derived alphas; structural 1-day stale is preferable
        # to a biased projection.
        if hist_volume is not None and sym in hist_volume.index:
            yv = hist_volume.get(sym)
            live_data["volume"][sym] = float(yv) if pd.notna(yv) else np.nan
        else:
            live_data["volume"][sym] = np.nan

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

    # VWAP — prefer IB's live VWAP, fall back to typical price (H+L+C)/3
    if "vwap" in matrices:
        vwap_row = {}
        for sym in ref_cols:
            ibv = ib_vwap.get(sym) if ib_vwap else None
            if ibv is not None and ibv > 0:
                vwap_row[sym] = float(ibv)
                source = (vwap_sources or {}).get(sym, "ib_stream")
                if source == "quote_tape":
                    n_vwap_tape += 1
                elif source == "fmp_intraday":
                    n_vwap_fmp_intraday += 1
                else:
                    n_vwap_ib += 1
                continue
            h = live_data["high"].get(sym)
            l = live_data["low"].get(sym)
            c = live_data["close"].get(sym)
            if h and l and c and not (np.isnan(h) or np.isnan(l) or np.isnan(c)):
                vwap_row[sym] = (h + l + c) / 3.0
                n_vwap_fallback += 1
            else:
                vwap_row[sym] = np.nan
        vwap_live = pd.DataFrame(
            {col: [vwap_row.get(col, np.nan)] for col in ref_cols},
            index=[today],
        )
        extended["vwap"] = pd.concat([matrices["vwap"], vwap_live])
        log.info("  VWAP: quote_tape=%s fmp_intraday=%s ib_stream=%s "
                 "typical_price_fallback=%s",
                 n_vwap_tape, n_vwap_fmp_intraday, n_vwap_ib, n_vwap_fallback)

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


def fetch_ib_live_vwap(ib, tickers: list[str], batch_size: int = 50,
                       settle_sec: float = 4.0) -> dict[str, float]:
    """Pull each ticker's live (intraday) VWAP from IB.

    IB delivers a running daily VWAP via the standard real-time market-data
    stream; ib_insync surfaces it as `Ticker.vwap`. Requires either a real-time
    market-data subscription or `reqMarketDataType(3)` for delayed (which is
    what most paper accounts have).

    Returns: {symbol: vwap_float}. Missing/zero values are dropped.
    """
    from ib_insync import Stock
    out = {}
    if ib is None:
        return out
    try:
        ib.reqMarketDataType(3)   # delayed feed (paper accounts lack realtime)
    except Exception as e:
        log.warning(f"  ib.reqMarketDataType(3) failed: {e}")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        reqs = []
        for sym in batch:
            try:
                c = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(c)
                td = ib.reqMktData(c, snapshot=False)
                reqs.append((sym, c, td))
            except Exception:
                continue
        ib.sleep(settle_sec)
        for sym, c, td in reqs:
            v = getattr(td, "vwap", None)
            if v is not None and v > 0:
                out[sym] = float(v)
            ib.cancelMktData(c)
    log.info(f"  IB VWAP: {len(out)}/{len(tickers)} tickers received")
    return out


def append_live_bar(matrices: dict[str, pd.DataFrame],
                    ib=None,
                    vwap_tickers: Optional[list[str]] = None
                    ) -> tuple[dict[str, pd.DataFrame], dict, list[str]]:
    """
    Full pipeline: fetch → detect corp actions → construct → return extended matrices.

    `vwap_tickers` should be the active strategy universe. VWAP is expensive
    to improve, so inactive names can safely use the cheaper fallback while
    symbols that can trade receive quote-tape, FMP intraday, or IB estimates.

    Returns: (extended_matrices, live_quotes, flagged_tickers)
    """
    tickers = matrices["close"].columns.tolist()
    vwap_tickers = sorted(set(vwap_tickers or tickers).intersection(tickers))
    hist_close = matrices["close"].iloc[-1]
    today = dt.date.today()

    # Check if matrices already include today
    last_date = matrices["close"].index[-1].date()
    if last_date >= today:
        log.info(f"  Matrices already include today ({last_date}), skipping live bar")
        return matrices, {}, []

    # Calendar-aware staleness — refuse to append a live bar if we'd be
    # stitching it onto a cache with missing trading days. The downstream
    # rolling-window operators read across the hole.
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=str(last_date),
                              end_date=str(today))
        sched_dates = [d.date() for d in sched.index]
        prior_days = [d for d in sched_dates if d < today]
        if prior_days:
            expected_last = prior_days[-1]
            if last_date != expected_last:
                missing_n = (expected_last - last_date).days
                log.error(f"  REFUSING to append live bar: cache last={last_date} "
                          f"but previous trading day is {expected_last} "
                          f"({missing_n} calendar-day gap). Run data refresh first.")
                raise RuntimeError(
                    f"Cache stale by {missing_n} calendar days; refresh required "
                    f"before live trading."
                )
    except ImportError:
        log.warning("  pandas_market_calendars not installed; cannot run "
                    "trading-day staleness check in live_bar")

    log.info(f"  Historical data ends: {last_date}")
    log.info(f"  Fetching live quotes for {len(tickers)} tickers...")

    # Step 1: Fetch live quotes (FMP)
    t0 = time.time()
    quotes = fetch_fmp_live_quotes(tickers)
    log.info(f"  Quotes fetched in {time.time()-t0:.1f}s")
    save_fmp_quote_snapshot(quotes)

    if len(quotes) < 50:
        log.error(f"  Only {len(quotes)} quotes received — aborting live bar")
        return matrices, quotes, []

    # Step 1b: Build the best available live VWAP map for tradeable names.
    live_vwap: dict[str, float] = {}
    vwap_sources: dict[str, str] = {}
    if ib is not None and ENABLE_IB_LIVE_VWAP:
        try:
            t1 = time.time()
            ib_vwap = fetch_ib_live_vwap(ib, vwap_tickers)
            log.info(f"  IB VWAP fetched in {time.time()-t1:.1f}s "
                     f"({len(ib_vwap)} tickers)")
            live_vwap.update(ib_vwap)
            vwap_sources.update({sym: "ib_stream" for sym in ib_vwap})
        except Exception as e:
            log.warning(f"  IB VWAP fetch failed; will use non-IB VWAP sources: {e}")
    elif ib is not None:
        log.info("  IB live VWAP disabled; set ENABLE_IB_LIVE_VWAP=1 to use it")

    missing_vwap = [sym for sym in vwap_tickers if sym not in live_vwap]
    if ENABLE_FMP_INTRADAY_VWAP and missing_vwap:
        fmp_vwap, _ = fetch_fmp_intraday_vwap(missing_vwap, today=today)
        for sym, vwap in fmp_vwap.items():
            if sym not in live_vwap:
                live_vwap[sym] = vwap
                vwap_sources[sym] = "fmp_intraday"
    elif missing_vwap:
        log.info("  FMP intraday VWAP disabled; %s active tickers will use fallback",
                 len(missing_vwap))

    missing_vwap = [sym for sym in vwap_tickers if sym not in live_vwap]
    if missing_vwap:
        tape_vwap, _ = load_quote_tape_vwap(missing_vwap, today=today)
        for sym, vwap in tape_vwap.items():
            live_vwap[sym] = vwap
            vwap_sources[sym] = "quote_tape"
    final_missing_vwap = [sym for sym in vwap_tickers if sym not in live_vwap]
    if final_missing_vwap:
        log.warning("  Live VWAP unavailable for %s active tickers; "
                    "using typical-price fallback for those symbols",
                    len(final_missing_vwap))

    # Step 2: Detect corporate actions
    log.info("  Checking for splits/dividends...")
    clean, flagged = detect_corporate_actions(quotes, hist_close)

    # Step 3: Construct and append live bar
    log.info("  Constructing live bar...")
    extended = construct_live_bar(
        matrices,
        quotes,
        flagged,
        ib_vwap=live_vwap,
        vwap_sources=vwap_sources,
    )

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
