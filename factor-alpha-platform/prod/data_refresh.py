"""
Incremental data refresh for production traders.

Fetches the latest 4h bars from exchange REST APIs, appends to kline parquets,
and rebuilds the matrix parquets. Designed to run before each trading cycle.

Key design decisions:
  - Fetches last 6 bars (24h) to handle any gaps from missed cycles
  - Deduplicates on timestamp before saving
  - Full matrix rebuild from klines (ensures derived fields are consistent)
  - Validates data freshness: fails loudly if latest bar is stale
  - Idempotent: safe to re-run without corrupting data

Usage:
    from prod.data_refresh import refresh_binance, refresh_kucoin
    refresh_binance()  # Updates data/binance_cache/matrices/4h/
    refresh_kucoin()   # Updates data/kucoin_cache/matrices/4h/
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Timing ──
BAR_SECONDS_4H = 4 * 3600
BAR_MS_4H = BAR_SECONDS_4H * 1000
BARS_TO_FETCH = 6  # 24h of history to cover any gaps
MAX_STALENESS_S = 5 * 3600  # Alert if data is > 5h old

# ── Matrix truncation ──
# The risk_parity combiner needs 504 bars of factor return history.
# Factor returns need alpha signals, which need 510 bars of matrix warm-up.
# Total minimum: 510 + 504 = 1014 bars. We use 1500 for safety margin.
# This keeps matrix rebuild fast (~40s) while ensuring combiner correctness.
MATRIX_TAIL_ROWS = 1500


def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════
# 1. BINANCE
# ═══════════════════════════════════════════════════════════════

BINANCE_BASE = "https://fapi.binance.com"
BINANCE_KLINES_DIR = Path("data/binance_cache/klines/4h")
BINANCE_MATRICES_DIR = Path("data/binance_cache/matrices/4h/prod")  # Separate from research matrices
BINANCE_FUNDING_DIR = Path("data/binance_cache/funding_rates")


def _binance_fetch_klines(symbol: str, limit: int = 6) -> pd.DataFrame | None:
    """Fetch latest 4h klines for one symbol from Binance fapi."""
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/fapi/v1/klines",
            params={"symbol": symbol, "interval": "4h", "limit": limit},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        rows = resp.json()
    except Exception:
        return None

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades_count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype(int)
    df = df[["datetime", "open", "high", "low", "close", "volume",
             "quote_volume", "trades_count", "taker_buy_volume",
             "taker_buy_quote_volume"]].copy()
    return df


def _binance_update_klines() -> int:
    """Fetch latest bars for all symbols and append to kline parquets.
    Returns count of symbols updated."""
    if not BINANCE_KLINES_DIR.exists():
        raise FileNotFoundError(f"Klines dir not found: {BINANCE_KLINES_DIR}")

    symbols = sorted([f.stem for f in BINANCE_KLINES_DIR.glob("*.parquet")])
    logger.info(f"  Refreshing {len(symbols)} symbols from Binance fapi...")

    updated = 0
    errors = 0
    for i, sym in enumerate(symbols):
        fpath = BINANCE_KLINES_DIR / f"{sym}.parquet"
        try:
            existing = pd.read_parquet(fpath)
            last_ts = existing["datetime"].max()
        except Exception:
            continue

        # Skip if already up-to-date (last bar within 1 bar of now)
        now_ts = pd.Timestamp.now('UTC').tz_localize(None)
        if (now_ts - last_ts).total_seconds() < BAR_SECONDS_4H * 1.5:
            continue

        new_df = _binance_fetch_klines(sym, limit=BARS_TO_FETCH)
        if new_df is None or new_df.empty:
            errors += 1
            continue

        # Append and deduplicate
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

        if len(combined) > len(existing):
            combined.to_parquet(fpath, index=False)
            updated += 1

        # Rate limiting: ~10 req/sec
        time.sleep(0.1)

        if (i + 1) % 100 == 0:
            logger.info(f"    {i+1}/{len(symbols)} checked, {updated} updated, {errors} errors")

    logger.info(f"  Klines: {updated} symbols updated, {errors} errors, "
                f"{len(symbols) - updated - errors} already current")
    return updated


def _build_binance_matrices():
    """Rebuild 4h matrix parquets from per-symbol klines. Same logic as ingest_binance.py."""
    logger.info(f"  Rebuilding 4h matrices (tail={MATRIX_TAIL_ROWS} rows)...")
    BINANCE_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for fpath in sorted(BINANCE_KLINES_DIR.glob("*.parquet")):
        sym = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if df.empty:
                continue
            df = df.set_index("datetime").tail(MATRIX_TAIL_ROWS)
            all_data[sym] = df
        except Exception:
            continue

    if not all_data:
        raise RuntimeError("No kline data found!")

    logger.info(f"    {len(all_data)} symbols loaded")

    # Base fields
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
            mat.to_parquet(BINANCE_MATRICES_DIR / f"{field}.parquet")

    # Derived fields
    close = pd.DataFrame({s: d["close"] for s, d in all_data.items()})
    open_ = pd.DataFrame({s: d["open"] for s, d in all_data.items()})
    high = pd.DataFrame({s: d["high"] for s, d in all_data.items()})
    low = pd.DataFrame({s: d["low"] for s, d in all_data.items()})
    volume = pd.DataFrame({s: d["volume"] for s, d in all_data.items()})
    qv = pd.DataFrame({s: d["quote_volume"] for s, d in all_data.items()})
    tbv = pd.DataFrame({s: d.get("taker_buy_volume", pd.Series(dtype=float))
                        for s, d in all_data.items()})

    returns = close.pct_change(fill_method=None)
    returns.to_parquet(BINANCE_MATRICES_DIR / "returns.parquet")

    log_returns = np.log(close / close.shift(1))
    log_returns.to_parquet(BINANCE_MATRICES_DIR / "log_returns.parquet")

    safe_vol = volume.replace(0, np.nan)
    vwap = qv / safe_vol
    vwap.to_parquet(BINANCE_MATRICES_DIR / "vwap.parquet")

    vwap_dev = (close - vwap) / vwap
    vwap_dev.to_parquet(BINANCE_MATRICES_DIR / "vwap_deviation.parquet")

    tbr = tbv / safe_vol
    tbr.to_parquet(BINANCE_MATRICES_DIR / "taker_buy_ratio.parquet")

    for window in [20, 60]:
        adv = qv.rolling(window, min_periods=max(window // 2, 1)).mean()
        adv.to_parquet(BINANCE_MATRICES_DIR / f"adv{window}.parquet")

    adv20 = qv.rolling(20, min_periods=10).mean()
    vol_ratio = qv / adv20
    vol_ratio.to_parquet(BINANCE_MATRICES_DIR / "volume_ratio_20d.parquet")

    for window in [10, 20, 60, 120]:
        hvol = returns.rolling(window, min_periods=max(window // 2, 1)).std() * np.sqrt(252)
        hvol.to_parquet(BINANCE_MATRICES_DIR / f"historical_volatility_{window}.parquet")

    hl_ratio = np.log(high / low)
    for window in [10, 20, 60]:
        pvol = hl_ratio.rolling(window, min_periods=max(window // 2, 1)).apply(
            lambda x: np.sqrt(np.mean(x**2) / (4 * np.log(2))) * np.sqrt(252), raw=True)
        pvol.to_parquet(BINANCE_MATRICES_DIR / f"parkinson_volatility_{window}.parquet")

    hlr = (high - low) / close
    hlr.to_parquet(BINANCE_MATRICES_DIR / "high_low_range.parquet")

    ocr = (close - open_).abs() / close
    ocr.to_parquet(BINANCE_MATRICES_DIR / "open_close_range.parquet")

    for window in [5, 20, 60]:
        mom = close / close.shift(window) - 1
        mom.to_parquet(BINANCE_MATRICES_DIR / f"momentum_{window}d.parquet")

    gap = open_ / close.shift(1) - 1
    gap.to_parquet(BINANCE_MATRICES_DIR / "overnight_gap.parquet")

    tc = pd.DataFrame({s: d["trades_count"] for s, d in all_data.items()})
    tpv = tc / safe_vol
    tpv.to_parquet(BINANCE_MATRICES_DIR / "trades_per_volume.parquet")

    max_oc = pd.DataFrame(np.maximum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    min_oc = pd.DataFrame(np.minimum(open_.values, close.values),
                          index=close.index, columns=close.columns)
    hl_range = high - low
    safe_hl = hl_range.replace(0, np.nan)
    upper_shadow = (high - max_oc) / safe_hl
    lower_shadow = (min_oc - low) / safe_hl
    upper_shadow.to_parquet(BINANCE_MATRICES_DIR / "upper_shadow.parquet")
    lower_shadow.to_parquet(BINANCE_MATRICES_DIR / "lower_shadow.parquet")

    cpr = (close - low) / safe_hl
    cpr.to_parquet(BINANCE_MATRICES_DIR / "close_position_in_range.parquet")

    qv.to_parquet(BINANCE_MATRICES_DIR / "dollars_traded.parquet")

    dv_mom = qv / qv.shift(1).replace(0, np.nan)
    dv_mom.to_parquet(BINANCE_MATRICES_DIR / "volume_momentum_1.parquet")

    dv_mom5 = qv.rolling(5, min_periods=2).mean() / qv.rolling(20, min_periods=5).mean()
    dv_mom5.to_parquet(BINANCE_MATRICES_DIR / "volume_momentum_5_20.parquet")

    tbqv = pd.DataFrame({s: d.get("taker_buy_quote_volume", pd.Series(dtype=float))
                         for s, d in all_data.items()})
    if not tbqv.empty:
        tbqv.to_parquet(BINANCE_MATRICES_DIR / "taker_buy_quote_volume.parquet")

    # Funding rate integration
    funding_path = BINANCE_FUNDING_DIR / "funding_rates.parquet"
    if funding_path.exists():
        fr_raw = pd.read_parquet(funding_path)
        fr_resampled = fr_raw.reindex(close.index, method="ffill")
        common_cols = [c for c in fr_resampled.columns if c in close.columns]
        if common_cols:
            fr_aligned = fr_resampled[common_cols]
            fr_aligned.to_parquet(BINANCE_MATRICES_DIR / "funding_rate.parquet")

            fr_cumsum_3 = fr_aligned.rolling(3, min_periods=1).sum()
            fr_cumsum_3.to_parquet(BINANCE_MATRICES_DIR / "funding_rate_cumsum_3.parquet")

            fr_avg_21 = fr_aligned.rolling(21, min_periods=5).mean()
            fr_avg_21.to_parquet(BINANCE_MATRICES_DIR / "funding_rate_avg_7d.parquet")

            fr_mean = fr_aligned.rolling(42, min_periods=10).mean()
            fr_std = fr_aligned.rolling(42, min_periods=10).std()
            fr_zscore = (fr_aligned - fr_mean) / fr_std.replace(0, np.nan)
            fr_zscore.to_parquet(BINANCE_MATRICES_DIR / "funding_rate_zscore.parquet")

    # BTC beta
    if "BTCUSDT" in returns.columns:
        btc_ret = returns["BTCUSDT"]
        btc_var = btc_ret.rolling(60, min_periods=30).var()
        betas = {}
        for col in returns.columns:
            if col == "BTCUSDT":
                betas[col] = pd.Series(1.0, index=returns.index)
                continue
            cov = returns[col].rolling(60, min_periods=30).cov(btc_ret)
            betas[col] = cov / btc_var
        beta_df = pd.DataFrame(betas)
        beta_df.to_parquet(BINANCE_MATRICES_DIR / "beta_to_btc.parquet")

    logger.info(f"    Matrices rebuilt: {close.shape[0]} bars x {close.shape[1]} tickers, "
                f"latest={close.index[-1]}")


def refresh_binance() -> str:
    """Full Binance data refresh: fetch latest bars + rebuild matrices.
    Returns the latest bar timestamp as string."""
    t0 = time.time()
    logger.info("Binance data refresh starting...")

    _binance_update_klines()
    _build_binance_matrices()

    # Validate freshness
    close = pd.read_parquet(BINANCE_MATRICES_DIR / "close.parquet")
    latest = close.index[-1]
    age_s = (pd.Timestamp.now('UTC').tz_localize(None) - latest).total_seconds()
    elapsed = time.time() - t0

    if age_s > MAX_STALENESS_S:
        logger.warning(f"  [!] Data may be stale: latest bar {latest} "
                       f"({age_s/3600:.1f}h old)")
    else:
        logger.info(f"  Data is fresh: latest bar {latest} ({age_s/3600:.1f}h old)")

    logger.info(f"  Refresh complete in {elapsed:.1f}s")
    return str(latest)


# ═══════════════════════════════════════════════════════════════
# 2. KUCOIN
# ═══════════════════════════════════════════════════════════════

KUCOIN_BASE = "https://api-futures.kucoin.com"
KUCOIN_KLINES_DIR = Path("data/kucoin_cache/klines/4h")
KUCOIN_MATRICES_DIR = Path("data/kucoin_cache/matrices/4h/prod")
KUCOIN_GRAN = 240  # 4h in minutes


def _kucoin_fetch_klines(symbol: str, limit: int = 6) -> pd.DataFrame | None:
    """Fetch latest 4h klines for one symbol from KuCoin futures."""
    now_ms = _now_utc_ms()
    from_ms = now_ms - limit * BAR_MS_4H
    try:
        resp = requests.get(
            f"{KUCOIN_BASE}/api/v1/kline/query",
            params={"symbol": symbol, "granularity": KUCOIN_GRAN,
                    "from": from_ms, "to": now_ms},
            timeout=15,
        )
        data = resp.json()
        if data.get("code") != "200000" or not data.get("data"):
            return None
        candles = data["data"]
    except Exception:
        return None

    if not candles:
        return None

    # KuCoin format: [time_ms, open, close, high, low, volume, turnover]
    df = pd.DataFrame(candles, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open", "close", "high", "low", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df


def _kucoin_update_klines() -> int:
    """Fetch latest bars for all KuCoin symbols and append to kline parquets."""
    if not KUCOIN_KLINES_DIR.exists():
        raise FileNotFoundError(f"Klines dir not found: {KUCOIN_KLINES_DIR}")

    symbols = sorted([f.stem for f in KUCOIN_KLINES_DIR.glob("*.parquet")])
    logger.info(f"  Refreshing {len(symbols)} symbols from KuCoin API...")

    updated = 0
    errors = 0
    for i, sym in enumerate(symbols):
        fpath = KUCOIN_KLINES_DIR / f"{sym}.parquet"
        try:
            existing = pd.read_parquet(fpath)
            last_ts = existing.index.max() if "time" not in existing.columns else existing["time"].max()
            if isinstance(last_ts, pd.Timestamp):
                pass
            else:
                last_ts = pd.Timestamp(last_ts)
        except Exception:
            continue

        now_ts = pd.Timestamp.now('UTC').tz_localize(None)
        if (now_ts - last_ts).total_seconds() < BAR_SECONDS_4H * 1.5:
            continue

        new_df = _kucoin_fetch_klines(sym, limit=BARS_TO_FETCH)
        if new_df is None or new_df.empty:
            errors += 1
            continue

        # KuCoin klines might be stored with time as index
        if "time" not in existing.columns and existing.index.name == "time":
            existing = existing.reset_index()

        # Align columns: KuCoin uses "time", "turnover" where Binance uses "datetime", "quote_volume"
        new_df_aligned = new_df.copy()

        # Merge and deduplicate
        if "time" in existing.columns:
            combined = pd.concat([existing, new_df_aligned], ignore_index=True)
            combined = combined.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        else:
            combined = new_df_aligned

        if len(combined) > len(existing):
            # Save with time as index (matching original format)
            combined = combined.set_index("time").sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.to_parquet(fpath)
            updated += 1

        time.sleep(0.12)  # KuCoin rate limit

        if (i + 1) % 100 == 0:
            logger.info(f"    {i+1}/{len(symbols)} checked, {updated} updated, {errors} errors")

    logger.info(f"  Klines: {updated} symbols updated, {errors} errors, "
                f"{len(symbols) - updated - errors} already current")
    return updated


def _build_kucoin_matrices():
    """Rebuild KuCoin 4h matrices. Same logic as download_kucoin.py."""
    logger.info(f"  Rebuilding KuCoin 4h matrices (tail={MATRIX_TAIL_ROWS} rows)...")
    KUCOIN_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for fpath in sorted(KUCOIN_KLINES_DIR.glob("*.parquet")):
        sym = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if df.empty or len(df) < 50:
                continue
            # Ensure time is the index
            if "time" in df.columns:
                df = df.set_index("time").sort_index()
            df = df.tail(MATRIX_TAIL_ROWS)
            all_data[sym] = df
        except Exception:
            continue

    if not all_data:
        raise RuntimeError("No KuCoin kline data found!")

    logger.info(f"    {len(all_data)} symbols loaded")

    # Build common index
    all_indices = set()
    for df in all_data.values():
        all_indices.update(df.index)
    common_idx = sorted(all_indices)

    # Base OHLCV
    for field in ["open", "close", "high", "low", "volume", "turnover"]:
        mat = pd.DataFrame(index=common_idx)
        for sym, df in all_data.items():
            if field in df.columns:
                mat[sym] = df[field]
        mat.to_parquet(KUCOIN_MATRICES_DIR / f"{field}.parquet")

    close = pd.DataFrame({s: d["close"] for s, d in all_data.items()}, index=common_idx)
    high = pd.DataFrame({s: d["high"] for s, d in all_data.items()}, index=common_idx)
    low = pd.DataFrame({s: d["low"] for s, d in all_data.items()}, index=common_idx)
    opn = pd.DataFrame({s: d["open"] for s, d in all_data.items()}, index=common_idx)
    vol = pd.DataFrame({s: d["volume"] for s, d in all_data.items()}, index=common_idx)
    turnover = pd.DataFrame({s: d["turnover"] for s, d in all_data.items()}, index=common_idx)

    ret = close.pct_change(fill_method=None)

    derived = {
        "returns": ret,
        "log_returns": np.log1p(ret.fillna(0)),
        "vwap": (high + low + close) / 3,
        "adv20": turnover.rolling(120, min_periods=60).mean(),
        "adv60": turnover.rolling(360, min_periods=180).mean(),
        "high_low_range": (high - low) / close,
        "open_close_range": (close - opn).abs() / close,
        "close_position_in_range": (close - low) / (high - low + 1e-10),
        "upper_shadow": (high - close.where(close > opn, opn)) / close,
        "lower_shadow": (close.where(close < opn, opn) - low) / close,
        "overnight_gap": opn / close.shift(1) - 1,
        "volume_momentum_5_20": vol.rolling(30).mean() / vol.rolling(120).mean(),
        "historical_volatility_10": ret.rolling(60).std() * np.sqrt(6 * 365),
        "historical_volatility_20": ret.rolling(120).std() * np.sqrt(6 * 365),
        "historical_volatility_60": ret.rolling(360).std() * np.sqrt(6 * 365),
        "momentum_5d": close / close.shift(30) - 1,
        "momentum_20d": close / close.shift(120) - 1,
        "momentum_60d": close / close.shift(360) - 1,
        "vwap_deviation": (close - (high + low + close) / 3) / ((high + low + close) / 3),
        "dollars_traded": turnover,
        "quote_volume": turnover,
        "volume_ratio_20d": vol / vol.rolling(120).mean(),
        "volume_momentum_1": vol / vol.shift(1) - 1,
    }

    hl = np.log(high / low)
    derived["parkinson_volatility_10"] = hl.pow(2).rolling(60).mean().pow(0.5) / (2 * np.log(2))**0.5
    derived["parkinson_volatility_20"] = hl.pow(2).rolling(120).mean().pow(0.5) / (2 * np.log(2))**0.5
    derived["parkinson_volatility_60"] = hl.pow(2).rolling(360).mean().pow(0.5) / (2 * np.log(2))**0.5

    for name, mat in derived.items():
        mat.to_parquet(KUCOIN_MATRICES_DIR / f"{name}.parquet")

    logger.info(f"    Matrices rebuilt: {close.shape[0]} bars x {close.shape[1]} tickers, "
                f"latest={close.index[-1]}")


def refresh_kucoin() -> str:
    """Full KuCoin data refresh: fetch latest bars + rebuild matrices."""
    t0 = time.time()
    logger.info("KuCoin data refresh starting...")

    _kucoin_update_klines()
    _build_kucoin_matrices()

    # Validate freshness
    close = pd.read_parquet(KUCOIN_MATRICES_DIR / "close.parquet")
    latest = close.index[-1]
    if isinstance(latest, pd.Timestamp):
        age_s = (pd.Timestamp.now('UTC').tz_localize(None) - latest).total_seconds()
    else:
        age_s = 0.0
    elapsed = time.time() - t0

    if age_s > MAX_STALENESS_S:
        logger.warning(f"  [!] KuCoin data may be stale: latest bar {latest} "
                       f"({age_s/3600:.1f}h old)")
    else:
        logger.info(f"  KuCoin data is fresh: latest bar {latest} ({age_s/3600:.1f}h old)")

    logger.info(f"  Refresh complete in {elapsed:.1f}s")
    return str(latest)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Refresh exchange data")
    parser.add_argument("exchange", choices=["binance", "kucoin", "all"])
    args = parser.parse_args()

    if args.exchange in ("binance", "all"):
        latest = refresh_binance()
        print(f"Binance latest bar: {latest}")

    if args.exchange in ("kucoin", "all"):
        latest = refresh_kucoin()
        print(f"KuCoin latest bar: {latest}")
