"""Incremental FMP EOD refresh for the equity production cache.

The MOC strategy uses a delay-0 live bar for today's auction decision, but the
historical FMP parquet cache still must be fresh through the latest fully closed
NYSE session. This module owns that refresh loop: fetch recent FMP EOD rows,
detect late vendor fills/revisions, rebuild matrices only when needed, and
verify the active production universe has usable data for the target bar.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.integrity import expected_nyse_dates, previous_nyse_trading_day


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "adjClose", "changePercent"]
PRICE_MATRIX_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]
PRICE_DERIVED_FIELDS = [
    "returns",
    "dollars_traded",
    "adv20",
    "adv60",
    "log_returns",
    "cap",
    "market_cap",
]
VOLATILITY_WINDOWS = [10, 20, 30, 60, 90, 120, 150, 180]
MARKET_CAP_BANDS: dict[str, tuple[float, float]] = {
    "MCAP_100M_500M": (100_000_000.0, 500_000_000.0),
    "MCAP_200M_1B": (200_000_000.0, 1_000_000_000.0),
    "MCAP_500M_2B": (500_000_000.0, 2_000_000_000.0),
    "MCAP_2B_10B": (2_000_000_000.0, 10_000_000_000.0),
}

log = logging.getLogger(__name__)


def latest_closed_nyse_session(
    as_of: dt.datetime | pd.Timestamp | None = None,
    *,
    close_hour: int = 16,
    close_minute: int = 0,
    availability_buffer_minutes: int = 30,
) -> dt.date:
    """Return the EOD bar date the cache should have by `as_of`.

    Before 16:30 ET on a trading day, the latest full historical bar should be
    the prior NYSE session. At/after 16:30 ET, the current session is eligible
    for EOD refresh attempts.
    """
    eastern = ZoneInfo("America/New_York")
    if as_of is None:
        local = pd.Timestamp.now(tz=eastern)
    else:
        local = pd.Timestamp(as_of)
        if local.tzinfo is None:
            local = local.tz_localize(eastern)
        else:
            local = local.tz_convert(eastern)

    current_date = local.date()
    trading_days = expected_nyse_dates(current_date - dt.timedelta(days=10), current_date)
    close_ready = (
        dt.datetime.combine(current_date, dt.time(close_hour, close_minute), tzinfo=eastern)
        + dt.timedelta(minutes=availability_buffer_minutes)
    )
    if current_date in trading_days and local >= close_ready:
        return current_date
    return previous_nyse_trading_day(current_date)


def _read_price_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in PRICE_COLUMNS if c in df.columns]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[keep]


def _normalize_fmp_prices(data: Any) -> pd.DataFrame:
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty or "date" not in df:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in PRICE_COLUMNS if c in df.columns]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[cols]


def _default_fmp_get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    from src.data import bulk_download

    bulk_download.API_KEY = os.environ.get("FMP_API_KEY", bulk_download.API_KEY)
    return bulk_download.fmp_get(endpoint, params)


def fetch_fmp_eod_prices(
    symbol: str,
    start_date: dt.date,
    *,
    fmp_get: Callable[[str, dict[str, Any] | None], Any] | None = None,
) -> pd.DataFrame:
    getter = fmp_get or _default_fmp_get
    data = getter("stable/historical-price-eod/full", {"symbol": symbol, "from": start_date.isoformat()})
    return _normalize_fmp_prices(data)


def merge_price_history(existing: pd.DataFrame, fetched: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Merge fetched FMP rows and report whether the cached history changed."""
    if fetched.empty:
        return existing, False
    existing = existing.copy()
    fetched = fetched.copy()
    for col in PRICE_COLUMNS:
        if col not in existing:
            existing[col] = np.nan
        if col not in fetched:
            fetched[col] = np.nan
    existing = existing[PRICE_COLUMNS].sort_index()
    fetched = fetched[PRICE_COLUMNS].sort_index()
    merged = pd.concat([existing, fetched])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    fetched_index = fetched.index.unique()
    before = existing.reindex(fetched_index)[PRICE_COLUMNS]
    after = merged.reindex(fetched_index)[PRICE_COLUMNS]
    changed = not _numeric_price_frames_equal(before, after)
    return merged, changed


def _numeric_price_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Compare price rows after dtype normalization and tiny float tolerance."""
    left = left.reindex(columns=PRICE_COLUMNS)
    right = right.reindex(columns=PRICE_COLUMNS)
    if not left.index.equals(right.index) or list(left.columns) != list(right.columns):
        return False
    left_values = left.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right_values = right.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isclose(left_values, right_values, rtol=1e-10, atol=1e-8, equal_nan=True).all())


def _matrix_latest_date(cache_dir: Path) -> dt.date | None:
    close_path = cache_dir / "matrices" / "close.parquet"
    if not close_path.exists():
        return None
    close = pd.read_parquet(close_path)
    if close.empty:
        return None
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    return close.index.max().date()


def _metadata_tickers(cache_dir: Path) -> list[str]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tickers = metadata.get("tickers")
    if not isinstance(tickers, list):
        return []
    return sorted(str(t) for t in tickers if t)


def _metadata_start_date(cache_dir: Path) -> dt.date | None:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = metadata.get("start_date")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return pd.Timestamp(raw).date()
    except Exception:
        return None


def _matrix_symbols(cache_dir: Path) -> list[str]:
    metadata_tickers = _metadata_tickers(cache_dir)
    if metadata_tickers:
        return metadata_tickers
    market_cap_metric_path = cache_dir / "matrices" / "market_cap_metric.parquet"
    if market_cap_metric_path.exists():
        market_cap_metric = pd.read_parquet(market_cap_metric_path)
        return sorted(str(c) for c in market_cap_metric.columns)
    close_path = cache_dir / "matrices" / "close.parquet"
    if close_path.exists():
        close = pd.read_parquet(close_path)
        return sorted(str(c) for c in close.columns)
    price_dir = cache_dir / "prices"
    return sorted(p.stem for p in price_dir.glob("*.parquet"))


def _load_profiles(cache_dir: Path, symbols: list[str]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        path = cache_dir / "profiles" / f"{sym}.json"
        if not path.exists():
            continue
        try:
            profiles[sym] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return profiles


def _load_prices(cache_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = cache_dir / "prices" / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = _read_price_parquet(path)
        except Exception:
            continue
        if not df.empty:
            prices[sym] = df
    return prices


def _update_one_symbol(
    cache_dir: Path,
    symbol: str,
    *,
    expected_date: dt.date,
    overlap_days: int,
    fmp_get: Callable[[str, dict[str, Any] | None], Any] | None,
) -> dict[str, Any]:
    price_path = cache_dir / "prices" / f"{symbol}.parquet"
    try:
        existing = _read_price_parquet(price_path) if price_path.exists() else pd.DataFrame(columns=PRICE_COLUMNS)
        if existing.empty:
            start_date = expected_date - dt.timedelta(days=max(overlap_days, 30))
        else:
            start_date = min(existing.index.max().date(), expected_date) - dt.timedelta(days=overlap_days)
        fetched = fetch_fmp_eod_prices(symbol, start_date, fmp_get=fmp_get)
        if fetched.empty:
            return {"symbol": symbol, "status": "empty", "changed": False, "latest": None}
        merged, changed = merge_price_history(existing, fetched)
        if changed:
            price_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(price_path)
        latest = merged.index.max().date() if not merged.empty else None
        return {
            "symbol": symbol,
            "status": "updated" if changed else "unchanged",
            "changed": changed,
            "latest": latest.isoformat() if latest else None,
        }
    except Exception as exc:
        return {"symbol": symbol, "status": "failed", "changed": False, "latest": None, "error": str(exc)}


def build_market_cap_band_universes(cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Build MCAP_* production universes from the current market_cap matrix."""
    matrices_dir = cache_dir / "matrices"
    universes_dir = cache_dir / "universes"
    market_cap_path = matrices_dir / "market_cap.parquet"
    close_path = matrices_dir / "close.parquet"
    if not market_cap_path.exists() or not close_path.exists():
        return {}
    market_cap = pd.read_parquet(market_cap_path)
    close = pd.read_parquet(close_path)
    if not isinstance(market_cap.index, pd.DatetimeIndex):
        market_cap.index = pd.to_datetime(market_cap.index)
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    common_index = market_cap.index.intersection(close.index)
    columns = sorted(set(market_cap.columns) | set(close.columns))
    market_cap = market_cap.reindex(index=common_index, columns=columns)
    close = close.reindex(index=common_index, columns=columns)
    tradable = close.notna()

    universes_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in MARKET_CAP_BANDS.items():
        mask = (market_cap >= lo) & (market_cap < hi) & tradable
        mask = mask.fillna(False).astype(bool)
        mask.to_parquet(universes_dir / f"{name}.parquet")
        summaries[name] = {
            "rows": int(mask.shape[0]),
            "columns": int(mask.shape[1]),
            "last_members": int(mask.iloc[-1].sum()) if len(mask) else 0,
            "avg_members": float(mask.sum(axis=1).mean()) if len(mask) else 0.0,
        }
    return summaries


def rebuild_classification_matrices(cache_dir: Path) -> dict[str, Any]:
    """Re-encode sector/industry/subindustry from classifications.json.

    `bulk_download.build_matrices()` still has an old fallback that copies
    industry into subindustry. Production neutralization needs the repaired
    SIC-based hierarchy, so every EOD rebuild re-applies the authoritative
    classification JSON to the matrix parquets.
    """
    matrices_dir = cache_dir / "matrices"
    close = pd.read_parquet(matrices_dir / "close.parquet")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    tickers = [str(c) for c in close.columns]
    classifications = json.loads((cache_dir / "classifications.json").read_text(encoding="utf-8"))

    stats: dict[str, Any] = {"tickers": len(tickers)}
    labels_by_level: dict[str, dict[str, str]] = {}
    for level in ("sector", "industry", "subindustry"):
        labels = {
            ticker: str((classifications.get(ticker) or {}).get(level) or "Unknown")
            for ticker in tickers
        }
        labels_by_level[level] = labels
        values = sorted(set(labels.values()))
        encoder = {label: i for i, label in enumerate(values)}
        row = pd.Series({ticker: encoder[labels[ticker]] for ticker in tickers}, dtype="int32")
        mat = pd.DataFrame([row.to_numpy()] * len(close.index), index=close.index, columns=tickers, dtype="int32")
        mat.to_parquet(matrices_dir / f"{level}.parquet")
        decoder = {str(i): label for label, i in encoder.items()}
        (cache_dir / f"{level}_decoder.json").write_text(json.dumps(decoder, indent=2, sort_keys=True), encoding="utf-8")
        stats[f"{level}_groups"] = len(values)

    pd.Series(labels_by_level["sector"], name="sector").to_frame().to_parquet(
        matrices_dir / "_sector_groups.parquet"
    )
    pd.Series(labels_by_level["industry"], name="industry").to_frame().to_parquet(
        matrices_dir / "_industry_groups.parquet"
    )
    same = sum(
        1
        for ticker in tickers
        if labels_by_level["industry"][ticker] == labels_by_level["subindustry"][ticker]
    )
    stats["industry_subindustry_same_pct"] = same / max(len(tickers), 1) * 100.0
    return stats


def _concat_price_matrix(
    prices: dict[str, pd.DataFrame],
    tickers: list[str],
    all_dates: pd.DatetimeIndex,
    field: str,
) -> pd.DataFrame:
    pieces = {
        sym: pd.to_numeric(df[field], errors="coerce")
        for sym, df in prices.items()
        if field in df.columns
    }
    if not pieces:
        return pd.DataFrame(index=all_dates, columns=tickers, dtype=float)
    mat = pd.concat(pieces, axis=1)
    if not isinstance(mat.index, pd.DatetimeIndex):
        mat.index = pd.to_datetime(mat.index)
    mat = mat.sort_index()
    if mat.index.duplicated().any():
        mat = mat[~mat.index.duplicated(keep="last")]
    return mat.reindex(index=all_dates, columns=tickers).astype(float)


def _read_existing_matrix(path: Path, all_dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df.reindex(index=all_dates, columns=tickers).ffill()


def _matrix_matches_shape(path: Path, all_dates: pd.DatetimeIndex, tickers: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if not isinstance(df, pd.DataFrame):
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    return df.index.equals(all_dates) and list(df.columns) == tickers


def _forward_fill_existing_matrices(
    cache_dir: Path,
    all_dates: pd.DatetimeIndex,
    tickers: list[str],
    *,
    skip_fields: set[str],
) -> dict[str, Any]:
    matrices_dir = cache_dir / "matrices"
    rewritten: list[str] = []
    for path in sorted(matrices_dir.glob("*.parquet")):
        name = path.stem
        if name in skip_fields or name.startswith("_"):
            continue
        aligned = _read_existing_matrix(path, all_dates, tickers)
        if aligned is None:
            continue
        aligned.to_parquet(path)
        rewritten.append(name)
    return {"fields_rewritten": len(rewritten), "field_sample": rewritten[:30]}


def _build_fast_price_matrices(
    cache_dir: Path,
    prices: dict[str, pd.DataFrame],
    symbols: list[str],
) -> dict[str, Any]:
    tickers = sorted(sym for sym in symbols if sym in prices)
    if not tickers:
        raise RuntimeError(f"No cached prices available under {cache_dir / 'prices'}")
    latest_price_date = max(pd.to_datetime(prices[sym].index).max().date() for sym in tickers)
    first_price_date = min(pd.to_datetime(prices[sym].index).min().date() for sym in tickers)
    start_date = _metadata_start_date(cache_dir) or first_price_date
    all_dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in expected_nyse_dates(start_date, latest_price_date)))
    matrices_dir = cache_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    shape_changed = not _matrix_matches_shape(matrices_dir / "close.parquet", all_dates, tickers)

    matrices: dict[str, pd.DataFrame] = {}
    for field in PRICE_MATRIX_FIELDS:
        matrices[field] = _concat_price_matrix(prices, tickers, all_dates, field)

    close = matrices["close"]
    volume = matrices["volume"]
    high = matrices.get("high")
    low = matrices.get("low")
    matrices["returns"] = close.pct_change()
    matrices["dollars_traded"] = close * volume
    matrices["adv20"] = matrices["dollars_traded"].rolling(20).mean()
    matrices["adv60"] = matrices["dollars_traded"].rolling(60).mean()
    ratio = close / close.shift(1)
    matrices["log_returns"] = np.log(ratio.where(ratio > 0))

    shares_out = _read_existing_matrix(matrices_dir / "shares_out.parquet", all_dates, tickers)
    market_cap_metric = _read_existing_matrix(matrices_dir / "market_cap_metric.parquet", all_dates, tickers)
    market_cap_existing = _read_existing_matrix(matrices_dir / "market_cap.parquet", all_dates, tickers)
    if market_cap_metric is not None:
        market_cap = market_cap_metric
    elif market_cap_existing is not None:
        market_cap = market_cap_existing
    elif shares_out is not None:
        market_cap = close * shares_out
    else:
        market_cap = close
    matrices["market_cap"] = market_cap
    matrices["cap"] = market_cap

    for window in VOLATILITY_WINDOWS:
        matrices[f"historical_volatility_{window}"] = matrices["returns"].rolling(window).std() * np.sqrt(252)
        if high is not None and low is not None:
            ratio_hl = np.log(high / low.replace(0, np.nan)) ** 2
            matrices[f"parkinson_volatility_{window}"] = np.sqrt(
                ratio_hl.rolling(window).mean() / (4 * np.log(2))
            )

    for name, mat in matrices.items():
        mat.to_parquet(matrices_dir / f"{name}.parquet")

    skip_fields = set(matrices) | {"sector", "industry", "subindustry"}
    ffill_summary = (
        _forward_fill_existing_matrices(cache_dir, all_dates, tickers, skip_fields=skip_fields)
        if shape_changed
        else {"fields_rewritten": 0, "field_sample": []}
    )
    return {
        "symbols_loaded": len(tickers),
        "matrix_rows": int(close.shape[0]),
        "matrix_cols": int(close.shape[1]),
        "matrix_end": str(pd.Timestamp(close.index.max()).date()),
        "price_fields_rebuilt": sorted(matrices),
        "forward_fill_summary": ffill_summary,
    }


def _update_metadata(cache_dir: Path, summary: dict[str, Any], fields: list[str]) -> None:
    metadata_path = cache_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata.update({
        "tickers": _metadata_tickers(cache_dir) or _matrix_symbols(cache_dir),
        "n_tickers": int(summary["matrix_cols"]),
        "n_days": int(summary["matrix_rows"]),
        "end_date": summary["matrix_end"],
    })
    if "start_date" not in metadata:
        close = pd.read_parquet(cache_dir / "matrices" / "close.parquet")
        metadata["start_date"] = str(pd.Timestamp(close.index.min()))
    metadata["fields"] = sorted(set(metadata.get("fields", [])) | set(fields))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")


def rebuild_equity_cache_from_prices(cache_dir: Path, symbols: list[str]) -> dict[str, Any]:
    """Incrementally rebuild matrices and universes from cached prices.

    EOD refreshes should not replay every fundamental file. Price-derived
    matrices are rebuilt from the canonical price parquet cache, while existing
    fundamental/static matrices are forward-filled onto new dates when needed.
    """
    from src.data import bulk_download

    canonical_symbols = _metadata_tickers(cache_dir) or symbols
    prices = _load_prices(cache_dir, canonical_symbols)
    matrix_summary = _build_fast_price_matrices(cache_dir, prices, canonical_symbols)
    old_cache_dir = bulk_download.CACHE_DIR
    try:
        bulk_download.CACHE_DIR = cache_dir
        bulk_download.validate_no_calendar_holes(cache_dir / "matrices" / "close.parquet")
    finally:
        bulk_download.CACHE_DIR = old_cache_dir
    classification_summary = rebuild_classification_matrices(cache_dir)
    band_summaries = build_market_cap_band_universes(cache_dir)
    _update_metadata(cache_dir, matrix_summary, matrix_summary["price_fields_rebuilt"])
    return {
        **matrix_summary,
        "classification_summary": classification_summary,
        "mcap_universes": band_summaries,
    }


def _active_universe_coverage(cache_dir: Path, universe_name: str, expected_date: dt.date) -> dict[str, Any]:
    close_path = cache_dir / "matrices" / "close.parquet"
    universe_path = cache_dir / "universes" / f"{universe_name}.parquet"
    if not close_path.exists() or not universe_path.exists():
        return {"coverage": 0.0, "active": 0, "missing": [], "reason": "missing close or universe parquet"}
    matrices = {}
    for field in ["open", "high", "low", "close", "volume"]:
        path = cache_dir / "matrices" / f"{field}.parquet"
        if not path.exists():
            return {"coverage": 0.0, "active": 0, "missing": [], "reason": f"missing {field}.parquet"}
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        matrices[field] = df
    universe = pd.read_parquet(universe_path)
    if not isinstance(universe.index, pd.DatetimeIndex):
        universe.index = pd.to_datetime(universe.index)
    ts = pd.Timestamp(expected_date)
    if ts not in universe.index or ts not in matrices["close"].index:
        return {"coverage": 0.0, "active": 0, "missing": [], "reason": f"{expected_date} not in cache"}
    active = universe.loc[ts].fillna(False).astype(bool)
    active_symbols = [str(s) for s, is_active in active.items() if bool(is_active)]
    if not active_symbols:
        return {"coverage": 0.0, "active": 0, "missing": [], "reason": "no active symbols"}
    ok = pd.Series(True, index=active_symbols)
    for field, df in matrices.items():
        ok &= df.reindex(index=[ts], columns=active_symbols).iloc[0].notna()
    missing = sorted(ok.index[~ok].tolist())
    return {
        "coverage": float(ok.mean()),
        "active": len(active_symbols),
        "missing": missing[:50],
        "missing_count": len(missing),
        "reason": "",
    }


def refresh_equity_eod_cache(
    root: Path = PROJECT_ROOT,
    *,
    expected_date: dt.date | str | None = None,
    universe_name: str = "MCAP_100M_500M",
    symbols: list[str] | None = None,
    recheck_recent: bool = False,
    overlap_days: int = 7,
    max_workers: int = 5,
    min_active_coverage: float = 0.99,
    fmp_get: Callable[[str, dict[str, Any] | None], Any] | None = None,
) -> dict[str, Any]:
    """Refresh the equity EOD cache and return a structured status payload."""
    root = Path(root)
    cache_dir = root / "data" / "fmp_cache"
    if expected_date is None:
        target_date = latest_closed_nyse_session()
    else:
        target_date = pd.Timestamp(expected_date).date()

    t0 = time.time()
    matrix_before = _matrix_latest_date(cache_dir)
    selected_symbols = symbols or _matrix_symbols(cache_dir)
    if not selected_symbols:
        raise RuntimeError("No equity symbols found for EOD refresh")

    needs_fetch = recheck_recent or matrix_before is None or matrix_before < target_date
    update_rows: list[dict[str, Any]] = []
    if needs_fetch:
        log.info(
            "Refreshing FMP EOD prices: target=%s matrix_before=%s symbols=%d recheck_recent=%s",
            target_date,
            matrix_before,
            len(selected_symbols),
            recheck_recent,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _update_one_symbol,
                    cache_dir,
                    sym,
                    expected_date=target_date,
                    overlap_days=overlap_days,
                    fmp_get=fmp_get,
                )
                for sym in selected_symbols
            ]
            for fut in as_completed(futures):
                update_rows.append(fut.result())

    changed_symbols = sorted(row["symbol"] for row in update_rows if row.get("changed"))
    failed_symbols = sorted(row["symbol"] for row in update_rows if row.get("status") == "failed")
    latest_dates = [
        pd.Timestamp(row["latest"]).date()
        for row in update_rows
        if row.get("latest")
    ]
    latest_price_date = max(latest_dates) if latest_dates else matrix_before
    should_rebuild = bool(changed_symbols) or (
        matrix_before is not None and latest_price_date is not None and latest_price_date > matrix_before
    ) or (matrix_before is None and latest_price_date is not None)

    rebuild_summary: dict[str, Any] | None = None
    if should_rebuild:
        rebuild_summary = rebuild_equity_cache_from_prices(cache_dir, selected_symbols)

    matrix_after = _matrix_latest_date(cache_dir)
    coverage = _active_universe_coverage(cache_dir, universe_name, target_date)
    loaded_expected = matrix_after is not None and matrix_after >= target_date
    coverage_ok = float(coverage.get("coverage") or 0.0) >= min_active_coverage

    if loaded_expected and coverage_ok:
        status = "completed" if should_rebuild else "up_to_date"
        message = f"Equity EOD cache is fresh through {matrix_after}."
    elif not loaded_expected:
        status = "waiting_for_vendor"
        message = f"FMP EOD target {target_date} is not loaded yet; latest matrix date is {matrix_after}."
    else:
        status = "incomplete_coverage"
        message = (
            f"Equity EOD target {target_date} loaded but active coverage "
            f"{float(coverage.get('coverage') or 0.0):.3f} is below {min_active_coverage:.3f}."
        )

    return {
        "status": status,
        "message": message,
        "expected_bar_date": target_date.isoformat(),
        "matrix_end_before": matrix_before.isoformat() if matrix_before else None,
        "matrix_end_after": matrix_after.isoformat() if matrix_after else None,
        "latest_price_date_seen": latest_price_date.isoformat() if latest_price_date else None,
        "symbols_checked": len(selected_symbols) if needs_fetch else 0,
        "symbols_updated": len(changed_symbols),
        "symbols_failed": len(failed_symbols),
        "changed_symbols_sample": changed_symbols[:50],
        "failed_symbols_sample": failed_symbols[:50],
        "rebuilt": should_rebuild,
        "rebuild_summary": rebuild_summary,
        "active_coverage": coverage,
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Refresh FMP equity EOD cache.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-date", default=None)
    parser.add_argument("--universe", default="MCAP_100M_500M")
    parser.add_argument("--recheck-recent", action="store_true")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--min-active-coverage", type=float, default=0.99)
    parser.add_argument("--fail-if-incomplete", action="store_true")
    args = parser.parse_args()

    result = refresh_equity_eod_cache(
        args.root,
        expected_date=args.expected_date,
        universe_name=args.universe,
        recheck_recent=args.recheck_recent,
        overlap_days=args.overlap_days,
        max_workers=args.workers,
        min_active_coverage=args.min_active_coverage,
    )
    print(json.dumps(result, indent=2, default=str))
    if args.fail_if_incomplete and result["status"] not in {"completed", "up_to_date"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
