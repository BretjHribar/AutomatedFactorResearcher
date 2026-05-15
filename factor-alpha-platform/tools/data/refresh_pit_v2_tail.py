"""Refresh the tail of the PIT v2 equity matrix set used by AIPT.

This is intentionally an incremental tail refresher. The historical PIT v2
matrix set contains delisted names and older history that the production FMP
matrix set does not carry, so a full rebuild from production matrices would
silently drop research history. Instead, this script:

1. Reads the existing PIT v2 matrices through their current last date.
2. Loads current per-symbol FMP price parquets for the missing trading days.
3. Appends only the missing rows.
4. Recomputes daily price/liquidity fields on a lookback window and forward
   fills PIT fundamental fields onto the new dates.
5. Extends universes_pit/membership using same-day finite close.

It does not fetch from FMP by itself. Run the normal equity EOD refresh first.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIT_DIR = ROOT / "data" / "fmp_cache" / "matrices_pit_v2"
PRICES_DIR = ROOT / "data" / "fmp_cache" / "prices"
MEMBERSHIP_PATH = ROOT / "data" / "fmp_cache" / "universes_pit" / "membership.parquet"

PRICE_FIELDS = ["open", "high", "low", "close", "volume", "vwap"]
VOL_WINDOWS = [10, 20, 30, 60, 90, 120, 150, 180]
MOM_WINDOWS = [5, 20, 60, 120, 252]

RECOMPUTED_FIELDS = set(PRICE_FIELDS) | {
    "returns",
    "log_returns",
    "dollars_traded",
    "adv20",
    "adv60",
    "high_low_range",
    "open_close_range",
    "close_position_in_range",
    "vwap_deviation",
    "turnover",
    "volume_ratio_20d",
    "volume_momentum_1",
    "volume_momentum_5_20",
    "cap",
    "market_cap",
    "enterprise_value",
    "earnings_yield",
    "book_to_market",
    "free_cashflow_yield",
    "fcf_yield_metric",
    "fcf_per_share",
    "ev_to_ebitda",
    "ev_to_revenue",
    "ev_to_sales",
    "roe",
    "roa",
    "return_equity",
    "return_assets",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "asset_turnover",
    "debt_to_equity",
    "debt_to_assets",
    "current_ratio",
    "turnover",
}
RECOMPUTED_FIELDS |= {f"historical_volatility_{w}" for w in VOL_WINDOWS}
RECOMPUTED_FIELDS |= {f"parkinson_volatility_{w}" for w in VOL_WINDOWS}
RECOMPUTED_FIELDS |= {f"momentum_{w}d" for w in MOM_WINDOWS}


def _read_matrix(name: str) -> pd.DataFrame | None:
    path = PIT_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def _price_symbols() -> list[str]:
    return sorted(path.stem for path in PRICES_DIR.glob("*.parquet"))


def _latest_price_date(symbols: list[str]) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for sym in symbols:
        path = PRICES_DIR / f"{sym}.parquet"
        try:
            df = pd.read_parquet(path, columns=["close"])
        except Exception:
            continue
        if df.empty:
            continue
        idx = pd.to_datetime(df.index)
        ts = pd.Timestamp(idx.max()).normalize()
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        raise RuntimeError(f"No readable price parquets in {PRICES_DIR}")
    return latest


def _load_price_window(symbols: list[str], dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    frames = {
        field: pd.DataFrame(np.nan, index=dates, columns=symbols, dtype="float64")
        for field in PRICE_FIELDS
    }
    wanted = set(dates)
    for i, sym in enumerate(symbols, start=1):
        path = PRICES_DIR / f"{sym}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[df.index.intersection(wanted)]
        if df.empty:
            continue
        for field in PRICE_FIELDS:
            if field in df:
                frames[field].loc[df.index, sym] = pd.to_numeric(df[field], errors="coerce")
        if i % 1000 == 0:
            print(f"  loaded prices for {i}/{len(symbols)} symbols", flush=True)
    return frames


def _matrix_tail(name: str, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    df = _read_matrix(name)
    if df is None or df.empty:
        return pd.DataFrame(np.nan, index=dates, columns=symbols, dtype="float64")
    return df.reindex(index=dates, columns=symbols).ffill()


def _build_outputs(calc_dates: pd.DatetimeIndex, symbols: list[str]) -> dict[str, pd.DataFrame]:
    print(f"[1/4] Loading price window {calc_dates[0].date()} -> {calc_dates[-1].date()} for {len(symbols)} symbols")
    raw_prices = _load_price_window(symbols, calc_dates)
    outputs: dict[str, pd.DataFrame] = {}
    for field in PRICE_FIELDS:
        existing = _matrix_tail(field, calc_dates, symbols)
        # Preserve existing PIT history on overlap dates; use raw prices to fill
        # the new dates and any missing overlap values.
        outputs[field] = existing.combine_first(raw_prices[field])

    close = outputs["close"]
    open_ = outputs["open"]
    high = outputs["high"]
    low = outputs["low"]
    volume = outputs["volume"]
    vwap = outputs["vwap"]

    outputs["returns"] = close.pct_change(fill_method=None)
    ratio = close / close.shift(1)
    outputs["log_returns"] = np.log(ratio.where(ratio > 0))
    outputs["dollars_traded"] = close * volume
    outputs["adv20"] = outputs["dollars_traded"].rolling(20, min_periods=10).mean()
    outputs["adv60"] = outputs["dollars_traded"].rolling(60, min_periods=20).mean()
    outputs["high_low_range"] = _safe_div(high - low, close)
    outputs["open_close_range"] = _safe_div(close - open_, open_)
    outputs["close_position_in_range"] = _safe_div(close - low, high - low)
    outputs["vwap_deviation"] = _safe_div(close - vwap, close)
    outputs["volume_ratio_20d"] = _safe_div(volume, volume.rolling(20, min_periods=10).mean())
    outputs["volume_momentum_1"] = volume.pct_change(fill_method=None)
    outputs["volume_momentum_5_20"] = _safe_div(
        volume.rolling(5, min_periods=2).mean(),
        volume.rolling(20, min_periods=10).mean(),
    ) - 1.0

    for window in MOM_WINDOWS:
        outputs[f"momentum_{window}d"] = close.pct_change(window, fill_method=None)
    for window in VOL_WINDOWS:
        outputs[f"historical_volatility_{window}"] = outputs["log_returns"].rolling(
            window, min_periods=max(5, window // 4)
        ).std() * math.sqrt(252)
        log_hl = np.log(high / low.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        pk = (log_hl ** 2) / (4 * math.log(2))
        outputs[f"parkinson_volatility_{window}"] = np.sqrt(
            pk.rolling(window, min_periods=max(5, window // 4)).mean()
        ) * math.sqrt(252)

    fund_names = {
        "shares_out",
        "total_debt",
        "cash",
        "total_equity",
        "eps",
        "free_cashflow",
        "ebitda",
        "revenue",
        "net_income",
        "total_assets",
        "gross_profit",
        "operating_income",
        "total_current_assets",
        "total_current_liabilities",
        "capex",
        "rd_expense",
        "assets",
        "debt",
    }
    fund = {name: _matrix_tail(name, calc_dates, symbols) for name in fund_names}

    market_cap = close * fund["shares_out"]
    prev_market_cap = _matrix_tail("market_cap", calc_dates, symbols)
    market_cap = market_cap.combine_first(prev_market_cap)
    outputs["market_cap"] = market_cap
    outputs["cap"] = market_cap
    outputs["enterprise_value"] = market_cap.add(fund["total_debt"], fill_value=np.nan).sub(
        fund["cash"], fill_value=np.nan
    )
    outputs["earnings_yield"] = _safe_div(fund["eps"], close)
    outputs["book_to_market"] = _safe_div(fund["total_equity"], market_cap)
    outputs["free_cashflow_yield"] = _safe_div(fund["free_cashflow"], market_cap)
    outputs["fcf_yield_metric"] = outputs["free_cashflow_yield"]
    outputs["fcf_per_share"] = _safe_div(fund["free_cashflow"], fund["shares_out"])
    outputs["ev_to_ebitda"] = _safe_div(outputs["enterprise_value"], fund["ebitda"])
    outputs["ev_to_revenue"] = _safe_div(outputs["enterprise_value"], fund["revenue"])
    outputs["ev_to_sales"] = outputs["ev_to_revenue"]
    outputs["roe"] = _safe_div(fund["net_income"], fund["total_equity"])
    outputs["roa"] = _safe_div(fund["net_income"], fund["total_assets"])
    outputs["return_equity"] = outputs["roe"]
    outputs["return_assets"] = outputs["roa"]
    outputs["gross_margin"] = _safe_div(fund["gross_profit"], fund["revenue"])
    outputs["operating_margin"] = _safe_div(fund["operating_income"], fund["revenue"])
    outputs["net_margin"] = _safe_div(fund["net_income"], fund["revenue"])
    outputs["asset_turnover"] = _safe_div(fund["revenue"], fund["total_assets"])
    outputs["debt_to_equity"] = _safe_div(fund["total_debt"], fund["total_equity"])
    outputs["debt_to_assets"] = _safe_div(fund["total_debt"], fund["total_assets"])
    outputs["current_ratio"] = _safe_div(fund["total_current_assets"], fund["total_current_liabilities"])
    outputs["turnover"] = _safe_div(outputs["dollars_traded"], market_cap)

    # Keep direct fundamental aliases fresh too when those files exist.
    outputs["capex"] = fund["capex"]
    outputs["rd_expense"] = fund["rd_expense"]
    outputs["assets"] = fund["assets"].combine_first(fund["total_assets"])
    outputs["debt"] = fund["debt"].combine_first(fund["total_debt"])
    return outputs


def _write_matrix(name: str, tail: pd.DataFrame, new_dates: pd.DatetimeIndex, symbols: list[str], *, dry_run: bool) -> dict[str, Any]:
    path = PIT_DIR / f"{name}.parquet"
    old = _read_matrix(name)
    if old is None:
        old = pd.DataFrame(index=pd.DatetimeIndex([]), columns=symbols)
    cols = sorted(set(old.columns) | set(symbols))
    old = old.reindex(columns=cols)
    tail = tail.reindex(index=new_dates, columns=cols)
    combined = pd.concat([old.loc[old.index < new_dates[0]], tail]).sort_index()
    coverage = float(combined.tail(1).notna().mean(axis=1).iloc[0]) if not combined.empty else 0.0
    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        combined.to_parquet(tmp)
        os.replace(tmp, path)
    return {
        "field": name,
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "last": str(combined.index[-1]) if len(combined.index) else None,
        "last_coverage": coverage,
    }


def _write_generic_ffill(name: str, calc_dates: pd.DatetimeIndex, new_dates: pd.DatetimeIndex, symbols: list[str], *, dry_run: bool) -> dict[str, Any]:
    tail = _matrix_tail(name, calc_dates, symbols).loc[new_dates]
    return _write_matrix(name, tail, new_dates, symbols, dry_run=dry_run)


def _refresh_membership(new_dates: pd.DatetimeIndex, close_tail: pd.DataFrame, symbols: list[str], *, dry_run: bool) -> dict[str, Any]:
    if MEMBERSHIP_PATH.exists():
        membership = pd.read_parquet(MEMBERSHIP_PATH)
        if not isinstance(membership.index, pd.DatetimeIndex):
            membership.index = pd.to_datetime(membership.index)
        membership = membership.sort_index()
    else:
        membership = pd.DataFrame(index=pd.DatetimeIndex([]), columns=symbols, dtype=bool)
    cols = sorted(set(membership.columns) | set(symbols))
    membership = membership.reindex(columns=cols).fillna(False).astype(bool)
    tail = close_tail.reindex(index=new_dates, columns=cols).notna() & close_tail.reindex(index=new_dates, columns=cols).gt(0)
    combined = pd.concat([membership.loc[membership.index < new_dates[0]], tail]).sort_index()
    if not dry_run:
        MEMBERSHIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MEMBERSHIP_PATH.with_suffix(MEMBERSHIP_PATH.suffix + ".tmp")
        combined.to_parquet(tmp)
        os.replace(tmp, MEMBERSHIP_PATH)
    return {
        "path": str(MEMBERSHIP_PATH.relative_to(ROOT)),
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "last": str(combined.index[-1]) if len(combined.index) else None,
        "last_members": int(combined.tail(1).sum(axis=1).iloc[0]) if not combined.empty else 0,
    }


def refresh(*, lookback_rows: int = 320, dry_run: bool = False) -> dict[str, Any]:
    t0 = time.time()
    close = _read_matrix("close")
    if close is None or close.empty:
        raise RuntimeError(f"{PIT_DIR / 'close.parquet'} missing or empty")
    existing_last = pd.Timestamp(close.index[-1]).normalize()
    price_symbols = _price_symbols()
    target_last = _latest_price_date(price_symbols)
    if target_last <= existing_last:
        return {
            "status": "up_to_date",
            "existing_last": str(existing_last),
            "target_last": str(target_last),
            "elapsed_sec": time.time() - t0,
        }

    new_dates = pd.DatetimeIndex([d for d in close.index.union(pd.date_range(existing_last, target_last, freq="B")) if existing_last < d <= target_last])
    if new_dates.empty:
        return {
            "status": "up_to_date",
            "existing_last": str(existing_last),
            "target_last": str(target_last),
            "elapsed_sec": time.time() - t0,
        }
    lookback_start_pos = max(0, len(close.index) - lookback_rows)
    calc_dates = pd.DatetimeIndex(close.index[lookback_start_pos:]).union(new_dates).sort_values()
    symbols = sorted(set(close.columns) | set(price_symbols))
    outputs = _build_outputs(calc_dates, symbols)

    print(f"[2/4] Writing {len(RECOMPUTED_FIELDS)} recomputed fields for {len(new_dates)} new dates")
    written = []
    for name in sorted(RECOMPUTED_FIELDS):
        if name in outputs:
            written.append(_write_matrix(name, outputs[name].loc[new_dates], new_dates, symbols, dry_run=dry_run))

    print("[3/4] Forward-filling remaining PIT fields")
    for path in sorted(PIT_DIR.glob("*.parquet")):
        name = path.stem
        if name.startswith("_") or name in RECOMPUTED_FIELDS:
            continue
        written.append(_write_generic_ffill(name, calc_dates, new_dates, symbols, dry_run=dry_run))

    print("[4/4] Refreshing PIT membership")
    membership = _refresh_membership(new_dates, outputs["close"].loc[new_dates], symbols, dry_run=dry_run)
    manifest = {
        "status": "dry_run" if dry_run else "completed",
        "started_existing_last": str(existing_last),
        "target_last": str(target_last),
        "new_dates": [str(d.date()) for d in new_dates],
        "n_symbols": len(symbols),
        "n_fields_written": len(written),
        "field_sample": written[:10],
        "membership": membership,
        "elapsed_sec": time.time() - t0,
        "generated_at": datetime.now().isoformat(),
    }
    if not dry_run:
        (PIT_DIR / "_tail_refresh_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Append current FMP data to PIT v2 matrices.")
    parser.add_argument("--lookback-rows", type=int, default=320)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(refresh(lookback_rows=args.lookback_rows, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
