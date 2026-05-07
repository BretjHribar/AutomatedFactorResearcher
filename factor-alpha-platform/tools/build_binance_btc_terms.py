"""Build Binance 4h research matrices denominated in BTC terms.

The Binance futures cache is USDT-quoted. This script creates an isolated
matrix directory where point prices are expressed as symbol/BTC at the same
4h bar and USDT quote-notional fields are expressed as BTC notional.

The generated fields intentionally follow the existing Binance matrix
semantics: 5/20/60 bar momentum names, 20/60 bar ADV, and sqrt(252) volatility
annualization. That keeps converted-data discovery comparable to prior
Binance research runs.

Notes on OHLC:
    Open and close can be converted exactly at the 4h bar endpoints using
    same-bar BTC open/close. A true BTC-relative high/low requires synchronized
    intrabar data. The local Binance 5m cache starts much later than the 4h
    history, so the default "range_envelope" mode uses high/BTC_low and
    low/BTC_high to produce a conservative, non-inverted range envelope.

Example:
    python tools/build_binance_btc_terms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

PASSTHROUGH_FIELDS = (
    "funding_rate",
    "funding_rate_cumsum_3",
    "funding_rate_avg_7d",
    "funding_rate_zscore",
)


@dataclass(frozen=True)
class BinanceBtcTermsSummary:
    source_klines_dir: str
    source_matrices_dir: str
    output_matrices_dir: str
    source_universe_path: str
    output_universe_path: str
    btc_symbol: str
    ohlc_mode: str
    drop_btc_symbol: bool
    n_symbols_loaded: int
    n_symbols_written: int
    n_bars: int
    start: str
    end: str
    high_low_inversions_after_sanitize: int
    passthrough_fields_written: list[str]
    fields_written: list[str]
    caveat: str


def _read_binance_klines(klines_dir: Path, min_rows: int) -> dict[str, pd.DataFrame]:
    all_data: dict[str, pd.DataFrame] = {}
    required = ("open", "high", "low", "close", "volume", "quote_volume")
    optional = ("trades_count", "taker_buy_volume", "taker_buy_quote_volume")
    for fpath in sorted(klines_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fpath)
            if df.empty or len(df) < min_rows:
                continue
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            elif "time" in df.columns:
                df = df.set_index("time")
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if not set(required).issubset(df.columns):
                continue
            cols = [c for c in required + optional if c in df.columns]
            all_data[fpath.stem] = df[cols].apply(pd.to_numeric, errors="coerce")
        except Exception:
            continue
    return all_data


def _matrix(all_data: Mapping[str, pd.DataFrame], field: str, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {sym: df[field] for sym, df in all_data.items() if field in df.columns},
        index=index,
    )


def _df_fmax(*frames: pd.DataFrame) -> pd.DataFrame:
    values = frames[0].to_numpy(dtype=float, copy=True)
    for frame in frames[1:]:
        values = np.fmax(values, frame.to_numpy(dtype=float, copy=False))
    return pd.DataFrame(values, index=frames[0].index, columns=frames[0].columns)


def _df_fmin(*frames: pd.DataFrame) -> pd.DataFrame:
    values = frames[0].to_numpy(dtype=float, copy=True)
    for frame in frames[1:]:
        values = np.fmin(values, frame.to_numpy(dtype=float, copy=False))
    return pd.DataFrame(values, index=frames[0].index, columns=frames[0].columns)


def _div_by_series(df: pd.DataFrame, denom: pd.Series) -> pd.DataFrame:
    return df.div(denom.replace(0, np.nan), axis=0)


def _rolling_beta_to_btc(returns_btc_terms: pd.DataFrame, btc_usdt_close: pd.Series) -> pd.DataFrame:
    btc_usdt_ret = btc_usdt_close.pct_change(fill_method=None)
    btc_var = btc_usdt_ret.rolling(60, min_periods=30).var()
    betas: dict[str, pd.Series] = {}
    for col in returns_btc_terms.columns:
        cov = returns_btc_terms[col].rolling(60, min_periods=30).cov(btc_usdt_ret)
        betas[col] = cov / btc_var
    return pd.DataFrame(betas, index=returns_btc_terms.index)


def _copy_passthrough_fields(
    matrices: dict[str, pd.DataFrame],
    source_matrices_dir: Path,
    *,
    index: pd.Index,
    columns: pd.Index,
    btc_symbol: str,
) -> list[str]:
    written: list[str] = []
    for name in PASSTHROUGH_FIELDS:
        fpath = source_matrices_dir / f"{name}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        if btc_symbol in df.columns:
            df = df.drop(columns=[btc_symbol])
        cols = [c for c in columns if c in df.columns]
        if not cols:
            continue
        matrices[name] = df.reindex(index=index, columns=cols).reindex(columns=columns)
        written.append(name)
    return written


def build_btc_terms_matrices(
    all_data: Mapping[str, pd.DataFrame],
    *,
    btc_symbol: str = "BTCUSDT",
    ohlc_mode: str = "range_envelope",
    drop_btc_symbol: bool = True,
    source_matrices_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Convert in-memory Binance kline data into BTC-denominated matrices."""
    if btc_symbol not in all_data:
        raise ValueError(f"{btc_symbol} not found in input klines")
    if ohlc_mode not in {"range_envelope", "field_sync", "close_anchor"}:
        raise ValueError("ohlc_mode must be one of: range_envelope, field_sync, close_anchor")

    all_idx: set[pd.Timestamp] = set()
    for df in all_data.values():
        all_idx.update(pd.to_datetime(df.index))
    index = pd.DatetimeIndex(sorted(all_idx))

    open_usdt = _matrix(all_data, "open", index)
    high_usdt = _matrix(all_data, "high", index)
    low_usdt = _matrix(all_data, "low", index)
    close_usdt = _matrix(all_data, "close", index)
    volume = _matrix(all_data, "volume", index)
    quote_volume_usdt = _matrix(all_data, "quote_volume", index)
    trades_count = _matrix(all_data, "trades_count", index)
    taker_buy_volume = _matrix(all_data, "taker_buy_volume", index)
    taker_buy_quote_volume_usdt = _matrix(all_data, "taker_buy_quote_volume", index)

    btc_open = open_usdt[btc_symbol]
    btc_high = high_usdt[btc_symbol]
    btc_low = low_usdt[btc_symbol]
    btc_close = close_usdt[btc_symbol]
    btc_typical = ((btc_high + btc_low + btc_close) / 3.0).replace(0, np.nan)

    open_btc = _div_by_series(open_usdt, btc_open)
    close_btc = _div_by_series(close_usdt, btc_close)

    if ohlc_mode == "range_envelope":
        high_raw = _div_by_series(high_usdt, btc_low)
        low_raw = _div_by_series(low_usdt, btc_high)
    elif ohlc_mode == "field_sync":
        high_raw = _div_by_series(high_usdt, btc_high)
        low_raw = _div_by_series(low_usdt, btc_low)
    else:
        high_raw = _div_by_series(high_usdt, btc_close)
        low_raw = _div_by_series(low_usdt, btc_close)

    high_btc = _df_fmax(high_raw, low_raw, open_btc, close_btc)
    low_btc = _df_fmin(high_raw, low_raw, open_btc, close_btc)
    inversions = int(((high_btc < low_btc) & high_btc.notna() & low_btc.notna()).sum().sum())

    quote_volume_btc = _div_by_series(quote_volume_usdt, btc_typical)
    taker_buy_quote_volume_btc = _div_by_series(taker_buy_quote_volume_usdt, btc_typical)
    returns = close_btc.pct_change(fill_method=None)
    log_returns = np.log(close_btc / close_btc.shift(1))
    safe_volume = volume.replace(0, np.nan)
    vwap = quote_volume_btc / safe_volume
    safe_close = close_btc.replace(0, np.nan)
    high_low_abs = high_btc - low_btc
    safe_hl = high_low_abs.replace(0, np.nan)
    max_oc = _df_fmax(open_btc, close_btc)
    min_oc = _df_fmin(open_btc, close_btc)

    matrices: dict[str, pd.DataFrame] = {
        "open": open_btc,
        "high": high_btc,
        "low": low_btc,
        "close": close_btc,
        "volume": volume,
        "quote_volume": quote_volume_btc,
        "dollars_traded": quote_volume_btc,
        "returns": returns,
        "log_returns": log_returns,
        "vwap": vwap,
        "vwap_deviation": (close_btc - vwap) / vwap.replace(0, np.nan),
        "adv20": quote_volume_btc.rolling(20, min_periods=10).mean(),
        "adv60": quote_volume_btc.rolling(60, min_periods=30).mean(),
        "volume_ratio_20d": quote_volume_btc / quote_volume_btc.rolling(20, min_periods=10).mean(),
        "high_low_range": high_low_abs / safe_close,
        "open_close_range": (close_btc - open_btc).abs() / safe_close,
        "close_position_in_range": (close_btc - low_btc) / safe_hl,
        "upper_shadow": (high_btc - max_oc) / safe_hl,
        "lower_shadow": (min_oc - low_btc) / safe_hl,
        "momentum_5d": close_btc / close_btc.shift(5) - 1,
        "momentum_20d": close_btc / close_btc.shift(20) - 1,
        "momentum_60d": close_btc / close_btc.shift(60) - 1,
        "volume_momentum_1": quote_volume_btc / quote_volume_btc.shift(1).replace(0, np.nan),
        "volume_momentum_5_20": (
            quote_volume_btc.rolling(5, min_periods=2).mean()
            / quote_volume_btc.rolling(20, min_periods=5).mean()
        ),
        "overnight_gap": open_btc / close_btc.shift(1) - 1,
    }

    if not trades_count.empty:
        matrices["trades_count"] = trades_count
        matrices["trades_per_volume"] = trades_count / safe_volume
    if not taker_buy_volume.empty:
        matrices["taker_buy_volume"] = taker_buy_volume
        matrices["taker_buy_ratio"] = taker_buy_volume / safe_volume
    if not taker_buy_quote_volume_btc.empty:
        matrices["taker_buy_quote_volume"] = taker_buy_quote_volume_btc

    for window in [10, 20, 60, 120]:
        matrices[f"historical_volatility_{window}"] = (
            returns.rolling(window, min_periods=max(window // 2, 1)).std() * np.sqrt(252)
        )

    hl_ratio = np.log(high_btc / low_btc.replace(0, np.nan))
    for window in [10, 20, 60]:
        matrices[f"parkinson_volatility_{window}"] = (
            hl_ratio.pow(2).rolling(window, min_periods=max(window // 2, 1)).mean().pow(0.5)
            / (4 * np.log(2)) ** 0.5
            * np.sqrt(252)
        )

    matrices["beta_to_btc"] = _rolling_beta_to_btc(returns, close_usdt[btc_symbol])

    if drop_btc_symbol:
        for name, mat in list(matrices.items()):
            if btc_symbol in mat.columns:
                matrices[name] = mat.drop(columns=[btc_symbol])

    passthrough_written: list[str] = []
    if source_matrices_dir is not None:
        passthrough_written = _copy_passthrough_fields(
            matrices,
            source_matrices_dir,
            index=index,
            columns=matrices["close"].columns,
            btc_symbol=btc_symbol,
        )

    metadata = {
        "n_symbols_loaded": len(all_data),
        "n_symbols_written": int(matrices["close"].shape[1]),
        "n_bars": int(matrices["close"].shape[0]),
        "start": str(matrices["close"].index.min()),
        "end": str(matrices["close"].index.max()),
        "high_low_inversions_after_sanitize": inversions,
        "passthrough_fields_written": passthrough_written,
        "fields_written": sorted(matrices),
    }
    return matrices, metadata


def _write_matrices(matrices: Mapping[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, mat in matrices.items():
        mat.to_parquet(out_dir / f"{name}.parquet")


def _write_btc_universe(
    source_path: Path,
    output_path: Path,
    matrix_index: pd.Index,
    matrix_columns: pd.Index,
    *,
    btc_symbol: str,
    extend_to_matrix_index: bool = False,
) -> None:
    uni = pd.read_parquet(source_path)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index, errors="coerce")
        uni = uni[uni.index.notna()]
    if btc_symbol in uni.columns:
        uni = uni.drop(columns=[btc_symbol])
    cols = [c for c in uni.columns if c in matrix_columns]
    index = matrix_index if extend_to_matrix_index else uni.index.intersection(matrix_index)
    out = (
        uni[cols]
        .reindex(index=index)
        .fillna(False)
        .infer_objects(copy=False)
        .astype(bool)
        .reindex(columns=matrix_columns, fill_value=False)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BTC-denominated Binance 4h matrices.")
    p.add_argument("--klines-dir", type=Path, default=ROOT / "data/binance_cache/klines/4h")
    p.add_argument("--source-matrices-dir", type=Path, default=ROOT / "data/binance_cache/matrices/4h")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/binance_cache/matrices/4h_btc")
    p.add_argument("--source-universe", type=Path, default=ROOT / "data/binance_cache/universes/BINANCE_TOP30_4h.parquet")
    p.add_argument("--output-universe", type=Path, default=ROOT / "data/binance_cache/universes/BINANCE_TOP30_BTC_4h.parquet")
    p.add_argument("--btc-symbol", default="BTCUSDT")
    p.add_argument("--min-rows", type=int, default=250)
    p.add_argument("--ohlc-mode", choices=["range_envelope", "field_sync", "close_anchor"], default="range_envelope")
    p.add_argument("--keep-btc-symbol", action="store_true")
    p.add_argument(
        "--extend-universe-to-matrix-index",
        action="store_true",
        help="Extend the output universe through the full matrix index with missing activity filled false.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    all_data = _read_binance_klines(args.klines_dir, args.min_rows)
    matrices, metadata = build_btc_terms_matrices(
        all_data,
        btc_symbol=args.btc_symbol,
        ohlc_mode=args.ohlc_mode,
        drop_btc_symbol=not args.keep_btc_symbol,
        source_matrices_dir=args.source_matrices_dir,
    )
    _write_matrices(matrices, args.out_dir)
    _write_btc_universe(
        args.source_universe,
        args.output_universe,
        matrices["close"].index,
        matrices["close"].columns,
        btc_symbol=args.btc_symbol,
        extend_to_matrix_index=args.extend_universe_to_matrix_index,
    )

    summary = BinanceBtcTermsSummary(
        source_klines_dir=str(args.klines_dir),
        source_matrices_dir=str(args.source_matrices_dir),
        output_matrices_dir=str(args.out_dir),
        source_universe_path=str(args.source_universe),
        output_universe_path=str(args.output_universe),
        btc_symbol=args.btc_symbol,
        ohlc_mode=args.ohlc_mode,
        drop_btc_symbol=not args.keep_btc_symbol,
        caveat=(
            "High/low are synthetic BTC-relative range envelopes unless --ohlc-mode "
            "is changed; exact ratio high/low needs synchronized intrabar bars."
        ),
        **metadata,
    )
    summary_path = args.out_dir / "_btc_terms_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    print(json.dumps(asdict(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
