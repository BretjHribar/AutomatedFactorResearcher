"""Build KuCoin 4h research matrices denominated in BTC terms.

The source KuCoin futures klines are USDT-quoted. This script creates a
separate matrix directory where point-in-time prices are expressed as
symbol/XBT at the same 4h bar and quote-notional fields are expressed as BTC
notional. It intentionally writes a new dataset so the live USDT pipeline is
not touched.

Notes on OHLC:
    Open and close can be converted exactly at the 4h bar endpoints using
    same-bar XBT open/close. A true BTC-relative high/low requires synchronized
    intrabar data, which is not available in the local KuCoin 4h cache. The
    default "range_envelope" mode uses high/XBT_low and low/XBT_high to produce
    a conservative, non-inverted range envelope.

Example:
    python tools/build_kucoin_btc_terms.py
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


@dataclass(frozen=True)
class BtcTermsSummary:
    source_klines_dir: str
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
    fields_written: list[str]
    caveat: str


def _read_kucoin_klines(klines_dir: Path, min_rows: int) -> dict[str, pd.DataFrame]:
    all_data: dict[str, pd.DataFrame] = {}
    for fpath in sorted(klines_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fpath)
            if df.empty or len(df) < min_rows:
                continue
            if "time" in df.columns:
                df = df.set_index("time")
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
            df = df[~df.index.duplicated(keep="last")]
            cols = [c for c in ("open", "high", "low", "close", "volume", "turnover") if c in df.columns]
            if len(cols) < 6:
                continue
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


def build_btc_terms_matrices(
    all_data: Mapping[str, pd.DataFrame],
    *,
    btc_symbol: str = "XBTUSDTM",
    ohlc_mode: str = "range_envelope",
    drop_btc_symbol: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Convert in-memory KuCoin kline data into BTC-denominated matrices."""
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
    turnover_usdt = _matrix(all_data, "turnover", index)

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

    # Make the synthetic OHLC internally consistent even when 4h source ranges
    # cannot identify the exact intrabar ratio high/low.
    high_btc = _df_fmax(high_raw, low_raw, open_btc, close_btc)
    low_btc = _df_fmin(high_raw, low_raw, open_btc, close_btc)
    inversions = int(((high_btc < low_btc) & high_btc.notna() & low_btc.notna()).sum().sum())

    turnover_btc = _div_by_series(turnover_usdt, btc_typical)
    ret = close_btc.pct_change(fill_method=None)
    vwap = (high_btc + low_btc + close_btc) / 3.0
    safe_close = close_btc.replace(0, np.nan)
    safe_range = (high_btc - low_btc).replace(0, np.nan)
    max_oc = _df_fmax(open_btc, close_btc)
    min_oc = _df_fmin(open_btc, close_btc)

    matrices: dict[str, pd.DataFrame] = {
        "open": open_btc,
        "high": high_btc,
        "low": low_btc,
        "close": close_btc,
        "volume": volume,
        "turnover": turnover_btc,
        "returns": ret,
        "log_returns": np.log1p(ret.fillna(0)),
        "vwap": vwap,
        "adv20": turnover_btc.rolling(120, min_periods=60).mean(),
        "adv60": turnover_btc.rolling(360, min_periods=180).mean(),
        "high_low_range": (high_btc - low_btc) / safe_close,
        "open_close_range": (close_btc - open_btc).abs() / safe_close,
        "close_position_in_range": (close_btc - low_btc) / (safe_range + 1e-10),
        "upper_shadow": (high_btc - max_oc) / safe_close,
        "lower_shadow": (min_oc - low_btc) / safe_close,
        "volume_momentum_5_20": volume.rolling(30).mean() / volume.rolling(120).mean(),
        "historical_volatility_10": ret.rolling(60).std() * np.sqrt(6 * 365),
        "historical_volatility_20": ret.rolling(120).std() * np.sqrt(6 * 365),
        "historical_volatility_60": ret.rolling(360).std() * np.sqrt(6 * 365),
        "historical_volatility_120": ret.rolling(720, min_periods=360).std() * np.sqrt(6 * 365),
        "momentum_5d": close_btc / close_btc.shift(30) - 1,
        "momentum_20d": close_btc / close_btc.shift(120) - 1,
        "momentum_60d": close_btc / close_btc.shift(360) - 1,
        "vwap_deviation": (close_btc - vwap) / vwap.replace(0, np.nan),
        "dollars_traded": turnover_btc,
        "quote_volume": turnover_btc,
        "volume_ratio_20d": volume / volume.rolling(120).mean(),
        "volume_momentum_1": volume / volume.shift(1) - 1,
    }

    hl = np.log(high_btc / low_btc.replace(0, np.nan))
    matrices["parkinson_volatility_10"] = hl.pow(2).rolling(60).mean().pow(0.5) / (2 * np.log(2)) ** 0.5
    matrices["parkinson_volatility_20"] = hl.pow(2).rolling(120).mean().pow(0.5) / (2 * np.log(2)) ** 0.5
    matrices["parkinson_volatility_60"] = hl.pow(2).rolling(360).mean().pow(0.5) / (2 * np.log(2)) ** 0.5

    # Keep the field name for alpha compatibility. On a BTC-relative return
    # matrix this is beta to the original BTC-USDT move; useful as a residual
    # market sensitivity estimate, but no longer a raw USD beta.
    btc_ret_usdt = close_usdt[btc_symbol].pct_change(fill_method=None)
    btc_var = btc_ret_usdt.rolling(60, min_periods=30).var()
    betas = {}
    for col in ret.columns:
        cov = ret[col].rolling(60, min_periods=30).cov(btc_ret_usdt)
        betas[col] = cov / btc_var
    matrices["beta_to_btc"] = pd.DataFrame(betas, index=index)

    if drop_btc_symbol:
        for name, mat in list(matrices.items()):
            if btc_symbol in mat.columns:
                matrices[name] = mat.drop(columns=[btc_symbol])

    metadata = {
        "n_symbols_loaded": len(all_data),
        "n_symbols_written": int(matrices["close"].shape[1]),
        "n_bars": int(matrices["close"].shape[0]),
        "start": str(matrices["close"].index.min()),
        "end": str(matrices["close"].index.max()),
        "high_low_inversions_after_sanitize": inversions,
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
    p = argparse.ArgumentParser(description="Build BTC-denominated KuCoin 4h matrices.")
    p.add_argument("--klines-dir", type=Path, default=ROOT / "data/kucoin_cache/klines/4h")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/kucoin_cache/matrices/4h_btc")
    p.add_argument("--source-universe", type=Path, default=ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet")
    p.add_argument("--out-universe", type=Path, default=ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_BTC_4h.parquet")
    p.add_argument("--btc-symbol", default="XBTUSDTM")
    p.add_argument("--ohlc-mode", default="range_envelope", choices=["range_envelope", "field_sync", "close_anchor"])
    p.add_argument("--keep-btc-symbol", action="store_true")
    p.add_argument(
        "--extend-universe-to-matrices",
        action="store_true",
        help="Extend the universe calendar to all matrix bars with False for bars outside the source universe.",
    )
    p.add_argument("--min-rows", type=int, default=50)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    print(f"Loading KuCoin klines from {args.klines_dir}...")
    all_data = _read_kucoin_klines(args.klines_dir, min_rows=args.min_rows)
    print(f"  loaded {len(all_data)} symbols")

    matrices, meta = build_btc_terms_matrices(
        all_data,
        btc_symbol=args.btc_symbol,
        ohlc_mode=args.ohlc_mode,
        drop_btc_symbol=not args.keep_btc_symbol,
    )
    print(
        f"  built {len(matrices)} BTC-term fields, "
        f"{meta['n_bars']} bars x {meta['n_symbols_written']} tickers"
    )
    print(f"  span {meta['start']} -> {meta['end']}")

    print(f"Writing matrices to {args.out_dir}...")
    _write_matrices(matrices, args.out_dir)

    print(f"Writing BTC-term universe to {args.out_universe}...")
    _write_btc_universe(
        args.source_universe,
        args.out_universe,
        matrices["close"].index,
        matrices["close"].columns,
        btc_symbol=args.btc_symbol,
        extend_to_matrix_index=args.extend_universe_to_matrices,
    )

    summary = BtcTermsSummary(
        source_klines_dir=str(args.klines_dir.relative_to(ROOT) if args.klines_dir.is_relative_to(ROOT) else args.klines_dir),
        output_matrices_dir=str(args.out_dir.relative_to(ROOT) if args.out_dir.is_relative_to(ROOT) else args.out_dir),
        source_universe_path=str(args.source_universe.relative_to(ROOT) if args.source_universe.is_relative_to(ROOT) else args.source_universe),
        output_universe_path=str(args.out_universe.relative_to(ROOT) if args.out_universe.is_relative_to(ROOT) else args.out_universe),
        btc_symbol=args.btc_symbol,
        ohlc_mode=args.ohlc_mode,
        drop_btc_symbol=not args.keep_btc_symbol,
        caveat=(
            "Open/close are same-bar endpoint conversions. True BTC-relative 4h high/low "
            "requires intrabar synchronized data; range_envelope uses high/XBT_low and "
            "low/XBT_high as a conservative non-inverted range."
        ),
        **meta,
    )
    summary_path = args.out_dir / "_btc_terms_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
