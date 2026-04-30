"""
Rebuild research matrices from the (now-fixed) kline parquets.
Same feature set as prod build but WITHOUT tail truncation — full history.

Writes to data/kucoin_cache/matrices/4h/ (research dir).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KLINES_DIR = PROJECT_ROOT / "data/kucoin_cache/klines/4h"
RESEARCH_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
PROD_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h/prod"


def build_research_matrices():
    print(f"Loading klines from {KLINES_DIR}...")
    all_data = {}
    for fpath in sorted(KLINES_DIR.glob("*.parquet")):
        sym = fpath.stem
        try:
            df = pd.read_parquet(fpath)
            if df.empty or len(df) < 50:
                continue
            if "time" in df.columns:
                df = df.set_index("time").sort_index()
            # NO tail limit for research
            all_data[sym] = df
        except Exception:
            continue
    print(f"  loaded {len(all_data)} symbols")

    # Common index
    all_idx = set()
    for df in all_data.values():
        all_idx.update(df.index)
    common_idx = sorted(all_idx)
    print(f"  common index: {len(common_idx)} bars, {common_idx[0]} to {common_idx[-1]}")

    # Base OHLCV
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    for field in ["open", "close", "high", "low", "volume", "turnover"]:
        d = {}
        for sym, df in all_data.items():
            if field in df.columns:
                d[sym] = df[field]
        mat = pd.DataFrame(d, index=common_idx)
        mat.to_parquet(RESEARCH_DIR / f"{field}.parquet")
    print("  base OHLCV written")

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
        "volume_momentum_5_20": vol.rolling(30).mean() / vol.rolling(120).mean(),
        "historical_volatility_10": ret.rolling(60).std() * np.sqrt(6 * 365),
        "historical_volatility_20": ret.rolling(120).std() * np.sqrt(6 * 365),
        "historical_volatility_60": ret.rolling(360).std() * np.sqrt(6 * 365),
        "historical_volatility_120": ret.rolling(720, min_periods=360).std() * np.sqrt(6 * 365),
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

    # Beta to BTC
    btc_sym = "XBTUSDTM"
    if btc_sym in ret.columns:
        btc_ret = ret[btc_sym]
        btc_var = btc_ret.rolling(60, min_periods=30).var()
        betas = {}
        for col in ret.columns:
            if col == btc_sym:
                betas[col] = pd.Series(1.0, index=ret.index)
                continue
            cov = ret[col].rolling(60, min_periods=30).cov(btc_ret)
            betas[col] = cov / btc_var
        derived["beta_to_btc"] = pd.DataFrame(betas)

    for name, mat in derived.items():
        mat.to_parquet(RESEARCH_DIR / f"{name}.parquet")
    print(f"  wrote {len(derived)} derived features")
    print(f"DONE — {close.shape[0]} bars x {close.shape[1]} tickers in {RESEARCH_DIR}")


if __name__ == "__main__":
    build_research_matrices()
