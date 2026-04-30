"""
Validate Binance 4h klines after a fresh ingest.

Same checks as validate_binance_daily.py but for 4h interval.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
KLINES_DIR = ROOT / "data/binance_cache/klines/4h"
MATRICES_DIR = ROOT / "data/binance_cache/matrices/4h"
EX_INFO = ROOT / "data/binance_cache/exchange_info.parquet"

results = []
def add(check, status, detail):
    results.append((check, status, detail))
    sym = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"  {sym} {check}: {detail}", flush=True)


def main():
    print("=" * 80)
    print("BINANCE 4H DATA VALIDATION")
    print("=" * 80)

    if not EX_INFO.exists():
        add("exchange_info", "FAIL", "missing")
        return _summary()
    ex = pd.read_parquet(EX_INFO)
    files = sorted(KLINES_DIR.glob("*.parquet"))
    add("file_count", "PASS" if len(files) >= 0.7 * len(ex) else "WARN",
        f"{len(files)} kline files (vs {len(ex)} symbols)")

    for sym in ["BTCUSDT", "ETHUSDT"]:
        fp = KLINES_DIR / f"{sym}.parquet"
        if not fp.exists():
            add(f"{sym}/exists", "FAIL", "missing")
            continue
        df = pd.read_parquet(fp)
        if df.empty:
            add(f"{sym}/empty", "FAIL", "0 rows")
            continue
        first = df["datetime"].min()
        last = df["datetime"].max()
        rows = len(df)
        # Expected: 6 bars/day
        expected = (last - first).total_seconds() / (4*3600) + 1
        cov = rows / expected if expected > 0 else 0
        days_stale = (datetime.utcnow() - last.to_pydatetime()).total_seconds() / 86400
        status = "PASS" if cov > 0.95 and days_stale < 0.5 else "WARN" if cov > 0.9 else "FAIL"
        add(f"{sym}/span", status,
            f"{rows} bars {first.date()} -> {last.date()} (cov={cov*100:.1f}%, {days_stale:.1f}d stale)")

    # Coverage by year
    print()
    daily_active = pd.Series(0, index=pd.date_range("2019-09-01", "2026-12-31", freq="D"))
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=["datetime"])
        except Exception:
            continue
        if df.empty:
            continue
        idx = pd.to_datetime(df["datetime"]).dt.normalize()
        daily_active.loc[idx[idx.isin(daily_active.index)]] += 1
    print("  Coverage by year (avg active 4h-symbols per day-bucket):")
    for y in range(2019, 2027):
        sub = daily_active.loc[f"{y}-01-01":f"{y}-12-31"]
        sub = sub[sub > 0]
        if len(sub):
            # Each symbol contributes 6 bars/day → divide by 6
            print(f"    {y}: {sub.mean()/6:>5.0f}  (max {sub.max()/6:.0f}, min {sub.min()/6:.0f})")
    add("year_coverage", "PASS", "yearly counts printed above")

    # Spot-check OHLC positivity on 100 files
    bad_price = []
    for fp in files[:100]:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df.empty: continue
        for col in ["open","high","low","close"]:
            if (df[col] <= 0).any():
                bad_price.append((fp.stem, col)); break
    add("price_positivity", "PASS" if not bad_price else "FAIL",
        f"{len(bad_price)} files with non-positive OHLC out of 100" if bad_price else "all 100 sample files have positive OHLC")

    # Gap check on top symbols (gap > 12h = 3 missing bars)
    for sym in ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]:
        fp = KLINES_DIR / f"{sym}.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp)
        if df.empty: continue
        dts = pd.to_datetime(df["datetime"]).sort_values()
        gaps = dts.diff().dropna()
        max_gap = gaps.max()
        n_big = (gaps > pd.Timedelta(hours=8)).sum()  # > 2 bars missing
        status = "PASS" if max_gap <= pd.Timedelta(hours=8) else "WARN"
        add(f"{sym}/gaps", status,
            f"max gap {max_gap.total_seconds()/3600:.0f}h, {n_big} gaps > 8h")

    # BTC sanity
    btc_fp = KLINES_DIR / "BTCUSDT.parquet"
    if btc_fp.exists():
        btc = pd.read_parquet(btc_fp).set_index("datetime")
        # Apr 2024: BTC ~60-72K
        apr24 = btc.loc["2024-04-01":"2024-04-30", "close"]
        if len(apr24) > 0:
            mean_p = apr24.mean()
            ok = 50000 < mean_p < 80000
            add("btc_apr24_sanity", "PASS" if ok else "FAIL",
                f"BTC Apr 2024 mean = ${mean_p:,.0f}")
        nov21 = btc.loc["2021-11-01":"2021-11-30", "close"]
        if len(nov21) > 0:
            max_p = nov21.max()
            ok = 60000 < max_p < 75000
            add("btc_nov21_ath", "PASS" if ok else "WARN",
                f"BTC Nov 2021 max = ${max_p:,.0f}")

    # Matrix consistency
    close_path = MATRICES_DIR / "close.parquet"
    if close_path.exists():
        close = pd.read_parquet(close_path)
        add("matrix/close.shape", "PASS",
            f"{close.shape}, {close.index.min()} -> {close.index.max()}")
        ret_path = MATRICES_DIR / "returns.parquet"
        if ret_path.exists():
            ret = pd.read_parquet(ret_path)
            add("matrix/returns_aligned", "PASS" if ret.shape == close.shape else "FAIL",
                f"returns.shape = {ret.shape}")
    else:
        add("matrix/close", "WARN", "matrices not yet built")

    return _summary()


def _summary():
    p = sum(1 for _,s,_ in results if s == "PASS")
    w = sum(1 for _,s,_ in results if s == "WARN")
    f = sum(1 for _,s,_ in results if s == "FAIL")
    print()
    print("=" * 80)
    print(f"SUMMARY: {p} PASS / {w} WARN / {f} FAIL")
    print("=" * 80)
    out = ROOT / "data/binance_cache/validation_4h_report.json"
    out.write_text(json.dumps([{"check":c,"status":s,"detail":d} for c,s,d in results], indent=2))
    print(f"Report: {out}")
    sys.exit(1 if f > 0 else 0)


if __name__ == "__main__":
    main()
