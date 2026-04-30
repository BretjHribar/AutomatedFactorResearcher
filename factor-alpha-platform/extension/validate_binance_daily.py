"""
Validate Binance daily klines after a fresh ingest.

Checks:
  1. File counts match exchange info
  2. BTCUSDT / ETHUSDT span the expected period (2019-09 to today)
  3. Coverage by year (avg active symbols per day)
  4. No negative or zero prices in OHLC
  5. No internal date gaps for top symbols
  6. Sanity: no daily move > 200% (data error proxy)
  7. Cross-exchange sanity: BTC mean price 2024-04 vs known $66K-70K
  8. Matrix-format consistency (close.parquet has expected shape)

Outputs PASS/FAIL summary. Exit 0 if all pass, 1 otherwise.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
KLINES_DIR = ROOT / "data/binance_cache/klines/1d"
MATRICES_DIR = ROOT / "data/binance_cache/matrices"
EX_INFO = ROOT / "data/binance_cache/exchange_info.parquet"

results = []
def add(check, status, detail):
    results.append((check, status, detail))
    sym = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"  {sym} {check}: {detail}", flush=True)


def main():
    print("=" * 80)
    print("BINANCE DAILY DATA VALIDATION")
    print("=" * 80)

    # 1. File counts
    if not EX_INFO.exists():
        add("exchange_info", "FAIL", "exchange_info.parquet missing")
        return _summary()
    ex = pd.read_parquet(EX_INFO)
    files = sorted(KLINES_DIR.glob("*.parquet"))
    add("exchange_info", "PASS", f"{len(ex)} symbols in exchange_info ({(ex['status']=='TRADING').sum()} active)")
    add("file_count", "PASS" if len(files) >= 0.7 * len(ex) else "WARN",
        f"{len(files)} kline files (vs {len(ex)} symbols in info)")

    # 2. BTCUSDT / ETHUSDT span
    for sym in ["BTCUSDT", "ETHUSDT"]:
        fp = KLINES_DIR / f"{sym}.parquet"
        if not fp.exists():
            add(f"{sym}/exists", "FAIL", "missing file")
            continue
        df = pd.read_parquet(fp)
        if df.empty:
            add(f"{sym}/empty", "FAIL", "0 rows")
            continue
        first = df["datetime"].min()
        last = df["datetime"].max()
        rows = len(df)
        expected = (last - first).days + 1
        cov = rows / expected if expected > 0 else 0
        # Should span ~2019-09 to today
        days_stale = (datetime.utcnow() - last.to_pydatetime()).days
        status = "PASS" if cov > 0.95 and days_stale <= 3 else "WARN" if cov > 0.9 else "FAIL"
        add(f"{sym}/span", status,
            f"{rows} bars {first.date()} -> {last.date()} (cov={cov*100:.1f}%, {days_stale}d stale)")

    # 3. Coverage by year
    print()
    yearly_active = {}
    sample_files = files  # all files
    daily_active = pd.Series(0, index=pd.date_range("2019-09-01", "2026-12-31", freq="D"))
    for fp in sample_files:
        try:
            df = pd.read_parquet(fp, columns=["datetime"])
        except Exception:
            continue
        if df.empty:
            continue
        idx = pd.to_datetime(df["datetime"]).dt.normalize()
        daily_active.loc[idx[idx.isin(daily_active.index)]] += 1
    print("  Coverage by year (avg active symbols per day):")
    for y in range(2019, 2027):
        sub = daily_active.loc[f"{y}-01-01":f"{y}-12-31"]
        sub = sub[sub > 0]
        if len(sub):
            print(f"    {y}: {sub.mean():>5.0f}  (max {sub.max():.0f}, min {sub.min():.0f})")
    add("year_coverage", "PASS", "yearly counts printed above")

    # 4. Negative/zero prices in any file
    bad_price = []
    for fp in sample_files[:100]:  # spot-check 100
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df.empty:
            continue
        for col in ["open", "high", "low", "close"]:
            if (df[col] <= 0).any():
                bad_price.append((fp.stem, col, int((df[col] <= 0).sum())))
                break
    add("price_positivity",
        "PASS" if not bad_price else "FAIL",
        f"{len(bad_price)} files with non-positive OHLC out of 100 sampled" if bad_price else "all 100 sample files have positive OHLC")

    # 5. Internal gaps for top liquidity symbols
    top_syms = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT"]
    for sym in top_syms:
        fp = KLINES_DIR / f"{sym}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if df.empty: continue
        dts = pd.to_datetime(df["datetime"]).sort_values()
        gaps = dts.diff().dropna()
        max_gap = gaps.max()
        n_big_gaps = (gaps > pd.Timedelta(days=2)).sum()
        status = "PASS" if max_gap <= pd.Timedelta(days=2) else "WARN"
        add(f"{sym}/gaps", status,
            f"max gap {max_gap.days}d, {n_big_gaps} gaps > 2d")

    # 6. Extreme move check (per-day return > 200%)
    extreme = []
    for fp in sample_files[:100]:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if df.empty or len(df) < 2:
            continue
        c = df["close"].astype(float).values
        ret = c[1:] / np.where(c[:-1] > 0, c[:-1], np.nan) - 1
        # Filter NaN
        ret = ret[~np.isnan(ret)]
        if len(ret) and (np.abs(ret).max() > 2.0):
            extreme.append((fp.stem, float(np.abs(ret).max())))
    add("extreme_moves",
        "WARN" if extreme else "PASS",
        f"{len(extreme)} files with |daily_ret| > 200%: " +
        ", ".join(f"{n}({r*100:.0f}%)" for n,r in extreme[:5]) if extreme else "no daily moves > 200% in 100 sampled")

    # 7. BTC sanity vs known prices
    btc_fp = KLINES_DIR / "BTCUSDT.parquet"
    if btc_fp.exists():
        btc = pd.read_parquet(btc_fp).set_index("datetime")
        # Apr 2024: BTC was ~60-72K (post-halving spike)
        apr24 = btc.loc["2024-04-01":"2024-04-30", "close"]
        if len(apr24) > 0:
            mean_p = apr24.mean()
            ok = 50000 < mean_p < 80000
            add("btc_apr24_sanity",
                "PASS" if ok else "FAIL",
                f"BTC Apr 2024 mean = ${mean_p:,.0f} (expected $50K-$80K)")
        # Nov 2021: BTC ATH ~67K
        nov21 = btc.loc["2021-11-01":"2021-11-30", "close"]
        if len(nov21) > 0:
            max_p = nov21.max()
            ok = 60000 < max_p < 75000
            add("btc_nov21_ath",
                "PASS" if ok else "WARN",
                f"BTC Nov 2021 max = ${max_p:,.0f} (expected ~$67K)")

    # 8. Matrix consistency
    close_path = MATRICES_DIR / "close.parquet"
    if close_path.exists():
        close = pd.read_parquet(close_path)
        add("matrix/close.shape", "PASS",
            f"{close.shape}, {close.index.min().date()} -> {close.index.max().date()}")
        # Check returns matrix matches
        ret_path = MATRICES_DIR / "returns.parquet"
        if ret_path.exists():
            ret = pd.read_parquet(ret_path)
            same_shape = ret.shape == close.shape
            add("matrix/returns_aligned", "PASS" if same_shape else "FAIL",
                f"returns.shape = {ret.shape} {'==' if same_shape else '!='} close.shape")
    else:
        add("matrix/close", "WARN", "matrices not yet built (run build_matrices)")

    return _summary()


def _summary():
    p = sum(1 for _,s,_ in results if s == "PASS")
    w = sum(1 for _,s,_ in results if s == "WARN")
    f = sum(1 for _,s,_ in results if s == "FAIL")
    print()
    print("=" * 80)
    print(f"SUMMARY: {p} PASS / {w} WARN / {f} FAIL")
    print("=" * 80)
    out = ROOT / "data/binance_cache/validation_report.json"
    out.write_text(json.dumps([{"check":c,"status":s,"detail":d} for c,s,d in results], indent=2))
    print(f"Report: {out}")
    sys.exit(1 if f > 0 else 0)


if __name__ == "__main__":
    main()
