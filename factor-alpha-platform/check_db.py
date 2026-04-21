"""Universe size diagnostic — shows exactly how many symbols survive each filter."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

# Load universe parquet
uni = pd.read_parquet("data/fmp_cache/universes/TOP2000TOP3000.parquet")

# ── Pre-filter: raw universe ──────────────────────────────────────────────
print("="*60)
print("  UNIVERSE: TOP2000TOP3000")
print("="*60)
print(f"  Raw tickers in parquet:          {len(uni.columns)}")

# Filter 1: US-only (no .L .TO .V suffixes)
us_mask = [t for t in uni.columns if not any(t.endswith(s) for s in [".L",".TO",".V",".AX"])]
uni_us = uni[us_mask]
print(f"  After US-only filter:            {len(uni_us.columns)}  (removed {len(uni.columns)-len(uni_us.columns)})")

# Filter 2: 30% coverage cutoff
coverage = uni_us.sum(axis=0) / len(uni_us)
valid = sorted(coverage[coverage > 0.30].index.tolist())
print(f"  After >30% coverage cutoff:      {len(valid)}  (removed {len(us_mask)-len(valid)})")

# ── Daily membership stats ────────────────────────────────────────────────
uni_valid = uni_us[valid]

periods = {
    "Full history  (2016-now)":   (None, None),
    "Train         (2016-2023)":  ("2016-01-01", "2023-01-01"),
    "Val           (2023-mid24)": ("2023-01-01", "2024-07-01"),
    "Test          (mid24-now)":  ("2024-07-01", None),
    "Recent 60d":                 (-60, None),
}

print(f"\n  Daily membership (stocks/day in universe after ALL filters):")
print(f"  {'Period':<30} {'Avg':>6} {'Min':>6} {'Max':>6} {'Last':>6}")
print(f"  {'-'*56}")
for label, (s, e) in periods.items():
    if s is None and e is None:
        sl = uni_valid
    elif isinstance(s, int):
        sl = uni_valid.iloc[s:]
    else:
        sl = uni_valid.loc[s:e] if e else uni_valid.loc[s:]
    daily = sl.sum(axis=1)
    last = int(daily.iloc[-1]) if len(daily) > 0 else 0
    print(f"  {label:<30} {daily.mean():>6.0f} {daily.min():>6.0f} {daily.max():>6.0f} {last:>6}")

# ── Yearly breakdown ──────────────────────────────────────────────────────
print(f"\n  Year-by-year average:")
print(f"  {'Year':<6} {'Avg stocks/day':>16} {'Total unique tickers':>22}")
for yr in range(2016, 2027):
    try:
        sl = uni_valid.loc[str(yr)]
        if len(sl) == 0:
            continue
        avg = sl.sum(axis=1).mean()
        unique = int(sl.any(axis=0).sum())
        print(f"  {yr:<6} {avg:>16.0f} {unique:>22}")
    except Exception:
        pass
