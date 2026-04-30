"""Are fundamentals updating in late 2025, or just carrying forward stale values?"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest_voc_equities_sweep import load_data_universe

print("Loading TOP2000...")
matrices, tickers, dates, close_vals, chars = load_data_universe("TOP2000")

# Check value-change frequency for each char
# A "stale" matrix has columns where the value barely changes month-over-month
fundamentals = ["book_to_market", "earnings_yield", "free_cashflow_yield",
                "ev_to_ebitda", "ev_to_revenue", "roe", "roa",
                "gross_margin", "operating_margin", "net_margin",
                "asset_turnover", "cap", "debt_to_equity", "current_ratio"]

# For each char, count how many tickers had ANY value-change in given month
def changed_frac(df, start, end):
    """Fraction of tickers whose value changed between start and end."""
    s = df.loc[start:end]
    if len(s) < 2:
        return np.nan
    first = s.iloc[0]
    last = s.iloc[-1]
    diff = (first - last).abs()
    valid = first.notna() & last.notna()
    if valid.sum() == 0:
        return np.nan
    changed = (diff > 1e-12) & valid
    return changed.sum() / valid.sum()

print(f"\n{'='*100}")
print(f"FRACTION OF TICKERS WHERE CHAR VALUE CHANGED WITHIN A 30-BAR WINDOW")
print(f"(low value = stale/forward-filled values)")
print(f"{'='*100}")

windows = [
    ("2024-04-01", "2024-04-30", "Apr 2024"),
    ("2024-10-01", "2024-10-31", "Oct 2024"),
    ("2025-04-01", "2025-04-30", "Apr 2025"),
    ("2025-08-01", "2025-08-31", "Aug 2025"),
    ("2025-10-01", "2025-10-31", "Oct 2025"),
    ("2025-11-01", "2025-11-30", "Nov 2025"),
    ("2025-12-01", "2025-12-31", "Dec 2025"),
    ("2026-01-02", "2026-01-31", "Jan 2026"),
    ("2026-02-01", "2026-02-26", "Feb 2026"),
]

print(f"{'char':<28}" + "".join(f"{w[2]:>10}" for w in windows))
for cn in fundamentals:
    if cn not in matrices:
        continue
    df = matrices[cn]
    fracs = [changed_frac(df, w[0], w[1]) for w in windows]
    print(f"{cn:<28}" + "".join(
        (f"{f*100:>9.1f}%" if not np.isnan(f) else "      n/a") for f in fracs))

# Also check time-series quantities (these SHOULD update every bar)
print(f"\n{'─'*100}")
print(f"Time-series chars (should change frequently):")
ts_chars = ["log_returns", "historical_volatility_20", "parkinson_volatility_20",
            "adv20", "dollars_traded"]
print(f"{'char':<28}" + "".join(f"{w[2]:>10}" for w in windows))
for cn in ts_chars:
    if cn not in matrices:
        continue
    df = matrices[cn]
    fracs = [changed_frac(df, w[0], w[1]) for w in windows]
    print(f"{cn:<28}" + "".join(
        (f"{f*100:>9.1f}%" if not np.isnan(f) else "      n/a") for f in fracs))
