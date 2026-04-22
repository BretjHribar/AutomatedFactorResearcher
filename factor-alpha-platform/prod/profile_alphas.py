"""Profile alpha execution times and analyze lookback requirements."""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# Load config
with open("prod/config/binance.json") as f:
    cfg = json.load(f)

alphas = cfg["alphas"]

# ── Part 1: Lookback Analysis ──
print("=" * 80)
print("ALPHA LOOKBACK ANALYSIS")
print("=" * 80)

for a in alphas:
    expr = a["expression"]
    # Find all numeric params in the expression
    all_nums = [int(n) for n in re.findall(r'[\(,]\s*(\d+)', expr) if int(n) > 1]
    
    # Find nested rolling: sma(ts_xxx(..., N), M) -> effective = N + M
    nested = re.findall(r'sma\(ts_\w+\([^)]+,\s*(\d+)\)[^)]*,\s*(\d+)\)', expr)
    nested_sums = [int(a) + int(b) for a, b in nested]
    
    # sma(Decay_exp(ts_xxx(..., N), decay), M) -> N + M  
    nested2 = re.findall(r'sma\(Decay_exp\(ts_\w+\([^)]+,\s*(\d+)\).*?,\s*(\d+)\)', expr)
    nested_sums2 = [int(a) + int(b) for a, b in nested2]
    
    # Derived field lookbacks used as inputs
    if 'adv60' in expr:
        all_nums.append(360)  # adv60 = rolling(360)
    if 'adv20' in expr:
        all_nums.append(120)
    if 'historical_volatility_60' in expr:
        all_nums.append(360)
    if 'parkinson_volatility_20' in expr:
        all_nums.append(120)
    if 'funding_rate_avg_7d' in expr:
        all_nums.append(21)
    if 'funding_rate_zscore' in expr:
        all_nums.append(42)
    if 'beta_to_btc' in expr:
        all_nums.append(60)
    if 'momentum_60d' in expr:
        all_nums.append(360)
    
    # The effective lookback is the MAX of all individual and nested windows
    max_lb = max(all_nums + nested_sums + nested_sums2 + [0])
    
    # For sma(ts_delta(s_log_1p(adv60), 30), 120): adv60=360 + ts_delta=30 + sma=120 = 510
    # More conservative: sum all nested chains
    if 'sma(ts_delta(s_log_1p(adv60), 30), 120)' in expr:
        max_lb = max(max_lb, 360 + 30 + 120)  # 510
    
    print(f"  {a['id']:<6}  lookback={max_lb:>4} bars = {max_lb*4:>5}h = {max_lb*4/24:>5.1f}d  | {expr[:60]}")

print()

# ── Part 2: Matrix Derived Field Lookbacks ──
print("MATRIX DERIVED FIELD MAX LOOKBACKS")
print("-" * 50)
derived = [
    ("historical_volatility_120", 120),
    ("adv60 (used as input to alphas)", 60),
    ("parkinson_volatility_60", 60),
    ("historical_volatility_60", 60),
    ("funding_rate_zscore", 42),
    ("beta_to_btc", 60),
    ("momentum_60d", 60),
    ("adv20", 20),
]
for name, lb in derived:
    print(f"  {name:<40} {lb:>4} bars")

print()
print("WORST CASE: sma(ts_delta(s_log_1p(adv60), 30), 120)")
print("  adv60 raw rolling window in matrix = 360 bars (from download_kucoin)")
print("  But in Binance matrices, adv60 = qv.rolling(60).mean()")
print("  The alpha then does: sma(ts_delta(s_log_1p(adv60), 30), 120)")
print("  Total: 60 (adv60) + 30 (ts_delta) + 120 (sma) = 210 bars")
print("  Conservative with warm-up: 300 bars")

# ── Part 3: Profile Alpha Execution Times ──
print()
print("=" * 80)
print("ALPHA EXECUTION TIME PROFILING (13822 bars x 639 tickers)")
print("=" * 80)

from src.operators.fastexpression import FastExpressionEngine

matrices_dir = Path("data/binance_cache/matrices/4h")
matrices = {}
for f in matrices_dir.glob("*.parquet"):
    matrices[f.stem] = pd.read_parquet(f)

engine = FastExpressionEngine(data_fields=matrices)

times = []
for a in alphas:
    t0 = time.time()
    try:
        result = engine.evaluate(a["expression"])
        elapsed = time.time() - t0
        n_active = result.iloc[-1].notna().sum()
        times.append((a["id"], elapsed, n_active, a["expression"][:55]))
    except Exception as e:
        elapsed = time.time() - t0
        times.append((a["id"], elapsed, 0, f"ERROR: {e}"))

# Sort by time descending
times.sort(key=lambda x: -x[1])

print(f"\n{'ID':<6} {'Time':>7} {'Active':>6}  Expression")
print("-" * 80)
for aid, t, n, expr in times:
    marker = " ***SLOW***" if t > 3.0 else ""
    print(f"  {aid:<6} {t:>6.2f}s {n:>5}  {expr}{marker}")

total = sum(t for _, t, _, _ in times)
print(f"\n  TOTAL: {total:.1f}s")
print(f"  Top 3 slowest: {sum(t for _, t, _, _ in times[:3]):.1f}s ({sum(t for _, t, _, _ in times[:3])/total*100:.0f}% of total)")
