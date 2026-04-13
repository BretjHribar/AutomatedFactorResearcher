"""
diag_per_alpha.py - Run each TOP50 alpha on val and test periods to identify which
alphas degrade and which remain predictive. Uses the exact same eval infrastructure
as eval_alpha_5m.py but extends the date range to cover val and test.

Run from: factor-alpha-platform directory
"""
import sys, os, time, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

# Configuration
UNIVERSE = "BINANCE_TOP50"
INTERVAL = "5m"
DB_PATH = "data/alphas_5m.db"
BARS_PER_DAY = 288
COVERAGE_CUTOFF = 0.3

# Full date window (train + val + test)
FULL_START = "2025-12-01"
FULL_END   = "2026-03-27"

# Split definitions
TRAIN_START, TRAIN_END = "2025-12-01", "2026-02-01"
VAL_START,   VAL_END   = "2026-02-01", "2026-03-01"
TEST_START,  TEST_END  = "2026-03-01", "2026-03-27"

# ─── Load data ───────────────────────────────────────────────────────────────
uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
universe_df = pd.read_parquet(uni_path)
coverage = universe_df.sum(axis=0) / len(universe_df)
valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
print(f"Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
t0 = time.time()
matrices = {}
for fp in sorted(mat_dir.glob("*.parquet")):
    df = pd.read_parquet(fp)
    cols = [c for c in valid_tickers if c in df.columns]
    if cols:
        matrices[fp.stem] = df[cols]
print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)

close_full = matrices["close"]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def evaluate_expression(expr, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expr)

def mean_ic_for_period(alpha_full, start, end):
    """Cross-sectional Spearman IC, alpha[t] vs return[t+1], for a date range."""
    from scipy import stats as sp_stats
    
    close = close_full
    fwd_ret = close.pct_change().shift(-1)    # forward 1-bar return
    
    mask = (alpha_full.index >= start) & (alpha_full.index < end)
    a = alpha_full.loc[mask]
    r = fwd_ret.loc[mask]
    
    uni_mask = universe_df[valid_tickers].reindex(index=a.index, columns=a.columns).fillna(False)
    a = a.where(uni_mask, np.nan)
    
    ics = []
    for t in range(len(a)):
        row_a = a.iloc[t].dropna()
        row_r = r.iloc[t].dropna()
        common = row_a.index.intersection(row_r.index)
        if len(common) < 5:
            continue
        ic, _ = sp_stats.spearmanr(row_a[common], row_r[common])
        if not np.isnan(ic):
            ics.append(ic)
    
    return float(np.mean(ics)) if ics else float('nan')

# ─── Load alphas from DB ──────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT a.id, a.expression FROM alphas a WHERE a.universe = ? ORDER BY a.id",
    (UNIVERSE,)
).fetchall()
conn.close()

print(f"\nFound {len(rows)} {UNIVERSE} alphas\n")
print(f"{'ID':>4} | {'Train IC':>10} | {'Val IC':>10} | {'Test IC':>10} | {'Val-Train':>10} | {'Test-Train':>11}")
print("-" * 75)

results = []
for alpha_id, expr in rows:
    try:
        # Evaluate on entire date range at once for efficiency
        alpha_full = evaluate_expression(expr, matrices)
        if alpha_full is None or alpha_full.empty:
            print(f"#{alpha_id:>3} | FAILED - empty alpha")
            continue
        
        train_ic = mean_ic_for_period(alpha_full, TRAIN_START, TRAIN_END)
        val_ic   = mean_ic_for_period(alpha_full, VAL_START,   VAL_END)
        test_ic  = mean_ic_for_period(alpha_full, TEST_START,  TEST_END)
        
        val_deg  = val_ic  - train_ic
        test_deg = test_ic - train_ic
        
        print(f"#{alpha_id:>3} | {train_ic:>+10.5f} | {val_ic:>+10.5f} | {test_ic:>+10.5f} | {val_deg:>+10.5f} | {test_deg:>+11.5f}")
        results.append((alpha_id, train_ic, val_ic, test_ic, val_deg, test_deg))
    except Exception as e:
        print(f"#{alpha_id:>3} | ERROR: {e}")

# ─── Summary ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(results, columns=["id","train_ic","val_ic","test_ic","val_deg","test_deg"])

print(f"\n{'='*75}")
print(f"PORTFOLIO IC SUMMARY")
print(f"{'='*75}")
print(f"{'Metric':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
print(f"{'Mean IC':<25} {df['train_ic'].mean():>+10.5f} {df['val_ic'].mean():>+10.5f} {df['test_ic'].mean():>+10.5f}")
print(f"{'Median IC':<25} {df['train_ic'].median():>+10.5f} {df['val_ic'].median():>+10.5f} {df['test_ic'].median():>+10.5f}")
print(f"{'% Positive IC alphas':<25} {(df['train_ic']>0).mean():>10.1%} {(df['val_ic']>0).mean():>10.1%} {(df['test_ic']>0).mean():>10.1%}")

print(f"\n{'WORST alphas in TEST (largest degradation)':}")
print(df.sort_values('test_ic').head(10)[['id','train_ic','val_ic','test_ic','test_deg']].to_string(index=False))

print(f"\n{'BEST alphas in TEST (still working)':}")
print(df.sort_values('test_ic', ascending=False).head(10)[['id','train_ic','val_ic','test_ic','test_deg']].to_string(index=False))
