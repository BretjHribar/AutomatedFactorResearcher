"""Deep diagnostic: Per-factor validation Sharpe and correlation structure."""
import sys, os, sqlite3, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars

DB_PATH = "data/alphas.db"
VAL_START, VAL_END = "2024-09-01", "2025-03-01"
UNIVERSE = "BINANCE_TOP50"
INTERVAL = "4h"
BOOKSIZE = 2_000_000.0
BARS_PER_DAY = 6
COVERAGE_CUTOFF = 0.3

# Load data
mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
universe_df = pd.read_parquet(uni_path)
coverage = universe_df.sum(axis=0) / len(universe_df)
valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

matrices = {}
for fp in sorted(mat_dir.glob("*.parquet")):
    df = pd.read_parquet(fp)
    cols = [c for c in valid_tickers if c in df.columns]
    if cols:
        matrices[fp.stem] = df[cols]

for name in list(matrices.keys()):
    matrices[name] = matrices[name].loc[VAL_START:VAL_END]
universe_val = universe_df[valid_tickers].loc[VAL_START:VAL_END]

close = matrices["close"]
returns_pct = close.pct_change()
tickers = close.columns.tolist()
dates = close.index

engine = FastExpressionEngine(data_fields=matrices)

# Load alphas
conn = sqlite3.connect(DB_PATH)
alphas = conn.execute("SELECT id, expression FROM alphas WHERE archived=0 ORDER BY id").fetchall()
conn.close()
print(f"Found {len(alphas)} active alphas\n")

# Evaluate each alpha individually on validation SET, with multiple max_weight / neutralization configs
print(f"{'ID':>4} {'Sharpe(mw=0.10)':>16} {'Sharpe(mw=0.05)':>16} {'Sharpe(mw=0.03)':>16} {'TO(0.05)':>10} {'Expression':>40}")
print("-" * 110)

alpha_pnls = {}
per_alpha_results = []

for alpha_id, expression in alphas:
    try:
        alpha_df = engine.evaluate(expression)
        if alpha_df is None or alpha_df.empty:
            continue
    except:
        continue

    results_row = {'id': alpha_id, 'expr': expression}
    for mw in [0.10, 0.05, 0.03]:
        try:
            r = simulate_vectorized_polars(
                alpha_df=alpha_df, returns_df=returns_pct, close_df=close,
                universe_df=universe_val, booksize=BOOKSIZE,
                max_stock_weight=mw, decay=0, delay=0,
                neutralization='market', fees_bps=5.0,
                bars_per_day=BARS_PER_DAY,
            )
            results_row[f'sr_{mw}'] = r.sharpe
            results_row[f'to_{mw}'] = r.turnover
            results_row[f'dd_{mw}'] = r.max_drawdown
            if mw == 0.05:
                alpha_pnls[alpha_id] = np.array(r.daily_pnl)
        except:
            results_row[f'sr_{mw}'] = 0
            results_row[f'to_{mw}'] = 0

    per_alpha_results.append(results_row)
    sr10 = results_row.get('sr_0.1', 0)
    sr05 = results_row.get('sr_0.05', 0)
    sr03 = results_row.get('sr_0.03', 0)
    to05 = results_row.get('to_0.05', 0)
    print(f"#{alpha_id:3d} {sr10:+16.3f} {sr05:+16.3f} {sr03:+16.3f} {to05:10.3f} {expression[:40]:>40}")

# Correlation matrix of daily PnLs
print(f"\n\n{'='*70}")
print("PAIRWISE PNL CORRELATION MATRIX (top half)")
print(f"{'='*70}")
ids = sorted(alpha_pnls.keys())
min_len = min(len(v) for v in alpha_pnls.values())
pnl_mat = np.column_stack([alpha_pnls[i][:min_len] for i in ids])
corr_mat = np.corrcoef(pnl_mat.T)

print(f"\n{'':>6}", end='')
for i in ids:
    print(f"  #{i:<4}", end='')
print()
for i, id_i in enumerate(ids):
    print(f"#{id_i:<4}", end='  ')
    for j, id_j in enumerate(ids):
        c = corr_mat[i, j]
        print(f"{c:+.2f} ", end='')
    print()

# Summary stats
upper = corr_mat[np.triu_indices(len(ids), k=1)]
print(f"\nCorrelation stats: mean={np.mean(upper):.3f} median={np.median(upper):.3f} "
      f"p90={np.percentile(upper, 90):.3f} max={np.max(upper):.3f}")

# Best individual alphas
print(f"\n\n{'='*70}")
print("TOP INDIVIDUAL ALPHAS BY VALIDATION SHARPE (mw=0.05, 5bps fees)")
print(f"{'='*70}")
per_alpha_results.sort(key=lambda x: x.get('sr_0.05', 0), reverse=True)
for r in per_alpha_results[:10]:
    print(f"  #{r['id']:3d} SR={r.get('sr_0.05',0):+.3f} TO={r.get('to_0.05',0):.3f} DD={r.get('dd_0.05',0):.3f} | {r['expr'][:55]}")

# Theoretical combined Sharpe
srs = np.array([r.get('sr_0.05', 0) for r in per_alpha_results])
srs_pos = srs[srs > 0]
if len(srs_pos) > 0:
    est_combined = np.sqrt(np.sum(srs_pos**2)) * 0.5  # rough estimate accounting for some correlation
    print(f"\nTheoretical max combined Sharpe (if uncorrelated): {np.sqrt(np.sum(srs_pos**2)):.2f}")
    print(f"Estimated combined (with ~0.5 avg corr): {est_combined:.2f}")
