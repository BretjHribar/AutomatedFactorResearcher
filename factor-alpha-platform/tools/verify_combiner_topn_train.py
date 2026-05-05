"""Step-1 validator: prove combiner_topn_train (in src/portfolio/combiners.py)
reproduces the canonical crypto baseline.

Targets:
  top_n = 30  -> TRAIN SR ≈ +5.50
  top_n = 50  -> TRAIN SR ≈ +5.30   (the chosen prod-research crypto config)
"""
import sqlite3, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import combiner_topn_train

UNIVERSE_PATH = ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
MATRICES_DIR  = ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH       = ROOT / "data/alphas.db"
COVERAGE_CUTOFF = 0.3
BARS_PER_YEAR = 6 * 365
TRAIN_START = pd.Timestamp("2023-09-01")
TRAIN_END   = pd.Timestamp("2025-09-01")

# Load matrices like update_wq_alphas_db.py
uni = pd.read_parquet(UNIVERSE_PATH)
cov = uni.sum(axis=0) / len(uni)
valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
matrices = {}
for fp in sorted(MATRICES_DIR.glob("*.parquet")):
    if fp.parent.name == "prod":
        continue
    df = pd.read_parquet(fp)
    cols = [c for c in valid if c in df.columns]
    if cols:
        matrices[fp.stem] = df[cols]
tickers = sorted(set(matrices["close"].columns))
for k, v in matrices.items():
    matrices[k] = v[[t for t in tickers if t in v.columns]]
returns = matrices["returns"]
print(f"Loaded {len(matrices)} fields, {len(tickers)} tickers, bars={len(matrices['close'])}")

# Load all archived=0 crypto/4h alphas with stored train Sharpe
con = sqlite3.connect(str(DB_PATH))
rows = con.execute("""
    SELECT a.id, a.expression, e.sharpe_is
    FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
    WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
    ORDER BY a.id""").fetchall()
print(f"  {len(rows)} crypto alphas in DB")

engine = FastExpressionEngine(data_fields=matrices)
alpha_signals = {}
train_sharpes = {}
for aid, expr, sr in rows:
    try:
        alpha_signals[aid] = engine.evaluate(expr)
        train_sharpes[aid] = sr
    except Exception as e:
        print(f"  skip a{aid}: {e}")
print(f"  evaluated {len(alpha_signals)} alphas")

# Backtest helper
def stats(w, label):
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    port = (w_a.shift(1) * r_a).sum(axis=1)
    to   = (w_a - w_a.shift(1)).abs().sum(axis=1)
    m = (port.index >= TRAIN_START) & (port.index <= TRAIN_END)
    p = port[m].dropna()
    ann = np.sqrt(BARS_PER_YEAR)
    sr = p.mean() / (p.std(ddof=1) + 1e-12) * ann
    ra = p.mean() * BARS_PER_YEAR
    print(f"  {label:18s}  n={len(p):>5d}  TRAIN_SR={sr:+5.2f}  ret_ann={ra*100:+6.1f}%  "
          f"TO={to.loc[p.index].mean():.3f}")
    return sr

print()
print("=== combiner_topn_train via library ===")
results = {}
for top_n in (30, 50, 60, 100):
    w = combiner_topn_train(alpha_signals, matrices, uni, returns,
                             train_sharpes=train_sharpes, top_n=top_n)
    results[top_n] = stats(w, f"top_{top_n}")

print()
print("=== assertions ===")
def check(actual, target, tol, label):
    ok = abs(actual - target) <= tol
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {label}: actual={actual:+.2f}  target={target:+.2f}  tol=±{tol}")
    return ok

ok30 = check(results[30], 5.50, 0.10, "top_30 TRAIN SR")
ok50 = check(results[50], 5.30, 0.10, "top_50 TRAIN SR")
print()
print("PASS" if (ok30 and ok50) else "FAIL — combiner_topn_train does not reproduce canonical baseline")
