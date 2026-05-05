"""Reproduce the equal-weight combination of A20-A29 using the EXACT framework
from update_wq_alphas_db.py. Answer: does the combined portfolio hit SR>5 on TRAIN?
"""
import sqlite3, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.operators.fastexpression import FastExpressionEngine

UNIVERSE_PATH = ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
MATRICES_DIR  = ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH       = ROOT / "data/alphas.db"
COVERAGE_CUTOFF = 0.3
BARS_PER_YEAR = 6 * 365
TRAIN_START = pd.Timestamp("2023-09-01")
TRAIN_END   = pd.Timestamp("2025-09-01")

# Load like update_wq_alphas_db.py
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
print(f"Loaded {len(matrices)} fields, {len(tickers)} tickers, bars={len(matrices['close'])}")

returns = matrices["returns"]
engine = FastExpressionEngine(data_fields=matrices)


def signal_to_portfolio(sig):
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


def stats(port, to, label):
    p = port.dropna()
    ann = np.sqrt(BARS_PER_YEAR)
    sr = p.mean() / (p.std(ddof=1) + 1e-12) * ann
    ra = p.mean() * BARS_PER_YEAR
    print(f"  {label:18s}  n={len(p):>5d}  SR={sr:>+5.2f}  ret_ann={ra*100:>+6.1f}%  TO={to.loc[p.index].mean():>5.3f}")
    return sr


# Choose alpha set via env var ALPHASET in {a20a29, top30, top60, all}
import os
ALPHASET = os.environ.get("ALPHASET", "a20a29")
con = sqlite3.connect(str(DB_PATH))
if ALPHASET == "a20a29":
    rows = con.execute("""SELECT a.id, a.expression FROM alphas a
        WHERE a.id BETWEEN 20 AND 29 ORDER BY a.id""").fetchall()
elif ALPHASET == "top30":
    rows = con.execute("""SELECT a.id, a.expression FROM alphas a
        JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY e.sharpe_is DESC LIMIT 30""").fetchall()
elif ALPHASET == "top60":
    rows = con.execute("""SELECT a.id, a.expression FROM alphas a
        JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY e.sharpe_is DESC LIMIT 60""").fetchall()
elif ALPHASET == "all":
    rows = con.execute("""SELECT a.id, a.expression FROM alphas a
        JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id""").fetchall()
else:
    raise SystemExit(f"unknown ALPHASET={ALPHASET}")
print(f"alpha set = {ALPHASET}, n={len(rows)}")

print(f"\nEvaluating {len(rows)} alphas A20-A29...")

# Each alpha → individual portfolio
ws = {}
for aid, expr in rows:
    sig = engine.evaluate(expr)
    w_a = signal_to_portfolio(sig)
    ws[aid] = w_a

# Method A: average the SIGNALS, then convert (matches "combiner" pattern)
sig_sum = None
for aid, expr in rows:
    sig = engine.evaluate(expr).replace([np.inf, -np.inf], np.nan)
    # z-score each signal cross-sectionally first (so they're on same scale)
    s_zs = sig.sub(sig.mean(axis=1), axis=0)
    s_std = sig.std(axis=1).replace(0, np.nan)
    s_zs = s_zs.div(s_std, axis=0)
    sig_sum = s_zs if sig_sum is None else sig_sum.add(s_zs, fill_value=0)
sig_avg = sig_sum / len(rows)
w_avg_sig = signal_to_portfolio(sig_avg)

# Method B: average the WEIGHTS (each alpha's L1-normed portfolio)
w_sum = None
for aid in ws:
    w_sum = ws[aid] if w_sum is None else w_sum.add(ws[aid], fill_value=0)
w_avg_w = w_sum / len(rows)
# Re-L1-normalize
gross = w_avg_w.abs().sum(axis=1).replace(0, np.nan)
w_avg_w = w_avg_w.div(gross, axis=0).fillna(0)


def backtest_train(w, label):
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    port = (w_a.shift(1) * r_a).sum(axis=1)
    to   = (w_a - w_a.shift(1)).abs().sum(axis=1)
    m = (port.index >= TRAIN_START) & (port.index <= TRAIN_END)
    return stats(port[m], to.loc[port[m].index], label)


print()
print("=== TRAIN window 2023-09-01 → 2025-09-01 ===")
print()
print("Individual alphas:")
for aid, w_a in ws.items():
    backtest_train(w_a, f"A{aid}")
print()
print("Combined portfolios:")
backtest_train(w_avg_sig, "avg(z-scored sig)")
backtest_train(w_avg_w, "avg(weights)")

# Try a few other combinations
print()
print("Sanity — full-history variants of combined:")
for label, w in [("avg(z-sig)", w_avg_sig), ("avg(w)", w_avg_w)]:
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    port = (w_a.shift(1) * r_a).sum(axis=1)
    to   = (w_a - w_a.shift(1)).abs().sum(axis=1)
    stats(port, to, f"{label} FULL")
