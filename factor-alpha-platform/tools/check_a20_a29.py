"""Reproduce stored evals for A20-A29 using the EXACT framework from
update_wq_alphas_db.py — answers: does the current data match what produced
the stored DB sharpes?
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

con = sqlite3.connect(str(DB_PATH))
rows = con.execute("""
    SELECT a.id, a.expression,
           e.sharpe_is, e.turnover, e.ic_mean, e.return_ann, e.train_start, e.train_end, e.n_bars
    FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
    WHERE a.id BETWEEN 20 AND 29 ORDER BY a.id""").fetchall()

print()
print(f"{'id':>3} | {'DB_SR':>6} {'my_SR':>6} {'dSR':>5} | "
      f"{'DB_TO':>6} {'my_TO':>6} | {'DB_IC':>8} {'my_IC':>8} | "
      f"{'DB_RA%':>7} {'my_RA%':>7} | n_bars")
print("-" * 100)
for (aid, expr, sr_db, to_db, ic_db, ra_db, tstart, tend, nb_db) in rows:
    sig = engine.evaluate(expr).replace([np.inf, -np.inf], np.nan)
    demean = sig.sub(sig.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    w = demean.div(gross, axis=0).fillna(0)
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    port = (w_a.shift(1) * r_a).sum(axis=1).dropna()
    to = (w_a - w_a.shift(1)).abs().sum(axis=1)
    m = (port.index >= pd.Timestamp(tstart)) & (port.index <= pd.Timestamp(tend))
    p = port[m]
    tt = to.loc[p.index]
    ann = np.sqrt(BARS_PER_YEAR)
    sr_my = p.mean() / (p.std(ddof=1) + 1e-12) * ann
    to_my = tt.mean()
    ra_my = p.mean() * BARS_PER_YEAR
    r_next = r_a.shift(-1).loc[w_a.index]
    ic_vals = []
    for ts in p.index:
        wi = w_a.loc[ts]; ri = r_next.loc[ts]
        mask = wi.notna() & ri.notna() & (wi.abs() > 1e-12)
        if mask.sum() >= 10:
            ic_vals.append(wi[mask].rank().corr(ri[mask].rank()))
    ic_my = float(np.mean(ic_vals)) if ic_vals else float("nan")
    print(f"{aid:>3} | {sr_db:>+6.2f} {sr_my:>+6.2f} {sr_my-sr_db:>+5.2f} | "
          f"{to_db:>6.3f} {to_my:>6.3f} | {ic_db:>+8.4f} {ic_my:>+8.4f} | "
          f"{ra_db*100:>+6.1f}% {ra_my*100:>+6.1f}% | DB={nb_db} my={len(p)}")
