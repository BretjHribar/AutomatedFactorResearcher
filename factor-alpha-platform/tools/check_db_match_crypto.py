"""Reproduce stored TRAIN sharpe for a sample of crypto alphas — verifies
that the data + framework today is consistent with what produced the DB
values. If reproductions don't match, the data has drifted (or the eval
framework has).

Uses the EXACT framework from update_wq_alphas_db.py (which wrote the DB
values on 2026-04-24). Reads matrices/universe the same way it did then.

Output: side-by-side DB vs reproduced for ~30 alphas spanning the SR range.
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
DB_PATH       = ROOT / "data/alphas.db"   # original crypto DB
COVERAGE_CUTOFF = 0.3
BARS_PER_YEAR = 6 * 365
N_SAMPLE = 30   # how many alphas to test

# Load matrices exactly like update_wq_alphas_db.py
print(f"Loading matrices from {MATRICES_DIR.relative_to(ROOT)}")
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
print(f"  loaded {len(matrices)} fields, {len(tickers)} tickers, "
      f"bars={len(matrices['close'])}")

returns = matrices["returns"]
engine = FastExpressionEngine(data_fields=matrices)


def signal_to_portfolio(sig):
    """Same as update_wq_alphas_db.signal_to_portfolio."""
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


def compute_stats(w, start, end):
    """delay=0 (crypto 24/7): trade at close T, earn close T -> close T+1."""
    common_idx = w.index.intersection(returns.index)
    w_a = w.loc[common_idx].fillna(0)
    r = returns.loc[common_idx].fillna(0)
    port = (w_a * r.shift(-1)).sum(axis=1)
    to = (w_a - w_a.shift(1)).abs().sum(axis=1)
    m = (port.index >= start) & (port.index <= end)
    p = port[m].dropna()
    tt = to.loc[p.index]
    ann = np.sqrt(BARS_PER_YEAR)
    sr = float(p.mean() / (p.std(ddof=1) + 1e-12) * ann)
    ra = float(p.mean() * BARS_PER_YEAR)
    return sr, ra, float(tt.mean()), int(len(p))


# Sample 30 alphas spanning the SR distribution: top 10, mid 10, bottom 10
con = sqlite3.connect(str(DB_PATH))
all_alphas = con.execute("""
    SELECT a.id, a.expression, e.sharpe_is, e.turnover, e.return_ann, e.n_bars,
           e.train_start, e.train_end
    FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
    WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
    ORDER BY e.sharpe_is DESC""").fetchall()
print(f"  {len(all_alphas)} crypto alphas in DB")

# Pick: top 10, mid 10, bottom 10
N = N_SAMPLE // 3
mid_start = (len(all_alphas) - N) // 2
sample = all_alphas[:N] + all_alphas[mid_start:mid_start + N] + all_alphas[-N:]

print()
print(f"{'rank':>5s} {'id':>4s} | {'DB_SR':>6s} {'my_SR':>7s} {'dSR':>6s} | "
      f"{'DB_TO':>6s} {'my_TO':>6s} | {'DB_ra%':>7s} {'my_ra%':>8s} | n_bars")
print("-" * 105)

diffs = []
for rank_in_full, (aid, expr, sr_db, to_db, ra_db, nb_db, tstart, tend) in enumerate(
        sample, start=1):
    # Recover the alpha's full position in the all-sorted ranking
    full_rank = next(i for i, a in enumerate(all_alphas, start=1) if a[0] == aid)
    try:
        sig = engine.evaluate(expr)
        w = signal_to_portfolio(sig)
        sr_my, ra_my, to_my, n_my = compute_stats(
            w, pd.Timestamp(tstart), pd.Timestamp(tend))
    except Exception as e:
        print(f"{full_rank:>5d} {aid:>4d}  ERROR: {type(e).__name__}: {str(e)[:60]}")
        continue
    d_sr = sr_my - sr_db
    diffs.append(d_sr)
    print(f"{full_rank:>5d} {aid:>4d} | {sr_db:>+6.2f} {sr_my:>+7.2f} {d_sr:>+5.2f}  | "
          f"{to_db:>6.3f} {to_my:>6.3f} | {ra_db*100:>+6.1f}% {ra_my*100:>+7.1f}% | "
          f"{nb_db}/{n_my}")

print()
print(f"|d_SR| stats over {len(diffs)} alphas:")
print(f"  mean: {np.mean(np.abs(diffs)):.3f}")
print(f"  max:  {np.max(np.abs(diffs)):.3f}")
print(f"  > 0.5 SR diff: {sum(1 for d in diffs if abs(d) > 0.5)} of {len(diffs)}")
print(f"  > 1.0 SR diff: {sum(1 for d in diffs if abs(d) > 1.0)} of {len(diffs)}")
