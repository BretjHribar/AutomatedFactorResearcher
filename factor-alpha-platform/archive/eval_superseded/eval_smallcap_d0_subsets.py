"""
Test single-alpha and subset portfolios. The 6-alpha equal_weight gets net SR
~5.9 train / 4.5 full, which is BELOW the strongest single alpha (#36 SR=9.4
gross train) — combining redundant alphas dilutes the signal.

Try:
  1. Single alpha #36 alone (the strongest)
  2. Top-2 most distinct: #36 + #39 (corr 0.58, lowest)
  3. Top-3 distinct: #36 + #33 + #39 (best diversification subset)
  4. All 6 with each combiner (already done — for comparison)
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, ".")
from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import combiner_billions, combiner_equal

UNIV='TOP1500TOP2500'; MAX_W=0.001; FEE_BPS=3.0
TRAIN_END='2022-01-01'; VAL_END='2024-01-01'

uni = pd.read_parquet(f'data/fmp_cache/universes/{UNIV}.parquet').astype(bool)
if not isinstance(uni.index, pd.DatetimeIndex):
    uni.index = pd.to_datetime(uni.index)
cov = uni.sum(axis=0)/len(uni); valid = sorted(cov[cov>0.5].index.tolist())
uni = uni[valid]; dates = uni.index; tickers = uni.columns.tolist()

mats = {}
for fp in sorted(Path('data/fmp_cache/matrices').glob('*.parquet')):
    if fp.stem.startswith('_'): continue
    try: df = pd.read_parquet(fp)
    except: continue
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce'); df = df[df.index.notna()]
    cc = [c for c in df.columns if c in tickers]
    if cc: mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)

close = mats['close']
ret = close.pct_change(fill_method=None)
engine = FastExpressionEngine(data_fields=mats)

cls = pd.read_parquet('data/fmp_cache/matrices/subindustry.parquet')
groups = cls.iloc[-1].reindex(tickers)

def proc(sig):
    s = sig.astype(float).where(uni, np.nan)
    for g in groups.dropna().unique():
        m = (groups==g).values
        if m.any():
            sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    abs_sum = s.abs().sum(axis=1).replace(0, np.nan)
    s = s.div(abs_sum, axis=0).clip(-MAX_W, MAX_W).fillna(0)
    return s

# Pull alphas
conn = sqlite3.connect('data/alpha_results.db')
rows = conn.execute('''SELECT a.id, a.expression, MAX(e.sharpe_is) FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
                       WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%' GROUP BY a.id HAVING MAX(e.sharpe_is)>=5 ORDER BY MAX(e.sharpe_is) DESC''').fetchall()

print("Loading + normalizing alphas ...")
raw = {aid: engine.evaluate(expr) for aid, expr, sr in rows}
normed = {aid: proc(r) for aid, r in raw.items()}

ann = np.sqrt(252)

def evaluate_signal(combined, name):
    combined = combined.div(combined.abs().sum(axis=1).replace(0, np.nan), axis=0)
    combined = combined.clip(-MAX_W, MAX_W).fillna(0)
    nx = ret.shift(-1)
    pnl_g = (combined * nx).sum(axis=1).fillna(0)
    to = combined.diff().abs().sum(axis=1) / 2.0
    pnl_n = pnl_g - to * (FEE_BPS / 1e4)

    def sr_ann(s): return float(s.mean()/s.std()*ann) if s.std()>0 else float('nan')
    def cum(s): return float((s.sum()))
    train_g = pnl_g.loc[:TRAIN_END]; train_n = pnl_n.loc[:TRAIN_END]
    val_g   = pnl_g.loc[TRAIN_END:VAL_END]; val_n = pnl_n.loc[TRAIN_END:VAL_END]
    test_g  = pnl_g.loc[VAL_END:]; test_n = pnl_n.loc[VAL_END:]
    full_g  = pnl_g; full_n = pnl_n

    print(f"\n=== {name} ===  (avg TO={to.mean()*100:.1f}%/day, fee={FEE_BPS} bps oneway)")
    print(f"  {'split':6s}  {'gross SR':>8s}  {'net SR':>7s}  {'gross/yr':>8s}  {'net/yr':>8s}")
    for nm, g, n in [('TRAIN', train_g, train_n), ('VAL', val_g, val_n), ('TEST', test_g, test_n), ('FULL', full_g, full_n)]:
        print(f"  {nm:6s}  {sr_ann(g):>+8.2f}  {sr_ann(n):>+7.2f}  {g.mean()*252*100:>+7.1f}%  {n.mean()*252*100:>+7.1f}%")
    return pnl_n

# 1. Single alpha #36
evaluate_signal(normed[36], "#36 alone (close-pos-range)")

# 2. Single alpha #32 (best multiplicative)
evaluate_signal(normed[32], "#32 alone (VWAP × 5d)")

# 3. Top-2 most distinct (#36 + #39, corr 0.58)
combined = (normed[36] + normed[39]) / 2
evaluate_signal(combined, "Equal #36+#39 (top-2 distinct)")

# 4. Top-3 distinct (#36 + #33 + #39)
combined = (normed[36] + normed[33] + normed[39]) / 3
evaluate_signal(combined, "Equal #36+#33+#39 (top-3 distinct)")

# 5. Pure equal-weight all 6
combined = sum(normed.values()) / len(normed)
evaluate_signal(combined, "Equal all 6")

# 6. Billions on top-3 distinct subset
sub_raw = {36: raw[36], 33: raw[33], 39: raw[39]}
billions_3 = combiner_billions(sub_raw, mats, uni, ret, optim_lookback=60, max_wt=MAX_W)
evaluate_signal(billions_3, "Billions on #36+#33+#39")

# 7. Billions on all 6
billions_all = combiner_billions(raw, mats, uni, ret, optim_lookback=60, max_wt=MAX_W)
evaluate_signal(billions_all, "Billions all 6")
