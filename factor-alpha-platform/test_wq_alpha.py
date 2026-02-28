"""Compute 2019-2023 Sharpe for the WQ alpha replication"""
import os, json, time
import numpy as np
import pandas as pd

from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim import simulate_vectorized
from src.agent.gp_engine import _build_classifications

print("Loading data...")
universe_df = pd.read_parquet("data/fmp_cache/universes/TOP3000.parquet")
ticker_coverage = universe_df.sum(axis=0) / len(universe_df)
tickers = sorted(ticker_coverage[ticker_coverage > 0.05].index.tolist())

needed = ['close', 'open', 'sales', 'invested_capital', 'return_equity', 'liabilities_curr', 'assets', 'volume']
matrices = {}
for f in needed:
    path = f"data/fmp_cache/matrices/{f}.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        valid = [c for c in tickers if c in df.columns]
        matrices[f] = df[valid]

# Apply universe mask
for f, mat in matrices.items():
    cc = mat.columns.intersection(universe_df.columns)
    ci = mat.index.intersection(universe_df.index)
    if len(cc) > 0 and len(ci) > 0:
        mask = universe_df.loc[ci, cc]
        matrices[f] = mat.loc[ci, cc].where(mask)

with open("data/fmp_cache/classifications.json") as f:
    all_cls = json.load(f)
cls_raw = {k: v for k, v in all_cls.items() if k in tickers}
classifications = _build_classifications(cls_raw)

engine = FastExpressionEngine()
for f, mat in matrices.items():
    engine.add_field(f, mat)
for level in ['sector', 'industry', 'subindustry']:
    if level in classifications:
        engine.add_group(level, classifications[level])

close = matrices["close"]
open_prices = matrices.get("open")

print(f"  {close.shape[0]} days x {close.shape[1]} tickers")

# Evaluate
t0 = time.time()
print("\nEvaluating alpha expression...")
alpha_df = engine.evaluate(
    "rank(ts_regression(sales, invested_capital, 252, lag=126, rettype=2)) * "
    "rank(ts_zscore(divide(return_equity, invested_capital), 252)) * "
    "rank(group_rank(divide(liabilities_curr, assets), subindustry))"
)
print(f"  Eval done in {time.time()-t0:.1f}s")

# Simulate FULL period
returns_df = close.pct_change().shift(-1)
sim_full = simulate_vectorized(
    alpha_df=alpha_df, returns_df=returns_df,
    close_df=close, open_df=open_prices,
    classifications=classifications,
    booksize=20_000_000.0, max_stock_weight=0.01,
    delay=1, neutralization="subindustry", fees_bps=0.0,
)

# Compute 2019-2023 sub-period Sharpe
pnl = sim_full.daily_pnl
booksize = 20_000_000

# 2019-2023 window (WQ measurement period)
mask_2019_2023 = (pnl.index.year >= 2019) & (pnl.index.year <= 2023)
pnl_sub = pnl[mask_2019_2023]
ret_sub = pnl_sub / (booksize * 0.5)
sharpe_sub = float(ret_sub.mean() / ret_sub.std() * np.sqrt(252)) if ret_sub.std() > 0 else 0

print(f"\n{'='*70}")
print(f"  FULL PERIOD (all data):")
print(f"    Sharpe:   {sim_full.sharpe:+.4f}")
print(f"    Returns:  {sim_full.returns_ann:+.2%}")
print(f"    Turnover: {sim_full.turnover:.4f}")
print(f"    Max DD:   {sim_full.max_drawdown:.2%}")
print(f"    PnL:      ${sim_full.total_pnl:,.0f}")

print(f"\n  2019-2023 WINDOW (WQ measurement period):")
print(f"    Sharpe:   {sharpe_sub:+.4f}")
cum_ret = ret_sub.sum()
print(f"    Cum Ret:  {cum_ret:+.2%}")
print(f"    Avg Ann:  {cum_ret/5:+.2%}")

# Year by year
print(f"\n  Year-by-year:")
for yr in range(2016, 2027):
    yr_mask = pnl.index.year == yr
    yr_pnl = pnl[yr_mask]
    if len(yr_pnl) == 0:
        continue
    yr_ret = yr_pnl / (booksize * 0.5)
    yr_sharpe = float(yr_ret.mean() / yr_ret.std() * np.sqrt(252)) if yr_ret.std() > 0 else 0
    yr_cum = yr_pnl.cumsum()
    yr_dd = float((yr_cum - yr_cum.cummax()).min() / booksize)
    tag = " <-- WQ window" if 2019 <= yr <= 2023 else ""
    print(f"    {yr}: Sharpe={yr_sharpe:+.2f}  Ret={yr_ret.sum():+.2%}  DD={yr_dd:+.2%}{tag}")

print(f"\n  WQ reported:     Sharpe = 2.1  (2019-2023)")
print(f"  Our replication: Sharpe = {sharpe_sub:+.4f}  (2019-2023)")
print(f"  Gap: {abs(2.1 - sharpe_sub):.4f}")
print(f"{'='*70}")
