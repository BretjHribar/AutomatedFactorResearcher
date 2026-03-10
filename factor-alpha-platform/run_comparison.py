"""
Comprehensive comparison: SimpleAvg vs QP with matching costs.
1) SimpleAvg at various fee levels (fast, ~1s each)
2) QP trade_aversion sweep (50 tickers, ~8s each to find optimal)
3) Best QP on full sample
"""
import sys, os, json, sqlite3, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

RF = 'comparison_results.txt'
out = open(RF, 'w')
def log(msg):
    print(msg, flush=True)
    out.write(msg + '\n')
    out.flush()

log("=" * 80)
log("COMPREHENSIVE COMPARISON: SimpleAvg vs QP")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 80)

# ── Load data ──
t0 = time.time()
log("\n[1/4] Loading data...")
universe_df = pd.read_parquet('data/fmp_cache/universes/TOP1000.parquet')
ui = universe_df.loc['2020-01-01':'2024-01-01']
tc = ui.sum(axis=0) / len(ui)
tickers = sorted(tc[tc > 0.3].index.tolist())
log(f"  {len(tickers)} tickers")

matrices = {}
mdir = 'data/fmp_cache/matrices_clean'
for fn in sorted(os.listdir(mdir)):
    if not fn.endswith('.parquet'): continue
    df = pd.read_parquet(f'{mdir}/{fn}')
    vc = [c for c in tickers if c in df.columns]
    if vc: matrices[fn.replace('.parquet', '')] = df[vc]
for f, m in matrices.items():
    if isinstance(m, pd.DataFrame) and m.shape[1] > 1:
        cc = m.columns.intersection(universe_df.columns)
        ci = m.index.intersection(universe_df.index)
        if len(cc) > 0 and len(ci) > 0:
            matrices[f] = m.loc[ci, cc].where(universe_df.loc[ci, cc])

with open('data/fmp_cache/classifications.json') as f: all_cls = json.load(f)
cls = {k: v for k, v in all_cls.items() if k in tickers}
from src.operators.fastexpression import FastExpressionEngine
engine = FastExpressionEngine(data_fields=matrices)
cs = {}
for lev in ['sector', 'industry', 'subindustry']:
    mp = {s: cd.get(lev, 'Unk') for s, cd in cls.items() if isinstance(cd, dict)}
    if mp: cs[lev] = pd.Series(mp)
for gn, gs in cs.items(): engine.add_group(gn, gs)

conn = sqlite3.connect('data/alpha_gp_pipeline.db')
cur = conn.cursor()
cur.execute("SELECT DISTINCT a.expression FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id WHERE e.sharpe >= 1.0 ORDER BY e.sharpe DESC")
alpha_expressions = [r[0] for r in cur.fetchall()]
conn.close()
log(f"  {len(alpha_expressions)} alphas, load={time.time()-t0:.0f}s")

# ── Part 1: SimpleAvg at various fee levels ──
log("\n" + "=" * 80)
log("[2/4] SIMPLE AVERAGE — full universe, various fee levels")
log("=" * 80)

from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

# Evaluate and combine alpha signals
log("  Evaluating alphas...", )
alpha_dfs = []
for i, expr in enumerate(alpha_expressions):
    try:
        a = engine.evaluate(expr)
        if a is not None and not a.empty:
            alpha_dfs.append(a)
    except:
        pass
log(f"  {len(alpha_dfs)} alphas evaluated")

# Rank-normalize and average (matching the GP pipeline)
returns_df = matrices['returns']
close_df = matrices['close']
common_cols = sorted(set.intersection(*[set(a.columns) for a in alpha_dfs]) & set(returns_df.columns))
common_idx = sorted(set.intersection(*[set(a.index) for a in alpha_dfs]) & set(returns_df.index))

# Build combined signal
combined = None
for a in alpha_dfs:
    ranked = a.reindex(index=common_idx, columns=common_cols).rank(axis=1, pct=True) - 0.5
    if combined is None:
        combined = ranked
    else:
        combined = combined + ranked
combined = combined / len(alpha_dfs)

log(f"  Combined signal: {combined.shape}")

# Run at various fee levels
log(f"\n  {'Fees':>6} | {'Sharpe':>7} | {'PnL':>12} | {'Turnover':>8} | {'Fitness':>7}")
log(f"  {'-'*55}")

sa_results = {}
for fees in [0, 3, 5, 6, 8, 10, 12]:
    result = sim_vec(
        alpha_df=combined,
        returns_df=returns_df,
        close_df=close_df,
        classifications=cs,
        universe_df=universe_df,
        booksize=20_000_000.0,
        max_stock_weight=0.01,
        decay=0,
        delay=1,
        neutralization="subindustry",
        fees_bps=float(fees),
    )
    sa_results[fees] = result
    log(f"  {fees:>4}bp | {result.sharpe:+7.2f} | ${result.total_pnl:>11,.0f} | "
        f"{result.turnover:>7.1%} | {result.fitness:>7.2f}")

# Compute effective all-in cost for SimpleAvg to match QP cost model
# QP has: 6bps linear + sqrt impact + borrow
# Let's compute what the impact and borrow would be post-hoc on SimpleAvg positions
log("\n  Post-hoc cost computation (matching QP cost model):")
positions = sa_results[0].positions  # zero-fee positions
pos_np = positions.values
trades = np.diff(pos_np, axis=0, prepend=0)
trade_notional_daily = np.sum(np.abs(trades), axis=1)

# Linear: 6bps on trade notional
linear_total = np.sum(trade_notional_daily) * 6e-4

# Impact: for each stock each day, impact = 0.1 * sigma * sqrt(|trade|/ADV) * |trade|
adv_df = matrices.get('adv20', matrices.get('dollars_traded', pd.DataFrame()))
vol_df = matrices.get('historical_volatility_60', pd.DataFrame())
if not adv_df.empty and not vol_df.empty:
    adv_vals = adv_df.reindex(index=positions.index, columns=positions.columns).fillna(1e6).values
    vol_vals = vol_df.reindex(index=positions.index, columns=positions.columns).fillna(0.02).values
    safe_adv = np.maximum(adv_vals, 1e3)
    sigma = vol_vals / np.sqrt(252)  # daily vol
    abs_trades = np.abs(trades)
    participation = abs_trades / safe_adv
    impact_daily = 0.1 * sigma * np.sqrt(participation) * abs_trades
    impact_total = np.nansum(impact_daily)
else:
    impact_total = 0

# Borrow: 0.12 bps/day on short notional
short_pos = np.minimum(pos_np, 0)
short_notional_daily = np.sum(np.abs(short_pos), axis=1)
borrow_total = np.sum(short_notional_daily) * 0.12e-4

total_all_in = linear_total + impact_total + borrow_total
gross_pnl = sa_results[0].total_pnl
net_pnl = gross_pnl - total_all_in

n_days = len(positions)
daily_pnl_gross = sa_results[0].daily_pnl.values
daily_costs = np.zeros(n_days)
daily_costs += trade_notional_daily * 6e-4
if not adv_df.empty and not vol_df.empty:
    daily_costs += np.nansum(impact_daily, axis=1)
daily_costs += short_notional_daily * 0.12e-4
daily_pnl_net = daily_pnl_gross - daily_costs
net_sharpe = np.mean(daily_pnl_net) / np.std(daily_pnl_net) * np.sqrt(252) if np.std(daily_pnl_net) > 0 else 0

log(f"  Linear (6bps):  ${linear_total:>12,.0f}")
log(f"  Impact (sqrt):  ${impact_total:>12,.0f}")
log(f"  Borrow (0.12bp):${borrow_total:>12,.0f}")
log(f"  Total costs:    ${total_all_in:>12,.0f}")
log(f"  Gross PnL:      ${gross_pnl:>12,.0f}")
log(f"  Net PnL:        ${net_pnl:>12,.0f}")
log(f"  Net Sharpe:     {net_sharpe:+.2f}")

# ── Part 2: QP trade_aversion sweep (50 tickers for speed) ──
log("\n" + "=" * 80)
log("[3/4] QP TRADE AVERSION SWEEP — 50 tickers, 1yr")
log("=" * 80)

from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline

tickers_small = tickers[:50]
matrices_small = {}
for f in matrices:
    cols = [c for c in tickers_small if c in matrices[f].columns]
    if cols:
        matrices_small[f] = matrices[f][cols].loc['2019-06-01':'2021-01-01']
cls_small = {k: v for k, v in cls.items() if k in tickers_small}
engine_small = FastExpressionEngine(data_fields=matrices_small)
for gn, gs in cs.items():
    engine_small.add_group(gn, gs)

log(f"\n  {'TradeAv':>8} | {'Sharpe':>7} | {'Gross':>7} | {'PnL':>12} | {'TO':>6} | {'Costs':>10} | {'Time':>5}")
log(f"  {'-'*70}")

best_ta, best_sharpe = 0.0, -999
for ta in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    t1 = time.time()
    config = PipelineConfig(
        is_start='2020-01-01', oos_start='2020-07-01', warmup_days=60,
        booksize=20_000_000.0, risk_aversion=500.0,
        slippage_bps=3.0, commission_bps=3.0, impact_coeff=0.1, borrow_cost_bps=0.12,
        ema_halflife_risk=60, ema_halflife_alpha=60,
        dollar_neutral=True, sector_neutral=True,
        max_position_pct_gmv=0.01, max_position_pct_adv=0.05, delay=1,
        raw_signal_mode=True, trade_aversion=ta,
    )
    pipeline = IsichenkoPipeline(config)
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices_small, classifications=cls_small,
        universe_df=universe_df, expr_engine=engine_small,
    )
    dt = time.time() - t1
    if results:
        s = results.get('full', {})
        if s:
            log(f"  {ta:>8.3f} | {s['sharpe']:+7.2f} | {s['gross_sharpe']:+7.2f} | "
                f"${s['cum_pnl']:>11,.0f} | {s['avg_turnover']:5.1%} | "
                f"${s['total_tcost']:>9,.0f} | {dt:4.0f}s")
            if s['sharpe'] > best_sharpe:
                best_sharpe = s['sharpe']
                best_ta = ta

log(f"\n  Best trade_aversion: {best_ta} (Sharpe={best_sharpe:+.2f})")

# ── Part 3: Best QP on full sample ──
log("\n" + "=" * 80)
log(f"[4/4] BEST QP ON FULL SAMPLE — trade_aversion={best_ta}")
log("=" * 80)

t1 = time.time()
config = PipelineConfig(
    is_start='2020-01-01', oos_start='2023-01-01', warmup_days=120,
    booksize=20_000_000.0, risk_aversion=500.0,
    slippage_bps=3.0, commission_bps=3.0, impact_coeff=0.1, borrow_cost_bps=0.12,
    ema_halflife_risk=60, ema_halflife_alpha=120,
    dollar_neutral=True, sector_neutral=True,
    max_position_pct_gmv=0.01, max_position_pct_adv=0.05, delay=1,
    raw_signal_mode=True, trade_aversion=best_ta,
)
pipeline = IsichenkoPipeline(config)
results = pipeline.run(
    alpha_expressions=alpha_expressions,
    matrices=matrices, classifications=cls,
    universe_df=universe_df, expr_engine=engine,
)
elapsed = time.time() - t1

if results:
    log(f"\n  Results ({elapsed:.0f}s):")
    log(f"  {'Period':>15} | {'NetShr':>6} | {'GrsShr':>6} | {'PnL':>12} | {'DD':>7} | {'TO':>6} | {'Costs':>10} | {'AvgGMV':>10}")
    log(f"  {'-'*90}")
    for period in ['full']:
        s = results.get(period, {})
        if s:
            log(f"  {s['label']:>15} | {s['sharpe']:+6.2f} | {s['gross_sharpe']:+6.2f} | "
                f"${s['cum_pnl']:>11,.0f} | {s['max_drawdown']:+6.1%} | "
                f"{s['avg_turnover']:5.1%} | ${s['total_tcost']:>9,.0f} | ${s['avg_gmv']:>9,.0f}")

# ── Final comparison table ──
log("\n" + "=" * 80)
log("FINAL COMPARISON (FULL period, same cost model)")
log("=" * 80)
log(f"  {'Method':>25} | {'Sharpe':>7} | {'PnL':>12} | {'TO/day':>7} | {'Costs':>10}")
log(f"  {'-'*75}")
log(f"  {'SimpleAvg (0 bps)':>25} | {sa_results[0].sharpe:+7.2f} | ${sa_results[0].total_pnl:>11,.0f} | {sa_results[0].turnover:>6.1%} | $        0")
log(f"  {'SimpleAvg (6bps flat)':>25} | {sa_results[6].sharpe:+7.2f} | ${sa_results[6].total_pnl:>11,.0f} | {sa_results[6].turnover:>6.1%} | (flat 6bp)")
log(f"  {'SimpleAvg (full cost model)':>25} | {net_sharpe:+7.2f} | ${net_pnl:>11,.0f} | {sa_results[0].turnover:>6.1%} | ${total_all_in:>9,.0f}")

if results:
    s = results['full']
    log(f"  {'QP (ta=' + str(best_ta) + ')':>25} | {s['sharpe']:+7.2f} | ${s['cum_pnl']:>11,.0f} | {s['avg_turnover']:>6.1%} | ${s['total_tcost']:>9,.0f}")

log(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total runtime: {time.time()-t0:.0f}s")
out.close()
print(f"\n✅ Results in {RF}", flush=True)
