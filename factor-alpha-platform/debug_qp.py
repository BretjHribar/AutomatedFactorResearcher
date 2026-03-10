"""Debug QP optimizer v3 — write results to file."""
import sys, os, json, sqlite3, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

out = open('debug_qp_results.txt', 'w')
def log(msg):
    print(msg, flush=True)
    out.write(msg + '\n')
    out.flush()

log("=" * 70)
log("QP OPTIMIZER DEBUG v3 — Fixed impact, results to file")
log("=" * 70)

t0 = time.time()
log("\n[1] Loading data...")
universe_df = pd.read_parquet('data/fmp_cache/universes/TOP1000.parquet')
ui = universe_df.loc['2020-01-01':'2021-01-01']
tc = ui.sum(axis=0) / len(ui)
tickers = sorted(tc[tc > 0.3].index.tolist())[:50]
log(f"  {len(tickers)} tickers")

matrices = {}
mdir = 'data/fmp_cache/matrices_clean'
for fn in sorted(os.listdir(mdir)):
    if not fn.endswith('.parquet'): continue
    df = pd.read_parquet(f'{mdir}/{fn}')
    vc = [c for c in tickers if c in df.columns]
    if vc: matrices[fn.replace('.parquet', '')] = df[vc]
for f in list(matrices.keys()):
    matrices[f] = matrices[f].loc['2019-06-01':'2021-01-01']

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

log("\n[2] Impact cost check:")
trade, adv_val, sig = 200_000, 10_000_000, 0.015
old = 0.1 * sig / np.sqrt(adv_val) * trade**2
new = 0.1 * sig * np.sqrt(trade/adv_val) * trade
log(f"  OLD: ${old:,.0f}  NEW: ${new:,.0f}  Linear(6bp): ${trade*6e-4:,.0f}")

log("\n[3] Kappa sweep (1yr, 50 tickers):")
log(f"{'kappa':>6} | {'Period':>12} | {'NetShr':>6} | {'GrsShr':>6} | {'PnL':>12} | {'DD':>7} | {'TO':>6} | {'Costs':>10} | {'Time':>5}")
log("-" * 95)

from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline

for kappa in [10, 50, 100, 500]:
    t1 = time.time()
    config = PipelineConfig(
        is_start='2020-01-01', oos_start='2020-07-01', warmup_days=60,
        booksize=20_000_000.0, risk_aversion=float(kappa),
        slippage_bps=3.0, commission_bps=3.0, impact_coeff=0.1, borrow_cost_bps=0.12,
        ema_halflife_risk=60, ema_halflife_alpha=60,
        dollar_neutral=True, sector_neutral=True,
        max_position_pct_gmv=0.01, max_position_pct_adv=0.05, delay=1,
        raw_signal_mode=True, trade_aversion=0.0,
    )
    pipeline = IsichenkoPipeline(config)
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices, classifications=cls,
        universe_df=universe_df, expr_engine=engine,
    )
    dt = time.time() - t1
    if results:
        for period in ['full', 'is', 'oos']:
            s = results.get(period, {})
            if s:
                log(f"{kappa:6d} | {s['label']:>12s} | {s['sharpe']:+6.2f} | {s['gross_sharpe']:+6.2f} | ${s['cum_pnl']:>11,.0f} | {s['max_drawdown']:+6.1%} | {s['avg_turnover']:5.1%} | ${s['total_tcost']:>9,.0f} | {dt:4.0f}s")
    log("")

out.close()
print("\n✅ Results written to debug_qp_results.txt", flush=True)
