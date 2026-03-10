"""Full sample QP pipeline run — all tickers, 2020-2026."""
import sys, os, json, sqlite3, warnings, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

RESULTS_FILE = 'full_qp_results.txt'
out = open(RESULTS_FILE, 'w')
def log(msg):
    print(msg, flush=True)
    out.write(msg + '\n')
    out.flush()

log("=" * 70)
log("FULL SAMPLE QP RUN — All tickers, 2020-2026, kappa=500")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

# ── Load data ──
t0 = time.time()
log("\n[1/3] Loading data...")

universe_df = pd.read_parquet('data/fmp_cache/universes/TOP1000.parquet')
ui = universe_df.loc['2020-01-01':'2024-01-01']
tc = ui.sum(axis=0) / len(ui)
tickers = sorted(tc[tc > 0.3].index.tolist())
log(f"  {len(tickers)} tickers")

matrices = {}
mdir = 'data/fmp_cache/matrices_clean'
for fn in sorted(os.listdir(mdir)):
    if not fn.endswith('.parquet'):
        continue
    df = pd.read_parquet(f'{mdir}/{fn}')
    vc = [c for c in tickers if c in df.columns]
    if vc:
        matrices[fn.replace('.parquet', '')] = df[vc]

# Apply universe mask
for f, m in matrices.items():
    if isinstance(m, pd.DataFrame) and m.shape[1] > 1:
        cc = m.columns.intersection(universe_df.columns)
        ci = m.index.intersection(universe_df.index)
        if len(cc) > 0 and len(ci) > 0:
            matrices[f] = m.loc[ci, cc].where(universe_df.loc[ci, cc])

with open('data/fmp_cache/classifications.json') as f:
    all_cls = json.load(f)
cls = {k: v for k, v in all_cls.items() if k in tickers}

from src.operators.fastexpression import FastExpressionEngine
engine = FastExpressionEngine(data_fields=matrices)
cs = {}
for lev in ['sector', 'industry', 'subindustry']:
    mp = {s: cd.get(lev, 'Unk') for s, cd in cls.items() if isinstance(cd, dict)}
    if mp:
        cs[lev] = pd.Series(mp)
for gn, gs in cs.items():
    engine.add_group(gn, gs)

conn = sqlite3.connect('data/alpha_gp_pipeline.db')
cur = conn.cursor()
cur.execute("""SELECT DISTINCT a.expression 
               FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id 
               WHERE e.sharpe >= 1.0 ORDER BY e.sharpe DESC""")
alpha_expressions = [r[0] for r in cur.fetchall()]
conn.close()

log(f"  {len(alpha_expressions)} alphas")
log(f"  {len(matrices)} matrices")
log(f"  Data range: {matrices['close'].index[0]} to {matrices['close'].index[-1]}")
log(f"  Load time: {time.time()-t0:.0f}s")

# ── Run both modes: EqWt+QP and IC-Wt+QP ──
from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline

configs = [
    ("EqWt+QP (raw)", True),
    ("IC-Wt+QP", False),
]

log("\n[2/3] Running backtests...")
log(f"  Config: kappa=500, max_pos=1%, slippage=3bp, commission=3bp")
log(f"  Period: IS=2020-01-01 to 2023-01-01, OOS=2023-01-01+")
log("")

all_results = {}
for mode_name, raw_mode in configs:
    log(f"{'='*70}")
    log(f"  {mode_name}")
    log(f"{'='*70}")
    
    t1 = time.time()
    config = PipelineConfig(
        is_start='2020-01-01',
        oos_start='2023-01-01',
        warmup_days=120,
        booksize=20_000_000.0,
        risk_aversion=500.0,
        slippage_bps=3.0,
        commission_bps=3.0,
        impact_coeff=0.1,
        borrow_cost_bps=0.12,
        ema_halflife_risk=60,
        ema_halflife_alpha=120,
        dollar_neutral=True,
        sector_neutral=True,
        max_position_pct_gmv=0.01,
        max_position_pct_adv=0.05,
        delay=1,
        raw_signal_mode=raw_mode,
        trade_aversion=0.0,
    )
    
    pipeline = IsichenkoPipeline(config)
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices,
        classifications=cls,
        universe_df=universe_df,
        expr_engine=engine,
    )
    elapsed = time.time() - t1
    all_results[mode_name] = results
    
    if results:
        log(f"\n  Results for {mode_name} ({elapsed:.0f}s):")
        log(f"  {'Period':>15} | {'NetShr':>6} | {'GrsShr':>6} | {'PnL':>12} | {'DD':>8} | {'TO':>6} | {'Costs':>10} | {'AvgGMV':>10}")
        log(f"  {'-'*85}")
        for period in ['full', 'is', 'oos']:
            s = results.get(period, {})
            if s:
                log(f"  {s['label']:>15} | {s['sharpe']:+6.2f} | {s['gross_sharpe']:+6.2f} | "
                    f"${s['cum_pnl']:>11,.0f} | {s['max_drawdown']:+7.1%} | "
                    f"{s['avg_turnover']:5.1%} | ${s['total_tcost']:>9,.0f} | ${s['avg_gmv']:>9,.0f}")
    log("")

# ── Summary comparison ──
log("\n[3/3] SUMMARY COMPARISON")
log("=" * 70)
log(f"{'Mode':>15} | {'IS Net':>8} | {'IS Gross':>8} | {'OOS Net':>8} | {'OOS Gross':>9} | {'OOS PnL':>10}")
log("-" * 70)

for mode_name in [n for n, _ in configs]:
    r = all_results.get(mode_name, {})
    is_s = r.get('is', {})
    oos_s = r.get('oos', {})
    if is_s and oos_s:
        log(f"{mode_name:>15} | {is_s['sharpe']:+8.2f} | {is_s['gross_sharpe']:+8.2f} | "
            f"{oos_s['sharpe']:+8.2f} | {oos_s['gross_sharpe']:+9.2f} | ${oos_s['cum_pnl']:>9,.0f}")

log(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total runtime: {time.time()-t0:.0f}s")
out.close()
print(f"\n✅ Full results in {RESULTS_FILE}", flush=True)
