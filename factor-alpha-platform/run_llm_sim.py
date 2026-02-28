"""
LLM-style 20-trial alpha evaluation — TOP1000 universe.
Loads data the same way as run_gp_top1000.py, evaluates like Gemini calling WQ.
"""
import os, json, time, logging
import numpy as np
import pandas as pd

from src.data.context_research import InMemoryDataContext
from src.data.alpha_database import AlphaDatabase
from src.operators.fastexpression import FastExpressionEngine, create_engine_from_context
from src.simulation.vectorized_sim import simulate_vectorized
from src.agent.gp_engine import _build_classifications

logging.basicConfig(level=logging.WARNING)

# ── Load TOP1000 ──────────────────────────────────────────────────────
print("Loading TOP1000 universe...")
universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
ticker_coverage = universe_df.sum(axis=0) / len(universe_df)
top1000_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())
print(f"  {len(top1000_tickers)} tickers")

matrices = {}
for fname in os.listdir("data/fmp_cache/matrices"):
    if not fname.endswith(".parquet") or fname.startswith("_"): continue
    field = fname.replace(".parquet", "")
    df = pd.read_parquet(os.path.join("data/fmp_cache/matrices", fname))
    valid_cols = [c for c in top1000_tickers if c in df.columns]
    matrices[field] = df[valid_cols]

# Apply universe mask
for field, mat in matrices.items():
    common_cols = mat.columns.intersection(universe_df.columns)
    common_idx = mat.index.intersection(universe_df.index)
    if len(common_cols) > 0 and len(common_idx) > 0:
        mask = universe_df.loc[common_idx, common_cols]
        matrices[field] = mat.loc[common_idx, common_cols].where(mask)

with open("data/fmp_cache/classifications.json") as f:
    all_cls = json.load(f)
cls_raw = {k: v for k, v in all_cls.items() if k in top1000_tickers}

ctx = InMemoryDataContext()
ctx.load_from_matrices(matrices, cls_raw)
engine = create_engine_from_context(ctx)
classifications = _build_classifications(cls_raw)

close = matrices["close"]
open_prices = matrices.get("open")
returns_df = close.pct_change().shift(-1)

print(f"  {close.shape[0]} days × {close.shape[1]} tickers, {len(matrices)} fields")
print(f"  GICS sub-industries: {classifications['subindustry'].nunique()}")

db = AlphaDatabase("data/alpha.db")
run_id = db.create_run(strategy="llm_sim_20_trials", llm_model="gemini_simulated",
                       config={"neutralization": "subindustry", "delay": 1,
                               "booksize": 20_000_000, "universe": "TOP1000"})

# ── 20 Alpha Expressions ─────────────────────────────────────────────
TRIALS = [
    ("Earnings yield",          "rank(earnings_yield)"),
    ("Book-to-price",           "rank(divide(bookvalue_ps, close))"),
    ("Sales-to-EV",             "rank(divide(revenue, enterprise_value))"),
    ("60d momentum",            "rank(ts_delta(close, 60))"),
    ("Short-term reversal",     "-1 * rank(ts_delta(close, 5))"),
    ("Vol-adj momentum",        "rank(divide(ts_delta(close, 20), ts_std_dev(close, 20)))"),
    ("ROE quality",             "rank(roe)"),
    ("Cash flow yield",         "rank(free_cashflow_yield)"),
    ("EBITDA margin",           "rank(ebitda_margin)"),
    ("VWAP deviation",          "-1 * rank(divide(subtract(close, vwap), vwap))"),
    ("Volume surge",            "rank(divide(volume, ts_mean(volume, 20)))"),
    ("20d rank",                "rank(ts_rank(close, 20))"),
    ("Low vol factor",          "-1 * rank(historical_volatility_20)"),
    ("Parkinson vol revert",    "-1 * rank(parkinson_volatility_30)"),
    ("Revenue growth",          "rank(ts_delta(revenue, 60))"),
    ("FCF/share growth",        "rank(ts_delta(fcf_per_share, 60))"),
    ("Gross margin trend",      "rank(ts_delta(gross_margin, 60))"),
    ("Value + momentum",        "add(rank(book_to_market), rank(ts_delta(close, 20)))"),
    ("Quality + reversal",      "add(rank(roe), -1 * rank(ts_delta(close, 5)))"),
    ("Vol-adj value",           "rank(divide(earnings_yield, historical_volatility_30))"),
]

print(f"\n{'='*80}")
print(f"  20 LLM-style trials | subindustry neut | delay=1 | $20MM | TOP1000")
print(f"{'='*80}")

results = []
stored = 0

for i, (name, expr) in enumerate(TRIALS, 1):
    print(f"\n── Trial {i}/20: {name} ──")
    print(f"   {expr}")
    t0 = time.time()

    try:
        alpha_df = engine.evaluate(expr)
        if alpha_df is None or not isinstance(alpha_df, pd.DataFrame):
            print(f"   ❌ Engine returned None")
            results.append({"i": i, "name": name, "expr": expr, "status": "fail"})
            continue

        sim = simulate_vectorized(
            alpha_df=alpha_df, returns_df=returns_df,
            close_df=close, open_df=open_prices,
            classifications=classifications,
            booksize=20_000_000.0, max_stock_weight=0.01,
            delay=1, neutralization="subindustry", fees_bps=0.0,
        )
        elapsed = time.time() - t0

        # Coverage check
        n_positions = sim.positions.notna().sum(axis=1).mean()
        coverage = n_positions / max(sim.positions.shape[1], 1)

        print(f"   Sharpe:  {sim.sharpe:+.4f}   Fitness: {sim.fitness:+.4f}")
        print(f"   Returns: {sim.returns_ann:+.2%}   Turnover: {sim.turnover:.4f}")
        print(f"   Max DD:  {sim.max_drawdown:.2%}   Margin: {sim.margin_bps:+.1f} bps")
        print(f"   PSR:     {sim.psr:.3f}   Coverage: {coverage:.1%}   [{elapsed:.1f}s]")
        print(f"   Total PnL: ${sim.total_pnl:,.0f}")

        # Year-by-year
        pnl = sim.daily_pnl
        for yr in sorted(set(d.year for d in pnl.index)):
            yr_pnl = pnl[pnl.index.year == yr]
            yr_ret = yr_pnl / 10_000_000
            yr_sharpe = float(yr_ret.mean() / yr_ret.std() * np.sqrt(252)) if yr_ret.std() > 0 else 0
            yr_cum = yr_pnl.cumsum()
            yr_dd = float((yr_cum - yr_cum.cummax()).min() / 20_000_000)
            print(f"     {yr}: Sharpe={yr_sharpe:+.2f} Ret={yr_ret.sum():+.2%} DD={yr_dd:+.2%}")

        results.append({
            "i": i, "name": name, "expr": expr, "status": "ok",
            "sharpe": sim.sharpe, "fitness": sim.fitness,
            "ret": sim.returns_ann, "turnover": sim.turnover,
            "dd": sim.max_drawdown, "margin": sim.margin_bps,
            "psr": sim.psr, "pnl": sim.total_pnl,
        })

        # Store in DB: insert_alpha + insert_evaluation
        params = {"neutralization": "subindustry", "delay": 1,
                  "booksize": 20_000_000, "universe": "TOP1000"}
        alpha_id = db.insert_alpha(
            expression=expr, params=params,
            reasoning=name, source="llm_sim",
            run_id=run_id, trial_index=i,
        )
        db.insert_evaluation(
            alpha_id=alpha_id, source="local",
            sharpe=float(sim.sharpe), fitness=float(sim.fitness),
            turnover=float(sim.turnover), max_drawdown=float(sim.max_drawdown),
            returns_ann=float(sim.returns_ann), profit_dollar=float(sim.total_pnl),
            margin_bps=float(sim.margin_bps),
            passed_checks=1 if abs(sim.sharpe) > 0.3 and sim.turnover > 0.001 else 0,
            metrics={"psr": float(sim.psr), "coverage": float(coverage),
                     "source": "llm_sim", "trial": i, "name": name},
        )
        stored += 1
        tag = "✅" if abs(sim.sharpe) > 0.5 else "📦"
        print(f"   {tag} Stored as alpha #{alpha_id}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback; traceback.print_exc()
        results.append({"i": i, "name": name, "expr": expr, "status": "error", "error": str(e)})

db.finish_run(run_id, status="completed")

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  RESULTS SUMMARY — {stored} alphas stored in run #{run_id}")
print(f"{'='*80}")
print(f"{'#':<3} {'Name':<22} {'Sharpe':>8} {'Fitness':>8} {'Ret':>8} {'T/O':>7} {'DD':>8} {'PnL':>12}")
print("-" * 80)
ok = [r for r in results if r["status"] == "ok"]
for r in sorted(ok, key=lambda x: abs(x["sharpe"]), reverse=True):
    print(f"{r['i']:<3} {r['name']:<22} {r['sharpe']:+8.4f} {r['fitness']:+8.4f} "
          f"{r['ret']:+8.2%} {r['turnover']:7.4f} {r['dd']:+8.2%} ${r['pnl']:>11,.0f}")

# ── DB verification ──────────────────────────────────────────────────
print(f"\n── Database Verification ──")
top = db.get_top_alphas(metric="sharpe", limit=20)
print(f"Total alphas in DB: {len(top)}")
for a in top:
    print(f"  ID={a['alpha_id']:>3} Sharpe={a.get('sharpe',0):+.4f} "
          f"Fitness={a.get('fitness',0):+.4f} T/O={a.get('turnover',0):.4f} "
          f"Pass={a.get('passed_checks',0)} | {a.get('expression','')[:55]}")

# Also show the stats
stats = db.get_stats()
print(f"\n  DB Stats: {json.dumps(stats, indent=2)}")
