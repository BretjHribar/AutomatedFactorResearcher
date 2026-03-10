"""
Run the Isichenko Pipeline at multiple Sharpe thresholds.
Loads alphas from the GP database and runs the pipeline for each threshold.
"""
import os, sys, json, time, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.isichenko import PipelineConfig, IsichenkoPipeline
from src.operators.fastexpression import FastExpressionEngine


def load_alphas_from_db(db_path: str, min_sharpe: float) -> list[str]:
    """Load alpha expressions from DB at given Sharpe threshold."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT a.expression
        FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
        WHERE e.sharpe >= ?
        ORDER BY e.sharpe DESC
    """, (min_sharpe,))
    exprs = [row[0] for row in cur.fetchall()]
    conn.close()
    return exprs


def run_pipeline_for_threshold(
    threshold: float,
    config: PipelineConfig,
    matrices: dict,
    classifications: dict,
    universe_df: pd.DataFrame,
    engine: FastExpressionEngine,
) -> dict:
    """Run the full pipeline for a given Sharpe threshold."""
    alpha_expressions = load_alphas_from_db("data/alpha_gp_pipeline.db", threshold)
    
    print(f"\n{'#'*70}")
    print(f"  PIPELINE RUN: Sharpe >= {threshold} ({len(alpha_expressions)} alphas)")
    print(f"{'#'*70}")
    
    for i, expr in enumerate(alpha_expressions[:10]):
        print(f"  {i+1:3d}. {expr[:65]}")
    if len(alpha_expressions) > 10:
        print(f"  ... and {len(alpha_expressions) - 10} more")
    
    pipeline = IsichenkoPipeline(config)
    t0 = time.time()
    results = pipeline.run(
        alpha_expressions=alpha_expressions,
        matrices=matrices,
        classifications=classifications,
        universe_df=universe_df,
        expr_engine=engine,
    )
    elapsed = time.time() - t0
    
    if not results:
        print(f"  Pipeline returned no results!")
        return {"threshold": threshold, "n_alphas": len(alpha_expressions), "error": True}
    
    print(f"\n  ── Results (elapsed: {elapsed:.0f}s) ──")
    for period in ["full", "is", "oos"]:
        s = results.get(period, {})
        if not s:
            continue
        print(f"\n  {s['label']} ({s['n_days']} days):")
        print(f"    Net Sharpe:     {s['sharpe']:+.2f}")
        print(f"    Gross Sharpe:   {s['gross_sharpe']:+.2f}")
        print(f"    Ann Return:     {s['ann_return']:+.1%}")
        print(f"    Max Drawdown:   {s['max_drawdown']:.1%}")
        print(f"    Calmar:         {s['calmar']:.2f}")
        print(f"    Cum PnL:        ${s['cum_pnl']:,.0f}")
        print(f"    Daily Turnover: {s.get('avg_daily_turnover', 0):.1%}")
        print(f"    Tx Costs:       ${s.get('total_costs', 0):,.0f}")
    
    return {
        "threshold": threshold,
        "n_alphas": len(alpha_expressions),
        "elapsed": elapsed,
        "full": results.get("full", {}),
        "is": results.get("is", {}),
        "oos": results.get("oos", {}),
    }


def main():
    t_start = time.time()
    
    # ── Config ──
    config = PipelineConfig(
        is_start="2020-01-01",
        oos_start="2024-01-01",
        warmup_days=120,
        booksize=20_000_000.0,
        risk_aversion=1e-6,
        slippage_bps=1.0,
        impact_coeff=0.1,
        ema_halflife_risk=60,
        ema_halflife_alpha=120,
        dollar_neutral=True,
        sector_neutral=True,
        max_position_pct_gmv=0.02,
        max_position_pct_adv=0.05,
        delay=1,
    )
    
    # ── Load Data (once) ──
    print("Loading data...")
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
    ui = universe_df.loc[config.is_start:config.oos_start]
    tc = ui.sum(axis=0) / len(ui)
    tickers = sorted(tc[tc > config.min_coverage].index.tolist())
    print(f"  Universe: TOP1000, {len(tickers)} tickers")
    
    matrices = {}
    mdir = "data/fmp_cache/matrices_clean" if os.path.isdir("data/fmp_cache/matrices_clean") else "data/fmp_cache/matrices"
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".parquet") or fn.startswith("_"):
            continue
        df = pd.read_parquet(f"{mdir}/{fn}")
        vc = [c for c in tickers if c in df.columns]
        if vc:
            matrices[fn.replace(".parquet", "")] = df[vc]
    print(f"  Loaded {len(matrices)} data fields")
    
    # Apply universe mask
    for f, m in matrices.items():
        if isinstance(m, pd.DataFrame) and m.shape[1] > 1:
            cc = m.columns.intersection(universe_df.columns)
            ci = m.index.intersection(universe_df.index)
            if len(cc) > 0 and len(ci) > 0:
                matrices[f] = m.loc[ci, cc].where(universe_df.loc[ci, cc])
    
    with open("data/fmp_cache/classifications.json") as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in tickers}
    
    # Expression engine
    engine = FastExpressionEngine(data_fields=matrices)
    cs = {}
    for lev in ["sector", "industry", "subindustry"]:
        mp = {s: cd.get(lev, "Unk") for s, cd in classifications.items() if isinstance(cd, dict)}
        if mp:
            cs[lev] = pd.Series(mp)
    for gn, gs in cs.items():
        engine.add_group(gn, gs)
    
    print(f"\n  Data loaded in {time.time() - t_start:.0f}s")
    
    # ── Run at 3 thresholds ──
    thresholds = [0.9, 0.75, 0.6]
    all_results = []
    
    for threshold in thresholds:
        result = run_pipeline_for_threshold(
            threshold, config, matrices, classifications, universe_df, engine
        )
        all_results.append(result)
    
    # ── Comparison Summary ──
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Threshold':>11s} | {'#Alpha':>6s} | {'IS Sharpe':>10s} | {'OOS Sharpe':>10s} | {'OOS Cum PnL':>12s} | {'Max DD':>8s} | {'Time':>6s}")
    print("-" * 80)
    for r in all_results:
        if r.get("error"):
            print(f"  >= {r['threshold']:.2f}  | {r['n_alphas']:6d} | {'ERROR':>10s} |")
            continue
        is_s = r.get("is", {})
        oos_s = r.get("oos", {})
        print(f"  >= {r['threshold']:.2f}  | {r['n_alphas']:6d} "
              f"| {is_s.get('sharpe', 0):>+10.2f} "
              f"| {oos_s.get('sharpe', 0):>+10.2f} "
              f"| ${oos_s.get('cum_pnl', 0):>11,.0f} "
              f"| {oos_s.get('max_drawdown', 0):>7.1%} "
              f"| {r.get('elapsed', 0):>5.0f}s")
    
    print(f"\n  Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
