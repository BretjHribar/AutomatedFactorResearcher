"""
Run average-signal backtests at multiple Sharpe thresholds.

Instead of the Isichenko pipeline (OLS + QP), this simply:
1. Evaluates each alpha expression
2. Cross-sectionally ranks each alpha (to normalize scales)
3. Averages all ranked signals
4. Runs the vectorized simulation on the combined signal
"""
import os, sys, json, time, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as simulate_vectorized
from src.operators.fastexpression import FastExpressionEngine


def load_alphas_from_db(db_path: str, min_sharpe: float) -> list[str]:
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


def run_avg_backtest(
    threshold: float,
    engine: FastExpressionEngine,
    matrices: dict,
    classifications: dict,
    universe_df: pd.DataFrame,
):
    """Average signals → simulate."""
    exprs = load_alphas_from_db("data/alpha_gp_pipeline.db", threshold)
    
    print(f"\n{'#'*70}")
    print(f"  AVERAGING BACKTEST: Sharpe >= {threshold} ({len(exprs)} alphas)")
    print(f"{'#'*70}")
    
    # Evaluate all alphas
    signals = []
    failed = 0
    for i, expr in enumerate(exprs):
        try:
            alpha = engine.evaluate(expr)
            if alpha is not None and not alpha.empty and alpha.notna().sum().sum() > 0:
                # Cross-sectional rank (normalize to [0,1])
                ranked = alpha.rank(axis=1, pct=True)
                # Demean each row
                ranked = ranked.sub(ranked.mean(axis=1), axis=0)
                signals.append(ranked)
                if i < 10:
                    print(f"  ✅ {i+1:3d}. {expr[:60]}")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if i < 5:
                print(f"  ❌ {i+1:3d}. FAILED: {str(e)[:50]}")
    
    if len(exprs) > 10:
        print(f"  ... {len(exprs) - 10} more")
    print(f"\n  Loaded: {len(signals)}/{len(exprs)} alphas ({failed} failed)")
    
    if not signals:
        print("  No valid signals!")
        return None
    
    # Average all ranked signals
    combined = signals[0].copy()
    for s in signals[1:]:
        # Align
        cc = combined.columns.intersection(s.columns)
        ci = combined.index.intersection(s.index)
        combined = combined.reindex(index=ci, columns=cc)
        combined = combined.add(s.reindex(index=ci, columns=cc), fill_value=0)
    combined = combined / len(signals)
    
    print(f"  Combined signal shape: {combined.shape}")
    
    # Simulate
    t0 = time.time()
    
    # Build classifications dict for sim
    cs = {}
    for lev in ["sector", "industry", "subindustry"]:
        mp = {s: cd.get(lev, "Unk") for s, cd in classifications.items() if isinstance(cd, dict)}
        if mp:
            cs[lev] = pd.Series(mp)
    
    result = simulate_vectorized(
        alpha_df=combined,
        returns_df=matrices["returns"],
        close_df=matrices["close"],
        open_df=matrices["open"],
        universe_df=universe_df,
        classifications=cs,
        booksize=20_000_000.0,
        max_stock_weight=0.01,
        delay=1,
        neutralization="subindustry",
        fees_bps=0.0,
    )
    elapsed = time.time() - t0
    
    # IS/OOS split
    oos_start = "2024-01-01"
    pnl = result.daily_pnl
    is_pnl = pnl[pnl.index < oos_start]
    oos_pnl = pnl[pnl.index >= oos_start]
    
    is_sharpe = (is_pnl.mean() / is_pnl.std() * np.sqrt(252)) if is_pnl.std() > 0 else 0
    oos_sharpe = (oos_pnl.mean() / oos_pnl.std() * np.sqrt(252)) if oos_pnl.std() > 0 else 0
    full_sharpe = result.sharpe
    
    is_cum = is_pnl.sum()
    oos_cum = oos_pnl.sum()
    
    # Max drawdown
    cum = result.cumulative_pnl
    oos_cum_series = cum[cum.index >= oos_start] - cum[cum.index >= oos_start].iloc[0]
    oos_dd = (oos_cum_series - oos_cum_series.cummax()).min() / 20_000_000
    
    full_dd = result.max_drawdown
    
    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Full:  Sharpe={full_sharpe:+.2f}  PnL=${result.total_pnl:>12,.0f}  DD={full_dd:.1%}")
    print(f"    IS:    Sharpe={is_sharpe:+.2f}  PnL=${is_cum:>12,.0f}")
    print(f"    OOS:   Sharpe={oos_sharpe:+.2f}  PnL=${oos_cum:>12,.0f}  DD={oos_dd:.1%}")
    
    return {
        "threshold": threshold,
        "n_alphas": len(signals),
        "full_sharpe": full_sharpe,
        "is_sharpe": float(is_sharpe),
        "oos_sharpe": float(oos_sharpe),
        "is_cum": float(is_cum),
        "oos_cum": float(oos_cum),
        "full_dd": full_dd,
        "oos_dd": float(oos_dd),
        "elapsed": elapsed,
    }


def main():
    t_start = time.time()
    
    print("Loading data...")
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
    coverage = universe_df.mean(axis=0)
    tickers = sorted(coverage[coverage >= 0.3].index.tolist())
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
    
    print(f"  Data loaded in {time.time() - t_start:.0f}s")
    
    # Run at 3 thresholds
    thresholds = [1.0, 0.9, 0.75]
    all_results = []
    
    for threshold in thresholds:
        r = run_avg_backtest(threshold, engine, matrices, classifications, universe_df)
        if r:
            all_results.append(r)
    
    # Comparison
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON: AVERAGED SIGNALS")
    print(f"{'='*70}")
    print(f"{'Threshold':>11s} | {'#Alpha':>6s} | {'Full Sharpe':>11s} | {'IS Sharpe':>10s} | {'OOS Sharpe':>10s} | {'OOS PnL':>12s} | {'OOS DD':>8s}")
    print("-" * 85)
    for r in all_results:
        print(f"  >= {r['threshold']:.2f}  | {r['n_alphas']:6d} "
              f"| {r['full_sharpe']:>+11.2f} "
              f"| {r['is_sharpe']:>+10.2f} "
              f"| {r['oos_sharpe']:>+10.2f} "
              f"| ${r['oos_cum']:>11,.0f} "
              f"| {r['oos_dd']:>7.1%}")
    
    print(f"\n  (Previous curated 6α Isichenko: IS=+0.48, OOS=+0.49, PnL=+$1.5M)")
    print(f"\n  Total elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
