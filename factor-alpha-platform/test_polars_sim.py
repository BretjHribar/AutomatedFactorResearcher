"""
Compare pandas vs polars vectorized simulation for correctness and speed.
"""
import sys, os, time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import json

from src.simulation.vectorized_sim import simulate_vectorized
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
from src.operators.fastexpression import FastExpressionEngine

def load_data():
    """Load matrices and setup."""
    mdir = "data/fmp_cache/matrices_clean"
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
    
    coverage = universe_df.mean(axis=0)
    tickers = sorted(coverage[coverage >= 0.3].index.tolist())
    
    matrices = {}
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(mdir, fn))
        vc = [c for c in tickers if c in df.columns]
        if vc:
            matrices[fn.replace(".parquet", "")] = df[vc]
    
    with open("data/fmp_cache/classifications.json") as f:
        all_cls = json.load(f)
    
    classifications = {}
    for level in ["sector", "industry", "subindustry"]:
        mp = {s: cd.get(level, "Unk") for s, cd in all_cls.items() if isinstance(cd, dict) and s in tickers}
        if mp:
            classifications[level] = pd.Series(mp)
    
    return matrices, classifications, universe_df, tickers


def compare_results(r1, r2, label: str):
    """Compare two VectorizedSimResult objects."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # Scalar metrics
    metrics = ['sharpe', 'fitness', 'turnover', 'returns_ann', 'max_drawdown', 'margin_bps', 'psr', 'total_pnl']
    all_close = True
    for m in metrics:
        v1 = getattr(r1, m)
        v2 = getattr(r2, m)
        diff = abs(v1 - v2)
        rel_diff = diff / max(abs(v1), 1e-10) * 100
        ok = "OK" if diff < 1e-6 or rel_diff < 0.01 else "DIFF"
        if ok == "DIFF":
            all_close = False
        print(f"  {m:15s}: pandas={v1:>15.6f}  polars={v2:>15.6f}  diff={diff:.2e} ({rel_diff:.4f}%) [{ok}]")
    
    # Time series
    pnl_diff = (r1.daily_pnl - r2.daily_pnl).abs()
    max_pnl_diff = pnl_diff.max()
    mean_pnl_diff = pnl_diff.mean()
    print(f"\n  daily_pnl:      max_diff={max_pnl_diff:.6f}  mean_diff={mean_pnl_diff:.6f}")
    
    to_diff = (r1.daily_turnover - r2.daily_turnover).abs()
    print(f"  daily_turnover: max_diff={to_diff.max():.6f}  mean_diff={to_diff.mean():.6f}")
    
    cum_diff = (r1.cumulative_pnl - r2.cumulative_pnl).abs()
    print(f"  cumulative_pnl: max_diff={cum_diff.max():.6f}  mean_diff={cum_diff.mean():.6f}")
    
    # Match check
    if all_close and max_pnl_diff < 1.0:
        print(f"\n  >>> MATCH <<<")
    else:
        print(f"\n  >>> MISMATCH <<<")
    
    return all_close and max_pnl_diff < 1.0


def main():
    print("Loading data...")
    matrices, classifications, universe_df, tickers = load_data()
    
    returns = matrices['returns']
    close = matrices['close']
    open_prices = matrices['open']
    
    # Setup expression engine
    engine = FastExpressionEngine(data_fields=matrices)
    for gn, gs in classifications.items():
        engine.add_group(gn, gs)
    
    # Test expressions
    test_alphas = [
        "rank(change_in_working_capital)",
        "Inverse(ArgMax(total_debt, 20))",
        "ts_min(volume, 35)",
        "sqrt(ts_rank(working_capital, 18))",
        "ts_entropy(stock_based_compensation, 38)",
        "log10(npfadd(debt_to_equity, -0.6120160888134647))",
    ]
    
    all_match = True
    pandas_times = []
    polars_times = []
    
    for expr in test_alphas:
        print(f"\n{'#'*60}")
        print(f"  Testing: {expr[:55]}")
        print(f"{'#'*60}")
        
        alpha = engine.evaluate(expr)
        if alpha is None or alpha.empty:
            print(f"  SKIP: empty signal")
            continue
        
        # Common args
        kwargs = dict(
            alpha_df=alpha,
            returns_df=returns,
            close_df=close,
            open_df=open_prices,
            universe_df=universe_df,
            classifications=classifications,
            booksize=20_000_000.0,
            max_stock_weight=0.01,
            decay=0,
            delay=1,
            neutralization="subindustry",
            fees_bps=0.0,
        )
        
        # Pandas version
        t0 = time.perf_counter()
        r_pd = simulate_vectorized(**kwargs)
        t_pd = time.perf_counter() - t0
        pandas_times.append(t_pd)
        print(f"\n  Pandas:  {t_pd:.3f}s  Sharpe={r_pd.sharpe:+.4f}  PnL=${r_pd.total_pnl:,.0f}")
        
        # Polars version
        t0 = time.perf_counter()
        r_pl = simulate_vectorized_polars(**kwargs)
        t_pl = time.perf_counter() - t0
        polars_times.append(t_pl)
        print(f"  Polars:  {t_pl:.3f}s  Sharpe={r_pl.sharpe:+.4f}  PnL=${r_pl.total_pnl:,.0f}")
        print(f"  Speedup: {t_pd/t_pl:.1f}x")
        
        match = compare_results(r_pd, r_pl, expr[:55])
        if not match:
            all_match = False
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  Alphas tested:  {len(pandas_times)}")
    print(f"  All match:      {'YES' if all_match else 'NO'}")
    print(f"  Avg Pandas:     {np.mean(pandas_times):.3f}s")
    print(f"  Avg Polars:     {np.mean(polars_times):.3f}s")
    print(f"  Avg Speedup:    {np.mean(pandas_times)/np.mean(polars_times):.1f}x")
    print(f"  Total Pandas:   {sum(pandas_times):.3f}s")
    print(f"  Total Polars:   {sum(polars_times):.3f}s")


if __name__ == "__main__":
    main()
