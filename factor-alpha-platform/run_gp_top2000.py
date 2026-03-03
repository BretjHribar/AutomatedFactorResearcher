"""
GP Alpha Runner — TOP1000 universe with clean data.

Runs GP alpha discovery on the TOP1000 universe (removes penny stocks).
Uses cleaned matrices. Stores results in the main alpha DB.
Target: 25+ alphas with Sharpe >= 1.25 at delay=1.
"""
import sys, os, json, logging, sqlite3
import numpy as np
import pandas as pd

from src.data.context_research import InMemoryDataContext
from src.data.alpha_database import AlphaDatabase
from src.agent.gp_engine import GPAlphaEngine, GPConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

def main():
    # ── Keep existing DB (append new alphas) ──
    db_path = "data/alpha_gp_pipeline.db"
    import sqlite3 as _sq
    _c = _sq.connect(db_path)
    _existing = _c.execute('SELECT COUNT(*) FROM evaluations WHERE sharpe >= 0.5').fetchone()[0]
    _c.close()
    print(f"Keeping existing DB: {_existing} alphas with Sharpe >= 0.5")
    
    # ── Load TOP2000 universe ──
    print("\nLoading TOP1000 universe...")
    universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")
    
    # Find tickers in TOP2000 on ≥30% of days
    ticker_coverage = universe_df.sum(axis=0) / len(universe_df)
    top1000_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())
    print(f"  TOP1000 tickers (>30% coverage): {len(top1000_tickers)}")
    
    # ── Load CLEAN matrices ──
    print("Loading clean matrices...")
    mdir = "data/fmp_cache/matrices_clean"
    if not os.path.isdir(mdir):
        print(f"  WARNING: {mdir} not found, falling back to raw matrices")
        mdir = "data/fmp_cache/matrices"
    
    with open("data/fmp_cache/classifications.json") as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in top1000_tickers}
    
    matrices = {}
    for fname in sorted(os.listdir(mdir)):
        if not fname.endswith(".parquet") or fname.startswith("_"):
            continue
        field = fname.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(mdir, fname))
        valid_cols = [c for c in top1000_tickers if c in df.columns]
        if valid_cols:
            matrices[field] = df[valid_cols]
    
    print(f"  Loaded {len(matrices)} fields, {len(top1000_tickers)} tickers")
    print(f"  Matrix shape: {matrices['close'].shape[0]} days × {matrices['close'].shape[1]} tickers")
    
    # ── Apply universe mask ──
    print("Applying universe mask...")
    for field, mat in matrices.items():
        if isinstance(mat, pd.DataFrame):
            cc = mat.columns.intersection(universe_df.columns)
            ci = mat.index.intersection(universe_df.index)
            if len(cc) > 0 and len(ci) > 0:
                matrices[field] = mat.loc[ci, cc].where(universe_df.loc[ci, cc])
    
    # ── Build context ──
    ctx = InMemoryDataContext()
    ctx.load_from_matrices(matrices, classifications)
    print(f"  Context: {len(ctx._price_matrices)} fields, {len(ctx._classifications)} classifications")
    
    # ── GP Config ──
    config = GPConfig(
        population_size=300,
        n_generations=40,            # shorter rounds, more restarts
        max_tree_depth=8,            # deeper trees for more complex alphas
        booksize=20_000_000.0,
        neutralization="subindustry",
        delay=1,
        fees_bps=0.0,
        corr_cutoff=0.7,             # relaxed for more diversity
        lookback_range=40,
        include_advanced=True,
    )
    
    db = AlphaDatabase(db_path)
    engine = GPAlphaEngine.from_context(ctx, config=config, db=db)
    
    # Print setup
    if engine.classifications is not None:
        for level in ["sector", "industry", "subindustry"]:
            grp = engine.classifications.get(level)
            if grp is not None:
                print(f"  GICS {level}: {grp.nunique()} unique groups")
    
    print(f"  Features ({len(engine.feature_names)}): {engine.feature_names[:10]}...")
    print(f"  Book: ${config.booksize:,.0f}, Delay: {config.delay}, Neutralization: {config.neutralization}")
    print(f"  Population: {config.population_size}, Generations: {config.n_generations}")
    
    # ── Run GP (multiple rounds to get 25+ alphas) ──
    target_count = 100
    target_sharpe = 0.5
    round_num = 0
    max_rounds = 10
    
    while round_num < max_rounds:
        round_num += 1
        # Different random seed each round for diversity
        np.random.seed(42 + round_num * 1000)
        print(f"\n{'='*60}")
        print(f"  GP ROUND {round_num}/{max_rounds} (seed={42 + round_num * 1000})")
        print(f"{'='*60}")
        
        # Re-create engine each round (fresh population)
        engine = GPAlphaEngine.from_context(ctx, config=config, db=db)
        
        results = engine.run(
            n_generations=config.n_generations,
            population_size=config.population_size,
            verbose=True,
        )
        
        print(f"\n  Best: {results['best_expression'][:60]}")
        print(f"  Fitness: {results['best_fitness']:.4f}")
        
        # Check DB count
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM evaluations WHERE sharpe >= {target_sharpe}")
        qualifying = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM evaluations WHERE sharpe >= 1.0")
        good = cur.fetchone()[0]
        conn.close()
        
        print(f"\n  📊 DB Status: {qualifying} alphas ≥ {target_sharpe} Sharpe, {good} ≥ 1.0 Sharpe")
        
        if qualifying >= target_count:
            print(f"\n  ✅ TARGET REACHED: {qualifying} alphas with Sharpe ≥ {target_sharpe}")
            break
        
        print(f"  Need {target_count - qualifying} more alphas ≥ {target_sharpe} Sharpe")
    
    # ── Final summary ──
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT a.expression, e.sharpe, e.fitness, e.turnover
        FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
        WHERE e.sharpe >= 1.0
        ORDER BY e.sharpe DESC
    """)
    rows = cur.fetchall()
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"  FINAL: {len(rows)} alphas with Sharpe ≥ 1.0")
    print(f"{'='*60}")
    for i, (expr, sharpe, fitness, turnover) in enumerate(rows[:20], 1):
        print(f"  {i:2d}. Sharpe={sharpe:+.2f} T/O={turnover:.3f} | {expr[:65]}")


if __name__ == "__main__":
    main()
