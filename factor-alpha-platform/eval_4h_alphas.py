"""
Evaluate 4h GP alphas — IS/OOS with rank-average and QP combination.
"""
import sqlite3
import sys
import time
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

# ── Config ──
DATA_DIR = Path("data/binance_cache")
MATRICES_DIR = DATA_DIR / "matrices" / "4h"
UNIVERSE_DIR = DATA_DIR / "universes"
DB_PATH = "data/alpha_gp_crypto_4h.db"
TRAIN_END = "2024-04-27"
BOOKSIZE = 2_000_000.0
MAX_WEIGHT = 0.05

def log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    # ── Load data ──
    log("Loading 4h data...")
    matrices = {}
    for fpath in sorted(MATRICES_DIR.glob("*.parquet")):
        matrices[fpath.stem] = pd.read_parquet(fpath)
    log(f"  {len(matrices)} matrices loaded")

    universe = pd.read_parquet(UNIVERSE_DIR / "BINANCE_TOP50_4h.parquet")
    
    # Valid tickers
    coverage = universe.sum(axis=0) / len(universe)
    valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    # Split
    train_end_ts = pd.Timestamp(TRAIN_END)
    train_matrices = {n: df.loc[:train_end_ts] for n, df in matrices.items()}
    test_matrices = {n: df.loc[train_end_ts:] for n, df in matrices.items()}
    train_universe = universe.loc[:train_end_ts]
    test_universe = universe.loc[train_end_ts:]

    train_returns = train_matrices["returns"]
    test_returns = test_matrices["returns"]
    train_close = train_matrices.get("close")
    test_close = test_matrices.get("close")
    all_returns = matrices["returns"]
    all_close = matrices.get("close")
    all_universe = universe

    log(f"  Train: {train_returns.shape[0]} bars")
    log(f"  Test:  {test_returns.shape[0]} bars")

    # ── Load alphas from DB ──
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover
                   FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
                   ORDER BY e.sharpe DESC""")
    db_alphas = cur.fetchall()
    conn.close()
    
    # Deduplicate by expression
    seen = set()
    unique_alphas = []
    for row in db_alphas:
        if row[1] not in seen:
            seen.add(row[1])
            unique_alphas.append(row)
    
    log(f"\n  {len(unique_alphas)} unique alphas from DB")

    # ── Evaluate each alpha IS and OOS ──
    train_engine = FastExpressionEngine(data_fields={
        n: train_matrices[n] for n in matrices if n in train_matrices
    })
    test_engine = FastExpressionEngine(data_fields={
        n: test_matrices[n] for n in matrices if n in test_matrices
    })
    full_engine = FastExpressionEngine(data_fields=matrices)

    results = []
    print(f"\n{'#':>3} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'IS TO':>6} | {'OOS TO':>6} | {'OOS PnL':>10} | Expression")
    print("-" * 110)

    # Store ranked signals for combination
    train_ranked_signals = []
    test_ranked_signals = []
    full_ranked_signals = []
    
    for i, (aid, expr, is_sharpe, fitness, turnover) in enumerate(unique_alphas):
        try:
            # IS evaluation
            is_alpha = train_engine.evaluate(expr)
            if is_alpha is None or is_alpha.empty:
                continue
            is_result = sim_vec(
                alpha_df=is_alpha, returns_df=train_returns, close_df=train_close,
                universe_df=train_universe, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                decay=0, delay=0, neutralization="market", fees_bps=0.0,
            )

            # OOS evaluation
            oos_alpha = test_engine.evaluate(expr)
            if oos_alpha is None or oos_alpha.empty:
                continue
            oos_result = sim_vec(
                alpha_df=oos_alpha, returns_df=test_returns, close_df=test_close,
                universe_df=test_universe, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                decay=0, delay=0, neutralization="market", fees_bps=0.0,
            )

            results.append({
                "expression": expr,
                "is_sharpe": is_result.sharpe,
                "oos_sharpe": oos_result.sharpe,
                "is_turnover": is_result.turnover,
                "oos_turnover": oos_result.turnover,
                "oos_pnl": oos_result.total_pnl,
            })

            print(f"{i+1:3d} | {is_result.sharpe:+9.3f} | {oos_result.sharpe:+10.3f} | "
                  f"{is_result.turnover:5.0%} | {oos_result.turnover:5.0%} | "
                  f"${oos_result.total_pnl:>9,.0f} | {expr[:50]}")

            # Store ranked signals for combination
            for engine_obj, store, label in [
                (train_engine, train_ranked_signals, "train"),
                (test_engine, test_ranked_signals, "test"),
                (full_engine, full_ranked_signals, "full"),
            ]:
                try:
                    a = engine_obj.evaluate(expr)
                    if a is not None and not a.empty:
                        ranked = a.rank(axis=1, pct=True) - 0.5
                        store.append(ranked)
                except:
                    pass

        except Exception as e:
            print(f"  Error evaluating {expr[:40]}: {e}")

    if not results:
        print("\nNo alphas evaluated successfully!")
        return

    # ── Summary ──
    res_df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"INDIVIDUAL ALPHA SUMMARY ({len(res_df)} alphas)")
    print(f"{'='*80}")
    print(f"  IS Sharpe:  mean={res_df['is_sharpe'].mean():+.3f}  max={res_df['is_sharpe'].max():+.3f}")
    print(f"  OOS Sharpe: mean={res_df['oos_sharpe'].mean():+.3f}  max={res_df['oos_sharpe'].max():+.3f}")
    n_pos = (res_df['oos_sharpe'] > 0).sum()
    print(f"  OOS Sharpe > 0: {n_pos}/{len(res_df)} ({100*n_pos/len(res_df):.0f}%)")
    corr = res_df['is_sharpe'].corr(res_df['oos_sharpe'], method='spearman')
    print(f"  IS vs OOS rank correlation: {corr:+.3f}")

    # ── COMBINATION 1: Equal-weight rank average ──
    print(f"\n{'='*80}")
    print(f"COMBINATION: EQUAL-WEIGHT RANK AVERAGE ({len(train_ranked_signals)} signals)")
    print(f"{'='*80}")
    
    for period_name, signals, ret, cl, uni in [
        ("TRAIN", train_ranked_signals, train_returns, train_close, train_universe),
        ("TEST", test_ranked_signals, test_returns, test_close, test_universe),
        ("FULL", full_ranked_signals, all_returns, all_close, all_universe),
    ]:
        if not signals:
            continue
        combined = signals[0].copy()
        for s in signals[1:]:
            combined = combined.add(s, fill_value=0)
        combined = combined / len(signals)

        for fees in [0.0, 3.0, 5.0, 10.0]:
            r = sim_vec(
                alpha_df=combined, returns_df=ret, close_df=cl, universe_df=uni,
                booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
                neutralization="market", fees_bps=fees,
            )
            print(f"  {period_name:5s} [{fees:4.0f}bps]: Sharpe={r.sharpe:+.3f} "
                  f"PnL=${r.total_pnl:>10,.0f} TO={r.turnover:.0%} DD={r.max_drawdown:.1%}")

    # ── COMBINATION 2: Simple QP (mean-variance) ──
    print(f"\n{'='*80}")
    print(f"COMBINATION: QP MEAN-VARIANCE OPTIMIZATION")
    print(f"{'='*80}")
    
    if len(train_ranked_signals) >= 2:
        try:
            # Build daily PnL matrix for QP
            train_pnl_series = []
            for s in train_ranked_signals:
                r = sim_vec(
                    alpha_df=s, returns_df=train_returns, close_df=train_close,
                    universe_df=train_universe, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                    decay=0, delay=0, neutralization="market", fees_bps=0.0,
                )
                train_pnl_series.append(r.daily_pnl)
            
            pnl_matrix = pd.concat(train_pnl_series, axis=1).fillna(0)
            pnl_matrix.columns = range(len(train_pnl_series))
            
            # Mean-variance weights
            mu = pnl_matrix.mean().values
            cov = pnl_matrix.cov().values
            
            # Simple inverse-variance weighting (regularized)
            n = len(mu)
            reg_cov = cov + np.eye(n) * np.trace(cov) * 0.1
            try:
                inv_cov = np.linalg.inv(reg_cov)
                raw_weights = inv_cov @ mu
                weights = raw_weights / np.sum(np.abs(raw_weights))
                weights = np.clip(weights, -0.5, 0.5)
                weights = weights / np.sum(np.abs(weights))
            except:
                weights = np.ones(n) / n
            
            print(f"  QP weights: {[f'{w:.3f}' for w in weights]}")
            
            # Build weighted combination
            for period_name, signals, ret, cl, uni in [
                ("TRAIN", train_ranked_signals, train_returns, train_close, train_universe),
                ("TEST", test_ranked_signals, test_returns, test_close, test_universe),
                ("FULL", full_ranked_signals, all_returns, all_close, all_universe),
            ]:
                if not signals:
                    continue
                combined_qp = None
                for j, s in enumerate(signals):
                    if combined_qp is None:
                        combined_qp = s * weights[j]
                    else:
                        combined_qp = combined_qp.add(s * weights[j], fill_value=0)
                
                for fees in [0.0, 3.0, 5.0, 10.0]:
                    r = sim_vec(
                        alpha_df=combined_qp, returns_df=ret, close_df=cl, universe_df=uni,
                        booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
                        neutralization="market", fees_bps=fees,
                    )
                    print(f"  {period_name:5s} [{fees:4.0f}bps]: Sharpe={r.sharpe:+.3f} "
                          f"PnL=${r.total_pnl:>10,.0f} TO={r.turnover:.0%} DD={r.max_drawdown:.1%}")
        except Exception as e:
            print(f"  QP failed: {e}")
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()
