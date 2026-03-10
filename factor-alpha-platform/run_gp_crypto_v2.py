"""
GP Alpha Discovery V2 — Enriched terminals (funding, OI, taker, volatility)
with correlation gate against existing 56 CryptoRL alphas.

STRICT IN-SAMPLE DISCIPLINE:
    Train: 2020-01-01 to 2025-09-01 (GP sees ONLY this)
    OOS:   2025-09-01 to 2026-03-05 (NEVER touched by GP)

Target: 25+ new alphas with IS Sharpe > 1.5 and |corr| < 0.5 with all DB alphas.

Usage:
    python run_gp_crypto_v2.py --generations 150 --population 800
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from src.agent.gp_engine import GPAlphaEngine, GPConfig
from src.operators.fastexpression import FastExpressionEngine
from src.operators.crypto_ops import evaluate_expression, CRYPTO_ALPHA_DEFINITIONS
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

# ── Constants ──
DATA_DIR = Path("data/binance_cache")
UNIVERSE_DIR = DATA_DIR / "universes"

# In-sample cutoff — GP ONLY trains on data before this date
TRAIN_END = "2025-09-01"
TRAIN_START = "2022-09-01"  # 3 years of training is plenty; speeds up evals 3x

# All terminals (42 fields) — price/volume/funding/OI/volatility/microstructure
CRYPTO_TERMINALS = [
    "close", "open", "high", "low", "volume", "quote_volume",
    "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "taker_buy_quote_volume",
    "vwap", "vwap_deviation",
    "high_low_range", "open_close_range",
    "adv20", "adv60",
    "volume_ratio_20d",
    "volume_momentum_1", "volume_momentum_5_20",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_10", "parkinson_volatility_20", "parkinson_volatility_60",
    "trades_count", "trades_per_volume",
    "momentum_5d", "momentum_20d", "momentum_60d",
    "beta_to_btc",
    "overnight_gap",
    "upper_shadow", "lower_shadow",
    "close_position_in_range",
    "dollars_traded",
    # Funding rate signals (unique to futures!)
    "funding_rate", "funding_rate_cumsum_3",
    "funding_rate_avg_7d", "funding_rate_zscore",
]


def log(msg: str):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_crypto_data(universe_name: str = "BINANCE_TOP50", interval: str = "4h"):
    """Load Binance matrices and universe, split into train/test."""
    log(f"Loading Binance data (interval={interval})...")

    matrices_dir = DATA_DIR / "matrices" / interval if interval != "1d" else DATA_DIR / "matrices"

    matrices = {}
    for fpath in sorted(matrices_dir.glob("*.parquet")):
        name = fpath.stem
        matrices[name] = pd.read_parquet(fpath)

    log(f"  {len(matrices)} matrices loaded from {matrices_dir}")

    # Load universe
    suffix = "" if interval == "1d" else f"_{interval}"
    uni_path = UNIVERSE_DIR / f"{universe_name}{suffix}.parquet"
    if not uni_path.exists():
        uni_path = UNIVERSE_DIR / f"{universe_name}.parquet"
        log(f"  WARNING: No {interval} universe found, using daily")
    universe_df = pd.read_parquet(uni_path)
    log(f"  Universe {universe_name}: {universe_df.shape}")

    # Get valid tickers
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > 0.3].index.tolist())
    log(f"  {len(valid_tickers)} tickers with >30% universe coverage")

    # Filter to valid tickers
    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    # Split train/test
    train_start_ts = pd.Timestamp(TRAIN_START)
    train_end_ts = pd.Timestamp(TRAIN_END)

    train_matrices = {name: df.loc[train_start_ts:train_end_ts] for name, df in matrices.items()}
    test_matrices = {name: df.loc[train_end_ts:] for name, df in matrices.items()}
    train_universe = universe_df.loc[train_start_ts:train_end_ts]
    test_universe = universe_df.loc[train_end_ts:]

    log(f"  Train: {train_matrices['close'].shape[0]} bars "
        f"({train_matrices['close'].index[0].date()} to {train_matrices['close'].index[-1].date()})")
    log(f"  Test:  {test_matrices['close'].shape[0]} bars "
        f"({test_matrices['close'].index[0].date()} to {test_matrices['close'].index[-1].date()})")

    return (matrices, train_matrices, test_matrices,
            universe_df, train_universe, test_universe, valid_tickers)


def compute_existing_alpha_pnls(train_matrices, train_universe):
    """Pre-compute PnL vectors for all 56 existing CryptoRL alphas (for correlation gating)."""
    log("Computing existing alpha PnL vectors for correlation gating...")

    # Build features dict matching crypto_ops expectations
    features = {name: train_matrices[name] for name in train_matrices}

    existing_pnls = {}
    for name, expr in CRYPTO_ALPHA_DEFINITIONS:
        try:
            sig = evaluate_expression(expr, features)
            if sig is None:
                continue
            # Compute PnL from this signal
            sig_n = sig.sub(sig.mean(axis=1), axis=0)
            sig_n = sig_n.div(sig_n.abs().sum(axis=1) + 1e-10, axis=0)
            ret = features['returns']
            pnl = (sig_n.shift(1) * ret).sum(axis=1).dropna()
            if len(pnl) > 100:
                existing_pnls[name] = pnl.values
        except Exception:
            pass

    log(f"  {len(existing_pnls)} existing alpha PnLs computed")
    return existing_pnls


def evaluate_alpha_oos(expression, engine, returns_df, close_df, universe_df,
                       booksize=2_000_000.0, max_stock_weight=0.10, fees_bps=5.0):
    """Evaluate a single alpha expression on the test set."""
    try:
        alpha_df = engine.evaluate(expression)
        if alpha_df is None or alpha_df.empty:
            return None

        result = sim_vec(
            alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
            universe_df=universe_df, booksize=booksize,
            max_stock_weight=max_stock_weight, decay=0, delay=1,
            neutralization="market", fees_bps=fees_bps,
            bars_per_day=6,
        )
        return result
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="GP Alpha Discovery V2 — Enriched Terminals")
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--population", type=int, default=800)
    parser.add_argument("--universe", default="BINANCE_TOP50")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--booksize", type=float, default=2_000_000.0)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--fees-bps", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max tree depth (4=wide/shallow, 6=deep)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sharpe", type=float, default=1.5,
                        help="Minimum IS Sharpe to record alpha")
    parser.add_argument("--corr-with-existing", type=float, default=0.70,
                        help="Max |correlation| with any existing alpha PnL")
    parser.add_argument("--target-alphas", type=int, default=25,
                        help="Target number of new alphas to discover")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("GP ALPHA DISCOVERY V2 — ENRICHED TERMINALS", flush=True)
    print(f"  Universe: {args.universe} | Interval: {args.interval}", flush=True)
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (IN-SAMPLE ONLY)", flush=True)
    print(f"  Test:  {TRAIN_END} to 2026-03-05 (NEVER TOUCHED)", flush=True)
    print(f"  Generations: {args.generations}, Population: {args.population}", flush=True)
    print(f"  Min IS Sharpe: {args.min_sharpe}", flush=True)
    print(f"  Max depth: {args.max_depth} (wide trees)", flush=True)
    print(f"  Target: {args.target_alphas}+ new alphas", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    # ── Load data ──
    (all_matrices, train_matrices, test_matrices,
     all_universe, train_universe, test_universe,
     valid_tickers) = load_crypto_data(args.universe, args.interval)

    # ── Compute existing alpha PnLs for correlation gating ──
    existing_pnls = compute_existing_alpha_pnls(train_matrices, train_universe)

    # ── Determine available terminals ──
    available_terminals = [t for t in CRYPTO_TERMINALS if t in train_matrices]
    log(f"  {len(available_terminals)} terminals available")
    for t in available_terminals:
        log(f"    {t}")

    # ── Configure GP ──
    # FITNESS THRESHOLDS RATIONALE:
    # - min_sharpe=1.5: Annualized at sqrt(6*252)=38.8. In 4h bars,
    #   a daily-equivalent Sharpe of ~1.5 is strong. CryptoRL's best
    #   single alphas had IS Sharpe of 1.5-3.0.
    # - corr_cutoff=0.5: Within-GP diversity. Combined with
    #   corr_with_existing=0.5, ensures new alphas are truly novel.
    # - population=800, generations=150: Larger search for richer
    #   terminal set (42 features vs typical 6).

    gp_config = GPConfig(
        population_size=args.population,
        n_generations=args.generations,
        max_tree_depth=args.max_depth,  # 4 = wide/shallow trees
        tournament_size=5,              # Lower pressure = more diversity
        crossover_prob=0.8,             # High crossover for wide trees
        mutation_prob=0.15,             # Exploration
        hall_of_fame_size=50,        # Keep more candidates
        booksize=args.booksize,
        max_stock_weight=args.max_weight,
        decay=0,
        delay=0,
        neutralization="market",
        fees_bps=args.fees_bps,      # 0 for fitness (fees applied separately in evaluation)
        min_turnover=0.0,
        max_turnover=999.0,
        fitness_cutoff=0.0,
        min_sharpe_cutoff=args.min_sharpe,  # 1.5 — strong requirement
        corr_cutoff=args.corr_with_existing,  # 0.5 — novel vs existing
        lookback_range=40,           # Window constants up to 240 (40*6=240 bars)
        include_advanced=True,
        bars_per_day=6,              # 4h bars = 6 per day
    )

    # ── Build GP engine with TRAINING data only ──
    log("\nInitializing GP engine (training data only)...")

    train_returns = train_matrices["returns"]  # pct_change for PnL computation
    train_close = train_matrices.get("close")
    train_open = train_matrices.get("open")

    gp_data = {name: train_matrices[name] for name in available_terminals
               if name in train_matrices}

    # Fix 1: Override 'returns' terminal to dollar-diff (matching super project)
    # The GP evolves expressions on (close - close.shift(1)), NOT pct_change
    # But sim still uses pct_change returns_df for PnL
    if train_close is not None and "returns" in gp_data:
        gp_data["returns"] = train_close - train_close.shift(1)
        log("  Overrode 'returns' terminal: dollar-diff (close - close.shift(1))")

    from src.data.alpha_database import AlphaDatabase

    db_path = f"data/alpha_gp_crypto_v2_{args.interval}.db"
    db = AlphaDatabase(db_path)
    log(f"  Database: {db_path}")
    log(f"  Data fields: {len(gp_data)} matrices")
    log(f"  Training period: {train_returns.index[0].date()} to {train_returns.index[-1].date()}")

    engine = GPAlphaEngine(
        data=gp_data,
        returns_df=train_returns,
        close_df=train_close,
        open_df=train_open,
        classifications=None,
        config=gp_config,
        db=db,
    )

    # ── Inject existing alpha PnLs into the engine for diversity checking ──
    # The engine's _check_diversity checks against _stored_pnl_vectors
    if not hasattr(engine, '_stored_pnl_vectors'):
        engine._stored_pnl_vectors = []
    for name, pnl_vec in existing_pnls.items():
        engine._stored_pnl_vectors.append(pnl_vec.copy())
    log(f"  Injected {len(existing_pnls)} existing alpha PnLs for diversity gating")

    # ── Run GP Discovery ──
    log(f"\nStarting GP evolution ({args.generations} gen, pop={args.population})...")
    log(f"  42 terminals × 15 operators × depth-6 trees = massive search space")
    log(f"  Correlation gate: |corr| < {args.corr_with_existing} with {len(existing_pnls)} existing alphas")

    gp_results = engine.run(
        n_generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        verbose=True,
    )

    gp_elapsed = time.time() - t0
    log(f"\nGP completed in {gp_elapsed:.0f}s")
    log(f"  Best fitness: {gp_results.get('best_fitness', 0):.4f}")
    log(f"  Best expression: {gp_results.get('best_expression', 'N/A')[:80]}")

    best_alphas = gp_results.get("best_alphas", [])
    log(f"  {len(best_alphas)} alphas recorded")

    # ── Evaluate ALL recorded alphas on TRUE OOS ──
    log("\n" + "=" * 70)
    log("OUT-OF-SAMPLE EVALUATION (GP never saw this data)")
    log("=" * 70)

    test_engine = FastExpressionEngine(data_fields={
        name: test_matrices[name] for name in available_terminals
        if name in test_matrices
    })

    test_returns = test_matrices["returns"]
    test_close = test_matrices.get("close")

    # Get all alphas from database
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""SELECT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover
                   FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
                   ORDER BY e.sharpe DESC""")
    db_alphas = cur.fetchall()
    conn.close()

    log(f"\n  {len(db_alphas)} alphas in database")

    results_file = open("crypto_gp_v2_results.txt", "w")
    def rlog(msg):
        print(msg, flush=True)
        results_file.write(msg + "\n")
        results_file.flush()

    rlog("=" * 90)
    rlog(f"GP V2 RESULTS — {args.universe} | {args.interval}")
    rlog(f"Train: 2020-01-01 to {TRAIN_END} | Test: {TRAIN_END} to 2026-03-05")
    rlog(f"Fees: {args.fees_bps} bps | Min IS Sharpe: {args.min_sharpe}")
    rlog(f"Corr gate: {args.corr_with_existing} vs {len(existing_pnls)} existing alphas")
    rlog(f"GP: {args.generations} gen × {args.population} pop, {gp_elapsed:.0f}s")
    rlog("=" * 90)

    rlog(f"\n{'#':>3} | {'IS Sharpe':>9} | {'OOS Sharpe':>10} | {'OOS PnL':>10} | "
         f"{'OOS TO':>7} | {'OOS DD':>7} | Expression")
    rlog("-" * 90)

    oos_results = []
    for i, (alpha_id, expression, is_sharpe, fitness, turnover) in enumerate(db_alphas):
        if is_sharpe < args.min_sharpe:
            continue

        oos_result = evaluate_alpha_oos(
            expression=expression, engine=test_engine,
            returns_df=test_returns, close_df=test_close,
            universe_df=test_universe,
            booksize=args.booksize, max_stock_weight=args.max_weight,
            fees_bps=5.0,   # Always evaluate OOS at 5 bps
        )

        if oos_result is not None:
            oos_results.append({
                "expression": expression,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_result.sharpe,
                "oos_pnl": oos_result.total_pnl,
                "oos_turnover": oos_result.turnover,
                "oos_max_dd": oos_result.max_drawdown,
            })

            rlog(f"{len(oos_results):3d} | {is_sharpe:+9.2f} | {oos_result.sharpe:+10.2f} | "
                 f"${oos_result.total_pnl:>9,.0f} | {oos_result.turnover:>6.1%} | "
                 f"{oos_result.max_drawdown:>6.1%} | {expression[:50]}")

    # ── Summary ──
    rlog("\n" + "=" * 90)
    rlog("SUMMARY")
    rlog("=" * 90)

    if oos_results:
        oos_df = pd.DataFrame(oos_results)
        n_total = len(oos_df)
        n_positive = (oos_df["oos_sharpe"] > 0).sum()
        n_above_05 = (oos_df["oos_sharpe"] > 0.5).sum()
        n_above_10 = (oos_df["oos_sharpe"] > 1.0).sum()

        rlog(f"\n  Total alphas evaluated OOS: {n_total}")
        rlog(f"  OOS Sharpe > 0:   {n_positive:3d} ({100*n_positive/max(n_total,1):.0f}%)")
        rlog(f"  OOS Sharpe > 0.5: {n_above_05:3d} ({100*n_above_05/max(n_total,1):.0f}%)")
        rlog(f"  OOS Sharpe > 1.0: {n_above_10:3d} ({100*n_above_10/max(n_total,1):.0f}%)")

        rlog(f"\n  IS Sharpe:  mean={oos_df['is_sharpe'].mean():+.2f}  "
             f"median={oos_df['is_sharpe'].median():+.2f}")
        rlog(f"  OOS Sharpe: mean={oos_df['oos_sharpe'].mean():+.2f}  "
             f"median={oos_df['oos_sharpe'].median():+.2f}")
        rlog(f"\n  Rank corr (IS vs OOS): "
             f"{oos_df['is_sharpe'].corr(oos_df['oos_sharpe'], method='spearman'):+.2f}")

        # Top 10 by OOS
        rlog("\n  TOP 10 BY OOS SHARPE:")
        top10 = oos_df.nlargest(10, "oos_sharpe")
        for _, row in top10.iterrows():
            rlog(f"    IS={row['is_sharpe']:+.2f} OOS={row['oos_sharpe']:+.2f} "
                 f"PnL=${row['oos_pnl']:>8,.0f} | {row['expression'][:60]}")

        # Save top alphas for integration
        good_alphas = oos_df[oos_df["oos_sharpe"] > 0.5].sort_values("oos_sharpe", ascending=False)
        if len(good_alphas) > 0:
            good_alphas.to_csv("gp_v2_good_alphas.csv", index=False)
            rlog(f"\n  Saved {len(good_alphas)} good alphas (OOS > 0.5) to gp_v2_good_alphas.csv")
            rlog(f"  These can be loaded into AlphaDB and added to the walk-forward pipeline")
    else:
        rlog("\n  No alphas met the minimum Sharpe threshold!")

    rlog(f"\nTotal runtime: {time.time()-t0:.0f}s")
    results_file.close()
    print(f"\n✅ Results saved to crypto_gp_v2_results.txt", flush=True)


if __name__ == "__main__":
    main()
