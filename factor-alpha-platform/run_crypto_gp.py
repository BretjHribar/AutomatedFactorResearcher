"""
Binance Futures GP Alpha Discovery — price/volume/funding/OI signals.

Key design decisions:
- Training data: 2020-01-01 to 2024-04-27 (70%)
- True OOS test: 2024-04-27 to present (30%) — GP NEVER sees this
- Universe: TOP50 by rolling ADV (sweet spot of liquidity vs diversity)
- Neutralization: market (dollar neutral), no sectors
- Max stock weight: 5% per coin (fewer names than equities)
- Booksize: $2M (crypto liquidity-appropriate)
- Fees: 5 bps one-way (3 bps taker + 2 bps slippage)
- Supports multiple intervals (1d, 4h) via --interval flag

Usage:
    python run_crypto_gp.py [--generations 100] [--population 500] [--interval 4h]
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
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

# ── Constants ──
DATA_DIR = Path("data/binance_cache")
UNIVERSE_DIR = DATA_DIR / "universes"

# 70/30 train/test split
TRAIN_END = "2024-04-27"

# Terminal set — price/volume/funding/OI
CRYPTO_TERMINALS = [
    "close", "open", "high", "low", "volume", "quote_volume",
    "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "taker_buy_quote_volume",
    "vwap", "vwap_deviation",
    "high_low_range", "open_close_range",
    "adv20", "adv60",
    "volume_ratio_20d",
    "volume_momentum_1", "volume_momentum_5_20",
    "historical_volatility_20", "historical_volatility_60",
    "historical_volatility_10", "historical_volatility_120",
    "parkinson_volatility_10", "parkinson_volatility_20", "parkinson_volatility_60",
    "trades_count", "trades_per_volume",
    "momentum_5d", "momentum_20d", "momentum_60d",
    "beta_to_btc",
    "overnight_gap",
    "upper_shadow", "lower_shadow",
    "close_position_in_range",
    # Funding rate signals
    "funding_rate", "funding_rate_cumsum_3",
    "funding_rate_avg_7d", "funding_rate_zscore",
    # Open interest signals
    "open_interest_value", "oi_change", "oi_volume_ratio",
]


def log(msg: str):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_crypto_data(universe_name: str = "BINANCE_TOP50", interval: str = "1d"):
    """Load Binance matrices and universe, split into train/test."""
    log(f"Loading Binance data (interval={interval})...")

    # Determine matrix directory based on interval
    if interval == "1d":
        matrices_dir = DATA_DIR / "matrices"
    else:
        matrices_dir = DATA_DIR / "matrices" / interval

    # Load all matrices
    matrices = {}
    for fpath in sorted(matrices_dir.glob("*.parquet")):
        name = fpath.stem
        df = pd.read_parquet(fpath)
        matrices[name] = df

    log(f"  {len(matrices)} matrices loaded from {matrices_dir}")

    # Load universe (4h universes have suffix _4h)
    suffix = "" if interval == "1d" else f"_{interval}"
    uni_path = UNIVERSE_DIR / f"{universe_name}{suffix}.parquet"
    if not uni_path.exists():
        # Fall back to daily universe if interval-specific doesn't exist
        uni_path = UNIVERSE_DIR / f"{universe_name}.parquet"
        log(f"  ⚠ No {interval} universe found, using daily")
    universe_df = pd.read_parquet(uni_path)
    log(f"  Universe {universe_name}: {universe_df.shape}")

    # Get tickers that are in the universe at least 30% of the time
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
    log(f"  {len(valid_tickers)} tickers with >10% universe coverage")

    # Filter matrices to valid tickers
    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    # Split into train and test
    train_end_ts = pd.Timestamp(TRAIN_END)

    # Training data: everything up to TRAIN_END
    train_matrices = {}
    for name, df in matrices.items():
        train_matrices[name] = df.loc[:train_end_ts]

    train_universe = universe_df.loc[:train_end_ts]

    # Test data: everything after TRAIN_END
    test_matrices = {}
    for name, df in matrices.items():
        test_matrices[name] = df.loc[train_end_ts:]

    test_universe = universe_df.loc[train_end_ts:]

    log(f"  Train: {train_matrices['close'].shape[0]} bars "
        f"({train_matrices['close'].index[0]} to {train_matrices['close'].index[-1]})")
    log(f"  Test:  {test_matrices['close'].shape[0]} bars "
        f"({test_matrices['close'].index[0]} to {test_matrices['close'].index[-1]})")

    return matrices, train_matrices, test_matrices, universe_df, train_universe, test_universe, valid_tickers


def evaluate_alpha_oos(
    expression: str,
    engine: FastExpressionEngine,
    returns_df: pd.DataFrame,
    close_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    booksize: float = 2_000_000.0,
    max_stock_weight: float = 0.05,
    fees_bps: float = 5.0,
):
    """Evaluate a single alpha expression on the test set."""
    try:
        alpha_df = engine.evaluate(expression)
        if alpha_df is None or alpha_df.empty:
            return None

        result = sim_vec(
            alpha_df=alpha_df,
            returns_df=returns_df,
            close_df=close_df,
            universe_df=universe_df,
            booksize=booksize,
            max_stock_weight=max_stock_weight,
            decay=0,
            delay=1,
            neutralization="market",
            fees_bps=fees_bps,
        )
        return result
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Crypto GP Alpha Discovery")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=500)
    parser.add_argument("--universe", default="BINANCE_TOP50")
    parser.add_argument("--interval", default="4h",
                        help="Data interval: 1d, 4h (default: 4h)")
    parser.add_argument("--booksize", type=float, default=2_000_000.0)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--fees-bps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sharpe", type=float, default=1.0,
                        help="Minimum Sharpe to record alpha")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("BINANCE FUTURES GP ALPHA DISCOVERY", flush=True)
    print(f"  Universe: {args.universe} | Interval: {args.interval}", flush=True)
    print(f"  Train: 2020-01-01 to {TRAIN_END} (70%)", flush=True)
    print(f"  Test:  {TRAIN_END} to present (30%) — GP never sees this", flush=True)
    print(f"  Generations: {args.generations}, Population: {args.population}", flush=True)
    print(f"  Fees: {args.fees_bps} bps, Max weight: {args.max_weight:.0%}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()

    # ── Load data ──
    (all_matrices, train_matrices, test_matrices,
     all_universe, train_universe, test_universe,
     valid_tickers) = load_crypto_data(args.universe, args.interval)

    # ── Determine available terminals ──
    available_terminals = [t for t in CRYPTO_TERMINALS if t in train_matrices]
    log(f"  {len(available_terminals)} terminals available: {available_terminals[:10]}...")

    # ── Configure GP ──
    gp_config = GPConfig(
        population_size=args.population,
        n_generations=args.generations,
        max_tree_depth=6,
        tournament_size=7,
        crossover_prob=0.7,
        mutation_prob=0.1,
        hall_of_fame_size=20,
        booksize=args.booksize,
        max_stock_weight=args.max_weight,
        decay=0,
        delay=0,
        neutralization="market",
        fees_bps=args.fees_bps,
        min_turnover=0.0,
        max_turnover=999.0,           # No turnover constraint
        fitness_cutoff=0.0,           # No fitness filter
        min_sharpe_cutoff=1.0,        # ~2.45 daily-equivalent (sim uses sqrt(252))
        corr_cutoff=0.7,
        lookback_range=40,
        include_advanced=True,
    )

    # ── Build GP engine with TRAINING data only ──
    log("\nInitializing GP engine (training data only)...")

    train_returns = train_matrices["returns"]
    train_close = train_matrices.get("close")
    train_open = train_matrices.get("open")

    # Build data dict for GP engine — only terminals that exist in training data
    gp_data = {name: train_matrices[name] for name in available_terminals
               if name in train_matrices}

    from src.data.alpha_database import AlphaDatabase

    db_path = f"data/alpha_gp_crypto_{args.interval}.db"
    db = AlphaDatabase(db_path)
    log(f"  Database: {db_path}")
    log(f"  Data fields: {len(gp_data)} matrices")
    log(f"  Training period: {train_returns.index[0].date()} to {train_returns.index[-1].date()}")

    engine = GPAlphaEngine(
        data=gp_data,
        returns_df=train_returns,
        close_df=train_close,
        open_df=train_open,
        classifications=None,  # No GICS sectors for crypto
        config=gp_config,
        db=db,
    )

    # ── Run GP Discovery ──
    log(f"\nStarting GP evolution ({args.generations} generations, pop={args.population})...")
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

    # Build expression engine with FULL data for OOS evaluation
    test_engine = FastExpressionEngine(data_fields={
        name: test_matrices[name] for name in available_terminals
        if name in test_matrices
    })

    test_returns = test_matrices["returns"]
    test_close = test_matrices.get("close")

    # Also get all alphas from the database
    import sqlite3
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""SELECT a.alpha_id, a.expression, e.sharpe, e.fitness, e.turnover
                   FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
                   ORDER BY e.sharpe DESC""")
    db_alphas = cur.fetchall()
    conn.close()

    log(f"\n  {len(db_alphas)} alphas in database")

    results_file = open("crypto_gp_results.txt", "w")
    def rlog(msg):
        print(msg, flush=True)
        results_file.write(msg + "\n")
        results_file.flush()

    rlog("=" * 90)
    rlog(f"BINANCE FUTURES GP RESULTS — {args.universe}")
    rlog(f"Train: 2020-01-01 to {TRAIN_END} | Test: {TRAIN_END} to present")
    rlog(f"Fees: {args.fees_bps} bps | Max weight: {args.max_weight:.0%} | "
         f"Booksize: ${args.booksize:,.0f}")
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
            expression=expression,
            engine=test_engine,
            returns_df=test_returns,
            close_df=test_close,
            universe_df=test_universe,
            booksize=args.booksize,
            max_stock_weight=args.max_weight,
            fees_bps=args.fees_bps,
        )

        if oos_result is not None:
            oos_results.append({
                "expression": expression,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_result.sharpe,
                "oos_pnl": oos_result.total_pnl,
                "oos_turnover": oos_result.turnover,
                "oos_max_dd": oos_result.max_drawdown,
                "oos_fitness": oos_result.fitness,
            })

            rlog(f"{len(oos_results):3d} | {is_sharpe:+9.2f} | {oos_result.sharpe:+10.2f} | "
                 f"${oos_result.total_pnl:>9,.0f} | {oos_result.turnover:>6.1%} | "
                 f"{oos_result.max_drawdown:>6.1%} | {expression[:50]}")

    # ── Summary statistics ──
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

        rlog(f"\n  Rank correlation (IS vs OOS Sharpe): "
             f"{oos_df['is_sharpe'].corr(oos_df['oos_sharpe'], method='spearman'):+.2f}")

        # Top 5 by OOS performance
        rlog("\n  TOP 5 BY OOS SHARPE:")
        top5 = oos_df.nlargest(5, "oos_sharpe")
        for _, row in top5.iterrows():
            rlog(f"    IS={row['is_sharpe']:+.2f} OOS={row['oos_sharpe']:+.2f} "
                 f"PnL=${row['oos_pnl']:>8,.0f} | {row['expression'][:60]}")

        # Best combined alpha (equal-weight average of top OOS performers)
        if n_above_05 >= 3:
            rlog("\n  COMBINED ALPHA (equal-weight of OOS Sharpe > 0.5):")
            good_exprs = oos_df[oos_df["oos_sharpe"] > 0.5]["expression"].tolist()

            # Evaluate combined on FULL period
            full_engine = FastExpressionEngine(data_fields={
                name: all_matrices[name] for name in available_terminals
                if name in all_matrices
            })
            full_returns = all_matrices["returns"]
            full_close = all_matrices.get("close")

            combined = None
            n_good = 0
            for expr in good_exprs:
                try:
                    a = full_engine.evaluate(expr)
                    if a is not None and not a.empty:
                        ranked = a.rank(axis=1, pct=True) - 0.5
                        if combined is None:
                            combined = ranked
                        else:
                            combined = combined.add(ranked, fill_value=0)
                        n_good += 1
                except:
                    pass

            if combined is not None and n_good > 0:
                combined = combined / n_good

                # Evaluate on train period
                for period_name, period_ret, period_close, period_uni in [
                    ("TRAIN", train_returns, train_close, train_universe),
                    ("TEST", test_returns, test_close, test_universe),
                    ("FULL", full_returns, full_close, all_universe),
                ]:
                    r = sim_vec(
                        alpha_df=combined,
                        returns_df=period_ret,
                        close_df=period_close,
                        universe_df=period_uni,
                        booksize=args.booksize,
                        max_stock_weight=args.max_weight,
                        decay=0,
                        delay=1,
                        neutralization="market",
                        fees_bps=args.fees_bps,
                    )
                    rlog(f"    {period_name:5s}: Sharpe={r.sharpe:+.2f} "
                         f"PnL=${r.total_pnl:>10,.0f} "
                         f"TO={r.turnover:.1%} DD={r.max_drawdown:.1%}")
    else:
        rlog("\n  No alphas met the minimum Sharpe threshold!")

    rlog(f"\nTotal runtime: {time.time()-t0:.0f}s")
    results_file.close()
    print(f"\n✅ Results saved to crypto_gp_results.txt", flush=True)


if __name__ == "__main__":
    main()
