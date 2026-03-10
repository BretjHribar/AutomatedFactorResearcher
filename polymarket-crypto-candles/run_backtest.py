"""
run_backtest.py — Main backtest runner with iterative signal optimization.

Usage:
    python run_backtest.py                    # Run full pipeline
    python run_backtest.py --scan             # Scan all signals
    python run_backtest.py --optimize         # Optimize signal combination
    python run_backtest.py --holdout          # Evaluate on holdout set
    python run_backtest.py --all              # Full pipeline: scan + optimize + holdout
"""
import sys, os, time, argparse, sqlite3, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SYMBOLS, INTERVALS, DATA_DIR, DB_PATH,
                     TRAIN_START, TRAIN_END, HOLDOUT_START, HOLDOUT_END,
                     SYMBOL_NAMES, BLENDED_TAKER_FEE, INITIAL_CAPITAL, BASE_TRADE_SIZE)
from signals import compute_signals, compute_target, get_all_signals
from backtest_engine import run_backtest, run_combined_backtest, evaluate_signal, BacktestResult


BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}


# ============================================================================
# DATABASE
# ============================================================================

def ensure_db():
    """Create the signals database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS signal_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        split TEXT NOT NULL,
        win_rate REAL,
        sharpe REAL,
        net_pnl REAL,
        max_dd REAL,
        avg_edge REAL,
        profit_factor REAL,
        total_trades INTEGER,
        trades_per_day REAL,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS optimized_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        signal_names TEXT NOT NULL,
        weights TEXT NOT NULL,
        train_sharpe REAL,
        train_win_rate REAL,
        holdout_sharpe REAL,
        holdout_win_rate REAL,
        holdout_pnl REAL,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    return conn


def save_signal_result(conn, signal_name, symbol, interval, split, metrics):
    conn.execute("""INSERT INTO signal_results
        (signal_name, symbol, interval, split, win_rate, sharpe, net_pnl,
         max_dd, avg_edge, profit_factor, total_trades, trades_per_day)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (signal_name, symbol, interval, split,
         metrics["win_rate"], metrics["sharpe"], metrics["net_pnl"],
         metrics["max_dd"], metrics["avg_edge"], metrics["profit_factor"],
         metrics["trades"], metrics["trades_per_day"]))
    conn.commit()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_klines(symbol: str, interval: str) -> pd.DataFrame:
    """Load kline data for a symbol/interval."""
    path = DATA_DIR / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    return pd.read_parquet(path)


def split_data(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Split data by date range."""
    return df.loc[start:end].copy()


# ============================================================================
# SIGNAL SCANNING
# ============================================================================

def scan_all_signals(symbol: str, interval: str, conn, verbose: bool = True):
    """Scan all signals for a symbol/interval on the train set."""
    print(f"\n{'='*70}")
    print(f"SCANNING: {SYMBOL_NAMES[symbol]} {interval}")
    print(f"{'='*70}")

    df = load_klines(symbol, interval)
    train_df = split_data(df, TRAIN_START, TRAIN_END)

    if len(train_df) < 100:
        print(f"  WARNING: Only {len(train_df)} bars in train set, skipping")
        return []

    # Compute all signals and target
    signals_df = compute_signals(train_df)
    target = compute_target(train_df)
    bpd = BARS_PER_DAY.get(interval, 288)

    results = []
    for col in signals_df.columns:
        sig = signals_df[col]
        valid = sig.dropna()
        if len(valid) < 50:
            continue

        metrics = evaluate_signal(sig, target, bars_per_day=bpd, label=col)
        results.append(metrics)
        save_signal_result(conn, col, symbol, interval, "train", metrics)

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    if verbose:
        print(f"\n  Top 20 signals by Sharpe (train {TRAIN_START} to {TRAIN_END}):")
        print(f"  {'Signal':<35} {'WinRate':>8} {'Sharpe':>8} {'PnL':>10} {'PF':>6} {'Trades/d':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")
        for r in results[:20]:
            print(f"  {r['name']:<35} {r['win_rate']:>7.1%} {r['sharpe']:>8.2f} "
                  f"${r['net_pnl']:>9,.0f} {r['profit_factor']:>5.2f} {r['trades_per_day']:>7.1f}")

    return results


# ============================================================================
# SIGNAL SELECTION & OPTIMIZATION
# ============================================================================

def select_top_signals(conn, symbol: str, interval: str, min_sharpe: float = 0.3,
                       min_win_rate: float = 0.51, max_signals: int = 20) -> list:
    """Select top signals from DB based on train performance."""
    rows = conn.execute("""
        SELECT signal_name, win_rate, sharpe, profit_factor
        FROM signal_results
        WHERE symbol=? AND interval=? AND split='train'
          AND sharpe > ? AND win_rate > ?
        ORDER BY sharpe DESC
        LIMIT ?
    """, (symbol, interval, min_sharpe, min_win_rate, max_signals * 3)).fetchall()

    if not rows:
        return []

    # Greedy correlation-aware selection
    df = load_klines(symbol, interval)
    train_df = split_data(df, TRAIN_START, TRAIN_END)
    signals_df = compute_signals(train_df, signal_names=[r[0] for r in rows])

    selected = []
    selected_signals = []

    for name, wr, sr, pf in rows:
        if name not in signals_df.columns:
            continue
        sig = signals_df[name].dropna()
        if len(sig) < 50:
            continue

        # Check correlation with already selected
        too_correlated = False
        for sel_name in selected:
            sel_sig = signals_df[sel_name].dropna()
            common = sig.index.intersection(sel_sig.index)
            if len(common) < 50:
                continue
            corr = np.corrcoef(sig.loc[common].values, sel_sig.loc[common].values)[0, 1]
            if abs(corr) > 0.65:
                too_correlated = True
                break

        if not too_correlated:
            selected.append(name)
            if len(selected) >= max_signals:
                break

    print(f"\n  Selected {len(selected)} orthogonal signals for {SYMBOL_NAMES[symbol]} {interval}")
    for s in selected:
        row = [r for r in rows if r[0] == s][0]
        print(f"    {s:<35} WR={row[1]:.1%} SR={row[2]:.2f}")

    return selected


def optimize_weights(symbol: str, interval: str, signal_names: list,
                     method: str = "equal") -> tuple:
    """
    Optimize signal combination weights on train data.
    Returns (weights, train_result).
    """
    df = load_klines(symbol, interval)
    train_df = split_data(df, TRAIN_START, TRAIN_END)
    signals_df = compute_signals(train_df, signal_names=signal_names)
    target = compute_target(train_df)
    bpd = BARS_PER_DAY.get(interval, 288)

    n_signals = len(signal_names)
    valid_cols = [c for c in signal_names if c in signals_df.columns]
    signals_matrix = signals_df[valid_cols].copy()

    # Normalize each signal to zero mean, unit variance
    for col in valid_cols:
        s = signals_matrix[col]
        mu = s.mean()
        std = s.std()
        if std > 0:
            signals_matrix[col] = (s - mu) / std

    if method == "equal":
        weights = np.ones(len(valid_cols)) / len(valid_cols)

    elif method == "sharpe_weighted":
        # Weight by individual signal Sharpe
        sharpes = []
        for col in valid_cols:
            r = run_backtest(signals_matrix[col], target, bars_per_day=bpd)
            sharpes.append(max(r.sharpe, 0.01))
        weights = np.array(sharpes)
        weights = weights / weights.sum()

    elif method == "optimize":
        # Scipy optimization to maximize Sharpe
        def neg_sharpe(w):
            w = w / max(np.abs(w).sum(), 1e-10)
            combined = (signals_matrix[valid_cols].values * w[np.newaxis, :]).sum(axis=1)
            sig_series = pd.Series(combined, index=signals_matrix.index)
            result = run_backtest(sig_series, target, bars_per_day=bpd)
            return -result.sharpe

        x0 = np.ones(len(valid_cols)) / len(valid_cols)
        bounds = [(0, 1) for _ in valid_cols]  # Long-only weights
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        try:
            opt_result = minimize(neg_sharpe, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints,
                                  options={"maxiter": 200, "ftol": 1e-8})
            weights = opt_result.x
            weights = weights / max(weights.sum(), 1e-10)
        except Exception as e:
            print(f"  Optimization failed: {e}, using equal weights")
            weights = np.ones(len(valid_cols)) / len(valid_cols)

    # Evaluate final combination
    combined = (signals_matrix[valid_cols].values * weights[np.newaxis, :]).sum(axis=1)
    combined_series = pd.Series(combined, index=signals_matrix.index)
    train_result = run_backtest(combined_series, target, bars_per_day=bpd)

    return weights, train_result, valid_cols


def evaluate_holdout(symbol: str, interval: str, signal_names: list,
                     weights: np.ndarray, train_stats: dict = None) -> BacktestResult:
    """Evaluate the model on holdout data."""
    df = load_klines(symbol, interval)

    # Compute normalization stats from TRAIN data (avoid look-ahead)
    train_df = split_data(df, TRAIN_START, TRAIN_END)
    train_signals = compute_signals(train_df, signal_names=signal_names)
    norm_stats = {}
    for col in signal_names:
        if col in train_signals.columns:
            norm_stats[col] = {
                "mean": train_signals[col].mean(),
                "std": train_signals[col].std()
            }

    # Now evaluate on holdout
    holdout_df = split_data(df, HOLDOUT_START, HOLDOUT_END)
    holdout_signals = compute_signals(holdout_df, signal_names=signal_names)
    target = compute_target(holdout_df)
    bpd = BARS_PER_DAY.get(interval, 288)

    # Normalize using TRAIN statistics
    valid_cols = [c for c in signal_names if c in holdout_signals.columns and c in norm_stats]
    for col in valid_cols:
        s = holdout_signals[col]
        mu = norm_stats[col]["mean"]
        std = norm_stats[col]["std"]
        if std > 0:
            holdout_signals[col] = (s - mu) / std

    w = weights[:len(valid_cols)]
    combined = (holdout_signals[valid_cols].values * w[np.newaxis, :]).sum(axis=1)
    combined_series = pd.Series(combined, index=holdout_signals.index)

    return run_backtest(combined_series, target, bars_per_day=bpd)


# ============================================================================
# FULL PIPELINE
# ============================================================================

def run_full_pipeline():
    """Run the complete pipeline: scan → select → optimize → holdout."""
    conn = ensure_db()

    all_results = {}

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            key = f"{SYMBOL_NAMES[symbol]}_{interval}"
            print(f"\n{'#'*70}")
            print(f"# PIPELINE: {key}")
            print(f"{'#'*70}")

            try:
                # Step 1: Scan all signals on train
                scan_results = scan_all_signals(symbol, interval, conn)
                if not scan_results:
                    print(f"  No valid signals found, skipping")
                    continue

                # Step 2: Select orthogonal top signals
                selected = select_top_signals(conn, symbol, interval,
                                               min_sharpe=0.2, min_win_rate=0.505,
                                               max_signals=15)
                if len(selected) < 3:
                    print(f"  Only {len(selected)} signals selected, trying with looser gates")
                    selected = select_top_signals(conn, symbol, interval,
                                                   min_sharpe=0.05, min_win_rate=0.50,
                                                   max_signals=15)
                if len(selected) < 2:
                    print(f"  Still too few signals, skipping")
                    continue

                # Step 3: Optimize weights (try multiple methods)
                best_method = None
                best_result = None
                best_weights = None
                best_cols = None

                for method in ["equal", "sharpe_weighted", "optimize"]:
                    weights, result, valid_cols = optimize_weights(
                        symbol, interval, selected, method=method
                    )
                    print(f"\n  {method:20s}: Train Sharpe={result.sharpe:.2f} "
                          f"WR={result.win_rate:.1%} PnL=${result.net_pnl:,.0f} "
                          f"MDD=${result.max_drawdown:,.0f} PF={result.profit_factor:.2f}")

                    if best_result is None or result.sharpe > best_result.sharpe:
                        best_method = method
                        best_result = result
                        best_weights = weights
                        best_cols = valid_cols

                print(f"\n  BEST METHOD: {best_method} (Train Sharpe={best_result.sharpe:.2f})")

                # Step 4: Evaluate on holdout
                holdout_result = evaluate_holdout(symbol, interval, best_cols, best_weights)

                print(f"\n  {'='*60}")
                print(f"  HOLDOUT RESULTS ({HOLDOUT_START} to {HOLDOUT_END}):")
                print(f"  {'='*60}")
                print(f"  Sharpe:       {holdout_result.sharpe:.2f}")
                print(f"  Win Rate:     {holdout_result.win_rate:.1%}")
                print(f"  Net PnL:      ${holdout_result.net_pnl:,.0f}")
                print(f"  Max Drawdown: ${holdout_result.max_drawdown:,.0f}")
                print(f"  Profit Factor:{holdout_result.profit_factor:.2f}")
                print(f"  Trades/Day:   {holdout_result.trades_per_day:.1f}")

                # Save to DB
                conn.execute("""INSERT INTO optimized_models
                    (model_name, symbol, interval, signal_names, weights,
                     train_sharpe, train_win_rate, holdout_sharpe, holdout_win_rate, holdout_pnl)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (f"{best_method}_{key}", symbol, interval,
                     json.dumps(best_cols), json.dumps(best_weights.tolist()),
                     best_result.sharpe, best_result.win_rate,
                     holdout_result.sharpe, holdout_result.win_rate,
                     holdout_result.net_pnl))
                conn.commit()

                all_results[key] = {
                    "method": best_method,
                    "signals": best_cols,
                    "weights": best_weights,
                    "train": best_result,
                    "holdout": holdout_result,
                }

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY — ALL MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Train SR':>10} {'Hold SR':>10} {'Hold WR':>10} {'Hold PnL':>12} {'Hold MDD':>12} {'PF':>8}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")

    for key, r in all_results.items():
        print(f"{key:<20} {r['train'].sharpe:>10.2f} {r['holdout'].sharpe:>10.2f} "
              f"{r['holdout'].win_rate:>9.1%} ${r['holdout'].net_pnl:>11,.0f} "
              f"${r['holdout'].max_drawdown:>11,.0f} {r['holdout'].profit_factor:>7.2f}")

    conn.close()
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Polymarket Candle Backtest Runner")
    parser.add_argument("--scan", action="store_true", help="Scan all signals")
    parser.add_argument("--optimize", action="store_true", help="Optimize signal combination")
    parser.add_argument("--holdout", action="store_true", help="Evaluate on holdout")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    parser.add_argument("--symbol", type=str, default=None, help="Specific symbol")
    parser.add_argument("--interval", type=str, default=None, help="Specific interval")
    args = parser.parse_args()

    if args.all or (not args.scan and not args.optimize and not args.holdout):
        run_full_pipeline()
    elif args.scan:
        conn = ensure_db()
        symbols = [args.symbol] if args.symbol else SYMBOLS
        intervals = [args.interval] if args.interval else INTERVALS
        for s in symbols:
            for i in intervals:
                scan_all_signals(s, i, conn)
        conn.close()


if __name__ == "__main__":
    main()
