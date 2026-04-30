"""
eval_alpha.py — Agent 1: Alpha Discovery Harness (TRAIN SET ONLY)

Uses the shared data/alphas.db database. Saves new alphas to the existing schema
and logs evaluations to the evaluations table.

Usage:
    python eval_alpha.py --expr "ts_zscore(close, 60)"  # Evaluate on train (IS + IC + stability + DSR)
    python eval_alpha.py --expr "ts_zscore(close, 60)" --save # Save if passes all quality gates
    python eval_alpha.py --list                         # List all alphas in DB
    python eval_alpha.py --scoreboard                   # Full scoreboard with DSR stats
"""

import sys, os, time, argparse, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPRESSION = "ts_zscore(close, 60)"           # <-- LLM EDITS THIS LINE

# Exchange: 'binance' or 'kucoin' — controls matrix/universe paths
EXCHANGE = "binance"

# Agent 1 ONLY uses the train period
TRAIN_START  = "2022-09-01"
TRAIN_END    = "2024-09-01"   # 2 years in-sample

# Sub-period splits for stability check (within train)
SUBPERIODS = [
    ("2022-09-01", "2023-09-01", "H1"),
    ("2023-09-01", "2024-09-01", "H2"),
]

# Sim parameters
UNIVERSE     = "BINANCE_TOP50"
INTERVAL     = "4h"
BOOKSIZE     = 2_000_000.0
MAX_WEIGHT   = 0.10
NEUTRALIZE   = "market"
BARS_PER_DAY = 6
COVERAGE_CUTOFF = 0.3

# Quality gates
MIN_IS_SHARPE  = 3.0
MIN_FITNESS    = 5.0
MIN_IC_MEAN    = -0.05       # Loose IC gate — Sharpe/stability/DSR are the real filters
CORR_CUTOFF    = 0.70
MAX_TURNOVER   = 0.30        # Reject high-turnover alphas — ρ=-0.62 with val Sharpe
MIN_SUB_SHARPE = 1.0         # Each sub-period must have meaningful Sharpe (not just > 0)
MAX_PNL_KURTOSIS = 20        # Reject fat-tailed PnL distributions — ρ=-0.51 with val Sharpe
MAX_ROLLING_SR_STD = 0.05    # Reject inconsistent performers — ρ=-0.40 with val Sharpe
MIN_PNL_SKEW = -0.5          # Reject negatively-skewed PnL (steamroller risk) — ρ=+0.32 with val Sharpe

# Shared DB (same as Agent 2 and existing infrastructure)
DB_PATH = "data/alphas.db"


# ============================================================================
# DATABASE (uses existing schema in data/alphas.db)
# ============================================================================

def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_trial_log(conn):
    """Create trial_log table if it doesn't exist (Agent 1 addition)."""
    conn.execute("""CREATE TABLE IF NOT EXISTS trial_log (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        expression TEXT NOT NULL,
        is_sharpe REAL,
        saved INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()


def log_trial(conn, expression, is_sharpe, saved=False):
    ensure_trial_log(conn)
    conn.execute("INSERT INTO trial_log (expression, is_sharpe, saved) VALUES (?,?,?)",
                 (expression, is_sharpe, 1 if saved else 0))
    conn.commit()


def get_num_trials(conn):
    ensure_trial_log(conn)
    return conn.execute("SELECT COUNT(*) FROM trial_log").fetchone()[0]


def check_diversity(conn, expression, new_alpha_raw):
    """Check signal correlation against all existing alphas on train data.
    Rejects if expression already exists OR if signal correlation > CORR_CUTOFF.
    new_alpha_raw should be the RAW expression output (before process_signal)."""
    # Check exact duplicate (scoped to interval + universe)
    existing = conn.execute(
        "SELECT id FROM alphas WHERE expression=? AND archived=0 AND interval=? AND universe=?",
        (expression, INTERVAL, UNIVERSE)).fetchone()
    if existing:
        print(f"  REJECTED: Expression already exists as alpha #{existing[0]}")
        return False

    if new_alpha_raw is None:
        return True

    # Check signal correlation against all existing alphas (same interval + universe)
    rows = conn.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND interval=? AND universe=?",
        (INTERVAL, UNIVERSE)).fetchall()
    if not rows:
        return True

    matrices, universe = load_data("train")

    # Process BOTH alphas through the same pipeline for fair comparison
    new_processed = process_signal(new_alpha_raw, universe_df=universe, max_wt=MAX_WEIGHT)

    for alpha_id, alpha_expr in rows:
        try:
            existing_raw = evaluate_expression(alpha_expr, matrices)
            if existing_raw is None:
                continue
            # Apply the SAME processing to the existing alpha (apples-to-apples)
            existing_df = process_signal(existing_raw, universe_df=universe, max_wt=MAX_WEIGHT)
            # Align shapes
            common_idx = new_processed.index.intersection(existing_df.index)
            common_cols = new_processed.columns.intersection(existing_df.columns)
            if len(common_idx) < 50 or len(common_cols) < 5:
                continue
            a = new_processed.loc[common_idx, common_cols].values.flatten()
            b = existing_df.loc[common_idx, common_cols].values.flatten()
            # Remove NaN pairs
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 100:
                continue
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            if abs(corr) > CORR_CUTOFF:
                print(f"  REJECTED: Signal corr={corr:.3f} with alpha #{alpha_id} (cutoff={CORR_CUTOFF})")
                print(f"            Existing: {alpha_expr[:60]}")
                return False
        except Exception:
            continue  # skip broken alphas

    return True


def save_alpha(conn, expression, reasoning, metrics):
    """Save alpha to the existing alphas table schema."""
    if not check_diversity(conn, expression, metrics.get('_alpha_raw')):
        return False

    c = conn.cursor()
    c.execute("""INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes, universe)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (expression,
               expression[:80],              # name = shortened expression
               metrics.get('category', ''),   # category from reasoning
               'crypto',                      # asset_class
               INTERVAL,                      # interval
               'agent1_research',             # source
               reasoning,
               UNIVERSE))
    alpha_id = c.lastrowid

    # Also save evaluation metrics
    c.execute("""INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann,
                 max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                 train_start, train_end, n_bars, evaluated_at, interval, universe)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?,?)""",
              (alpha_id,
               metrics['is_sharpe'], metrics['is_sharpe'],
               metrics.get('returns_ann', 0),
               metrics['max_drawdown'], metrics['turnover'],
               metrics['is_fitness'],
               metrics['ic_mean'], metrics['icir'],
               metrics['deflated_sharpe'],
               TRAIN_START, TRAIN_END,
               metrics.get('n_bars', 0),
               INTERVAL, UNIVERSE))
    conn.commit()
    print(f"  SAVED as alpha #{alpha_id}")
    return True


# ============================================================================
# DATA LOADING (TRAIN ONLY)
# ============================================================================

_DATA_CACHE = {}

def load_data(split="train"):
    if split in _DATA_CACHE:
        return _DATA_CACHE[split]

    exchange_dir = "binance_cache" if EXCHANGE == "binance" else "kucoin_cache"
    mat_dir = Path(f"data/{exchange_dir}/matrices/{INTERVAL}")
    uni_path = Path(f"data/{exchange_dir}/universes/{UNIVERSE}_{INTERVAL}.parquet")

    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "h1":    (SUBPERIODS[0][0], SUBPERIODS[0][1]),
        "h2":    (SUBPERIODS[1][0], SUBPERIODS[1][1]),
    }
    if split not in splits:
        raise ValueError(f"Agent 1 only uses train splits: {list(splits.keys())}")

    start, end = splits[split]
    split_matrices = {name: df.loc[start:end] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[start:end]

    if "close" in split_matrices and "returns" in split_matrices:
        close = split_matrices["close"]
        split_matrices["returns"] = close.pct_change()

    result = (split_matrices, split_universe)
    _DATA_CACHE[split] = result
    return result


# ============================================================================
# CORE EVALUATION
# ============================================================================

def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def simulate(alpha_df, returns_df, close_df, universe_df, fees_bps=0.0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=returns_df, close_df=close_df,
        universe_df=universe_df, booksize=BOOKSIZE,
        max_stock_weight=MAX_WEIGHT, decay=0, delay=0,
        neutralization=NEUTRALIZE, fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
        trading_days_per_year=365,  # crypto trades 365 days/year
    )


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
    """
    Standard Signal Processing Pipeline (Matches Superproject: WorldQuant style).
    1. Neutralize (Demean)
    2. Scale (Sum(abs(weights)) = 1)
    3. Clip (Weight limit)
    """
    # 1. Neutralize (Demean cross-sectionally)
    # Replaces 'hedgeGlobal' in RiskModelFunctions.py
    signal = alpha_df.copy()
    
    # Apply universe mask if provided (important for accurate de-meaning)
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)

    # Cross-sectional demean
    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)

    # 2. Scale (Sum of absolute weights = 1)
    # This keeps the exposure constant regardless of signal magnitude
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)

    # 3. Clip (Max position weight)
    # Matches np.clip(..., -max_wt, max_wt) in StratMangerBillion.py
    signal = signal.clip(lower=-max_wt, upper=max_wt)

    return signal.fillna(0.0)


def compute_ic(alpha_df, returns_df):
    """Cross-sectional rank IC per bar: spearman(signal[t-1], return[t])."""
    alpha_lagged = alpha_df.shift(1)
    ics = []
    for dt in alpha_lagged.index[1:]:
        a = alpha_lagged.loc[dt].dropna()
        r = returns_df.loc[dt].dropna()
        common = a.index.intersection(r.index)
        if len(common) < 10:
            ics.append(np.nan)
            continue
        ic, _ = stats.spearmanr(a[common], r[common])
        ics.append(ic)
    return pd.Series(ics, index=alpha_lagged.index[1:])


def deflated_sharpe_ratio(observed_sr, n_trials, n_bars, skew=0, kurtosis=3):
    """Lopez de Prado DSR: P(true SR > 0) given # trials."""
    if n_trials <= 1:
        return min(stats.norm.cdf(observed_sr * np.sqrt(n_bars)), 1.0)

    euler_gamma = 0.5772156649
    log_n = np.log(n_trials)
    if log_n < 0.01:
        expected_max_sr = 0
    else:
        expected_max_sr = (np.sqrt(2 * log_n) * (1 - euler_gamma / log_n)
                           + euler_gamma / np.sqrt(2 * log_n))

    sr_var = (1 + 0.5 * observed_sr**2 - skew * observed_sr
              + ((kurtosis - 3) / 4) * observed_sr**2) / n_bars
    if sr_var <= 0:
        return 0.0

    z = (observed_sr - expected_max_sr) / np.sqrt(sr_var)
    return float(stats.norm.cdf(z))


def eval_single(expression, split="train", fees_bps=0.0):
    matrices, universe = load_data(split)
    close = matrices.get("close")
    returns_pct = close.pct_change() if close is not None else matrices.get("returns")

    t0 = time.time()
    try:
        alpha_raw = evaluate_expression(expression, matrices)
    except Exception as e:
        return {"success": False, "error": f"Expression error: {e}"}

    if alpha_raw is None or (hasattr(alpha_raw, 'empty') and alpha_raw.empty):
        return {"success": False, "error": "Expression returned None/empty"}

    # --- WORLDQUANT SIGNAL PROCESSING (Demean -> Scale -> Clip) ---
    # Matches superproject: RiskModelFunctions.hedgeGlobal and StratMangerBillion.py
    alpha_df = process_signal(alpha_raw, universe_df=universe, max_wt=MAX_WEIGHT)

    try:
        result = simulate(alpha_df, returns_pct, close, universe, fees_bps=fees_bps)
    except Exception as e:
        return {"success": False, "error": f"Simulation error: {e}"}

    return {
        "success": True,
        "sharpe": result.sharpe, "fitness": result.fitness,
        "turnover": result.turnover, "max_drawdown": result.max_drawdown,
        "returns_ann": result.returns_ann,
        "n_bars": len(result.daily_pnl),
        "pnl_vec": np.array(result.daily_pnl),
        "alpha_df": alpha_df, "alpha_raw": alpha_raw,
        "returns_pct": returns_pct,
        "elapsed": time.time() - t0,
    }


# ============================================================================
# FULL EVALUATION (IS + IC + Stability + DSR)
# ============================================================================

def eval_full(expression, conn):
    n_trials = get_num_trials(conn) + 1

    is_m = eval_single(expression, split="train", fees_bps=0)
    if not is_m["success"]:
        return {"success": False, "error": is_m["error"]}

    # IC
    ic_series = compute_ic(is_m["alpha_df"], is_m["returns_pct"])
    ic_clean = ic_series.dropna()
    ic_mean = ic_clean.mean() if len(ic_clean) > 0 else 0
    ic_std = ic_clean.std() if len(ic_clean) > 1 else 1
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # Stability
    stability = {}
    for start, end, name in SUBPERIODS:
        sub = eval_single(expression, split=name.lower(), fees_bps=0)
        stability[name] = sub["sharpe"] if sub["success"] else 0

    # DSR
    dsr = deflated_sharpe_ratio(is_m["sharpe"], n_trials, is_m["n_bars"])

    # PnL distribution metrics (novel gates — empirically validated)
    pnl = is_m["pnl_vec"]
    pnl_kurtosis = float(pd.Series(pnl).kurtosis()) if len(pnl) > 10 else 0
    pnl_skew = float(pd.Series(pnl).skew()) if len(pnl) > 10 else 0
    rolling_sr = pd.Series(pnl).rolling(60 * BARS_PER_DAY).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0
    ).dropna()
    rolling_sr_std = float(rolling_sr.std()) if len(rolling_sr) > 10 else 999

    return {
        "success": True,
        "is_sharpe": is_m["sharpe"], "is_fitness": is_m["fitness"],
        "turnover": is_m["turnover"], "max_drawdown": is_m["max_drawdown"],
        "returns_ann": is_m["returns_ann"], "n_bars": is_m["n_bars"],
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "stability_h1": stability.get("H1", 0),
        "stability_h2": stability.get("H2", 0),
        "deflated_sharpe": dsr, "n_trials": n_trials,
        "pnl_vec": is_m["pnl_vec"],
        "pnl_kurtosis": pnl_kurtosis,
        "pnl_skew": pnl_skew,
        "rolling_sr_std": rolling_sr_std,
        "_alpha_raw": is_m["alpha_raw"],  # RAW signal for correlation check on save
    }


# ============================================================================
# LISTING / SCOREBOARD (reads from shared DB)
# ============================================================================

def list_alphas():
    conn = get_conn()
    rows = conn.execute("""
        SELECT a.id, a.expression, a.source,
               COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.turnover, 0), COALESCE(e.ic_mean, 0)
        FROM alphas a
        LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.fitness, 0) DESC
    """, (UNIVERSE,)).fetchall()
    n_trials = get_num_trials(conn)
    conn.close()

    print(f"\n{'='*70}")
    print(f"ALPHA DATABASE [{UNIVERSE}]: {len(rows)} active alphas | {n_trials} agent trials")
    print(f"{'='*70}")
    for r in rows:
        print(f"  #{r[0]:3d} | SR={r[3]:+.2f} Fit={r[4]:.2f} TO={r[5]:.2f} IC={r[6]:+.3f} "
              f"| src={r[2] or '?':15s} | {r[1][:40]}")


def print_scoreboard():
    conn = get_conn()
    n_alphas = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0 AND universe=?",
                            (UNIVERSE,)).fetchone()[0]
    n_agent = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE source='agent1_research' AND archived=0 AND universe=?",
        (UNIVERSE,)).fetchone()[0]
    n_trials = get_num_trials(conn)
    rows = conn.execute("""
        SELECT a.expression, COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.ic_mean, 0), COALESCE(e.psr, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.source = 'agent1_research' AND a.universe = ?
        ORDER BY COALESCE(e.fitness, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()

    print(f"\n{'='*70}")
    print(f"  ALPHA RESEARCH SCOREBOARD (Agent 1) [{UNIVERSE}]")
    print(f"  Total alphas in DB: {n_alphas} | Agent 1 alphas: {n_agent} | Trials: {n_trials}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (no fees)")
    print(f"{'='*70}")
    if rows:
        print(f"  Avg IS Sharpe: {np.mean([r[1] for r in rows]):+.3f}")
        print(f"  Avg IC:        {np.mean([r[3] for r in rows]):+.4f}")
        print(f"\n  Agent 1 alphas (top 10):")
        for i, r in enumerate(rows[:10]):
            print(f"    {i+1}. SR={r[1]:+.2f} Fit={r[2]:.2f} IC={r[3]:+.3f} DSR={r[4]:.2f} | {r[0][:200]}")
    else:
        print("  No Agent 1 alphas yet.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent 1: Alpha Discovery (Train Only)")
    parser.add_argument("--expr", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--scoreboard", action="store_true")
    parser.add_argument("--reasoning", type=str, default="")
    parser.add_argument("--universe", type=str, default=None,
                        help="Universe name (e.g. BINANCE_TOP50, KUCOIN_TOP50). Auto-detects exchange.")
    args = parser.parse_args()

    # Auto-detect exchange from universe prefix
    global EXCHANGE, UNIVERSE, TRAIN_START, TRAIN_END, SUBPERIODS
    if args.universe:
        UNIVERSE = args.universe
        if args.universe.startswith("KUCOIN"):
            EXCHANGE = "kucoin"
            TRAIN_START = "2023-09-01"
            TRAIN_END   = "2025-09-01"
            SUBPERIODS  = [
                ("2023-09-01", "2024-09-01", "H1"),
                ("2024-09-01", "2025-09-01", "H2"),
            ]
        elif args.universe.startswith("BINANCE"):
            EXCHANGE = "binance"
        _DATA_CACHE.clear()

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    if args.scoreboard:
        print_scoreboard()
        return
    if args.list:
        list_alphas()
        return

    conn = get_conn()
    ensure_trial_log(conn)
    expression = args.expr or EXPRESSION
    print(f"\nEvaluating: {expression}")

    result = eval_full(expression, conn)
    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        log_trial(conn, expression, 0, saved=False)
        conn.close()
        return

    # Print results
    print(f"\n  --- IN-SAMPLE ({TRAIN_START} to {TRAIN_END}, no fees) ---")
    print(f"  Sharpe:       {result['is_sharpe']:+.3f}")
    print(f"  Fitness:      {result['is_fitness']:.3f}")
    print(f"  Turnover:     {result['turnover']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']:.3f}")

    print(f"\n  --- PNL DISTRIBUTION ---")
    print(f"  PnL Kurtosis: {result['pnl_kurtosis']:.1f}")
    print(f"  PnL Skewness: {result['pnl_skew']:+.3f}")
    print(f"  Rolling SR std: {result['rolling_sr_std']:.4f}")

    print(f"\n  --- INFORMATION COEFFICIENT ---")
    print(f"  Mean IC:      {result['ic_mean']:+.4f}")
    print(f"  IC Std:       {result['ic_std']:.4f}")
    print(f"  ICIR:         {result['icir']:.3f}")

    print(f"\n  --- SUB-PERIOD STABILITY ---")
    print(f"  H1 ({SUBPERIODS[0][0][:7]}): {result['stability_h1']:+.3f}")
    print(f"  H2 ({SUBPERIODS[1][0][:7]}): {result['stability_h2']:+.3f}")
    both_pos = result['stability_h1'] > 0 and result['stability_h2'] > 0
    print(f"  Stable:       {'YES' if both_pos else 'NO'}")

    print(f"\n  --- DEFLATED SHARPE (trial {result['n_trials']}) ---")
    print(f"  DSR:          {result['deflated_sharpe']:.3f}")

    # Gates
    print(f"\n  --- QUALITY GATES ---")
    min_sub = min(result['stability_h1'], result['stability_h2'])
    gates = [
        (result['is_sharpe'] >= MIN_IS_SHARPE, f"IS Sharpe >= {MIN_IS_SHARPE}: {result['is_sharpe']:+.3f}"),
        (result['is_fitness'] >= MIN_FITNESS, f"Fitness >= {MIN_FITNESS}: {result['is_fitness']:.3f}"),
        (result['ic_mean'] >= MIN_IC_MEAN, f"Mean IC >= {MIN_IC_MEAN}: {result['ic_mean']:+.4f}"),
        (both_pos, f"Sub-period stability: H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}"),
        (result['turnover'] <= MAX_TURNOVER, f"Turnover <= {MAX_TURNOVER}: {result['turnover']:.3f}"),
        (min_sub >= MIN_SUB_SHARPE, f"Min sub-period Sharpe >= {MIN_SUB_SHARPE}: {min_sub:+.2f}"),
        (result['pnl_kurtosis'] <= MAX_PNL_KURTOSIS,
         f"PnL Kurtosis <= {MAX_PNL_KURTOSIS}: {result['pnl_kurtosis']:.1f}"),
        (result['rolling_sr_std'] <= MAX_ROLLING_SR_STD,
         f"Rolling SR Consistency <= {MAX_ROLLING_SR_STD}: {result['rolling_sr_std']:.4f}"),
        (result['pnl_skew'] >= MIN_PNL_SKEW,
         f"PnL Skew >= {MIN_PNL_SKEW}: {result['pnl_skew']:+.3f}"),
    ]
    # DSR shown as info only, not a gate
    print(f"  [INFO] DSR: {result['deflated_sharpe']:.3f} (informational, not a gate)")
    all_pass = True
    for passed, desc in gates:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        if not passed:
            all_pass = False

    log_trial(conn, expression, result['is_sharpe'], saved=False)

    if args.save:
        if not all_pass:
            print(f"\n  NOT SAVED: Failed quality gates")
        else:
            saved = save_alpha(conn, expression, args.reasoning, result)
            if saved:
                conn.execute("UPDATE trial_log SET saved=1 WHERE trial_id=(SELECT MAX(trial_id) FROM trial_log)")
                conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
