"""
eval_alpha_equity.py — Agent 1: Equities Alpha Discovery Harness (TRAIN SET ONLY)

Uses the shared data/alpha_results.db database (equities-specific).
Saves new alphas to the database and logs evaluations to the trial_log table.

Usage:
    python eval_alpha_equity.py --expr "rank(earnings_yield)"          # Evaluate IS + IC + stability + DSR
    python eval_alpha_equity.py --expr "rank(earnings_yield)" --save   # Save if passes all quality gates
    python eval_alpha_equity.py --list                                  # List all alphas in DB
    python eval_alpha_equity.py --scoreboard                            # Full scoreboard with DSR stats

Asset Class: US Equities (daily bars, TOP1000 by ADV20)
Data source: data/fmp_cache/matrices/
This script is EXCLUSIVELY for equities. Do NOT use eval_alpha.py (crypto).
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

# Agent 1 ONLY uses the train period (in-sample)
TRAIN_START = "2016-01-01"
TRAIN_END   = "2023-01-01"   # ~7 years in-sample

# Sub-period splits for stability check (within train)
SUBPERIODS = [
    ("2016-01-01", "2019-07-01", "H1"),
    ("2019-07-01", "2023-01-01", "H2"),
]

# Sim parameters
UNIVERSE      = "TOP1000"
BOOKSIZE      = 10_000_000.0
MAX_WEIGHT    = 0.005          # 0.5% max per stock (1000 stock universe)
NEUTRALIZE    = "subindustry"  # GICS 8-digit — best practice for equities
BARS_PER_DAY  = 1              # Daily bars
DELAY         = 1              # Mandatory for equities (avoid lookahead)
DECAY         = 0              # Linear decay applied to signal (0 = no decay; e.g. 5 = 5-day linear weighted avg)
COVERAGE_CUTOFF = 0.5          # Ticker must appear in universe > 50% of days

# Quality gates (equities — different from crypto)
MIN_IS_SHARPE  = 1.25          # Delay=1 equity signals - higher bar than delay=0
MIN_FITNESS    = 1.0           # Fitness = Sharpe * sqrt(|return_ann| / max(turnover, 0.125))
MIN_IC_MEAN    = 0.0           # Mean IC must be positive (strict for equities)
CORR_CUTOFF    = 0.65          # Reject if |corr| > 0.65 with existing alpha
MIN_SUB_SHARPE = 0.5           # Each sub-period must be positive (bull + bear)
MAX_TURNOVER   = 0.40          # Reject high-turnover alphas (same as crypto)
MAX_PNL_KURTOSIS  = 20        # Reject fat-tailed PnL distributions
MAX_ROLLING_SR_STD = 0.25     # Reject inconsistent performers (calibrated for daily equity bars; crypto gate was 0.05 for 4h)
MIN_PNL_SKEW   = -0.5         # Reject negatively-skewed PnL (steamroller risk)

# Equities DB (separate from crypto alphas.db)
DB_PATH = "data/alpha_results.db"

# Data directory
MATRICES_DIR  = Path("data/fmp_cache/matrices")
UNIVERSES_DIR = Path("data/fmp_cache/universes")


# ============================================================================
# DATABASE
# ============================================================================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn):
    """Create tables if they don't exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS alphas (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        expression  TEXT    NOT NULL,
        name        TEXT,
        category    TEXT,
        asset_class TEXT    DEFAULT 'equities',
        interval    TEXT    DEFAULT 'daily',
        source      TEXT,
        notes       TEXT,
        archived    INTEGER DEFAULT 0,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        alpha_id    INTEGER NOT NULL REFERENCES alphas(id),
        sharpe_is   REAL,
        sharpe_train REAL,
        return_ann  REAL,
        max_drawdown REAL,
        turnover    REAL,
        fitness     REAL,
        ic_mean     REAL,
        ic_ir       REAL,
        psr         REAL,
        delay       INTEGER,
        decay       INTEGER,
        train_start TEXT,
        train_end   TEXT,
        n_bars      INTEGER,
        evaluated_at TEXT   DEFAULT (datetime('now'))
    )""")
    # Migrate existing DBs that lack delay/decay columns
    for col, typedef in (("delay", "INTEGER"), ("decay", "INTEGER")):
        try:
            conn.execute(f"ALTER TABLE evaluations ADD COLUMN {col} {typedef}")
        except Exception:
            pass  # column already exists
    conn.execute("""CREATE TABLE IF NOT EXISTS trial_log (
        trial_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        expression  TEXT    NOT NULL,
        is_sharpe   REAL,
        saved       INTEGER DEFAULT 0,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")
    conn.commit()


def log_trial(conn, expression, is_sharpe, saved=False):
    conn.execute("INSERT INTO trial_log (expression, is_sharpe, saved) VALUES (?,?,?)",
                 (expression, is_sharpe, 1 if saved else 0))
    conn.commit()


def get_num_trials(conn):
    return conn.execute("SELECT COUNT(*) FROM trial_log").fetchone()[0]


def check_diversity(conn, expression, new_alpha_raw):
    """Check signal correlation against all existing alphas on train data.
    Rejects if expression already exists OR if signal correlation > CORR_CUTOFF.
    new_alpha_raw: the RAW expression output (before process_signal).
    """
    # Check exact duplicate
    existing = conn.execute(
        "SELECT id FROM alphas WHERE expression=? AND archived=0", (expression,)
    ).fetchone()
    if existing:
        print(f"  REJECTED: Expression already exists as alpha #{existing[0]}")
        return False

    if new_alpha_raw is None:
        return True

    rows = conn.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND asset_class='equities'"
    ).fetchall()
    if not rows:
        return True

    matrices, universe, classifications = load_data("train")

    new_processed = process_signal(
        new_alpha_raw, universe_df=universe,
        classifications=classifications, max_wt=MAX_WEIGHT
    )

    for alpha_id, alpha_expr in rows:
        try:
            existing_raw = evaluate_expression(alpha_expr, matrices)
            if existing_raw is None:
                continue
            existing_df = process_signal(
                existing_raw, universe_df=universe,
                classifications=classifications, max_wt=MAX_WEIGHT
            )
            common_idx  = new_processed.index.intersection(existing_df.index)
            common_cols = new_processed.columns.intersection(existing_df.columns)
            if len(common_idx) < 50 or len(common_cols) < 10:
                continue
            a = new_processed.loc[common_idx, common_cols].values.flatten()
            b = existing_df.loc[common_idx, common_cols].values.flatten()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 200:
                continue
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            if abs(corr) > CORR_CUTOFF:
                print(f"  REJECTED: Signal corr={corr:.3f} with alpha #{alpha_id} "
                      f"(cutoff={CORR_CUTOFF})")
                print(f"            Existing: {alpha_expr[:80]}")
                return False
        except Exception:
            continue

    return True


def save_alpha(conn, expression, reasoning, metrics):
    """Save alpha to the equities alphas table."""
    if not check_diversity(conn, expression, metrics.get('_alpha_raw')):
        return False

    c = conn.cursor()
    c.execute("""INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?)""",
              (expression,
               expression[:80],
               metrics.get('category', ''),
               'equities',
               'daily',
               'agent1_equity_research',
               reasoning))
    alpha_id = c.lastrowid

    c.execute("""INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann,
                 max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                 delay, decay, train_start, train_end, n_bars, evaluated_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
              (alpha_id,
               metrics['is_sharpe'], metrics['is_sharpe'],
               metrics.get('returns_ann', 0),
               metrics['max_drawdown'], metrics['turnover'],
               metrics['is_fitness'],
               metrics['ic_mean'], metrics['icir'],
               metrics['deflated_sharpe'],
               DELAY, DECAY,
               TRAIN_START, TRAIN_END,
               metrics.get('n_bars', 0)))
    conn.commit()
    print(f"  SAVED as alpha #{alpha_id}")
    return True


# ============================================================================
# DATA LOADING (TRAIN ONLY)
# ============================================================================

_DATA_CACHE = {}


def load_data(split="train"):
    """Load equity matrix data for the given split. Returns (matrices, universe, classifications)."""
    if split in _DATA_CACHE:
        return _DATA_CACHE[split]

    # Load universe mask (TOP1000 by ADV20)
    uni_path = UNIVERSES_DIR / f"{UNIVERSE}.parquet"
    universe_df = pd.read_parquet(uni_path)
    # Convert index to DatetimeIndex if needed
    if not isinstance(universe_df.index, pd.DatetimeIndex):
        universe_df.index = pd.to_datetime(universe_df.index)

    # Filter tickers by coverage
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    # Load all matrix files
    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        # Skip classification/group parquets — they are not numeric signal matrices
        if fp.stem.startswith("_"):
            continue
        try:
            df = pd.read_parquet(fp)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            cols = [c for c in valid_tickers if c in df.columns]
            if cols:
                matrices[fp.stem] = df[cols]
        except Exception:
            continue  # skip unreadable files

    # Load GICS classifications for sector neutralization
    # sector/subindustry are label matrices (string dtype)
    classifications = {}
    for level in ("sector", "industry", "subindustry"):
        if level in matrices:
            # Take the most recent non-null value per ticker as a static label
            label_mat = matrices.pop(level)
            # Use last row with data as the canonical label
            labels = label_mat.ffill().iloc[-1]
            classifications[level] = labels

    # Define splits
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "h1":    (SUBPERIODS[0][0], SUBPERIODS[0][1]),
        "h2":    (SUBPERIODS[1][0], SUBPERIODS[1][1]),
    }
    if split not in splits:
        raise ValueError(f"Agent 1 only uses train splits: {list(splits.keys())}")

    start, end = splits[split]

    split_matrices = {}
    for name, df in matrices.items():
        sliced = df.loc[start:end]
        if len(sliced) > 0:
            split_matrices[name] = sliced

    split_universe = universe_df[valid_tickers].loc[start:end]

    # Recompute returns from close if available (clip extreme values from data errors)
    if "close" in split_matrices:
        raw_ret = split_matrices["close"].pct_change(fill_method=None)
        # Null out extreme returns (>50% daily) - these are data errors, reverse splits, etc.
        MAX_DAILY_RETURN = 0.5
        raw_ret = raw_ret.where(raw_ret.abs() <= MAX_DAILY_RETURN, np.nan)
        split_matrices["returns"] = raw_ret
        split_matrices["log_returns"] = np.log1p(raw_ret.fillna(0))

    result = (split_matrices, split_universe, classifications)
    _DATA_CACHE[split] = result
    return result


# ============================================================================
# CORE EVALUATION
# ============================================================================

def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def simulate(alpha_df, returns_df, close_df, universe_df, classifications, fees_bps=0.0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df,
        returns_df=returns_df,
        close_df=close_df,
        universe_df=universe_df,
        classifications=classifications,
        booksize=BOOKSIZE,
        max_stock_weight=MAX_WEIGHT,
        decay=DECAY,
        delay=DELAY,
        neutralization=NEUTRALIZE,
        fees_bps=fees_bps,
        bars_per_day=BARS_PER_DAY,
    )


def process_signal(alpha_df, universe_df=None, classifications=None, max_wt=MAX_WEIGHT):
    """
    Standard Equities Signal Processing Pipeline.
    1. Apply universe mask
    2. Sector/subindustry demean (if classifications provided)
    3. Scale (Sum(abs(weights)) = 1)
    4. Clip (Position weight limit)
    """
    signal = alpha_df.copy().astype(float)

    # Apply universe mask
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)

    # Sector-relative demeaning (within subindustry group)
    if classifications is not None and NEUTRALIZE in classifications:
        groups = classifications[NEUTRALIZE]
        group_labels = groups.reindex(signal.columns)
        for grp in group_labels.dropna().unique():
            col_mask = (group_labels == grp).values
            if col_mask.any():
                grp_data = signal.iloc[:, col_mask]
                grp_mean = grp_data.mean(axis=1)
                signal.iloc[:, col_mask] = grp_data.sub(grp_mean, axis=0)
    else:
        # Fallback: market-wide demean
        row_mean = signal.mean(axis=1)
        signal = signal.sub(row_mean, axis=0)

    # Scale: sum of absolute weights = 1
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)

    # Clip positions
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
        if len(common) < 20:
            ics.append(np.nan)
            continue
        ic, _ = stats.spearmanr(a[common], r[common])
        ics.append(ic)
    return pd.Series(ics, index=alpha_lagged.index[1:])


def deflated_sharpe_ratio(observed_sr, n_trials, n_bars, skew=0, kurtosis=3):
    """Lopez de Prado DSR: P(true SR > 0) given number of trials."""
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
    """Evaluate a single expression on the given data split."""
    matrices, universe, classifications = load_data(split)
    close = matrices.get("close")
    returns_pct = matrices.get("returns")
    if returns_pct is None and close is not None:
        returns_pct = close.pct_change()

    t0 = time.time()
    try:
        alpha_raw = evaluate_expression(expression, matrices)
    except Exception as e:
        return {"success": False, "error": f"Expression error: {e}"}

    if alpha_raw is None or (hasattr(alpha_raw, 'empty') and alpha_raw.empty):
        return {"success": False, "error": "Expression returned None/empty"}

    # Equity signal processing: universe mask -> sector demean -> scale -> clip
    alpha_df = process_signal(
        alpha_raw, universe_df=universe,
        classifications=classifications, max_wt=MAX_WEIGHT
    )

    try:
        result = simulate(alpha_df, returns_pct, close, universe, classifications,
                          fees_bps=fees_bps)
    except Exception as e:
        return {"success": False, "error": f"Simulation error: {e}"}

    return {
        "success": True,
        "sharpe": result.sharpe,
        "fitness": result.fitness,
        "turnover": result.turnover,
        "max_drawdown": result.max_drawdown,
        "returns_ann": result.returns_ann,
        "n_bars": len(result.daily_pnl),
        "pnl_vec": np.array(result.daily_pnl),
        "alpha_df": alpha_df,
        "alpha_raw": alpha_raw,
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

    # IC (cross-sectional rank IC)
    ic_series = compute_ic(is_m["alpha_df"], is_m["returns_pct"])
    ic_clean = ic_series.dropna()
    ic_mean = ic_clean.mean() if len(ic_clean) > 0 else 0
    ic_std  = ic_clean.std()  if len(ic_clean) > 1 else 1
    icir    = ic_mean / ic_std if ic_std > 0 else 0

    # Sub-period stability
    stability = {}
    for start, end, name in SUBPERIODS:
        sub = eval_single(expression, split=name.lower(), fees_bps=0)
        stability[name] = sub["sharpe"] if sub["success"] else 0

    # DSR (Deflated Sharpe Ratio)
    dsr = deflated_sharpe_ratio(is_m["sharpe"], n_trials, is_m["n_bars"])

    # PnL distribution metrics (matching crypto harness)
    pnl = is_m["pnl_vec"]
    pnl_kurtosis = float(pd.Series(pnl).kurtosis()) if len(pnl) > 10 else 0
    pnl_skew     = float(pd.Series(pnl).skew())     if len(pnl) > 10 else 0
    rolling_sr   = pd.Series(pnl).rolling(60).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0
    ).dropna()
    rolling_sr_std = float(rolling_sr.std()) if len(rolling_sr) > 10 else 999

    return {
        "success": True,
        "is_sharpe": is_m["sharpe"],
        "is_fitness": is_m["fitness"],
        "turnover": is_m["turnover"],
        "max_drawdown": is_m["max_drawdown"],
        "returns_ann": is_m["returns_ann"],
        "n_bars": is_m["n_bars"],
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "stability_h1": stability.get("H1", 0),
        "stability_h2": stability.get("H2", 0),
        "deflated_sharpe": dsr,
        "n_trials": n_trials,
        "pnl_vec": is_m["pnl_vec"],
        "pnl_kurtosis": pnl_kurtosis,
        "pnl_skew": pnl_skew,
        "rolling_sr_std": rolling_sr_std,
        "_alpha_raw": is_m["alpha_raw"],  # RAW signal for correlation check on save
    }


# ============================================================================
# LISTING / SCOREBOARD
# ============================================================================

def list_alphas():
    conn = get_conn()
    rows = conn.execute("""
        SELECT a.id, a.expression, a.source,
               COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.turnover, 0), COALESCE(e.ic_mean, 0)
        FROM alphas a
        LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class = 'equities'
        ORDER BY COALESCE(e.fitness, 0) DESC
    """).fetchall()
    n_trials = get_num_trials(conn)
    conn.close()

    print(f"\n{'='*75}")
    print(f"EQUITIES ALPHA DATABASE: {len(rows)} active alphas | {n_trials} agent trials")
    print(f"{'='*75}")
    for r in rows:
        print(f"  #{r[0]:3d} | SR={r[3]:+.2f} Fit={r[4]:.2f} TO={r[5]:.3f} IC={r[6]:+.4f} "
              f"| src={r[2] or '?':25s} | {r[1][:50]}")


def print_scoreboard():
    conn = get_conn()
    n_alphas = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND asset_class='equities'"
    ).fetchone()[0]
    n_agent = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE source='agent1_equity_research' AND archived=0"
    ).fetchone()[0]
    n_trials = get_num_trials(conn)
    rows = conn.execute("""
        SELECT a.expression, COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.ic_mean, 0), COALESCE(e.psr, 0), COALESCE(e.turnover, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.source = 'agent1_equity_research'
        ORDER BY COALESCE(e.fitness, 0) DESC
    """).fetchall()
    conn.close()

    print(f"\n{'='*75}")
    print(f"  EQUITIES ALPHA RESEARCH SCOREBOARD (Agent 1)")
    print(f"  Total equities alphas in DB: {n_alphas} | Agent 1 alphas: {n_agent} | Trials: {n_trials}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (no fees)")
    print(f"  Universe: {UNIVERSE} | Neutralization: {NEUTRALIZE} | Delay: {DELAY}d | Decay: {DECAY}")
    print(f"{'='*75}")
    if rows:
        print(f"  Avg IS Sharpe:  {np.mean([r[1] for r in rows]):+.3f}")
        print(f"  Avg Mean IC:    {np.mean([r[3] for r in rows]):+.4f}")
        print(f"  Avg Turnover:   {np.mean([r[5] for r in rows]):.4f}")
        print(f"\n  Agent 1 equities alphas (top 15):")
        for i, r in enumerate(rows[:15]):
            print(f"    {i+1:2d}. SR={r[1]:+.2f} Fit={r[2]:.2f} IC={r[3]:+.4f} TO={r[5]:.4f} "
                  f"DSR={r[4]:.2f} | {r[0][:80]}")
    else:
        print("  No equities Agent 1 alphas yet. Run --expr to start researching.")
    print(f"{'='*75}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agent 1: Equities Alpha Discovery (Train Only, daily bars)"
    )
    parser.add_argument("--expr",      type=str, default=None, help="Alpha expression to evaluate")
    parser.add_argument("--save",      action="store_true",    help="Save if all quality gates pass")
    parser.add_argument("--list",      action="store_true",    help="List all alphas in DB")
    parser.add_argument("--scoreboard",action="store_true",    help="Print full scoreboard")
    parser.add_argument("--reasoning", type=str, default="",   help="Economic reasoning (used with --save)")
    parser.add_argument("--decay",     type=int, default=None,
                        help="Override global DECAY (linear decay window, 0=off). "
                             "Higher decay = smoother positions = lower turnover = better Fitness. "
                             "Useful when signal turnover > 0.125 and Fitness is the binding gate.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    if args.scoreboard:
        print_scoreboard()
        return
    if args.list:
        list_alphas()
        return

    if not args.expr:
        parser.print_help()
        return

    # Allow --decay to override the global at runtime
    global DECAY
    if args.decay is not None:
        DECAY = args.decay

    conn = get_conn()
    expression = args.expr
    print(f"\nEvaluating (equities): {expression}")
    print(f"  Universe: {UNIVERSE} | Period: {TRAIN_START} -> {TRAIN_END} | Delay: {DELAY}d | Decay: {DECAY} | Neutral: {NEUTRALIZE}")

    result = eval_full(expression, conn)
    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        log_trial(conn, expression, 0, saved=False)
        conn.close()
        return

    # ── Print Results ──────────────────────────────────────────────────────
    print(f"\n  --- IN-SAMPLE ({TRAIN_START} to {TRAIN_END}, no fees) ---")
    print(f"  IS Sharpe:    {result['is_sharpe']:+.3f}   (gate: >= {MIN_IS_SHARPE})")
    print(f"  Fitness:      {result['is_fitness']:.3f}   (gate: >= {MIN_FITNESS})")
    print(f"  Turnover:     {result['turnover']:.4f}  (daily; equities penalties are mild)")
    print(f"  Max Drawdown: {result['max_drawdown']:.3f}")
    print(f"  Ann. Return:  {result['returns_ann']:+.3f}")

    print(f"\n  --- INFORMATION COEFFICIENT ---")
    print(f"  Mean IC:      {result['ic_mean']:+.4f}   (gate: > {MIN_IC_MEAN})")
    print(f"  IC Std:       {result['ic_std']:.4f}")
    print(f"  ICIR:         {result['icir']:.3f}")

    print(f"\n  --- SUB-PERIOD STABILITY ---")
    print(f"  H1 ({SUBPERIODS[0][0][:7]} to {SUBPERIODS[0][1][:7]}): {result['stability_h1']:+.3f}")
    print(f"  H2 ({SUBPERIODS[1][0][:7]} to {SUBPERIODS[1][1][:7]}): {result['stability_h2']:+.3f}")
    both_pos = result['stability_h1'] > MIN_SUB_SHARPE and result['stability_h2'] > MIN_SUB_SHARPE
    print(f"  Stable:       {'YES (both >= ' + str(MIN_SUB_SHARPE) + ')' if both_pos else 'NO'}")

    print(f"\n  --- DEFLATED SHARPE (trial #{result['n_trials']}) ---")
    print(f"  DSR:          {result['deflated_sharpe']:.3f}  (informational)")

    print(f"\n  --- PNL DISTRIBUTION ---")
    print(f"  PnL Kurtosis:   {result['pnl_kurtosis']:.1f}   (gate: <= {MAX_PNL_KURTOSIS})")
    print(f"  PnL Skewness:   {result['pnl_skew']:+.3f}  (gate: >= {MIN_PNL_SKEW})")
    print(f"  Rolling SR std: {result['rolling_sr_std']:.4f} (gate: <= {MAX_ROLLING_SR_STD})")

    # ── Quality Gates ──────────────────────────────────────────────────────
    print(f"\n  --- QUALITY GATES ---")
    gates = [
        (result['is_sharpe'] >= MIN_IS_SHARPE,
         f"IS Sharpe >= {MIN_IS_SHARPE}: {result['is_sharpe']:+.3f}"),
        (result['is_fitness'] >= MIN_FITNESS,
         f"Fitness >= {MIN_FITNESS}: {result['is_fitness']:.3f}"),
        (result['ic_mean'] > MIN_IC_MEAN,
         f"Mean IC > {MIN_IC_MEAN}: {result['ic_mean']:+.4f}"),
        (result['stability_h1'] > MIN_SUB_SHARPE and result['stability_h2'] > MIN_SUB_SHARPE,
         f"Sub-period Sharpe > {MIN_SUB_SHARPE}: H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}"),
        (result['turnover'] <= MAX_TURNOVER,
         f"Turnover <= {MAX_TURNOVER}: {result['turnover']:.4f}"),
        (result['pnl_kurtosis'] <= MAX_PNL_KURTOSIS,
         f"PnL Kurtosis <= {MAX_PNL_KURTOSIS}: {result['pnl_kurtosis']:.1f}"),
        (result['rolling_sr_std'] <= MAX_ROLLING_SR_STD,
         f"Rolling SR Consistency <= {MAX_ROLLING_SR_STD}: {result['rolling_sr_std']:.4f}"),
        (result['pnl_skew'] >= MIN_PNL_SKEW,
         f"PnL Skew >= {MIN_PNL_SKEW}: {result['pnl_skew']:+.3f}"),
    ]
    print(f"  [INFO] DSR: {result['deflated_sharpe']:.3f} (informational, not a gate)")
    all_pass = True
    for passed, desc in gates:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  [OK] ALL GATES PASSED -- use --save to add to database")
    else:
        print(f"\n  [FAIL] GATE(S) FAILED -- iterate on the expression")

    log_trial(conn, expression, result['is_sharpe'], saved=False)

    if args.save:
        if not all_pass:
            print(f"\n  NOT SAVED: Failed quality gates above")
        else:
            saved = save_alpha(conn, expression, args.reasoning, result)
            if saved:
                conn.execute(
                    "UPDATE trial_log SET saved=1 WHERE trial_id=(SELECT MAX(trial_id) FROM trial_log)"
                )
                conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
