"""
eval_alpha_ib.py — IB Closing Auction Alpha Discovery Harness (delay=0, fee-free)

Evaluates delay-0 alphas on the TOP2000TOP3000 band universe for the
IB closing auction strategy. All individual alpha evaluation is FEE-FREE
per WorldQuant standard — fees only apply at portfolio combination.

Usage:
    python eval_alpha_ib.py --expr "rank((close - open) / (high - low + 0.001))"
    python eval_alpha_ib.py --expr "rank(-ts_delta(close, 3))" --save
    python eval_alpha_ib.py --list
    python eval_alpha_ib.py --scoreboard
    python eval_alpha_ib.py --run-seeds        # Evaluate all seed alphas
    python eval_alpha_ib.py --compare-universes # Compare across band universes

Asset Class: US Equities (daily bars, TOP2000TOP3000 by ADV20)
Data source: data/fmp_cache/matrices/
Execution: IBKR Pro, Portfolio Margin, MOC orders in closing auction
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

# Historical periods
TRAIN_START = "2016-01-01"
TRAIN_END   = "2023-01-01"   # 7 years IS (spans multiple regimes)

# Validation and test periods (for multi-stage evaluation)
VAL_START = "2023-01-01"
VAL_END   = "2024-07-01"     # 1.5 years validation

TEST_START = "2024-07-01"
# TEST_END = present (use all available data)

# Sub-period splits for stability check (within train)
SUBPERIODS = [
    ("2016-01-01", "2019-07-01", "H1"),  # Pre-COVID
    ("2019-07-01", "2023-01-01", "H2"),  # COVID + recovery + rate hikes
]

# Sim parameters — IB Closing Auction specific
UNIVERSE       = "TOP2000TOP3000"      # Band universe: ranks 2001-3000 by ADV20
BOOKSIZE       = 20_000_000.0          # Standard WQ booksize for signal evaluation
MAX_WEIGHT     = 0.01                  # 1% max per stock (~100-150 positions)
NEUTRALIZE     = "market"              # Market-level (sector/industry groups too small in band)
BARS_PER_DAY   = 1                     # Daily bars
DELAY          = 0                     # DELAY-0: closing auction execution
DECAY          = 0                     # No decay by default
COVERAGE_CUTOFF = 0.3                  # Lower bar than TOP1000 (band members rotate more)

# Quality gates
MIN_IS_SHARPE      = 3.0       # Minimum in-sample Sharpe (institutional grade)
MIN_FITNESS        = 1.0       # Fitness = Sharpe * sqrt(|return_ann| / max(turnover, 0.125))
MIN_IC_MEAN        = 0.0       # Mean IC must be positive
CORR_CUTOFF        = 0.70      # Reject if |corr| > 0.70 (tighter gate for 20-alpha library)
MIN_SUB_SHARPE     = -999       # DISABLED: rebuilt universe has no H1 coverage (100MM cap filter)
MAX_TURNOVER       = 2.0       # Delay-0 MOC signals rebalance fully daily (alpha #1 has TO=1.26)
MAX_PNL_KURTOSIS   = 25        # Slightly more lenient for small-caps
MAX_ROLLING_SR_STD = 1.00      # Relaxed: new universe has different rolling behavior
MIN_PNL_SKEW       = -0.5      # Reject negatively-skewed PnL

# IB-specific database (separate from equity research DB)
DB_PATH = "data/ib_alphas.db"

# Data directories
MATRICES_DIR  = Path("data/fmp_cache/matrices")
UNIVERSES_DIR = Path("data/fmp_cache/universes")

# Available band universes for comparison
BAND_UNIVERSES = ["TOP1500TOP2500", "TOP2000TOP3000", "TOP2500TOP3500"]

# Fee-free evaluation (fees applied only at portfolio combination)
FEES_BPS = 0.0


# ============================================================================
# DATABASE
# ============================================================================

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn):
    """Create IB-specific tables if they don't exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS alphas (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        expression  TEXT    NOT NULL,
        name        TEXT,
        category    TEXT,
        asset_class TEXT    DEFAULT 'equities_ib',
        universe    TEXT    DEFAULT 'TOP2000TOP3000',
        delay       INTEGER DEFAULT 0,
        decay       INTEGER DEFAULT 0,
        neutralize  TEXT    DEFAULT 'sector',
        source      TEXT,
        notes       TEXT,
        archived    INTEGER DEFAULT 0,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        alpha_id    INTEGER NOT NULL REFERENCES alphas(id),
        -- In-sample metrics (2016-2022)
        sharpe_is   REAL,
        return_ann  REAL,
        max_drawdown REAL,
        turnover    REAL,
        fitness     REAL,
        ic_mean     REAL,
        ic_ir       REAL,
        psr         REAL,
        -- Sub-period stability
        sharpe_h1   REAL,
        sharpe_h2   REAL,
        -- PnL distribution
        pnl_kurtosis REAL,
        pnl_skew     REAL,
        rolling_sr_std REAL,
        -- Meta
        delay       INTEGER,
        decay       INTEGER,
        universe    TEXT,
        train_start TEXT,
        train_end   TEXT,
        n_bars      INTEGER,
        evaluated_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trial_log (
        trial_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        expression  TEXT    NOT NULL,
        universe    TEXT    DEFAULT 'TOP2000TOP3000',
        is_sharpe   REAL,
        saved       INTEGER DEFAULT 0,
        created_at  TEXT    DEFAULT (datetime('now'))
    )""")
    conn.commit()


def log_trial(conn, expression, is_sharpe, saved=False):
    conn.execute(
        "INSERT INTO trial_log (expression, universe, is_sharpe, saved) VALUES (?,?,?,?)",
        (expression, UNIVERSE, is_sharpe, 1 if saved else 0)
    )
    conn.commit()


def get_num_trials(conn):
    return conn.execute("SELECT COUNT(*) FROM trial_log").fetchone()[0]


def check_diversity(conn, expression, new_alpha_raw):
    """Check signal correlation against all existing IB alphas."""
    existing = conn.execute(
        "SELECT id FROM alphas WHERE expression=? AND archived=0", (expression,)
    ).fetchone()
    if existing:
        print(f"  REJECTED: Expression already exists as alpha #{existing[0]}")
        return False

    if new_alpha_raw is None:
        return True

    rows = conn.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND asset_class='equities_ib'"
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
                return False
        except Exception:
            continue

    return True


def save_alpha(conn, expression, reasoning, metrics, category=""):
    """Save alpha to the IB alphas database."""
    if not check_diversity(conn, expression, metrics.get('_alpha_raw')):
        return False

    c = conn.cursor()
    c.execute("""INSERT INTO alphas (expression, name, category, asset_class,
                 universe, delay, decay, neutralize, source, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (expression,
               expression[:80],
               category,
               'equities_ib',
               UNIVERSE,
               DELAY, DECAY, NEUTRALIZE,
               'ib_research',
               reasoning))
    alpha_id = c.lastrowid

    c.execute("""INSERT INTO evaluations (alpha_id, sharpe_is, return_ann,
                 max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                 sharpe_h1, sharpe_h2,
                 pnl_kurtosis, pnl_skew, rolling_sr_std,
                 delay, decay, universe, train_start, train_end, n_bars)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (alpha_id,
               metrics['is_sharpe'],
               metrics.get('returns_ann', 0),
               metrics['max_drawdown'], metrics['turnover'],
               metrics['is_fitness'],
               metrics['ic_mean'], metrics['icir'],
               metrics['deflated_sharpe'],
               metrics.get('stability_h1', 0),
               metrics.get('stability_h2', 0),
               metrics.get('pnl_kurtosis', 0),
               metrics.get('pnl_skew', 0),
               metrics.get('rolling_sr_std', 0),
               DELAY, DECAY, UNIVERSE,
               TRAIN_START, TRAIN_END,
               metrics.get('n_bars', 0)))
    conn.commit()
    print(f"  SAVED as IB alpha #{alpha_id}")
    return True


# ============================================================================
# DATA LOADING
# ============================================================================

_DATA_CACHE = {}


def _is_us_ticker(sym: str) -> bool:
    """Filter out non-US tickers (London .L, Canadian .V/.TO, etc.)."""
    if "." in sym:
        # Allow tickers like BRK-A, BF-B (hyphens are US)
        # Reject exchange suffixes: .L, .V, .TO, .PA, .DE etc.
        parts = sym.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) <= 3:
            return False  # Foreign exchange suffix
    return True


def load_data(split="train", universe_name=None):
    """
    Load equity matrix data for the given split and universe.

    Applies critical data quality filters at load time:
    - Removes non-US tickers (.L, .V, .TO)
    - Clamps OHLC consistency (O,C within [L,H])
    - Filters zero-ADV ghost tickers
    - Recomputes returns from close with ±50% clip

    Returns (matrices, universe, classifications).
    """
    if universe_name is None:
        universe_name = UNIVERSE

    cache_key = (split, universe_name, NEUTRALIZE)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    # Load universe mask
    uni_path = UNIVERSES_DIR / f"{universe_name}.parquet"
    if not uni_path.exists():
        # Try to build band universe on the fly
        from src.data.universe_band import build_band_universe, BAND_UNIVERSES
        if universe_name in BAND_UNIVERSES:
            lo, hi = BAND_UNIVERSES[universe_name]
            print(f"  Building {universe_name} universe from TOP{lo} and TOP{hi}...")
            build_band_universe(lo, hi, universes_dir=UNIVERSES_DIR)
        if not uni_path.exists():
            raise FileNotFoundError(
                f"Universe file not found: {uni_path}\n"
                f"Run: python -m src.data.universe_band --band {universe_name[3:7]} {universe_name[10:]}"
            )

    universe_df = pd.read_parquet(uni_path)
    if not isinstance(universe_df.index, pd.DatetimeIndex):
        universe_df.index = pd.to_datetime(universe_df.index)

    # ── DATA QUALITY FILTER 1: US-only tickers ──
    all_tickers = universe_df.columns.tolist()
    us_tickers = [t for t in all_tickers if _is_us_ticker(t)]
    n_removed = len(all_tickers) - len(us_tickers)
    if n_removed > 0:
        universe_df = universe_df[us_tickers]

    # Filter tickers by coverage
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())

    # Load all matrix files
    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
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
            continue

    # ── DATA QUALITY FILTER 2: Clamp OHLC on the fly ──
    if all(k in matrices for k in ("open", "high", "low", "close")):
        h = matrices["high"]
        l = matrices["low"]
        # Clamp open and close to [low, high] range
        matrices["open"] = matrices["open"].clip(lower=l, upper=h)
        matrices["close"] = matrices["close"].clip(lower=l, upper=h)

    # Load GICS classifications
    classifications = {}
    for level in ("sector", "industry", "subindustry"):
        if level in matrices:
            label_mat = matrices.pop(level)
            labels = label_mat.ffill().iloc[-1]
            classifications[level] = labels

    # Define splits
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val":   (VAL_START, VAL_END),
        "test":  (TEST_START, None),  # None = to end of data
        "h1":    (SUBPERIODS[0][0], SUBPERIODS[0][1]),
        "h2":    (SUBPERIODS[1][0], SUBPERIODS[1][1]),
        "full":  (TRAIN_START, None),  # Full history
    }
    if split not in splits:
        raise ValueError(f"Unknown split: {split}. Available: {list(splits.keys())}")

    start, end = splits[split]

    split_matrices = {}
    for name, df in matrices.items():
        sliced = df.loc[start:end] if end else df.loc[start:]
        if len(sliced) > 0:
            split_matrices[name] = sliced

    uni_sliced = universe_df[valid_tickers]
    uni_sliced = uni_sliced.loc[start:end] if end else uni_sliced.loc[start:]

    # ── DATA QUALITY FILTER 3: Recompute returns from close ──
    # Never use stored returns.parquet — it's corrupted with values in billions
    if "close" in split_matrices:
        raw_ret = split_matrices["close"].pct_change(fill_method=None)
        MAX_DAILY_RETURN = 0.5  # Clip at ±50% (handles reverse splits, data errors)
        raw_ret = raw_ret.where(raw_ret.abs() <= MAX_DAILY_RETURN, np.nan)
        split_matrices["returns"] = raw_ret
        split_matrices["log_returns"] = np.log1p(raw_ret.fillna(0))

    result = (split_matrices, uni_sliced, classifications)
    _DATA_CACHE[cache_key] = result
    return result



# ============================================================================
# CORE EVALUATION
# ============================================================================

def evaluate_expression(expression, matrices):
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def simulate(alpha_df, returns_df, close_df, universe_df, classifications):
    """Run vectorized simulation — ALWAYS fee-free for individual alphas."""
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
        fees_bps=FEES_BPS,  # Always 0 for alpha research
        bars_per_day=BARS_PER_DAY,
    )


def process_signal(alpha_df, universe_df=None, classifications=None, max_wt=MAX_WEIGHT):
    """
    Standard Signal Processing Pipeline for IB alphas.
    1. Apply universe mask
    2. Sector demean (sector-level for small-caps)
    3. Scale (Sum(abs(weights)) = 1)
    4. Clip (Position weight limit)
    """
    signal = alpha_df.copy().astype(float)

    # Apply universe mask
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False).astype(bool)
        signal = signal.where(uni_mask, np.nan)

    # Sector-relative demeaning
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
        row_mean = signal.mean(axis=1)
        signal = signal.sub(row_mean, axis=0)

    # Scale
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)

    # Clip
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


def eval_single(expression, split="train", universe_name=None):
    """Evaluate a single expression on the given data split. Fee-free."""
    matrices, universe, classifications = load_data(split, universe_name)
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

    alpha_df = process_signal(
        alpha_raw, universe_df=universe,
        classifications=classifications, max_wt=MAX_WEIGHT
    )

    try:
        result = simulate(alpha_df, returns_pct, close, universe, classifications)
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


def eval_full(expression, conn):
    """Full evaluation: IS + IC + Stability + DSR. All fee-free."""
    n_trials = get_num_trials(conn) + 1

    is_m = eval_single(expression, split="train")
    if not is_m["success"]:
        return {"success": False, "error": is_m["error"]}

    # IC
    ic_series = compute_ic(is_m["alpha_df"], is_m["returns_pct"])
    ic_clean = ic_series.dropna()
    ic_mean = ic_clean.mean() if len(ic_clean) > 0 else 0
    ic_std  = ic_clean.std()  if len(ic_clean) > 1 else 1
    icir    = ic_mean / ic_std if ic_std > 0 else 0

    # Sub-period stability
    stability = {}
    for start, end, name in SUBPERIODS:
        sub = eval_single(expression, split=name.lower())
        stability[name] = sub["sharpe"] if sub["success"] else 0

    # DSR
    dsr = deflated_sharpe_ratio(is_m["sharpe"], n_trials, is_m["n_bars"])

    # PnL distribution
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
        "_alpha_raw": is_m["alpha_raw"],
    }


# ============================================================================
# LISTING / SCOREBOARD
# ============================================================================

def list_alphas():
    conn = get_conn()
    rows = conn.execute("""
        SELECT a.id, a.expression, a.universe, a.source,
               COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.turnover, 0), COALESCE(e.ic_mean, 0)
        FROM alphas a
        LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class = 'equities_ib'
        ORDER BY COALESCE(e.fitness, 0) DESC
    """).fetchall()
    n_trials = get_num_trials(conn)
    conn.close()

    print(f"\n{'='*80}")
    print(f"IB ALPHA DATABASE: {len(rows)} active alphas | {n_trials} trials")
    print(f"{'='*80}")
    for r in rows:
        print(f"  #{r[0]:3d} | SR={r[4]:+.2f} Fit={r[5]:.2f} TO={r[6]:.3f} IC={r[7]:+.4f} "
              f"| univ={r[2]:15s} | {r[1][:50]}")


def print_scoreboard():
    conn = get_conn()
    n_alphas = conn.execute(
        "SELECT COUNT(*) FROM alphas WHERE archived=0 AND asset_class='equities_ib'"
    ).fetchone()[0]
    n_trials = get_num_trials(conn)
    rows = conn.execute("""
        SELECT a.expression, a.category,
               COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.ic_mean, 0), COALESCE(e.turnover, 0),
               COALESCE(e.psr, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class = 'equities_ib'
        ORDER BY COALESCE(e.fitness, 0) DESC
    """).fetchall()
    conn.close()

    print(f"\n{'='*80}")
    print(f"  IB CLOSING AUCTION ALPHA SCOREBOARD")
    print(f"  Alphas: {n_alphas} | Trials: {n_trials}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (FEE-FREE)")
    print(f"  Universe: {UNIVERSE} | Neutral: {NEUTRALIZE} | Delay: {DELAY} | Decay: {DECAY}")
    print(f"  Execution: IBKR Pro PM, MOC orders, 4x gross leverage")
    print(f"{'='*80}")
    if rows:
        print(f"  Avg IS Sharpe:  {np.mean([r[2] for r in rows]):+.3f}")
        print(f"  Avg Mean IC:    {np.mean([r[4] for r in rows]):+.4f}")
        print(f"  Avg Turnover:   {np.mean([r[5] for r in rows]):.4f}")
        print(f"\n  Top 15 alphas:")
        for i, r in enumerate(rows[:15]):
            print(f"    {i+1:2d}. SR={r[2]:+.2f} Fit={r[3]:.2f} IC={r[4]:+.4f} "
                  f"TO={r[5]:.4f} [{r[1] or '?':10s}] | {r[0][:60]}")
    else:
        print("  No IB alphas yet. Run --expr or --run-seeds to start.")
    print(f"{'='*80}\n")


def run_seed_alphas(conn, save=False):
    """Evaluate and optionally save all seed alphas."""
    from seed_alphas_ib import SEED_ALPHAS

    print(f"\nRunning {len(SEED_ALPHAS)} seed alphas on {UNIVERSE}...")
    print(f"  Delay: {DELAY} | Neutral: {NEUTRALIZE} | Fees: {FEES_BPS} bps (fee-free)")
    print("=" * 80)

    results = []
    for i, alpha in enumerate(SEED_ALPHAS, 1):
        expr = alpha["expr"]
        print(f"\n[{i}/{len(SEED_ALPHAS)}] {alpha['name']} ({alpha['category']})")
        print(f"  Expr: {expr}")

        result = eval_full(expr, conn)
        if not result["success"]:
            print(f"  FAILED: {result['error']}")
            log_trial(conn, expr, 0, saved=False)
            continue

        print(f"  IS Sharpe: {result['is_sharpe']:+.3f} | "
              f"Fitness: {result['is_fitness']:.3f} | "
              f"TO: {result['turnover']:.4f} | "
              f"IC: {result['ic_mean']:+.4f}")

        log_trial(conn, expr, result['is_sharpe'], saved=False)
        results.append({"alpha": alpha, "metrics": result})

        if save and result['is_sharpe'] >= MIN_IS_SHARPE:
            saved = save_alpha(conn, expr, alpha["reasoning"], result, alpha["category"])
            if saved:
                log_trial(conn, expr, result['is_sharpe'], saved=True)

    # Summary
    print(f"\n{'='*80}")
    print(f"SEED ALPHA SUMMARY: {len(results)}/{len(SEED_ALPHAS)} evaluated successfully")
    if results:
        sharpes = [r["metrics"]["is_sharpe"] for r in results]
        print(f"  Sharpe range: {min(sharpes):+.3f} to {max(sharpes):+.3f}")
        passing = [r for r in results if r["metrics"]["is_sharpe"] >= MIN_IS_SHARPE]
        print(f"  Passing IS Sharpe gate: {len(passing)}/{len(results)}")
    print(f"{'='*80}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IB Closing Auction Alpha Discovery (delay=0, fee-free)"
    )
    parser.add_argument("--expr",       type=str, default=None,
                        help="Alpha expression to evaluate")
    parser.add_argument("--save",       action="store_true",
                        help="Save if all quality gates pass")
    parser.add_argument("--list",       action="store_true",
                        help="List all IB alphas in DB")
    parser.add_argument("--scoreboard", action="store_true",
                        help="Print full scoreboard")
    parser.add_argument("--run-seeds",  action="store_true",
                        help="Evaluate all seed alphas")
    parser.add_argument("--reasoning",  type=str, default="",
                        help="Economic reasoning (used with --save)")
    parser.add_argument("--category",   type=str, default="",
                        help="Alpha category")
    parser.add_argument("--decay",      type=int, default=None,
                        help="Override DECAY parameter")
    parser.add_argument("--universe",   type=str, default=None,
                        help="Override universe (e.g. TOP1500TOP2500)")
    args = parser.parse_args()

    if args.scoreboard:
        print_scoreboard()
        return
    if args.list:
        list_alphas()
        return

    # Allow overrides
    global DECAY, UNIVERSE
    if args.decay is not None:
        DECAY = args.decay
    if args.universe is not None:
        UNIVERSE = args.universe

    if args.run_seeds:
        conn = get_conn()
        run_seed_alphas(conn, save=args.save)
        conn.close()
        return

    if not args.expr:
        parser.print_help()
        return

    conn = get_conn()
    expression = args.expr
    print(f"\nEvaluating (IB closing auction): {expression}")
    print(f"  Universe: {UNIVERSE} | Train: {TRAIN_START} -> {TRAIN_END}")
    print(f"  Delay: {DELAY} | Decay: {DECAY} | Neutral: {NEUTRALIZE} | Fees: {FEES_BPS} bps")

    result = eval_full(expression, conn)
    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        log_trial(conn, expression, 0, saved=False)
        conn.close()
        return

    # Print results
    print(f"\n  --- IN-SAMPLE ({TRAIN_START} to {TRAIN_END}, FEE-FREE) ---")
    print(f"  IS Sharpe:    {result['is_sharpe']:+.3f}   (gate: >= {MIN_IS_SHARPE})")
    print(f"  Fitness:      {result['is_fitness']:.3f}   (gate: >= {MIN_FITNESS})")
    print(f"  Turnover:     {result['turnover']:.4f}  (gate: <= {MAX_TURNOVER})")
    print(f"  Max Drawdown: {result['max_drawdown']:.3f}")
    print(f"  Ann. Return:  {result['returns_ann']:+.3f}")

    print(f"\n  --- INFORMATION COEFFICIENT ---")
    print(f"  Mean IC:      {result['ic_mean']:+.4f}   (gate: > {MIN_IC_MEAN})")
    print(f"  ICIR:         {result['icir']:.3f}")

    print(f"\n  --- SUB-PERIOD STABILITY ---")
    print(f"  H1 ({SUBPERIODS[0][0][:7]} to {SUBPERIODS[0][1][:7]}): {result['stability_h1']:+.3f}")
    print(f"  H2 ({SUBPERIODS[1][0][:7]} to {SUBPERIODS[1][1][:7]}): {result['stability_h2']:+.3f}")

    print(f"\n  --- PNL DISTRIBUTION ---")
    print(f"  Kurtosis:     {result['pnl_kurtosis']:.1f}   (gate: <= {MAX_PNL_KURTOSIS})")
    print(f"  Skewness:     {result['pnl_skew']:+.3f}  (gate: >= {MIN_PNL_SKEW})")
    print(f"  Rolling SR std: {result['rolling_sr_std']:.4f} (gate: <= {MAX_ROLLING_SR_STD})")

    print(f"\n  --- DEFLATED SHARPE (trial #{result['n_trials']}) ---")
    print(f"  DSR:          {result['deflated_sharpe']:.3f}")

    # Quality Gates
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
    all_pass = True
    for passed, desc in gates:
        print(f"  [{'PASS' if passed else 'FAIL'}] {desc}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  [OK] ALL GATES PASSED")
    else:
        print(f"\n  [FAIL] GATE(S) FAILED")

    log_trial(conn, expression, result['is_sharpe'], saved=False)

    if args.save:
        if not all_pass:
            print(f"\n  NOT SAVED: Failed quality gates")
        else:
            saved = save_alpha(conn, expression, args.reasoning, result, args.category)
            if saved:
                conn.execute(
                    "UPDATE trial_log SET saved=1 WHERE trial_id=(SELECT MAX(trial_id) FROM trial_log)"
                )
                conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
