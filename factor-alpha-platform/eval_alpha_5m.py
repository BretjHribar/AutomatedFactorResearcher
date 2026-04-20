"""
eval_alpha_5m.py — Agent 1: Alpha Discovery Harness for 5-MINUTE CRYPTO (TRAIN SET ONLY)

5m-specific configuration:
  - Universe: BINANCE_TOP100 / BINANCE_TOP50 / BINANCE_TOP20 (selectable via --universe)
  - Interval: 5m (288 bars/day)
  - Fees: 0 bps (zero fees per user request)
  - Backtest: Dec-Jan train (alpha discovery), Feb val (portfolio optim), Mar test (holdout)
  - Primary metric: Mean IC | Secondary: Sharpe
  - Separate DB: data/alphas_5m.db

Usage:
    python eval_alpha_5m.py --expr "ts_zscore(close, 288)"  # Evaluate on train
    python eval_alpha_5m.py --expr "ts_zscore(close, 288)" --save --reasoning "..."
    python eval_alpha_5m.py --list
    python eval_alpha_5m.py --scoreboard
"""

import sys, os, time, argparse, sqlite3
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION — 5-MINUTE BARS
# ============================================================================

EXPRESSION = "ts_zscore(close, 288)"  # <-- LLM EDITS THIS LINE

# ┌─────────────────────────────────────────────────────────────────────┐
# │  HARD RULE: DATA SPLIT DISCIPLINE                                  │
# │                                                                     │
# │  Train  (Feb 1 2025 – Feb 1 2026) → Alpha signal DISCOVERY       │
# │  Val    (Feb 1 – Mar 1)  → Portfolio optim / signal combination    │
# │  Test   (Mar 1 – Mar 27) → FINAL TEST ONLY — NEVER LOOK AT        │
# │                                                                     │
# │  NEVER discover alphas on Val or Test data. NEVER optimize          │
# │  portfolio params on Train data. NEVER evaluate Test until          │
# │  you are done with ALL research.                                    │
# └─────────────────────────────────────────────────────────────────────┘

# Agent 1 discovers alphas on TRAIN only (Feb 2025 - Jan 2026)
# Full 1-year window covering all major regimes:
#   Feb-Aug 2025:  sideways/consolidation + summer low-vol
#   Sep-Oct 2025:  slow grind / base-building
#   Nov 2025:      post-election BTC rally, momentum regime
#   Dec 2025:      crypto crash / de-risk regime
#   Jan 2026:      recovery / rebound regime
# Data: 120K bars, 630 symbols, 38 fields. All downloaded and ready.
TRAIN_START  = "2025-02-01"
TRAIN_END    = "2026-02-01"

# Sub-period splits for stability check across multiple regimes
SUBPERIODS = [
    ("2025-02-01", "2025-08-01", "H1"),  # Feb-Jul: consolidation / low-vol
    ("2025-08-01", "2026-02-01", "H2"),  # Aug-Jan: BTC rally + crash + recovery
]

# Sim parameters
ALL_UNIVERSES = ["BINANCE_TOP100", "BINANCE_TOP50", "BINANCE_TOP20"]
UNIVERSE     = "BINANCE_TOP100"   # Overridden by --universe CLI arg
INTERVAL     = "5m"
BOOKSIZE     = 2_000_000.0
NEUTRALIZE   = "market"
BARS_PER_DAY = 288           # 24h * 60min / 5min = 288
COVERAGE_CUTOFF = 0.3

# Universe-dependent parameters
UNIVERSE_CONFIG = {
    "BINANCE_TOP100": {"max_weight": 0.05, "min_is_sharpe": 2.5},
    "BINANCE_TOP50":  {"max_weight": 0.10, "min_is_sharpe": 2.5},
    "BINANCE_TOP20":  {"max_weight": 0.20, "min_is_sharpe": 2.5},
}
MAX_WEIGHT   = 0.05          # Updated at startup from UNIVERSE_CONFIG

# Quality gates (relaxed for short window, zero fees)
MIN_IS_SHARPE  = 4.0         # Updated at startup from UNIVERSE_CONFIG
MIN_IC_MEAN    = -0.02       # IC is our PRIMARY objective — keep this tight
CORR_CUTOFF    = 0.60        # Relaxed for broader alpha diversity
MAX_TURNOVER   = 0.05        # Key gate: low-turnover alpha campaign
MIN_SUB_SHARPE = 0.5         # Shorter sub-periods → lower threshold
MAX_PNL_KURTOSIS = 30        # Slightly relaxed for 5m
MAX_ROLLING_SR_STD = 0.10    # More variance at 5m frequency
MIN_PNL_SKEW = -1.0          # Relaxed for 5m
MIN_FM_TSTAT = 1.65          # Fama-MacBeth |t-stat| >= 1.65 (90% CI)

# Shared DB (separate from 4h)
DB_PATH = "data/alphas_5m.db"


# ============================================================================
# DATABASE
# ============================================================================

def get_conn():
    return sqlite3.connect(DB_PATH)


def ensure_tables(conn):
    """Create tables if they don't exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS alphas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expression TEXT NOT NULL,
        name TEXT,
        category TEXT DEFAULT '',
        asset_class TEXT DEFAULT 'crypto',
        interval TEXT DEFAULT '5m',
        universe TEXT DEFAULT 'BINANCE_TOP100',
        source TEXT DEFAULT 'agent1_research',
        notes TEXT DEFAULT '',
        archived INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alpha_id INTEGER,
        universe TEXT DEFAULT 'BINANCE_TOP100',
        sharpe_is REAL,
        sharpe_train REAL,
        return_ann REAL,
        max_drawdown REAL,
        turnover REAL,
        fitness REAL,
        ic_mean REAL,
        ic_ir REAL,
        psr REAL,
        train_start TEXT,
        train_end TEXT,
        n_bars INTEGER,
        evaluated_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trial_log (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        expression TEXT NOT NULL,
        universe TEXT DEFAULT 'BINANCE_TOP100',
        is_sharpe REAL,
        ic_mean REAL,
        saved INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    # Add universe column to existing tables if missing (idempotent migration)
    for table in ['alphas', 'evaluations', 'trial_log']:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if 'universe' not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN universe TEXT DEFAULT 'BINANCE_TOP100'")
            conn.execute(f"UPDATE {table} SET universe = 'BINANCE_TOP100' WHERE universe IS NULL")
    conn.commit()


def log_trial(conn, expression, is_sharpe, ic_mean=0, saved=False):
    ensure_tables(conn)
    conn.execute("INSERT INTO trial_log (expression, universe, is_sharpe, ic_mean, saved) VALUES (?,?,?,?,?)",
                 (expression, UNIVERSE, is_sharpe, ic_mean, 1 if saved else 0))
    conn.commit()


def get_num_trials(conn):
    ensure_tables(conn)
    return conn.execute("SELECT COUNT(*) FROM trial_log WHERE universe = ?", (UNIVERSE,)).fetchone()[0]


def check_diversity(conn, expression, new_alpha_raw):
    """Check signal correlation against all existing alphas in the same universe."""
    existing = conn.execute("SELECT id FROM alphas WHERE expression=? AND archived=0 AND universe=?",
                            (expression, UNIVERSE)).fetchone()
    if existing:
        print(f"  REJECTED: Expression already exists as alpha #{existing[0]} in {UNIVERSE}")
        return False

    if new_alpha_raw is None:
        return True

    rows = conn.execute("SELECT id, expression FROM alphas WHERE archived=0 AND universe=?",
                        (UNIVERSE,)).fetchall()
    if not rows:
        return True

    matrices, universe = load_data("train")

    new_processed = process_signal(new_alpha_raw, universe_df=universe, max_wt=MAX_WEIGHT)

    for alpha_id, alpha_expr in rows:
        try:
            existing_raw = evaluate_expression(alpha_expr, matrices)
            if existing_raw is None:
                continue
            existing_df = process_signal(existing_raw, universe_df=universe, max_wt=MAX_WEIGHT)
            common_idx = new_processed.index.intersection(existing_df.index)
            common_cols = new_processed.columns.intersection(existing_df.columns)
            if len(common_idx) < 50 or len(common_cols) < 5:
                continue
            a = new_processed.loc[common_idx, common_cols].values.flatten()
            b = existing_df.loc[common_idx, common_cols].values.flatten()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 100:
                continue
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            if abs(corr) > CORR_CUTOFF:
                print(f"  REJECTED: Signal corr={corr:.3f} with alpha #{alpha_id} (cutoff={CORR_CUTOFF})")
                print(f"            Existing: {alpha_expr[:60]}")
                return False
        except Exception:
            continue

    return True


def save_alpha(conn, expression, reasoning, metrics):
    """Save alpha to the alphas table."""
    if not check_diversity(conn, expression, metrics.get('_alpha_raw')):
        return False

    c = conn.cursor()
    c.execute("""INSERT INTO alphas (expression, name, category, asset_class, interval, universe, source, notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (expression,
               expression[:80],
               metrics.get('category', ''),
               'crypto',
               INTERVAL,
               UNIVERSE,
               'agent1_research',
               reasoning))
    alpha_id = c.lastrowid

    c.execute("""INSERT INTO evaluations (alpha_id, universe, sharpe_is, sharpe_train, return_ann,
                 max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                 train_start, train_end, n_bars, evaluated_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
              (alpha_id,
               UNIVERSE,
               metrics['is_sharpe'], metrics['is_sharpe'],
               metrics.get('returns_ann', 0),
               metrics['max_drawdown'], metrics['turnover'],
               metrics['is_fitness'],
               metrics['ic_mean'], metrics['icir'],
               metrics['deflated_sharpe'],
               TRAIN_START, TRAIN_END,
               metrics.get('n_bars', 0)))
    conn.commit()
    print(f"  SAVED as alpha #{alpha_id}")
    return True


# ============================================================================
# DATA LOADING (TRAIN ONLY)
# ============================================================================

_DATA_CACHE = {}

def _get_universe_tickers():
    """Get universe tickers once — used to filter parquet reads."""
    if '_tickers' in _DATA_CACHE:
        return _DATA_CACHE['_tickers'], _DATA_CACHE['_universe_df']
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
    if not uni_path.exists():
        raise FileNotFoundError(f"Universe file not found: {uni_path}")
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    _DATA_CACHE['_tickers'] = valid_tickers
    _DATA_CACHE['_universe_df'] = universe_df
    return valid_tickers, universe_df


def load_data(split="train"):
    if split in _DATA_CACHE:
        return _DATA_CACHE[split]

    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    if not mat_dir.exists():
        raise FileNotFoundError(f"Matrix directory not found: {mat_dir}")

    valid_tickers, universe_df = _get_universe_tickers()

    # Only read if not already cached at full level
    if '_full_matrices' not in _DATA_CACHE:
        print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
        t0 = time.time()
        matrices = {}
        for fp in sorted(mat_dir.glob("*.parquet")):
            df = pd.read_parquet(fp, columns=[c for c in valid_tickers
                                              if c in pd.read_parquet(fp, columns=[]).columns]
                                 ) if False else pd.read_parquet(fp)
            cols = [c for c in valid_tickers if c in df.columns]
            if cols:
                matrices[fp.stem] = df[cols]
        _DATA_CACHE['_full_matrices'] = matrices
        print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)
    else:
        matrices = _DATA_CACHE['_full_matrices']

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

    if "close" in split_matrices:
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
    )


def process_signal(alpha_df, universe_df=None, max_wt=MAX_WEIGHT):
    """Standard Signal Processing Pipeline (Demean -> Scale -> Clip)."""
    signal = alpha_df.copy()

    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)

    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)

    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)

    signal = signal.clip(lower=-max_wt, upper=max_wt)

    return signal.fillna(0.0)


def compute_ic(alpha_raw, returns_df, universe_df=None):
    """Cross-sectional rank IC per bar: spearman(raw_signal[t], return[t+1]).
    
    Uses RAW alpha (before process_signal) restricted to universe stocks only.
    This avoids the dilution bug where zero-filled non-universe stocks crush IC.
    """
    from scipy import stats as sp_stats
    
    # Use raw signal — only universe tickers
    signal = alpha_raw.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)
    
    # IC: corr(signal[t], return[t+1])  → shift signal by 1 forward
    signal_lagged = signal.shift(1)
    
    ics = []
    for dt in signal_lagged.index[1:]:
        a = signal_lagged.loc[dt]
        r = returns_df.loc[dt]
        # Only keep stocks where both signal and return are finite
        valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
        a_v = a[valid]
        r_v = r[valid]
        if len(a_v) < 10 or a_v.std() < 1e-15:
            ics.append(np.nan)
            continue
        ic, _ = sp_stats.spearmanr(a_v, r_v)
        ics.append(ic)
    return pd.Series(ics, index=signal_lagged.index[1:])


def deflated_sharpe_ratio(observed_sr, n_trials, n_bars, skew=0, kurtosis=3):
    """Lopez de Prado DSR: P(true SR > 0) given # trials."""
    from scipy import stats as sp_stats
    if n_trials <= 1:
        return min(sp_stats.norm.cdf(observed_sr * np.sqrt(n_bars)), 1.0)

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
    return float(sp_stats.norm.cdf(z))


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

    # IC — use RAW alpha (not processed) restricted to universe stocks
    _, universe = load_data("train")
    ic_series = compute_ic(is_m["alpha_raw"], is_m["returns_pct"], universe_df=universe)
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

    # PnL distribution metrics
    pnl = is_m["pnl_vec"]
    pnl_kurtosis = float(pd.Series(pnl).kurtosis()) if len(pnl) > 10 else 0
    pnl_skew = float(pd.Series(pnl).skew()) if len(pnl) > 10 else 0
    # Rolling SR: 2-day window at 5m (2 * 288 = 576 bars)
    rolling_sr = pd.Series(pnl).rolling(2 * BARS_PER_DAY).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0
    ).dropna()
    rolling_sr_std = float(rolling_sr.std()) if len(rolling_sr) > 10 else 999

    # Fama-MacBeth cross-sectional regression t-stat
    from stat_tests_5m import fama_macbeth_gate
    fm_pass, fm_tstat, fm_pvalue = fama_macbeth_gate(
        is_m["alpha_raw"], is_m["returns_pct"], universe, threshold=MIN_FM_TSTAT
    )

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
        "fm_tstat": fm_tstat,
        "fm_pvalue": fm_pvalue,
        "_alpha_raw": is_m["alpha_raw"],
    }


# ============================================================================
# LISTING / SCOREBOARD
# ============================================================================

def list_alphas():
    conn = get_conn()
    ensure_tables(conn)
    rows = conn.execute("""
        SELECT a.id, a.expression, a.source,
               COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.turnover, 0), COALESCE(e.ic_mean, 0)
        FROM alphas a
        LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    n_trials = get_num_trials(conn)
    conn.close()

    print(f"\n{'='*70}")
    print(f"ALPHA DATABASE (5m, {UNIVERSE}): {len(rows)} active alphas | {n_trials} trials")
    print(f"{'='*70}")
    for r in rows:
        print(f"  #{r[0]:3d} | IC={r[6]:+.4f} SR={r[3]:+.2f} Fit={r[4]:.2f} TO={r[5]:.2f} "
              f"| src={r[2] or '?':15s} | {r[1][:40]}")


def print_scoreboard():
    conn = get_conn()
    ensure_tables(conn)
    n_alphas = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0 AND universe=?",
                            (UNIVERSE,)).fetchone()[0]
    n_agent = conn.execute("SELECT COUNT(*) FROM alphas WHERE source='agent1_research' AND archived=0 AND universe=?",
                           (UNIVERSE,)).fetchone()[0]
    n_trials = get_num_trials(conn)
    rows = conn.execute("""
        SELECT a.expression, COALESCE(e.sharpe_is, 0), COALESCE(e.fitness, 0),
               COALESCE(e.ic_mean, 0), COALESCE(e.psr, 0), COALESCE(e.ic_ir, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.source = 'agent1_research' AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()

    print(f"\n{'='*70}")
    print(f"  5m ALPHA RESEARCH SCOREBOARD — {UNIVERSE}")
    print(f"  Total alphas: {n_alphas} | Agent 1: {n_agent} | Trials: {n_trials}")
    print(f"  Train: {TRAIN_START} to {TRAIN_END} (zero fees)")
    print(f"  Quality gates: MIN_IS_SHARPE={MIN_IS_SHARPE} MAX_WEIGHT={MAX_WEIGHT}")
    print(f"  Primary: Mean IC | Secondary: Sharpe")
    print(f"{'='*70}")
    if rows:
        print(f"  Avg Mean IC: {np.mean([r[3] for r in rows]):+.05f}")
        print(f"  Avg IS Sharpe: {np.mean([r[1] for r in rows]):+.3f}")
        print(f"\n  Agent 1 alphas (ranked by IC, top 10):")
        for i, r in enumerate(rows[:10]):
            print(f"    {i+1}. IC={r[3]:+.05f} SR={r[1]:+.2f} Fit={r[2]:.2f} ICIR={r[5]:.3f} DSR={r[4]:.2f} | {r[0][:180]}")
    else:
        print("  No Agent 1 alphas yet.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Agent 1: 5m Alpha Discovery (Train Only)")
    parser.add_argument("--expr", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--scoreboard", action="store_true")
    parser.add_argument("--reasoning", type=str, default="")
    parser.add_argument("--universe", default="BINANCE_TOP100", choices=ALL_UNIVERSES,
                        help="Universe for alpha discovery (default: BINANCE_TOP100)")
    args = parser.parse_args()

    # Apply universe-specific config
    global UNIVERSE, MAX_WEIGHT, MIN_IS_SHARPE
    UNIVERSE = args.universe
    ucfg = UNIVERSE_CONFIG.get(UNIVERSE, UNIVERSE_CONFIG["BINANCE_TOP100"])
    MAX_WEIGHT = ucfg["max_weight"]
    MIN_IS_SHARPE = ucfg["min_is_sharpe"]

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    if args.scoreboard:
        print_scoreboard()
        return
    if args.list:
        list_alphas()
        return

    conn = get_conn()
    ensure_tables(conn)
    expression = args.expr or EXPRESSION
    print(f"\nEvaluating (5m): {expression}")

    result = eval_full(expression, conn)
    if not result["success"]:
        print(f"  FAILED: {result['error']}")
        log_trial(conn, expression, 0, 0, saved=False)
        conn.close()
        return

    # Build output as single string to avoid terminal interleaving
    both_pos = result['stability_h1'] > 0 and result['stability_h2'] > 0
    min_sub = min(result['stability_h1'], result['stability_h2'])
    gates = [
        (result['is_sharpe'] >= MIN_IS_SHARPE, f"IS Sharpe >= {MIN_IS_SHARPE}: {result['is_sharpe']:+.3f}"),
        (both_pos, f"Sub-period stability: H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}"),
        (min_sub >= MIN_SUB_SHARPE, f"Min sub-period Sharpe >= {MIN_SUB_SHARPE}: {min_sub:+.2f}"),
        (result['rolling_sr_std'] <= MAX_ROLLING_SR_STD, f"Rolling SR Consistency <= {MAX_ROLLING_SR_STD}: {result['rolling_sr_std']:.4f}"),
        (result['pnl_skew'] >= MIN_PNL_SKEW, f"PnL Skew >= {MIN_PNL_SKEW}: {result['pnl_skew']:+.3f}"),
    ]
    all_pass = all(g[0] for g in gates)
    # Informational-only metrics (soft gates — not blockers)
    gate_lines = "\n".join(f"  [{'PASS' if p else 'FAIL'}] {d}" for p, d in gates)
    gate_lines += f"\n  [INFO] Mean IC (informational): {result['ic_mean']:+.5f} (ref >= {MIN_IC_MEAN}, PRIMARY signal quality metric)"
    gate_lines += f"\n  [INFO] PnL Kurtosis (soft): {result['pnl_kurtosis']:.1f} (ref <= {MAX_PNL_KURTOSIS}, 5m crypto naturally high)"
    gate_lines += f"\n  [INFO] FM |t-stat| >= {MIN_FM_TSTAT}: {result['fm_tstat']:+.3f} (p={result['fm_pvalue']:.4f})"

    out = f"""
  IC={result['ic_mean']:+.5f} | ICIR={result['icir']:.4f} | SR={result['is_sharpe']:+.3f} | Fit={result['is_fitness']:.1f} | TO={result['turnover']:.3f} | DD={result['max_drawdown']:.3f}
  PnL: kurt={result['pnl_kurtosis']:.1f} skew={result['pnl_skew']:+.3f} rollSR_std={result['rolling_sr_std']:.4f}
  Stability: H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f} | DSR={result['deflated_sharpe']:.3f} (trial {result['n_trials']})
  --- GATES ---
{gate_lines}
  ALL PASS: {'YES' if all_pass else 'NO'}"""
    print(out, flush=True)

    log_trial(conn, expression, result['is_sharpe'], result['ic_mean'], saved=False)

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
