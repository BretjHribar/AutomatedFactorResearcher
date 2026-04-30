"""
PROPER OOS evaluation — vectorized sim, no look-ahead bias.

Approach:
1. Select ALL alphas with IS Sharpe >= 1.0 AND IS Fitness >= 1.0 (training criteria only)
2. Combine ALL qualifying alphas via rank-normalization + equal-weight average
3. Run vectorized sim on TRAIN, OOS, and FULL — same sim the GP trained against
4. Also run with delay=0 for comparison
5. Handle delistings

NO cherry-picking OOS winners. The combined signal is fixed before seeing OOS.
"""
import sys, os, sqlite3, time, math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, '.')

from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

TRAIN_END = "2024-04-27"
DATA_DIR = Path("data/binance_cache")
DB_PATH = "data/alpha_gp_crypto.db"
RESULTS_FILE = "crypto_oos_pipeline.txt"

BOOKSIZE = 2_000_000.0
MAX_WEIGHT = 0.05
FEES = 5.0  # bps one-way

out = open(RESULTS_FILE, 'w', encoding='ascii', errors='replace')
def log(msg):
    print(msg, flush=True)
    out.write(msg + '\n')
    out.flush()

log("=" * 90)
log("CRYPTO OOS — PROPER EVALUATION (NO LOOK-AHEAD BIAS)")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 90)
log(f"\nRules:")
log(f"  1. Alpha selection: IS Sharpe >= 1.0 AND IS Fitness >= 1.0 (training-set only)")
log(f"  2. ALL qualifying alphas combined (rank-norm + equal-weight)")  
log(f"  3. No cherry-picking — signal fixed BEFORE seeing OOS")
log(f"  4. Both delay=1 and delay=0 compared")
log(f"  5. Delistings handled (zeroed out)")
log(f"  6. Fees: {FEES} bps | Book: ${BOOKSIZE:,.0f} | Max wt: {MAX_WEIGHT:.0%}")

# ── Load data ──
t0 = time.time()
log("\n[1/5] Loading data...")

matrices = {}
for f in sorted((DATA_DIR / "matrices").glob("*.parquet")):
    matrices[f.stem] = pd.read_parquet(f)
log(f"  {len(matrices)} matrices loaded")

universe_df = pd.read_parquet(DATA_DIR / "universes/BINANCE_TOP50.parquet")

# Handle delistings
close = matrices['close'].copy()
returns = matrices['returns'].copy()
n_delisted = 0
for col in close.columns:
    last_valid = close[col].last_valid_index()
    if last_valid is not None and last_valid < close.index[-1]:
        returns.loc[returns.index > last_valid, col] = 0.0
        n_delisted += 1
matrices['returns'] = returns
matrices['close'] = close
log(f"  {n_delisted} delisted coins handled (returns zeroed after delist)")

# Get tickers
all_tickers = sorted(universe_df.columns[universe_df.any()].tolist())
log(f"  {len(all_tickers)} tickers ever in universe")

for name in list(matrices.keys()):
    cols = [c for c in all_tickers if c in matrices[name].columns]
    if cols:
        matrices[name] = matrices[name][cols]
    else:
        del matrices[name]

# Split
train_matrices = {n: m.loc[:TRAIN_END].copy() for n, m in matrices.items()}
test_matrices = {n: m.loc[TRAIN_END:].copy() for n, m in matrices.items()}
train_uni = universe_df.loc[:TRAIN_END]
test_uni = universe_df.loc[TRAIN_END:]

log(f"  Train: {train_matrices['returns'].shape[0]} days "
    f"({train_matrices['returns'].index[0].date()} to {train_matrices['returns'].index[-1].date()})")
log(f"  Test:  {test_matrices['returns'].shape[0]} days "
    f"({test_matrices['returns'].index[0].date()} to {test_matrices['returns'].index[-1].date()})")

TERMINALS = [t for t in [
    "close", "open", "high", "low", "volume", "returns", "log_returns",
    "taker_buy_ratio", "taker_buy_volume", "vwap", "vwap_deviation",
    "high_low_range", "open_close_range", "adv20", "adv60",
    "volume_ratio_20d", "historical_volatility_20", "historical_volatility_60",
    "momentum_5d", "momentum_20d", "momentum_60d", "beta_to_btc",
    "overnight_gap", "upper_shadow", "lower_shadow",
    "close_position_in_range", "trades_count", "trades_per_volume",
    "parkinson_volatility_20", "quote_volume",
] if t in matrices]

log(f"  {len(TERMINALS)} terminals | Load time: {time.time()-t0:.0f}s")

# ── Get qualifying alphas (IS criteria ONLY) ──
log("\n[2/5] Selecting alphas by TRAINING-SET criteria...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""SELECT a.expression, e.sharpe, e.fitness, e.turnover
               FROM alphas a JOIN evaluations e ON a.alpha_id=e.alpha_id
               WHERE e.sharpe >= 1.0 AND e.fitness >= 1.0
               ORDER BY e.fitness DESC""")
qualifying = cur.fetchall()
conn.close()
expressions = [row[0] for row in qualifying]
log(f"  {len(qualifying)} alphas with IS Sharpe >= 1.0 AND IS Fitness >= 1.0")

# ── Build combined signals ──
log("\n[3/5] Building combined signal from ALL qualifying alphas...")

def build_combined(engine, expressions, label=""):
    """Combine all expressions by rank-normalization + equal-weight average."""
    combined = None
    n_good = 0
    for expr in expressions:
        try:
            a = engine.evaluate(expr)
            if a is not None and not a.empty:
                ranked = a.rank(axis=1, pct=True) - 0.5
                if combined is None:
                    combined = ranked.copy()
                else:
                    combined = combined.add(ranked, fill_value=0)
                n_good += 1
        except:
            pass
    if combined is not None and n_good > 0:
        combined = combined / n_good
    log(f"  {label}: {n_good}/{len(expressions)} alphas combined")
    return combined

# Build for each period
train_engine = FastExpressionEngine(data_fields={n: train_matrices[n] for n in TERMINALS if n in train_matrices})
test_engine = FastExpressionEngine(data_fields={n: test_matrices[n] for n in TERMINALS if n in test_matrices})
full_engine = FastExpressionEngine(data_fields={n: matrices[n] for n in TERMINALS if n in matrices})

train_signal = build_combined(train_engine, expressions, "TRAIN")
test_signal = build_combined(test_engine, expressions, "OOS")
full_signal = build_combined(full_engine, expressions, "FULL")

# ── Evaluate with delay=1 (what GP trained on) ──
log("\n[4/5] Evaluating combined signal...")
log(f"\n{'='*90}")
log(f"  DELAY=1 (signal at close T -> trade at open T+1) -- REALISTIC")
log(f"{'='*90}")

log(f"\n  {'Period':>5} | {'Sharpe':>7} | {'PnL':>12} | {'Ann Ret':>9} | {'Turnover':>9} | {'MaxDD':>7}")
log("  " + "-" * 70)

for period_name, signal, returns, close, open_df, uni in [
    ("TRAIN", train_signal, train_matrices['returns'], train_matrices.get('close'), train_matrices.get('open'), train_uni),
    ("OOS", test_signal, test_matrices['returns'], test_matrices.get('close'), test_matrices.get('open'), test_uni),
    ("FULL", full_signal, matrices['returns'], matrices.get('close'), matrices.get('open'), universe_df),
]:
    if signal is None:
        continue
    r = sim_vec(
        alpha_df=signal, returns_df=returns, close_df=close, open_df=open_df,
        universe_df=uni, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
        decay=0, delay=1, neutralization="market", fees_bps=FEES,
    )
    ann_ret = r.total_pnl / (BOOKSIZE * 0.5) / max(1, len(returns) / 365) * 100
    log(f"  {period_name:>5} | {r.sharpe:+7.2f} | ${r.total_pnl:>11,.0f} | {ann_ret:>8.1f}% | "
        f"{r.turnover:>8.1%} | {r.max_drawdown:>6.1%}")

# ── Evaluate with delay=0 (upper bound) ──
log(f"\n{'='*90}")
log(f"  DELAY=0 (signal at close T -> trade at close T) -- UPPER BOUND")
log(f"{'='*90}")

log(f"\n  {'Period':>5} | {'Sharpe':>7} | {'PnL':>12} | {'Ann Ret':>9} | {'Turnover':>9} | {'MaxDD':>7}")
log("  " + "-" * 70)

for period_name, signal, returns, close, open_df, uni in [
    ("TRAIN", train_signal, train_matrices['returns'], train_matrices.get('close'), train_matrices.get('open'), train_uni),
    ("OOS", test_signal, test_matrices['returns'], test_matrices.get('close'), test_matrices.get('open'), test_uni),
    ("FULL", full_signal, matrices['returns'], matrices.get('close'), matrices.get('open'), universe_df),
]:
    if signal is None:
        continue
    r = sim_vec(
        alpha_df=signal, returns_df=returns, close_df=close, open_df=open_df,
        universe_df=uni, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
        decay=0, delay=0, neutralization="market", fees_bps=FEES,
    )
    ann_ret = r.total_pnl / (BOOKSIZE * 0.5) / max(1, len(returns) / 365) * 100
    log(f"  {period_name:>5} | {r.sharpe:+7.2f} | ${r.total_pnl:>11,.0f} | {ann_ret:>8.1f}% | "
        f"{r.turnover:>8.1%} | {r.max_drawdown:>6.1%}")

# ── Individual alpha OOS diagnostics ──
log(f"\n{'='*90}")
log("[5/5] Individual alpha OOS diagnostics (delay=1)")
log(f"{'='*90}")

oos_sharpes = []
for expr in expressions:
    try:
        a = test_engine.evaluate(expr)
        if a is not None and not a.empty:
            r = sim_vec(
                alpha_df=a, returns_df=test_matrices['returns'],
                close_df=test_matrices.get('close'), open_df=test_matrices.get('open'),
                universe_df=test_uni, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                decay=0, delay=1, neutralization="market", fees_bps=FEES,
            )
            oos_sharpes.append(r.sharpe)
        else:
            oos_sharpes.append(np.nan)
    except:
        oos_sharpes.append(np.nan)

valid_oos = [s for s in oos_sharpes if not np.isnan(s)]
n_pos = sum(1 for s in valid_oos if s > 0)
n_05 = sum(1 for s in valid_oos if s > 0.5)
n_10 = sum(1 for s in valid_oos if s > 1.0)

log(f"\n  Individual alpha OOS survival ({len(valid_oos)} evaluated):")
log(f"    OOS Sharpe > 0:   {n_pos:3d} ({100*n_pos/max(len(valid_oos),1):.0f}%)")
log(f"    OOS Sharpe > 0.5: {n_05:3d} ({100*n_05/max(len(valid_oos),1):.0f}%)")
log(f"    OOS Sharpe > 1.0: {n_10:3d} ({100*n_10/max(len(valid_oos),1):.0f}%)")
log(f"    Mean:  {np.nanmean(valid_oos):+.2f}")
log(f"    Median:{np.nanmedian(valid_oos):+.2f}")

from scipy.stats import spearmanr
is_s = [qualifying[i][1] for i in range(len(qualifying)) if i < len(oos_sharpes) and not np.isnan(oos_sharpes[i])]
oos_s = [oos_sharpes[i] for i in range(len(qualifying)) if i < len(oos_sharpes) and not np.isnan(oos_sharpes[i])]
if len(is_s) > 5:
    rho, _ = spearmanr(is_s, oos_s)
    log(f"    Spearman(IS, OOS): {rho:+.3f}")

log(f"\nCompleted: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Total runtime: {time.time()-t0:.0f}s")
out.close()
print(f"\n✅ Results saved to {RESULTS_FILE}", flush=True)
