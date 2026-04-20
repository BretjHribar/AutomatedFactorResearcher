"""
run_factor_max.py -- Factor MAX Strategy (Bali, Cakici, Whitelaw style)

Implements the Factor MAX signal combination from:
  "Factor MAX" - factors with the highest maximum daily return in the
  prior month continue to outperform (underreaction to factor-level news).

Adapted to our 5m crypto alpha context:
  - Each "factor" = one of our 19 alpha signals
  - MAX = largest single-bar PnL of each alpha in the prior lookback window
  - Weight alphas proportional to their MAX rank
  - Compare with existing combiners (Billion Alphas, PCA, Return-Weighted)

The paper shows this is distinct from factor momentum and captures
attention-driven underreaction to extreme factor-level events.
"""

import sys, os, time, sqlite3
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_alpha_5m as ea
import eval_portfolio_5m as ep

# ============================================================================
# CONFIGURATION
# ============================================================================

FEES_BPS     = 0.0
UNIVERSE     = "BINANCE_TOP50"
INTERVAL     = "5m"
BOOKSIZE     = ea.BOOKSIZE
BARS_PER_DAY = ea.BARS_PER_DAY
NEUTRALIZE   = ea.NEUTRALIZE
DB_PATH      = ea.DB_PATH

ea.UNIVERSE = UNIVERSE
ucfg = ea.UNIVERSE_CONFIG.get(UNIVERSE, ea.UNIVERSE_CONFIG["BINANCE_TOP100"])
MAX_WEIGHT = ucfg["max_weight"]
ea.MAX_WEIGHT = MAX_WEIGHT

SPLITS = {
    "train": ("2025-02-01", "2026-02-01"),
    "val":   ("2026-02-01", "2026-03-01"),
    "test":  ("2026-03-01", "2026-03-27"),
}

# Factor MAX params
LOOKBACK_BARS = 288 * 5   # 5 days (paper uses 1 month; we use 5d for 5m crypto)
REWEIGHT_EVERY = 288      # Recompute MAX weights every day (paper: monthly)
SIGNAL_SMOOTH = 12        # 1-hour EMA on final composite
GAMMA = 0.01              # ts_autonorm gamma


# ============================================================================
# FACTOR MAX COMBINER
# ============================================================================

def compute_composite_factor_max(
    alphas,           # list of (id, expression, ic_mean, sharpe_is)
    matrices,         # dict of {field: DataFrame}
    universe_df,      # DataFrame of universe membership
    returns_df,       # DataFrame of returns
    lookback_bars=288*5,
    reweight_every=288,
    gamma=0.01,
    signal_smooth=12,
):
    """Factor MAX signal combination.
    
    Algorithm:
      1. Evaluate each alpha expression -> per-bar, per-ticker signal
      2. Compute each alpha's PnL series (simple signal * forward return)
      3. Rolling window: compute MAX = max single-bar PnL for each alpha
      4. Rank alphas by MAX -> weight proportional to rank
      5. Combine alpha signals using Factor MAX weights
    
    This captures: factors with recent extreme positive performance
    tend to continue performing well (underreaction).
    """
    n_alphas = len(alphas)
    if n_alphas == 0:
        return None
    
    print(f"  Factor MAX: computing per-alpha signals ({n_alphas} alphas)...", flush=True)
    
    # Step 1: Evaluate all alpha signals
    alpha_signals = {}  # {alpha_id: DataFrame}
    for aid, expr, ic_mean, sharpe_is in alphas:
        try:
            sig = ea.evaluate_expression(expr, matrices)
            if sig is not None and not sig.empty:
                alpha_signals[aid] = sig
        except Exception as e:
            continue
    
    print(f"  Factor MAX: {len(alpha_signals)}/{n_alphas} alphas loaded", flush=True)
    
    if len(alpha_signals) < 2:
        print("  Factor MAX: need at least 2 alphas")
        return None
    
    # Step 2: Compute per-alpha L/S PnL at each bar
    # PnL_i(t) = sum_j[ signal_i(j,t) * return(j,t+1) ]  (cross-sectional)
    all_dates = returns_df.index
    tickers = returns_df.columns.tolist()
    T = len(all_dates)
    
    alpha_ids = sorted(alpha_signals.keys())
    n_active = len(alpha_ids)
    
    # Pre-compute per-alpha bar-level returns (signal * forward return)
    alpha_bar_pnl = {}  # {alpha_id: Series of bar-level PnL}
    for aid in alpha_ids:
        sig = alpha_signals[aid].reindex(index=all_dates, columns=tickers).fillna(0.0)
        # Cross-sectional z-score
        sig_vals = sig.values
        mu = np.nanmean(sig_vals, axis=1, keepdims=True)
        sigma = np.nanstd(sig_vals, axis=1, keepdims=True)
        sigma = np.where(sigma > 1e-10, sigma, 1.0)
        sig_normed = (sig_vals - mu) / sigma
        sig_normed = np.nan_to_num(sig_normed, nan=0.0)
        
        # Dollar-neutral: demean
        sig_normed = sig_normed - np.mean(sig_normed, axis=1, keepdims=True)
        
        # PnL = signal(t) * return(t+1)
        ret_vals = returns_df.reindex(index=all_dates, columns=tickers).fillna(0.0).values
        # Use contemporaneous for simplicity (signal * same-bar return is the IC proxy)
        # Actually, use shifted: signal at t predicts return at t+1
        bar_pnl = np.sum(sig_normed[:-1] * ret_vals[1:], axis=1)
        bar_pnl = np.concatenate([[0.0], bar_pnl])  # Pad first bar
        alpha_bar_pnl[aid] = pd.Series(bar_pnl, index=all_dates)
    
    # Step 3 & 4: Rolling MAX and rank-based weights
    print(f"  Factor MAX: computing rolling MAX (lookback={lookback_bars} bars, "
          f"reweight every {reweight_every} bars)...", flush=True)
    
    # Build weight matrix: (T, n_active) -> weight for each alpha at each bar
    weight_matrix = np.zeros((T, n_active))
    current_weights = np.ones(n_active) / n_active  # Equal weight initially
    
    for t in range(T):
        if t % reweight_every == 0 and t >= lookback_bars:
            # Compute MAX for each alpha over [t-lookback : t]
            max_vals = np.zeros(n_active)
            for i, aid in enumerate(alpha_ids):
                pnl_window = alpha_bar_pnl[aid].iloc[t-lookback_bars:t].values
                if len(pnl_window) > 0:
                    max_vals[i] = np.max(pnl_window)  # MAX = largest single-bar PnL
                else:
                    max_vals[i] = 0.0
            
            # Rank: higher MAX -> higher weight (cross-sectional rank)
            ranks = np.argsort(np.argsort(max_vals)).astype(float)  # 0 to n_active-1
            # Center ranks: subtract mean so weights sum to ~0 influence
            # But we want long high-MAX, short low-MAX
            # Weight = (rank - mean_rank) / sum(|rank - mean_rank|)
            mean_rank = np.mean(ranks)
            w_raw = ranks - mean_rank
            w_sum = np.sum(np.abs(w_raw))
            if w_sum > 0:
                current_weights = w_raw / w_sum
            else:
                current_weights = np.ones(n_active) / n_active
        
        weight_matrix[t] = current_weights
    
    # Step 5: Combine alpha signals using Factor MAX weights
    print(f"  Factor MAX: assembling composite signal...", flush=True)
    
    composite = np.zeros((T, len(tickers)))
    
    for i, aid in enumerate(alpha_ids):
        sig = alpha_signals[aid].reindex(index=all_dates, columns=tickers).fillna(0.0)
        sig_vals = sig.values
        
        # Per-alpha normalization: EWM z-score (same as ts_autonorm)
        ewm_mean = np.zeros_like(sig_vals)
        ewm_var = np.zeros_like(sig_vals)
        alpha_ema = gamma
        
        for t in range(T):
            if t == 0:
                ewm_mean[t] = sig_vals[t]
                ewm_var[t] = np.zeros(len(tickers))
            else:
                ewm_mean[t] = alpha_ema * sig_vals[t] + (1 - alpha_ema) * ewm_mean[t-1]
                diff = sig_vals[t] - ewm_mean[t]
                ewm_var[t] = alpha_ema * diff**2 + (1 - alpha_ema) * ewm_var[t-1]
            
            std = np.sqrt(ewm_var[t] + 1e-10)
            normed = (sig_vals[t] - ewm_mean[t]) / std
            normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            normed = np.clip(normed, -5, 5)
            
            composite[t] += weight_matrix[t, i] * normed
    
    # Signal smoothing
    composite_df = pd.DataFrame(composite, index=all_dates, columns=tickers)
    if signal_smooth > 0:
        composite_df = composite_df.ewm(span=signal_smooth, min_periods=1).mean()
    
    # Print weight summary
    final_weights = weight_matrix[-1]
    top_idx = np.argsort(np.abs(final_weights))[::-1][:5]
    print(f"  Factor MAX: final weights (top 5 by |w|):")
    for idx in top_idx:
        aid = alpha_ids[idx]
        print(f"    alpha #{aid}: w={final_weights[idx]:+.4f}")
    
    n_pos = np.sum(final_weights > 0.01)
    n_neg = np.sum(final_weights < -0.01)
    print(f"  Factor MAX: pos={n_pos} neg={n_neg} |w| sum={np.sum(np.abs(final_weights)):.4f}")
    
    return composite_df


# ============================================================================
# DATA LOADING
# ============================================================================

def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.universe = ?
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """, (UNIVERSE,)).fetchall()
    conn.close()
    return rows


def load_full_data():
    mat_dir = Path(f"data/binance_cache/matrices/{INTERVAL}")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > ea.COVERAGE_CUTOFF].index.tolist())
    
    print(f"  Loading 5m matrices ({len(valid_tickers)} tickers)...", flush=True)
    t0 = time.time()
    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s", flush=True)
    
    ea._DATA_CACHE['_full_matrices'] = matrices
    ea._DATA_CACHE['_tickers'] = valid_tickers
    ea._DATA_CACHE['_universe_df'] = universe_df
    
    return matrices, universe_df, valid_tickers


def slice_data(matrices, universe_df, valid_tickers, split):
    start, end = SPLITS[split]
    split_matrices = {name: df.loc[start:end] for name, df in matrices.items()}
    split_universe = universe_df[valid_tickers].loc[start:end]
    if "close" in split_matrices:
        split_matrices["returns"] = split_matrices["close"].pct_change()
    return split_matrices, split_universe


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0_total = time.time()
    
    alphas = load_alphas()
    print(f"\n{'='*80}")
    print(f"  FACTOR MAX COMPARISON")
    print(f"  Universe: {UNIVERSE} | Fees: {FEES_BPS} bps | Alphas: {len(alphas)}")
    print(f"  MAX lookback: {LOOKBACK_BARS} bars ({LOOKBACK_BARS/288:.0f}d)")
    print(f"  Reweight every: {REWEIGHT_EVERY} bars ({REWEIGHT_EVERY/288:.1f}d)")
    print(f"{'='*80}")
    
    if not alphas:
        print("  No alphas found!")
        return
    
    full_matrices, universe_df, valid_tickers = load_full_data()
    
    full_start, full_end = "2025-02-01", "2026-03-27"
    full_mat = {name: df.loc[full_start:full_end] for name, df in full_matrices.items()}
    full_uni = universe_df[valid_tickers].loc[full_start:full_end]
    if "close" in full_mat:
        full_mat["returns"] = full_mat["close"].pct_change()
    full_returns = full_mat.get("returns")
    
    # Build composites
    COMBINERS = {
        "Billion Alphas": {
            "func": "billion_alphas",
            "kwargs": {
                "lookback_days": 15, "retrain_every": 288,
                "gamma": 0.01, "signal_smooth": 12,
            },
        },
        "Factor MAX (5d)": {
            "func": "factor_max",
            "kwargs": {
                "lookback_bars": LOOKBACK_BARS,
                "reweight_every": REWEIGHT_EVERY,
                "gamma": GAMMA,
                "signal_smooth": SIGNAL_SMOOTH,
            },
        },
        "Factor MAX (10d)": {
            "func": "factor_max",
            "kwargs": {
                "lookback_bars": 288 * 10,
                "reweight_every": REWEIGHT_EVERY,
                "gamma": GAMMA,
                "signal_smooth": SIGNAL_SMOOTH,
            },
        },
        "Factor MAX (20d)": {
            "func": "factor_max",
            "kwargs": {
                "lookback_bars": 288 * 20,
                "reweight_every": REWEIGHT_EVERY,
                "gamma": GAMMA,
                "signal_smooth": SIGNAL_SMOOTH,
            },
        },
    }
    
    composites = {}
    for cname, cfg in COMBINERS.items():
        print(f"\n  Building composite: {cname}...", flush=True)
        func_name = cfg["func"]
        kwargs = cfg["kwargs"]
        
        if func_name == "billion_alphas":
            composite = ep.compute_composite_billion_alphas(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        elif func_name == "factor_max":
            composite = compute_composite_factor_max(
                alphas, full_mat, full_uni, full_returns, **kwargs
            )
        else:
            raise ValueError(f"Unknown combiner: {func_name}")
        
        if composite is not None:
            composites[cname] = composite
    
    if not composites:
        print("  No composites built!")
        return
    
    # Run simulations
    all_results = {}
    
    for cname, composite in composites.items():
        all_results[cname] = {}
        for split_name, (start, end) in SPLITS.items():
            comp_slice = composite.loc[start:end]
            split_mat, split_uni = slice_data(full_matrices, universe_df, valid_tickers, split_name)
            close = split_mat.get("close")
            returns = split_mat.get("returns")
            alpha_processed = ea.process_signal(comp_slice, universe_df=split_uni, max_wt=MAX_WEIGHT)
            sim = ea.simulate(
                alpha_df=alpha_processed, returns_df=returns,
                close_df=close, universe_df=split_uni, fees_bps=FEES_BPS,
            )
            all_results[cname][split_name] = sim
    
    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  FACTOR MAX COMPARISON RESULTS")
    print(f"  ({len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps)")
    print(f"{'='*90}")
    
    for cname in composites:
        print(f"\n  -- {cname} --")
        print(f"  {'Split':<8} {'Sharpe':>8} {'Ret%':>9} {'RetAnn%':>10} {'MaxDD%':>8} {'TO':>8} {'Fitness':>8}")
        print(f"  {'-'*70}")
        for split_name in ["train", "val", "test"]:
            s = all_results[cname][split_name]
            ret_pct = s.total_pnl / BOOKSIZE * 100
            print(f"  {split_name:<8} {s.sharpe:+8.3f} {ret_pct:+8.2f}% "
                  f"{s.returns_ann*100:+9.2f}% {s.max_drawdown*100:7.2f}% "
                  f"{s.turnover:7.4f} {s.fitness:7.2f}")
    
    # Chart
    print(f"\n  Generating charts...", flush=True)
    
    colors = {
        "Billion Alphas":    "#4CAF50",
        "Factor MAX (5d)":   "#E91E63",
        "Factor MAX (10d)":  "#9C27B0",
        "Factor MAX (20d)":  "#FF5722",
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(20, 14),
                              gridspec_kw={'height_ratios': [3, 1.5]})
    
    ax1 = axes[0]
    
    # Split backgrounds
    ref_label = list(composites.keys())[0]
    offset = 0
    for split_name in ["train", "val", "test"]:
        n = len(all_results[ref_label][split_name].daily_pnl)
        ax1.axvspan(offset, offset + n, alpha=0.06,
                   color=['#2196F3', '#FF9800', '#F44336'][["train","val","test"].index(split_name)])
        if split_name != "train":
            ax1.axvline(x=offset, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        offset += n
    
    for cname in composites:
        running_offset = 0.0
        bar_off = 0
        all_x, all_y = [], []
        
        for split_name in ["train", "val", "test"]:
            s = all_results[cname][split_name]
            cum = s.cumulative_pnl.values / BOOKSIZE * 100 + running_offset
            n = len(cum)
            x = np.arange(bar_off, bar_off + n)
            all_x.extend(x)
            all_y.extend(cum)
            running_offset = cum[-1]
            bar_off += n
        
        color = colors.get(cname, '#999999')
        sr_train = all_results[cname]["train"].sharpe
        sr_val   = all_results[cname]["val"].sharpe
        sr_test  = all_results[cname]["test"].sharpe
        lw = 2.5 if "Billion" in cname else 2.0
        ax1.plot(all_x, all_y, color=color, linewidth=lw,
                label=f"{cname}  (Train={sr_train:+.2f} | Val={sr_val:+.2f} | Test={sr_test:+.2f})")
    
    ax1.set_title(f"Factor MAX vs Billion Alphas -- {len(alphas)} alphas, {UNIVERSE}, {FEES_BPS:.0f} bps\n"
                  f"Factor MAX: sort alphas by max single-bar PnL, go long high-MAX / short low-MAX",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.25)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    # Bar chart
    ax2 = axes[1]
    comb_names = list(composites.keys())
    n_c = len(comb_names)
    x_pos = np.arange(3)
    bar_width = 0.8 / n_c
    
    for i, cname in enumerate(comb_names):
        sharpes = [all_results[cname][s].sharpe for s in ["train", "val", "test"]]
        color = colors.get(cname, '#999999')
        bars = ax2.bar(x_pos + i * bar_width - 0.4 + bar_width/2, sharpes, bar_width,
                      label=cname, color=color, alpha=0.8, edgecolor='white')
        for j, v in enumerate(sharpes):
            ax2.text(x_pos[j] + i * bar_width - 0.4 + bar_width/2,
                    v + (0.15 if v >= 0 else -0.35),
                    f"{v:+.2f}", ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["TRAIN", "VAL", "TEST"], fontsize=12, fontweight='bold')
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.set_title("Sharpe by Split x Combiner", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "factor_max_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved: {chart_path}")
    
    elapsed = time.time() - t0_total
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
