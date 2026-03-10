#!/usr/bin/env python3
"""
Walk-Forward Crypto Pipeline — runs from the unified AlphaDB.

Loads alpha expressions from the database, runs the full walk-forward
backtest, and writes evaluation results + selection history back.

Usage:
    python run_walkforward_db.py --days 480 --fee-bps 5 --top-n 50
    python run_walkforward_db.py --category momentum --days 480
"""

import argparse
import json
import math
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

# ── Local imports ────────────────────────────────────────────────
sys.path.insert(0, ".")
from src.data.alpha_db import AlphaDB
from src.operators.crypto_ops import (
    evaluate_expression, sma, rank, stddev, ts_min, ts_max, ts_sum,
    ts_zscore, correlation, sign, abs, div, mul, delta, delay,
    CRYPTO_ALPHA_DEFINITIONS,
)

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

ANN_FACTOR = math.sqrt(6 * 252)


# ═════════════════════════════════════════════════════════════════
# DATA LOADING (from verified cached matrices)
# ═════════════════════════════════════════════════════════════════

from pathlib import Path
DATA_DIR = Path("data/binance_cache")

def load_data(top_n: int = 100, days: int = 480, interval: str = "4h"):
    """Load pre-built 4h matrices from the verified cache."""
    matrices_dir = DATA_DIR / "matrices" / interval
    universe_path = DATA_DIR / f"universes/BINANCE_TOP{top_n}_{interval}.parquet"
    
    if not universe_path.exists():
        # Fall back to TOP100 if exact universe not available
        universe_path = DATA_DIR / f"universes/BINANCE_TOP100_{interval}.parquet"
        print(f"  Using TOP100 universe (TOP{top_n} not found)")
    
    # Load all matrices
    matrices = {}
    for fpath in sorted(matrices_dir.glob("*.parquet")):
        matrices[fpath.stem] = pd.read_parquet(fpath)
    print(f"  {len(matrices)} matrices loaded from {matrices_dir}")
    
    # Load universe and get valid tickers
    universe_df = pd.read_parquet(universe_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
    print(f"  {len(valid_tickers)} tickers with >10% universe coverage")
    
    # Filter to valid tickers
    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]
    
    # Build features dict (compatible with evaluate_expression)
    features = {}
    for name in ["open", "high", "low", "close", "volume", "quote_volume",
                  "returns", "adv20"]:
        if name in matrices:
            features[name] = matrices[name]
    
    # Compute returns/adv20 if not pre-built
    if "returns" not in features and "close" in features:
        features["returns"] = features["close"].pct_change()
    if "adv20" not in features and "volume" in features:
        features["adv20"] = sma(features["volume"], 30)
    
    idx = features["close"].index
    valid_syms = features["close"].columns.tolist()
    
    return features, idx, valid_syms


# ═════════════════════════════════════════════════════════════════
# ALPHA SELECTION + QP COMBINATION
# ═════════════════════════════════════════════════════════════════

def compute_period_sharpe(signal, returns, start, end):
    if end - start < 60:
        return None
    sig = signal.iloc[start:end]
    ret = returns.iloc[start:end]
    sig_n = sig.sub(sig.mean(axis=1), axis=0)
    sig_n = sig_n.div(sig_n.abs().sum(axis=1) + 1e-10, axis=0)
    pnl = (sig_n.shift(1) * ret).sum(axis=1).dropna()
    if len(pnl) < 30 or pnl.std() < 1e-12:
        return None
    return float((pnl.mean() / pnl.std()) * ANN_FACTOR)


def compute_signal_correlation(s1, s2, start, end):
    correlations = []
    for d in range(max(start + 30, end - 50), end):
        if d >= len(s1):
            continue
        v1 = s1.iloc[d].dropna()
        v2 = s2.iloc[d].dropna()
        common = v1.index.intersection(v2.index)
        if len(common) >= 10:
            c = v1[common].corr(v2[common])
            if not np.isnan(c):
                correlations.append(builtins_abs(c))
    return np.mean(correlations) if correlations else 1.0

# Keep Python's abs separate from our operator abs
import builtins
builtins_abs = builtins.abs


def select_alphas(alpha_signals, returns, t, train_bars=720, val_bars=360,
                  max_corr=0.65, max_alphas=12):
    train_start = t - train_bars - val_bars
    train_end = t - val_bars
    if train_start < 200:
        return []
    
    metrics = {}
    for name, sig in alpha_signals.items():
        train_sr = compute_period_sharpe(sig, returns, train_start, train_end)
        val_sr = compute_period_sharpe(sig, returns, train_end, t)
        if train_sr is not None and val_sr is not None:
            metrics[name] = {"train": train_sr, "val": val_sr}
    
    passing = [n for n, m in metrics.items() if m["train"] > 0.3 and m["val"] > 0.0]
    passing.sort(key=lambda n: metrics[n]["val"] * 0.7 + metrics[n]["train"] * 0.3, reverse=True)
    
    selected = []
    for name in passing:
        is_ortho = all(
            compute_signal_correlation(alpha_signals[name], alpha_signals[s], train_start, t)
            <= max_corr
            for s in selected
        )
        if is_ortho:
            selected.append(name)
        if len(selected) >= max_alphas:
            break
    
    return selected, metrics


class QPCombiner:
    def __init__(self, signals, names, returns, lookback=45):
        self.signals = signals
        self.names = names
        self.returns = returns
        self.lookback = lookback
        self.factor_returns = {}
        for name in names:
            sig = signals[name]
            sig_n = sig.sub(sig.mean(axis=1), axis=0)
            sig_n = sig_n.div(sig_n.abs().sum(axis=1) + 1e-10, axis=0)
            self.factor_returns[name] = (sig_n.shift(1) * returns).sum(axis=1)
        self.fr_df = pd.DataFrame(self.factor_returns)
    
    def get_weights(self, t_idx):
        n = len(self.names)
        if t_idx < self.lookback + 20 or n == 0:
            return np.ones(n) / max(n, 1)
        fr = self.fr_df.iloc[max(0, t_idx - self.lookback - 1):t_idx - 1]
        mu = fr.mean().values
        cov = fr.cov().values + 0.02 * np.eye(n)
        def objective(w):
            return -np.dot(w, mu) + 0.5 * np.dot(w, cov @ w)
        bounds = [(0, 1)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n
        try:
            result = minimize(objective, w0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"maxiter": 50})
            return result.x if result.success else w0
        except Exception:
            return w0
    
    def combine(self, t_idx):
        if not self.names:
            return None, 0, np.array([])
        weights = self.get_weights(t_idx)
        n_active = int(np.sum(weights > 0.05))
        combined = pd.Series(0.0, index=self.signals[self.names[0]].columns)
        for i, name in enumerate(self.names):
            if weights[i] > 0.01:
                combined += weights[i] * self.signals[name].iloc[t_idx - 1]
        combined = combined.sub(combined.mean())
        combined = combined.div(combined.abs().sum() + 1e-10)
        return combined, n_active, weights


# ═════════════════════════════════════════════════════════════════
# CVXPY OPTIMIZER
# ═════════════════════════════════════════════════════════════════

def estimate_covariance(returns, shrinkage=0.9):
    """Ledoit-Wolf shrinkage toward scaled identity (matching CryptoRL)."""
    r = returns.fillna(0)
    n = r.shape[1]
    if len(r) < 10 or n < 2:
        return np.eye(n) * 0.0004
    sample_cov = r.cov().values
    sample_cov = np.nan_to_num(sample_cov, nan=0.0)
    avg_var = np.diag(sample_cov).mean()
    target = np.eye(n) * avg_var
    shrunk = shrinkage * target + (1 - shrinkage) * sample_cov
    min_eig = np.linalg.eigvalsh(shrunk).min()
    if min_eig < 0:
        shrunk -= np.eye(n) * (min_eig - 1e-8)
    return shrunk


def compute_pca_loadings(returns, n_factors=3):
    """PCA via SVD on centered returns (matching CryptoRL)."""
    r = returns.fillna(0)
    hist = r.iloc[-90:] if len(r) > 90 else r
    if len(hist) < 20 or hist.shape[1] < n_factors + 1:
        return None
    try:
        X = hist.values
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return Vt.T[:, :n_factors]
    except Exception:
        return None


_optimizer_stats = {"converged": 0, "failed": 0, "fallback": 0}


def optimize_portfolio(signal, returns, current_holdings,
                       max_position=0.04, max_gross_lev=0.80,
                       max_turnover=0.08, tx_cost_bps=10.0,
                       risk_aversion=1.0, n_pca_factors=3):
    if not CVXPY_AVAILABLE:
        _optimizer_stats["fallback"] += 1
        return signal.copy(), False
    
    assets = signal.index
    n = len(assets)
    r = returns.reindex(columns=assets).fillna(0)
    alpha = signal.fillna(0).values.astype(float)
    Sigma = estimate_covariance(r)
    pca = compute_pca_loadings(r, n_pca_factors)
    
    h_prev = np.zeros(n)
    if current_holdings is not None:
        h_prev = current_holdings.reindex(assets, fill_value=0).values
    
    h = cp.Variable(n)
    trade = h - h_prev
    tx_cost = (tx_cost_bps * 1e-4) * cp.norm(trade, 1)
    objective = cp.Maximize(
        alpha @ h - (risk_aversion / 2) * cp.quad_form(h, Sigma) - tx_cost
    )
    constraints = [
        cp.sum(h) == 0,
        cp.norm(h, "inf") <= max_position,
        cp.norm(h, 1) <= max_gross_lev,
        cp.norm(trade, 1) <= max_turnover,  # Always apply (matching CryptoRL)
    ]
    if pca is not None:
        for k in range(pca.shape[1]):
            constraints.append(pca[:, k] @ h == 0)
    
    problem = cp.Problem(objective, constraints)
    try:
        try:
            problem.solve(solver=cp.ECOS, verbose=False, max_iters=200)
        except Exception:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=500)
        if problem.status in ("optimal", "optimal_inaccurate") and h.value is not None:
            _optimizer_stats["converged"] += 1
            return pd.Series(h.value, index=assets), True
        else:
            _optimizer_stats["failed"] += 1
    except Exception:
        _optimizer_stats["failed"] += 1
    
    # Fallback: use forecast scaled to respect approximate turnover budget
    # This prevents the cascade failure where raw forecast (0.8 gross) creates
    # infeasibility on next bar with turnover constraint (0.08).
    _optimizer_stats["fallback"] += 1
    if current_holdings is not None:
        h_prev_s = current_holdings.reindex(assets, fill_value=0)
        # Scale forecast to be within turnover budget of h_prev
        trade = signal - h_prev_s
        trade_norm = trade.abs().sum()
        if trade_norm > max_turnover:
            scale = max_turnover / trade_norm
            return h_prev_s + trade * scale, False
        return signal.copy(), False
    else:
        # First bar: scale forecast to max_turnover gross leverage
        gross = signal.abs().sum()
        if gross > max_turnover:
            return signal * (max_turnover / gross), False
        return signal.copy(), False


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Walk-forward from AlphaDB")
    parser.add_argument("--days", type=int, default=480)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--category", type=str, default=None,
                        help="Filter alphas by category (momentum, reversal, etc.)")
    parser.add_argument("--source", type=str, default=None,
                        help="Filter alphas by source (cryptorl, gp, manual)")
    parser.add_argument("--db", type=str, default="data/alphas.db")
    args = parser.parse_args()

    # ── Load alphas from database ────────────────────────────────
    db = AlphaDB(args.db)
    alphas_rows = db.list_alphas(
        asset_class="crypto", interval="4h",
        category=args.category, source=args.source,
    )
    if not alphas_rows:
        print("No alphas found in database matching filters!")
        return
    
    alpha_exprs = {r["name"]: r["expression"] for r in alphas_rows}
    alpha_id_map = {r["name"]: r["id"] for r in alphas_rows}
    print(f"\n{'='*70}")
    print(f"LOADED {len(alpha_exprs)} ALPHAS FROM DATABASE")
    print(f"{'='*70}")
    if args.category:
        print(f"  Category filter: {args.category}")
    if args.source:
        print(f"  Source filter: {args.source}")

    # ── Load data ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LOADING DATA: top {args.top_n} symbols, {args.days} days, 4h")
    print(f"{'='*70}")
    features, idx, valid_syms = load_data(args.top_n, args.days)
    n_bars = len(idx)
    print(f"  {len(valid_syms)} symbols, {n_bars} bars ({idx[0].date()} → {idx[-1].date()})")

    # ── Compute alpha signals ────────────────────────────────────
    print(f"\nComputing {len(alpha_exprs)} alpha signals...")
    alpha_signals = {}
    for name, expr in alpha_exprs.items():
        sig = evaluate_expression(expr, features)
        if sig is not None:
            alpha_signals[name] = sig
    print(f"  {len(alpha_signals)} alphas computed ({len(alpha_exprs) - len(alpha_signals)} failed)")

    # ── Create run in database ───────────────────────────────────
    run_config = {
        "days": args.days, "fee_bps": args.fee_bps, "top_n": args.top_n,
        "category_filter": args.category, "source_filter": args.source,
        "n_symbols": len(valid_syms), "n_bars": n_bars,
        "train_bars": 720, "val_bars": 360, "reeval_interval": 6,
    }
    run_id = db.create_run(
        name=f"walkforward_{'_'.join(filter(None, [args.category, args.source, '4h']))}", 
        asset_class="crypto", interval="4h",
        universe=f"BINANCE_TOP{args.top_n}", fee_bps=args.fee_bps,
        config=run_config,
    )

    # ── Walk-forward backtest ────────────────────────────────────
    train_bars, val_bars = 720, 360
    start_bar = train_bars + val_bars + 200
    fee_bps = args.fee_bps
    
    rolling_vol = features["returns"].mean(axis=1).rolling(30).std() * ANN_FACTOR
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD BACKTEST (run_id={run_id})")
    print(f"{'='*70}")
    print(f"  Train: {train_bars} bars | Val: {val_bars} bars | Re-eval: every 6 bars")
    print(f"  Fee: {fee_bps} bps | Start bar: {start_bar}/{n_bars}")
    print(f"  Simulating {n_bars - start_bar} bars...")
    
    _optimizer_stats.update({"converged": 0, "failed": 0, "fallback": 0})
    
    results = []
    prev_holdings = None
    current_selected = []
    combiner = None
    equity = 1.0
    peak_equity = 1.0
    
    for t in range(start_bar, n_bars):
        # Alpha selection
        if (t - start_bar) % 6 == 0 or not current_selected:
            sel_result = select_alphas(alpha_signals, features["returns"], t, train_bars, val_bars)
            sel, metrics = sel_result
            if sel:
                current_selected = sel
                combiner = QPCombiner(
                    {n: alpha_signals[n] for n in sel},
                    sel, features["returns"], 45,
                )
                # Record selections in database
                sel_records = []
                for s_name in sel:
                    if s_name in alpha_id_map:
                        sel_records.append({
                            "alpha_id": alpha_id_map[s_name],
                            "train_sharpe": metrics[s_name]["train"],
                            "val_sharpe": metrics[s_name]["val"],
                        })
                if sel_records:
                    db.add_selections(run_id, t, str(idx[t]), sel_records)
        
        if not current_selected or combiner is None:
            continue
        
        # Universe
        trailing = features["adv20"].iloc[max(0, t - 60):t].mean()
        valid = trailing.dropna()
        if len(valid) < 20:
            continue
        universe = valid.nlargest(args.top_n).index.tolist()
        
        # Combine
        forecast, n_active, qp_weights = combiner.combine(t)
        if forecast is None:
            continue
        forecast = forecast.reindex(universe).fillna(0)
        forecast = forecast.sub(forecast.mean())
        forecast = forecast.div(forecast.abs().sum() + 1e-10)
        
        # Dynamic scaling
        cur_vol = rolling_vol.iloc[t] if t < len(rolling_vol) and not np.isnan(rolling_vol.iloc[t]) else 0.3
        vol_scale = min(1.0, 0.20 / (cur_vol + 0.01))
        dd = (equity - peak_equity) / peak_equity
        dd_scale = max(0.25, 1.0 + dd * 4) if dd < -0.05 else 1.0
        forecast = forecast * vol_scale * dd_scale
        
        # Optimize
        hist_ret = features["returns"].iloc[max(0, t - 120):t]  # CryptoRL walkforward_v8.py line 353
        cur = prev_holdings.reindex(forecast.index, fill_value=0) if prev_holdings is not None else None
        holdings, converged = optimize_portfolio(
            forecast, hist_ret, cur,
            max_position=0.04, max_gross_lev=0.80, max_turnover=0.08,
            tx_cost_bps=fee_bps * 2, risk_aversion=1.0, n_pca_factors=3,
        )
        
        # PnL
        actual_ret = features["returns"].iloc[t].reindex(universe, fill_value=0)
        pnl_gross = float((holdings * actual_ret).sum())
        turnover = float((holdings - (prev_holdings.reindex(holdings.index, fill_value=0)
                          if prev_holdings is not None else 0)).abs().sum())
        pnl_net = pnl_gross - turnover * fee_bps * 1e-4
        equity *= (1 + pnl_net)
        peak_equity = max(peak_equity, equity)
        
        results.append({
            "datetime": idx[t], "pnl_gross": pnl_gross, "pnl_net": pnl_net,
            "equity": equity, "turnover": turnover,
            "n_alphas": len(current_selected), "n_active": n_active,
            "gross_lev": float(holdings.abs().sum()), "converged": converged,
        })
        prev_holdings = holdings.copy()
        
        if (t - start_bar) % 200 == 0 and t > start_bar:
            sr = (np.mean([r["pnl_net"] for r in results]) /
                  (np.std([r["pnl_net"] for r in results]) + 1e-12)) * ANN_FACTOR
            conv = _optimizer_stats["converged"]
            total = conv + _optimizer_stats["failed"]
            conv_rate = conv / total * 100 if total > 0 else 0
            print(f"  Bar {t}/{n_bars}: equity={equity:.3f}, SR={sr:+.2f}, "
                  f"alphas={len(current_selected)}, optimizer={conv_rate:.0f}% converged")
    
    if not results:
        print("No results!")
        return
    
    # ── Results ──────────────────────────────────────────────────
    df = pd.DataFrame(results).set_index("datetime")
    sr_net = float((df["pnl_net"].mean() / (df["pnl_net"].std() + 1e-12)) * ANN_FACTOR)
    sr_gross = float((df["pnl_gross"].mean() / (df["pnl_gross"].std() + 1e-12)) * ANN_FACTOR)
    total_return = float(df["equity"].iloc[-1] - 1)
    max_dd = float(((df["equity"] - df["equity"].cummax()) / df["equity"].cummax()).min())
    
    conv = _optimizer_stats["converged"]
    total_opt = conv + _optimizer_stats["failed"]
    print(f"\n  Optimizer: {conv}/{total_opt} converged ({conv/total_opt*100:.0f}%), "
          f"{_optimizer_stats['fallback']} fallbacks")
    
    print(f"\n{'='*70}")
    print(f"RESULTS (fee = {fee_bps} bps one-way)")
    print(f"{'='*70}")
    print(f"  Sharpe (net):    {sr_net:+.2f}")
    print(f"  Sharpe (gross):  {sr_gross:+.2f}")
    recent = df["pnl_net"].iloc[-400:] if len(df) > 400 else df["pnl_net"]
    sr_recent = float((recent.mean() / (recent.std() + 1e-12)) * ANN_FACTOR)
    print(f"  Recent 400-bar:  {sr_recent:+.2f}")
    print(f"  Total Return:    {total_return:+.1%}")
    print(f"  Max Drawdown:    {max_dd:+.1%}")
    print(f"  Avg Turnover:    {df['turnover'].mean():.3f}")
    print(f"  Avg Gross Lev:   {df['gross_lev'].mean():.2f}")
    print(f"  Avg Alphas:      {df['n_alphas'].mean():.1f}")
    print(f"  Bars Simulated:  {len(df)}")
    print(f"  Date Range:      {df.index[0].date()} → {df.index[-1].date()}")
    
    # ── Save to database ─────────────────────────────────────────
    db.complete_run(run_id,
                    sharpe_net=sr_net, sharpe_gross=sr_gross,
                    total_return=total_return, max_drawdown=max_dd,
                    n_alphas_tested=len(alpha_signals),
                    n_alphas_passed=len(current_selected))
    
    # Store per-alpha evaluations for alphas that were ever selected
    selected_names = set()
    for row in db.get_selection_history(run_id):
        selected_names.add(row["name"])
    
    for name in selected_names:
        if name in alpha_id_map:
            # Get the most recent metrics for this alpha
            sr_train = compute_period_sharpe(alpha_signals[name], features["returns"],
                                             n_bars - train_bars - val_bars, n_bars - val_bars)
            sr_val = compute_period_sharpe(alpha_signals[name], features["returns"],
                                           n_bars - val_bars, n_bars)
            db.add_evaluation(
                alpha_id_map[name], run_id,
                sharpe_train=sr_train, sharpe_val=sr_val,
                test_start=str(df.index[0].date()),
                test_end=str(df.index[-1].date()),
                n_bars=len(df),
            )
    
    print(f"\n  Results saved to database (run_id={run_id})")
    
    # ── Chart ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(df.index, df["equity"], color="#00d4aa", linewidth=1.5)
    axes[0].fill_between(df.index, 1, df["equity"], alpha=0.15, color="#00d4aa")
    axes[0].axhline(1, color="gray", ls="--", alpha=0.3)
    axes[0].set_title(f"Walk-Forward Crypto Pipeline  |  Sharpe={sr_net:+.2f}  |  Return={total_return:+.1%}",
                      fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.2)
    
    axes[1].fill_between(df.index, df["turnover"], alpha=0.5, color="#ff6b6b")
    axes[1].set_ylabel("Turnover")
    axes[1].grid(alpha=0.2)
    
    axes[2].plot(df.index, df["n_alphas"], color="#4ecdc4", drawstyle="steps-post")
    axes[2].set_ylabel("# Alphas")
    axes[2].grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("walkforward_db_pipeline.png", dpi=150)
    print(f"  Chart saved: walkforward_db_pipeline.png")
    
    # ── Fee sensitivity ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FEE SENSITIVITY SWEEP")
    print(f"{'='*70}")
    for fee in [0, 3, 5, 10]:
        pnl_adj = df["pnl_gross"] - df["turnover"] * fee * 1e-4
        sr = float((pnl_adj.mean() / (pnl_adj.std() + 1e-12)) * ANN_FACTOR)
        ret = float((1 + pnl_adj).prod() - 1)
        print(f"  {fee:2d} bps: Sharpe={sr:+.2f}, Return={ret:+.1%}")
    
    db.close()


if __name__ == "__main__":
    main()
