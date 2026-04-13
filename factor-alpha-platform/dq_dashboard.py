"""
Data Quality & Outlier Detection Dashboard for 5m Crypto Klines
================================================================

Interactive Plotly Dash dashboard that runs a comprehensive data quality
analysis pipeline on the 5m matrices and displays live diagnostics.

References:
  - Brownlees & Gallo (2006): HF data cleaning, rolling MAD filter
  - Barndorff-Nielsen & Shephard (2004): Bipower variation for jump detection
  - Andersen, Bollerslev, Diebold & Labys (2003): Realized variance analysis
  - Boudt, Croux, Laurent (2011): Robust estimation of intraday volatility

Usage:
    python dq_dashboard.py [--universe TOP100] [--port 8050]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# ── Dash / Plotly ──
try:
    import dash
    from dash import dcc, html, Input, Output, State, dash_table
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("Installing dash and plotly...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dash", "plotly", "-q"])
    import dash
    from dash import dcc, html, Input, Output, State, dash_table
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px


# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR   = Path("data/binance_cache")
MAT_DIR    = DATA_DIR / "matrices" / "5m"
UNI_DIR    = DATA_DIR / "universes"
BARS_PER_DAY = 288

# DQ thresholds (match eval_portfolio_5m.py)
DQ_NAN_TICKER_THRESHOLD    = 0.30
DQ_STALE_TICKER_THRESHOLD  = 0.20
DQ_MIN_QUOTE_VOL           = 500.0
DQ_STALE_RUN_BARS          = 6
DQ_ZERO_VOL_BARS           = 6
DQ_RETURN_HARD_CAP         = 0.15
DQ_RETURN_ADAPTIVE_MULT    = 5.0
DQ_RETURN_ADAPTIVE_FLOOR   = 0.02
DQ_QV_SPIKE_MULT           = 50.0
DQ_TICK_SIZE_RATIO         = 0.40


# ============================================================================
# DATA LOADING & ANALYSIS
# ============================================================================

def load_matrices(universe="TOP100"):
    """Load all 5m matrices for the given universe."""
    uni_path = UNI_DIR / f"BINANCE_{universe}_5m.parquet"
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > 0.3].index.tolist())

    matrices = {}
    for fp in sorted(MAT_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    return matrices, valid_tickers, universe_df


def analyze_completeness(matrices, valid_tickers):
    """Per-ticker data completeness analysis."""
    close = matrices.get("close")
    if close is None:
        return pd.DataFrame()

    results = []
    for ticker in valid_tickers:
        if ticker not in close.columns:
            continue
        s = close[ticker]
        total = len(s)
        valid = s.notna().sum()
        nan_pct = s.isna().mean() * 100
        # First and last valid bar
        valid_idx = s.dropna().index
        if len(valid_idx) > 0:
            first_bar = valid_idx[0]
            last_bar = valid_idx[-1]
            active_days = (last_bar - first_bar).days
        else:
            first_bar = last_bar = None
            active_days = 0
        results.append({
            "ticker": ticker, "total_bars": total, "valid_bars": int(valid),
            "nan_pct": round(nan_pct, 2), "first_bar": str(first_bar)[:10],
            "last_bar": str(last_bar)[:10], "active_days": active_days
        })
    return pd.DataFrame(results).sort_values("nan_pct", ascending=False)


def analyze_staleness(matrices, valid_tickers):
    """Detect stale price runs per ticker."""
    close = matrices.get("close")
    if close is None:
        return pd.DataFrame()

    results = []
    diffs = close.diff(axis=0)
    for ticker in valid_tickers:
        if ticker not in close.columns:
            continue
        s = diffs[ticker]
        is_stale = (s == 0) & close[ticker].notna()
        # Overall stale fraction
        stale_frac = is_stale.mean() * 100
        # Longest stale run
        if is_stale.any():
            ri = (~is_stale).cumsum()
            run_lens = is_stale.groupby(ri).sum()
            max_run = int(run_lens.max())
            n_runs_gt6 = int((run_lens >= DQ_STALE_RUN_BARS).sum())
        else:
            max_run = 0
            n_runs_gt6 = 0
        results.append({
            "ticker": ticker, "stale_frac_pct": round(stale_frac, 2),
            "max_stale_run": max_run, "runs_gt_30min": n_runs_gt6,
            "status": "EXCLUDE" if stale_frac > DQ_STALE_TICKER_THRESHOLD * 100 else "OK"
        })
    return pd.DataFrame(results).sort_values("stale_frac_pct", ascending=False)


def analyze_returns(matrices, valid_tickers):
    """Return distribution analysis including outlier detection."""
    close = matrices.get("close")
    if close is None:
        return {}, pd.DataFrame()

    returns = close.pct_change()
    qv = matrices.get("quote_volume")

    # Per-ticker return stats
    ticker_stats = []
    for ticker in valid_tickers:
        if ticker not in returns.columns:
            continue
        r = returns[ticker].dropna()
        if len(r) < 100:
            continue
        # Hard cap violations
        n_hard = int((r.abs() > DQ_RETURN_HARD_CAP).sum())
        # Jump detection using Barndorff-Nielsen & Shephard bipower variation
        abs_r = r.abs()
        abs_r_shift = abs_r.shift(1)
        bv = (np.pi / 2) * (abs_r * abs_r_shift).rolling(BARS_PER_DAY).mean()
        rv = (r ** 2).rolling(BARS_PER_DAY).mean()
        # Jump ratio: RV/BV > 1 indicates jumps (diffusive component < realized)
        jump_ratio = (rv / bv.clip(lower=1e-15)).fillna(1)
        n_jumps = int((jump_ratio > 3.0).sum())
        # Adaptive violations
        mar = abs_r.rolling(BARS_PER_DAY, min_periods=50).median().shift(1)
        adaptive_cap = (mar * DQ_RETURN_ADAPTIVE_MULT).clip(lower=DQ_RETURN_ADAPTIVE_FLOOR)
        n_adaptive = int((abs_r > adaptive_cap).sum())
        # QV spike
        n_qv = 0
        if qv is not None and ticker in qv.columns:
            qv_t = qv[ticker].reindex(r.index)
            qv_med = qv_t.rolling(BARS_PER_DAY, min_periods=50).median().shift(1)
            qv_spike = (qv_t > qv_med * DQ_QV_SPIKE_MULT) & (abs_r > 0.02)
            n_qv = int(qv_spike.sum())

        ticker_stats.append({
            "ticker": ticker,
            "mean_ret": round(r.mean() * 1e4, 3),  # bps
            "std_ret": round(r.std() * 100, 4),  # %
            "skew": round(r.skew(), 2),
            "kurtosis": round(r.kurtosis(), 1),
            "min_ret_pct": round(r.min() * 100, 2),
            "max_ret_pct": round(r.max() * 100, 2),
            "n_hard_cap": n_hard,
            "n_adaptive": n_adaptive,
            "n_bns_jumps": n_jumps,
            "n_qv_spike": n_qv,
            "total_outliers": n_hard + n_adaptive + n_qv,
        })

    df_stats = pd.DataFrame(ticker_stats).sort_values("total_outliers", ascending=False)

    # Global stats
    all_ret = returns.values.flatten()
    all_ret = all_ret[~np.isnan(all_ret)]
    global_stats = {
        "n_bars": len(all_ret),
        "mean_bps": round(np.mean(all_ret) * 1e4, 3),
        "std_pct": round(np.std(all_ret) * 100, 4),
        "skew": round(float(pd.Series(all_ret).skew()), 2),
        "kurtosis": round(float(pd.Series(all_ret).kurtosis()), 1),
        "pct_gt_5pct": round((np.abs(all_ret) > 0.05).mean() * 100, 4),
        "pct_gt_10pct": round((np.abs(all_ret) > 0.10).mean() * 100, 4),
        "pct_gt_15pct": round((np.abs(all_ret) > 0.15).mean() * 100, 4),
    }
    return global_stats, df_stats


def analyze_liquidity(matrices, valid_tickers):
    """Quote volume distribution analysis."""
    qv = matrices.get("quote_volume")
    if qv is None:
        return pd.DataFrame()

    results = []
    for ticker in valid_tickers:
        if ticker not in qv.columns:
            continue
        s = qv[ticker].dropna()
        if len(s) < 100:
            continue
        results.append({
            "ticker": ticker,
            "median_qv": round(s.median(), 0),
            "mean_qv": round(s.mean(), 0),
            "p5_qv": round(s.quantile(0.05), 0),
            "p95_qv": round(s.quantile(0.95), 0),
            "pct_below_500": round((s < DQ_MIN_QUOTE_VOL).mean() * 100, 2),
            "zero_vol_pct": round((s == 0).mean() * 100, 2),
        })
    return pd.DataFrame(results).sort_values("median_qv", ascending=True)


def analyze_hloc_sanity(matrices, valid_tickers):
    """HLOC consistency checks."""
    close = matrices.get("close")
    high = matrices.get("high")
    low = matrices.get("low")
    if close is None or high is None or low is None:
        return pd.DataFrame()

    results = []
    for ticker in valid_tickers:
        if ticker not in close.columns:
            continue
        c = close[ticker]
        h = high[ticker] if ticker in high.columns else pd.Series(dtype=float)
        lo = low[ticker] if ticker in low.columns else pd.Series(dtype=float)
        valid = c.notna() & h.notna() & lo.notna()
        n_valid = valid.sum()
        if n_valid < 100:
            continue
        n_hl_bad = int(((h < lo) & valid).sum())
        n_close_outside = int((((c < lo) | (c > h)) & valid).sum())
        n_flat = int(((h == lo) & (h == c) & valid).sum())
        n_neg = int(((c <= 0) & c.notna()).sum())
        total_issues = n_hl_bad + n_close_outside + n_flat + n_neg
        results.append({
            "ticker": ticker, "valid_bars": int(n_valid),
            "high_lt_low": n_hl_bad, "close_outside": n_close_outside,
            "flat_bars": n_flat, "neg_price": n_neg,
            "total_issues": total_issues,
            "issue_pct": round(total_issues / n_valid * 100, 4)
        })
    return pd.DataFrame(results).sort_values("total_issues", ascending=False)


def analyze_cross_sectional(matrices, valid_tickers):
    """Cross-sectional dispersion and correlation over time."""
    close = matrices.get("close")
    if close is None:
        return pd.DataFrame()

    returns = close[valid_tickers].pct_change()
    # Daily cross-sectional stats
    daily_idx = range(0, len(returns), BARS_PER_DAY)
    cs_stats = []
    for i in daily_idx:
        chunk = returns.iloc[i:i+BARS_PER_DAY]
        if len(chunk) < BARS_PER_DAY // 2:
            continue
        mean_ret = chunk.mean(axis=0)
        date = chunk.index[0]
        cs_stats.append({
            "date": date,
            "cs_dispersion": round(mean_ret.std() * 100, 4),
            "cs_mean": round(mean_ret.mean() * 100, 4),
            "cs_skew": round(float(mean_ret.skew()), 2),
            "n_tickers_active": int(chunk.notna().any(axis=0).sum()),
            "pct_nan": round(chunk.isna().mean().mean() * 100, 2),
        })
    return pd.DataFrame(cs_stats)


def analyze_tick_discretization(matrices, valid_tickers):
    """Tick-size discretization analysis.

    Detects tickers where the minimum price increment (tick size) is large
    relative to the price, producing quantized returns.  Example: CRV trades
    at ~$0.50 with a tick of $0.001 = 0.2% per tick, meaning most 5m bars
    show zero change — not because the market is dead, but because the
    price literally can't move in sub-tick increments.

    This is fundamentally different from genuine staleness (delisted/dead
    tickers) and taints alpha signals via:
      1. Quantized return distribution (many exact zeros + occasional jumps)
      2. Noisy cross-sectional ranks (random ties at zero-change)
      3. False momentum/mean-reversion signals from tick-level noise
    """
    close = matrices.get("close")
    if close is None:
        return pd.DataFrame()

    results = []
    diffs = close.diff(axis=0)
    for ticker in valid_tickers:
        if ticker not in close.columns:
            continue
        c = close[ticker].dropna()
        if len(c) < 500:
            continue
        d = diffs[ticker].dropna()
        # Zero-change fraction
        zero_frac = (d == 0).mean() * 100
        # Median absolute return
        ret = c.pct_change().dropna().abs()
        med_abs_ret = ret.median() * 100
        # Unique price levels
        n_unique = c.nunique()
        # Tick size estimate: median of non-zero absolute price changes
        non_zero_diffs = d[d != 0].abs()
        if len(non_zero_diffs) > 10:
            est_tick = non_zero_diffs.quantile(0.10)  # 10th percentile of moves
            med_price = c.median()
            tick_pct = (est_tick / med_price) * 100 if med_price > 0 else 0
        else:
            est_tick = 0
            tick_pct = 0
        # Diagnosis
        if zero_frac > DQ_TICK_SIZE_RATIO * 100:
            status = "EXCLUDE"
        elif zero_frac > DQ_STALE_TICKER_THRESHOLD * 100:
            status = "WARN"
        else:
            status = "OK"
        results.append({
            "ticker": ticker,
            "zero_change_pct": round(zero_frac, 2),
            "med_abs_ret_pct": round(med_abs_ret, 4),
            "n_unique_prices": n_unique,
            "est_tick_size": round(est_tick, 6),
            "tick_pct_price": round(tick_pct, 4),
            "med_price": round(c.median(), 4),
            "status": status,
        })
    return pd.DataFrame(results).sort_values("zero_change_pct", ascending=False)


def build_tick_discretization_fig(tick_df):
    """Scatter: zero-change% vs median |return|% — exposes coarse tick sizes."""
    if tick_df.empty:
        return go.Figure()
    fig = go.Figure()
    colors = tick_df["status"].map({"EXCLUDE": "#ef4444", "WARN": "#f59e0b", "OK": "#10b981"})
    fig.add_trace(go.Scatter(
        x=tick_df["zero_change_pct"], y=tick_df["med_abs_ret_pct"],
        mode="markers+text", text=tick_df["ticker"],
        textposition="top center", textfont=dict(size=7),
        marker=dict(
            size=np.clip(tick_df["tick_pct_price"] * 40, 6, 40),
            color=colors, opacity=0.8,
            line=dict(width=1, color="#1e1b4b"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Zero-change: %{x:.1f}%<br>"
            "Med |ret|: %{y:.4f}%<br>"
            "Tick/Price: %{customdata[0]:.4f}%<br>"
            "Med Price: $%{customdata[1]:.2f}"
            "<extra></extra>"
        ),
        customdata=np.column_stack([
            tick_df["tick_pct_price"].values,
            tick_df["med_price"].values,
        ]),
    ))
    # Threshold lines
    fig.add_vline(x=DQ_TICK_SIZE_RATIO * 100, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Exclude ({DQ_TICK_SIZE_RATIO*100:.0f}%)")
    fig.add_vline(x=DQ_STALE_TICKER_THRESHOLD * 100, line_dash="dot", line_color="#f59e0b",
                  annotation_text=f"Warn ({DQ_STALE_TICKER_THRESHOLD*100:.0f}%)")
    fig.update_layout(
        template="plotly_dark",
        title="Tick-Size Discretization (size = tick/price ratio)",
        xaxis_title="Zero-Change Fraction (%)",
        yaxis_title="Median |Return| (%)",
        height=450, margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def build_return_distribution_fig(matrices, valid_tickers):
    """Build return distribution histogram with outlier thresholds."""
    close = matrices.get("close")
    if close is None:
        return go.Figure()

    returns = close[valid_tickers].pct_change()
    all_ret = returns.values.flatten()
    all_ret = all_ret[~np.isnan(all_ret)]
    # Clip for visualization
    clipped = np.clip(all_ret, -0.10, 0.10)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clipped * 100, nbinsx=500, name="5m Returns",
        marker_color="#6366f1", opacity=0.8
    ))
    # Threshold lines
    for thresh, color, label in [
        (DQ_RETURN_HARD_CAP * 100, "#ef4444", f"Hard Cap ({DQ_RETURN_HARD_CAP*100:.0f}%)"),
        (-DQ_RETURN_HARD_CAP * 100, "#ef4444", ""),
        (DQ_RETURN_ADAPTIVE_FLOOR * 100, "#f59e0b", f"Adaptive Floor ({DQ_RETURN_ADAPTIVE_FLOOR*100:.0f}%)"),
        (-DQ_RETURN_ADAPTIVE_FLOOR * 100, "#f59e0b", ""),
    ]:
        fig.add_vline(x=thresh, line_dash="dash", line_color=color,
                      annotation_text=label if label else None)

    fig.update_layout(
        template="plotly_dark",
        title="5m Return Distribution (clipped to ±10% for display)",
        xaxis_title="Return (%)", yaxis_title="Count",
        yaxis_type="log", height=400,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def build_staleness_heatmap(matrices, valid_tickers, top_n=40):
    """Build a heatmap of daily staleness by ticker."""
    close = matrices.get("close")
    if close is None:
        return go.Figure()

    diffs = close[valid_tickers].diff(axis=0)
    is_stale = (diffs == 0) & close[valid_tickers].notna()

    # Resample to daily stale fraction
    daily_stale = is_stale.resample("1D").mean() * 100

    # Pick worst tickers
    worst = daily_stale.mean().nlargest(top_n).index.tolist()
    data = daily_stale[worst].T

    fig = go.Figure(data=go.Heatmap(
        z=data.values, x=[str(d)[:10] for d in data.columns],
        y=data.index.tolist(), colorscale="YlOrRd",
        colorbar=dict(title="Stale %"),
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"Daily Staleness Heatmap (Top {top_n} Worst Tickers)",
        height=max(400, top_n * 18), margin=dict(l=120, r=30, t=50, b=40),
    )
    return fig


def build_liquidity_timeseries(matrices, valid_tickers):
    """Daily aggregate quote volume over time."""
    qv = matrices.get("quote_volume")
    if qv is None:
        return go.Figure()

    daily_qv = qv[valid_tickers].resample("1D").sum().sum(axis=1)
    daily_n = qv[valid_tickers].resample("1D").apply(lambda x: (x > 0).any(axis=0).sum())

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Daily Total Quote Volume ($)", "Active Tickers"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=daily_qv.index, y=daily_qv.values / 1e9,
        fill="tozeroy", fillcolor="rgba(99,102,241,0.3)",
        line=dict(color="#6366f1"), name="Quote Vol ($B)"
    ), row=1, col=1)

    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=60, r=30, t=50, b=40),
        showlegend=False,
    )
    return fig


def build_cs_dispersion_fig(cs_df):
    """Cross-sectional dispersion timeseries."""
    if cs_df.empty:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Cross-Sectional Dispersion (%)", "NaN % by Day"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=cs_df["date"], y=cs_df["cs_dispersion"],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.3)",
        line=dict(color="#10b981"), name="CS Dispersion"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cs_df["date"], y=cs_df["pct_nan"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.3)",
        line=dict(color="#ef4444"), name="NaN %"
    ), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=60, r=30, t=50, b=40), showlegend=False,
    )
    return fig


def build_outlier_scatter(df_returns):
    """Scatter plot of outlier stats per ticker."""
    if df_returns.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_returns["kurtosis"], y=df_returns["total_outliers"],
        mode="markers+text", text=df_returns["ticker"],
        textposition="top center", textfont=dict(size=8),
        marker=dict(
            size=df_returns["std_ret"] * 50,
            color=df_returns["skew"],
            colorscale="RdBu_r", showscale=True,
            colorbar=dict(title="Skew"),
            line=dict(width=1, color="#1e1b4b"),
        ),
        hovertemplate="<b>%{text}</b><br>Kurt: %{x:.1f}<br>Outliers: %{y}<br>StdRet: %{customdata:.3f}%<extra></extra>",
        customdata=df_returns["std_ret"],
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Per-Ticker Outlier Map (size=volatility, color=skew)",
        xaxis_title="Kurtosis", yaxis_title="Total Outliers",
        height=500, margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def build_jump_detection_fig(matrices, ticker):
    """BNS jump detection for a single ticker."""
    close = matrices.get("close")
    if close is None or ticker not in close.columns:
        return go.Figure()

    r = close[ticker].pct_change().dropna()
    abs_r = r.abs()
    abs_r_shift = abs_r.shift(1)
    bv = (np.pi / 2) * (abs_r * abs_r_shift).rolling(BARS_PER_DAY).mean()
    rv = (r ** 2).rolling(BARS_PER_DAY).mean()
    jump_ratio = (rv / bv.clip(lower=1e-15)).fillna(1)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=[f"{ticker} Returns", "RV vs BV", "Jump Ratio (RV/BV)"],
                        vertical_spacing=0.06, row_heights=[0.4, 0.3, 0.3])
    fig.add_trace(go.Scatter(
        x=r.index, y=r.values * 100, mode="lines",
        line=dict(color="#6366f1", width=0.5), name="Return %"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rv.index, y=rv.values * 1e4, mode="lines",
        line=dict(color="#ef4444"), name="RV"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bv.index, y=bv.values * 1e4, mode="lines",
        line=dict(color="#10b981"), name="BV"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=jump_ratio.index, y=jump_ratio.values, mode="lines",
        line=dict(color="#f59e0b"), name="Jump Ratio"
    ), row=3, col=1)
    fig.add_hline(y=3.0, line_dash="dash", line_color="#ef4444", row=3, col=1,
                  annotation_text="Jump Threshold (3x)")
    fig.update_layout(
        template="plotly_dark", height=600,
        margin=dict(l=60, r=30, t=50, b=40), showlegend=False,
    )
    return fig


# ============================================================================
# DASH APP
# ============================================================================

def create_app(universe="TOP100"):
    print(f"Loading {universe} data...", flush=True)
    matrices, valid_tickers, universe_df = load_matrices(universe)
    print(f"  Loaded {len(matrices)} fields, {len(valid_tickers)} tickers", flush=True)
    print(f"  Date range: {matrices['close'].index[0]} -> {matrices['close'].index[-1]}", flush=True)

    print("Running analysis...", flush=True)
    completeness_df = analyze_completeness(matrices, valid_tickers)
    staleness_df = analyze_staleness(matrices, valid_tickers)
    global_ret_stats, returns_df = analyze_returns(matrices, valid_tickers)
    liquidity_df = analyze_liquidity(matrices, valid_tickers)
    hloc_df = analyze_hloc_sanity(matrices, valid_tickers)
    cs_df = analyze_cross_sectional(matrices, valid_tickers)
    tick_df = analyze_tick_discretization(matrices, valid_tickers)
    print("Analysis complete.", flush=True)

    # Summary cards data
    close = matrices["close"]
    n_tickers = len(valid_tickers)
    n_bars = close.shape[0]
    date_range = f"{str(close.index[0])[:10]} to {str(close.index[-1])[:10]}"
    total_cells = n_tickers * n_bars
    nan_pct = round(close[valid_tickers].isna().mean().mean() * 100, 2)
    n_excluded_stale = len(staleness_df[staleness_df["status"] == "EXCLUDE"]) if not staleness_df.empty else 0
    total_outliers = int(returns_df["total_outliers"].sum()) if not returns_df.empty else 0
    total_hloc = int(hloc_df["total_issues"].sum()) if not hloc_df.empty else 0
    n_tick_exclude = len(tick_df[tick_df["status"] == "EXCLUDE"]) if not tick_df.empty else 0
    n_tick_warn = len(tick_df[tick_df["status"] == "WARN"]) if not tick_df.empty else 0

    # ── Build App ──
    app = dash.Dash(__name__, title=f"DQ Dashboard — {universe}")

    # Styles
    CARD_STYLE = {
        "background": "linear-gradient(135deg, #1e1b4b 0%, #312e81 100%)",
        "borderRadius": "12px", "padding": "20px", "textAlign": "center",
        "boxShadow": "0 4px 20px rgba(0,0,0,0.3)", "flex": "1", "minWidth": "200px",
    }
    CARD_TITLE = {"color": "#a5b4fc", "fontSize": "13px", "marginBottom": "5px", "fontWeight": "500"}
    CARD_VALUE = {"color": "#e0e7ff", "fontSize": "28px", "fontWeight": "700"}
    SECTION_STYLE = {
        "background": "#1e1b4b", "borderRadius": "12px", "padding": "20px",
        "marginBottom": "20px", "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
    }
    SECTION_TITLE = {"color": "#c7d2fe", "fontSize": "18px", "fontWeight": "600", "marginBottom": "15px"}
    TABLE_STYLE = {
        "overflowX": "auto", "backgroundColor": "#0f0d2e",
        "color": "#e0e7ff", "border": "1px solid #312e81",
    }
    TABLE_HEADER_STYLE = {
        "backgroundColor": "#1e1b4b", "color": "#a5b4fc",
        "fontWeight": "bold", "border": "1px solid #312e81",
    }
    TABLE_CELL_STYLE = {
        "backgroundColor": "#0f0d2e", "color": "#e0e7ff",
        "border": "1px solid #1e1b4b", "fontSize": "12px",
    }

    app.layout = html.Div(style={
        "backgroundColor": "#0a0820", "minHeight": "100vh", "padding": "30px",
        "fontFamily": "'Inter', -apple-system, sans-serif",
    }, children=[
        # ── Header ──
        html.Div(style={"textAlign": "center", "marginBottom": "30px"}, children=[
            html.H1("Data Quality & Outlier Detection", style={
                "color": "#e0e7ff", "fontSize": "32px", "fontWeight": "800",
                "background": "linear-gradient(90deg, #818cf8, #c084fc)",
                "-webkit-background-clip": "text",
                "-webkit-text-fill-color": "transparent",
                "marginBottom": "5px",
            }),
            html.P(f"{universe} | {n_tickers} tickers | {n_bars:,} bars | {date_range}", style={
                "color": "#7c8aad", "fontSize": "14px",
            }),
        ]),

        # ── Summary Cards ──
        html.Div(style={"display": "flex", "gap": "15px", "flexWrap": "wrap", "marginBottom": "25px"}, children=[
            html.Div(style=CARD_STYLE, children=[
                html.Div("Total Cells", style=CARD_TITLE),
                html.Div(f"{total_cells:,}", style=CARD_VALUE),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Missing Data %", style=CARD_TITLE),
                html.Div(f"{nan_pct}%", style={**CARD_VALUE,
                    "color": "#ef4444" if nan_pct > 5 else "#10b981"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Excluded (Stale)", style=CARD_TITLE),
                html.Div(f"{n_excluded_stale}", style={**CARD_VALUE,
                    "color": "#f59e0b" if n_excluded_stale > 5 else "#10b981"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Return Outliers", style=CARD_TITLE),
                html.Div(f"{total_outliers:,}", style={**CARD_VALUE,
                    "color": "#ef4444" if total_outliers > 1000 else "#10b981"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("HLOC Violations", style=CARD_TITLE),
                html.Div(f"{total_hloc:,}", style={**CARD_VALUE,
                    "color": "#ef4444" if total_hloc > 100 else "#10b981"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Kurtosis (Global)", style=CARD_TITLE),
                html.Div(f"{global_ret_stats.get('kurtosis', 'N/A')}", style={**CARD_VALUE,
                    "color": "#f59e0b" if global_ret_stats.get('kurtosis', 0) > 20 else "#10b981"}),
            ]),
            html.Div(style=CARD_STYLE, children=[
                html.Div("Tick Discretized", style=CARD_TITLE),
                html.Div(f"{n_tick_exclude} / {n_tick_warn}",
                         style={**CARD_VALUE, "fontSize": "24px",
                    "color": "#ef4444" if n_tick_exclude > 0 else "#10b981"}),
                html.Div("excl / warn", style={"color": "#6b7280", "fontSize": "11px"}),
            ]),
        ]),

        # ── Row 1: Return Distribution + Outlier Map ──
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Return Distribution", style=SECTION_TITLE),
                dcc.Graph(figure=build_return_distribution_fig(matrices, valid_tickers), config={"displayModeBar": False}),
                html.Div(style={"display": "flex", "gap": "20px", "marginTop": "10px", "flexWrap": "wrap"}, children=[
                    html.Span(f">5%: {global_ret_stats.get('pct_gt_5pct', 'N/A')}%", style={"color": "#f59e0b", "fontSize": "12px"}),
                    html.Span(f">10%: {global_ret_stats.get('pct_gt_10pct', 'N/A')}%", style={"color": "#ef4444", "fontSize": "12px"}),
                    html.Span(f">15%: {global_ret_stats.get('pct_gt_15pct', 'N/A')}%", style={"color": "#dc2626", "fontSize": "12px"}),
                    html.Span(f"Skew: {global_ret_stats.get('skew', 'N/A')}", style={"color": "#a5b4fc", "fontSize": "12px"}),
                ]),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Outlier Map (Per Ticker)", style=SECTION_TITLE),
                dcc.Graph(figure=build_outlier_scatter(returns_df), config={"displayModeBar": False}),
            ]),
        ]),

        # ── Row 2: Tick Discretization + Staleness Heatmap ──
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Tick-Size Discretization", style=SECTION_TITLE),
                html.P("Tickers with coarse tick sizes produce quantized returns that poison alpha signals. "
                       "CRV is a classic example: actively traded but price can't move in sub-tick increments.",
                       style={"color": "#7c8aad", "fontSize": "12px", "marginBottom": "10px"}),
                dcc.Graph(figure=build_tick_discretization_fig(tick_df), config={"displayModeBar": False}),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Staleness Heatmap (Daily, Top 40 Worst)", style=SECTION_TITLE),
                dcc.Graph(figure=build_staleness_heatmap(matrices, valid_tickers, top_n=40), config={"displayModeBar": False}),
            ]),
        ]),

        # ── Row 3: CS Dispersion + Liquidity ──
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Cross-Sectional Dispersion & Coverage", style=SECTION_TITLE),
                dcc.Graph(figure=build_cs_dispersion_fig(cs_df), config={"displayModeBar": False}),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Aggregate Liquidity", style=SECTION_TITLE),
                dcc.Graph(figure=build_liquidity_timeseries(matrices, valid_tickers), config={"displayModeBar": False}),
            ]),
        ]),

        # ── Row 4: Jump Detection (Interactive) ──
        html.Div(style=SECTION_STYLE, children=[
            html.H3("Barndorff-Nielsen & Shephard Jump Detection", style=SECTION_TITLE),
            html.P("RV/BV > 3 indicates significant price jumps (discontinuous moves vs diffusive volatility)",
                   style={"color": "#7c8aad", "fontSize": "12px", "marginBottom": "10px"}),
            dcc.Dropdown(
                id="jump-ticker-select",
                options=[{"label": t, "value": t} for t in valid_tickers[:50]],
                value=valid_tickers[0] if valid_tickers else None,
                style={"backgroundColor": "#1e1b4b", "color": "#e0e7ff", "marginBottom": "10px"},
            ),
            dcc.Graph(id="jump-chart", config={"displayModeBar": False}),
        ]),

        # ── Row 5: Tables ──
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Completeness (Worst First)", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=completeness_df.head(30).to_dict("records") if not completeness_df.empty else [],
                    columns=[{"name": c, "id": c} for c in completeness_df.columns] if not completeness_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                    style_data_conditional=[
                        {"if": {"filter_query": "{nan_pct} > 30"}, "backgroundColor": "#7f1d1d", "color": "#fca5a5"},
                        {"if": {"filter_query": "{nan_pct} > 10 && {nan_pct} <= 30"}, "backgroundColor": "#78350f", "color": "#fbbf24"},
                    ],
                ),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Staleness Analysis", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=staleness_df.head(30).to_dict("records") if not staleness_df.empty else [],
                    columns=[{"name": c, "id": c} for c in staleness_df.columns] if not staleness_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                    style_data_conditional=[
                        {"if": {"filter_query": '{status} = "EXCLUDE"'}, "backgroundColor": "#7f1d1d", "color": "#fca5a5"},
                    ],
                ),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Return Outlier Stats (Per Ticker)", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=returns_df.head(30).to_dict("records") if not returns_df.empty else [],
                    columns=[{"name": c, "id": c} for c in returns_df.columns] if not returns_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                    style_data_conditional=[
                        {"if": {"filter_query": "{total_outliers} > 100"}, "backgroundColor": "#7f1d1d", "color": "#fca5a5"},
                    ],
                ),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("HLOC Sanity Violations", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=hloc_df.head(30).to_dict("records") if not hloc_df.empty else [],
                    columns=[{"name": c, "id": c} for c in hloc_df.columns] if not hloc_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                    style_data_conditional=[
                        {"if": {"filter_query": "{total_issues} > 50"}, "backgroundColor": "#7f1d1d", "color": "#fca5a5"},
                    ],
                ),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Tick Discretization Analysis", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=tick_df.head(30).to_dict("records") if not tick_df.empty else [],
                    columns=[{"name": c, "id": c} for c in tick_df.columns] if not tick_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                    style_data_conditional=[
                        {"if": {"filter_query": '{status} = "EXCLUDE"'}, "backgroundColor": "#7f1d1d", "color": "#fca5a5"},
                        {"if": {"filter_query": '{status} = "WARN"'}, "backgroundColor": "#78350f", "color": "#fbbf24"},
                    ],
                ),
            ]),
            html.Div(style=SECTION_STYLE, children=[
                html.H3("Liquidity Analysis", style=SECTION_TITLE),
                dash_table.DataTable(
                    data=liquidity_df.head(30).to_dict("records") if not liquidity_df.empty else [],
                    columns=[{"name": c, "id": c} for c in liquidity_df.columns] if not liquidity_df.empty else [],
                    style_table=TABLE_STYLE, style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE, page_size=15,
                ),
            ]),
        ]),

        # ── Footer ──
        html.Div(style={"textAlign": "center", "marginTop": "30px", "padding": "20px"}, children=[
            html.P("References: Brownlees & Gallo (2006), Barndorff-Nielsen & Shephard (2004), "
                   "Andersen, Bollerslev, Diebold & Labys (2003), Boudt, Croux, Laurent (2011)",
                   style={"color": "#4b5563", "fontSize": "11px"}),
            html.P(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   style={"color": "#4b5563", "fontSize": "11px"}),
        ]),
    ])

    # ── Callbacks ──
    @app.callback(
        Output("jump-chart", "figure"),
        Input("jump-ticker-select", "value"),
    )
    def update_jump_chart(ticker):
        if not ticker:
            return go.Figure()
        return build_jump_detection_fig(matrices, ticker)

    return app


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Dashboard")
    parser.add_argument("--universe", default="TOP100", choices=["TOP100", "TOP50", "TOP20"])
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(universe=args.universe)
    print(f"\n  Dashboard running at http://localhost:{args.port}", flush=True)
    app.run(debug=args.debug, port=args.port)
