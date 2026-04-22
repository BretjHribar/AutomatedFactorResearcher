"""
prod/stats/dashboard.py — Unified Performance Dashboard

Reads the unified trade logs from all exchanges and produces:
  1. Live equity curves (per exchange + combined)
  2. Daily PnL breakdown
  3. Summary statistics

Usage:
  python prod/stats/dashboard.py                  # Show all exchanges
  python prod/stats/dashboard.py --exchange binance  # Binance only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_ROOT = PROJECT_ROOT / "prod" / "logs"
OUT_DIR = PROJECT_ROOT / "prod" / "stats" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCHANGES = {
    "ib": {"name": "IB Equities", "color": "#2196F3", "perf_dir": "performance"},
    "binance": {"name": "Binance Futures", "color": "#F0B90B", "perf_dir": "binance/performance"},
    "kucoin": {"name": "KuCoin Futures", "color": "#23AF91", "perf_dir": "kucoin/performance"},
}


def load_equity_csv(exchange: str) -> pd.DataFrame | None:
    """Load the equity CSV for an exchange."""
    info = EXCHANGES.get(exchange)
    if not info:
        return None
    
    perf_dir = LOGS_ROOT / info["perf_dir"]
    
    # Find the equity CSV
    for pattern in [f"equity_{exchange}.csv", "equity_*.csv"]:
        files = list(perf_dir.glob(pattern))
        if files:
            df = pd.read_csv(files[0], parse_dates=["timestamp"])
            return df
    
    return None


def load_trade_logs(exchange: str) -> list[dict]:
    """Load all trade log JSONs for an exchange."""
    log_dirs = {
        "ib": LOGS_ROOT / "trades",
        "binance": LOGS_ROOT / "binance" / "trades",
        "kucoin": LOGS_ROOT / "kucoin" / "trades",
    }
    
    log_dir = log_dirs.get(exchange)
    if not log_dir or not log_dir.exists():
        return []
    
    logs = []
    for f in sorted(log_dir.glob("trade_*.json")):
        with open(f) as fp:
            logs.append(json.load(fp))
    return logs


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics from equity CSV."""
    if df is None or len(df) == 0:
        return {}
    
    pnl = df["pnl_bar"]
    cum = df["cumulative_pnl"]
    
    # Assume 6 bars/day for crypto, 1 for equities
    exchange = df["exchange"].iloc[0]
    bars_per_year = 6 * 365 if exchange != "ib" else 252
    
    sr = pnl.mean() / pnl.std() * np.sqrt(bars_per_year) if pnl.std() > 0 else 0
    
    # Drawdown from cumulative PnL
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()
    
    return {
        "n_bars": len(df),
        "total_pnl": cum.iloc[-1],
        "avg_pnl_bar": pnl.mean(),
        "sharpe": sr,
        "max_dd": max_dd,
        "win_rate": (pnl > 0).sum() / len(pnl) * 100 if len(pnl) > 0 else 0,
        "avg_gmv": df["gmv"].mean(),
        "avg_positions": df["n_positions"].mean(),
    }


def print_summary(stats: dict, name: str):
    """Pretty-print summary stats."""
    if not stats:
        print(f"  {name}: No data")
        return
    
    print(f"\n  {name}")
    print(f"  {'─'*50}")
    print(f"    Bars:        {stats['n_bars']:>8d}")
    print(f"    Total PnL:   ${stats['total_pnl']:>10,.2f}")
    print(f"    Sharpe:      {stats['sharpe']:>8.2f}")
    print(f"    Max DD:      ${stats['max_dd']:>10,.2f}")
    print(f"    Win Rate:    {stats['win_rate']:>7.1f}%")
    print(f"    Avg GMV:     ${stats['avg_gmv']:>10,.0f}")
    print(f"    Avg Pos:     {stats['avg_positions']:>8.0f}")


def plot_equity_curves(exchange_filter: str | None = None):
    """Plot equity curves for all exchanges."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Multi-Exchange Alpha Portfolio — Live Equity Curves",
                 fontsize=16, fontweight="bold")
    
    all_stats = {}
    has_data = False
    
    for exchange, info in EXCHANGES.items():
        if exchange_filter and exchange != exchange_filter:
            continue
        
        df = load_equity_csv(exchange)
        if df is None or len(df) == 0:
            continue
        
        has_data = True
        stats = compute_stats(df)
        all_stats[exchange] = stats
        
        # Panel 1: Cumulative PnL
        axes[0].plot(df["timestamp"], df["cumulative_pnl"],
                     label=f"{info['name']} (SR={stats['sharpe']:.1f})",
                     color=info["color"], linewidth=1.5)
        
        # Panel 2: Bar PnL
        axes[1].bar(df["timestamp"], df["pnl_bar"],
                    alpha=0.5, color=info["color"], width=0.01)
        
        # Panel 3: GMV
        axes[2].plot(df["timestamp"], df["gmv"],
                     color=info["color"], linewidth=1, alpha=0.7)
    
    if not has_data:
        print("No equity data found. Run traders first.")
        plt.close()
        return
    
    # Combined equity (if multiple exchanges)
    if len(all_stats) > 1:
        combined = None
        for exchange in EXCHANGES:
            df = load_equity_csv(exchange)
            if df is not None and len(df) > 0:
                if combined is None:
                    combined = df[["timestamp", "cumulative_pnl"]].copy()
                    combined.columns = ["timestamp", "cum"]
                else:
                    merged = df[["timestamp", "cumulative_pnl"]].copy()
                    merged.columns = ["timestamp", "cum"]
                    combined = pd.merge_asof(
                        combined.sort_values("timestamp"),
                        merged.sort_values("timestamp"),
                        on="timestamp", suffixes=("", "_new")
                    )
                    combined["cum"] = combined["cum"].fillna(0) + combined.get("cum_new", 0).fillna(0)
                    combined = combined[["timestamp", "cum"]]
        
        if combined is not None:
            axes[0].plot(combined["timestamp"], combined["cum"],
                         label="Combined", color="white", linewidth=2, linestyle="--")
    
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_facecolor("#1a1a2e")
    
    axes[1].set_ylabel("Bar PnL ($)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#1a1a2e")
    
    axes[2].set_ylabel("GMV ($)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_facecolor("#1a1a2e")
    
    fig.set_facecolor("#0f0f23")
    for ax in axes:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    fig.suptitle("Multi-Exchange Alpha Portfolio — Live Equity Curves",
                 fontsize=16, fontweight="bold", color="white")
    
    plt.tight_layout()
    out_path = OUT_DIR / "equity_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Equity curve saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Trading Dashboard")
    parser.add_argument("--exchange", choices=["ib", "binance", "kucoin"],
                        help="Filter to a single exchange")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  MULTI-EXCHANGE ALPHA PORTFOLIO — DASHBOARD")
    print("=" * 60)
    
    for exchange, info in EXCHANGES.items():
        if args.exchange and exchange != args.exchange:
            continue
        
        df = load_equity_csv(exchange)
        stats = compute_stats(df) if df is not None else {}
        print_summary(stats, info["name"])
        
        # Recent trades
        logs = load_trade_logs(exchange)
        if logs:
            print(f"    Recent: {len(logs)} trade logs")
            last = logs[-1]
            print(f"    Last:   {last.get('timestamp', '?')}")
            print(f"    Orders: {last.get('orders', {}).get('n_orders', 0)}")
    
    plot_equity_curves(args.exchange)


if __name__ == "__main__":
    main()
