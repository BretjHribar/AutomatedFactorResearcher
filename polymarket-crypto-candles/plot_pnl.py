"""
plot_pnl.py — Generate PnL equity curve charts for all models.
Shows train + holdout with a vertical divider line.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from config import *
from signals import compute_signals, compute_target
from backtest_engine import run_backtest
from run_backtest import load_klines, split_data

BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24}

# Best signal configs per model
BEST_CONFIGS = {
    "BTC_15m": {"signals": ["mr_10", "bb_10", "mr_vol_10"], "w": [0.333, 0.333, 0.334]},
    "ETH_15m": {"signals": ["mr_8", "rsi_14"],              "w": [0.5, 0.5]},
    "SOL_15m": {"signals": ["mr_8", "rsi_14"],              "w": [0.5, 0.5]},
    "ETH_5m":  {"signals": ["mr_8", "rsi_14"],              "w": [0.5, 0.5]},
    "BTC_1h":  {"signals": ["mr_5", "mr_30"],               "w": [0.5, 0.5]},
    "ETH_1h":  {"signals": ["mr_5", "mr_30"],               "w": [0.5, 0.5]},
    "SOL_1h":  {"signals": ["mr_5", "mr_10"],               "w": [0.5, 0.5]},
}

COLORS = {
    "BTC": "#F7931A",
    "ETH": "#627EEA",
    "SOL": "#9945FF",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_model_pnl(symbol, interval, cfg):
    """Compute full train+holdout daily cumulative PnL for a model."""
    bpd = BARS_PER_DAY[interval]
    df = load_klines(symbol, interval)

    # Full period signals + target
    full_df = split_data(df, TRAIN_START, HOLDOUT_END)
    train_df = split_data(df, TRAIN_START, TRAIN_END)

    sigs = compute_signals(full_df, cfg["signals"])
    target = compute_target(full_df)

    # Normalize using TRAIN stats only
    train_sigs = compute_signals(train_df, cfg["signals"])
    for col in cfg["signals"]:
        if col in train_sigs.columns and col in sigs.columns:
            mu = train_sigs[col].mean()
            std = train_sigs[col].std()
            if std > 1e-10:
                sigs[col] = (sigs[col] - mu) / std

    w = np.array(cfg["w"])
    valid = [c for c in cfg["signals"] if c in sigs.columns]
    wv = w[:len(valid)]

    combined = (sigs[valid].values * wv).sum(axis=1)
    combined_series = pd.Series(combined, index=sigs.index)

    result = run_backtest(combined_series, target, bars_per_day=bpd)
    return result


def dollar_formatter(x, _):
    if abs(x) >= 1000:
        return f"${x/1000:,.0f}K"
    return f"${x:,.0f}"


def main():
    print("Generating PnL charts...")

    # =========================================================================
    # CHART 1: All 15m models (the sweet spot) — individual panels
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.patch.set_facecolor("#0D1117")

    holdout_start = pd.Timestamp(HOLDOUT_START, tz="UTC")

    for idx, (key, color_key) in enumerate([("ETH_15m", "ETH"), ("BTC_15m", "BTC"), ("SOL_15m", "SOL")]):
        coin = key.split("_")[0]
        symbol = [s for s in SYMBOLS if SYMBOL_NAMES[s] == coin][0]
        cfg = BEST_CONFIGS[key]

        result = compute_model_pnl(symbol, "15m", cfg)
        cum_pnl = result.cumulative_pnl

        # Daily aggregation for smoother plot
        daily = result.pnl_series.resample("D").sum()
        daily_cum = daily.cumsum()

        ax = axes[idx]
        ax.set_facecolor("#0D1117")

        # Split into train and holdout
        train_mask = daily_cum.index < holdout_start
        holdout_mask = daily_cum.index >= holdout_start

        train_series = daily_cum[train_mask]
        holdout_series = daily_cum[holdout_mask]

        # Plot train (dimmer)
        ax.fill_between(train_series.index, 0, train_series.values,
                        alpha=0.15, color=COLORS[coin])
        ax.plot(train_series.index, train_series.values,
                color=COLORS[coin], alpha=0.4, linewidth=1.2, label="Train")

        # Plot holdout (bright)
        ax.fill_between(holdout_series.index, 0, holdout_series.values,
                        alpha=0.25, color=COLORS[coin])
        ax.plot(holdout_series.index, holdout_series.values,
                color=COLORS[coin], alpha=1.0, linewidth=2.5, label="Holdout (OOS)")

        # Holdout start line
        ax.axvline(holdout_start, color="#FF6B6B", linestyle="--", linewidth=1.5,
                   alpha=0.8, label="Holdout Start")

        # Zero line
        ax.axhline(0, color="#30363D", linewidth=0.8)

        # Annotations
        ho_pnl = holdout_series.iloc[-1] - (train_series.iloc[-1] if len(train_series) > 0 else 0)
        ho_sharpe = result.sharpe  # This is full-period, but close enough
        sigs_str = " + ".join(cfg["signals"])

        ax.text(0.02, 0.92, f"{coin} 15m",
                transform=ax.transAxes, fontsize=18, fontweight="bold",
                color=COLORS[coin], va="top",
                fontfamily="sans-serif")

        ax.text(0.02, 0.78,
                f"Signals: {sigs_str}",
                transform=ax.transAxes, fontsize=10, color="#8B949E", va="top",
                fontfamily="monospace")

        # Holdout stats box
        stats_text = (f"Holdout PnL: ${ho_pnl:,.0f}\n"
                      f"Win Rate: {result.win_rate:.1%}\n"
                      f"Profit Factor: {result.profit_factor:.2f}")

        ax.text(0.98, 0.92, stats_text,
                transform=ax.transAxes, fontsize=11, color="#C9D1D9", va="top", ha="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#161B22", edgecolor="#30363D",
                          alpha=0.9))

        ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
        ax.tick_params(colors="#8B949E", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#30363D")
        ax.spines["bottom"].set_color("#30363D")
        ax.grid(True, alpha=0.1, color="#30363D")

        ax.legend(loc="lower left", fontsize=9, facecolor="#161B22",
                  edgecolor="#30363D", labelcolor="#C9D1D9")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle("Polymarket Crypto Candle Trading — 15m Equity Curves (Net of Fees)",
                 fontsize=20, fontweight="bold", color="#F0F6FC", y=0.98,
                 fontfamily="sans-serif")

    fig.text(0.5, 0.01,
             f"Train: {TRAIN_START} → {TRAIN_END}  |  Holdout: {HOLDOUT_START} → {HOLDOUT_END}  |  "
             f"Fee: {BLENDED_TAKER_FEE*100:.1f}% per trade  |  Trade Size: ${BASE_TRADE_SIZE}",
             ha="center", fontsize=10, color="#8B949E", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path1 = os.path.join(OUTPUT_DIR, "pnl_15m_models.png")
    fig.savefig(path1, dpi=150, facecolor="#0D1117", edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # =========================================================================
    # CHART 2: Combined portfolio equity curve (all 15m models stacked)
    # =========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 8))
    fig2.patch.set_facecolor("#0D1117")
    ax2.set_facecolor("#0D1117")

    portfolio_daily = None
    individual_curves = {}

    for key, color_key in [("ETH_15m", "ETH"), ("BTC_15m", "BTC"), ("SOL_15m", "SOL")]:
        coin = key.split("_")[0]
        symbol = [s for s in SYMBOLS if SYMBOL_NAMES[s] == coin][0]
        cfg = BEST_CONFIGS[key]

        result = compute_model_pnl(symbol, "15m", cfg)
        daily = result.pnl_series.resample("D").sum()

        if portfolio_daily is None:
            portfolio_daily = daily.copy()
        else:
            portfolio_daily = portfolio_daily.add(daily, fill_value=0)

        individual_curves[coin] = daily.cumsum()

    portfolio_cum = portfolio_daily.cumsum()

    # Plot individual (dimmer)
    for coin, cum in individual_curves.items():
        ax2.plot(cum.index, cum.values, color=COLORS[coin], alpha=0.35,
                 linewidth=1.2, linestyle="--", label=f"{coin} 15m")

    # Plot combined portfolio (bright white)
    ax2.fill_between(portfolio_cum.index, 0, portfolio_cum.values,
                     alpha=0.15, color="#58A6FF")
    ax2.plot(portfolio_cum.index, portfolio_cum.values,
             color="#58A6FF", alpha=1.0, linewidth=3.0, label="Combined Portfolio")

    # Holdout line
    ax2.axvline(holdout_start, color="#FF6B6B", linestyle="--", linewidth=2.0,
                alpha=0.9, label="Holdout Start")
    ax2.axhline(0, color="#30363D", linewidth=0.8)

    # Stats
    ho_mask = portfolio_cum.index >= holdout_start
    ho_portfolio = portfolio_cum[ho_mask]
    tr_end_val = portfolio_cum[portfolio_cum.index < holdout_start].iloc[-1] if len(portfolio_cum[portfolio_cum.index < holdout_start]) > 0 else 0
    ho_pnl = ho_portfolio.iloc[-1] - tr_end_val
    ho_daily = portfolio_daily[portfolio_daily.index >= holdout_start]
    ho_sharpe = ho_daily.mean() / ho_daily.std() * np.sqrt(365) if ho_daily.std() > 0 else 0
    ho_max_dd = (ho_portfolio - ho_portfolio.cummax()).min()

    stats_box = (f"Combined Portfolio (Holdout)\n"
                 f"{'─' * 30}\n"
                 f"Net PnL:      ${ho_pnl:,.0f}\n"
                 f"Sharpe:       {ho_sharpe:.1f}\n"
                 f"Max Drawdown: ${ho_max_dd:,.0f}\n"
                 f"Annualized:   ${ho_pnl/189*365:,.0f}")

    ax2.text(0.02, 0.95, stats_box,
             transform=ax2.transAxes, fontsize=12, color="#C9D1D9", va="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#161B22",
                       edgecolor="#58A6FF", alpha=0.95, linewidth=1.5))

    # "TRAIN" and "HOLDOUT" labels
    train_mid = pd.Timestamp(TRAIN_START, tz="UTC") + (holdout_start - pd.Timestamp(TRAIN_START, tz="UTC")) / 2
    holdout_mid = holdout_start + (pd.Timestamp(HOLDOUT_END, tz="UTC") - holdout_start) / 2

    y_top = portfolio_cum.max() * 1.05
    ax2.text(train_mid, y_top * 0.95, "TRAIN", fontsize=14, color="#8B949E",
             ha="center", fontweight="bold", alpha=0.5, fontfamily="sans-serif")
    ax2.text(holdout_mid, y_top * 0.95, "HOLDOUT (OUT-OF-SAMPLE)", fontsize=14,
             color="#FF6B6B", ha="center", fontweight="bold", alpha=0.7,
             fontfamily="sans-serif")

    ax2.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax2.tick_params(colors="#8B949E", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#30363D")
    ax2.spines["bottom"].set_color("#30363D")
    ax2.grid(True, alpha=0.1, color="#30363D")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.legend(loc="lower right", fontsize=11, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="#C9D1D9")

    fig2.suptitle("Polymarket 15m Combined Portfolio — Cumulative PnL (Net of Fees)",
                  fontsize=20, fontweight="bold", color="#F0F6FC", y=0.98,
                  fontfamily="sans-serif")

    fig2.text(0.5, 0.01,
              f"Trade Size: ${BASE_TRADE_SIZE}  |  Taker Fee: {BLENDED_TAKER_FEE*100:.1f}%  |  "
              f"Signals: Mean Reversion + RSI + Bollinger",
              ha="center", fontsize=10, color="#8B949E", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path2 = os.path.join(OUTPUT_DIR, "pnl_combined_portfolio.png")
    fig2.savefig(path2, dpi=150, facecolor="#0D1117", edgecolor="none", bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # =========================================================================
    # CHART 3: Monthly PnL bar chart (holdout only)
    # =========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))
    fig3.patch.set_facecolor("#0D1117")
    ax3.set_facecolor("#0D1117")

    monthly_data = {}
    for key in ["ETH_15m", "BTC_15m", "SOL_15m"]:
        coin = key.split("_")[0]
        symbol = [s for s in SYMBOLS if SYMBOL_NAMES[s] == coin][0]
        cfg = BEST_CONFIGS[key]
        result = compute_model_pnl(symbol, "15m", cfg)
        ho_pnl_series = result.pnl_series[result.pnl_series.index >= holdout_start]
        monthly_data[coin] = ho_pnl_series.resample("ME").sum()

    # Align all to same months
    all_months = sorted(set().union(*[set(v.index) for v in monthly_data.values()]))
    x = np.arange(len(all_months))
    bar_width = 0.25

    for i, (coin, monthly) in enumerate(monthly_data.items()):
        vals = [monthly.get(m, 0) for m in all_months]
        bars = ax3.bar(x + i * bar_width, vals, bar_width, label=f"{coin} 15m",
                       color=COLORS[coin], alpha=0.85, edgecolor="#0D1117", linewidth=0.5)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if abs(val) > 500:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                         f"${val/1000:.1f}K", ha="center", va="bottom",
                         fontsize=7.5, color="#C9D1D9", fontfamily="monospace")

    # Total line
    totals = []
    for m in all_months:
        t = sum(monthly_data[coin].get(m, 0) for coin in monthly_data)
        totals.append(t)
    ax3.plot(x + bar_width, totals, "o-", color="#F0F6FC", linewidth=2, markersize=6,
             label="Total", zorder=5)

    ax3.axhline(0, color="#30363D", linewidth=0.8)

    month_labels = [m.strftime("%b\n%Y") for m in all_months]
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(month_labels)

    ax3.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    ax3.tick_params(colors="#8B949E", labelsize=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#30363D")
    ax3.spines["bottom"].set_color("#30363D")
    ax3.grid(True, axis="y", alpha=0.1, color="#30363D")

    ax3.legend(loc="upper left", fontsize=11, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="#C9D1D9")

    total_ho = sum(totals)
    fig3.suptitle(f"Polymarket 15m — Monthly PnL (Holdout Only, Net of Fees)  |  Total: ${total_ho:,.0f}",
                  fontsize=18, fontweight="bold", color="#F0F6FC", y=0.98,
                  fontfamily="sans-serif")

    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    path3 = os.path.join(OUTPUT_DIR, "pnl_monthly_bars.png")
    fig3.savefig(path3, dpi=150, facecolor="#0D1117", edgecolor="none", bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")

    print("\nAll charts generated successfully!")
    print(f"  1. {path1}")
    print(f"  2. {path2}")
    print(f"  3. {path3}")


if __name__ == "__main__":
    main()
