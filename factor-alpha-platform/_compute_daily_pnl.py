"""Compute exact daily PnL from the Billions backtest at $248k GMV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import sqlite3

# Reuse the pipeline to get exact returns
from run_ib_portfolio import (
    load_full_data, load_alphas, load_alpha_signals,
    combiner_billions, process_signal, simulate,
    UNIVERSE, BOOKSIZE as BACKTEST_BOOKSIZE
)

GMV = 248_189  # From dry-run

matrices, universe_df, classifications, valid_tickers = load_full_data()
close_df = matrices["close"]
returns_df = matrices["returns"]

alphas = load_alphas()
alpha_signals = load_alpha_signals(alphas, matrices)
combined = combiner_billions(alpha_signals, matrices, universe_df, returns_df)

# Process the combined signal
processed = process_signal(combined, universe_df=universe_df)

# Simulate at BACKTEST booksize ($20M)
r0 = simulate(processed, returns_df, close_df, universe_df, fees_bps=0.0)
r_fee = simulate(processed, returns_df, close_df, universe_df, fees_bps=0.50)

# Test period: 2024-07-01 onward
test_mask = close_df.index >= "2024-07-01"
test_dates = close_df.index[test_mask]
n_test_days = len(test_dates)

# Get daily PnL arrays
pnl_0bps = np.array(r0.daily_pnl)
pnl_fee  = np.array(r_fee.daily_pnl)

# Test-period PnL (at $20M booksize)
test_pnl_0bps = pnl_0bps[test_mask[:len(pnl_0bps)]]
test_pnl_fee  = pnl_fee[test_mask[:len(pnl_fee)]]

mean_daily_pnl_20M_0bps = test_pnl_0bps.mean()
mean_daily_pnl_20M_fee  = test_pnl_fee.mean()
std_daily_pnl_20M = test_pnl_fee.std()

# Now scale to $248k
scale = GMV / BACKTEST_BOOKSIZE
mean_daily_248k_0bps = mean_daily_pnl_20M_0bps * scale
mean_daily_248k_fee  = mean_daily_pnl_20M_fee * scale
std_daily_248k = std_daily_pnl_20M * scale

ann_ret_248k = mean_daily_248k_fee * 252
sr_test = (mean_daily_248k_fee / std_daily_248k) * np.sqrt(252) if std_daily_248k > 0 else 0

print("=" * 60)
print("  DAILY PnL ESTIMATE AT $248k GMV")
print("=" * 60)
print()
print(f"  Backtest booksize:     ${BACKTEST_BOOKSIZE:,.0f}")
print(f"  Live GMV (from dry-run): ${GMV:,.0f}")
print(f"  Scale factor:          {scale:.6f}")
print()
print(f"  Test period:           2024-07-01 to {test_dates[-1].date()}")
print(f"  Test trading days:     {n_test_days}")
print()
print("  --- At $20M booksize (backtest) ---")
print(f"  Mean daily PnL (0 bps):  ${mean_daily_pnl_20M_0bps:,.0f}")
print(f"  Mean daily PnL (0.5bps): ${mean_daily_pnl_20M_fee:,.0f}")
print(f"  Daily vol:               ${std_daily_pnl_20M:,.0f}")
print()
print("  --- Scaled to $248k GMV ---")
print(f"  Mean daily PnL (0 bps):  ${mean_daily_248k_0bps:,.2f}")
print(f"  Mean daily PnL (0.5bps): ${mean_daily_248k_fee:,.2f}")
print(f"  Daily vol:               ${std_daily_248k:,.2f}")
print(f"  Ann return:              ${ann_ret_248k:,.0f} ({ann_ret_248k/GMV*100:.1f}%)")
print(f"  Test Sharpe:             {sr_test:.2f}")
print()
print(f"  Monthly PnL (0.5bps):    ${mean_daily_248k_fee * 21:,.0f}")
print(f"  Annual PnL (0.5bps):     ${ann_ret_248k:,.0f}")
print()
print(f"  Daily commission (~0.5bps):")
print(f"    ${(mean_daily_248k_0bps - mean_daily_248k_fee):,.2f}/day")
print(f"    ${(mean_daily_248k_0bps - mean_daily_248k_fee)*252:,.0f}/year")
