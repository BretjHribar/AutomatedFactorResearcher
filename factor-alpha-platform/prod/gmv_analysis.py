"""
Analyze order distribution and simulate minimum trade thresholds.

If we skip orders where |diff_shares| is tiny (below a dollar threshold),
we save commissions but introduce tracking error vs the ideal signal.

This uses today's actual trade log to model the impact.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load today's trade log
log_path = Path("prod/logs/trades/trade_2026-04-21.json")
with open(log_path) as f:
    trade_log = json.load(f)

target = trade_log["target_portfolio"]  # {ticker: shares}
diffs = trade_log["order_diffs"]        # {ticker: diff_shares}

# We need prices to compute dollar values
# Load from the matrices
import sys
sys.path.insert(0, ".")
import eval_alpha_ib
eval_alpha_ib.UNIVERSE = "TOP2000TOP3000"
eval_alpha_ib.NEUTRALIZE = "market"
matrices, _, _ = eval_alpha_ib.load_data("full")
last_close = matrices["close"].iloc[-1]

# Compute dollar value of each order
order_dollars = {}
for sym, shares in diffs.items():
    price = last_close.get(sym, 0)
    if price > 0:
        order_dollars[sym] = abs(shares) * price

# Sort by dollar value
sorted_orders = sorted(order_dollars.items(), key=lambda x: x[1])

print("=" * 80)
print("  ORDER SIZE DISTRIBUTION (Today's 208 orders)")
print("=" * 80)

# Distribution
dollar_vals = np.array(list(order_dollars.values()))
print(f"\n  Total orders: {len(dollar_vals)}")
print(f"  Total $ traded: ${dollar_vals.sum():,.0f}")
print(f"  Mean order: ${dollar_vals.mean():,.0f}")
print(f"  Median order: ${np.median(dollar_vals):,.0f}")
print(f"  Min order: ${dollar_vals.min():,.0f}")
print(f"  Max order: ${dollar_vals.max():,.0f}")

# Percentile breakdown
print(f"\n  Percentile breakdown:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{p}: ${np.percentile(dollar_vals, p):,.0f}")

# Histogram
print(f"\n  Order size distribution:")
buckets = [0, 100, 250, 500, 1000, 2000, 3000, 5000, 10000]
for i in range(len(buckets) - 1):
    lo, hi = buckets[i], buckets[i+1]
    count = ((dollar_vals >= lo) & (dollar_vals < hi)).sum()
    print(f"    ${lo:>5,} - ${hi:>5,}: {count:3d} orders ({count/len(dollar_vals)*100:.0f}%)")
count = (dollar_vals >= buckets[-1]).sum()
print(f"    ${buckets[-1]:>5,}+      : {count:3d} orders ({count/len(dollar_vals)*100:.0f}%)")

# ─── Simulate Minimum Trade Thresholds ────────────────────────────────
print(f"\n{'='*80}")
print(f"  MINIMUM TRADE THRESHOLD ANALYSIS")
print(f"  (Skip orders below threshold → save commissions, lose some alpha)")
print(f"{'='*80}")

IBKR_MIN_PER_ORDER = 0.35
IBKR_PER_SHARE = 0.0035
MEDIAN_PRICE = 22.43

# For each GMV level and each threshold, compute impact
GMV_LEVELS = [100_000, 200_000, 500_000]
THRESHOLDS = [0, 50, 100, 200, 500, 1000]

for gmv in GMV_LEVELS:
    print(f"\n  --- GMV = ${gmv:,} ---")
    print(f"  {'Threshold':>10} {'Orders':>8} {'Saved':>8} {'$ Traded':>10} {'Fee/Day':>10} {'Fee/Yr':>10} {'Fee%':>7} {'Track Err':>10}")
    print(f"  {'-'*76}")
    
    scale = gmv / 500_000  # Scale from $500k baseline
    
    for thresh in THRESHOLDS:
        # Scale order values for this GMV
        scaled_vals = dollar_vals * scale
        
        # Filter: keep only orders above threshold
        mask = scaled_vals >= thresh
        kept_orders = mask.sum()
        kept_vals = scaled_vals[mask]
        skipped_vals = scaled_vals[~mask]
        
        # Dollar traded
        total_traded = kept_vals.sum()
        
        # Commission for kept orders
        # Each order: max($0.35, shares * $0.0035)
        daily_commission = 0
        for val in kept_vals:
            shares = val / MEDIAN_PRICE
            cost = max(IBKR_MIN_PER_ORDER, shares * IBKR_PER_SHARE)
            daily_commission += cost
        
        annual_commission = daily_commission * 252
        fee_pct = annual_commission / gmv * 100
        
        # Tracking error: dollar value of skipped orders / total target
        total_target = scaled_vals.sum()
        tracking_err = skipped_vals.sum() / total_target * 100 if total_target > 0 else 0
        
        saved = len(dollar_vals) - kept_orders
        
        print(f"  ${thresh:>9,} {kept_orders:>7,} {saved:>7,} ${total_traded:>9,.0f} "
              f"${daily_commission:>8,.1f} ${annual_commission:>9,.0f} {fee_pct:>6.1f}% {tracking_err:>9.1f}%")
    
    # Net return estimate for best threshold
    print(f"\n  Best threshold estimate:")
    gross_return_pct = 76.2  # ~76% gross (before any fees)
    
    for thresh in [0, 200, 500]:
        scaled_vals_t = dollar_vals * scale
        mask_t = scaled_vals_t >= thresh
        daily_comm = 0
        for val in scaled_vals_t[mask_t]:
            shares = val / MEDIAN_PRICE
            daily_comm += max(IBKR_MIN_PER_ORDER, shares * IBKR_PER_SHARE)
        annual_comm = daily_comm * 252
        fee_drag = annual_comm / gmv * 100
        track_err = scaled_vals_t[~mask_t].sum() / scaled_vals_t.sum() * 100
        # Tracking error reduces alpha proportionally (conservative estimate)
        net_ret = gross_return_pct - fee_drag - (track_err * 0.5)
        net_dollar = gmv * net_ret / 100
        print(f"    Thresh=${thresh}: fee={fee_drag:.1f}% + track_err={track_err:.1f}% -> "
              f"net ~{net_ret:.1f}% = ${net_dollar:,.0f}/yr")
