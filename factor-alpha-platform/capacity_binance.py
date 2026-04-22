"""
Estimate capacity for Binance Crypto TOP50/TOP100 portfolio.
No MOC orders — must trade limit/market orders during the bar.
Key constraint: market impact, participation rate.
"""
import pandas as pd
import numpy as np
from pathlib import Path

MATRICES_DIR = Path("data/binance_cache/matrices/4h")
close = pd.read_parquet(MATRICES_DIR / "close.parquet")
qv = pd.read_parquet(MATRICES_DIR / "quote_volume.parquet")
vol = pd.read_parquet(MATRICES_DIR / "volume.parquet")

# Build universes
adv20 = qv.rolling(120, min_periods=60).mean()
rank = adv20.rank(axis=1, ascending=False)

# Last snapshot for capacity analysis
last_adv = adv20.iloc[-1].dropna().sort_values(ascending=False)
last_close = close.iloc[-1].dropna()

print("=" * 90)
print("  BINANCE FUTURES CAPACITY ANALYSIS")
print("  Portfolio: 25 alphas, equal-weight, market-neutral")
print("  Turnover: 0.196 per 4h bar (from backtest)")
print("  Execution: Limit orders during 4h bar (no MOC available)")
print("=" * 90)

TO_PER_BAR = 0.196  # from backtest
BARS_PER_DAY = 6

for uni_name, n_top in [("TOP50", 50), ("TOP100", 100)]:
    top_n = last_adv.head(n_top)
    
    print(f"\n{'─'*90}")
    print(f"  {uni_name} Universe")
    print(f"{'─'*90}")
    
    # ADV stats (per 4h bar, in USDT)
    print(f"\n  ADV20 per 4h bar (USDT):")
    print(f"    Min:    ${top_n.min():>15,.0f}")
    print(f"    P10:    ${top_n.quantile(0.10):>15,.0f}")
    print(f"    P25:    ${top_n.quantile(0.25):>15,.0f}")
    print(f"    Median: ${top_n.median():>15,.0f}")
    print(f"    P75:    ${top_n.quantile(0.75):>15,.0f}")
    print(f"    Mean:   ${top_n.mean():>15,.0f}")
    print(f"    Max:    ${top_n.max():>15,.0f}")
    print(f"    Total:  ${top_n.sum():>15,.0f}")
    
    # Bottom 5 (capacity bottleneck)
    print(f"\n  Bottom 5 by ADV (capacity bottleneck):")
    for sym, adv in top_n.tail(5).items():
        print(f"    {sym:<20s} ADV=${adv:>12,.0f}/4h  Price=${last_close.get(sym, 0):.4f}")
    
    # Capacity estimation
    # Rule of thumb: can execute up to X% of ADV without significant impact
    # Conservative: 1% of ADV per bar
    # Moderate: 5% of ADV per bar
    # Aggressive: 10% of ADV per bar
    
    print(f"\n  Capacity Estimates (GMV that can be traded):")
    print(f"  {'Participation':>15s} {'Per-Symbol Cap':>15s} {'Portfolio Cap':>15s} {'Daily Traded':>15s}")
    print(f"  {'Rate':>15s} {'(bottleneck)':>15s} {'(total GMV)':>15s} {'(notional)':>15s}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for pct_name, pct in [("0.5% (ultra safe)", 0.005), ("1% (conservative)", 0.01), 
                           ("2% (moderate)", 0.02), ("5% (aggressive)", 0.05),
                           ("10% (very aggr.)", 0.10)]:
        # Per-symbol max trade = pct * ADV_per_bar
        # Trade per bar per symbol ≈ GMV / n_symbols * TO_PER_BAR
        # So: GMV / n * TO = pct * ADV_min
        # GMV = pct * ADV_min * n / TO
        
        bottleneck_adv = top_n.min()
        per_sym_cap = pct * bottleneck_adv  # max notional per bar
        # Each symbol's trade per bar = (GMV / n_syms) * turnover
        # GMV = per_sym_cap * n_syms / turnover
        port_cap_bottleneck = per_sym_cap * n_top / TO_PER_BAR
        
        # But more realistically, use harmonic mean of ADVs
        # since positions aren't equal weight
        # Use P10 ADV as practical bottleneck
        p10_adv = top_n.quantile(0.10)
        port_cap_p10 = pct * p10_adv * n_top / TO_PER_BAR
        
        # Daily traded = GMV * TO * bars/day
        daily_traded = port_cap_p10 * TO_PER_BAR * BARS_PER_DAY
        
        print(f"  {pct_name:>17s} ${per_sym_cap:>13,.0f} ${port_cap_p10:>13,.0f} ${daily_traded:>13,.0f}")
    
    # Impact model
    print(f"\n  Market Impact Model (Kyle's lambda):")
    print(f"  Assuming linear impact: slippage_bps = k * sqrt(trade_notional / ADV)")
    print(f"  With k = 10 bps (conservative for crypto):")
    
    for gmv in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]:
        # Per symbol trade per bar
        trade_per_sym = gmv / n_top * TO_PER_BAR
        
        # Impact on bottleneck
        bottom_adv = top_n.quantile(0.10)
        participation = trade_per_sym / bottom_adv
        impact_bps = 10 * np.sqrt(participation)  # Kyle's lambda
        
        # Median symbol
        med_adv = top_n.median()
        participation_med = trade_per_sym / med_adv
        impact_med = 10 * np.sqrt(participation_med)
        
        # Total cost = taker + tick + impact
        total_cost_bps = 1.7 + 2.8 + impact_bps  # taker + median tick + impact
        
        # Annual drag from impact alone
        impact_drag = impact_bps / 10000 * TO_PER_BAR * BARS_PER_DAY * 365
        total_drag = total_cost_bps / 10000 * TO_PER_BAR * BARS_PER_DAY * 365
        
        # Net return (backtest gross was ~370%)
        gross_ret = 3.70  # 370% annual
        net_ret = gross_ret - total_drag
        
        ok = "✓" if net_ret > 1.0 else ("⚠" if net_ret > 0 else "✗")
        
        print(f"    GMV ${gmv:>11,}: partic={participation*100:.2f}% impact={impact_bps:.1f}bps "
              f"total={total_cost_bps:.1f}bps/trade drag={total_drag*100:.1f}%/yr "
              f"net={net_ret*100:.0f}%/yr {ok}")

# Execution considerations
print(f"\n{'='*90}")
print(f"  EXECUTION CONSIDERATIONS")
print(f"{'='*90}")
print(f"""
  Unlike IB equities with MOC orders, Binance execution requires:
  
  1. TWAP/VWAP: Spread each 4h bar's trades over the bar duration
     - 4 hours = 240 minutes of execution window
     - Can slice into 60 child orders (1 per 4 min)
     - Reduces impact significantly vs market-on-close
  
  2. Limit orders at mid: Post passive orders and wait for fills
     - Earn maker rebate (-0.5 bps at VIP9!) instead of paying taker
     - Risk: may not fill, requiring aggressive take at bar end
  
  3. Iceberg orders: Binance supports native iceberg
     - Hide large orders behind small visible clips
  
  4. Funding rate drag: Not modeled in backtest
     - 8h funding ≈ ±0.01% per period
     - For market-neutral: long and short cancel out
     - Net drag is small if portfolio is balanced
  
  5. Liquidation risk: Perps can be liquidated
     - Use low leverage (1-2x) or isolated margin per position
     - Max position size limited by margin requirements
""")
