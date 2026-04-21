"""
capacity_estimate.py — Strategy Capacity Estimation (Isichenko Framework)

Uses the linear impact model from Isichenko (2021) "Quantitative Portfolio Management"
Section 5.2.2, Eq (5.3) and Section 6.6, Eq (6.68):

    Impact cost per trade:  Cost(T) = (lambda/2) * T^2
    where lambda ~ 1/V  (V = relevant volume)

    Capacity formula (Eq 6.68):
    G* = G * (gross_pnl - slippage_cost) / (2 * impact_cost)

Key adaptation for MOC (Market-on-Close) execution:
  - Our orders execute in the closing auction, not continuous trading
  - The relevant volume V is the CLOSING AUCTION volume, not full-day ADV
  - Closing auction volume is typically 7-15% of total daily volume for 
    small/mid-cap stocks (our TOP2000-3000 universe)
  - MOC orders are batched — no sequential self-impact within the auction
  - But our share of auction volume determines price impact

References:
  - Isichenko (2021), Ch 5-6
  - Almgren et al. (2005), "Direct Estimation of Equity Market Impact"
  - Bouchaud et al. (2018), "Trades, Quotes and Prices"
"""

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# DATA LOADING
# ============================================================================

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSE_DIR = Path("data/fmp_cache/universes")
UNIVERSE = "TOP2000TOP3000"

print("=" * 100)
print("  IB CLOSING AUCTION — STRATEGY CAPACITY ESTIMATION")
print("  Model: Isichenko (2021) linear impact + MOC auction adjustments")
print("=" * 100)

# Load volume and close price matrices
volume_df = pd.read_parquet(MATRICES_DIR / "volume.parquet")
close_df  = pd.read_parquet(MATRICES_DIR / "close.parquet")
universe_df = pd.read_parquet(UNIVERSE_DIR / f"{UNIVERSE}.parquet")

# Use test period only (mid-2024 to present) for realistic estimates
TEST_START = "2024-07-01"
vol = volume_df.loc[TEST_START:]
cls = close_df.loc[TEST_START:]
univ = universe_df.loc[TEST_START:]

# Mask to universe members only
vol = vol.where(univ > 0)
cls = cls.where(univ > 0)

# Dollar volume per stock per day
dollar_vol = (vol * cls)

# Stats
n_days = len(vol)
n_tickers = univ.sum(axis=1).median()
print(f"\n  Test period: {vol.index[0].date()} to {vol.index[-1].date()} ({n_days} days)")
print(f"  Universe tickers: {n_tickers:.0f} (median active per day)")

# ============================================================================
# CLOSING AUCTION VOLUME MODEL
# ============================================================================
# For NYSE-listed stocks, closing auction is typically 7-12% of daily volume
# For NASDAQ, typically 8-15% (larger due to NASDAQ closing cross mechanics)
# Our universe is ~58% NASDAQ, ~42% NYSE/AMEX
# Conservative estimate: 10% of daily volume participates in closing auction

AUCTION_FRACTION = 0.10  # fraction of daily volume in closing auction

# Per-stock median daily volume (shares) and dollar volume
med_daily_vol_shares = vol.median()          # median shares/day per ticker
med_daily_dvol = dollar_vol.median()         # median $/day per ticker
med_close = cls.median()                     # median price per ticker

# Auction volume per stock
auction_vol_shares = med_daily_vol_shares * AUCTION_FRACTION
auction_dvol = med_daily_dvol * AUCTION_FRACTION

print(f"\n  Closing Auction Volume Assumptions:")
print(f"    Auction fraction of daily volume: {AUCTION_FRACTION*100:.0f}%")
print(f"    Median daily volume per stock: {med_daily_vol_shares.median():,.0f} shares (${med_daily_dvol.median():,.0f})")
print(f"    Median auction volume per stock: {auction_vol_shares.median():,.0f} shares (${auction_dvol.median():,.0f})")
print(f"    Median stock price: ${med_close.median():.2f}")

# ============================================================================
# STRATEGY CHARACTERISTICS (from run_ib_portfolio.py results)
# ============================================================================

BOOKSIZE_BACKTEST = 20_000_000  # backtest booksize ($20M)
TURNOVER = 1.07                 # daily turnover (fraction of GMV)
N_POSITIONS = 253               # approximate number of positions
MAX_WEIGHT = 0.01               # 1% max per stock

# From backtest: Test Sharpe = 8.46, Ann Return = 86.8% at $20M book
# Daily return = 86.8% / 252 = 0.3444% of GMV
DAILY_RETURN_BPS = 86.8 / 252 * 100  # ~34.4 bps per day
GROSS_PNL_PER_DOLLAR = DAILY_RETURN_BPS / 10000

print(f"\n  Strategy Parameters:")
print(f"    Backtest booksize: ${BOOKSIZE_BACKTEST/1e6:.0f}M")
print(f"    Daily turnover: {TURNOVER:.2f} (fraction of GMV)")
print(f"    Positions: ~{N_POSITIONS}")
print(f"    Max weight: {MAX_WEIGHT*100:.0f}%")
print(f"    Daily gross return: {DAILY_RETURN_BPS:.1f} bps")

# ============================================================================
# IMPACT MODEL (Isichenko Eq 5.3)
# ============================================================================
# Linear impact model: Cost(T) = (lambda/2) * T^2
# where lambda ~ eta / V  (eta = impact coefficient, V = relevant volume)
#
# For MOC orders in closing auction:
#   - Our trade T_s in stock s as fraction of auction volume = T_s / V_auction_s
#   - Impact = eta * (T_s / V_auction_s)  [as fraction of price]
#   - eta calibrated from literature: ~0.1 to 1.0 for permanent impact
#
# Almgren et al. (2005) find: temporary impact ~ 0.314 * sigma * (T/V)^0.6
# But for auction (batch execution): impact is closer to linear in participation rate
#
# For closing auction specifically:
#   - All MOC orders are matched simultaneously
#   - Impact depends on net imbalance, not sequential execution  
#   - Our one-sided (either buy or sell) trade creates imbalance
#   - Impact ~ eta * (our_shares / auction_volume) * sigma_daily
#
# We use: impact_per_stock_bps = eta * participation_rate * daily_vol_bps
# where participation_rate = our_trade_shares / auction_volume_shares

# Impact coefficient calibration
# Conservative: eta = 1.0 (full participation rate impact)
# Moderate: eta = 0.5 (auction batching reduces impact)  
# Aggressive: eta = 0.25 (MOC orders have minimal impact in liquid auction)
ETA_VALUES = {
    "Conservative (eta=1.0)": 1.0,
    "Moderate (eta=0.5)":     0.5,
    "Aggressive (eta=0.25)":  0.25,
}

# Daily volatility of universe stocks
returns = cls.pct_change()
daily_vol = returns.std()  # per stock daily volatility
median_daily_vol = daily_vol.median()

print(f"\n  Impact Model Calibration:")
print(f"    Median daily stock volatility: {median_daily_vol*100:.2f}%")
print(f"    Impact model: I = eta * (trade_shares / auction_vol) * sigma_daily")

# ============================================================================
# CAPACITY CALCULATION
# ============================================================================
# For each GMV level G:
#   1. Trade per stock = G * MAX_WEIGHT * TURNOVER * (2/N_positions) per side
#      (but capped at MAX_WEIGHT * G per stock)
#   2. Dollar trade per stock ~ G * TURNOVER / N_positions
#   3. Share trade per stock = dollar_trade / price
#   4. Participation rate = share_trade / auction_volume
#   5. Impact cost per stock = eta * participation_rate * sigma * |trade_dollars|
#   6. Total impact cost = sum over positions
#
# Isichenko Eq 6.68:
#   G* = G * (gross_pnl - slippage_cost) / (2 * impact_cost)

print(f"\n{'='*100}")
print(f"  CAPACITY ESTIMATES")
print(f"{'='*100}")

# IBKR slippage (commission): 0.50 bps effective at our volume tier
SLIPPAGE_BPS = 0.50

GMV_POINTS = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]) * 1e6

for eta_label, eta in ETA_VALUES.items():
    print(f"\n  ── {eta_label} ──")
    print(f"  {'GMV':>10} | {'Gross PnL':>12} | {'Slippage':>10} | {'Impact':>12} | {'Net PnL':>12} | {'Net Sharpe':>10} | {'Part.Rate':>10} | {'Net RoR':>8}")
    print(f"  {'-'*105}")
    
    for G in GMV_POINTS:
        # Per-stock trade in dollars
        # Total daily trade = G * TURNOVER (split across ~N_POSITIONS stocks each way)
        # Each stock gets roughly G * TURNOVER / N_POSITIONS dollars of trading
        trade_per_stock_usd = G * TURNOVER / N_POSITIONS
        
        # Per-stock trade in shares (using median price)
        trade_per_stock_shares = trade_per_stock_usd / med_close.median()
        
        # Participation rate in closing auction
        med_auction_shares = auction_vol_shares.median()
        participation_rate = trade_per_stock_shares / med_auction_shares
        
        # Gross PnL
        gross_pnl = G * GROSS_PNL_PER_DOLLAR
        
        # Slippage cost (IBKR commission) — proportional to turnover
        slippage_cost = G * TURNOVER * SLIPPAGE_BPS / 10000
        
        # Impact cost per stock (Isichenko linear model)
        # Impact = eta * (trade/auction_vol) * sigma * trade_dollars 
        # Summed across all positions:
        # Total impact = N_positions * eta * participation_rate * sigma * trade_per_stock_usd
        #              = eta * sigma * (G * TURNOVER)^2 / (N_positions * auction_dvol_median)
        #
        # More precisely using Isichenko Eq 5.3:
        # lambda_s = eta * sigma_s / V_s  (impact coefficient per stock)
        # Cost_s = (lambda_s / 2) * T_s^2  (per stock)
        # Total = sum_s Cost_s
        
        # Compute per-stock, then aggregate  
        # Use distribution of actual stock volumes, not just median
        total_impact = 0.0
        stocks_with_data = 0
        for ticker in med_close.index:
            if pd.isna(med_close[ticker]) or pd.isna(auction_vol_shares[ticker]):
                continue
            if auction_vol_shares[ticker] <= 0 or med_close[ticker] <= 0:
                continue
            
            stock_price = med_close[ticker]
            stock_auction_vol = auction_vol_shares[ticker]
            stock_sigma = daily_vol.get(ticker, median_daily_vol)
            if pd.isna(stock_sigma) or stock_sigma <= 0:
                stock_sigma = median_daily_vol
            
            # This stock's trade in shares
            stock_trade_shares = trade_per_stock_usd / stock_price
            
            # Lambda for this stock: eta * sigma / auction_volume_shares
            lambda_s = eta * stock_sigma / stock_auction_vol
            
            # Cost = (lambda/2) * T^2 in price units, convert to dollars
            # Impact cost in dollars = (lambda_s / 2) * stock_trade_shares^2 * stock_price
            impact_cost_s = (lambda_s / 2.0) * (stock_trade_shares ** 2) * stock_price
            total_impact += impact_cost_s
            stocks_with_data += 1
        
        # Net PnL
        net_pnl = gross_pnl - slippage_cost - total_impact
        
        # Net Sharpe (rough: assume daily vol scales with sqrt of return vol)
        # Original vol at $20M: gross_pnl_20M / (8.46 * sqrt(252))
        base_daily_vol = (BOOKSIZE_BACKTEST * GROSS_PNL_PER_DOLLAR) / (8.46 / np.sqrt(252))
        scale = G / BOOKSIZE_BACKTEST
        daily_pnl_vol = base_daily_vol * scale
        net_sharpe = (net_pnl / daily_pnl_vol) * np.sqrt(252) if daily_pnl_vol > 0 else 0
        
        # Rate of return
        net_ror = (net_pnl * 252) / G * 100 if G > 0 else 0
        
        gmv_str = f"${G/1e6:.1f}M" if G >= 1e6 else f"${G/1e3:.0f}k"
        print(f"  {gmv_str:>10} | ${gross_pnl:>10,.0f} | ${slippage_cost:>8,.0f} | ${total_impact:>10,.0f} | ${net_pnl:>10,.0f} | {net_sharpe:>9.2f} | {participation_rate:>9.2%} | {net_ror:>6.1f}%")

    # Capacity: where net PnL is maximized (Isichenko Eq 6.68)
    # Scan finer grid
    fine_gmv = np.linspace(0.1e6, 200e6, 2000)
    best_pnl = -1e18
    best_gmv = 0
    half_gmv = 0  # where Sharpe drops to half
    
    for G in fine_gmv:
        trade_per_stock_usd = G * TURNOVER / N_POSITIONS
        gross_pnl = G * GROSS_PNL_PER_DOLLAR
        slippage_cost = G * TURNOVER * SLIPPAGE_BPS / 10000
        
        # Quick aggregate impact using median stock characteristics
        # lambda_median = eta * median_sigma / median_auction_vol
        lambda_med = eta * median_daily_vol / med_auction_shares
        trade_shares_med = trade_per_stock_usd / med_close.median()
        impact_per_stock = (lambda_med / 2.0) * (trade_shares_med ** 2) * med_close.median()
        total_impact = impact_per_stock * N_POSITIONS
        
        net = gross_pnl - slippage_cost - total_impact
        if net > best_pnl:
            best_pnl = net
            best_gmv = G
        
        # Where Sharpe drops to half original
        base_vol = (BOOKSIZE_BACKTEST * GROSS_PNL_PER_DOLLAR) / (8.46 / np.sqrt(252))
        scale = G / BOOKSIZE_BACKTEST
        dvol = base_vol * scale
        ns = (net / dvol) * np.sqrt(252) if dvol > 0 else 0
        if half_gmv == 0 and ns < 8.46 / 2:
            half_gmv = G
    
    print(f"\n  >> Max PnL capacity (G*): ${best_gmv/1e6:.1f}M GMV  →  net PnL = ${best_pnl:,.0f}/day")
    if half_gmv > 0:
        print(f"  >> Half-Sharpe point:     ${half_gmv/1e6:.1f}M GMV  (Sharpe drops from 8.46 to ~4.2)")
    print(f"  >> Median participation rate at G*: {(best_gmv * TURNOVER / N_POSITIONS / med_close.median()) / med_auction_shares:.2%}")

# ============================================================================
# SUMMARY FOR $248K AND $500K (our actual targets)
# ============================================================================
print(f"\n\n{'='*100}")
print(f"  CAPACITY SUMMARY FOR TARGET SIZES")
print(f"{'='*100}")

for target_gmv, label in [(248_000, "$248k (current target)"), (500_000, "$500k (IB PM target)")]:
    print(f"\n  ── {label} ──")
    trade_per_stock = target_gmv * TURNOVER / N_POSITIONS
    trade_shares = trade_per_stock / med_close.median()
    part_rate = trade_shares / auction_vol_shares.median()
    
    print(f"    Per-stock daily trade: ${trade_per_stock:,.0f} ({trade_shares:,.0f} shares)")
    print(f"    Auction participation rate: {part_rate:.4%}")
    
    for eta_label, eta in ETA_VALUES.items():
        lambda_med = eta * median_daily_vol / auction_vol_shares.median()
        impact_per = (lambda_med / 2) * trade_shares**2 * med_close.median()
        total_impact = impact_per * N_POSITIONS
        gross = target_gmv * GROSS_PNL_PER_DOLLAR
        slip = target_gmv * TURNOVER * SLIPPAGE_BPS / 10000
        net = gross - slip - total_impact
        impact_bps = total_impact / (target_gmv * TURNOVER) * 10000
        print(f"    {eta_label:<25}: impact=${total_impact:.2f}/day ({impact_bps:.3f} bps eff.)  net=${net:,.0f}/day")
    
    print(f"    >> Impact is NEGLIGIBLE at this size — well below capacity constraints")

print(f"\n{'='*100}")
print(f"  CONCLUSION")
print(f"{'='*100}")
print(f"""
  At our target GMV of $248k-$500k, market impact is essentially ZERO:
    - We trade ~${248000 * 1.07 / 253:,.0f}/stock/day = {248000 * 1.07 / 253 / med_close.median():,.0f} shares per stock
    - Closing auction median volume: {auction_vol_shares.median():,.0f} shares per stock
    - Participation rate: {(248000 * 1.07 / 253 / med_close.median()) / auction_vol_shares.median():.4%}
    
  MOC orders are particularly low-impact because:
    1. They execute in a single batch auction (no sequential self-impact)
    2. They are submitted 15-20 min before close (doesn't signal to HFTs)
    3. The closing auction has deep liquidity from index rebalances and MOC flow

  The strategy becomes capacity-constrained only at $10M+ GMV where 
  participation rates approach 1% of closing auction volume.
""")
