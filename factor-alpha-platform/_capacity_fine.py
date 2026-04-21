"""Fine-grid per-stock capacity scan to find the exact G* for each eta."""
import numpy as np, pandas as pd
from pathlib import Path

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSE_DIR = Path("data/fmp_cache/universes")
UNIVERSE = "TOP2000TOP3000"

volume_df = pd.read_parquet(MATRICES_DIR / "volume.parquet")
close_df  = pd.read_parquet(MATRICES_DIR / "close.parquet")
universe_df = pd.read_parquet(UNIVERSE_DIR / f"{UNIVERSE}.parquet")

TEST_START = "2024-07-01"
vol = volume_df.loc[TEST_START:].where(universe_df.loc[TEST_START:] > 0)
cls = close_df.loc[TEST_START:].where(universe_df.loc[TEST_START:] > 0)

AUCTION_FRAC = 0.10
med_close = cls.median()
auction_vol = vol.median() * AUCTION_FRAC
returns = cls.pct_change(fill_method=None)
daily_sigma = returns.std()
median_sigma = daily_sigma.median()

# Pre-filter to stocks with valid data
valid = med_close.notna() & (med_close > 0) & auction_vol.notna() & (auction_vol > 0)
tickers = valid[valid].index.tolist()
print(f"Valid tickers for impact calc: {len(tickers)}")

# Strategy params
TURNOVER = 1.07
N_POS = 253  # actual portfolio positions per day
GROSS_BPS = 86.8 / 252 * 100  # daily gross return in bps
GROSS_PER_DOLLAR = GROSS_BPS / 10000
SLIP_BPS = 0.50
BASE_SHARPE = 8.46

# Pre-compute per-stock lambda values 
lambdas = {}
for t in tickers:
    sig = daily_sigma.get(t, median_sigma)
    if pd.isna(sig) or sig <= 0:
        sig = median_sigma
    lambdas[t] = sig / auction_vol[t]  # lambda = sigma / V_auction (before eta)

def compute_impact_perstock(G, eta):
    """Compute total daily impact cost across all positions at GMV=G."""
    trade_per_stock_usd = G * TURNOVER / N_POS
    total = 0.0
    for t in tickers:
        price = med_close[t]
        trade_shares = trade_per_stock_usd / price
        lam = eta * lambdas[t]
        # Isichenko Eq 5.3: Cost = (lambda/2) * T^2 * price
        total += (lam / 2.0) * (trade_shares ** 2) * price
    return total

print("\nScanning fine GMV grid (per-stock impact)...\n")

for eta_label, eta in [("Conservative (eta=1.0)", 1.0), 
                        ("Moderate (eta=0.5)", 0.5), 
                        ("Aggressive (eta=0.25)", 0.25)]:
    # Fine grid from $50k to $50M
    gmv_grid = np.concatenate([
        np.linspace(50e3, 2e6, 200),
        np.linspace(2e6, 50e6, 200)
    ])
    
    best_pnl = -1e18
    best_gmv = 0
    half_sr_gmv = 0
    
    results = []
    for G in gmv_grid:
        gross = G * GROSS_PER_DOLLAR
        slip = G * TURNOVER * SLIP_BPS / 10000
        impact = compute_impact_perstock(G, eta)
        net = gross - slip - impact
        
        # Sharpe scaling: vol proportional to G
        base_vol = (20e6 * GROSS_PER_DOLLAR) / (BASE_SHARPE / np.sqrt(252))
        dvol = base_vol * (G / 20e6)
        sr = (net / dvol) * np.sqrt(252) if dvol > 0 else 0
        ror = (net * 252) / G * 100 if G > 0 else 0
        
        if net > best_pnl:
            best_pnl = net
            best_gmv = G
        if half_sr_gmv == 0 and sr < BASE_SHARPE / 2:
            half_sr_gmv = G
        
        results.append((G, gross, slip, impact, net, sr, ror))
    
    # Find exact sweet spot metrics
    gstar_gross = best_gmv * GROSS_PER_DOLLAR
    gstar_slip = best_gmv * TURNOVER * SLIP_BPS / 10000
    gstar_impact = compute_impact_perstock(best_gmv, eta)
    gstar_net = gstar_gross - gstar_slip - gstar_impact
    gstar_dvol = (20e6 * GROSS_PER_DOLLAR) / (BASE_SHARPE / np.sqrt(252)) * (best_gmv / 20e6)
    gstar_sr = (gstar_net / gstar_dvol) * np.sqrt(252) if gstar_dvol > 0 else 0
    gstar_ror = (gstar_net * 252) / best_gmv * 100
    gstar_part = (best_gmv * TURNOVER / N_POS / med_close.median()) / auction_vol.median()
    
    gmv_str = f"${best_gmv/1e6:.2f}M" if best_gmv >= 1e6 else f"${best_gmv/1e3:.0f}k"
    print(f"  {eta_label}")
    print(f"    G* (Max-PnL GMV):   {gmv_str}")
    print(f"    Net PnL at G*:      ${gstar_net:,.0f}/day  (${gstar_net*252:,.0f}/year)")
    print(f"    Sharpe at G*:       {gstar_sr:.2f}")
    print(f"    Ann Return on GMV:  {gstar_ror:.1f}%")
    print(f"    Auction part. rate: {gstar_part:.2f}%")
    print(f"    Impact at G*:       ${gstar_impact:,.0f}/day ({gstar_impact/(best_gmv*TURNOVER)*10000:.1f} bps)")
    print(f"    Gross at G*:        ${gstar_gross:,.0f}/day")
    if half_sr_gmv > 0:
        hs_str = f"${half_sr_gmv/1e6:.2f}M" if half_sr_gmv >= 1e6 else f"${half_sr_gmv/1e3:.0f}k"
        print(f"    Half-Sharpe (SR~{BASE_SHARPE/2:.1f}): {hs_str}")
    
    # Print a few key points
    print(f"\n    {'GMV':>10} | {'Net PnL':>10} | {'Sharpe':>8} | {'Ann RoR':>8} | {'Impact':>10}")
    print(f"    {'-'*60}")
    for G, gross, slip, impact, net, sr, ror in results:
        if G in [100e3, 248e3, 500e3, 1e6, 2e6, 5e6, 10e6] or abs(G - best_gmv) < 50e3:
            gs = f"${G/1e6:.2f}M" if G >= 1e6 else f"${G/1e3:.0f}k"
            print(f"    {gs:>10} | ${net:>8,.0f} | {sr:>7.2f} | {ror:>6.1f}% | ${impact:>8,.0f}")
    print()
