"""
universe_analysis.py — Analyse universe characteristics for experiment design

For each candidate universe, compute:
  1. Median/distribution of ADV, stock price, market cap
  2. MOC auction volume estimates
  3. Impact cost at $100k and $500k GMV using MOC Participation model (eta=0.08)
  4. Net PnL headroom vs current strategy performance
  5. Effective number of stocks (diversification)
"""
import numpy as np
import pandas as pd
from pathlib import Path

MATRICES_DIR = Path("data/fmp_cache/matrices")
UNIVERSE_DIR = Path("data/fmp_cache/universes")
TEST_START = "2024-07-01"

# Load base matrices
vol_df   = pd.read_parquet(MATRICES_DIR / "volume.parquet")
cls_df   = pd.read_parquet(MATRICES_DIR / "close.parquet")
cap_df   = pd.read_parquet(MATRICES_DIR / "cap.parquet")          # market cap
adv20_df = pd.read_parquet(MATRICES_DIR / "adv20.parquet")        # 20-day ADV

# Slice to test period for realistic estimates
vol  = vol_df.loc[TEST_START:]
cls  = cls_df.loc[TEST_START:]
cap  = cap_df.loc[TEST_START:]
adv20 = adv20_df.loc[TEST_START:]

# MOC model parameters
AUCTION_FRAC = 0.10
ETA_MOC = 0.08
TURNOVER = 1.07
TARGET_GMVS = [100_000, 500_000]

universes = sorted([f.stem for f in UNIVERSE_DIR.glob("*.parquet")])
print(f"Available universes: {universes}\n")

print("="*130)
print(f"  UNIVERSE ANALYSIS — ADV profile, market cap, and MOC impact at $100k / $500k GMV")
print("="*130)

# Header
print(f"\n  {'Universe':<22} | {'N(med)':>6} | {'Median ADV':>12} | {'Median Cap':>12} | "
      f"{'Med Price':>9} | {'ADV-bot10%':>10} | "
      f"{'Impact $100k':>13} | {'Impact $500k':>13} | {'Net/day $100k':>14} | {'Net/day $500k':>14}")
print(f"  {'-'*175}")

# Approximate daily gross PnL per dollar (from Billions at $20M SR=8.46)
GROSS_PER_DOLLAR = (86.8 / 252 / 100)

results = []
for uname in universes:
    try:
        univ = pd.read_parquet(UNIVERSE_DIR / f"{uname}.parquet").loc[TEST_START:]
    except Exception:
        continue

    # Mask to universe members
    mask = univ > 0
    v = vol.where(mask)
    c = cls.where(mask)
    cp = cap.where(mask)
    a = adv20.where(mask)

    # Per-stock medians across time
    med_vol   = v.median()
    med_cls   = c.median()
    med_cap   = cp.median()
    med_adv   = a.median()

    valid = med_vol.notna() & (med_vol > 0) & med_cls.notna() & (med_cls > 0)
    tickers = valid[valid].index.tolist()
    n_med = mask.sum(axis=1).median()

    # Auction volumes
    auction_vol = med_vol * AUCTION_FRAC

    # Daily return vol per stock
    rets = c.pct_change(fill_method=None)
    daily_sigma = rets.std()
    med_sigma = daily_sigma.median()

    # Impact at each GMV target using MOC Participation model
    impacts = {}
    for G in TARGET_GMVS:
        # Spread trade across positions (use actual n_med as N_POS)
        N_POS = max(int(n_med), 1)
        trade_per_stock_usd = G * TURNOVER / N_POS
        total_impact = 0.0
        for t in tickers:
            price = med_cls.get(t, np.nan)
            auc = auction_vol.get(t, np.nan)
            sig = daily_sigma.get(t, med_sigma)
            if pd.isna(price) or price <= 0 or pd.isna(auc) or auc <= 0:
                continue
            if pd.isna(sig) or sig <= 0:
                sig = med_sigma
            our_shares = trade_per_stock_usd / price
            participation = our_shares / auc
            frac_impact = ETA_MOC * sig * participation
            total_impact += frac_impact * trade_per_stock_usd / 2
        impacts[G] = total_impact

    # Net PnL per day
    nets = {}
    for G in TARGET_GMVS:
        gross = G * GROSS_PER_DOLLAR
        slip = G * TURNOVER * 0.50 / 10000
        nets[G] = gross - slip - impacts[G]

    # ADV bottom 10th percentile (most illiquid stocks in universe)
    adv_bot10 = med_adv.quantile(0.10)
    med_adv_val = med_adv.median()
    med_cap_val = med_cap.median()

    row = dict(
        universe=uname, n_med=n_med, med_adv=med_adv_val, med_cap=med_cap_val,
        med_price=med_cls.median(), adv_bot10=adv_bot10,
        impact_100k=impacts[100_000], impact_500k=impacts[500_000],
        net_100k=nets[100_000], net_500k=nets[500_000]
    )
    results.append(row)

    def fmt_m(v):
        if pd.isna(v): return "    N/A"
        if v >= 1e9: return f"${v/1e9:.2f}B"
        if v >= 1e6: return f"${v/1e6:.1f}M"
        return f"${v/1e3:.0f}k"

    print(f"  {uname:<22} | {n_med:>6.0f} | {fmt_m(med_adv_val):>12} | {fmt_m(med_cap_val):>12} | "
          f"${med_cls.median():>7.2f} | {fmt_m(adv_bot10):>10} | "
          f"${impacts[100_000]:>10,.2f} | ${impacts[500_000]:>10,.2f} | "
          f"${nets[100_000]:>11,.2f} | ${nets[500_000]:>11,.2f}")

# Best universes per target
print(f"\n\n  BEST UNIVERSES BY NET PNL")
print(f"  {'GMV':>8} | {'Universe':>22} | {'Net PnL/day':>12} | {'Impact/day':>12} | {'Impact as % gross':>18}")
df = pd.DataFrame(results)
for G, col_net, col_imp in [(100_000, 'net_100k', 'impact_100k'), (500_000, 'net_500k', 'impact_500k')]:
    gross = G * GROSS_PER_DOLLAR
    best = df.sort_values(col_net, ascending=False)
    for _, r in best.iterrows():
        print(f"  ${G/1e3:.0f}k     | {r['universe']:>22} | ${r[col_net]:>10,.2f} | ${r[col_imp]:>10,.2f} | {r[col_imp]/gross*100:>16.1f}%")
    print()
