"""Compute expected daily PnL for Binance and KuCoin portfolios."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import combiner_risk_parity, combiner_equal

GMV = 100_000

# === BINANCE ===
mdir = Path("data/binance_cache/matrices/4h")
matrices = {f.stem: pd.read_parquet(f) for f in mdir.glob("*.parquet")}
close = matrices["close"]
returns = matrices["returns"]

with open("prod/config/binance.json") as f:
    cfg = json.load(f)

engine = FastExpressionEngine(data_fields=matrices)
qv = matrices["quote_volume"]
adv20 = qv.rolling(20, min_periods=10).mean()
rank = adv20.rank(axis=1, ascending=False)
universe = rank <= 100

raw_alphas = {}
for a in cfg["alphas"]:
    try:
        sig = engine.evaluate(a["expression"])
        sig = sig.where(universe, np.nan)
        raw_alphas[a["id"]] = sig
    except Exception:
        pass

combined = combiner_risk_parity(raw_alphas, matrices, universe, returns, max_wt=0.10)

fwd_ret = close.pct_change(fill_method=None).shift(-1)
bar_pnl = (combined * fwd_ret).sum(axis=1)

oos_start = "2024-06-01"
oos = bar_pnl.loc[oos_start:]
daily = oos.resample("D").sum()

daily_gross = daily.mean() * GMV

turnover = combined.diff().abs().sum(axis=1)
avg_turnover = turnover.loc[oos_start:].mean()
cost_per_bar = avg_turnover * 0.00017 * GMV
cost_per_day = cost_per_bar * 6

daily_net = daily_gross - cost_per_day
sr = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() > 0 else 0

print("=== BINANCE (Risk Parity, 17 alphas, OOS since 2024-06) ===")
print(f"  Daily gross PnL:  ${daily_gross:,.0f}")
print(f"  Daily costs:      ${cost_per_day:,.0f}")
print(f"  Daily net PnL:    ${daily_net:,.0f}")
print(f"  Annual net:       ${daily_net*365:,.0f}")
print(f"  Sharpe (gross):   {sr:.1f}")
print(f"  Avg turnover/bar: {avg_turnover:.3f}")

# === KUCOIN ===
mdir_kc = Path("data/kucoin_cache/matrices/4h")
matrices_kc = {f.stem: pd.read_parquet(f) for f in mdir_kc.glob("*.parquet")}
close_kc = matrices_kc["close"]
returns_kc = close_kc.pct_change(fill_method=None)

qv_kc = matrices_kc.get("quote_volume", matrices_kc.get("turnover"))
adv20_kc = qv_kc.rolling(120, min_periods=60).mean()
rank_kc = adv20_kc.rank(axis=1, ascending=False)
universe_kc = rank_kc <= 100

engine_kc = FastExpressionEngine(data_fields=matrices_kc)
kc_alphas_raw = {}
for a in cfg["alphas"]:
    try:
        sig = engine_kc.evaluate(a["expression"])
        sig = sig.where(universe_kc, np.nan)
        if sig.iloc[-1].notna().sum() < 10:
            continue
        kc_alphas_raw[a["id"]] = sig
    except Exception:
        pass

combined_kc = combiner_equal(kc_alphas_raw, matrices_kc, universe_kc, returns_kc, max_wt=0.10)
fwd_ret_kc = close_kc.pct_change(fill_method=None).shift(-1)
bar_pnl_kc = (combined_kc * fwd_ret_kc).sum(axis=1)
oos_kc = bar_pnl_kc.loc[oos_start:]
daily_kc = oos_kc.resample("D").sum()
daily_gross_kc = daily_kc.mean() * GMV

turnover_kc = combined_kc.diff().abs().sum(axis=1)
avg_turnover_kc = turnover_kc.loc[oos_start:].mean()
cost_per_day_kc = avg_turnover_kc * 0.00060 * GMV * 6
daily_net_kc = daily_gross_kc - cost_per_day_kc

sr_kc = (daily_kc.mean() / daily_kc.std()) * np.sqrt(365) if daily_kc.std() > 0 else 0

print()
print(f"=== KUCOIN (Equal Weight, {len(kc_alphas_raw)} alphas, OOS since 2024-06) ===")
print(f"  Daily gross PnL:  ${daily_gross_kc:,.0f}")
print(f"  Daily costs:      ${cost_per_day_kc:,.0f}")
print(f"  Daily net PnL:    ${daily_net_kc:,.0f}")
print(f"  Annual net:       ${daily_net_kc*365:,.0f}")
print(f"  Sharpe (gross):   {sr_kc:.1f}")
