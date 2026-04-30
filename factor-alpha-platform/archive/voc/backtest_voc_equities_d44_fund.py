"""
TOP2000 PIT, D=44 = original 24 + 20 CLASSICAL FUNDAMENTAL RATIOS.
P scaled appropriately (test 2000, 4000, 8000), gamma rescaled by sqrt(24/D).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base


def add_extra_fundamentals(matrices):
    """20 classical fundamental ratios from raw PIT inputs."""
    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        return out.replace([np.inf, -np.inf], np.nan)

    # ── Original 10 (from d34_fund) ────────────────────────────────────────
    if "total_assets" in matrices:
        matrices["asset_growth"] = matrices["total_assets"].pct_change(252, fill_method=None)
    if "revenue" in matrices:
        matrices["sales_growth"] = matrices["revenue"].pct_change(252, fill_method=None)
    if "eps" in matrices:
        matrices["eps_growth"] = matrices["eps"].pct_change(252, fill_method=None)
    if all(f in matrices for f in ["net_income","operating_cashflow","total_assets"]):
        matrices["accruals"] = safe_div(
            matrices["net_income"] - matrices["operating_cashflow"], matrices["total_assets"])
    if all(f in matrices for f in ["cash","total_assets"]):
        matrices["cash_to_assets"] = safe_div(matrices["cash"], matrices["total_assets"])
    if all(f in matrices for f in ["dividends_paid","cap"]):
        ttm_div = matrices["dividends_paid"].abs().rolling(252, min_periods=63).sum()
        matrices["dividend_yield"] = safe_div(ttm_div, matrices["cap"])
    if all(f in matrices for f in ["dividends_paid","net_income"]):
        matrices["payout_ratio"] = safe_div(matrices["dividends_paid"].abs(), matrices["net_income"])
    if all(f in matrices for f in ["ebit","interest_expense"]):
        matrices["interest_coverage"] = safe_div(matrices["ebit"], matrices["interest_expense"])
    if all(f in matrices for f in ["net_debt","ebitda"]):
        matrices["net_debt_to_ebitda"] = safe_div(matrices["net_debt"], matrices["ebitda"])
    if all(f in matrices for f in ["total_current_assets","inventory","total_current_liabilities"]):
        matrices["quick_ratio"] = safe_div(
            matrices["total_current_assets"] - matrices["inventory"],
            matrices["total_current_liabilities"])

    # ── 10 NEW additions ──────────────────────────────────────────────────
    # Novy-Marx (2013) gross profitability — quality factor
    if all(f in matrices for f in ["gross_profit","total_assets"]):
        matrices["gross_profit_to_assets"] = safe_div(matrices["gross_profit"], matrices["total_assets"])

    # FCF margin
    if all(f in matrices for f in ["free_cashflow","revenue"]):
        matrices["fcf_to_revenue"] = safe_div(matrices["free_cashflow"], matrices["revenue"])

    # Investment intensity (capex is negative; take abs)
    if all(f in matrices for f in ["capex","revenue"]):
        matrices["capex_to_revenue"] = safe_div(matrices["capex"].abs(), matrices["revenue"])
    if all(f in matrices for f in ["capex","depreciation_amortization"]):
        matrices["capex_to_depreciation"] = safe_div(matrices["capex"].abs(),
                                                      matrices["depreciation_amortization"])

    # Buyback / issuance proxy
    if "shares_out" in matrices:
        matrices["shares_change_252d"] = matrices["shares_out"].pct_change(252, fill_method=None)

    # M&A activity / asset quality
    if all(f in matrices for f in ["goodwill","total_assets"]):
        matrices["goodwill_to_assets"] = safe_div(matrices["goodwill"], matrices["total_assets"])
    if all(f in matrices for f in ["intangibles","total_assets"]):
        matrices["intangibles_to_assets"] = safe_div(matrices["intangibles"], matrices["total_assets"])

    # Debt service capacity
    if all(f in matrices for f in ["operating_cashflow","total_debt"]):
        matrices["cf_to_debt"] = safe_div(matrices["operating_cashflow"], matrices["total_debt"])

    # Operating earnings yield
    if all(f in matrices for f in ["ebit","enterprise_value"]):
        matrices["ebit_to_ev"] = safe_div(matrices["ebit"], matrices["enterprise_value"])

    # Quality momentum: 12m change in ROE
    if "roe" in matrices:
        matrices["roe_change_252d"] = matrices["roe"].diff(252)

    return matrices


D44 = [
    # Original 24
    "log_returns",
    "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
    "book_to_market", "earnings_yield", "free_cashflow_yield",
    "ev_to_ebitda", "ev_to_revenue",
    "roe", "roa", "gross_margin", "operating_margin", "net_margin", "asset_turnover",
    "adv20", "adv60", "dollars_traded", "cap",
    "debt_to_equity", "current_ratio",
    # First 10 fundamentals (from d34_fund)
    "asset_growth", "sales_growth", "eps_growth",
    "accruals", "cash_to_assets",
    "dividend_yield", "payout_ratio",
    "interest_coverage", "net_debt_to_ebitda",
    "quick_ratio",
    # 10 new additional fundamentals
    "gross_profit_to_assets",   # Novy-Marx
    "fcf_to_revenue",            # FCF margin
    "capex_to_revenue",          # Investment intensity
    "capex_to_depreciation",     # Capex vs maintenance
    "shares_change_252d",        # Buyback/issuance
    "goodwill_to_assets",        # M&A intensity
    "intangibles_to_assets",     # Asset quality
    "cf_to_debt",                # Debt service
    "ebit_to_ev",                # Operating earnings yield
    "roe_change_252d",           # Quality momentum
]

RAW_FIELDS = ["total_assets","revenue","eps","net_income","operating_cashflow",
              "cash","dividends_paid","ebit","interest_expense","ebitda","net_debt",
              "total_current_assets","inventory","total_current_liabilities",
              "gross_profit","free_cashflow","capex","depreciation_amortization",
              "shares_out","goodwill","intangibles","total_debt","enterprise_value","roe"]


def patched_load_data():
    print(f"Loading PIT matrices from {base.PIT_DIR}...", flush=True)
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    fields_to_load = set(D44) | set(RAW_FIELDS) | {"close"}
    for name in fields_to_load:
        fp = base.PIT_DIR / f"{name}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[name] = df[cols]

    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    add_extra_fundamentals(matrices)
    available = [c for c in D44 if c in matrices]
    missing = [c for c in D44 if c not in matrices]
    print(f"  D={len(available)}/{len(D44)}  missing={missing}", flush=True)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    print(f"  T={len(dates)} N={len(tickers)}", flush=True)

    import json
    with open(base.CLASSIF_PATH) as fh:
        classifications = json.load(fh)
    return matrices, tickers, dates, close_vals, available, classifications


base.load_data = patched_load_data
base.CHAR_NAMES = D44
GAMMA_REF_D = 24
GAMMA_SCALE = float(np.sqrt(GAMMA_REF_D / len(D44)))
print(f"D={len(D44)}  GAMMA_SCALE = sqrt({GAMMA_REF_D}/{len(D44)}) = {GAMMA_SCALE:.4f}", flush=True)
base.GAMMA_GRID = [g * GAMMA_SCALE for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]

# Scale P proportionally — D=24 P=2000 optimum; D=44 P_opt ≈ 2000*(44/24) ≈ 3667 → test 2000,4000,8000
P_GRID = [2000, 4000, 8000]
LOG = base.RESULTS_DIR / "voc_equities_pit_d44_fund.csv"


def main():
    t0 = time.time()
    print(f"TOP2000 PIT D=44 (24 + 20 FUNDAMENTAL RATIOS) + γ-rescaled, P sweep {P_GRID}", flush=True)
    matrices, tickers, dates, close_vals, chars, classifications = base.load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    t1 = time.time()
    Z_panel, D = base.build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s", flush=True)

    rows = []
    for P in P_GRID:
        print(f"\n--- D={D} P={P} (gamma×{GAMMA_SCALE:.3f}) ---", flush=True)
        ts = time.time()
        df = base.run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                                          tickers, classifications, matrices, mode="baseline")
        m = base.split_metrics(df, dates, oos_start_idx)
        m.update({"D": D, "P": P, "gamma_scale": GAMMA_SCALE, "minutes": (time.time()-ts)/60})
        rows.append(m)
        pd.DataFrame(rows).to_csv(LOG, index=False)
        print(f"  D={D} P={P}  bars={len(df)}  TO={m.get('full_to',0)*100:5.1f}%  "
              f"IC={m.get('full_ic_p',0):+.4f}  IR={m.get('full_ir_p',0):+.2f}  "
              f"R²={m.get('full_r2',0):.4f}  ({(time.time()-ts)/60:.1f}min)", flush=True)
        for bps in base.TAKER_BPS_GRID:
            print(f"    fee={bps:g}bps  "
                  f"FULL: gSR={m.get('full_sr_g',0):+.2f} nSR={m.get(f'full_sr_n_{bps:g}bps',0):+.2f} "
                  f"ncum={m.get(f'full_ncum_{bps:g}bps',0):+.1f}%  |  "
                  f"VAL nSR={m.get(f'val_sr_n_{bps:g}bps',0):+.2f}  "
                  f"TEST nSR={m.get(f'test_sr_n_{bps:g}bps',0):+.2f}", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min — CSV: {LOG}")


if __name__ == "__main__":
    main()
