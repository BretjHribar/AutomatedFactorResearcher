"""
TOP2000 PIT, D=34 = original 24 + 10 CLASSICAL FUNDAMENTAL RATIOS
not currently in the chars (asset growth, sales growth, accruals,
dividend yield, interest coverage, etc).

These are well-established academic factors:
  asset_growth         (Cooper-Gulen-Schill 2008 investment anomaly)
  sales_growth         (growth/momentum)
  eps_growth           (earnings momentum)
  accruals             (Sloan 1996 anomaly: NI - CFO over assets)
  cash_to_assets       (financial slack)
  dividend_yield       (income/quality)
  payout_ratio         (capital allocation)
  interest_coverage    (credit quality, EBIT / interest expense)
  net_debt_to_ebitda   (leverage with cashflow lens)
  quick_ratio          (liquidity, (CA - inventory) / CL)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base


# ── Compute 10 new ratios from existing PIT raw fundamentals ────────────────
def add_fundamental_ratios(matrices):
    """In-place: add 10 new ratio matrices using raw PIT inputs from the matrices dict."""
    def safe_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a / b
        return out.replace([np.inf, -np.inf], np.nan)

    needed = ["total_assets","revenue","eps","net_income","operating_cashflow",
              "cash","dividends_paid","cap","ebit","interest_expense","ebitda",
              "net_debt","total_current_assets","inventory","total_current_liabilities"]
    miss = [f for f in needed if f not in matrices]
    if miss:
        print(f"  WARN: missing raw fields {miss} — some ratios won't compute", flush=True)

    # Year-over-year growth (use 252 trading days ~ 1 year)
    if "total_assets" in matrices:
        matrices["asset_growth"] = matrices["total_assets"].pct_change(252, fill_method=None)
    if "revenue" in matrices:
        matrices["sales_growth"] = matrices["revenue"].pct_change(252, fill_method=None)
    if "eps" in matrices:
        matrices["eps_growth"] = matrices["eps"].pct_change(252, fill_method=None)

    # Accruals = (NI - CFO) / total_assets
    if all(f in matrices for f in ["net_income","operating_cashflow","total_assets"]):
        matrices["accruals"] = safe_div(
            matrices["net_income"] - matrices["operating_cashflow"], matrices["total_assets"])

    # Financial slack
    if all(f in matrices for f in ["cash","total_assets"]):
        matrices["cash_to_assets"] = safe_div(matrices["cash"], matrices["total_assets"])

    # Dividend yield (dividends_paid is negative in cashflow → take abs); annualize ~ 4 quarters
    if all(f in matrices for f in ["dividends_paid","cap"]):
        # 4-quarter rolling sum approx via TTM: shift back 252 days, sum
        ttm_div = matrices["dividends_paid"].abs().rolling(252, min_periods=63).sum()
        matrices["dividend_yield"] = safe_div(ttm_div, matrices["cap"])

    # Payout = dividends/NI
    if all(f in matrices for f in ["dividends_paid","net_income"]):
        matrices["payout_ratio"] = safe_div(matrices["dividends_paid"].abs(), matrices["net_income"])

    # Interest coverage = EBIT / interest_expense
    if all(f in matrices for f in ["ebit","interest_expense"]):
        # interest_expense is positive; if 0, NaN
        matrices["interest_coverage"] = safe_div(matrices["ebit"], matrices["interest_expense"])

    # Net debt / EBITDA
    if all(f in matrices for f in ["net_debt","ebitda"]):
        matrices["net_debt_to_ebitda"] = safe_div(matrices["net_debt"], matrices["ebitda"])

    # Quick ratio = (current assets - inventory) / current liabilities
    if all(f in matrices for f in ["total_current_assets","inventory","total_current_liabilities"]):
        matrices["quick_ratio"] = safe_div(
            matrices["total_current_assets"] - matrices["inventory"],
            matrices["total_current_liabilities"])

    return matrices


# Override CHAR_NAMES to original 24 + 10 fundamental ratios
D34_FUND = [
    "log_returns",
    "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
    "book_to_market", "earnings_yield", "free_cashflow_yield",
    "ev_to_ebitda", "ev_to_revenue",
    "roe", "roa", "gross_margin", "operating_margin", "net_margin", "asset_turnover",
    "adv20", "adv60", "dollars_traded", "cap",
    "debt_to_equity", "current_ratio",
    # 10 NEW fundamental ratios -------------------------------------------
    "asset_growth", "sales_growth", "eps_growth",
    "accruals", "cash_to_assets",
    "dividend_yield", "payout_ratio",
    "interest_coverage", "net_debt_to_ebitda",
    "quick_ratio",
]


# Patch base.load_data to ALSO load raw fields needed for the 10 new ratios
RAW_FIELDS_NEEDED = ["total_assets","revenue","eps","net_income","operating_cashflow",
                     "cash","dividends_paid","ebit","interest_expense","ebitda",
                     "net_debt","total_current_assets","inventory","total_current_liabilities"]


def patched_load_data():
    """Load the original 24 chars + the raw fields needed for the 10 new ratios,
    compute the new ratios in-memory, then return matrices dict."""
    print(f"Loading PIT matrices from {base.PIT_DIR}...", flush=True)
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    # Load all chars + raw fields we need
    fields_to_load = set(D34_FUND) | set(RAW_FIELDS_NEEDED) | {"close"}
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

    # Compute the 10 new ratios
    add_fundamental_ratios(matrices)
    print(f"  After ratio compute, available chars from D34_FUND list:")
    available = [c for c in D34_FUND if c in matrices]
    missing = [c for c in D34_FUND if c not in matrices]
    print(f"    available={len(available)}/{len(D34_FUND)}  missing={missing}", flush=True)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    chars = available
    print(f"  Loaded D={len(chars)}, T={len(dates)}, N={len(tickers)}", flush=True)

    import json
    with open(base.CLASSIF_PATH) as fh:
        classifications = json.load(fh)
    return matrices, tickers, dates, close_vals, chars, classifications


base.load_data = patched_load_data
base.CHAR_NAMES = D34_FUND
GAMMA_REF_D = 24
GAMMA_SCALE = float(np.sqrt(GAMMA_REF_D / len(D34_FUND)))
print(f"D={len(D34_FUND)}  GAMMA_SCALE = sqrt({GAMMA_REF_D}/{len(D34_FUND)}) = {GAMMA_SCALE:.4f}", flush=True)
base.GAMMA_GRID = [g * GAMMA_SCALE for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
print(f"Rescaled GAMMA_GRID = {[round(g,4) for g in base.GAMMA_GRID]}", flush=True)

P_GRID = [1000, 2000, 5000]
LOG = base.RESULTS_DIR / "voc_equities_pit_d34_fund.csv"


def main():
    t0 = time.time()
    print(f"TOP2000 PIT D=34 (24 + 10 FUNDAMENTAL RATIOS) + γ-rescaled, P sweep {P_GRID}", flush=True)
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
