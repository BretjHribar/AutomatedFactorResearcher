"""
TOP2000 PIT, D=47 (24 baseline + 23 added high-coverage chars), P sweep.
Baseline mode only (no neutralization), 0/1/3 bps fee variants.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base

# ── Override CHAR_NAMES with the expanded 47-char list ───────────────────────
EXPANDED = [
    # Original 24 ----------------------------------------------------------
    "log_returns",
    "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
    "book_to_market", "earnings_yield", "free_cashflow_yield",
    "ev_to_ebitda", "ev_to_revenue",
    "roe", "roa", "gross_margin", "operating_margin", "net_margin",
    "asset_turnover",
    "adv20", "adv60", "dollars_traded", "cap",
    "debt_to_equity", "current_ratio",
    # Momentum (5) ---------------------------------------------------------
    "momentum_5d", "momentum_20d", "momentum_60d", "momentum_120d", "momentum_252d",
    # Vol horizons (6) -----------------------------------------------------
    "historical_volatility_10", "historical_volatility_30", "historical_volatility_90",
    "parkinson_volatility_10", "parkinson_volatility_30", "parkinson_volatility_90",
    # Microstructure (4) ---------------------------------------------------
    "high_low_range", "open_close_range", "close_position_in_range", "vwap_deviation",
    # Volume dynamics (4) --------------------------------------------------
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d", "turnover",
    # Beta + per-share value (4) -------------------------------------------
    "beta_to_btc", "bookvalue_ps", "tangible_book_per_share", "fcf_per_share",
    # Leverage variant -----------------------------------------------------
    "debt_to_assets",
]
base.CHAR_NAMES = EXPANDED   # monkey-patch BEFORE load_data is called

P_GRID = [1000, 2000, 5000]
LOG = base.RESULTS_DIR / "voc_equities_pit_d47.csv"


def main():
    t0 = time.time()
    print(f"TOP2000 PIT D=47, P sweep {P_GRID}, baseline mode, 0/1/3 bps", flush=True)
    matrices, tickers, dates, close_vals, chars, classifications = base.load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} chars={len(chars)} (target {len(EXPANDED)})", flush=True)
    t1 = time.time()
    Z_panel, D = base.build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s", flush=True)

    rows = []
    for P in P_GRID:
        print(f"\n--- D={D} P={P} ---", flush=True)
        ts = time.time()
        df = base.run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                                          tickers, classifications, matrices, mode="baseline")
        m = base.split_metrics(df, dates, oos_start_idx)
        m.update({"D": D, "P": P, "minutes": (time.time()-ts)/60})
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
