"""
P sweep on PIT data: TOP2000, D=24 chars, no neutralization, no smoothing.
Tests P ∈ {1000, 2000, 5000, 10000} to find the optimal RFF size for D=24.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_equities_neutralized import (
    load_data, build_Z_panel, run_with_neutralization, split_metrics,
    BARS_PER_YEAR, TRAIN_BARS, OOS_START, RESULTS_DIR, TAKER_BPS_GRID,
)

P_GRID = [1000, 2000, 5000, 10000]
LOG = RESULTS_DIR / "voc_equities_pit_psweep.csv"


def main():
    overall_t0 = time.time()
    print(f"PIT P-sweep on TOP2000 D=24 (baseline, no neutralization)")
    matrices, tickers, dates, close_vals, chars, classifications = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} OOS={oos_start_idx} chars={len(chars)}")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s")

    rows = []
    for P in P_GRID:
        print(f"\n--- P = {P} ---", flush=True)
        t0 = time.time()
        df = run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                                       tickers, classifications, matrices, mode="baseline")
        m = split_metrics(df, dates, oos_start_idx)
        m["P"] = P
        m["minutes"] = (time.time() - t0) / 60
        rows.append(m)
        pd.DataFrame(rows).to_csv(LOG, index=False)
        print(f"  P={P}  bars={len(df)}  TO={m.get('full_to',0)*100:5.1f}%  "
              f"IC={m.get('full_ic_p',0):+.4f}  IR={m.get('full_ir_p',0):+.2f}  "
              f"R²={m.get('full_r2',0):.4f}  ({(time.time()-t0)/60:.1f}min)", flush=True)
        for bps in TAKER_BPS_GRID:
            print(f"    fee={bps:g}bps  "
                  f"FULL: gSR={m.get('full_sr_g',0):+.2f} nSR={m.get(f'full_sr_n_{bps:g}bps',0):+.2f} "
                  f"ncum={m.get(f'full_ncum_{bps:g}bps',0):+.1f}%  |  "
                  f"VAL nSR={m.get(f'val_sr_n_{bps:g}bps',0):+.2f}  "
                  f"TEST nSR={m.get(f'test_sr_n_{bps:g}bps',0):+.2f}", flush=True)

    print(f"\nDONE in {(time.time()-overall_t0)/60:.1f} min — CSV: {LOG}")


if __name__ == "__main__":
    main()
