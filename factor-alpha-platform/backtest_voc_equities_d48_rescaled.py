"""
TOP2000 PIT, D=48 with gamma rescaled by sqrt(24/D) = sqrt(0.5) ≈ 0.707
to keep RFF projection variance K-invariant. P sweep concurrently.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base
from backtest_voc_equities_d47 import EXPANDED

base.CHAR_NAMES = EXPANDED                     # D=48
GAMMA_REF_D = 24
GAMMA_SCALE = float(np.sqrt(GAMMA_REF_D / len(EXPANDED)))
print(f"GAMMA_SCALE = sqrt({GAMMA_REF_D}/{len(EXPANDED)}) = {GAMMA_SCALE:.4f}", flush=True)
base.GAMMA_GRID = [g * GAMMA_SCALE for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
print(f"Rescaled GAMMA_GRID = {[round(g,4) for g in base.GAMMA_GRID]}", flush=True)

P_GRID = [1000, 2000, 5000]
LOG = base.RESULTS_DIR / "voc_equities_pit_d48_rescaled.csv"


def main():
    t0 = time.time()
    print(f"TOP2000 PIT D=48 + gamma-rescaled, P sweep {P_GRID}, baseline mode", flush=True)
    matrices, tickers, dates, close_vals, chars, classifications = base.load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} chars={len(chars)}", flush=True)
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
