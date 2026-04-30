"""
Equities AIPT sweep: universes × P values, no QP, no smoothing.

Universes: TOP500, TOP1000, TOP2000, TOP3000
P values:  1000, 2000, 5000, 10000

For each (universe, P) reports VAL/TEST splits + IC + R² + Sharpe + turnover.
Streams to data/aipt_results/voc_equities_sweep.csv.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from backtest_voc_equities import (
    load_data as _load_data_module,
    build_Z_panel, run_with_ic,
    BARS_PER_YEAR, TRAIN_BARS, OOS_START, COVERAGE_CUTOFF, TAKER_BPS,
    MATRICES_DIR, UNIVERSES_DIR, RESULTS_DIR, CHAR_NAMES,
)


def load_data_universe(universe_name: str):
    """Load matrices for a given universe (TOP500/TOP1000/etc)."""
    uni = pd.read_parquet(UNIVERSES_DIR / f"{universe_name}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.stem not in CHAR_NAMES + ["close"]:
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)

    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    available_chars = [c for c in CHAR_NAMES if c in matrices]
    return matrices, tickers, dates, close_vals, available_chars


def main():
    overall_t0 = time.time()
    UNIVERSES = ["TOP500", "TOP1000", "TOP2000", "TOP3000"]
    P_VALUES  = [1000, 2000, 5000, 10000]
    LOG_CSV   = RESULTS_DIR / "voc_equities_sweep.csv"

    print("=" * 100, flush=True)
    print(f"EQUITIES AIPT SWEEP  (universes={UNIVERSES}, P={P_VALUES})", flush=True)
    print("=" * 100, flush=True)

    all_rows = []
    for universe in UNIVERSES:
        print(f"\n#### {universe} ####", flush=True)
        t0 = time.time()
        matrices, tickers, dates, close_vals, chars = load_data_universe(universe)
        T_total = len(dates)
        oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
        start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
        print(f"  N={len(tickers)}  T={T_total}  OOS_start={oos_start_idx}  "
              f"chars={len(chars)}  load={time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
        print(f"  Z panel D={D} built in {time.time()-t1:.1f}s", flush=True)

        for P in P_VALUES:
            t2 = time.time()
            df = run_with_ic(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D)
            df["date"] = [dates[i] for i in df["bar_idx"]]
            n = len(df)
            split = n // 2
            ann = np.sqrt(BARS_PER_YEAR)

            def stats(sub):
                if len(sub) < 30 or sub["net_1bps"].std() < 1e-12:
                    return None
                g = sub["gross"].values
                nn = sub["net_1bps"].values
                return {
                    "sr_g": g.mean() / g.std(ddof=1) * ann,
                    "sr_n": nn.mean() / nn.std(ddof=1) * ann,
                    "to": sub["turnover"].mean(),
                    "ic_p": sub["ic_p"].mean(),
                    "ir_p": sub["ic_p"].mean() / sub["ic_p"].std(ddof=1) * ann if sub["ic_p"].std() > 1e-12 else 0.0,
                    "ic_s": sub["ic_s"].mean(),
                    "r2": sub["r2"].mean(),
                    "ncum": nn.sum() * 100,
                    "n_bars": len(sub),
                }

            full_s = stats(df)
            val_s  = stats(df.iloc[:split])
            test_s = stats(df.iloc[split:])

            row = {
                "universe": universe, "N": len(tickers), "P": P, "D": D,
                "n_oos": n, "minutes": (time.time()-t2)/60,
                "full_sr_g": full_s["sr_g"], "full_sr_n": full_s["sr_n"],
                "full_to":   full_s["to"],   "full_ic_p": full_s["ic_p"],
                "full_ir_p": full_s["ir_p"], "full_ic_s": full_s["ic_s"],
                "full_r2":   full_s["r2"],   "full_ncum": full_s["ncum"],
                "val_sr_n":  val_s["sr_n"],  "val_ic_p":  val_s["ic_p"],
                "val_ir_p":  val_s["ir_p"],  "val_to":    val_s["to"],
                "test_sr_n": test_s["sr_n"], "test_sr_g": test_s["sr_g"],
                "test_ic_p": test_s["ic_p"], "test_ir_p": test_s["ir_p"],
                "test_to":   test_s["to"],   "test_r2":   test_s["r2"],
                "test_ncum": test_s["ncum"],
            }
            all_rows.append(row)
            pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
            print(f"  {universe} P={P:>5}  "
                  f"FULL: gSR={full_s['sr_g']:+.2f} nSR={full_s['sr_n']:+.2f} "
                  f"TO={full_s['to']*100:.1f}% IC={full_s['ic_p']:+.4f} IR={full_s['ir_p']:+.2f}  "
                  f"VAL: nSR={val_s['sr_n']:+.2f} IC={val_s['ic_p']:+.4f}  "
                  f"TEST: nSR={test_s['sr_n']:+.2f} IC={test_s['ic_p']:+.4f}  "
                  f"({(time.time()-t2)/60:.1f}min)", flush=True)

        del Z_panel, matrices

    print(f"\n{'='*100}", flush=True)
    print(f"DONE in {(time.time()-overall_t0)/60:.1f} min", flush=True)
    print(f"CSV: {LOG_CSV}", flush=True)


if __name__ == "__main__":
    main()
