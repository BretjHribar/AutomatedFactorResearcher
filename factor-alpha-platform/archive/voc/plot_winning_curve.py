"""Re-run the winning config (D=3 block-CV, P=1000) saving per-bar series + plot equity curve."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base
from backtest_voc_equities_d44_fund import RAW_FIELDS, add_extra_fundamentals

selection = json.load(open(base.RESULTS_DIR / "selected_chars_block_cv.json"))
SELECTED = selection["selected"]
SIGNS = selection["signs"]
P = 1000   # winner from sweep
print(f"Winning config: D={len(SELECTED)} {SELECTED}  P={P}", flush=True)

base.CHAR_NAMES = SELECTED
GAMMA_REF_D = 24
GAMMA_SCALE = float(np.sqrt(GAMMA_REF_D / len(SELECTED)))
base.GAMMA_GRID = [g * GAMMA_SCALE for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]


def patched_load_data():
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())
    matrices = {}
    fields = set(SELECTED) | set(RAW_FIELDS) | {"close"}
    for name in fields:
        fp = base.PIT_DIR / f"{name}.parquet"
        if fp.exists():
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
    for c in SELECTED:
        if SIGNS.get(c, 1) < 0 and c in matrices:
            matrices[c] = -matrices[c]
    available = [c for c in SELECTED if c in matrices]
    close_vals = matrices["close"].values
    dates = matrices["close"].index
    with open(base.CLASSIF_PATH) as fh:
        classifications = json.load(fh)
    return matrices, tickers, dates, close_vals, available, classifications


base.load_data = patched_load_data
matrices, tickers, dates, close_vals, chars, classifications = base.load_data()
T_total = len(dates)
oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= base.OOS_START)
start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
print(f"  N={len(tickers)} T={T_total} OOS={oos_start_idx}", flush=True)

print("Building Z panel...", flush=True)
Z_panel, D = base.build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
print(f"  D={D} built", flush=True)

print(f"Running AIPT P={P}...", flush=True)
df = base.run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                                    tickers, classifications, matrices, mode="baseline")
df["date"] = [dates[i] for i in df["bar_idx"]]
print(f"  {len(df)} bars", flush=True)

# Save per-bar
out_csv = base.RESULTS_DIR / "voc_equities_winning_perbar.csv"
df.to_csv(out_csv, index=False)
print(f"CSV: {out_csv}")

# Stats per split + fee level
ann = np.sqrt(252)
oos_mask = df["bar_idx"] >= oos_start_idx
df_oos = df[oos_mask].reset_index(drop=True)
split = len(df_oos) // 2
val_end_date = df_oos["date"].iloc[split]
print(f"  VAL: {df_oos['date'].iloc[0]} → {df_oos['date'].iloc[split-1]}  ({split} bars)")
print(f"  TEST: {df_oos['date'].iloc[split]} → {df_oos['date'].iloc[-1]}  ({len(df_oos)-split} bars)")

def stats(sub):
    g = sub["gross"].values
    out = {"gSR": g.mean()/g.std(ddof=1)*ann, "TO": sub["turnover"].mean()*100,
           "IC": sub["ic_p"].mean()}
    for bps in [0, 1, 3]:
        nn = sub["gross"] - sub["turnover"] * bps / 10000.0 * 2.0
        nn = nn.values
        out[f"nSR_{bps}bp"] = nn.mean()/nn.std(ddof=1)*ann
        out[f"ncum_{bps}bp"] = nn.sum() * 100
    return out

print(f"\n  FULL OOS: {stats(df_oos)}")
print(f"  VAL:      {stats(df_oos.iloc[:split])}")
print(f"  TEST:     {stats(df_oos.iloc[split:])}")

# ── Plot: cumulative net return at 0/1/3 bps fees, with VAL/TEST shading ─
fig, ax = plt.subplots(figsize=(14, 8))
oos = df_oos.copy()
for bps, color in [(0, "tab:green"), (1, "tab:blue"), (3, "tab:red")]:
    nn = oos["gross"] - oos["turnover"] * bps / 10000.0 * 2.0
    cum = nn.cumsum() * 100
    final = cum.iloc[-1]
    s = stats(oos)
    ax.plot(oos["date"], cum, color=color, linewidth=1.8,
            label=f"net @ {bps}bp  (nSR={s[f'nSR_{bps}bp']:+.2f}, ncum={final:+.1f}%)")

ax.axvline(val_end_date, color="black", linestyle="--", alpha=0.6, linewidth=1, label="VAL/TEST split")
# Shade VAL and TEST without forcing y-bounds — use axvspan
ax.axvspan(oos["date"].iloc[0], val_end_date, alpha=0.07, color="orange", label="VAL")
ax.axvspan(val_end_date, oos["date"].iloc[-1], alpha=0.07, color="green", label="TEST")
ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)

ax.set_xlabel("Date")
ax.set_ylabel("Cumulative net return (%)")
chars_str = ", ".join(f"{('+' if SIGNS[c]==1 else '-')}{c}" for c in SELECTED)
ax.set_title(
    f"Winning Equities AIPT — TOP2000 PIT, D={D} block-CV chars, P={P}\n"
    f"Chars: {chars_str}\n"
    f"FULL OOS gross SR = +{stats(df_oos)['gSR']:.2f}, "
    f"TEST nSR @ 1bp = {stats(df_oos.iloc[split:])['nSR_1bp']:+.2f}, "
    f"TEST IC = {stats(df_oos.iloc[split:])['IC']:+.4f}",
    fontsize=12, fontweight="bold",
)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = base.RESULTS_DIR / "voc_equities_winning_curve.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nPNG: {out_png}")
