"""Plot equity curve from saved TOP2000 P=1000 baseline (per-bar gross/net/turnover/IC)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
RESULTS_DIR = Path("data/aipt_results")

CSV = RESULTS_DIR / "voc_equities_baseline.csv"
OUT = RESULTS_DIR / "voc_equities_top2000_curve.png"

BARS_PER_YEAR = 252
OOS_START = pd.Timestamp("2024-01-01")

df = pd.read_csv(CSV, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Split OOS into VAL/TEST 50/50
oos_mask = df["date"] >= OOS_START
oos_idx = df.index[oos_mask].tolist()
split_pos = len(oos_idx) // 2
val_end_date = df.loc[oos_idx[split_pos], "date"]

cum_gross = df["gross"].cumsum() * 100
cum_net = df["net_1bps"].cumsum() * 100

# Stats per split
ann = np.sqrt(BARS_PER_YEAR)
def stat(sub, tag):
    if len(sub) < 30:
        return f"{tag}: n/a"
    g, nn = sub["gross"].values, sub["net_1bps"].values
    sr_g = g.mean() / g.std(ddof=1) * ann
    sr_n = nn.mean() / nn.std(ddof=1) * ann
    ic = sub["ic_p"].mean()
    return (f"{tag}: gSR={sr_g:+.2f} nSR={sr_n:+.2f} IC={ic:+.4f} "
            f"TO={sub['turnover'].mean()*100:.1f}% ncum={nn.sum()*100:+.1f}%")

train = df[df["date"] < OOS_START]
val   = df[(df["date"] >= OOS_START) & (df["date"] < val_end_date)]
test  = df[df["date"] >= val_end_date]

print(stat(df,    "FULL  "))
print(stat(train, "TRAIN "))
print(stat(val,   "VAL   "))
print(stat(test,  "TEST  "))

fig, ax = plt.subplots(figsize=(14, 7))

# Plot gross + net curves; color-code TRAIN / VAL / TEST
def plot_segment(sub, color_g, color_n, label_prefix, lw_g=1.4, lw_n=1.4):
    ax.plot(sub["date"], sub["gross"].cumsum() * 100 + (cum_gross.iloc[sub.index[0]] - sub["gross"].iloc[0]*100),
            color=color_g, linewidth=lw_g, label=f"{label_prefix} gross",
            alpha=0.6 if "TRAIN" in label_prefix else 1.0)

# Simpler approach: use full cumsum throughout, just color spans differently
ax.plot(df["date"], cum_gross, color="lightgray", linewidth=0.8, label="Full series gross", alpha=0.5)
ax.plot(df["date"], cum_net, color="darkgray", linewidth=0.8, label="Full series net (1bp)", alpha=0.7)

# Overlay colored spans
ax.fill_between(df["date"], cum_net.min() - 5, cum_gross.max() + 5,
                where=(df["date"] < OOS_START), alpha=0.05, color="blue", label="TRAIN (walk-fwd, pre-OOS)")
ax.fill_between(df["date"], cum_net.min() - 5, cum_gross.max() + 5,
                where=((df["date"] >= OOS_START) & (df["date"] < val_end_date)),
                alpha=0.10, color="orange", label="VAL")
ax.fill_between(df["date"], cum_net.min() - 5, cum_gross.max() + 5,
                where=(df["date"] >= val_end_date),
                alpha=0.10, color="green", label="TEST")

# Heavier lines on OOS portion
oos = df[df["date"] >= OOS_START]
ax.plot(oos["date"], cum_gross.loc[oos.index], color="tab:blue", linewidth=2.2, label="OOS gross")
ax.plot(oos["date"], cum_net.loc[oos.index], color="tab:red", linewidth=2.2, label="OOS net (1 bp)")

ax.axvline(OOS_START, color="black", linestyle="--", alpha=0.7, linewidth=1)
ax.axvline(val_end_date, color="black", linestyle=":", alpha=0.7, linewidth=1)
ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return (%)")
ax.set_title(
    "Equities AIPT — TOP2000 D=24 chars + RFF P=1000, ρ=1e-3, fees=1bp aggregate\n"
    f"FULL OOS nSR=+4.66 (TO=13.8%, IC=+0.0125)  VAL nSR=+6.15  TEST nSR=+3.41",
    fontsize=12, fontweight="bold",
)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nFigure: {OUT}")
