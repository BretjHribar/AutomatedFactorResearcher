"""Quick summary chart for the tree-feature AIPT sweep."""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

CSV = Path("data/aipt_results/trees_sweep_results.csv")
OUT = Path("data/aipt_results/trees_sweep_summary.png")

df = pd.read_csv(CSV)
df = df.sort_values(["K_actual", "P_eff"]).reset_index(drop=True)

# Print summary table
cols = ["label", "K_actual", "P_eff", "val_sr_n", "val_sr_g", "val_to",
        "test_sr_n", "test_sr_g", "test_to", "test_ann_ret"]
print("\n" + "=" * 120)
print("TREE-FEATURE AIPT SWEEP — RESULTS")
print("=" * 120)
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

# Plot — 2x2 panels: VAL net SR, TEST net SR, VAL gross SR, TEST gross SR
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
colors = {480: "tab:blue", 966: "tab:orange", 1928: "tab:green"}
markers = {"ridge_only": "s", "rff_P2000": "o", "rff_P5000": "o",
           "rff_P10000": "o", "rff_P20000": "o", "rff_P40000": "o"}

for K_val in sorted(df["K_actual"].unique()):
    sub = df[df["K_actual"] == K_val].sort_values("P_eff")
    c = colors[K_val]
    axes[0, 0].plot(sub["P_eff"], sub["val_sr_n"], marker="o", color=c, lw=2,
                    label=f"K={K_val}")
    axes[0, 1].plot(sub["P_eff"], sub["test_sr_n"], marker="o", color=c, lw=2,
                    label=f"K={K_val}")
    axes[1, 0].plot(sub["P_eff"], sub["val_sr_g"], marker="o", color=c, lw=2,
                    label=f"K={K_val}")
    axes[1, 1].plot(sub["P_eff"], sub["test_sr_g"], marker="o", color=c, lw=2,
                    label=f"K={K_val}")

for ax, title in zip(axes.flat, ["VAL  Net SR (3 bps fees)",
                                 "TEST Net SR (3 bps fees)",
                                 "VAL  Gross SR",
                                 "TEST Gross SR"]):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("P (RFF size; left point = ridge-only no RFF)")
    ax.set_ylabel("Sharpe (annualized)")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

# Highlight nSR=3 line on VAL panel
axes[0, 0].axhline(3.0, color="red", linestyle="--", alpha=0.6,
                   label="target nSR=3.0")
axes[0, 0].legend(loc="best")

fig.suptitle(
    "AIPT on Random GP Trees (depth ≤ 3) — VAL: 2024-09-01→2025-03-01, "
    "TEST: 2025-03-01→",
    fontsize=13, fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT}")

# Print best per (K, period)
print("\n" + "=" * 120)
print("BEST PER (K, period) BY NET SHARPE")
print("=" * 120)
for K_val in sorted(df["K_actual"].unique()):
    sub = df[df["K_actual"] == K_val]
    best_val = sub.loc[sub["val_sr_n"].idxmax()]
    best_test = sub.loc[sub["test_sr_n"].idxmax()]
    print(f"\nK={K_val}:")
    print(f"  best VAL : {best_val['label']:<35}  "
          f"VAL nSR={best_val['val_sr_n']:+.3f}  TEST nSR={best_val['test_sr_n']:+.3f}")
    print(f"  best TEST: {best_test['label']:<35}  "
          f"VAL nSR={best_test['val_sr_n']:+.3f}  TEST nSR={best_test['test_sr_n']:+.3f}")

print("\n" + "=" * 120)
print("OVERALL BEST")
print("=" * 120)
overall_test = df.loc[df["test_sr_n"].idxmax()]
overall_val = df.loc[df["val_sr_n"].idxmax()]
print(f"  best VAL nSR : {overall_val['label']:<35}  "
      f"val={overall_val['val_sr_n']:+.3f}  test={overall_val['test_sr_n']:+.3f}  "
      f"gross_test={overall_val['test_sr_g']:+.2f}  TO={overall_val['test_to']:.2%}  "
      f"AnnR_test={overall_val['test_ann_ret']:+.1f}%")
print(f"  best TEST nSR: {overall_test['label']:<35}  "
      f"val={overall_test['val_sr_n']:+.3f}  test={overall_test['test_sr_n']:+.3f}  "
      f"gross_test={overall_test['test_sr_g']:+.2f}  TO={overall_test['test_to']:.2%}  "
      f"AnnR_test={overall_test['test_ann_ret']:+.1f}%")
