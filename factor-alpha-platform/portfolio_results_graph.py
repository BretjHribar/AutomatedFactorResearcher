"""
Re-run portfolio optimization on the 25 qualifying alphas and generate
cumulative PnL graph showing IS vs OOS regions.
"""
import sys, os, json, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.simulation.vectorized_sim import simulate_vectorized
from src.simulation.oos import fixed_split_oos
from src.portfolio.optimizer import PortfolioOptimizer
from src.operators.fastexpression import FastExpressionEngine

# ── Config ──
IS_START = "2017-01-01"
IS_END = "2022-12-31"
SPLIT_DATE = "2023-01-01"
UNIVERSE = "TOP3000"
DELAY = 1
NEUTRALIZATION = "subindustry"
BOOKSIZE = 20_000_000.0
DB_PATH = "data/alpha_gp_pipeline.db"
MATRICES_DIR = "data/fmp_cache/matrices"
CLS_PATH = "data/fmp_cache/classifications.json"

# ── Load qualifying alphas from DB ──
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute('''
    SELECT a.expression, e.sharpe, e.fitness, e.turnover
    FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
    WHERE e.sharpe >= 1.0
    ORDER BY e.sharpe DESC
''')
qualifying = [{"expression": r[0], "sharpe": r[1]} for r in cur.fetchall()]
conn.close()
print(f"Loaded {len(qualifying)} qualifying alphas from DB")

# ── Load full data ──
print("Loading data...")
universe_df = pd.read_parquet(f"data/fmp_cache/universes/{UNIVERSE}.parquet")
universe_is = universe_df.loc[IS_START:IS_END]
ticker_coverage = universe_is.sum(axis=0) / len(universe_is)
universe_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())

with open(CLS_PATH) as f:
    all_cls = json.load(f)
classifications = {k: v for k, v in all_cls.items() if k in universe_tickers}

matrices_full = {}
for fname in sorted(os.listdir(MATRICES_DIR)):
    if not fname.endswith(".parquet") or fname.startswith("_"):
        continue
    field = fname.replace(".parquet", "")
    df = pd.read_parquet(os.path.join(MATRICES_DIR, fname))
    valid_cols = [c for c in universe_tickers if c in df.columns]
    if len(valid_cols) > 0:
        matrices_full[field] = df[valid_cols]

# Apply mask
for field, mat in matrices_full.items():
    if isinstance(mat, pd.DataFrame) and mat.shape[1] > 1:
        common_cols = mat.columns.intersection(universe_df.columns)
        common_idx = mat.index.intersection(universe_df.index)
        if len(common_cols) > 0 and len(common_idx) > 0:
            mask = universe_df.loc[common_idx, common_cols]
            matrices_full[field] = mat.loc[common_idx, common_cols].where(mask)

# Build classification series
cls_series = {}
for level in ["sector", "industry", "subindustry"]:
    mapping = {}
    for sym, cls_data in classifications.items():
        if isinstance(cls_data, dict):
            mapping[sym] = cls_data.get(level, "Unknown")
    if mapping:
        cls_series[level] = pd.Series(mapping)

full_returns = matrices_full["close"].pct_change().shift(-1)
full_close = matrices_full["close"]
full_open = matrices_full.get("open")

print(f"Data: {full_close.shape[0]} days × {full_close.shape[1]} tickers")

# ── Build expression engine ──
expr_engine = FastExpressionEngine(data_fields=matrices_full)
for gn, gs in cls_series.items():
    expr_engine.add_group(gn, gs)

# ── Evaluate each alpha over full period ──
print("\nEvaluating alphas over full period...")
alpha_pnls_full = {}  # name -> daily_pnl Series (full period)
alpha_pnls_is = {}
alpha_pnls_oos = {}
oos_table = []

for i, alpha in enumerate(qualifying):
    expr = alpha["expression"]
    name = f"alpha_{i+1}"
    try:
        alpha_df = expr_engine.evaluate(expr)
        
        # Full-period simulation
        result_full = simulate_vectorized(
            alpha_df=alpha_df,
            returns_df=full_returns,
            close_df=full_close,
            open_df=full_open,
            classifications=cls_series,
            booksize=BOOKSIZE,
            delay=DELAY,
            neutralization=NEUTRALIZATION,
        )
        
        alpha_pnls_full[name] = result_full.daily_pnl
        
        # Split IS/OOS
        is_pnl = result_full.daily_pnl.loc[:IS_END]
        oos_pnl = result_full.daily_pnl.loc[SPLIT_DATE:]
        alpha_pnls_is[name] = is_pnl
        alpha_pnls_oos[name] = oos_pnl
        
        is_sharpe = is_pnl.mean() / is_pnl.std() * np.sqrt(252) if is_pnl.std() > 0 else 0
        oos_sharpe = oos_pnl.mean() / oos_pnl.std() * np.sqrt(252) if len(oos_pnl) > 20 and oos_pnl.std() > 0 else 0
        
        oos_table.append({
            "name": name, "expression": expr,
            "is_sharpe": is_sharpe, "oos_sharpe": oos_sharpe,
            "full_pnl": result_full.daily_pnl,
        })
        
        status = "✅" if oos_sharpe > 0.5 else "⚠️" if oos_sharpe > 0 else "❌"
        print(f"  {status} {name}: IS={is_sharpe:+.2f}  OOS={oos_sharpe:+.2f}  | {expr[:60]}")
    except Exception as e:
        print(f"  ❌ {name}: FAILED — {e}")

print(f"\nSuccessfully evaluated: {len(alpha_pnls_full)}/{len(qualifying)}")

# ── Portfolio Optimization (IS period) ──
print("\n" + "=" * 70)
print("  PORTFOLIO OPTIMIZATION (IS Period: 2017-2022)")
print("=" * 70)

optimizer = PortfolioOptimizer(booksize=BOOKSIZE)
for name, pnl in alpha_pnls_is.items():
    is_sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
    idx = int(name.split("_")[1]) - 1
    optimizer.add_alpha(
        name=name,
        daily_pnl=pnl,
        sharpe=is_sharpe,
        expression=qualifying[idx]["expression"],
    )

print(f"  Added {optimizer.n_alphas} alphas to optimizer\n")

all_results = optimizer.optimize_all()

print(f"  {'Method':<20} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'Turnover':>10}")
print(f"  {'-'*56}")
for method_name, r in all_results.items():
    print(f"  {method_name:<20} {r.sharpe:>+8.2f} {r.returns_ann:>9.1%} {r.max_drawdown:>7.1%} {r.turnover:>10.3f}")

best_method = max(all_results, key=lambda m: all_results[m].sharpe)
best = all_results[best_method]
print(f"\n  🏆 Best: {best_method} (IS Sharpe={best.sharpe:.2f})")

# Print weights
print(f"\n  Portfolio Weights ({best_method}):")
for name, w in sorted(best.weights.items(), key=lambda x: -x[1]):
    if w > 0.005:
        idx = int(name.split("_")[1]) - 1
        e = qualifying[idx]["expression"][:55] if idx < len(qualifying) else "?"
        print(f"    {name}: {w:>6.1%}  | {e}")

# Correlation analysis
corr_info = optimizer.compute_correlation()
print(f"\n  Correlation Analysis:")
print(f"    Avg pairwise correlation: {corr_info.avg_pairwise_corr:.3f}")
print(f"    Max pairwise correlation: {corr_info.max_pairwise_corr:.3f}")

# ── OOS Portfolio Performance ──
print("\n" + "=" * 70)
print("  OOS PORTFOLIO PERFORMANCE (2023+)")
print("=" * 70)

opt_oos = PortfolioOptimizer(booksize=BOOKSIZE)
for name, pnl in alpha_pnls_oos.items():
    oos_sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if len(pnl) > 20 and pnl.std() > 0 else 0
    idx = int(name.split("_")[1]) - 1
    opt_oos.add_alpha(
        name=name,
        daily_pnl=pnl,
        sharpe=oos_sharpe,
        expression=qualifying[idx]["expression"],
    )

if opt_oos.n_alphas >= 2:
    oos_port = opt_oos._evaluate_portfolio(best.weights, f"{best_method}_oos")
    
    print(f"\n  {'Metric':<15} {'IS':>12} {'OOS':>12} {'Decay':>10}")
    print(f"  {'-'*49}")
    print(f"  {'Sharpe':<15} {best.sharpe:>+12.2f} {oos_port.sharpe:>+12.2f} "
          f"{(1 - oos_port.sharpe/best.sharpe) if best.sharpe != 0 else 0:>9.0%}")
    print(f"  {'Ann Return':<15} {best.returns_ann:>11.1%} {oos_port.returns_ann:>11.1%}")
    print(f"  {'Max Drawdown':<15} {best.max_drawdown:>11.1%} {oos_port.max_drawdown:>11.1%}")
    print(f"  {'Turnover':<15} {best.turnover:>12.3f} {oos_port.turnover:>12.3f}")

# ══════════════════════════════════════════════════════════════════════
# GRAPH: Cumulative PnL with IS/OOS split
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating graph...")

# Build portfolio PnL over full period using IS-optimized weights
portfolio_pnl_full = pd.Series(0.0, index=list(alpha_pnls_full.values())[0].index)
for name, w in best.weights.items():
    if name in alpha_pnls_full and w > 0:
        pnl = alpha_pnls_full[name]
        common_idx = portfolio_pnl_full.index.intersection(pnl.index)
        portfolio_pnl_full.loc[common_idx] += w * pnl.loc[common_idx]

cum_pnl = portfolio_pnl_full.cumsum()

# Also build equal-weight portfolio for comparison
ew_pnl = pd.Series(0.0, index=portfolio_pnl_full.index)
n_alphas = len(alpha_pnls_full)
for name, pnl in alpha_pnls_full.items():
    common_idx = ew_pnl.index.intersection(pnl.index)
    ew_pnl.loc[common_idx] += pnl.loc[common_idx] / n_alphas
cum_ew = ew_pnl.cumsum()

# Individual top alphas
top_oos = sorted(oos_table, key=lambda x: x["oos_sharpe"], reverse=True)[:5]

# ── Plot ──
fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor("#0d1117")

# === Top Panel: Cumulative PnL ===
ax = axes[0]
ax.set_facecolor("#0d1117")

# IS/OOS shading
split_dt = pd.Timestamp(SPLIT_DATE)
ax.axvspan(cum_pnl.index[0], split_dt, alpha=0.08, color="#58a6ff", label="_IS region")
ax.axvspan(split_dt, cum_pnl.index[-1], alpha=0.08, color="#f0883e", label="_OOS region")
ax.axvline(split_dt, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8, label="IS/OOS Split")

# Portfolio lines
ax.plot(cum_pnl.index, cum_pnl.values / 1e6, color="#58a6ff", linewidth=2.5,
        label=f"Optimized Portfolio ({best_method}, Sharpe IS={best.sharpe:.2f})")
ax.plot(cum_ew.index, cum_ew.values / 1e6, color="#7ee787", linewidth=1.8, alpha=0.7,
        label=f"Equal Weight (n={n_alphas})")

# Top 3 individual alphas (thin lines)
colors_indiv = ["#d2a8ff", "#f778ba", "#ffa657"]
for j, entry in enumerate(top_oos[:3]):
    pnl = entry["full_pnl"]
    ax.plot(pnl.index, pnl.cumsum().values / 1e6, color=colors_indiv[j], linewidth=1.0, alpha=0.5,
            label=f"Top {j+1}: OOS={entry['oos_sharpe']:.1f} | {entry['expression'][:35]}...")

# Annotations
is_final = cum_pnl.loc[:IS_END].iloc[-1] / 1e6
oos_final = cum_pnl.iloc[-1] / 1e6
ax.annotate(f"IS End: ${is_final:.1f}M", xy=(split_dt, is_final),
            xytext=(-120, 30), textcoords="offset points",
            fontsize=10, color="#58a6ff", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.5))
ax.annotate(f"Total: ${oos_final:.1f}M", xy=(cum_pnl.index[-1], oos_final),
            xytext=(-120, -30), textcoords="offset points",
            fontsize=10, color="#f0883e", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f0883e", lw=1.5))

# IS label
mid_is = pd.Timestamp("2019-06-01")
ax.text(mid_is, ax.get_ylim()[1] * 0.95, "IN-SAMPLE\n2017—2022",
        ha="center", va="top", fontsize=12, fontweight="bold", color="#58a6ff", alpha=0.4)
mid_oos = pd.Timestamp("2024-03-01")
ax.text(mid_oos, ax.get_ylim()[1] * 0.95, "OUT-OF-SAMPLE\n2023—2025",
        ha="center", va="top", fontsize=12, fontweight="bold", color="#f0883e", alpha=0.4)

ax.set_ylabel("Cumulative PnL ($M)", fontsize=13, color="white")
ax.set_title("GP Alpha Portfolio — Cumulative PnL (IS vs OOS)", fontsize=16,
             fontweight="bold", color="white", pad=15)
ax.legend(loc="upper left", fontsize=9, facecolor="#161b22", edgecolor="#30363d",
          labelcolor="white", framealpha=0.9)
ax.tick_params(colors="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.grid(True, alpha=0.15, color="white")
for spine in ax.spines.values():
    spine.set_color("#30363d")

# === Bottom Panel: Drawdown ===
ax2 = axes[1]
ax2.set_facecolor("#0d1117")

# Compute drawdown
running_max = cum_pnl.cummax()
drawdown = (cum_pnl - running_max)  # in dollars
dd_pct = drawdown / running_max.replace(0, np.nan) * 100

ax2.fill_between(cum_pnl.index, 0, drawdown.values / 1e6, color="#f85149", alpha=0.4)
ax2.plot(cum_pnl.index, drawdown.values / 1e6, color="#f85149", linewidth=1)
ax2.axvline(split_dt, color="#f0883e", linewidth=2, linestyle="--", alpha=0.8)
ax2.axhline(0, color="#30363d", linewidth=0.5)

ax2.set_ylabel("Drawdown ($M)", fontsize=13, color="white")
ax2.set_xlabel("Date", fontsize=13, color="white")
ax2.tick_params(colors="white")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.grid(True, alpha=0.15, color="white")
for spine in ax2.spines.values():
    spine.set_color("#30363d")

plt.tight_layout()
out_path = os.path.abspath("data/gp_portfolio_performance.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"\n  Graph saved to {out_path}")
plt.close()

print("\n  ✅ Done!")
