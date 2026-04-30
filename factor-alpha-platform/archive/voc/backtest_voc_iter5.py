"""
Iter v5 — alpha SELECTION instead of "use them all".

iter4 finding: adding more alphas (36 → 49) hurt nSR (2.68 → 2.49). New alphas
introduced noise. Filter the DB to high-quality subsets and re-run best recipe.

Variants:
  topK_fitness_K  — top K by fitness score from evaluations table
  topK_sharpe_K   — top K by sharpe_is
  K ∈ {12, 18, 24, 30, 36}
  All run with iter3-best recipe: ridge=0.01, α=0.5 (no β-smoothing)
"""
from __future__ import annotations
import sys, time, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from src.operators.fastexpression import FastExpressionEngine
from backtest_voc_postfix import (
    BARS_PER_YEAR, TRAIN_BARS, MIN_TRAIN_BARS, REBAL_EVERY,
    OOS_START, COVERAGE_CUTOFF, Z_RIDGE, GAMMA_GRID, TAKER_BPS,
    UNIVERSE_PATH, MATRICES_DIR, RESULTS_DIR, CHAR_NAMES,
)
from backtest_voc_iter import smooth_z_panel, stats
from backtest_voc_iter2 import run_ewma_w
from backtest_voc_iter3 import build_Z_panel

DB_PATH = PROJECT_ROOT / "data/alphas.db"
SEED = 42
TARGET_NET_SR = 3.0
LOG_CSV  = RESULTS_DIR / "iter5_results.csv"
PLOT_OUT = RESULTS_DIR / "iter5_frontier.png"


def load_data_filtered(metric, top_k):
    """Load matrices + top-K alphas by chosen metric ('fitness' or 'sharpe_is')."""
    universe_df = pd.read_parquet(UNIVERSE_PATH)
    cov = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())

    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    available_chars = [c for c in CHAR_NAMES if c in matrices]

    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        f"SELECT a.id, a.expression, e.{metric} "
        f"FROM alphas a JOIN evaluations e ON e.alpha_id = a.id "
        f"WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h' "
        f"AND a.universe='KUCOIN_TOP100' AND e.{metric} IS NOT NULL "
        f"ORDER BY e.{metric} DESC LIMIT {top_k}"
    ).fetchall()
    con.close()

    engine = FastExpressionEngine(data_fields=matrices)
    alpha_dfs = []
    for aid, expr, score in rows:
        try:
            sig = engine.evaluate(expr).reindex(columns=tickers)
            alpha_dfs.append(sig)
        except Exception as e:
            print(f"  alpha {aid}: FAIL {e!r}")
    return matrices, tickers, dates, close_vals, available_chars, alpha_dfs


def run_variant(name, kwargs, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates):
    t0 = time.time()
    df = run_ewma_w(Z_panel=Z_panel, close_vals=close_vals, start_bar=start_bar,
                    T_total=T_total, oos_start_idx=oos_start_idx, D=D, **kwargs)
    df["date"] = [dates[i] for i in df["bar_idx"]]
    s = stats(df)
    s["name"] = name
    s["secs"] = time.time() - t0
    print(f"  {name:<48}  bars={s['bars']}  gSR={s['g_sr']:+.2f}  nSR={s['n_sr']:+.2f}  "
          f"TO={s['avg_to']*100:5.1f}%  ncum={s['n_cum']:+6.1f}%  ({s['secs']:.1f}s)")
    return s


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("Iter v5 — alpha SELECTION by IS metrics")

    results = []
    best = {"n_sr": -np.inf, "name": "(none)"}

    def log(s):
        nonlocal best
        results.append(s)
        if s["n_sr"] > best["n_sr"]:
            best = s.copy()
            print(f"  >> new best: {best['name']}  nSR={best['n_sr']:+.3f}")
        pd.DataFrame(results).to_csv(LOG_CSV, index=False)

    for metric in ["fitness", "sharpe_is"]:
        for K in [12, 18, 24, 30, 36, 49]:
            print(f"\n--- top-{K} alphas by {metric} ---")
            matrices, tickers, dates, close_vals, available_chars, alpha_dfs = \
                load_data_filtered(metric, K)
            T_total = len(dates)
            oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
            start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
            print(f"  loaded {len(alpha_dfs)} alphas  D will be {len(available_chars) + len(alpha_dfs)}")

            t1 = time.time()
            Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                                       start_bar, T_total, mode=3)
            print(f"  Z panel built in {time.time()-t1:.1f}s")

            log(run_variant(f"top{K}_{metric}_baseline", dict(P=1000, alpha=1.0),
                            Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))
            log(run_variant(f"top{K}_{metric}_ridge0.01_a0.5",
                            dict(P=1000, alpha=0.5, ridge=0.01),
                            Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))
            if best["n_sr"] >= TARGET_NET_SR:
                break
        if best["n_sr"] >= TARGET_NET_SR:
            break

    # ── done ──
    print("\n" + "=" * 100)
    print(f"Best: {best['name']}  nSR={best['n_sr']:+.3f}  TO={best['avg_to']*100:.1f}%  "
          f"ncum={best['n_cum']:+.1f}%")
    print(f"Target nSR={TARGET_NET_SR}: {'REACHED' if best['n_sr']>=TARGET_NET_SR else 'NOT reached'}")
    print(f"Total: {(time.time()-overall_t0)/60:.1f} min  ({len(results)} variants)")

    df_all = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.scatter(df_all["avg_to"]*100, df_all["n_sr"], s=70, alpha=0.7,
               c=range(len(df_all)), cmap="viridis", edgecolors="black", linewidths=0.5)
    for _, r in df_all.iterrows():
        ax.annotate(r["name"], (r["avg_to"]*100, r["n_sr"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.axhline(TARGET_NET_SR, color="red", linestyle="--", label=f"target nSR={TARGET_NET_SR}")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Avg turnover per bar (%)")
    ax.set_ylabel("Net Sharpe (3 bps taker)")
    ax.set_title(f"Iter v5 — alpha selection — best: {best['name']} nSR={best['n_sr']:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Figure: {PLOT_OUT}")
    print(f"Log:    {LOG_CSV}")


if __name__ == "__main__":
    main()
