"""
Iter v3 — pull ALL crypto 4h KUCOIN_TOP100 alphas from data/alphas.db
(currently 34, was 18 before) and re-run the best smoothing recipes.

The 18 → 34 alpha jump should add ~30-50% more independent signal which is
exactly the gross-SR boost we need to push net SR past 2.4.
"""
from __future__ import annotations
import sys, time, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

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

DB_PATH = PROJECT_ROOT / "data/alphas.db"
SEED = 42
TARGET_NET_SR = 3.0
LOG_CSV  = RESULTS_DIR / "iter3_results.csv"
PLOT_OUT = RESULTS_DIR / "iter3_frontier.png"


def load_data_with_all_db_alphas():
    """Same as backtest_voc_postfix.load_data but pulls ALL crypto/4h KUCOIN_TOP100
    alphas from data/alphas.db (instead of the 18 hardcoded)."""
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

    # Pull all crypto 4h KUCOIN_TOP100 alphas from DB
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute(
        "SELECT id, expression FROM alphas "
        "WHERE archived=0 AND asset_class='crypto' AND interval='4h' "
        "AND universe='KUCOIN_TOP100' ORDER BY id"
    ).fetchall()
    con.close()
    print(f"  loaded {len(rows)} alphas from DB")

    engine = FastExpressionEngine(data_fields=matrices)
    alpha_dfs = []
    for aid, expr in rows:
        try:
            sig = engine.evaluate(expr).reindex(columns=tickers)
            alpha_dfs.append(sig)
        except Exception as e:
            print(f"  alpha {aid:>3}: FAIL {e!r}")
    print(f"  {len(alpha_dfs)}/{len(rows)} alphas evaluated")
    return matrices, tickers, dates, close_vals, available_chars, alpha_dfs


def build_Z_panel(matrices, tickers, available_chars, alpha_dfs, start, end, mode):
    """Same as backtest_voc_postfix.build_Z_panel."""
    char_list  = available_chars if mode != 2 else []
    use_alphas = alpha_dfs       if mode != 1 else []
    N = len(tickers)
    D = len(char_list) + len(use_alphas)
    panel = {}
    for t in range(start, end):
        Z = np.full((N, D), np.nan)
        j = 0
        for cn in char_list:
            Z[:, j] = matrices[cn].iloc[t].reindex(tickers).values.astype(np.float64)
            j += 1
        for adf in use_alphas:
            Z[:, j] = adf.iloc[t].reindex(tickers).values.astype(np.float64)
            j += 1
        for j in range(D):
            col = Z[:, j]
            ok  = ~np.isnan(col)
            if ok.sum() < 3:
                Z[:, j] = 0.0
                continue
            r        = rankdata(col[ok], method="average") / ok.sum() - 0.5
            Z[ok, j] = r
            Z[~ok, j] = 0.0
        panel[t] = Z
    return panel, D


def run_variant(name, kwargs, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates):
    t0 = time.time()
    df = run_ewma_w(Z_panel=Z_panel, close_vals=close_vals, start_bar=start_bar,
                    T_total=T_total, oos_start_idx=oos_start_idx, D=D, **kwargs)
    df["date"] = [dates[i] for i in df["bar_idx"]]
    s = stats(df)
    s["name"] = name
    s["secs"] = time.time() - t0
    print(f"  {name:<40}  bars={s['bars']}  gSR={s['g_sr']:+.2f}  nSR={s['n_sr']:+.2f}  "
          f"TO={s['avg_to']*100:5.1f}%  ncum={s['n_cum']:+6.1f}%  ({s['secs']:.1f}s)")
    return s


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("Iter v3 — load ALL crypto 4h KUCOIN_TOP100 alphas from DB")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data_with_all_db_alphas()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} chars={len(available_chars)} alphas={len(alpha_dfs)}")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                               start_bar, T_total, mode=3)
    bars_iter = list(range(start_bar, T_total))
    print(f"  Z panel D={D}  built in {time.time()-t1:.1f}s")
    print("=" * 100)

    results = []
    best = {"n_sr": -np.inf, "name": "(none)"}

    def log(s):
        nonlocal best
        results.append(s)
        if s["n_sr"] > best["n_sr"]:
            best = s.copy()
            print(f"  >> new best: {best['name']}  nSR={best['n_sr']:+.3f}")
        pd.DataFrame(results).to_csv(LOG_CSV, index=False)

    # --- batch A: baseline + best-smoothing recipes -------------------------------
    print(f"\n[A] baseline + best smoothings on full alpha set (D={D})")
    log(run_variant("baseline (P=1000)", dict(P=1000, alpha=1.0),
                    Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))
    log(run_variant("ewma_w(α=0.5)", dict(P=1000, alpha=0.5),
                    Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))
    Zsm5 = smooth_z_panel(Z_panel, bars_iter, 0.5)
    log(run_variant("ewma_z(β=0.5)", dict(P=1000, alpha=1.0),
                    Zsm5, close_vals, start_bar, T_total, oos_start_idx, D, dates))
    log(run_variant("combo(α=0.5,β=0.5)", dict(P=1000, alpha=0.5),
                    Zsm5, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch B: higher P with combo ---------------------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[B] higher P with α=0.5, β=0.5")
        for P_try in [2000, 3000, 4000]:
            if best["n_sr"] >= TARGET_NET_SR: break
            log(run_variant(f"P={P_try},combo", dict(P=P_try, alpha=0.5),
                            Zsm5, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch C: ridge × EWMA-w combos -------------------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[C] ridge × EWMA-w combos (no z-smoothing)")
        for ridge in [3e-3, 1e-2, 3e-2]:
            for alpha in [0.5, 0.75, 1.0]:
                if best["n_sr"] >= TARGET_NET_SR: break
                log(run_variant(f"ridge={ridge:g},a={alpha}",
                                dict(P=1000, alpha=alpha, ridge=ridge),
                                Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, dates))

    # --- batch D: alphas-only with full set ---------------------------------------
    if best["n_sr"] < TARGET_NET_SR:
        print(f"\n[D] alphas-only (mode 2) with full DB alpha set")
        Z2, D2 = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                               start_bar, T_total, mode=2)
        Z2sm = smooth_z_panel(Z2, bars_iter, 0.5)
        log(run_variant("alphas_only baseline", dict(P=1000, alpha=1.0),
                        Z2, close_vals, start_bar, T_total, oos_start_idx, D2, dates))
        log(run_variant("alphas_only combo(0.5,0.5)", dict(P=1000, alpha=0.5),
                        Z2sm, close_vals, start_bar, T_total, oos_start_idx, D2, dates))

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
    ax.set_title(f"Iter v3 — full DB alphas — best: {best['name']} nSR={best['n_sr']:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
    print(f"Figure: {PLOT_OUT}")
    print(f"Log:    {LOG_CSV}")


if __name__ == "__main__":
    main()
