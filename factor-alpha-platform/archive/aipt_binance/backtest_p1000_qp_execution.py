"""
P=1000 AIPT signal, then execution QP from src/simulation/walkforward.py
(optimize_portfolio). QP settings tuned on a validation slice of the OOS
period; held-out test results reported.

QP objective (from walkforward.optimize_portfolio):
  maximize   alpha·h - (kappa/2)·h'Σh - tx_cost·||h - h_prev||_1
  subject to sum(h) = 0                 (dollar neutral)
             ||h||_inf <= max_position
             ||h||_1   <= max_gross_leverage
             ||h - h_prev||_1 <= max_turnover
             pca[:,k]·h = 0    for k in 1..n_pca_factors
"""
from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.walkforward import optimize_portfolio, WalkForwardConfig

# ── AIPT config (match aipt_trader.py / aipt_barbybar.py) ──────────────────
UNIVERSE        = "KUCOIN_TOP100"
INTERVAL        = "4h"
BARS_PER_YEAR   = 6 * 365
TRAIN_BARS      = 4380
MIN_TRAIN_BARS  = 1000
REBAL_EVERY     = 12
OOS_START       = "2024-09-01"
COVERAGE_CUTOFF = 0.3
Z_RIDGE         = 1e-3
SEED            = 42
GAMMA_GRID      = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
P_FACTORS       = 1000
TAKER_BPS       = 3.0   # KuCoin VIP12 taker

CHAR_NAMES = [
    "adv20","adv60","beta_to_btc","close_position_in_range",
    "dollars_traded","high_low_range",
    "historical_volatility_10","historical_volatility_20",
    "historical_volatility_60","historical_volatility_120",
    "log_returns","momentum_5d","momentum_20d","momentum_60d",
    "open_close_range",
    "parkinson_volatility_10","parkinson_volatility_20","parkinson_volatility_60",
    "quote_volume","turnover",
    "volume_momentum_1","volume_momentum_5_20","volume_ratio_20d","vwap_deviation",
]
MATRICES_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes" / f"{UNIVERSE}_{INTERVAL}.parquet"
RESULTS_DIR = PROJECT_ROOT / "data/aipt_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    uni = pd.read_parquet(UNIVERSE_PATH)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    raw = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid if c in df.columns]
        if cols:
            raw[fp.stem] = df[cols]
    close_df = raw["close"]
    tickers = [t for t in valid if t in close_df.columns]
    close_vals = close_df[tickers].values
    returns_mat = raw["returns"][tickers].values if "returns" in raw else None
    dates = close_df.index
    avail = [c for c in CHAR_NAMES if c in raw]
    return raw, tickers, dates, close_vals, returns_mat, len(avail), avail


def build_Z_panel(raw, tickers, avail, start, end):
    N, D = len(tickers), len(avail)
    panel = {}
    for t in range(start, end):
        Z = np.full((N, D), np.nan)
        for j, cn in enumerate(avail):
            Z[:, j] = raw[cn].iloc[t].reindex(tickers).values.astype(np.float64)
        for j in range(D):
            col = Z[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 3:
                Z[:, j] = 0.0
                continue
            r = rankdata(col[ok], method="average") / ok.sum() - 0.5
            Z[ok, j] = r
            Z[~ok, j] = 0.0
        panel[t] = Z
    return panel


def estimate_ridge_markowitz_woodbury(F_train, z, P):
    T = F_train.shape[0]
    mu = F_train.mean(axis=0)
    if P <= T:
        FF = (F_train.T @ F_train) / T
        A = z * np.eye(P) + FF
        return np.linalg.solve(A, mu)
    FFT = F_train @ F_train.T
    A_T = z * T * np.eye(T) + FFT
    return (mu - F_train.T @ np.linalg.solve(A_T, F_train @ mu)) / z


# ── Step 1: run AIPT P=1000 once and cache raw signals per OOS bar ───────
def generate_p1000_signals(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D):
    """Run P=1000 bar-by-bar and collect (date_idx, raw_w, R_t, valid_mask) per OOS bar.
    raw_w is the pre-normalization Markowitz SDF weight vector (NOT divided by sum|w|)."""
    P = P_FACTORS
    rng = np.random.default_rng(SEED)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history = []
    lambda_hat = None
    bars_since_rebal = REBAL_EVERY
    rows = []

    for t in range(start_bar, T_total - 1):
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)
        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v = S_t[valid]
        R_v = R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        # Rebal lambda
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if idx < (t + 1) and idx >= cutoff_low]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            lambda_hat = estimate_ridge_markowitz_woodbury(F_train, Z_RIDGE, P)
            bars_since_rebal = 0

        # Raw weights (cross-sectional Markowitz signal — not normalized here;
        # QP will handle gross/turnover/etc.)
        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        # Normalize to sum|w| = 1 so it's a comparable scale across bars
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        rows.append({
            "bar_idx": t + 1,
            "signal": w_norm.copy(),
            "R_t1": R_t1.copy(),
            "valid": valid.copy(),
        })
        bars_since_rebal += 1
    return rows


# ── Step 2: apply QP execution optimizer over a slice of bars ─────────────
def run_qp_on_slice(rows, tickers, close_df_all, config: WalkForwardConfig, lookback=120):
    """For each bar in rows, call optimize_portfolio with signal=rows[i]['signal'].
    Return DataFrame of (bar_idx, gross, net_3bps, turnover, converged)."""
    out = []
    prev_h = None
    prev_h_series = None
    for i, row in enumerate(rows):
        bar = row["bar_idx"]
        signal_arr = row["signal"]
        R_t1 = row["R_t1"]

        # Trailing window of returns for covariance + PCA
        lo = max(0, bar - lookback)
        ret_window = close_df_all.iloc[lo:bar].pct_change(fill_method=None).fillna(0)

        signal_ser = pd.Series(signal_arr, index=tickers)
        h_new_ser, converged = optimize_portfolio(
            signal_ser, ret_window, prev_h_series, config
        )
        h_new = h_new_ser.reindex(tickers).fillna(0).values

        # Convention (matches backtest_voc_postfix and aipt_barbybar): the
        # portfolio return attributed to this row is h_t @ R_{t+1}, i.e. holdings
        # formed at bar t earn the return over the t→t+1 interval.
        port_ret = float(h_new @ R_t1)
        turnover = float(np.abs(h_new - (prev_h if prev_h is not None else np.zeros_like(h_new))).sum() / 2.0)
        out.append({
            "bar_idx": bar, "gross": port_ret, "turnover": turnover,
            "converged": bool(converged),
        })
        prev_h = h_new
        prev_h_series = h_new_ser

    df = pd.DataFrame(out)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000 * 2
    return df


def summarize(df, label=""):
    if df.empty:
        return {"label": label, "n": 0}
    g = df["gross"].values
    n = df["net_3bps"].values
    ann = np.sqrt(BARS_PER_YEAR)
    return {
        "label": label,
        "n": len(df),
        "gross_mean_bps": g.mean() * 10000,
        "net_mean_bps": n.mean() * 10000,
        "gross_std_bps": g.std(ddof=1) * 10000,
        "net_std_bps": n.std(ddof=1) * 10000,
        "gross_sr_ann": (g.mean() / (g.std(ddof=1) + 1e-12)) * ann,
        "net_sr_ann": (n.mean() / (n.std(ddof=1) + 1e-12)) * ann,
        "gross_cum_pct": g.sum() * 100,
        "net_cum_pct": n.sum() * 100,
        "avg_turnover": df["turnover"].mean(),
        "converged_pct": df["converged"].mean() * 100,
    }


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] Loading data...", flush=True)
    raw, tickers, dates, close_vals, returns_mat, D, avail = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(0, oos_start_idx - TRAIN_BARS - 10)
    close_df_all = raw["close"]

    print(f"[{time.time()-t0:5.1f}s] N={len(tickers)} D={D} T={T_total} OOS={oos_start_idx}", flush=True)
    print(f"[{time.time()-t0:5.1f}s] Building Z panel for {T_total - start_bar} bars...", flush=True)
    Z_panel = build_Z_panel(raw, tickers, avail, start_bar, T_total)

    print(f"[{time.time()-t0:5.1f}s] Running P=1000 AIPT to precompute signals...", flush=True)
    rows = generate_p1000_signals(Z_panel, close_vals, start_bar, T_total, oos_start_idx, D)
    print(f"[{time.time()-t0:5.1f}s] Got {len(rows)} OOS bars with signals", flush=True)

    # Val / Test split — 50/50 by time
    n = len(rows)
    split = n // 2
    val_rows = rows[:split]
    test_rows = rows[split:]
    val_dates = (dates[val_rows[0]["bar_idx"]], dates[val_rows[-1]["bar_idx"]])
    test_dates = (dates[test_rows[0]["bar_idx"]], dates[test_rows[-1]["bar_idx"]])
    print(f"[{time.time()-t0:5.1f}s] Val: {val_dates[0]} -> {val_dates[1]} ({len(val_rows)} bars)", flush=True)
    print(f"[{time.time()-t0:5.1f}s] Test: {test_dates[0]} -> {test_dates[1]} ({len(test_rows)} bars)", flush=True)

    # Baseline: no QP — use raw P=1000 signal as holdings directly (like the original backtest)
    print(f"[{time.time()-t0:5.1f}s] Computing baseline (raw P=1000, no QP)...", flush=True)
    def raw_baseline(rows_):
        # Same convention as run_qp_on_slice: port_ret = w_t @ R_{t+1}
        out = []; prev = None
        for r in rows_:
            w = r["signal"]
            pr = float(w @ r["R_t1"])
            to = float(np.abs(w - (prev if prev is not None else np.zeros_like(w))).sum() / 2)
            out.append({"bar_idx": r["bar_idx"], "gross": pr, "turnover": to, "converged": True})
            prev = w
        d = pd.DataFrame(out)
        d["net_3bps"] = d["gross"] - d["turnover"] * TAKER_BPS / 10000 * 2
        return d
    base_val = raw_baseline(val_rows)
    base_test = raw_baseline(test_rows)
    base_val_s = summarize(base_val, "baseline_val")
    base_test_s = summarize(base_test, "baseline_test")
    print(f"[{time.time()-t0:5.1f}s]   baseline VAL:  SR_net={base_val_s['net_sr_ann']:+.2f} cum_net={base_val_s['net_cum_pct']:+.1f}% TO={base_val_s['avg_turnover']*100:.1f}%", flush=True)
    print(f"[{time.time()-t0:5.1f}s]   baseline TEST: SR_net={base_test_s['net_sr_ann']:+.2f} cum_net={base_test_s['net_cum_pct']:+.1f}% TO={base_test_s['avg_turnover']*100:.1f}%", flush=True)

    # ── Param grid for QP sweep ───────────────────────────────────────────
    # Trimmed to 24 configs (runs in ~15-25 min on val set).
    grid = []
    for max_turnover in [0.20, 0.40, 0.60]:
        for risk_aversion in [1.0, 10.0]:
            for tx_cost_bps in [3.0, 15.0]:
                for max_gross_leverage in [0.8, 1.0]:
                    for n_pca in [0, 3]:
                        grid.append({
                            "max_turnover": max_turnover,
                            "risk_aversion": risk_aversion,
                            "tx_cost_bps": tx_cost_bps,
                            "max_gross_leverage": max_gross_leverage,
                            "max_position": 0.08,
                            "n_pca_factors": n_pca,
                            "cov_shrinkage": 0.9,
                        })
    print(f"[{time.time()-t0:5.1f}s] Grid size: {len(grid)} configs", flush=True)

    results = []
    for i, gp in enumerate(grid):
        cfg = WalkForwardConfig(
            max_position=gp["max_position"],
            max_gross_leverage=gp["max_gross_leverage"],
            max_turnover=gp["max_turnover"],
            risk_aversion=gp["risk_aversion"],
            tx_cost_bps=gp["tx_cost_bps"],
            n_pca_factors=gp["n_pca_factors"],
            cov_shrinkage=gp["cov_shrinkage"],
            optimizer_lookback=120,
        )
        t1 = time.time()
        df_val = run_qp_on_slice(val_rows, tickers, close_df_all, cfg)
        s_val = summarize(df_val, f"val_cfg{i}")
        s_val.update(gp)
        results.append(s_val)
        dt = time.time() - t1
        if (i + 1) % 5 == 0 or i == len(grid) - 1:
            print(f"[{time.time()-t0:5.1f}s] [{i+1}/{len(grid)}] "
                  f"TO={gp['max_turnover']:.2f} kappa={gp['risk_aversion']:.1f} "
                  f"tx={gp['tx_cost_bps']:.0f}bps gross={gp['max_gross_leverage']:.1f} "
                  f"pca={gp['n_pca_factors']} | "
                  f"val_net_SR={s_val['net_sr_ann']:+.2f} cum={s_val['net_cum_pct']:+.1f}% "
                  f"TO={s_val['avg_turnover']*100:.1f}% conv={s_val['converged_pct']:.0f}% ({dt:.1f}s)",
                  flush=True)

    res_df = pd.DataFrame(results).sort_values("net_sr_ann", ascending=False)
    res_df.to_csv(RESULTS_DIR / "qp_val_sweep.csv", index=False)
    print(f"\n[{time.time()-t0:5.1f}s] TOP 10 configs by val net Sharpe:", flush=True)
    cols = ["max_turnover","risk_aversion","tx_cost_bps","max_gross_leverage",
            "n_pca_factors","net_sr_ann","net_cum_pct","gross_sr_ann","avg_turnover","converged_pct"]
    print(res_df[cols].head(10).to_string(index=False), flush=True)

    # Pick best by val net Sharpe — among configs that have at least 80% convergence
    best_candidates = res_df[res_df["converged_pct"] >= 80]
    if best_candidates.empty:
        best_candidates = res_df
    best = best_candidates.iloc[0]
    print(f"\n[{time.time()-t0:5.1f}s] BEST VAL CONFIG: "
          f"TO={best['max_turnover']:.2f} kappa={best['risk_aversion']:.1f} "
          f"tx={best['tx_cost_bps']:.0f} gross={best['max_gross_leverage']:.1f} pca={int(best['n_pca_factors'])}",
          flush=True)

    # Evaluate on TEST
    best_cfg = WalkForwardConfig(
        max_position=0.08,
        max_gross_leverage=best["max_gross_leverage"],
        max_turnover=best["max_turnover"],
        risk_aversion=best["risk_aversion"],
        tx_cost_bps=best["tx_cost_bps"],
        n_pca_factors=int(best["n_pca_factors"]),
        cov_shrinkage=0.9,
        optimizer_lookback=120,
    )
    print(f"[{time.time()-t0:5.1f}s] Evaluating best config on TEST...", flush=True)
    df_test = run_qp_on_slice(test_rows, tickers, close_df_all, best_cfg)
    df_test["date"] = [dates[i] for i in df_test["bar_idx"]]
    df_test.to_csv(RESULTS_DIR / "qp_best_test.csv", index=False)
    s_test = summarize(df_test, "best_test")

    # Also run best config on VAL for plot
    df_val_best = run_qp_on_slice(val_rows, tickers, close_df_all, best_cfg)
    df_val_best["date"] = [dates[i] for i in df_val_best["bar_idx"]]

    # Final report
    print(f"\n[{time.time()-t0:5.1f}s] ========= FINAL RESULTS =========")
    print(f"  Best config: max_TO={best['max_turnover']:.2f}  kappa={best['risk_aversion']:.1f}  "
          f"tx_cost={best['tx_cost_bps']:.0f}bps  gross_cap={best['max_gross_leverage']:.1f}  "
          f"PCA={int(best['n_pca_factors'])}")
    print()
    print(f"                  {'Baseline (no QP)':>20s}  {'QP-optimized':>20s}")
    print(f"  VAL Net Sharpe  {base_val_s['net_sr_ann']:>+19.2f}  "
          f"{summarize(df_val_best)['net_sr_ann']:>+19.2f}")
    print(f"  VAL Net cum %   {base_val_s['net_cum_pct']:>+19.1f}  "
          f"{summarize(df_val_best)['net_cum_pct']:>+19.1f}")
    print(f"  VAL Turnover    {base_val_s['avg_turnover']*100:>19.1f}  "
          f"{summarize(df_val_best)['avg_turnover']*100:>19.1f}")
    print()
    print(f"  TEST Net Sharpe {base_test_s['net_sr_ann']:>+19.2f}  "
          f"{s_test['net_sr_ann']:>+19.2f}")
    print(f"  TEST Net cum %  {base_test_s['net_cum_pct']:>+19.1f}  "
          f"{s_test['net_cum_pct']:>+19.1f}")
    print(f"  TEST Gross SR   {base_test_s['gross_sr_ann']:>+19.2f}  "
          f"{s_test['gross_sr_ann']:>+19.2f}")
    print(f"  TEST Gross cum% {base_test_s['gross_cum_pct']:>+19.1f}  "
          f"{s_test['gross_cum_pct']:>+19.1f}")
    print(f"  TEST Turnover   {base_test_s['avg_turnover']*100:>19.1f}  "
          f"{s_test['avg_turnover']*100:>19.1f}")
    print()

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    base_val["date"] = [dates[i] for i in base_val["bar_idx"]]
    base_test["date"] = [dates[i] for i in base_test["bar_idx"]]

    for ax, (series_base, series_qp, title) in zip(
        axes.flat,
        [
            (base_val, df_val_best, "VAL — Gross cum return"),
            (base_test, df_test, "TEST — Gross cum return"),
            (base_val, df_val_best, "VAL — Net cum return (3 bps)"),
            (base_test, df_test, "TEST — Net cum return (3 bps)"),
        ],
    ):
        key = "gross" if "Gross" in title else "net_3bps"
        ax.plot(series_base["date"], series_base[key].cumsum() * 100,
                label="P=1000 raw (no QP)", color="tab:blue", linewidth=1.5)
        ax.plot(series_qp["date"], series_qp[key].cumsum() * 100,
                label="P=1000 + QP (tuned on val)", color="tab:red", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("Cumulative return (%)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.legend(loc="upper left")

    fig.suptitle(
        f"P=1000 AIPT + Execution QP (walkforward.optimize_portfolio)\n"
        f"max_TO={best['max_turnover']:.2f}  kappa={best['risk_aversion']:.1f}  "
        f"tx_cost={best['tx_cost_bps']:.0f}bps  PCA={int(best['n_pca_factors'])}",
        fontsize=13,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "qp_execution_val_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[{time.time()-t0:5.1f}s] Figure: {out}")

    # Save best config to JSON
    (RESULTS_DIR / "qp_best_config.json").write_text(json.dumps({
        "max_position": 0.08,
        "max_gross_leverage": float(best["max_gross_leverage"]),
        "max_turnover": float(best["max_turnover"]),
        "risk_aversion": float(best["risk_aversion"]),
        "tx_cost_bps": float(best["tx_cost_bps"]),
        "n_pca_factors": int(best["n_pca_factors"]),
        "cov_shrinkage": 0.9,
        "optimizer_lookback": 120,
        "val_period": [str(val_dates[0]), str(val_dates[1])],
        "test_period": [str(test_dates[0]), str(test_dates[1])],
        "val_net_sr": summarize(df_val_best)["net_sr_ann"],
        "test_net_sr": s_test["net_sr_ann"],
        "test_net_cum_pct": s_test["net_cum_pct"],
    }, indent=2))
    print(f"[{time.time()-t0:5.1f}s] Best config saved.")
    print(f"[{time.time()-t0:5.1f}s] DONE. Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
