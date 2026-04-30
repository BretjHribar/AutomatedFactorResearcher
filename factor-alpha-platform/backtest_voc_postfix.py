"""
Post-fix backtest comparing three feature sets at P=1000:
  1. Random projections — chars only  (original behaviour)
  2. Alphas only        — 18 KuCoin 4h alpha signals as Z features
  3. Both              — chars + alpha signals concatenated as Z features

Alpha expressions are evaluated with FastExpressionEngine (identical to
backtest_wq_alphas_kucoin.py) so the per-bar cross-sectional values match.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.fastexpression import FastExpressionEngine

# ── Config ────────────────────────────────────────────────────────────────────
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
TAKER_BPS       = 3.0
P               = 1000
TAU_GRID        = [0.0, 0.1, 1.0, 10.0]   # quadratic turnover-penalty strengths

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

ALPHA_EXPRS = [
    "zscore_cs(true_divide(sma(log_returns, 120), df_max(sma(parkinson_volatility_20, 120), 0.001)))",
    "zscore_cs(sma(upper_shadow, 30))",
    "zscore_cs(sma(ts_zscore(turnover, 240), 60))",
    "Abs(momentum_20d)",
    "ts_zscore(ts_sum(multiply(momentum_20d, returns), 180), 180)",
    "add(ts_rank(low, 240), open_close_range)",
    "df_min(ts_rank(adv20, 90), Decay_exp(square(momentum_5d), 36))",
    "ts_zscore(rank(adv20), 240)",
    "stddev(ts_sum(momentum_20d, 60), 6)",
    "stddev(multiply(volume_momentum_5_20, momentum_20d), 18)",
    "ts_zscore(normalize(adv20), 90)",
    "Decay_exp(sqrt(momentum_20d), 6)",
    "log(ts_max(momentum_20d, 36))",
    "multiply(momentum_5d, add(momentum_20d, df_min(momentum_20d, momentum_5d)))",
    "stddev(stddev(momentum_20d, 60), 6)",
    "Abs(multiply(subtract(Decay_exp(momentum_20d, 12), zscore_cs(high)), parkinson_volatility_10))",
    "rank(ts_zscore(square(momentum_20d), 240))",
    "rank(ts_zscore(stddev(momentum_60d, 30), 240))",
]

MATRICES_DIR  = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes" / f"{UNIVERSE}_{INTERVAL}.parquet"
RESULTS_DIR   = PROJECT_ROOT / "data/aipt_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    """Load all matrices, evaluate alpha expressions, return everything needed."""
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

    # Consistent ticker set across all matrices (same pattern as backtest_wq_alphas_kucoin.py)
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    close_vals = matrices["close"].values
    dates      = matrices["close"].index
    available_chars = [c for c in CHAR_NAMES if c in matrices]

    # Evaluate alpha expressions with FastExpressionEngine — identical to
    # backtest_wq_alphas_kucoin.py so values are the same.
    engine = FastExpressionEngine(data_fields=matrices)
    alpha_dfs = []
    for i, expr in enumerate(ALPHA_EXPRS):
        try:
            sig = engine.evaluate(expr)
            sig = sig.reindex(columns=tickers)
            alpha_dfs.append(sig)
            print(f"  alpha {i+1:>2}: ok   {expr[:70]}")
        except Exception as e:
            print(f"  alpha {i+1:>2}: FAIL {e!r}  ({expr[:60]})")

    print(f"  {len(alpha_dfs)}/{len(ALPHA_EXPRS)} alphas evaluated")
    return matrices, tickers, dates, close_vals, available_chars, alpha_dfs


# ── Z-panel construction ──────────────────────────────────────────────────────

def build_Z_panel(matrices, tickers, available_chars, alpha_dfs, start, end, mode):
    """
    Build rank-normalised feature panel for bars [start, end).

    mode 1 — chars only
    mode 2 — alphas only
    mode 3 — chars + alphas
    """
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
        # Rank-normalise each feature column (same as original build_Z_panel)
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


# ── AIPT bar-by-bar backtest ──────────────────────────────────────────────────

def run_one(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, seed=SEED):
    """Bar-by-bar AIPT backtest. Returns DataFrame with gross/net_3bps/turnover."""
    rng    = np.random.default_rng(seed)
    n_pairs = P // 2
    theta  = rng.standard_normal((n_pairs, D))
    gamma  = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history     = []
    lambda_hat     = None
    bars_since_rebal = REBAL_EVERY
    prev_weights   = None
    rows           = []

    for t in range(start_bar, T_total - 1):
        Z_t  = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t  = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v  = S_t[valid]
        R_v  = R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train  = np.vstack(train)
            T_tr, P_tr = F_train.shape
            if P_tr <= T_tr:
                FF  = (F_train.T @ F_train) / T_tr
                A   = Z_RIDGE * np.eye(P_tr) + FF
                lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            else:
                mu       = F_train.mean(axis=0)
                FFT      = F_train @ F_train.T
                A_T      = Z_RIDGE * T_tr * np.eye(T_tr) + FFT
                inv_F_mu = np.linalg.solve(A_T, F_train @ mu)
                lambda_hat = (mu - F_train.T @ inv_F_mu) / Z_RIDGE
            bars_since_rebal = 0

        raw_w   = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_weights).sum() / 2.0) if prev_weights is not None else 0.0
        prev_weights = w_norm.copy()
        bars_since_rebal += 1

        rows.append({"bar_idx": t + 1, "gross": port_ret, "turnover": to})

    df = pd.DataFrame(rows)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def run_one_tc(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
               tau=0.0, seed=SEED):
    """Variant of run_one with an asset-space quadratic turnover penalty
    (Gârleanu-Pedersen). At each rebalance solve

        (FF/T + ρI + τ·SₜᵀSₜ/N) λ = μ + τ·(SₜᵀSₜ/N) λ_prev

    StS/N ≈ I for well-mixed sin/cos features, so τ is on the same scale as
    the existing ridge ρ = Z_RIDGE. tau=0 reproduces run_one numerically.
    """
    rng    = np.random.default_rng(seed)
    n_pairs = P // 2
    theta  = rng.standard_normal((n_pairs, D))
    gamma  = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history     = []
    lambda_hat     = None
    bars_since_rebal = REBAL_EVERY
    prev_weights   = None
    rows           = []

    for t in range(start_bar, T_total - 1):
        Z_t  = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t  = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v  = S_t[valid]
        R_v  = R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train  = np.vstack(train)
            T_tr, P_tr = F_train.shape
            if P_tr <= T_tr:
                FF  = (F_train.T @ F_train) / T_tr
                A   = Z_RIDGE * np.eye(P_tr) + FF
                rhs = F_train.mean(axis=0)
                if tau > 0 and lambda_hat is not None:
                    StS = (S_v.T @ S_v) / N_t
                    A   = A + tau * StS
                    rhs = rhs + tau * (StS @ lambda_hat)
                lambda_hat = np.linalg.solve(A, rhs)
            else:
                # Woodbury branch (P>T) — penalty not applied; not hit at P=1000.
                mu       = F_train.mean(axis=0)
                FFT      = F_train @ F_train.T
                A_T      = Z_RIDGE * T_tr * np.eye(T_tr) + FFT
                inv_F_mu = np.linalg.solve(A_T, F_train @ mu)
                lambda_hat = (mu - F_train.T @ inv_F_mu) / Z_RIDGE
            bars_since_rebal = 0

        raw_w   = np.zeros(Z_t.shape[0])
        raw_w[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_weights).sum() / 2.0) if prev_weights is not None else 0.0
        prev_weights = w_norm.copy()
        bars_since_rebal += 1

        rows.append({"bar_idx": t + 1, "gross": port_ret, "turnover": to})

    df = pd.DataFrame(rows)
    df["net_3bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data and evaluating alphas...")
    matrices, tickers, dates, close_vals, available_chars, alpha_dfs = load_data()
    T_total      = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar    = max(0, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)}  T={T_total}  OOS_start={oos_start_idx}  start_bar={start_bar}")
    print(f"  chars={len(available_chars)}  alphas={len(alpha_dfs)}")

    modes = {
        1: ("rp",     "RP (chars)"),
        2: ("alphas", "Alphas"),
        3: ("both",   "Both"),
    }

    results = []  # one row per (mode, tau)
    for mode, (short, label) in modes.items():
        D_c = len(available_chars) if mode != 2 else 0
        D_a = len(alpha_dfs)       if mode != 1 else 0
        print(f"\n[mode {mode}] {label}  D={D_c + D_a} ({D_c} chars + {D_a} alphas)")
        t1 = time.time()
        Z_panel, D = build_Z_panel(matrices, tickers, available_chars, alpha_dfs,
                                   start_bar, T_total, mode)
        print(f"  Z panel built in {time.time()-t1:.1f}s")

        for tau in TAU_GRID:
            t2 = time.time()
            if tau == 0.0:
                df = run_one(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D)
            else:
                df = run_one_tc(P, Z_panel, close_vals, start_bar, T_total,
                                oos_start_idx, D, tau=tau)
            df["date"] = [dates[i] for i in df["bar_idx"]]

            g_sr   = df["gross"].mean()    / df["gross"].std(ddof=1)    * np.sqrt(BARS_PER_YEAR)
            n_sr   = df["net_3bps"].mean() / df["net_3bps"].std(ddof=1) * np.sqrt(BARS_PER_YEAR)
            avg_to = df["turnover"].mean()
            g_cum  = df["gross"].sum()    * 100
            n_cum  = df["net_3bps"].sum() * 100
            print(f"  tau={tau:>5.2f}  bars={len(df)}  gSR={g_sr:+.2f}  nSR={n_sr:+.2f}  "
                  f"TO={avg_to*100:5.1f}%  gcum={g_cum:+6.1f}%  ncum={n_cum:+6.1f}%  "
                  f"({time.time()-t2:.1f}s)")

            csv_path = RESULTS_DIR / f"voc_P1000_{short}_tau{tau:g}.csv"
            df[["date", "gross", "net_3bps", "turnover"]].to_csv(csv_path, index=False)
            results.append({
                "mode": mode, "label": label, "tau": tau, "df": df,
                "g_sr": g_sr, "n_sr": n_sr, "avg_to": avg_to,
                "g_cum": g_cum, "n_cum": n_cum,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print(f"{'mode':<8}{'tau':>7}{'gross_SR':>10}{'net_SR':>9}{'TO%':>8}{'gcum%':>10}{'ncum%':>10}")
    print("-" * 86)
    for r in results:
        print(f"{r['label']:<8}{r['tau']:>7.2f}{r['g_sr']:>+10.2f}{r['n_sr']:>+9.2f}"
              f"{r['avg_to']*100:>7.1f}%{r['g_cum']:>+9.1f}%{r['n_cum']:>+9.1f}%")
    print("=" * 86)

    # ── Plot: SR/TO frontier + best curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors  = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    markers = {0.0: "o", 0.1: "s", 1.0: "^", 10.0: "D"}

    seen = set()
    for r in results:
        c = colors[r["mode"]]
        m = markers[r["tau"]]
        key = (r["mode"],)
        lbl = r["label"] if key not in seen else None
        seen.add(key)
        axes[0].scatter(r["avg_to"] * 100, r["n_sr"], color=c, marker=m, s=110,
                        edgecolors="black", linewidths=0.6, label=lbl)
        axes[0].annotate(f"τ={r['tau']:g}", (r["avg_to"] * 100, r["n_sr"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[0].set_xlabel("Average turnover per bar (%)")
    axes[0].set_ylabel("Net Sharpe (3 bps taker)")
    axes[0].set_title("SR/turnover frontier — quadratic TC penalty")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].legend(loc="lower right")

    for mode, (short, label) in modes.items():
        mode_results = [r for r in results if r["mode"] == mode]
        best = max(mode_results, key=lambda r: r["n_sr"])
        c = colors[mode]
        axes[1].plot(best["df"]["date"], best["df"]["net_3bps"].cumsum() * 100,
                     label=f"{best['label']}  τ={best['tau']:g}  nSR={best['n_sr']:+.2f}  "
                           f"TO={best['avg_to']*100:.1f}%",
                     color=c, linewidth=2)
    axes[1].set_title("Best net cum return per mode")
    axes[1].set_ylabel("Cumulative net return (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].legend(loc="upper left")

    fig.suptitle(
        f"AIPT P={P} with Gârleanu-Pedersen quadratic turnover penalty  (OOS: {OOS_START} →)",
        fontsize=13,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "voc_P1000_tau_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure: {out}")
    print(f"Total:  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
