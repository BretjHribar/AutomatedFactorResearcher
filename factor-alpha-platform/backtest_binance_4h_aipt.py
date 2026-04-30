"""
AIPT (DKKM) on Binance 4h futures — same pipeline as backtest_binance_daily_aipt.py.

  - INTERVAL = 4h
  - 6 bars/day, sqrt(6*365) annualization
  - REBAL_EVERY = 12 (every 2 days, like KuCoin script)
  - TRAIN_BARS = 4380 (~2 years 4h)
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

UNIVERSE        = "BINANCE_TOP100_4h"
INTERVAL        = "4h"
BARS_PER_YEAR   = 6.0 * 365     # 6 bars/day, 365d
TRAIN_BARS      = 4380          # ~2 years
MIN_TRAIN_BARS  = 1500
REBAL_EVERY     = 12            # every 2 days
OOS_START       = "2022-09-01"
COVERAGE_CUTOFF = 0.30
Z_RIDGE         = 1e-3
SEED            = 42
GAMMA_GRID      = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAKER_BPS       = 3.0
P               = 1000

CHAR_NAMES = [
    "adv20", "adv60", "beta_to_btc", "close_position_in_range",
    "dollars_traded", "high_low_range",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "log_returns", "momentum_5d", "momentum_20d", "momentum_60d",
    "open_close_range",
    "parkinson_volatility_10", "parkinson_volatility_20", "parkinson_volatility_60",
    "quote_volume",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d", "vwap_deviation",
]

MATRICES_DIR  = PROJECT_ROOT / "data/binance_cache/matrices/4h"
UNIVERSE_PATH = PROJECT_ROOT / "data/binance_cache/universes" / f"{UNIVERSE}.parquet"
RESULTS_DIR   = PROJECT_ROOT / "data/aipt_results/binance_4h"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print(f"Loading universe {UNIVERSE} from {UNIVERSE_PATH}", flush=True)
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Universe parquet missing: {UNIVERSE_PATH}")
    universe_df = pd.read_parquet(UNIVERSE_PATH)
    cov = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    print(f"  {len(valid_tickers)} tickers pass coverage > {COVERAGE_CUTOFF}", flush=True)

    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)
    close_vals = matrices["close"].values.astype(np.float64)
    dates = matrices["close"].index
    available_chars = [c for c in CHAR_NAMES if c in matrices]
    print(f"  T={len(dates)} N={len(tickers)} D_chars={len(available_chars)}", flush=True)
    print(f"  Date range: {dates.min()} -> {dates.max()}", flush=True)
    print(f"  Missing chars: {sorted(set(CHAR_NAMES) - set(available_chars))}", flush=True)
    return matrices, tickers, dates, close_vals, available_chars


def build_Z_panel(matrices, tickers, chars, start, end):
    N = len(tickers)
    D = len(chars)
    panel = {}
    for t in range(start, end):
        Z = np.full((N, D), np.nan)
        for j, cn in enumerate(chars):
            Z[:, j] = matrices[cn].iloc[t].reindex(tickers).values.astype(np.float64)
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
    return panel, D


def run_aipt(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, seed=SEED):
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history = []
    lambda_hat = None
    bars_since_rebal = REBAL_EVERY
    prev_weights = None
    rows = []

    for t in range(start_bar, T_total - 1):
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        c_t = close_vals[t]
        c_t1 = close_vals[t + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            R_t1 = np.where(c_t > 0, (c_t1 - c_t) / c_t, 0.0)
        R_t1 = np.nan_to_num(R_t1, nan=0.0, posinf=0.0, neginf=0.0)

        valid = (~np.isnan(Z_t).any(axis=1) & ~np.isnan(c_t) & ~np.isnan(c_t1) & (c_t > 0))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v = S_t[valid]
        R_v = R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            T_tr, P_tr = F_train.shape
            if P_tr <= T_tr:
                FF = (F_train.T @ F_train) / T_tr
                A = Z_RIDGE * np.eye(P_tr) + FF
                lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            else:
                mu = F_train.mean(axis=0)
                FFT = F_train @ F_train.T
                A_T = Z_RIDGE * T_tr * np.eye(T_tr) + FFT
                inv_F_mu = np.linalg.solve(A_T, F_train @ mu)
                lambda_hat = (mu - F_train.T @ inv_F_mu) / Z_RIDGE
            bars_since_rebal = 0

        raw_w = np.zeros(Z_t.shape[0])
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
    if df.empty:
        return df
    for bps in [0, 1, 3, 5]:
        df[f"net_{bps}bps"] = df["gross"] - df["turnover"] * bps / 10000.0 * 2.0
    return df


def metrics(df, ann=BARS_PER_YEAR):
    if df.empty:
        return {}
    out = {"n_bars": len(df)}
    for col in ["gross", "net_0bps", "net_1bps", "net_3bps", "net_5bps"]:
        if col in df.columns:
            r = df[col]
            mu, sd = r.mean(), r.std(ddof=1)
            out[f"{col}_sharpe"] = float(mu / sd * np.sqrt(ann)) if sd > 1e-12 else np.nan
            out[f"{col}_ann_ret"] = float(mu * ann)
    out["avg_turnover_per_bar"] = float(df["turnover"].mean())
    out["avg_turnover_per_day"] = float(df["turnover"].mean() * 6)
    return out


def main():
    t0 = time.time()
    print("=" * 80)
    print(f"BINANCE 4H AIPT  (P={P}, chars-only)")
    print("=" * 80)

    matrices, tickers, dates, close_vals, chars = load_data()
    T_total = len(dates)

    oos_start_idx = next((i for i, d in enumerate(dates) if str(d) >= OOS_START), T_total)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  oos_start_idx={oos_start_idx} ({dates[oos_start_idx]}), "
          f"start_bar={start_bar} ({dates[start_bar]})", flush=True)

    print("\n[1/2] Building Z panel...", flush=True)
    ts = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total)
    print(f"  D={D}  ({time.time()-ts:.1f}s)", flush=True)

    print(f"\n[2/2] Running AIPT (P={P}, train={TRAIN_BARS}, rebal={REBAL_EVERY}, seed={SEED})...", flush=True)
    ts = time.time()
    df = run_aipt(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, seed=SEED)
    print(f"  done ({time.time()-ts:.1f}s) — {len(df)} OOS bars", flush=True)

    if df.empty:
        print("  EMPTY result")
        return

    df["date"] = [dates[i] for i in df["bar_idx"]]
    df.to_csv(RESULTS_DIR / "binance_4h_returns.csv", index=False)
    m = metrics(df)
    print("\n## METRICS")
    for k, v in m.items():
        print(f"  {k:<28} {v if isinstance(v,int) else f'{v:.4f}'}")
    json.dump(m, open(RESULTS_DIR / "binance_4h_metrics.json", "w"), indent=2)

    eq = (1 + df["gross"]).cumprod()
    eq_net = (1 + df["net_3bps"]).cumprod()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], eq, label="gross", lw=1.4)
    ax.plot(df["date"], eq_net, label="net 3bps", lw=1.2, alpha=0.85)
    ax.axhline(1, color="gray", ls=":", alpha=0.4)
    ax.set_yscale("log")
    ax.set_title(f"Binance 4h AIPT — P={P}, D={D}, {dates[oos_start_idx].date()} → {df['date'].iloc[-1].date()}\n"
                 f"Gross SR={m['gross_sharpe']:.2f}  Net3bps SR={m['net_3bps_sharpe']:.2f}  TO/bar={m['avg_turnover_per_bar']*100:.1f}%")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "binance_4h_equity.png", dpi=110)
    plt.close(fig)

    print(f"\n## DONE ({(time.time()-t0)/60:.1f}min) — outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
