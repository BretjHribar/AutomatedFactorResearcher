"""
AIPT P=1000 on US equities (TOP500 universe, daily bars), no QP.

Same RFF + ridge-Markowitz pipeline as the crypto script. Differences:
  - Universe: TOP500 by ADV20 (membership changes daily)
  - Bars: daily (BARS_PER_YEAR=252)
  - DELAY=1 (mandatory for equities): signal at bar t uses chars at t-1
  - 24 hand-picked equity characteristics (momentum / value / quality / vol / liquidity)

Outputs Sharpe, IC, R², turnover for VAL/TEST (50/50 split of OOS 2024-01-01 →).
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

# ── Config ───────────────────────────────────────────────────────────────────
UNIVERSE_NAME   = "TOP2000"
BARS_PER_YEAR   = 252            # daily equities
TRAIN_BARS      = 1500           # ~6 years
MIN_TRAIN_BARS  = 500            # ~2 years
REBAL_EVERY     = 5              # weekly rebal (5 trading days)
OOS_START       = "2024-01-01"
COVERAGE_CUTOFF = 0.5
Z_RIDGE         = 1e-3
SEED            = 42
GAMMA_GRID      = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAKER_BPS       = 1.0            # equities: low commission + spread, ~1bp aggregate
P               = 1000

# 24 equity characteristics — momentum, value, quality, vol, liquidity, growth
CHAR_NAMES = [
    # momentum / reversal
    "log_returns",           # 1d returns
    "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
    # value
    "book_to_market", "earnings_yield", "free_cashflow_yield",
    "ev_to_ebitda", "ev_to_revenue",
    # quality / profitability
    "roe", "roa", "gross_margin", "operating_margin", "net_margin",
    "asset_turnover",
    # liquidity / size
    "adv20", "adv60", "dollars_traded", "cap",
    # leverage
    "debt_to_equity", "current_ratio",
]

MATRICES_DIR  = PROJECT_ROOT / "data/fmp_cache/matrices_clean"
UNIVERSES_DIR = PROJECT_ROOT / "data/fmp_cache/universes"
RESULTS_DIR   = PROJECT_ROOT / "data/aipt_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load equity matrices, restrict to TOP500-eligible tickers, common date index."""
    uni = pd.read_parquet(UNIVERSES_DIR / f"{UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    print(f"  {UNIVERSE_NAME} tickers passing {COVERAGE_CUTOFF*100:.0f}% coverage: {len(valid_tickers)}")

    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.stem not in CHAR_NAMES + ["close"]:
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]

    # Align all matrices on common date index
    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)

    # Restrict to consistent ticker set
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    available_chars = [c for c in CHAR_NAMES if c in matrices]
    missing = [c for c in CHAR_NAMES if c not in matrices]
    if missing:
        print(f"  Missing chars: {missing}")
    print(f"  Loaded {len(available_chars)} chars, T={len(dates)}, N={len(tickers)}")
    return matrices, tickers, dates, close_vals, available_chars


def build_Z_panel(matrices, tickers, available_chars, start, end, delay=1):
    """Z_t built from chars at t-delay (delay=1 = standard equities setup)."""
    N = len(tickers)
    D = len(available_chars)
    panel = {}
    for t in range(start, end):
        z_idx = t - delay
        if z_idx < 0:
            continue
        Z = np.full((N, D), np.nan)
        for j, cn in enumerate(available_chars):
            Z[:, j] = matrices[cn].iloc[z_idx].reindex(tickers).values.astype(np.float64)
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


def run_with_ic(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D, seed=SEED):
    """Standard AIPT P=1000 with per-bar IC + R² capture. No QP, no smoothing."""
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    fr_history, lambda_hat = [], None
    bars_since_rebal = REBAL_EVERY
    prev_w = None
    rows = []

    for t in range(start_bar, T_total - 1):
        if t not in Z_panel:
            continue
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        # next-bar return (DELAY=1: w formed from Z_{t-1}, applied at bar t, earns over t->t+1)
        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v, R_v = S_t[valid], R_t1[valid]
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
            FF = (F_train.T @ F_train) / T_tr
            A = Z_RIDGE * np.eye(P_tr) + FF
            lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            bars_since_rebal = 0

        pred_v = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        # IC + R²
        if pred_v.std() > 1e-12 and R_v.std() > 1e-12:
            ic_p = float(np.corrcoef(pred_v, R_v)[0, 1])
            ic_s = float(np.corrcoef(rankdata(pred_v), rankdata(R_v))[0, 1])
            r2 = ic_p ** 2
        else:
            ic_p = ic_s = r2 = 0.0

        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = pred_v
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = raw_w / abs_sum

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        bars_since_rebal += 1

        rows.append({
            "bar_idx": t + 1, "gross": port_ret, "turnover": to,
            "ic_p": ic_p, "ic_s": ic_s, "r2": r2,
        })

    df = pd.DataFrame(rows)
    df["net_1bps"] = df["gross"] - df["turnover"] * TAKER_BPS / 10000.0 * 2.0
    return df


def main():
    t0 = time.time()
    print("Loading equity data...")
    matrices, tickers, dates, close_vals, available_chars = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} OOS_start={oos_start_idx} start_bar={start_bar}")
    print(f"  OOS dates: {dates[oos_start_idx]} -> {dates[-1]}  ({T_total - oos_start_idx} bars)")

    print(f"\nBuilding Z panel for {T_total - start_bar} bars (DELAY=1)...")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, available_chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s")

    print(f"\nRunning AIPT P={P} (no QP, no smoothing)...")
    t2 = time.time()
    df = run_with_ic(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D)
    print(f"  Done in {time.time()-t2:.1f}s, {len(df)} OOS bars")
    df["date"] = [dates[i] for i in df["bar_idx"]]

    # Stats — full OOS
    ann = np.sqrt(BARS_PER_YEAR)
    g, n = df["gross"].values, df["net_1bps"].values
    g_sr = g.mean() / g.std(ddof=1) * ann
    n_sr = n.mean() / n.std(ddof=1) * ann
    avg_to = df["turnover"].mean()
    g_cum = g.sum() * 100
    n_cum = n.sum() * 100
    ic_p_mean = df["ic_p"].mean()
    ic_s_mean = df["ic_s"].mean()
    ir_p = df["ic_p"].mean() / df["ic_p"].std(ddof=1) * ann
    ir_s = df["ic_s"].mean() / df["ic_s"].std(ddof=1) * ann
    r2_mean = df["r2"].mean()
    r2_med = df["r2"].median()

    print("\n" + "=" * 90)
    print(f"FULL OOS ({dates[oos_start_idx]} -> {dates[-1]}, {len(df)} bars)")
    print("=" * 90)
    print(f"  gross_SR  = {g_sr:+.3f}    net_SR (1bp) = {n_sr:+.3f}    avg_TO = {avg_to*100:.1f}%")
    print(f"  gross_cum = {g_cum:+.1f}%    net_cum      = {n_cum:+.1f}%")
    print(f"  IC_Pear   = {ic_p_mean:+.5f}   IR_Pear   = {ir_p:+.3f}")
    print(f"  IC_Spear  = {ic_s_mean:+.5f}   IR_Spear  = {ir_s:+.3f}")
    print(f"  R²_mean   = {r2_mean:.5f}    R²_median = {r2_med:.5f}")

    # VAL/TEST 50/50 split
    n_oos = len(df)
    split = n_oos // 2
    for tag, sub in [("VAL", df.iloc[:split]), ("TEST", df.iloc[split:])]:
        gg, nn = sub["gross"].values, sub["net_1bps"].values
        if len(nn) > 1 and nn.std() > 1e-12:
            sr_g = gg.mean() / gg.std(ddof=1) * ann
            sr_n = nn.mean() / nn.std(ddof=1) * ann
            ic_p = sub["ic_p"].mean()
            ir_p = sub["ic_p"].mean() / sub["ic_p"].std(ddof=1) * ann
            print(f"\n  {tag} ({sub['date'].iloc[0]} -> {sub['date'].iloc[-1]}, {len(sub)} bars):")
            print(f"    gross_SR={sr_g:+.3f}  net_SR={sr_n:+.3f}  TO={sub['turnover'].mean()*100:.1f}%  "
                  f"IC_p={ic_p:+.5f}  IR_p={ir_p:+.3f}  R²={sub['r2'].mean():.5f}")

    out_csv = RESULTS_DIR / "voc_equities_baseline.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV: {out_csv}")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
