"""
Catchup script for AIPT traders after missed scheduled runs.

Processes each bar from state.last_bar_time + 1 to the latest matrix bar,
one at a time, updating state / equity CSV / trade JSON exactly as if the
scheduler had fired on every bar.

Usage:
    python prod/catchup_aipt.py aipt          # P=100
    python prod/catchup_aipt.py aipt_p1000    # P=1000
"""
from __future__ import annotations
import argparse, datetime as dt, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INTERVAL        = "4h"
TRAIN_BARS      = 4380
MIN_TRAIN_BARS  = 1000
REBAL_EVERY     = 12
COVERAGE_CUTOFF = 0.3
Z_RIDGE         = 1e-3
SEED            = 42
GAMMA_GRID      = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAKER_BPS       = 3.0
TARGET_GMV      = 100_000
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
MATRICES_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h/prod"
UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"


def generate_rff_params(D, P, seed=SEED):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((P // 2, D))
    gamma = rng.choice(GAMMA_GRID, size=P // 2)
    return theta, gamma


def build_Z(matrices, tickers, bar_idx, available_chars):
    N, D = len(tickers), len(available_chars)
    Z = np.full((N, D), np.nan)
    for j, cn in enumerate(available_chars):
        Z[:, j] = matrices[cn][tickers].iloc[bar_idx].reindex(tickers).values.astype(np.float64)
    for j in range(D):
        col = Z[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 3:
            Z[:, j] = 0.0
            continue
        r = rankdata(col[ok], method="average") / ok.sum() - 0.5
        Z[ok, j] = r
        Z[~ok, j] = 0.0
    return Z


def rff(Z, theta, gamma, P):
    proj = (Z @ theta.T) * gamma[None, :]
    S = np.empty((Z.shape[0], P))
    S[:, 0::2] = np.sin(proj)
    S[:, 1::2] = np.cos(proj)
    return S


def estimate_ridge_markowitz(F_train, z, P):
    T = F_train.shape[0]
    mu = F_train.mean(axis=0)
    if P <= T:
        FF = (F_train.T @ F_train) / T
        A = z * np.eye(P) + FF
        return np.linalg.solve(A, mu)
    # Woodbury for P > T (P=1000 case)
    FFT = F_train @ F_train.T
    A_T = z * T * np.eye(T) + FFT
    F_mu = F_train @ mu
    inv_F_mu = np.linalg.solve(A_T, F_mu)
    return (mu - F_train.T @ inv_F_mu) / z


def run_catchup(strategy: str):
    assert strategy in ("aipt", "aipt_p1000")
    P_FACTORS = 100 if strategy == "aipt" else 1000
    STATE_DIR = PROJECT_ROOT / f"prod/state/{strategy}"
    LOG_DIR   = PROJECT_ROOT / f"prod/logs/kucoin/{strategy}"
    TRADE_LOG_DIR = LOG_DIR / "trades"
    EQUITY_FILE = LOG_DIR / "performance" / f"equity_{strategy}.csv"

    print(f"=== CATCHUP: {strategy} (P={P_FACTORS}) ===")

    # Load matrices
    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        matrices[fp.stem] = pd.read_parquet(fp)

    # Load universe + filter tickers
    universe_df = pd.read_parquet(UNIVERSE_PATH)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    all_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    tickers = [t for t in all_tickers if t in matrices["close"].columns]
    N = len(tickers)
    close = matrices["close"][tickers]
    dates = close.index
    close_vals = close.values

    available_chars = [c for c in CHAR_NAMES if c in matrices]
    D = len(available_chars)
    theta, gamma = generate_rff_params(D, P_FACTORS, seed=SEED)

    # Load state
    state = json.loads((STATE_DIR / "aipt_state.json").read_text())
    w_data = np.load(STATE_DIR / "weights.npz")
    lambda_hat = w_data["lambda_hat"]
    prev_weights = w_data["prev_weights"]
    if prev_weights.size == 0:
        prev_weights = None
    bars_since_rebal = state["bars_since_rebal"]
    last_bar_time = state["last_bar_time"]
    if state["tickers"] != tickers:
        print(f"WARNING: state tickers != current tickers ({len(state['tickers'])} vs {len(tickers)})")

    # Load factor return history
    fr_idx = np.load(STATE_DIR / "factor_indices.npy", allow_pickle=True)
    fr_npz = np.load(STATE_DIR / "factor_returns.npz")
    fr_vecs = fr_npz["factor_returns"]
    fr_history = [(str(fr_idx[i]), fr_vecs[i]) for i in range(len(fr_idx))]
    # De-dup by timestamp (keep last)
    seen = {}
    for ts, v in fr_history:
        seen[ts] = v
    fr_history = sorted(seen.items())

    # Find bars to process (bars with index > state.last_bar_time)
    last_ts = pd.Timestamp(last_bar_time)
    bar_indices_to_process = [i for i, d in enumerate(dates) if d > last_ts]
    if not bar_indices_to_process:
        print(f"Nothing to process. State already at {last_bar_time}, matrix latest = {dates[-1]}")
        return
    print(f"State at {last_bar_time}. Matrix latest = {dates[-1]}.")
    print(f"Processing {len(bar_indices_to_process)} bar(s): {[str(dates[i]) for i in bar_indices_to_process]}")
    print()

    for t_last in bar_indices_to_process:
        bar_time = dates[t_last]
        # Factor return for this bar, using Z at t_last-1
        Z_prev = build_Z(matrices, tickers, t_last - 1, available_chars)
        S_prev = rff(Z_prev, theta, gamma, P_FACTORS)
        R_t = (close_vals[t_last] - close_vals[t_last - 1]) / close_vals[t_last - 1]
        R_t = np.nan_to_num(R_t, nan=0.0)
        valid = (~np.isnan(Z_prev).any(axis=1)
                 & ~np.isnan(close_vals[t_last]) & ~np.isnan(close_vals[t_last - 1]))
        N_t = int(valid.sum())
        if N_t >= 5:
            F_t = (1.0 / np.sqrt(N_t)) * (S_prev[valid].T @ R_t[valid])
            fr_history.append((str(bar_time), F_t))
        else:
            print(f"  {bar_time}: only {N_t} valid assets, skipping factor return")

        # Trim history
        fr_history = fr_history[-(TRAIN_BARS + 500):]

        # Should we re-estimate lambda?
        bars_since_rebal += 1
        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            train = [fr for ts, fr in fr_history if ts < str(bar_time)][-TRAIN_BARS:]
            if len(train) >= MIN_TRAIN_BARS:
                F_train = np.vstack(train)
                lambda_hat = estimate_ridge_markowitz(F_train, Z_RIDGE, P_FACTORS)
                bars_since_rebal = 0
                print(f"  {bar_time}: LAMBDA re-estimated from {len(train)} bars (|lam|_mean={abs(lambda_hat).mean():.4f})")

        # Target portfolio from Z at t_last
        Z_t = build_Z(matrices, tickers, t_last, available_chars)
        S_t = rff(Z_t, theta, gamma, P_FACTORS)
        valid_now = ~np.isnan(Z_t).any(axis=1)
        N_valid = int(valid_now.sum())
        raw_w = np.zeros(N)
        raw_w[valid_now] = (1.0 / np.sqrt(N_valid)) * (S_t[valid_now] @ lambda_hat)
        abs_sum = np.abs(raw_w).sum()
        if abs_sum < 1e-12:
            print(f"  {bar_time}: zero weights — SKIPPING")
            continue
        w_norm = raw_w / abs_sum
        n_long = int((w_norm > 1e-6).sum())
        n_short = int((w_norm < -1e-6).sum())

        # Realized return + turnover + fee
        if prev_weights is not None and len(prev_weights) == N:
            turnover = float(np.abs(w_norm - prev_weights).sum() / 2.0)
            port_return = float(prev_weights @ R_t)
        else:
            turnover = 0.0
            port_return = 0.0
        fee_drag = turnover * TAKER_BPS / 10000 * 2
        port_return_net = port_return - fee_drag

        # Write equity row
        ts_iso = dt.datetime.utcnow().isoformat()
        with open(EQUITY_FILE, "a") as f:
            f.write(f"{ts_iso},{bar_time},{n_long},{n_short},{turnover:.6f},"
                    f"{port_return:.8f},{port_return_net:.8f},{TARGET_GMV}\n")

        # Write trade JSON
        tag = "aipt" if strategy == "aipt" else "aipt_p1000"
        trade_path = TRADE_LOG_DIR / f"{tag}_{ts_iso.replace(':', '-')}.json"
        trade_json = {
            "timestamp": ts_iso, "bar_time": str(bar_time),
            "model": {"P": P_FACTORS, "z": Z_RIDGE, "seed": SEED, "rebal": REBAL_EVERY},
            "portfolio": {
                "n_long": n_long, "n_short": n_short, "target_gmv": TARGET_GMV,
                "turnover": round(turnover, 6),
                "port_return_gross": round(port_return, 8),
                "port_return_net": round(port_return_net, 8),
            },
            "weights": {t: round(float(w), 6) for t, w in zip(tickers, w_norm) if abs(w) > 1e-6},
        }
        trade_path.write_text(json.dumps(trade_json, indent=2))

        print(f"  {bar_time}: {n_long}L/{n_short}S  TO={turnover:.4f}  Gross={port_return*10000:+6.2f}bps  Net={port_return_net*10000:+6.2f}bps")

        prev_weights = w_norm
        last_bar_time = str(bar_time)

    # Save factor returns
    ts_arr = np.array([ts for ts, _ in fr_history], dtype="U30")
    vecs = np.vstack([v for _, v in fr_history])
    np.save(STATE_DIR / "factor_indices.npy", ts_arr)
    np.savez_compressed(STATE_DIR / "factor_returns.npz", factor_returns=vecs)

    # Save state
    state["last_bar_time"] = last_bar_time
    state["bars_since_rebal"] = bars_since_rebal
    (STATE_DIR / "aipt_state.json").write_text(json.dumps(state, indent=2))
    np.savez(STATE_DIR / "weights.npz", lambda_hat=lambda_hat, prev_weights=prev_weights)

    print(f"\nFinal state: last_bar_time={last_bar_time}, bars_since_rebal={bars_since_rebal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", choices=["aipt", "aipt_p1000"])
    args = parser.parse_args()
    run_catchup(args.strategy)
