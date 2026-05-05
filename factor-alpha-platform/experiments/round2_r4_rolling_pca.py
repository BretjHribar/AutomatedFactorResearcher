"""
Round 2 — R4: rolling-window PCA at signal stage.

Theory (from H5 + Bianchi-Babiak): static PCA on TRAIN was a wash. Bianchi-
Babiak found IPCA with TIME-VARYING loadings beats observable factors. A
practical proxy is rolling-window PCA: re-fit eigenvectors over the trailing
N bars at each rebalance, residualize the signal, then trade.

Hypothesis: rolling PCA outperforms static PCA at the same K. Sweet spot
window is ~120-360 bars (20-60 days) — long enough to be stable, short enough
to track changing factor structure.

Implementation note: rolling residualization for every bar is O(N²×T×T_window)
which is slow. Approximation: compute PCA loadings at each rebalance (every
20d = 120 bars) and hold constant in between. Eigenvalue decomposition of
~30×30 covariance is fast.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"experiments"))
from universe_experiments import (
    build_universe_topn_rebal, eval_signal,
    split_metrics, load_all,
    BENCHMARK_EXPR, BARS_PER_DAY, OUT, log,
)


def signal_to_portfolio_rolling_pca(sig, universe, returns,
                                     k: int = 3,
                                     window_bars: int = 360,
                                     update_every_bars: int = 120,
                                     max_wt: float = 0.10):
    """Demean → rolling PCA residualize → gross-norm → clip."""
    common = sig.columns.intersection(universe.columns)
    s = sig[common].replace([np.inf, -np.inf], np.nan)
    uni_mask = universe.reindex(index=s.index, columns=common).fillna(False).astype(bool)
    s = s.where(uni_mask, np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)

    ret = returns.reindex(index=s.index, columns=common).fillna(0)
    R = ret.values
    D = demean.fillna(0).values

    n_bars = len(s); n_cols = len(common)
    out = np.zeros_like(D)

    # Compute PCA loadings at each update bar; hold between
    U_K_current = np.zeros((n_cols, k))
    for i in range(n_bars):
        if i < window_bars:
            # Warmup: no PCA yet; pass through
            out[i] = D[i]
            continue
        if i == window_bars or (i - window_bars) % update_every_bars == 0:
            R_win = R[i-window_bars:i]
            R_win = R_win - R_win.mean(axis=0, keepdims=True)
            cov = R_win.T @ R_win / max(window_bars - 1, 1)
            cov = (cov + cov.T) * 0.5 + 1e-10 * np.eye(n_cols)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                U_K_current = eigvecs[:, order[:k]]
            except np.linalg.LinAlgError:
                pass  # keep previous
        # Project out current PCs
        d = D[i]
        out[i] = d - U_K_current @ (U_K_current.T @ d)

    out_df = pd.DataFrame(out, index=demean.index, columns=demean.columns)
    out_df = out_df.where(uni_mask, np.nan)
    gross = out_df.abs().sum(axis=1).replace(0, np.nan)
    w = out_df.div(gross, axis=0)
    w = w.clip(lower=-max_wt, upper=max_wt)
    return w.fillna(0)


def main():
    matrices = load_all()
    sig = eval_signal(BENCHMARK_EXPR, matrices)
    log("R4: rolling-PCA at signal stage")
    uni = build_universe_topn_rebal(matrices["adv20"], matrices["close"],
                                     top_n=30, rebal_bars=20*BARS_PER_DAY,
                                     min_history_days=365)
    rows = []
    configs = [
        ("none",        None, None, None),
        ("static_K3",   3, None, None),  # marker for static; handled specially
        ("rolling_K1_w360",  1, 360, 120),
        ("rolling_K3_w360",  3, 360, 120),
        ("rolling_K5_w360",  5, 360, 120),
        ("rolling_K3_w120",  3, 120, 30),
        ("rolling_K3_w720",  3, 720, 120),
        ("rolling_K3_w180",  3, 180, 60),
    ]
    for name, k, win, upd in configs:
        t0 = time.time()
        if k is None:
            # baseline: no risk model
            from universe_experiments import signal_to_portfolio
            w = signal_to_portfolio(sig, uni)
        elif name == "static_K3":
            from universe_experiments import signal_to_portfolio
            w = signal_to_portfolio(sig, uni, risk_residualize="pca",
                                    residualize_args={"k": 3, "returns": matrices["returns"]})
        else:
            w = signal_to_portfolio_rolling_pca(sig, uni, matrices["returns"],
                                                k=k, window_bars=win, update_every_bars=upd)
        m = split_metrics(w, matrices["returns"], fee_bps=3.0)
        m.update({"name": name, "k": k or 0, "window_bars": win or 0,
                  "update_every": upd or 0, "t_sec": time.time()-t0})
        rows.append(m)
        print(f"  {name:22s} TR={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} "
              f"DD={m['VT_dd_n']*100:+.0f}% TO={m['to_per_bar']:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT/"round2_r4_rolling_pca.csv", index=False, float_format="%.4f")
    log("R4 DONE")


if __name__ == "__main__":
    main()
