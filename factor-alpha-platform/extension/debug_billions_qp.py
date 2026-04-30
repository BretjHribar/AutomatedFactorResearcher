"""
Instrument strategy_billions_qp to print:
  - How many QP solves succeeded vs fell back to prev_w
  - The actual weights_norm at a few sample bars
  - Std-dev of weights across alphas (if 0, all uniform)

Helps determine WHY BillionsQP collapses to ProperEqual at N>=35.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import eval_portfolio as ep
import cvxpy as cp
from sklearn import linear_model


def instrumented_billions_qp(raw_signals, returns_pct, close, universe,
                              max_wt=ep.MAX_WEIGHT, optim_lookback=120, qp_lookback=120,
                              rebal_every=12, risk_aversion=2.0, tc_penalty=0.5,
                              ema_halflife=60):
    K = len(raw_signals)
    print(f"\n  K={K} alphas")

    normed = {aid: ep.proper_normalize_alpha(raw, universe, max_wt=max_wt)
              for aid, raw in raw_signals.items()}
    dates = close.index
    aid_list = list(normed.keys())
    n_bars = len(dates)
    ret_df = returns_pct.reindex(index=dates, columns=close.columns)

    # Factor returns
    fr_data = {}
    for aid, n in normed.items():
        lagged = n.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        nn = lagged.div(ab, axis=0)
        fr_data[aid] = (nn * ret_df).sum(axis=1)
    fr_df = pd.DataFrame(fr_data, index=dates)
    print(f"  fr_df: shape={fr_df.shape}, NaN%={fr_df.isna().mean().mean()*100:.1f}%")

    # Billions stage 2 — count successful updates
    seed_exp_ret = fr_df.rolling(optim_lookback, min_periods=optim_lookback//2).mean().shift(1).clip(lower=0)
    billions_mu = seed_exp_ret.copy()
    n_bil_updates = 0; n_bil_skip_std = 0
    reg = linear_model.LinearRegression(fit_intercept=False)
    for test_start in range(1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback
        if optim_end + 1 >= n_bars: break
        try:
            bil_df = fr_df.iloc[test_start:optim_end].copy()
            demeaned = bil_df - bil_df.mean(axis=0)
            sample_std = demeaned.std(axis=0).replace(0, np.nan)
            if sample_std.isna().any():
                n_bil_skip_std += 1; continue
            normalized = demeaned.divide(sample_std)
            A_is = normalized.fillna(0.0)
            sub_exp = seed_exp_ret.iloc[test_start:optim_end].divide(sample_std).fillna(0.0)
            reg.fit(A_is.values, sub_exp.values)
            residuals = pd.DataFrame(reg.predict(A_is.values) - sub_exp.values,
                                       index=sub_exp.index, columns=sub_exp.columns)
            opt_w = residuals.divide(sample_std)
            row_sums = opt_w.sum(axis=1).replace(0, np.nan)
            opt_w = opt_w.div(row_sums, axis=0)
            billions_mu.iloc[optim_end + 1] = opt_w.iloc[-1].values
            n_bil_updates += 1
        except Exception:
            pass
    print(f"  Billions stage 2: {n_bil_updates} successful updates, "
          f"{n_bil_skip_std} skipped due to NaN sample_std")

    # QP stage — count successful solves vs fallbacks
    weights_arr = np.zeros((n_bars, K))
    prev_w = np.ones(K) / K
    n_qp_optimal = 0; n_qp_fallback_solver = 0; n_qp_skip_window = 0
    for t in range(qp_lookback, n_bars, rebal_every):
        window = fr_df.iloc[max(0, t - qp_lookback):t].dropna(axis=1, how='all')
        if len(window) < 20 or window.shape[1] < 2:
            weights_arr[t:min(t+rebal_every, n_bars)] = prev_w
            n_qp_skip_window += 1; continue
        active_ids = list(window.columns)
        K_a = len(active_ids)
        mu_row = billions_mu.iloc[t]
        mu = np.array([mu_row.get(aid, 0.0) for aid in active_ids])
        cov_raw = window.cov().values
        cov = ep._ledoit_wolf_shrink(cov_raw) + 1e-8 * np.eye(K_a)
        alpha_prev = np.array([prev_w[aid_list.index(aid)] for aid in active_ids])
        alpha = cp.Variable(K_a, nonneg=True)
        delta = cp.Variable(K_a)
        prob = cp.Problem(
            cp.Maximize(mu @ alpha - 0.5*risk_aversion*cp.quad_form(alpha, cov)
                          - tc_penalty*cp.norm1(delta)),
            [cp.sum(alpha) == 1.0, alpha <= 0.5, delta == alpha - alpha_prev],
        )
        solved = False
        for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True)
                if prob.status in ['optimal','optimal_inaccurate'] and alpha.value is not None:
                    solved = True; break
            except Exception:
                continue
        if solved:
            w = np.clip(alpha.value, 0, None)
            wsum = w.sum()
            w = w / wsum if wsum > 1e-10 else alpha_prev
            n_qp_optimal += 1
        else:
            w = alpha_prev
            n_qp_fallback_solver += 1
        w_full = np.zeros(K)
        for i, aid in enumerate(active_ids):
            w_full[aid_list.index(aid)] = w[i]
        prev_w = w_full
        weights_arr[t:min(t+rebal_every, n_bars)] = w_full
    weights_arr[:qp_lookback] = 1.0 / K
    print(f"  QP stage: {n_qp_optimal} optimal solves, "
          f"{n_qp_fallback_solver} solver fallbacks, "
          f"{n_qp_skip_window} window-skip")

    # Inspect weights
    weights_df = pd.DataFrame(weights_arr, index=dates, columns=aid_list)
    weights_smooth = weights_df.ewm(halflife=ema_halflife, min_periods=1).mean()
    wsum = weights_smooth.sum(axis=1).replace(0, np.nan)
    weights_norm = weights_smooth.div(wsum, axis=0).fillna(0)
    print(f"\n  weights_norm stats:")
    print(f"    mean across alphas (per bar): mean={weights_norm.mean(axis=1).mean():.6f} "
          f"(uniform={1.0/K:.6f})")
    print(f"    std across alphas (per bar): mean={weights_norm.std(axis=1).mean():.6f} "
          f"(uniform → 0)")
    print(f"    rows where every alpha = 1/K (uniform):  "
          f"{int((weights_norm.std(axis=1) < 1e-10).sum())} / {len(weights_norm)}")

    # Sample 5 specific rows
    sample_bars = [qp_lookback, qp_lookback+50, qp_lookback+200, qp_lookback+500, n_bars-1]
    print(f"\n  Sample weight rows (first 5 alphas shown):")
    for b in sample_bars:
        if b < len(weights_norm):
            r = weights_norm.iloc[b]
            print(f"    bar {b} ({weights_norm.index[b].date()}): "
                  f"first 5 = {r.iloc[:5].values.round(6).tolist()}, "
                  f"all-equal? {bool(r.std() < 1e-10)}")


def main():
    print("Loading 190 alphas...")
    raw, ret, close, univ = ep.load_raw_alpha_signals()
    aids = sorted(raw.keys())
    for N in [25, 35, 50, 100]:
        print("\n" + "="*80)
        print(f"N = {N}")
        print("="*80)
        subset = {aid: raw[aid] for aid in aids[:N]}
        instrumented_billions_qp(subset, ret, close, univ)


if __name__ == "__main__":
    main()
