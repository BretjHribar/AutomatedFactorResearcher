"""
Risk-model-agnostic name-level QP for alpha → per-stock weights.

Single-day problem:
    max  α'w − ½λ (Σ_k ||L_k' w||² + s²·w²) − Σ_i κ_i |w_i − w_prev,i|
    s.t. ‖w‖₁ ≤ max_gross_leverage
         |w_i| ≤ max_w
         sum(w) = 0          (if dollar_neutral)

The risk model is supplied as (L_list, s²) by a callable plugged in from
src/portfolio/risk_model.py. The QP itself knows nothing about PCA / style
factors / Barra.

Walk-forward driver: run_walkforward(alpha_df, close_df, ret_df, uni_df,
                                       risk_model_fn, **config).
"""
from __future__ import annotations
import time
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import cvxpy as cp


def solve_qp(alpha: np.ndarray,
             w_prev: np.ndarray,
             prices: np.ndarray,
             L_list: List[np.ndarray],
             s2: np.ndarray,
             *,
             lambda_risk: float = 5.0,
             kappa_tc: float = 30.0,
             max_w=0.02,                            # scalar or per-name (N,) array
             commission_per_share: float = 0.0045,
             impact_bps: float = 0.5,
             dollar_neutral: bool = True,
             max_gross_leverage: float = 1.0,
             ) -> Optional[np.ndarray]:
    """Single-day QP solve. Returns weight vector (N,) or None on failure.

    max_w may be a scalar or a per-name (N,) array for ADV-style caps.
    """
    n = len(alpha)
    kappa = kappa_tc * (commission_per_share / np.maximum(prices, 0.01)
                        + impact_bps / 1e4)
    w = cp.Variable(n)
    risk_terms = [cp.sum_squares(L.T @ w) for L in L_list]
    risk_terms.append(cp.sum(cp.multiply(s2, cp.square(w))))
    risk = cp.sum(risk_terms) if len(risk_terms) > 1 else risk_terms[0]
    tc = cp.sum(cp.multiply(kappa, cp.abs(w - w_prev)))
    obj = cp.Maximize(alpha @ w - 0.5 * lambda_risk * risk - tc)
    cons = [cp.norm(w, 1) <= max_gross_leverage,
            cp.abs(w) <= max_w]
    if dollar_neutral:
        cons.append(cp.sum(w) == 0)
    try:
        cp.Problem(obj, cons).solve(solver=cp.OSQP, warm_start=True, verbose=False)
        return w.value
    except Exception:
        return None


def run_walkforward(alpha: pd.DataFrame,
                    close: pd.DataFrame,
                    ret: pd.DataFrame,
                    uni: pd.DataFrame,
                    risk_model_fn: Callable,
                    *,
                    lambda_risk: float = 5.0,
                    kappa_tc: float = 30.0,
                    max_w: float = 0.02,
                    commission_per_share: float = 0.0045,
                    impact_bps: float = 0.5,
                    vol_window: int = 60,
                    factor_window: int = 126,
                    dollar_neutral: bool = True,
                    max_gross_leverage: float = 1.0,
                    # ADV-cap (capacity) inputs — all optional. When all 3 are
                    # provided, per-name cap = min(max_w, moc_frac * max_moc_participation * ADV_i / book).
                    adv: Optional[pd.DataFrame] = None,
                    book: Optional[float] = None,
                    moc_frac: float = 0.10,
                    max_moc_participation: float = 0.30,
                    label: str = "QP",
                    verbose: bool = True,
                    ) -> pd.DataFrame:
    """Walk-forward QP solve.

    risk_model_fn(day_idx, idx, vol_today_active, ret_mat, factor_window)
        → (L_list, s²)

    If `adv` and `book` are provided, an ADV-style per-name cap is enforced
    via the QP constraint |w_i| ≤ min(max_w, moc_frac × max_moc_participation × ADV_i / book).
    """
    dates = alpha.index
    tickers = list(alpha.columns)
    n_names = len(tickers)
    vol = ret.rolling(vol_window, min_periods=20).std().shift(1).bfill().fillna(0.02)
    ret_mat = ret.values

    use_advcap = (adv is not None) and (book is not None) and (book > 0)
    if use_advcap:
        adv_mat = adv.reindex(index=dates, columns=tickers).values
    else:
        adv_mat = None

    w_qp = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_prev = np.zeros(n_names)
    n_solves = 0; n_fails = 0
    t0 = time.time()

    for i, _dt in enumerate(dates):
        a = alpha.iloc[i].values
        sig = vol.iloc[i].values
        active = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig > 0)
        if active.sum() < 10:
            w_qp.iloc[i] = w_prev
            continue
        idx = np.where(active)[0]
        a_s = a[idx]
        sig_s = sig[idx]
        wp_s = w_prev[idx]
        price_s = close.iloc[i].values[idx]

        if use_advcap:
            adv_s = adv_mat[i, idx]
            valid = (adv_s > 0) & np.isfinite(adv_s)
            adv_cap = np.where(valid,
                               moc_frac * max_moc_participation * adv_s / book,
                               0.0)
            cap_s = np.minimum(max_w, adv_cap)
        else:
            cap_s = max_w

        try:
            L_list, s2 = risk_model_fn(i, idx, sig_s, ret_mat, factor_window)
            sol = solve_qp(a_s, wp_s, price_s, L_list, s2,
                           lambda_risk=lambda_risk, kappa_tc=kappa_tc, max_w=cap_s,
                           commission_per_share=commission_per_share,
                           impact_bps=impact_bps,
                           dollar_neutral=dollar_neutral,
                           max_gross_leverage=max_gross_leverage)
            if sol is None:
                w_qp.iloc[i] = w_prev
                n_fails += 1
                continue
            n_solves += 1
        except Exception:
            n_fails += 1
            w_qp.iloc[i] = w_prev
            continue
        new_w = np.zeros(n_names)
        new_w[idx] = sol
        w_qp.iloc[i] = new_w
        w_prev = new_w

    elapsed = time.time() - t0
    if verbose:
        print(f"  [{label}] done in {elapsed:.0f}s  "
              f"solves={n_solves}  fails={n_fails}", flush=True)
    return w_qp
