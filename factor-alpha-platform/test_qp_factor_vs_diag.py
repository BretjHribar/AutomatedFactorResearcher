"""
Head-to-head QP risk-model test on SMALLCAP_D0 composite.

Three variants — IDENTICAL alpha composite, fees, constraints, t-cost. The only
thing that changes is the risk term in the objective.

  DIAG_60   : risk = sum_i sig60_i^2 * w_i^2          (current production)
  DIAG_126  : risk = sum_i sig126_i^2 * w_i^2         (window-control for FACTOR)
  FACTOR_K5 : risk = ||B' w||^2 + sum_i s_i^2 w_i^2,  B,s from K=5 PCA on 126d cov

For each variant: TRAIN/VAL/TEST/FULL gross + net SR, turnover, mean factor
exposure (only meaningful for FACTOR).
"""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
import cvxpy as cp

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

UNIV_NAME   = "MCAP_100M_500M"
MAX_W       = 0.02
BOOK        = 500_000
LAMBDA_RISK = 5.0
KAPPA_TC    = 30.0
TRAIN_END   = "2024-01-01"
VAL_END     = "2025-04-01"

COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

# Factor model params
FACTOR_WINDOW = 126
N_FACTORS     = 5

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"


def proc_signal(s, uni, cls):
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    ab = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(ab, axis=0).clip(-MAX_W, MAX_W).fillna(0)


def realistic_cost(w, close, book):
    pos = w * book
    trd = pos.diff().abs()
    safe = close.where(close > 0)
    shares = trd / safe
    pn_comm = (shares * COMMISSION_PER_SHARE).clip(lower=0)
    has = trd > 1.0
    pn_comm = pn_comm.where(~has, np.maximum(pn_comm, PER_ORDER_MIN)).where(has, 0)
    cost = (pn_comm.sum(axis=1)
            + (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
            + (trd * IMPACT_BPS / 1e4).sum(axis=1)
            + (-pos.clip(upper=0)).sum(axis=1) * (BORROW_BPS_ANNUAL / 1e4) / 252.0
           ) / book
    return cost


def qp_solve_diag(a_s, sig2_s, wp_s, kappa, max_w):
    n = len(a_s)
    w = cp.Variable(n)
    risk = cp.sum(cp.multiply(sig2_s, cp.square(w)))
    tc = cp.sum(cp.multiply(kappa, cp.abs(w - wp_s)))
    obj = cp.Maximize(a_s @ w - 0.5 * LAMBDA_RISK * risk - tc)
    cons = [cp.norm(w, 1) <= 1.0, cp.abs(w) <= max_w, cp.sum(w) == 0]
    cp.Problem(obj, cons).solve(solver=cp.OSQP, warm_start=True, verbose=False)
    return w.value


def qp_solve_factor(a_s, B_s, s2_s, wp_s, kappa, max_w):
    n = len(a_s)
    w = cp.Variable(n)
    factor_risk = cp.sum_squares(B_s.T @ w)            # ||B' w||^2
    spec_risk = cp.sum(cp.multiply(s2_s, cp.square(w)))
    tc = cp.sum(cp.multiply(kappa, cp.abs(w - wp_s)))
    obj = cp.Maximize(a_s @ w - 0.5 * LAMBDA_RISK * (factor_risk + spec_risk) - tc)
    cons = [cp.norm(w, 1) <= 1.0, cp.abs(w) <= max_w, cp.sum(w) == 0]
    cp.Problem(obj, cons).solve(solver=cp.OSQP, warm_start=True, verbose=False)
    return w.value


def estimate_factor_model(R_window, k):
    """K-factor PCA decomposition of sample cov.

    R_window: (T, N) demeaned returns of active names.
    Returns (B, s2) where B is (N, K) factor loadings, s2 is (N,) specific var.

    Σ ≈ B B' + diag(s²) such that diag(Σ) ≈ diag(sample_cov).
    """
    T, N = R_window.shape
    R = R_window - R_window.mean(axis=0, keepdims=True)
    cov = (R.T @ R) / max(T - 1, 1)
    cov = (cov + cov.T) * 0.5
    cov += 1e-10 * np.eye(N)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return np.zeros((N, k)), np.maximum(np.diag(cov), 1e-8)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    k_use = min(k, np.sum(eigvals > 1e-12))
    B = eigvecs[:, :k_use] * np.sqrt(np.maximum(eigvals[:k_use], 0.0))[None, :]
    if k_use < k:
        B = np.hstack([B, np.zeros((N, k - k_use))])
    common_var = np.sum(B**2, axis=1)
    total_var = np.diag(cov)
    s2 = np.maximum(total_var - common_var, 1e-8)
    return B, s2


def run_walkforward(name, alpha, close, ret, uni, dates, tickers, mode,
                     vol60, vol126, ret_mat):
    """mode in {'diag60','diag126','factor'}"""
    print(f"\n[{name}] starting walk-forward ({mode})...", flush=True)
    n_names = len(tickers)
    w_qp = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_prev = np.zeros(n_names)
    factor_var_pct_log = []
    n_solves = 0; n_fails = 0
    t0 = time.time()
    for i, dt in enumerate(dates):
        a = alpha.iloc[i].values
        sig60 = vol60.iloc[i].values
        active = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig60 > 0)
        if active.sum() < 10:
            w_qp.iloc[i] = w_prev; continue
        idx = np.where(active)[0]
        a_s = a[idx]; wp_s = w_prev[idx]
        price_s = close.iloc[i].values[idx]
        kappa = KAPPA_TC * (COMMISSION_PER_SHARE / np.maximum(price_s, 0.01) + IMPACT_BPS/1e4)
        try:
            if mode == "diag60":
                sig2_s = sig60[idx] ** 2
                sol = qp_solve_diag(a_s, sig2_s, wp_s, kappa, MAX_W)
            elif mode == "diag126":
                sig2_s = vol126.iloc[i].values[idx] ** 2
                if not np.all(np.isfinite(sig2_s)) or np.any(sig2_s <= 0):
                    sig2_s = np.where(np.isfinite(sig2_s) & (sig2_s > 0), sig2_s, sig60[idx] ** 2)
                sol = qp_solve_diag(a_s, sig2_s, wp_s, kappa, MAX_W)
            elif mode == "factor":
                if i < FACTOR_WINDOW + 1:
                    sig2_s = sig60[idx] ** 2
                    sol = qp_solve_diag(a_s, sig2_s, wp_s, kappa, MAX_W)
                else:
                    R_win = ret_mat[i - FACTOR_WINDOW : i, idx]
                    R_win = np.where(np.isfinite(R_win), R_win, 0.0)
                    B_s, s2_s = estimate_factor_model(R_win, N_FACTORS)
                    fv = float(np.sum(B_s**2)) / max(float(np.sum(B_s**2) + np.sum(s2_s)), 1e-12)
                    factor_var_pct_log.append(fv)
                    sol = qp_solve_factor(a_s, B_s, s2_s, wp_s, kappa, MAX_W)
            else:
                raise ValueError(mode)
            if sol is None:
                w_qp.iloc[i] = w_prev; n_fails += 1; continue
            n_solves += 1
        except Exception as e:
            n_fails += 1
            w_qp.iloc[i] = w_prev; continue
        new_w = np.zeros(n_names); new_w[idx] = sol
        w_qp.iloc[i] = new_w; w_prev = new_w
    elapsed = time.time() - t0
    avg_fv = np.mean(factor_var_pct_log) if factor_var_pct_log else float('nan')
    print(f"[{name}] done in {elapsed:.0f}s  solves={n_solves}  fails={n_fails}"
          + (f"  avg_factor_var_share={avg_fv*100:.1f}%" if mode == "factor" else ""), flush=True)
    return w_qp


def show(name, g, n, cost, w, ann):
    to = w.diff().abs().sum(axis=1).mean() / 2
    max_pos = w.abs().max(axis=1).max()
    print(f"\n=== {name} ===  TO={to*100:.1f}%/d  max|w|={max_pos*100:.2f}%  cost={cost.mean()*1e4:.2f}bps/d ({cost.mean()*252*100:.2f}%/yr)")
    rows = []
    for lab, sl in [("TRAIN", slice(None, TRAIN_END)),
                    ("VAL",   slice(TRAIN_END, VAL_END)),
                    ("TEST",  slice(VAL_END, None)),
                    ("FULL",  slice(None, None))]:
        gg = g.loc[sl]; nn = n.loc[sl]
        srg = gg.mean()/gg.std()*ann if gg.std()>0 else float('nan')
        srn = nn.mean()/nn.std()*ann if nn.std()>0 else float('nan')
        print(f"  {lab:6s}  SR_g={srg:+5.2f}  SR_n={srn:+5.2f}  ret_g={gg.mean()*252*100:+5.1f}%  ret_n={nn.mean()*252*100:+5.1f}%")
        rows.append((lab, srg, srn))
    return rows


def main():
    print(f"=== Loading {UNIV_NAME} ===", flush=True)
    uni = pd.read_parquet(UNIV_DIR / f"{UNIV_NAME}.parquet").astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0)/len(uni); valid = sorted(cov[cov>0.5].index.tolist())
    uni = uni[valid]; dates = uni.index; tickers = uni.columns.tolist()

    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce"); df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc: mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)

    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)

    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND (a.notes LIKE '%SMALLCAP_D0_v2%' OR a.notes LIKE '%SMALLCAP_D0_v3%')
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"Combining {len(rows)} alphas: {[r[0] for r in rows]}", flush=True)

    normed = [proc_signal(engine.evaluate(e), uni, cls) for _, e in rows]
    alpha = sum(normed) / len(normed)
    alpha = alpha.div(alpha.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    vol60  = ret.rolling(60,  min_periods=20).std().shift(1).bfill().fillna(0.02)
    vol126 = ret.rolling(126, min_periods=40).std().shift(1).bfill().fillna(0.02)
    ret_mat = ret.values  # (T, N), shifted-window access uses [i-W:i] which excludes day i (no look-ahead)

    print(f"Universe: {len(tickers)} tickers, {len(dates)} dates "
          f"({dates[0].date()} → {dates[-1].date()})", flush=True)
    print(f"Avg active names/day: {((~np.isnan(close.values)) & uni.values).sum(axis=1).mean():.0f}", flush=True)

    ann = np.sqrt(252)
    summary = {}

    for name, mode in [("DIAG_60",   "diag60"),
                       ("DIAG_126",  "diag126"),
                       ("FACTOR_K5", "factor")]:
        w_qp = run_walkforward(name, alpha, close, ret, uni, dates, tickers, mode,
                                vol60, vol126, ret_mat)
        g = (w_qp * ret.shift(-1)).sum(axis=1).fillna(0)
        cost = realistic_cost(w_qp, close, BOOK)
        n = g - cost
        rows_out = show(name, g, n, cost, w_qp, ann)
        summary[name] = {lab: (srg, srn) for lab, srg, srn in rows_out}

    print("\n" + "="*72)
    print("SUMMARY (net SR)")
    print("="*72)
    print(f"{'split':6s} | {'DIAG_60':>10s} | {'DIAG_126':>10s} | {'FACTOR_K5':>10s}")
    print("-"*72)
    for lab in ["TRAIN", "VAL", "TEST", "FULL"]:
        d60 = summary["DIAG_60"][lab][1]
        d126 = summary["DIAG_126"][lab][1]
        fk5 = summary["FACTOR_K5"][lab][1]
        print(f"{lab:6s} | {d60:+10.3f} | {d126:+10.3f} | {fk5:+10.3f}")


if __name__ == "__main__":
    main()
