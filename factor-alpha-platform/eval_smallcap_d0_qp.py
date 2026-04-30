"""
QP-with-t-cost optimizer for the SMALLCAP_D0 alpha set.

For each day t we solve:

    maximize_w   α'·w  -  λ_risk · w'Σw  -  κ · ||w − w_prev||_1

  s.t.   ||w||_1 ≤ 1
         |w_i|  ≤ max_w        (per-name cap)
         (optional) sum(w) = 0  (dollar-neutral)

Where:
  α        = combined alpha signal at t (the equal-weight composite, normalized)
  Σ        = diagonal covariance proxy (per-name realized vol²)
  κ        = expected per-dollar one-way trading cost (here computed from the
             realistic per-share fee model on trade-weighted-average share price)
  w_prev   = optimal w from previous bar

This naturally produces a no-trade region around w_prev whose width is set by
κ / λ_risk · σ_i  — closed-form for diagonal Σ.

We use cvxpy for the L1-cost objective.

The whole window is walked sequentially, then evaluated with the same realistic
per-share fee model used in eval_smallcap_d0_sweep.
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

# Same per-share fee model as eval_smallcap_d0_sweep
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"


def main():
    # config
    UNIV_NAME   = "MCAP_100M_500M"
    MAX_W       = 0.02
    BOOK        = 500_000
    LAMBDA_RISK = float(__import__("os").environ.get("LAMBDA_RISK", 5.0))
    KAPPA_TC    = 30.0     # t-cost penalty per dollar of |Δw| (linear cost coefficient)
    TRAIN_END   = "2024-01-01"
    VAL_END     = "2025-04-01"
    REBAL_EVERY = int(__import__("os").environ.get("REBAL_EVERY", 1))

    print(f"=== QP-with-tcost  universe={UNIV_NAME}  book=${BOOK:,.0f} ===")
    print(f"  max_w={MAX_W}  λ_risk={LAMBDA_RISK}  κ_tc={KAPPA_TC}  rebal_every={REBAL_EVERY}")

    # Load universe + matrices
    uni = pd.read_parquet(UNIV_DIR / f"{UNIV_NAME}.parquet").astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0)/len(uni)
    valid = sorted(cov[cov>0.5].index.tolist())
    uni = uni[valid]
    dates = uni.index; tickers = uni.columns.tolist()

    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce"); df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc:
            mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)

    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)

    # Build signal — equal-weight composite of v2/v3 alphas
    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND (a.notes LIKE '%SMALLCAP_D0_v2%' OR a.notes LIKE '%SMALLCAP_D0_v3%')
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"  {len(rows)} alphas combined")

    def proc(s):
        s = s.astype(float).where(uni, np.nan)
        for g in cls.dropna().unique():
            m = (cls == g).values
            if m.any():
                sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
        ab = s.abs().sum(axis=1).replace(0, np.nan)
        return s.div(ab, axis=0).clip(-MAX_W, MAX_W).fillna(0)

    normed = [proc(engine.evaluate(e)) for _, e in rows]
    alpha = sum(normed) / len(normed)
    alpha = alpha.div(alpha.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    # Rolling per-name vol estimate (60d, lagged)
    vol = ret.rolling(60, min_periods=20).std().shift(1).fillna(method="bfill").fillna(0.02)

    # Sequential QP solve
    print(f"  Walking {len(dates)} bars ...")
    n_names = len(tickers)
    w_opt = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_prev = np.zeros(n_names)
    t_start = time.time()

    for i, dt in enumerate(dates):
        if i % REBAL_EVERY != 0:
            w_opt.iloc[i] = w_prev
            continue

        a = alpha.iloc[i].values
        sig = vol.iloc[i].values
        active_mask = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig > 0)
        if active_mask.sum() < 10:
            w_opt.iloc[i] = w_prev
            continue

        idx = np.where(active_mask)[0]
        a_sub = a[idx]
        sig_sub = sig[idx]
        wp_sub = w_prev[idx]

        # Build per-name t-cost coefficient: cost ≈ commission_per_share / price_i  + IMPACT
        price_sub = close.iloc[i].values[idx]
        per_dollar_cost = (COMMISSION_PER_SHARE / np.maximum(price_sub, 0.01)) + IMPACT_BPS / 1e4
        kappa_i = KAPPA_TC * per_dollar_cost   # vector

        # CVXPY problem (rebuild for new size — could cache but fine for MVP)
        w = cp.Variable(len(idx))
        # diag covariance
        risk_quad = cp.sum(cp.multiply(sig_sub**2, cp.square(w)))
        tc = cp.sum(cp.multiply(kappa_i, cp.abs(w - wp_sub)))
        obj = cp.Maximize(a_sub @ w - 0.5 * LAMBDA_RISK * risk_quad - tc)
        constraints = [
            cp.norm(w, 1) <= 1.0,
            cp.abs(w) <= MAX_W,
            cp.sum(w) == 0,
        ]
        try:
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                w_opt.iloc[i] = w_prev
                continue
            sol = w.value
        except Exception:
            w_opt.iloc[i] = w_prev
            continue

        new_w = np.zeros(n_names)
        new_w[idx] = sol
        w_opt.iloc[i] = new_w
        w_prev = new_w

        if (i+1) % 200 == 0:
            print(f"    bar {i+1}/{len(dates)}  active={active_mask.sum()}  elapsed={time.time()-t_start:.0f}s")

    print(f"  QP walk done in {time.time()-t_start:.0f}s")

    # Realistic-fee evaluation
    nx = ret.shift(-1)
    g = (w_opt * nx).sum(axis=1).fillna(0)
    pos = w_opt * BOOK
    trd = pos.diff().abs()
    safe_close = close.where(close > 0)
    shares = trd / safe_close
    per_name_comm = (shares * COMMISSION_PER_SHARE).clip(lower=0)
    has_trade = trd > 1.0
    per_name_comm = per_name_comm.where(~has_trade, np.maximum(per_name_comm, PER_ORDER_MIN))
    per_name_comm = per_name_comm.where(has_trade, 0)
    cost_d = (per_name_comm.sum(axis=1)
              + (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
              + (trd * IMPACT_BPS / 1e4).sum(axis=1)
              + (-pos.clip(upper=0)).sum(axis=1) * (BORROW_BPS_ANNUAL / 1e4) / 252.0
             ) / BOOK
    n = g - cost_d.reindex(g.index).fillna(0)
    to = trd.sum(axis=1) / BOOK / 2.0
    n_active = (w_opt.abs() > 0).sum(axis=1)
    ann = np.sqrt(252)

    def stats(g_, n_, lab):
        srg = g_.mean()/g_.std()*ann if g_.std()>0 else float("nan")
        srn = n_.mean()/n_.std()*ann if n_.std()>0 else float("nan")
        return f"  {lab:6s}  SR_g={srg:+5.2f}  SR_n={srn:+5.2f}  ret_g={g_.mean()*252*100:+5.1f}%  ret_n={n_.mean()*252*100:+5.1f}%"

    print(f"\n  cost={cost_d.mean()*1e4:.2f}bps/d ({cost_d.mean()*252*100:.2f}%/yr drag)")
    print(f"  TO={to.mean()*100:.1f}%/d  n_active={n_active.mean():.0f}")
    print(stats(g.loc[:TRAIN_END], n.loc[:TRAIN_END], "TRAIN"))
    print(stats(g.loc[TRAIN_END:VAL_END], n.loc[TRAIN_END:VAL_END], "VAL"))
    print(stats(g.loc[VAL_END:], n.loc[VAL_END:], "TEST"))
    print(stats(g, n, "FULL"))


if __name__ == "__main__":
    main()
