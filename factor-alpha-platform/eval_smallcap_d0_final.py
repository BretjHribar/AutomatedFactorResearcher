"""
Final equity curve + summary for the optimal config:

  Universe:   MCAP_100M_500M  (micro-cap by market cap, 215 tickers, ~159 active)
  Alphas:     all SMALLCAP_D0_v2 + v3 (7 alphas)
  Combiner:   equal_weight
  Optimizer:  QP with linear t-cost penalty (κ=30, λ_risk=5)
  Book:       $500K
  Fees:       realistic per-share IB MOC ($0.0045 + $0.35 min, sec/finra,
              0.5bp impact, 50bp/yr borrow on shorts)
"""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_FINAL_equity.png"


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


def main():
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
    print(f"Combining {len(rows)} alphas: {[r[0] for r in rows]}")

    normed = [proc_signal(engine.evaluate(e), uni, cls) for _, e in rows]
    alpha = sum(normed) / len(normed)
    alpha = alpha.div(alpha.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    # ---- 1) Equal-weight (no QP) baseline ----
    eq_simple = alpha.copy()
    g_simple = (eq_simple * ret.shift(-1)).sum(axis=1).fillna(0)
    cost_simple = realistic_cost(eq_simple, close, BOOK)
    n_simple = g_simple - cost_simple

    # ---- 2) Equal-weight + no-trade-band 0.3 ----
    band = 0.3 * MAX_W
    w_band = pd.DataFrame(0.0, index=dates, columns=tickers)
    prev = pd.Series(0.0, index=tickers)
    for i, dt in enumerate(dates):
        target = alpha.iloc[i]
        diff = target - prev
        new = target.where(diff.abs() >= band, prev)
        w_band.iloc[i] = new
        prev = new
    g_band = (w_band * ret.shift(-1)).sum(axis=1).fillna(0)
    cost_band = realistic_cost(w_band, close, BOOK)
    n_band = g_band - cost_band

    # ---- 3) QP with t-cost (the optimal) ----
    print("Running QP walk-forward ...")
    vol = ret.rolling(60, min_periods=20).std().shift(1).bfill().fillna(0.02)
    n_names = len(tickers)
    w_qp = pd.DataFrame(0.0, index=dates, columns=tickers)
    w_prev = np.zeros(n_names)
    t0 = time.time()
    for i, dt in enumerate(dates):
        a = alpha.iloc[i].values; sig = vol.iloc[i].values
        active = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig > 0)
        if active.sum() < 10:
            w_qp.iloc[i] = w_prev; continue
        idx = np.where(active)[0]
        a_s = a[idx]; sig_s = sig[idx]; wp_s = w_prev[idx]
        price_s = close.iloc[i].values[idx]
        kappa = KAPPA_TC * (COMMISSION_PER_SHARE / np.maximum(price_s, 0.01) + IMPACT_BPS/1e4)
        w = cp.Variable(len(idx))
        risk = cp.sum(cp.multiply(sig_s**2, cp.square(w)))
        tc = cp.sum(cp.multiply(kappa, cp.abs(w - wp_s)))
        obj = cp.Maximize(a_s @ w - 0.5 * LAMBDA_RISK * risk - tc)
        cons = [cp.norm(w, 1) <= 1.0, cp.abs(w) <= MAX_W, cp.sum(w) == 0]
        try:
            cp.Problem(obj, cons).solve(solver=cp.OSQP, warm_start=True, verbose=False)
            sol = w.value
        except Exception:
            w_qp.iloc[i] = w_prev; continue
        new_w = np.zeros(n_names); new_w[idx] = sol
        w_qp.iloc[i] = new_w; w_prev = new_w
    print(f"QP done in {time.time()-t0:.0f}s")
    g_qp = (w_qp * ret.shift(-1)).sum(axis=1).fillna(0)
    cost_qp = realistic_cost(w_qp, close, BOOK)
    n_qp = g_qp - cost_qp

    # ---- Print summary ----
    ann = np.sqrt(252)
    def show(g, n, cost, w, name):
        to = w.diff().abs().sum(axis=1).mean() / 2
        print(f"\n=== {name} ===  TO={to*100:.1f}%/d  cost={cost.mean()*1e4:.2f}bps/d ({cost.mean()*252*100:.2f}%/yr)")
        for lab, sl in [("TRAIN", slice(None, TRAIN_END)),
                        ("VAL",   slice(TRAIN_END, VAL_END)),
                        ("TEST",  slice(VAL_END, None)),
                        ("FULL",  slice(None, None))]:
            gg = g.loc[sl]; nn = n.loc[sl]
            srg = gg.mean()/gg.std()*ann if gg.std()>0 else float('nan')
            srn = nn.mean()/nn.std()*ann if nn.std()>0 else float('nan')
            print(f"  {lab:6s}  SR_g={srg:+5.2f}  SR_n={srn:+5.2f}  ret_g={gg.mean()*252*100:+5.1f}%  ret_n={nn.mean()*252*100:+5.1f}%")

    show(g_simple, n_simple, cost_simple, eq_simple, "EW (no NTB)")
    show(g_band,   n_band,   cost_band,   w_band, "EW + NTB=0.3")
    show(g_qp,     n_qp,     cost_qp,     w_qp,   "QP + t-cost")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 7))
    eq1 = (1 + n_simple).cumprod()
    eq2 = (1 + n_band).cumprod()
    eq3 = (1 + n_qp).cumprod()
    sr1 = n_simple.mean()/n_simple.std()*ann
    sr2 = n_band.mean()/n_band.std()*ann
    sr3 = n_qp.mean()/n_qp.std()*ann
    ax.plot(eq1.index, eq1.values, color="C0", lw=1.0, label=f"equal-weight  net SR={sr1:+.2f}", alpha=0.6)
    ax.plot(eq2.index, eq2.values, color="C1", lw=1.0, label=f"EW + NTB(0.3)  net SR={sr2:+.2f}", alpha=0.7)
    ax.plot(eq3.index, eq3.values, color="C3", lw=1.5, label=f"QP + t-cost   net SR={sr3:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END),   color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 — combiner showdown @ ${BOOK:,.0f} book\n"
                 f"({len(rows)} alphas, {UNIV_NAME}, max_w={MAX_W}, realistic per-share IB MOC fees)")
    ax.set_ylabel("Equity (log, start=1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
