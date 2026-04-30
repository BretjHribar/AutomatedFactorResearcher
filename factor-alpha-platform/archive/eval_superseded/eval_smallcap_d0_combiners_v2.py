"""
Multi-combiner evaluator for the SMALLCAP_D0 alpha set.

Runs four signal combiners (equal_weight / adaptive / risk_parity / billions)
PLUS the QP-with-t-cost optimizer on whatever alphas are tagged
[SMALLCAP_D0_*] in the DB. Same MCAP_100M_500M / max_w 0.02 / IB MOC fee model
as eval_smallcap_d0_final.py.

VAL/TEST splits are reported here for ASSESSMENT ONLY — never used for selection.

Output:
  - prints TRAIN/VAL/TEST/FULL gross + net SR for each combiner
  - saves data/smallcap_d0_combiners_v2_equity.png
  - saves data/smallcap_d0_combiners_v2_summary.json
"""
from __future__ import annotations
import sys, sqlite3, json, time
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
from src.portfolio.combiners import (
    combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions,
    combiner_ic_weighted, combiner_sharpe_weighted, combiner_topn_sharpe,
)

UNIV_NAME   = "MCAP_100M_500M"
MAX_W       = 0.02
BOOK        = 500_000
LAMBDA_RISK = 5.0
KAPPA_TC    = 30.0
TRAIN_END   = "2024-01-01"
VAL_END     = "2025-04-01"

# Realistic per-share IB MOC fees
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_combiners_v2_equity.png"
OUT_JSON = ROOT / "data/smallcap_d0_combiners_v2_summary.json"


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


def normalize_clip(sig, mw=MAX_W):
    abs_sum = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(abs_sum, axis=0).clip(-mw, mw).fillna(0)


def split_stats(g, n, label):
    ann = np.sqrt(252)
    sg = g.mean()/g.std()*ann if g.std()>0 else float('nan')
    sn = n.mean()/n.std()*ann if n.std()>0 else float('nan')
    return dict(name=label,
                sharpe_gross=float(sg),
                sharpe_net=float(sn),
                ret_gross=float(g.mean()*252),
                ret_net=float(n.mean()*252))


def report(name, w, close, ret, train_end, val_end, book):
    nx = ret.shift(-1)
    g = (w * nx).sum(axis=1).fillna(0)
    cost = realistic_cost(w, close, book)
    n = g - cost
    ann = np.sqrt(252)
    to = w.diff().abs().sum(axis=1).mean() / 2

    s_train = split_stats(g.loc[:train_end], n.loc[:train_end], "TRAIN")
    s_val   = split_stats(g.loc[train_end:val_end], n.loc[train_end:val_end], "VAL")
    s_test  = split_stats(g.loc[val_end:], n.loc[val_end:], "TEST")
    s_full  = split_stats(g, n, "FULL")
    print(f"\n=== {name} ===  TO={to*100:.1f}%/d  cost={cost.mean()*1e4:.2f}bps/d ({cost.mean()*252*100:.2f}%/yr)")
    for s in (s_train, s_val, s_test, s_full):
        print(f"  {s['name']:6s}  SR_g={s['sharpe_gross']:+5.2f}  SR_n={s['sharpe_net']:+5.2f}  "
              f"ret_g={s['ret_gross']*100:+5.1f}%  ret_n={s['ret_net']*100:+5.1f}%")
    return n, dict(train=s_train, val=s_val, test=s_test, full=s_full,
                   turnover=float(to), cost_bps_d=float(cost.mean()*1e4))


def qp_combiner(alpha, close, ret, uni, n_names, dates):
    """QP with linear t-cost penalty — same as eval_smallcap_d0_final.py."""
    print("  Running QP walk-forward...")
    vol = ret.rolling(60, min_periods=20).std().shift(1).bfill().fillna(0.02)
    w_qp = pd.DataFrame(0.0, index=dates, columns=alpha.columns)
    w_prev = np.zeros(n_names)
    t0 = time.time()
    for i, dt in enumerate(dates):
        a = alpha.iloc[i].values; sig_v = vol.iloc[i].values
        active = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig_v > 0)
        if active.sum() < 10:
            w_qp.iloc[i] = w_prev; continue
        idx = np.where(active)[0]
        a_s = a[idx]; sig_s = sig_v[idx]; wp_s = w_prev[idx]
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
    print(f"  QP done in {time.time()-t0:.0f}s")
    return w_qp


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
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"Combining {len(rows)} alphas: {[r[0] for r in rows]}")

    # Pre-process raw signals once (raw -> universe-masked + subindustry-demeaned + L1-clipped)
    raw_signals = {}
    normed = {}
    for aid, expr in rows:
        try:
            r = engine.evaluate(expr)
            raw_signals[aid] = r
            normed[aid] = proc_signal(r, uni, cls)
        except Exception as e:
            print(f"  skip #{aid}: {e}")

    # ---- 1) equal-weight combiner (simple mean) ----
    eq_alpha = sum(normed.values()) / len(normed)
    eq_alpha = normalize_clip(eq_alpha)

    # ---- 2) adaptive (rolling expected-return weighted) ----
    print("\n  Running combiner_adaptive...")
    adapt_alpha = combiner_adaptive(raw_signals, mats, uni, ret, lookback=504, max_wt=MAX_W)
    adapt_alpha = normalize_clip(adapt_alpha)

    # ---- 3) risk parity ----
    print("  Running combiner_risk_parity...")
    rp_alpha = combiner_risk_parity(raw_signals, mats, uni, ret, lookback=504, max_wt=MAX_W)
    rp_alpha = normalize_clip(rp_alpha)

    # ---- 4) billions (Kakushadze) ----
    print("  Running combiner_billions...")
    bil_alpha = combiner_billions(raw_signals, mats, uni, ret, optim_lookback=60, max_wt=MAX_W)
    bil_alpha = normalize_clip(bil_alpha)

    # ---- 4b) IC-weighted ----
    print("  Running combiner_ic_weighted...")
    ic_alpha = combiner_ic_weighted(raw_signals, mats, uni, ret, lookback=126, max_wt=MAX_W)
    ic_alpha = normalize_clip(ic_alpha)

    # ---- 4c) Sharpe-weighted ----
    print("  Running combiner_sharpe_weighted...")
    shr_alpha = combiner_sharpe_weighted(raw_signals, mats, uni, ret, lookback=252, max_wt=MAX_W)
    shr_alpha = normalize_clip(shr_alpha)

    # ---- 4d) Top-N Sharpe (top 10 alphas each bar) ----
    print("  Running combiner_topn_sharpe (top 10)...")
    top10_alpha = combiner_topn_sharpe(raw_signals, mats, uni, ret, lookback=252, top_n=10, max_wt=MAX_W)
    top10_alpha = normalize_clip(top10_alpha)

    # ---- 5) QP on the equal-weight composite ----
    print("\n  Running QP optimizer on equal-weight composite...")
    qp_w = qp_combiner(eq_alpha, close, ret, uni, len(tickers), dates)

    # Reports
    summary = {}
    eq_curves = {}

    n_eq, summary['equal_weight'] = report("equal_weight", eq_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['equal_weight'] = (1 + n_eq).cumprod()

    n_ad, summary['adaptive']     = report("adaptive",     adapt_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['adaptive'] = (1 + n_ad).cumprod()

    n_rp, summary['risk_parity']  = report("risk_parity",  rp_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['risk_parity'] = (1 + n_rp).cumprod()

    n_bi, summary['billions']     = report("billions",     bil_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['billions'] = (1 + n_bi).cumprod()

    n_ic, summary['ic_weighted']  = report("ic_weighted",  ic_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['ic_weighted'] = (1 + n_ic).cumprod()

    n_sh, summary['sharpe_weighted'] = report("sharpe_weighted", shr_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['sharpe_weighted'] = (1 + n_sh).cumprod()

    n_t10, summary['topn_sharpe_10'] = report("top10_sharpe", top10_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['topn_sharpe_10'] = (1 + n_t10).cumprod()

    n_qp, summary['QP_tcost']     = report("QP + t-cost",  qp_w, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['QP_tcost'] = (1 + n_qp).cumprod()

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"equal_weight":"C0", "adaptive":"C1", "risk_parity":"C2", "billions":"C4",
              "ic_weighted":"C5", "sharpe_weighted":"C6", "topn_sharpe_10":"C7",
              "QP_tcost":"C3"}
    for name, eq in eq_curves.items():
        sr = summary[name]['full']['sharpe_net']
        lw = 1.6 if name == 'QP_tcost' else 1.0
        ax.plot(eq.index, eq.values, lw=lw, color=colors.get(name,"k"),
                label=f"{name:13s} net SR(full)={sr:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END), color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 — combiner showdown @ ${BOOK:,.0f} book\n"
                 f"({len(rows)} alphas, {UNIV_NAME}, max_w={MAX_W}, realistic per-share IB MOC fees)")
    ax.set_ylabel("Equity (log, start=1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")

    # JSON summary
    with open(OUT_JSON, "w") as f:
        json.dump({"alphas": [r[0] for r in rows], "summary": summary,
                   "config": dict(universe=UNIV_NAME, max_w=MAX_W, book=BOOK,
                                  train_end=TRAIN_END, val_end=VAL_END)}, f, indent=2)

    # Best
    print("\n=== BEST NET SR (FULL) ===")
    best = sorted(summary.items(), key=lambda x: -x[1]['full']['sharpe_net'])
    for name, s in best:
        print(f"  {name:13s}  TRAIN={s['train']['sharpe_net']:+5.2f}  "
              f"VAL={s['val']['sharpe_net']:+5.2f}  "
              f"TEST={s['test']['sharpe_net']:+5.2f}  "
              f"FULL={s['full']['sharpe_net']:+5.2f}")


if __name__ == "__main__":
    main()
