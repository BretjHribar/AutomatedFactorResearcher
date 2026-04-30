"""
Greedy-prune the 39 SMALLCAP_D0 alphas to a subset where all pairwise
correlations on TRAIN are < CORR_CAP. Then run all 8 combiners on that subset.

Pruning method:
  1. Compute pairwise correlation matrix on TRAIN-window normalized signals.
  2. Greedy by TRAIN-SR (DB sharpe_is): start with highest-SR alpha, then add
     each subsequent alpha if its max corr to the selected set is < CORR_CAP.
  3. Also report what max-independent-set looks like via networkx if available.

Then run all combiners on the pruned subset and print the FULL summary.
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

CORR_CAP    = 0.7
UNIV_NAME   = "MCAP_100M_500M"
MAX_W       = 0.02
BOOK        = 500_000
LAMBDA_RISK = 5.0
KAPPA_TC    = 30.0
TRAIN_START = "2020-01-01"
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
OUT_PNG  = ROOT / "data/smallcap_d0_pruned_combiners_equity.png"


def proc_signal(s, uni, cls):
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    ab = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(ab, axis=0).clip(-MAX_W, MAX_W).fillna(0)


def realistic_cost(w, close, book):
    pos = w * book; trd = pos.diff().abs()
    safe = close.where(close > 0); shares = trd / safe
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
    return dict(name=label, sharpe_gross=float(sg), sharpe_net=float(sn),
                ret_gross=float(g.mean()*252), ret_net=float(n.mean()*252))


def report(name, w, close, ret, train_end, val_end, book):
    nx = ret.shift(-1)
    g = (w * nx).sum(axis=1).fillna(0)
    cost = realistic_cost(w, close, book); n = g - cost
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
    print("  Running QP walk-forward...")
    vol = ret.rolling(60, min_periods=20).std().shift(1).bfill().fillna(0.02)
    w_qp = pd.DataFrame(0.0, index=dates, columns=alpha.columns)
    w_prev = np.zeros(n_names); t0 = time.time()
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
    print(f"=== Loading {UNIV_NAME} ===")
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
        SELECT a.id, a.expression, MAX(e.sharpe_is) as sr
          FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         GROUP BY a.id ORDER BY MAX(e.sharpe_is) DESC""").fetchall()
    print(f"\n=== {len(rows)} SMALLCAP_D0 alphas ranked by TRAIN SR ===")

    raw_signals = {}
    normed_signals = {}
    for aid, expr, sr in rows:
        try:
            r = engine.evaluate(expr)
            raw_signals[aid] = r
            normed_signals[aid] = proc_signal(r, uni, cls)
        except Exception as e:
            print(f"  skip #{aid}: {e}")

    # Compute pairwise correlation matrix on TRAIN (signal-flat-non-zero pearson)
    print("\n=== Computing pairwise correlations on TRAIN ===")
    train_mask = (dates >= TRAIN_START) & (dates < TRAIN_END)
    aid_list = [aid for aid, _, _ in rows if aid in normed_signals]
    n = len(aid_list)
    flat = {}
    for aid in aid_list:
        v = normed_signals[aid].loc[train_mask].values.flatten()
        m = ~np.isnan(v) & (v != 0)
        flat[aid] = (v, m)
    corr_mat = np.zeros((n, n))
    for i, ai in enumerate(aid_list):
        vi, mi = flat[ai]
        for j, aj in enumerate(aid_list):
            if i == j: corr_mat[i,j] = 1.0; continue
            if i > j:  corr_mat[i,j] = corr_mat[j,i]; continue
            vj, mj = flat[aj]
            mm = mi & mj
            if mm.sum() < 1000:
                corr_mat[i,j] = float('nan'); continue
            corr_mat[i,j] = float(np.corrcoef(vi[mm], vj[mm])[0,1])
    corr_df = pd.DataFrame(corr_mat, index=aid_list, columns=aid_list)

    # Greedy by TRAIN SR (rows already sorted desc)
    print(f"\n=== Greedy pruning to corr < {CORR_CAP} (rank by TRAIN SR) ===")
    sr_by_aid = {aid: sr for aid, _, sr in rows if aid in normed_signals}
    selected = []
    for aid, _, sr in rows:
        if aid not in normed_signals: continue
        if not selected:
            selected.append(aid); continue
        max_corr = max(abs(corr_df.loc[aid, s]) for s in selected if not np.isnan(corr_df.loc[aid, s]))
        if max_corr < CORR_CAP:
            selected.append(aid)
            print(f"  + #{aid:3d}  SR={sr:.2f}  max|corr to selected|={max_corr:.3f}")
        else:
            pass  # silently drop
    print(f"\n  Selected {len(selected)} of {n} alphas.")
    print(f"  IDs: {selected}")

    # Show pairwise corr of selected (sanity)
    sel_corr = corr_df.loc[selected, selected].abs()
    triu = sel_corr.where(np.triu(np.ones_like(sel_corr, dtype=bool), k=1))
    print(f"\n  Selected subset pairwise |corr|: min={triu.stack().min():.3f}, "
          f"median={triu.stack().median():.3f}, max={triu.stack().max():.3f}")

    # Build combiner inputs from SELECTED only
    sel_raw = {aid: raw_signals[aid] for aid in selected}
    sel_normed = {aid: normed_signals[aid] for aid in selected}

    eq_alpha = sum(sel_normed.values()) / len(sel_normed)
    eq_alpha = normalize_clip(eq_alpha)

    print("\n  Running combiner_adaptive...")
    adapt = normalize_clip(combiner_adaptive(sel_raw, mats, uni, ret, lookback=504, max_wt=MAX_W))
    print("  Running combiner_risk_parity...")
    rp = normalize_clip(combiner_risk_parity(sel_raw, mats, uni, ret, lookback=504, max_wt=MAX_W))
    print("  Running combiner_billions...")
    bil = normalize_clip(combiner_billions(sel_raw, mats, uni, ret, optim_lookback=60, max_wt=MAX_W))
    print("  Running combiner_ic_weighted...")
    ic = normalize_clip(combiner_ic_weighted(sel_raw, mats, uni, ret, lookback=126, max_wt=MAX_W))
    print("  Running combiner_sharpe_weighted...")
    shr = normalize_clip(combiner_sharpe_weighted(sel_raw, mats, uni, ret, lookback=252, max_wt=MAX_W))
    print("  Running combiner_topn_sharpe (top 10)...")
    top10 = normalize_clip(combiner_topn_sharpe(sel_raw, mats, uni, ret, lookback=252, top_n=10, max_wt=MAX_W))
    print("\n  Running QP optimizer on equal-weight composite...")
    qp_w = qp_combiner(eq_alpha, close, ret, uni, len(tickers), dates)

    summary = {}; eq_curves = {}
    n_eq, summary['equal_weight']    = report("equal_weight", eq_alpha, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['equal_weight'] = (1 + n_eq).cumprod()
    n_ad, summary['adaptive']        = report("adaptive", adapt, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['adaptive'] = (1 + n_ad).cumprod()
    n_rp, summary['risk_parity']     = report("risk_parity", rp, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['risk_parity'] = (1 + n_rp).cumprod()
    n_bi, summary['billions']        = report("billions", bil, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['billions'] = (1 + n_bi).cumprod()
    n_ic, summary['ic_weighted']     = report("ic_weighted", ic, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['ic_weighted'] = (1 + n_ic).cumprod()
    n_sh, summary['sharpe_weighted'] = report("sharpe_weighted", shr, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['sharpe_weighted'] = (1 + n_sh).cumprod()
    n_t, summary['topn_sharpe_10']   = report("top10_sharpe", top10, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['topn_sharpe_10'] = (1 + n_t).cumprod()
    n_qp, summary['QP_tcost']        = report("QP + t-cost", qp_w, close, ret, TRAIN_END, VAL_END, BOOK)
    eq_curves['QP_tcost'] = (1 + n_qp).cumprod()

    # Plot
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"equal_weight":"C0","adaptive":"C1","risk_parity":"C2","billions":"C4",
              "ic_weighted":"C5","sharpe_weighted":"C6","topn_sharpe_10":"C7","QP_tcost":"C3"}
    for name, eq in eq_curves.items():
        sr = summary[name]['full']['sharpe_net']
        lw = 1.6 if name == 'QP_tcost' else 1.0
        ax.plot(eq.index, eq.values, lw=lw, color=colors.get(name,"k"),
                label=f"{name:13s} net SR(full)={sr:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END), color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 — combiner showdown @ ${BOOK:,.0f} book\n"
                 f"({len(selected)} pruned alphas (corr<{CORR_CAP}), {UNIV_NAME}, max_w={MAX_W})")
    ax.set_ylabel("Equity (log, start=1.0)"); ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")

    print(f"\n=== BEST NET SR (FULL) on {len(selected)} pruned alphas ===")
    best = sorted(summary.items(), key=lambda x: -x[1]['full']['sharpe_net'])
    for name, s in best:
        print(f"  {name:15s}  TRAIN={s['train']['sharpe_net']:+5.2f}  "
              f"VAL={s['val']['sharpe_net']:+5.2f}  "
              f"TEST={s['test']['sharpe_net']:+5.2f}  "
              f"FULL={s['full']['sharpe_net']:+5.2f}")


if __name__ == "__main__":
    main()
