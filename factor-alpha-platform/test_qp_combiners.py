"""
Combiner × risk-model sweep on SMALLCAP_D0.

For every combiner ∈ {equal, billions, risk_parity, ic_weighted,
sharpe_weighted, topn_sharpe, adaptive}, run the QP under TWO risk models:
  - diagonal      (current production: ½λ Σ σ²_i w²_i)
  - style+pca     (Barra-ish: ½λ ||L_style' w||² + ||B_pca' w||² + s²·w²)

Reports TRAIN/VAL/TEST/FULL net SR for each (combiner × risk_model) pair.

Sanity goal: (equal × diagonal) must reproduce the prior baseline of
~4.92 FULL net SR from eval_smallcap_d0_final.py.
"""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.qp import run_walkforward
from src.portfolio.risk_model import (
    build_diagonal, build_pca, build_style, build_style_pca,
    build_style_factors,
)
from src.portfolio.combiners import (
    process_signal,
    combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions,
    combiner_ic_weighted, combiner_sharpe_weighted, combiner_topn_sharpe,
)

UNIV_NAME = "MCAP_100M_500M"
MAX_W = 0.02
BOOK = 500_000
LAMBDA_RISK = 5.0
KAPPA_TC = 30.0
TRAIN_END = "2024-01-01"
VAL_END = "2025-04-01"
N_PCA_FACTORS = 5

COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN = 0.35
SEC_FEE_PER_DOLLAR = 27.80e-6
SELL_FRACTION = 0.50
IMPACT_BPS = 0.5
BORROW_BPS_ANNUAL = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB = ROOT / "data/alpha_results.db"


def proc_signal_subind(s, uni, cls):
    """Match eval_smallcap_d0_final.proc_signal: subindustry-demean + L1 + clip."""
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]
            s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
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


def make_risk_model_fn(name, B_style_stack=None, k_pca=N_PCA_FACTORS):
    """Returns a callable matching the risk_model_fn signature in qp.py."""
    def fn(i, idx, vol_today, ret_mat, factor_window):
        if name == 'diagonal':
            return build_diagonal(vol_today)
        if i < factor_window + 1:
            return build_diagonal(vol_today)
        R_win = ret_mat[i - factor_window:i, idx]
        R_win = np.where(np.isfinite(R_win), R_win, 0.0)
        if name == 'pca':
            return build_pca(R_win, k_pca)
        if B_style_stack is None:
            return build_pca(R_win, k_pca)
        B_t = B_style_stack[i, idx, :]
        col_ok = np.all(np.isfinite(B_t), axis=0)
        if col_ok.sum() == 0:
            return build_pca(R_win, k_pca)
        B_style = B_t[:, col_ok]
        if name == 'style':
            return build_style(R_win, B_style)
        if name == 'style+pca':
            return build_style_pca(R_win, B_style, k_pca)
        raise ValueError(name)
    return fn


def split_metrics(g, n, w, ann):
    to = w.diff().abs().sum(axis=1).mean() / 2
    rows = []
    for lab, sl in [("TRAIN", slice(None, TRAIN_END)),
                    ("VAL",   slice(TRAIN_END, VAL_END)),
                    ("TEST",  slice(VAL_END, None)),
                    ("FULL",  slice(None, None))]:
        gg = g.loc[sl]; nn = n.loc[sl]
        srn = nn.mean() / nn.std() * ann if nn.std() > 0 else float('nan')
        rows.append((lab, srn))
    return rows, to


def main():
    print(f"=== Loading {UNIV_NAME} ===", flush=True)
    uni = pd.read_parquet(UNIV_DIR / f"{UNIV_NAME}.parquet").astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > 0.5].index.tolist())
    uni = uni[valid]
    dates = uni.index
    tickers = uni.columns.tolist()

    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc:
            mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)

    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)

    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND (a.notes LIKE '%SMALLCAP_D0_v2%' OR a.notes LIKE '%SMALLCAP_D0_v3%')
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"Loading {len(rows)} alphas...", flush=True)

    # Subindustry-pre-normalize each raw alpha (matches eval_smallcap_d0_final).
    # This is what 'raw' means going into the combiners.
    alpha_signals = {}
    for aid, expr in rows:
        sig = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        alpha_signals[aid] = proc_signal_subind(sig, uni, cls)

    # Style factors (Barra-ish, from risk_model.build_style_factors)
    print("Building style factors...", flush=True)
    style_factors = build_style_factors(mats)
    print(f"  Got {len(style_factors)} style factors: {list(style_factors.keys())}", flush=True)
    factor_names = list(style_factors.keys())
    n_factors = len(factor_names)
    n_dates = len(dates)
    n_names = len(tickers)
    B_style_stack = np.full((n_dates, n_names, n_factors), np.nan, dtype=np.float32)
    for k, fname in enumerate(factor_names):
        df = style_factors[fname].reindex(index=dates, columns=tickers)
        B_style_stack[:, :, k] = df.values.astype(np.float32)
    print(f"Universe: {n_names} tickers, {n_dates} dates "
          f"({dates[0].date()} → {dates[-1].date()})", flush=True)

    # Combiners (input alpha producers)
    combiner_specs = [
        ("equal",      lambda: combiner_equal(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
        ("billions",   lambda: combiner_billions(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
        ("risk_par",   lambda: combiner_risk_parity(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
        ("ic_wt",      lambda: combiner_ic_weighted(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
        ("sharpe_wt",  lambda: combiner_sharpe_weighted(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
        ("topn",       lambda: combiner_topn_sharpe(alpha_signals, mats, uni, ret, max_wt=MAX_W, top_n=10)),
        ("adaptive",   lambda: combiner_adaptive(alpha_signals, mats, uni, ret, max_wt=MAX_W)),
    ]
    combined_alphas = {}
    for cname, cfn in combiner_specs:
        t0 = time.time()
        print(f"  building combiner: {cname}...", flush=True)
        out = cfn()
        out = out.reindex(index=dates, columns=tickers).fillna(0)
        # Re-L1-normalize + clip to match eval_smallcap_d0_final.py's
        # post-combiner preprocessing. Without this, combiner_equal etc. return
        # `sum/N` with magnitude ~1/sqrt(N) — too small relative to per-name
        # κ t-cost, which starves the QP and drops SR.
        out = out.div(out.abs().sum(axis=1).replace(0, np.nan), axis=0)
        out = out.clip(-MAX_W, MAX_W).fillna(0)
        combined_alphas[cname] = out
        print(f"  combiner {cname} built in {time.time()-t0:.0f}s", flush=True)

    risk_model_specs = [
        ("diag",       make_risk_model_fn('diagonal')),
        ("style+pca",  make_risk_model_fn('style+pca', B_style_stack=B_style_stack)),
    ]

    ann = np.sqrt(252)
    results = {}

    for cname, alpha in combined_alphas.items():
        for rname, rfn in risk_model_specs:
            label = f"{cname}/{rname}"
            print(f"\n>>> {label}", flush=True)
            w = run_walkforward(alpha, close, ret, uni, rfn,
                                 lambda_risk=LAMBDA_RISK, kappa_tc=KAPPA_TC, max_w=MAX_W,
                                 commission_per_share=COMMISSION_PER_SHARE,
                                 impact_bps=IMPACT_BPS,
                                 label=label, verbose=True)
            g = (w * ret.shift(-1)).sum(axis=1).fillna(0)
            cost = realistic_cost(w, close, BOOK)
            n = g - cost
            rows_out, to = split_metrics(g, n, w, ann)
            cell = {lab: sr for lab, sr in rows_out}
            cell['_to'] = to
            cell['_cost_bps'] = cost.mean() * 1e4
            results[(cname, rname)] = cell
            print(f"  TO={to*100:.1f}%/d  cost={cost.mean()*1e4:.2f}bps/d "
                  f"TRAIN={cell['TRAIN']:+.2f} VAL={cell['VAL']:+.2f} "
                  f"TEST={cell['TEST']:+.2f} FULL={cell['FULL']:+.2f}", flush=True)

    # Final summary
    print("\n" + "=" * 90)
    print("SUMMARY: net SR by (combiner × risk_model)")
    print("=" * 90)
    print(f"{'combiner':10s} | {'risk':10s} | {'TRAIN':>7s} | {'VAL':>7s} | "
          f"{'TEST':>7s} | {'FULL':>7s} | {'TO%/d':>6s} | {'cost(bps)':>9s}")
    print("-" * 90)
    for cname in combined_alphas.keys():
        for rname, _ in risk_model_specs:
            r = results[(cname, rname)]
            print(f"{cname:10s} | {rname:10s} | "
                  f"{r['TRAIN']:+7.2f} | {r['VAL']:+7.2f} | "
                  f"{r['TEST']:+7.2f} | {r['FULL']:+7.2f} | "
                  f"{r['_to']*100:6.1f} | {r['_cost_bps']:9.2f}")

    # Sanity check
    base_full = results[("equal", "diag")]['FULL']
    print("\n" + "=" * 90)
    print(f"SANITY: equal/diag FULL net SR = {base_full:+.3f}  "
          f"(prior baseline ~4.92 from eval_smallcap_d0_final.py)")
    if abs(base_full - 4.92) > 0.30:
        print(f"  WARN: diverges from prior by {base_full - 4.92:+.3f} — "
              f"likely combiner_equal vs manual mean differs")


if __name__ == "__main__":
    main()
