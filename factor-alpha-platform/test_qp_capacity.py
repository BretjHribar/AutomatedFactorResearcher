"""
Capacity sweep on SMALLCAP_D0 — equal-weight combiner × {diag, style+pca}.

For each book size, runs the unified QP with the ADV-style per-name cap
    |w_i| ≤ min(MAX_W, MOC_FRAC × MAX_MOC_PARTICIPATION × ADV20_i / book)
which forces the optimizer to scale OUT of low-ADV names as book grows.

Reports for each book × risk_model:
  TRAIN / VAL / TEST / FULL net SR    (gross PnL minus realized fees)
  cost%/yr (realistic per-share IB MOC + impact + borrow)
  p99 MOC participation %             (worst-case execution share)
  cap_hit %                            (fraction of (date, name) at the cap)
  avg gross dollar exposure / book
  avg active-name count
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
from src.portfolio.combiners import combiner_equal

UNIV_NAME = "MCAP_100M_500M"
MAX_W = 0.02
LAMBDA_RISK = 5.0
KAPPA_TC = 30.0
TRAIN_END = "2024-01-01"
VAL_END = "2025-04-01"
N_PCA_FACTORS = 5

# Per-share IB MOC fee model
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN = 0.35
SEC_FEE_PER_DOLLAR = 27.80e-6
SELL_FRACTION = 0.50
IMPACT_BPS = 0.5
BORROW_BPS_ANNUAL = 50

# Capacity model
MOC_FRAC = 0.10               # fraction of ADV20 that prints in close auction
MAX_MOC_PARTICIPATION = 0.30  # max % of MOC print per name per day
NET_SR_THRESHOLD = 4.0
CAP_HIT_THRESHOLD = 5.0       # %

BOOKS = [100_000, 250_000, 500_000, 1_000_000, 1_500_000, 2_000_000]

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB = ROOT / "data/alpha_results.db"


def proc_signal_subind(s, uni, cls):
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
        if name == 'style+pca':
            return build_style_pca(R_win, B_style, k_pca)
        raise ValueError(name)
    return fn


def stats_for(w, close, ret, adv, book, dates, tickers, ann):
    nx = ret.shift(-1)
    g = (w * nx).sum(axis=1).fillna(0)
    cost = realistic_cost(w, close, book)
    n = g - cost
    s = lambda x: x.mean() / x.std() * ann if x.std() > 0 else float('nan')
    r = lambda x: x.mean() * 252 * 100
    sr_val   = s(n.loc[TRAIN_END:VAL_END])
    sr_test  = s(n.loc[VAL_END:])
    ret_val  = r(n.loc[TRAIN_END:VAL_END])
    ret_test = r(n.loc[VAL_END:])
    cost_yr = cost.mean() * 252 * 100
    pos_dollar = (w * book).abs()
    moc_dollar = (adv * MOC_FRAC).reindex(index=dates, columns=tickers)
    moc_part = (pos_dollar / moc_dollar.where(moc_dollar > 0)).where(pos_dollar > 0)
    active_part = moc_part.stack().dropna()
    if len(active_part) > 0:
        med_part = active_part.median() * 100
        p99_part = active_part.quantile(0.99) * 100
        cap_hit = (active_part > MAX_MOC_PARTICIPATION).mean() * 100
    else:
        med_part = p99_part = cap_hit = float('nan')
    gmv = w.abs().sum(axis=1).mean()
    n_active = (w.abs() > 1e-6).sum(axis=1).mean()
    return dict(book=book, cost_yr=cost_yr,
                sr_val=sr_val, sr_test=sr_test,
                ret_val=ret_val, ret_test=ret_test,
                med_moc_pct=med_part, p99_moc_pct=p99_part, cap_hit_pct=cap_hit,
                gmv=float(gmv), n_active=float(n_active))


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
    adv = mats.get("adv20")
    if adv is None:
        print("ERROR: adv20 matrix not found — required for capacity sweep", flush=True)
        return
    engine = FastExpressionEngine(data_fields=mats)

    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND (a.notes LIKE '%SMALLCAP_D0_v2%' OR a.notes LIKE '%SMALLCAP_D0_v3%')
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"Loading {len(rows)} alphas...", flush=True)

    alpha_signals = {}
    for aid, expr in rows:
        sig = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        alpha_signals[aid] = proc_signal_subind(sig, uni, cls)

    # Build equal-weight combiner alpha (the OOS-best combiner from sweep)
    print("Building equal-weight combiner...", flush=True)
    eq = combiner_equal(alpha_signals, mats, uni, ret, max_wt=MAX_W)
    eq = eq.reindex(index=dates, columns=tickers).fillna(0)
    eq = eq.div(eq.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    # Build style factor stack
    print("Building style factors...", flush=True)
    style_factors = build_style_factors(mats)
    factor_names = list(style_factors.keys())
    n_factors = len(factor_names)
    n_dates = len(dates); n_names = len(tickers)
    B_style_stack = np.full((n_dates, n_names, n_factors), np.nan, dtype=np.float32)
    for k, fname in enumerate(factor_names):
        df = style_factors[fname].reindex(index=dates, columns=tickers)
        B_style_stack[:, :, k] = df.values.astype(np.float32)
    print(f"  Style factors: {factor_names}", flush=True)

    risk_model_specs = [
        ("diag",      make_risk_model_fn('diagonal')),
        ("style+pca", make_risk_model_fn('style+pca', B_style_stack=B_style_stack)),
    ]

    ann = np.sqrt(252)
    all_results = {rname: [] for rname, _ in risk_model_specs}

    hdr = (f"{'Book':>10s}  {'cost%/yr':>9s}  "
           f"{'SR_val':>7s}  {'SR_test':>7s}  {'avgSR':>6s}  "
           f"{'ret_val%':>9s}  {'ret_test%':>10s}  {'avg_ret%':>9s}  "
           f"{'p99 MOC%':>8s}  {'cap_hit%':>8s}  {'gmv':>5s}  {'n_act':>6s}")
    fmt = (lambda r: (f"  ${r['book']/1000:>7.0f}K  {r['cost_yr']:>7.2f}%  "
                       f"{r['sr_val']:>+6.2f}  {r['sr_test']:>+6.2f}  "
                       f"{(r['sr_val'] + r['sr_test'])/2:>+5.2f}  "
                       f"{r['ret_val']:>+8.2f}%  {r['ret_test']:>+9.2f}%  "
                       f"{(r['ret_val'] + r['ret_test'])/2:>+8.2f}%  "
                       f"{r['p99_moc_pct']:>6.2f}%  {r['cap_hit_pct']:>6.2f}%  "
                       f"{r['gmv']:>5.2f}  {r['n_active']:>5.0f}"))

    for rname, rfn in risk_model_specs:
        print(f"\n=== Risk model: {rname} ===", flush=True)
        print(hdr, flush=True)
        for book in BOOKS:
            t0 = time.time()
            w = run_walkforward(eq, close, ret, uni, rfn,
                                 lambda_risk=LAMBDA_RISK, kappa_tc=KAPPA_TC, max_w=MAX_W,
                                 commission_per_share=COMMISSION_PER_SHARE,
                                 impact_bps=IMPACT_BPS,
                                 adv=adv, book=book,
                                 moc_frac=MOC_FRAC,
                                 max_moc_participation=MAX_MOC_PARTICIPATION,
                                 label=f"{rname}@${book/1e6:.2f}M",
                                 verbose=False)
            r = stats_for(w, close, ret, adv, book, dates, tickers, ann)
            r['_elapsed'] = time.time() - t0
            all_results[rname].append(r)
            print(fmt(r), flush=True)

    # Side-by-side summary by book — OOS-only (avg of VAL+TEST)
    print("\n" + "=" * 116)
    print("SIDE-BY-SIDE: equal × {diag, style+pca} — avg(VAL,TEST) net SR & ret%, cap_hit%")
    print("=" * 116)
    print(f"{'Book':>11s} | "
          f"{'d avgSR':>8s} {'d avg_ret%':>11s} {'d cap%':>7s}  | "
          f"{'s+p avgSR':>9s} {'s+p avg_ret%':>13s} {'s+p cap%':>9s}")
    print("-" * 116)
    for i, book in enumerate(BOOKS):
        d = all_results['diag'][i]
        s = all_results['style+pca'][i]
        d_avg_sr = (d['sr_val'] + d['sr_test']) / 2
        s_avg_sr = (s['sr_val'] + s['sr_test']) / 2
        d_avg_ret = (d['ret_val'] + d['ret_test']) / 2
        s_avg_ret = (s['ret_val'] + s['ret_test']) / 2
        print(f"  ${book/1e6:>7.2f}M | "
              f"{d_avg_sr:>+7.2f} {d_avg_ret:>+10.2f}% {d['cap_hit_pct']:>6.2f}% | "
              f"{s_avg_sr:>+8.2f} {s_avg_ret:>+12.2f}% {s['cap_hit_pct']:>8.2f}%")

    # Cutoffs — OOS avgSR threshold instead of FULL
    print(f"\n=== Capacity cutoffs (avg(VAL,TEST) SR ≥ {NET_SR_THRESHOLD}, cap_hit% ≤ {CAP_HIT_THRESHOLD}) ===")
    for rname in ['diag', 'style+pca']:
        df = pd.DataFrame(all_results[rname])
        df['avgSR'] = (df['sr_val'] + df['sr_test']) / 2
        df['avg_ret'] = (df['ret_val'] + df['ret_test']) / 2
        viable = df[(df['avgSR'] >= NET_SR_THRESHOLD)
                    & (df['cap_hit_pct'] <= CAP_HIT_THRESHOLD)]
        if len(viable) > 0:
            top = viable.iloc[-1]
            print(f"  {rname:10s}: max viable book = ${top['book']/1e6:.2f}M  "
                  f"(avgSR {top['avgSR']:+.2f}, avg_ret {top['avg_ret']:+.2f}%/yr, "
                  f"cost {top['cost_yr']:.2f}%/yr, cap_hit {top['cap_hit_pct']:.1f}%)")
        else:
            print(f"  {rname:10s}: NO viable book at threshold")


if __name__ == "__main__":
    main()
