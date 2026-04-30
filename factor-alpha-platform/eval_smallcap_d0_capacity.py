"""
Capacity analysis for the 47-alpha QP composite at MCAP_100M_500M / delay-0 / MOC.

Method:
  1. Load saved QP composite weights (recompute at $500K once — QP weights depend
     on cost-per-trade not book level, since commission_per_share is fixed).
  2. Sweep book sizes from $100K to $20M.
  3. For each book:
     - Recompute realistic per-share IB MOC cost (per-order min becomes negligible
       at large books; impact and commission scale linearly).
     - Compute net Sharpe (gross PnL is the same; net = gross - cost).
     - Compute MOC participation: largest daily position $ / (MOC_FRAC × ADV20).
     - Flag capacity hit if any name's MOC-share exceeds threshold.

Reports cutoffs:
  - Min book for net SR ≥ 4 (lower bound — fees too high below)
  - Max book for net SR ≥ 4 AND max MOC-share ≤ 30% (upper bound — liquidity)
  - Optimal book by net Sharpe
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
LAMBDA_RISK = 5.0
KAPPA_TC    = 30.0
TRAIN_END   = "2024-01-01"
VAL_END     = "2025-04-01"

# Per-share IB MOC fee model
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

# Capacity model
MOC_FRAC             = 0.10  # fraction of ADV20 that prints in the close auction
MAX_MOC_PARTICIPATION = 0.30  # max % of MOC print we'd take per name (any single day)
NET_SR_THRESHOLD     = 4.0   # call book "viable" if net SR ≥ this

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_capacity.png"


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


def qp_weights(alpha, close, ret, uni, n_names, dates):
    """Original QP — no ADV constraint. Weights independent of book size."""
    print("Running QP walk-forward (no ADV cap)...")
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
    print(f"QP done in {time.time()-t0:.0f}s")
    return w_qp


def qp_weights_advcap(alpha, close, ret, uni, n_names, dates, adv20, book):
    """QP with per-name ADV cap.

    Per-name weight constraint:
        |w_i| * book  <=  MOC_FRAC * ADV20_i * MAX_MOC_PARTICIPATION
    i.e. position dollars <= max acceptable share of MOC print.

    With this cap, the optimizer DELIBERATELY shrinks/zeros positions in low-ADV
    names so executable book size scales without losing alpha to slippage.

    Weights DO depend on book size now — must rerun per book.
    """
    print(f"Running QP+ADV-cap walk-forward (book=${book/1e6:.2f}M)...")
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
        adv_s = adv20.iloc[i].values[idx]
        # Per-name cap: |w_i| <= min(MAX_W, MOC_FRAC * MAX_PART * ADV_i / book)
        adv_cap = np.where(
            (adv_s > 0) & np.isfinite(adv_s),
            MOC_FRAC * MAX_MOC_PARTICIPATION * adv_s / book,
            0.0,
        )
        per_name_cap = np.minimum(MAX_W, adv_cap)
        kappa = KAPPA_TC * (COMMISSION_PER_SHARE / np.maximum(price_s, 0.01) + IMPACT_BPS/1e4)
        w = cp.Variable(len(idx))
        risk = cp.sum(cp.multiply(sig_s**2, cp.square(w)))
        tc = cp.sum(cp.multiply(kappa, cp.abs(w - wp_s)))
        obj = cp.Maximize(a_s @ w - 0.5 * LAMBDA_RISK * risk - tc)
        cons = [cp.norm(w, 1) <= 1.0, cp.abs(w) <= per_name_cap, cp.sum(w) == 0]
        try:
            cp.Problem(obj, cons).solve(solver=cp.OSQP, warm_start=True, verbose=False)
            sol = w.value
        except Exception:
            w_qp.iloc[i] = w_prev; continue
        new_w = np.zeros(n_names); new_w[idx] = sol
        w_qp.iloc[i] = new_w; w_prev = new_w
    print(f"  QP+ADV-cap done in {time.time()-t0:.0f}s")
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
    adv20 = mats.get("adv20")  # for capacity check
    if adv20 is None:
        print("WARN: adv20 not found"); return

    engine = FastExpressionEngine(data_fields=mats)

    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         GROUP BY a.id ORDER BY a.id""").fetchall()
    print(f"Combining {len(rows)} alphas: {[r[0] for r in rows]}")

    normed = []
    for aid, expr in rows:
        try:
            normed.append(proc_signal(engine.evaluate(expr), uni, cls))
        except Exception as e:
            print(f"skip #{aid}: {e}")
    eq_alpha = sum(normed) / len(normed)
    eq_alpha = normalize_clip(eq_alpha)

    # === Run baseline QP once (no ADV cap, weights book-independent) ===
    w_qp_nocap = qp_weights(eq_alpha, close, ret, uni, len(tickers), dates)

    # Sweep book sizes
    books = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
             5_000_000, 7_500_000, 10_000_000, 15_000_000, 20_000_000,
             50_000_000, 100_000_000]

    nx = ret.shift(-1)
    g_nocap = (w_qp_nocap * nx).sum(axis=1).fillna(0)
    ann = np.sqrt(252)

    def stats_for(weights, gpnl, book):
        cost = realistic_cost(weights, close, book)
        n = gpnl - cost
        s = lambda x: x.mean()/x.std()*ann if x.std()>0 else float('nan')
        sr_train = s(n.loc[:TRAIN_END]); sr_val = s(n.loc[TRAIN_END:VAL_END])
        sr_test = s(n.loc[VAL_END:]); sr_full = s(n)
        cost_yr = cost.mean()*252*100
        pos_dollar = (weights * book).abs()
        moc_dollar = (adv20 * MOC_FRAC).reindex(index=dates, columns=tickers)
        moc_part = (pos_dollar / moc_dollar.where(moc_dollar > 0)).where(pos_dollar > 0)
        active_part = moc_part.stack().dropna()
        if len(active_part) > 0:
            med_part = active_part.median()*100
            p99_part = active_part.quantile(0.99)*100
            cap_hit_pct = (active_part > MAX_MOC_PARTICIPATION).mean()*100
        else:
            med_part = p99_part = cap_hit_pct = float('nan')
        gmv = weights.abs().sum(axis=1).mean()  # avg gross dollar exposure as fraction of book
        n_active = (weights.abs() > 1e-6).sum(axis=1).mean()
        return dict(book=book, cost_yr=cost_yr, sr_train=sr_train, sr_val=sr_val,
                    sr_test=sr_test, sr_full=sr_full,
                    med_moc_pct=med_part, p99_moc_pct=p99_part, cap_hit_pct=cap_hit_pct,
                    gmv=float(gmv), n_active=float(n_active))

    hdr = f"{'Book':>10s}  {'cost%/yr':>9s}  {'TRAIN':>7s}  {'VAL':>7s}  {'TEST':>7s}  {'FULL':>7s}  {'p99 MOC%':>9s}  {'cap_hit%':>9s}  {'n_act':>6s}"
    fmt = lambda r: (f"  ${r['book']/1000:>7.0f}K  {r['cost_yr']:>7.2f}%  "
                     f"{r['sr_train']:>+6.2f}  {r['sr_val']:>+6.2f}  {r['sr_test']:>+6.2f}  {r['sr_full']:>+6.2f}  "
                     f"{r['p99_moc_pct']:>7.2f}%  {r['cap_hit_pct']:>7.2f}%  {r['n_active']:>5.0f}")

    print(f"\n=== A. QP NO-ADV-CAP (current default — book-independent weights) ===")
    print(hdr)
    nocap_results = []
    for book in books:
        r = stats_for(w_qp_nocap, g_nocap, book)
        nocap_results.append(r)
        print(fmt(r))

    print(f"\n=== B. QP +ADV-CAP (per-name cap = MOC_FRAC*{MAX_MOC_PARTICIPATION*100:.0f}%*ADV20 / book) ===")
    print(hdr)
    capped_results = []
    for book in books:
        w = qp_weights_advcap(eq_alpha, close, ret, uni, len(tickers), dates, adv20, book)
        g = (w * nx).sum(axis=1).fillna(0)
        r = stats_for(w, g, book)
        capped_results.append(r)
        print(fmt(r))

    df_nocap = pd.DataFrame(nocap_results)
    df_capped = pd.DataFrame(capped_results)

    print(f"\n=== CUTOFFS (net-SR threshold {NET_SR_THRESHOLD}, cap_hit% < 5%) ===")
    print("  A. NO-CAP:")
    viable = df_nocap[(df_nocap['sr_full'] >= NET_SR_THRESHOLD) & (df_nocap['cap_hit_pct'] <= 5.0)]
    if len(viable) > 0:
        print(f"    Max viable book: ${viable['book'].max()/1e6:.2f}M (cost {viable.iloc[-1]['cost_yr']:.2f}%/yr, SR {viable.iloc[-1]['sr_full']:.2f})")
    print("  B. ADV-CAP:")
    viable = df_capped[(df_capped['sr_full'] >= NET_SR_THRESHOLD) & (df_capped['cap_hit_pct'] <= 5.0)]
    if len(viable) > 0:
        print(f"    Max viable book: ${viable['book'].max()/1e6:.2f}M (cost {viable.iloc[-1]['cost_yr']:.2f}%/yr, SR {viable.iloc[-1]['sr_full']:.2f})")

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    ax = axes[0]
    ax.plot(df_nocap['book']/1e6, df_nocap['sr_full'], 'o-', label='No-cap FULL net SR', lw=1.5, color='C0')
    ax.plot(df_capped['book']/1e6, df_capped['sr_full'], 's-', label='ADV-cap FULL net SR', lw=1.5, color='C3')
    ax.axhline(NET_SR_THRESHOLD, ls=':', color='grey')
    ax.set_xscale("log"); ax.set_xlabel("Book size ($M, log)"); ax.set_ylabel("Net Sharpe (annualized)")
    ax.set_title(f"SMALLCAP_D0 Capacity — QP without vs with ADV-cap ({len(rows)} alphas)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_nocap['book']/1e6, df_nocap['cap_hit_pct'], 'o-', label='No-cap: % positions > 30% MOC', color='C0')
    ax2.plot(df_capped['book']/1e6, df_capped['cap_hit_pct'], 's-', label='ADV-cap: % positions > 30% MOC', color='C3')
    ax2.axhline(5.0, ls=':', color='grey', label='5% threshold')
    ax2.set_xscale("log"); ax2.set_xlabel("Book size ($M, log)"); ax2.set_ylabel("Capacity-violating positions (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
