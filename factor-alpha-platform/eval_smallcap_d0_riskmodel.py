"""
Compare two neutralization schemes for the SMALLCAP_D0_v2 alpha set:

  A. subindustry demean (current method)
  B. full risk model — residualize against beta, size, value, momentum,
     profitability, low_vol, growth, leverage AND industry dummies

For each scheme, build the equal-weight portfolio across the saved alphas and
report TRAIN / VAL / TEST gross + net Sharpe @ $500K with the realistic
per-share fee model.
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.risk_model import build_style_factors, neutralize

UNIV = "TOP1500TOP2500"
MAX_W = 0.001
TRAIN_END = "2024-01-01"   # the v2 alphas were trained on 2020-2024
VAL_END   = "2025-04-01"   # rough mid-point of OOS
BOOK = 500_000.0
COMMISSION_PER_SHARE = 0.0045
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_v2_riskmodel_compare.png"


def load_universe():
    df = pd.read_parquet(UNIV_DIR / f"{UNIV}.parquet").astype(bool)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    cov = df.sum(axis=0)/len(df)
    valid = sorted(cov[cov>0.5].index.tolist())
    return df[valid]


def load_matrices(tickers, dates):
    out = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce"); df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc: out[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)
    return out


def post_process(sig, uni, group_labels=None, factors=None, scheme="subindustry"):
    """Apply universe mask + neutralization (per scheme) + L1-normalize + clip."""
    s = sig.astype(float).where(uni, np.nan)
    if scheme == "subindustry" and group_labels is not None:
        for g in group_labels.dropna().unique():
            m = (group_labels == g).values
            if m.any():
                sub = s.iloc[:, m]
                s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    elif scheme == "riskmodel" and factors is not None:
        s = neutralize(s, factors, group_labels, include_industry=True)
    abs_sum = s.abs().sum(axis=1).replace(0, np.nan)
    s = s.div(abs_sum, axis=0).clip(-MAX_W, MAX_W).fillna(0)
    return s


def realistic_cost(combined_w, close, book):
    pos = combined_w * book
    trd = pos.diff().abs()
    safe = close.where(close > 0)
    shares = trd / safe
    commission_d = (shares * COMMISSION_PER_SHARE).sum(axis=1)
    sec_d        = (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
    impact_d     = (trd * IMPACT_BPS / 1e4).sum(axis=1)
    short_d      = (-pos.clip(upper=0)).sum(axis=1)
    borrow_d     = short_d * (BORROW_BPS_ANNUAL / 1e4) / 252.0
    daily_pct = (commission_d + sec_d + impact_d + borrow_d) / book
    return daily_pct


def evaluate(name, w, close, ret):
    nx = ret.shift(-1)
    g = (w * nx).sum(axis=1).fillna(0)
    cost = realistic_cost(w, close, BOOK)
    n = g - cost.reindex(g.index).fillna(0)
    ann = np.sqrt(252)
    def split(s_g, s_n, lab):
        srg = s_g.mean()/s_g.std()*ann if s_g.std()>0 else float("nan")
        srn = s_n.mean()/s_n.std()*ann if s_n.std()>0 else float("nan")
        return f"  {lab:6s}  SR_g={srg:+5.2f}  SR_n={srn:+5.2f}  ret_g={s_g.mean()*252*100:+6.1f}%  ret_n={s_n.mean()*252*100:+6.1f}%"
    print(f"\n=== {name} ===  cost={cost.mean()*1e4:.2f}bps/day → {cost.mean()*252*100:.2f}%/yr drag")
    print(split(g.loc[:TRAIN_END], n.loc[:TRAIN_END], "TRAIN"))
    print(split(g.loc[TRAIN_END:VAL_END], n.loc[TRAIN_END:VAL_END], "VAL"))
    print(split(g.loc[VAL_END:], n.loc[VAL_END:], "TEST"))
    print(split(g, n, "FULL"))
    return n


def main():
    print(f"=== Loading {UNIV} universe ===")
    uni = load_universe()
    dates = uni.index; tickers = uni.columns.tolist()
    print(f"  {len(tickers)} tickers, {len(dates)} dates")

    print("=== Loading matrices ===")
    mats = load_matrices(tickers, dates)
    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    engine = FastExpressionEngine(data_fields=mats)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)

    print("=== Building risk-model factor exposures ===")
    factors = build_style_factors(mats)
    print(f"  built {len(factors)} style factors: {list(factors.keys())}")
    n_industries = cls.dropna().unique().size
    print(f"  industry dummies: {n_industries-1} (one held out)")

    # Pull SMALLCAP_D0_v2 alphas
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT a.id, a.expression, MAX(e.sharpe_is)
          FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0_v2%'
         GROUP BY a.id ORDER BY MAX(e.sharpe_is) DESC
    """).fetchall()
    print(f"\n=== {len(rows)} v2 alphas to test ===")
    for aid, expr, sr in rows:
        print(f"  #{aid} SR={sr:.2f}  {expr[:80]}")
    if not rows:
        print("No v2 alphas yet"); return 1

    raw = {aid: engine.evaluate(expr) for aid, expr, _ in rows}

    # ---- Scheme A: subindustry demean only ----
    print("\n========== SCHEME A: subindustry demean ==========")
    normed_A = {aid: post_process(r, uni, cls, factors, scheme="subindustry") for aid, r in raw.items()}
    eq_A = sum(normed_A.values())/len(normed_A)
    eq_A = eq_A.div(eq_A.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)
    nA = evaluate("equal-weight (subindustry only)", eq_A, close, ret)

    # ---- Scheme B: full risk-model residualization ----
    print("\n========== SCHEME B: full risk-model residualize ==========")
    print("  (computing residuals — this may take 30-60s per alpha)")
    normed_B = {}
    for aid, r in raw.items():
        print(f"    neutralizing #{aid} ...")
        normed_B[aid] = post_process(r, uni, cls, factors, scheme="riskmodel")
    eq_B = sum(normed_B.values())/len(normed_B)
    eq_B = eq_B.div(eq_B.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)
    nB = evaluate("equal-weight (full risk model)", eq_B, close, ret)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 6.5))
    eqA = (1 + nA).cumprod()
    eqB = (1 + nB).cumprod()
    annA = nA.mean()/nA.std() * np.sqrt(252) if nA.std()>0 else float('nan')
    annB = nB.mean()/nB.std() * np.sqrt(252) if nB.std()>0 else float('nan')
    ax.plot(eqA.index, eqA.values, color="C0", lw=1.3, label=f"subindustry-only  net SR={annA:+.2f}")
    ax.plot(eqB.index, eqB.values, color="C3", lw=1.3, label=f"full risk-model    net SR={annB:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END),   color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0_v2 — risk-model neutralization @ ${BOOK:,.0f} book\n"
                 f"({len(rows)} alphas, train 2020-2024, equal-weight combiner)")
    ax.set_ylabel("Equity (log)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
