"""
Honest fee model for the SMALLCAP_D0 portfolio at $500K.

Costs modeled per-share on actual close prices (not assumed-bps):

  Commission (IBKR Tiered MOC):     $0.0035/share + $0.0010/share venue ≈ $0.0045/share
  SEC fee (sells only):             $27.80 per $1M sold = $0.0000278 / $traded
  FINRA TAF (sells only):           $0.000166/share, capped (small)
  Closing-auction imbalance/impact: ~0.5 bp on the trade notional (we are <0.5% of MOC print)
  Borrow cost on shorts:            ~50 bps/yr on the short side  (general collateral; small caps
                                    can be much higher but we'll model the avg)

For each (date, ticker) trade:
    shares     = |Δposition_$| / close_price
    commission = shares × $0.0045
    sec_fee    = 0.5 × |Δposition_$| × 27.8e-6   (only on sells; assume 50% are sells)
    impact     = |Δposition_$| × 0.5 bps
Daily cost  = sum over names + (gross_short × 50bps/yr ÷ 252)
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
from src.portfolio.combiners import combiner_billions, combiner_equal

# ── Config ────────────────────────────────────────────────────────────────
UNIVERSE  = "TOP1500TOP2500"
MAX_W     = 0.001
TRAIN_END = "2022-01-01"
VAL_END   = "2024-01-01"
BOOK      = 500_000.0   # the user's target book size

# Per-share cost model
COMMISSION_PER_SHARE = 0.0045        # $0.0035 IBKR + $0.0010 venue/closing-auction fees
SEC_FEE_PER_DOLLAR   = 27.80e-6      # SEC sec.31 fee (sells only)
SELL_FRACTION        = 0.50          # ~50% of trades are sells (long/short rebalances)
IMPACT_BPS           = 0.5           # bp on trade notional, MOC imbalance contribution
BORROW_BPS_ANNUAL    = 50            # 50 bps/yr on short-side gross — typical GC for small caps

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_realistic_fees_equity.png"


def load_universe():
    df = pd.read_parquet(UNIV_DIR / f"{UNIVERSE}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.values.dtype != bool:
        df = df.astype(bool)
    cov = df.sum(axis=0) / len(df)
    valid = sorted(cov[cov > 0.5].index.tolist())
    return df[valid]


def load_matrices(tickers, dates):
    out = {}
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
            out[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)
    return out


def proc(sig, uni, groups):
    s = sig.astype(float).where(uni, np.nan)
    for g in groups.dropna().unique():
        m = (groups == g).values
        if m.any():
            sub = s.iloc[:, m]
            s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    abs_sum = s.abs().sum(axis=1).replace(0, np.nan)
    s = s.div(abs_sum, axis=0).clip(-MAX_W, MAX_W).fillna(0)
    return s


def realistic_cost(combined_w, close, book):
    """Daily $ cost = commission + sec/finra + impact + borrow on shorts."""
    # $ position per (date, ticker) at this book size
    position_dollars = combined_w * book
    trade_dollars   = position_dollars.diff().abs()         # $ traded per name per day
    safe_close      = close.where(close > 0)
    shares_traded   = trade_dollars / safe_close

    commission_daily = (shares_traded * COMMISSION_PER_SHARE).sum(axis=1)
    sec_finra_daily  = (trade_dollars * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
    impact_daily     = (trade_dollars * IMPACT_BPS / 1e4).sum(axis=1)
    short_book = (-position_dollars.clip(upper=0)).sum(axis=1)        # gross short $
    borrow_daily     = short_book * (BORROW_BPS_ANNUAL / 1e4) / 252.0

    daily_cost_dollars = commission_daily + sec_finra_daily + impact_daily + borrow_daily
    daily_cost_pct = daily_cost_dollars / book
    return daily_cost_pct, dict(
        commission=commission_daily / book,
        sec_finra=sec_finra_daily / book,
        impact=impact_daily / book,
        borrow=borrow_daily / book,
    )


def main():
    print(f"=== Universe {UNIVERSE} | Book ${BOOK:,.0f} ===")
    uni = load_universe()
    dates = uni.index
    tickers = uni.columns.tolist()
    print(f"  {len(tickers)} tickers, {len(dates)} dates")

    print("=== Loading matrices ===")
    mats = load_matrices(tickers, dates)
    close = mats["close"]
    ret = close.pct_change(fill_method=None)
    engine = FastExpressionEngine(data_fields=mats)

    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet")
    groups = cls.iloc[-1].reindex(tickers)

    # Pull SR>=5 alphas
    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT a.id, a.expression, MAX(e.sharpe_is) sr
          FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         GROUP BY a.id HAVING sr>=5 ORDER BY sr DESC
    """).fetchall()
    print(f"\n=== {len(rows)} alphas with TRAIN SR>=5 ===")
    for aid, expr, sr in rows:
        print(f"  #{aid}  SR={sr:.2f}")

    # ---- Trade-weighted price profile (the question that started this) ---
    print(f"\n=== Trade-weighted price profile (alpha #36 used as representative) ===")
    aid_ref, expr_ref, _ = rows[0]   # #36 is highest-SR
    sig_ref = proc(engine.evaluate(expr_ref), uni, groups)
    to_ref = sig_ref.diff().abs()
    recent = to_ref.iloc[-252:]
    rprice = close.iloc[-252:].reindex_like(recent)
    mask = (recent.values > 0) & (~np.isnan(rprice.values))
    trades = recent.values[mask]
    prices = rprice.values[mask]
    w_avg_price = (trades * prices).sum() / trades.sum()
    p10 = np.percentile(prices, 10); p25 = np.percentile(prices, 25)
    p50 = np.percentile(prices, 50); p75 = np.percentile(prices, 75)
    p90 = np.percentile(prices, 90)
    print(f"  Distribution of close-prices on traded (date, name) pairs (last 252d):")
    print(f"    10%-ile  ${p10:6.2f}")
    print(f"    25%-ile  ${p25:6.2f}")
    print(f"    median   ${p50:6.2f}")
    print(f"    75%-ile  ${p75:6.2f}")
    print(f"    90%-ile  ${p90:6.2f}")
    print(f"    trade-weighted avg  ${w_avg_price:6.2f}")
    print(f"    → ${COMMISSION_PER_SHARE}/share at \${w_avg_price:.2f} avg = "
          f"{COMMISSION_PER_SHARE/w_avg_price*1e4:.2f} bps one-way commission alone")

    # ---- Build all 6 normed signals for the equal-weight portfolio ------
    raw = {aid: engine.evaluate(expr) for aid, expr, _ in rows}
    normed = {aid: proc(r, uni, groups) for aid, r in raw.items()}

    # equal-weight all 6 (and billions for comparison)
    eq6 = sum(normed.values()) / len(normed)
    eq6 = eq6.div(eq6.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    bil6 = combiner_billions(raw, mats, uni, ret, optim_lookback=60, max_wt=MAX_W)
    bil6 = bil6.div(bil6.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    # Top-1 (#36 alone)
    top1 = normed[36]
    top1 = top1.div(top1.abs().sum(axis=1).replace(0, np.nan), axis=0).clip(-MAX_W, MAX_W).fillna(0)

    portfolios = {
        "single #36":     top1,
        "equal_weight 6": eq6,
        "billions 6":     bil6,
    }

    # ---- Evaluate each ---------------------------------------------------
    ann = np.sqrt(252)
    nx = ret.shift(-1)

    def split_stats(name, gross, net, cost):
        def block(start, end, lab):
            g = gross.loc[start:end] if end else gross.loc[start:]
            n = net.loc[start:end] if end else net.loc[start:]
            srg = g.mean()/g.std()*ann if g.std()>0 else float('nan')
            srn = n.mean()/n.std()*ann if n.std()>0 else float('nan')
            return f"  {lab:6s}  SR_g={srg:+5.2f}  SR_n={srn:+5.2f}  ret_g={g.mean()*252*100:+6.1f}%  ret_n={n.mean()*252*100:+6.1f}%"
        print(f"\n=== {name} ===   avg daily cost: ${cost.mean()*BOOK:,.0f}  "
              f"({cost.mean()*1e4:.2f} bps/day → {cost.mean()*252*100:.2f}%/yr drag)")
        print(block(None, TRAIN_END, "TRAIN"))
        print(block(TRAIN_END, VAL_END, "VAL"))
        print(block(VAL_END, None, "TEST"))
        print(block(None, None, "FULL"))

    eq_curves = {}
    for name, w in portfolios.items():
        gross = (w * nx).sum(axis=1).fillna(0)
        cost_pct, comp = realistic_cost(w, close, BOOK)
        net = gross - cost_pct.reindex(gross.index).fillna(0)
        split_stats(name, gross, net, cost_pct)
        # Cost decomposition
        print("    avg daily cost decomposition (bps of book):")
        for k, v in comp.items():
            print(f"      {k:11s}  {v.mean()*1e4:>5.3f} bps/day  ({v.mean()*252*100:.2f}%/yr)")
        eq_curves[name] = (1 + net).cumprod()

    # ---- Plot the final equity curve --------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6.5))
    cmap = {"single #36": "C2", "equal_weight 6": "C0", "billions 6": "C3"}
    for name, eq in eq_curves.items():
        srg = ((eq.iloc[-1])**(252/len(eq)) - 1)
        # final SR for label
        net_pnl = eq.pct_change().dropna()
        sr = net_pnl.mean()/net_pnl.std()*ann
        ax.plot(eq.index, eq.values, color=cmap.get(name, "k"), lw=1.3,
                label=f"{name}  net SR={sr:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END),   color="grey", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 portfolio — REALISTIC fees @ ${BOOK:,.0f} book\n"
                 f"(per-share commission ${COMMISSION_PER_SHARE}, "
                 f"impact {IMPACT_BPS}bp, borrow {BORROW_BPS_ANNUAL}bp/yr on shorts)")
    ax.set_ylabel("Equity (log)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
