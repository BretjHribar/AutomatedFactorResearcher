"""
Parametric portfolio harness — sweep config knobs to maximize NET Sharpe at $500K.

Knobs:
  --universe NAME           e.g. TOP1500TOP2500, TOP500TOP1000, MCAP_500M_2B
  --max-weight F            per-name cap
  --decay N                 linear-decay smoothing on the combined signal
  --neutralize subindustry  or 'riskmodel' (full Barra-style residualize)
  --combiner equal|billions
  --no-trade-band F         drop trades smaller than F*MAX_W (skip tiny rebalances)
  --train-start / --train-end / --val-end / --test-end

Always reports:
  - TRAIN/VAL/TEST gross + NET SR @ $500K with the realistic per-share fee model
  - daily turnover (one-way)
  - cost decomposition (commission / sec / impact / borrow)
"""
from __future__ import annotations
import argparse, sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

# ── per-share cost constants ──────────────────────────────────────────────
COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35     # $0.35 minimum per order (Tiered)
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"


def load_universe(name):
    df = pd.read_parquet(UNIV_DIR / f"{name}.parquet").astype(bool)
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


def proc(sig, uni, group_labels=None):
    s = sig.astype(float).where(uni, np.nan)
    if group_labels is not None:
        for g in group_labels.dropna().unique():
            m = (group_labels == g).values
            if m.any():
                sub = s.iloc[:, m]
                s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    return s


def normalize_clip(sig, max_w):
    abs_sum = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(abs_sum, axis=0).clip(-max_w, max_w).fillna(0)


def realistic_cost_per_share(combined_w, close, book):
    """Per-share commission with $0.35/order minimum, plus other fees."""
    pos = combined_w * book
    trd = pos.diff().abs()
    safe = close.where(close > 0)
    shares = trd / safe
    # per-name commission = max(shares * cps, $0.35) when there's a trade
    per_name_comm = (shares * COMMISSION_PER_SHARE).clip(lower=0)
    has_trade = trd > 1.0  # only count trades > $1
    per_name_comm = per_name_comm.where(~has_trade, np.maximum(per_name_comm, PER_ORDER_MIN))
    per_name_comm = per_name_comm.where(has_trade, 0)
    commission_d = per_name_comm.sum(axis=1)
    sec_d        = (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
    impact_d     = (trd * IMPACT_BPS / 1e4).sum(axis=1)
    short_d      = (-pos.clip(upper=0)).sum(axis=1)
    borrow_d     = short_d * (BORROW_BPS_ANNUAL / 1e4) / 252.0
    daily_pct = (commission_d + sec_d + impact_d + borrow_d) / book
    decomp = dict(commission=commission_d/book, sec=sec_d/book, impact=impact_d/book, borrow=borrow_d/book)
    return daily_pct, decomp


def apply_no_trade_band(combined, max_w, band_frac):
    """Walk forward: only update position when |target - current| > band_frac * max_w.
    band_frac=0 disables. band_frac=0.2 means tolerate 20% of max_w drift per name."""
    if band_frac <= 0:
        return combined
    threshold = band_frac * max_w
    out = combined.copy()
    prev = pd.Series(0.0, index=combined.columns)
    for i, dt in enumerate(combined.index):
        target = combined.iloc[i]
        diff = target - prev
        # snap to previous where |diff| < threshold
        small = diff.abs() < threshold
        new = target.where(~small, prev)
        out.iloc[i] = new
        prev = new
    return out


def apply_decay(sig, decay_days):
    """Linear-decay smoothing: out[t] = sum_{k=0..N-1} (N-k)/N * sig[t-k] / sum_w."""
    if decay_days <= 0:
        return sig
    weights = np.arange(decay_days, 0, -1, dtype=float)
    weights /= weights.sum()
    return sig.rolling(decay_days, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]), raw=True
    )


def evaluate(name, w, close, ret, book, train_end, val_end):
    nx = ret.shift(-1)
    g = (w * nx).sum(axis=1).fillna(0)
    cost, comp = realistic_cost_per_share(w, close, book)
    n = g - cost.reindex(g.index).fillna(0)
    ann = np.sqrt(252)
    to_oneway = w.diff().abs().sum(axis=1) / 2

    def _sr(s): return s.mean()/s.std()*ann if s.std()>0 else float("nan")
    def stats(g_, n_):
        return f"SR_g={_sr(g_):+5.2f} SR_n={_sr(n_):+5.2f} ret_g={g_.mean()*252*100:+5.1f}% ret_n={n_.mean()*252*100:+5.1f}%"

    train_g = g.loc[:train_end]; train_n = n.loc[:train_end]
    val_g = g.loc[train_end:val_end]; val_n = n.loc[train_end:val_end]
    test_g = g.loc[val_end:]; test_n = n.loc[val_end:]
    print(f"  --- {name} ---  cost={cost.mean()*1e4:.2f}bps/d ({cost.mean()*252*100:.2f}%/yr), "
          f"TO={to_oneway.mean()*100:.1f}%/d, n_active={(w.iloc[-100:].abs()>0).sum(axis=1).mean():.0f}")
    print(f"    TRAIN  {stats(train_g, train_n)}")
    print(f"    VAL    {stats(val_g, val_n)}")
    print(f"    TEST   {stats(test_g, test_n)}")
    print(f"    FULL   {stats(g, n)}")
    return _sr(g.loc[:train_end]), _sr(n.loc[:train_end]), _sr(g), _sr(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="TOP1500TOP2500")
    ap.add_argument("--max-weight", type=float, default=0.001)
    ap.add_argument("--decay", type=int, default=0)
    ap.add_argument("--no-trade-band", type=float, default=0.0,
                    help="Threshold (×max_weight) below which we skip the rebalance.")
    ap.add_argument("--book", type=float, default=500_000)
    ap.add_argument("--train-end", default="2024-01-01")
    ap.add_argument("--val-end", default="2025-04-01")
    ap.add_argument("--alpha-tag", default="SMALLCAP_D0_v2")
    ap.add_argument("--name", default="run", help="Run label for printing")
    args = ap.parse_args()

    print(f"\n========== SWEEP RUN: {args.name} ==========")
    print(f"  universe={args.universe} max_w={args.max_weight} decay={args.decay} "
          f"band={args.no_trade_band} book=${args.book:,.0f}")

    uni = load_universe(args.universe)
    dates = uni.index; tickers = uni.columns.tolist()
    print(f"  {len(tickers)} tickers, {len(dates)} dates")
    mats = load_matrices(tickers, dates)
    close = mats["close"]; ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)

    conn = sqlite3.connect(DB)
    rows = conn.execute(f"""
        SELECT a.id, a.expression, MAX(e.sharpe_is) FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
         WHERE a.archived=0 AND a.notes LIKE '%{args.alpha_tag}%'
         GROUP BY a.id ORDER BY MAX(e.sharpe_is) DESC""").fetchall()
    print(f"  {len(rows)} alphas:  ", "  ".join(f"#{aid}/SR{sr:.1f}" for aid, _, sr in rows))

    # Build normed signals
    raw = {aid: engine.evaluate(expr) for aid, expr, _ in rows}
    normed = {aid: normalize_clip(proc(r, uni, cls), args.max_weight) for aid, r in raw.items()}

    # Equal-weight combine
    combined = sum(normed.values()) / len(normed)
    combined = normalize_clip(combined, args.max_weight)

    # Apply decay smoothing if requested
    if args.decay > 0:
        combined = apply_decay(combined, args.decay)
        combined = normalize_clip(combined, args.max_weight)

    # No-trade band
    if args.no_trade_band > 0:
        combined = apply_no_trade_band(combined, args.max_weight, args.no_trade_band)

    sr = evaluate("equal-weight", combined, close, ret, args.book, args.train_end, args.val_end)
    return sr


if __name__ == "__main__":
    main()
