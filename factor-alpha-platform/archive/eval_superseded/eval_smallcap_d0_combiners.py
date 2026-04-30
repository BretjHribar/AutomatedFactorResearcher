"""
Try all 4 combiners on the SR>=5 SMALLCAP_D0 alpha set.

For each combiner:
  - Compute combined daily signal across full window 2016-2026
  - Split TRAIN (2016-2022) / VAL (2022-2024) / TEST (2024-2026)
  - Compute gross + net Sharpe @ $500K with realistic IB MOC fees:
      commission ~3.5 bps round-trip (taker MOC, $0.005/share on $15 stock)
      slippage at MOC ~0 (fills at official close)
      total ~3.5 bps round-trip cost on the $ traded
  - Print summary table

Saved alphas are pulled from the DB filtered to SR>=5 train.
"""
from __future__ import annotations
import sys, sqlite3, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.combiners import (
    combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions,
)

# ── Config ────────────────────────────────────────────────────────────────
UNIVERSE        = "TOP1500TOP2500"
DELAY           = 0
MAX_WEIGHT      = 0.001
NEUTRALIZE      = "subindustry"
BOOK            = 500_000.0
TRAIN_END       = "2022-01-01"
VAL_END         = "2024-01-01"

# IB MOC cost model:
#   IBKR Tiered MOC commission ~ $0.0035/share + exchange ~ $0.001 = $0.0045/share
#   Avg small-cap price ~ $15 (TOP1500-2500 universe)
#   → ~ 3.0 bps per share, one-way
#   Slippage at the MOC auction itself ≈ 0 (you get the official close).
#   Borrow on shorts ~ 30 bps/yr ÷ 252 = 0.12 bps/day ÷ shorts side
# Round-trip cost ≈ 6 bps. We charge as half-spread effective fee per dollar
# traded one-way (turnover already counts each direction once).
FEE_BPS_ONE_WAY = 3.0   # bps per dollar traded one-way

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
OUT_PNG  = ROOT / "data/smallcap_d0_combiners_equity.png"
OUT_JSON = ROOT / "data/smallcap_d0_combiners_summary.json"


def load_universe():
    df = pd.read_parquet(UNIV_DIR / f"{UNIVERSE}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.values.dtype != bool:
        df = df.astype(bool)
    coverage = df.sum(axis=0) / len(df)
    valid = sorted(coverage[coverage > 0.5].index.tolist())
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
        common_cols = [c for c in df.columns if c in tickers]
        if not common_cols:
            continue
        out[fp.stem] = df.loc[df.index.isin(dates), common_cols].reindex(index=dates, columns=tickers)
    return out


def stats(r, name, fee_bps_oneway=0.0, turnover_per_bar=None):
    if len(r) < 2:
        return {"name": name, "n": int(len(r))}
    ann = np.sqrt(252)
    # Net = gross - turnover * fee_bps. turnover_per_bar is fractional one-way per day.
    if turnover_per_bar is not None and fee_bps_oneway > 0:
        cost_per_bar = turnover_per_bar * (fee_bps_oneway / 1e4)
        rn = r - cost_per_bar.reindex(r.index).fillna(0)
    else:
        rn = r
    sr_g = r.mean()  / r.std()  * ann if r.std()  > 0 else float("nan")
    sr_n = rn.mean() / rn.std() * ann if rn.std() > 0 else float("nan")
    return {
        "name": name, "n": int(len(r)),
        "sharpe_gross_ann": float(sr_g),
        "sharpe_net_ann":   float(sr_n),
        "ret_gross_ann":    float(r.mean()  * 252),
        "ret_net_ann":      float(rn.mean() * 252),
        "vol_ann":          float(r.std() * ann),
        "max_dd": float(((1 + rn).cumprod() / (1 + rn).cumprod().cummax() - 1).min()),
        "hit_rate": float((rn > 0).mean()),
        "turnover_avg": float(turnover_per_bar.mean()) if turnover_per_bar is not None else 0.0,
        "start": str(r.index.min().date()),
        "end":   str(r.index.max().date()),
    }


def main():
    print(f"=== Loading universe {UNIVERSE} ===")
    uni_df = load_universe()
    print(f"  {uni_df.shape[1]} tickers x {uni_df.shape[0]} dates")
    dates = uni_df.index
    tickers = uni_df.columns.tolist()

    print("=== Loading matrices ===")
    matrices = load_matrices(tickers, dates)
    print(f"  {len(matrices)} matrices")

    close = matrices["close"]
    returns_df = close.pct_change(fill_method=None)

    # Pull SMALLCAP_D0 alphas with SR>=5 (use latest evaluation per alpha)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, a.expression, MAX(e.sharpe_is) AS sr
          FROM alphas a
          JOIN evaluations e ON e.alpha_id = a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         GROUP BY a.id
        HAVING sr >= 5.0
         ORDER BY sr DESC
    """)
    rows = cur.fetchall()
    print(f"\n=== {len(rows)} SMALLCAP_D0 alphas with SR>=5 ===")
    for aid, expr, sr in rows:
        print(f"  #{aid}  SR={sr:.2f}  {expr[:90]}")
    if not rows:
        print("Nothing to combine"); return 1

    # Evaluate each raw signal
    engine = FastExpressionEngine(data_fields=matrices)
    raw_signals = {}
    for aid, expr, sr in rows:
        try:
            raw_signals[aid] = engine.evaluate(expr)
            print(f"  #{aid} ok")
        except Exception as e:
            print(f"  #{aid} ERROR: {e}")

    if not raw_signals:
        return 1

    # Run all 4 combiners
    combiners = {
        "equal_weight": lambda: combiner_equal(raw_signals, matrices, uni_df, returns_df,
                                                max_wt=MAX_WEIGHT),
        "adaptive":     lambda: combiner_adaptive(raw_signals, matrices, uni_df, returns_df,
                                                   lookback=504, max_wt=MAX_WEIGHT),
        "risk_parity":  lambda: combiner_risk_parity(raw_signals, matrices, uni_df, returns_df,
                                                      lookback=504, max_wt=MAX_WEIGHT),
        "billions":     lambda: combiner_billions(raw_signals, matrices, uni_df, returns_df,
                                                    optim_lookback=60, max_wt=MAX_WEIGHT),
    }

    eq_curves = {}
    summary = {}
    for name, fn in combiners.items():
        print(f"\n=== Running combiner: {name} ===")
        try:
            sig = fn()
        except Exception as e:
            print(f"  ERROR in combiner: {e}")
            continue
        # Final L1-normalize + clip (same as kucoin_trader does)
        sig = sig.div(sig.abs().sum(axis=1).replace(0, np.nan), axis=0)
        sig = sig.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT).fillna(0.0)

        # delay=0: signal at T applies to T's close → captures T to T+1 close-to-close
        next_ret = returns_df.shift(-1)
        pnl_g = (sig * next_ret).sum(axis=1)
        # one-way turnover per bar (fractional GMV)
        to = sig.diff().abs().sum(axis=1) / 2.0
        # net pnl
        cost_per_bar = to * (FEE_BPS_ONE_WAY / 1e4)
        pnl_n = pnl_g - cost_per_bar
        pnl_g = pnl_g.dropna()
        pnl_n = pnl_n.reindex(pnl_g.index).fillna(0)

        # Splits
        s_train = stats(pnl_g.loc[:TRAIN_END], "TRAIN", FEE_BPS_ONE_WAY, to.loc[:TRAIN_END])
        s_val   = stats(pnl_g.loc[TRAIN_END:VAL_END], "VAL", FEE_BPS_ONE_WAY, to.loc[TRAIN_END:VAL_END])
        s_test  = stats(pnl_g.loc[VAL_END:],   "TEST",  FEE_BPS_ONE_WAY, to.loc[VAL_END:])
        s_full  = stats(pnl_g, "FULL", FEE_BPS_ONE_WAY, to)

        print(f"  {'split':6s}  {'n':>5s}  {'gross':>6s}  {'net':>6s}  {'gross%/y':>8s}  {'net%/y':>8s}  {'TO':>5s}  {'DD':>6s}  hit")
        for s in (s_train, s_val, s_test, s_full):
            print(f"  {s['name']:6s}  {s['n']:>5d}  {s['sharpe_gross_ann']:>+6.2f}  "
                  f"{s['sharpe_net_ann']:>+6.2f}  {s['ret_gross_ann']*100:>+7.1f}%  "
                  f"{s['ret_net_ann']*100:>+7.1f}%  {s['turnover_avg']*100:>4.1f}%  "
                  f"{s['max_dd']*100:>+5.1f}%  {s['hit_rate']*100:.0f}%")

        eq_curves[name] = (pnl_n.reindex(dates).fillna(0).cumsum() + 1).copy()
        summary[name] = {"train": s_train, "val": s_val, "test": s_test, "full": s_full}

    # Plot
    fig, ax = plt.subplots(figsize=(13, 7))
    colors = {"equal_weight": "C0", "adaptive": "C1", "risk_parity": "C2", "billions": "C3"}
    for name, eq in eq_curves.items():
        srn = summary[name]["full"]["sharpe_net_ann"]
        ax.plot(eq.index, eq.values, lw=1.2, color=colors.get(name, "k"),
                label=f"{name}  net SR={srn:+.2f}")
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END),   color="grey", ls="--", lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], dates[0], pd.Timestamp(TRAIN_END), color="grey", alpha=0.04)
    ax.fill_betweenx([ymin, ymax], pd.Timestamp(VAL_END), dates[-1], color="grey", alpha=0.04)
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 combiner comparison — net of {FEE_BPS_ONE_WAY} bps one-way "
                 f"({len(rows)} alphas, {UNIVERSE}, delay=0)")
    ax.set_ylabel("Cumulative return (1 + Σ net daily returns), log")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved: {OUT_PNG}")

    with open(OUT_JSON, "w") as f:
        json.dump({"config": dict(universe=UNIVERSE, delay=DELAY, max_weight=MAX_WEIGHT,
                                  fee_bps_oneway=FEE_BPS_ONE_WAY, book=BOOK,
                                  alphas=[a[0] for a in rows]),
                   "summary": summary}, f, indent=2)

    # Print best
    print("\n=== BEST NET SR ON FULL WINDOW ===")
    best = sorted(summary.items(), key=lambda x: -x[1]["full"]["sharpe_net_ann"])
    for name, s in best:
        print(f"  {name:14s}  net SR (full) = {s['full']['sharpe_net_ann']:+.2f}  "
              f"train={s['train']['sharpe_net_ann']:+.2f}  "
              f"val={s['val']['sharpe_net_ann']:+.2f}  "
              f"test={s['test']['sharpe_net_ann']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
