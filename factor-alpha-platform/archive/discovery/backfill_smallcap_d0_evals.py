"""
Backfill missing evaluation metrics (return_ann, max_drawdown, ic_mean, ic_ir,
n_bars, psr) for SMALLCAP_D0 alphas saved with zero placeholders.

All metrics computed on TRAIN window (2020-01-01 to 2024-01-01) per the
no-VAL/TEST-peeking rule.

Updates rows in-place where the existing field is 0 / NULL.
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

UNIV_NAME   = "MCAP_100M_500M"
MAX_W       = 0.02
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-01-01"
BOOK        = 500_000

COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"


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
    print(f"=== Loading universe + matrices ({UNIV_NAME}) ===")
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

    train_mask = (dates >= TRAIN_START) & (dates < TRAIN_END)
    nx = ret.shift(-1)
    ann = np.sqrt(252)

    conn = sqlite3.connect(DB)
    rows = conn.execute("""
        SELECT a.id, a.expression, e.id as eval_id,
               e.return_ann, e.max_drawdown, e.ic_mean, e.ic_ir, e.n_bars, e.psr
          FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         ORDER BY a.id
    """).fetchall()

    print(f"\n=== Found {len(rows)} SMALLCAP_D0 alpha rows ===\n")

    needs_update = [r for r in rows if (r[3] in (0, 0.0, None) or r[4] in (0, 0.0, None)
                                         or r[5] in (0, 0.0, None) or r[7] in (0, None))]
    print(f"  {len(needs_update)} need backfill (have zero/NULL in return/dd/ic/n_bars)")

    for aid, expr, eval_id, ret_ann, mdd, ic, ic_ir, n_bars, psr in needs_update:
        try:
            raw = engine.evaluate(expr)
        except Exception as e:
            print(f"  #{aid} ERR: {e}"); continue

        sig = proc_signal(raw, uni, cls)
        sig_tr = sig.loc[train_mask]
        nx_tr  = nx.loc[train_mask]
        close_tr = close.loc[train_mask]

        g = (sig_tr * nx_tr).sum(axis=1).fillna(0)
        cost = realistic_cost(sig_tr, close_tr, BOOK)
        n = g - cost

        ret_ann_calc = float(g.mean() * 252)
        sr_train = float(g.mean()/g.std()*ann) if g.std() > 0 else 0.0

        eq = (1 + n).cumprod()
        running_max = eq.cummax()
        dd = (eq / running_max - 1.0)
        mdd_calc = float(dd.min())

        # Cross-sectional IC daily, lagged
        lagged = sig_tr.shift(1)
        ic_daily = lagged.corrwith(nx_tr, axis=1)
        ic_mean_calc = float(ic_daily.mean())
        ic_ir_calc = float(ic_daily.mean()/ic_daily.std()*np.sqrt(252)) if ic_daily.std() > 0 else 0.0

        n_bars_calc = int(train_mask.sum())

        # Probabilistic SR (Bailey-López de Prado): probability that true SR > 0
        # PSR = N( (SR - 0)/std(SR) ) — std(SR) ≈ sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))
        try:
            from scipy.stats import norm, skew, kurtosis
            sk = float(skew(g.dropna()))
            kt = float(kurtosis(g.dropna(), fisher=True))   # excess kurtosis
            T  = n_bars_calc
            sr_d = sr_train / np.sqrt(252)   # daily SR
            std_sr = np.sqrt((1 - sk*sr_d + (kt)/4.0 * sr_d**2) / max(T-1, 1))
            psr_calc = float(norm.cdf(sr_d / std_sr)) if std_sr > 0 else 0.5
        except Exception:
            psr_calc = 0.0

        conn.execute("""
            UPDATE evaluations
               SET return_ann = ?, max_drawdown = ?, ic_mean = ?, ic_ir = ?,
                   n_bars = ?, psr = ?
             WHERE id = ?
        """, (ret_ann_calc, mdd_calc, ic_mean_calc, ic_ir_calc, n_bars_calc, psr_calc, eval_id))

        print(f"  #{aid}  ret_ann={ret_ann_calc*100:+5.1f}%  mdd={mdd_calc*100:+5.1f}%  "
              f"ic={ic_mean_calc:+.4f}  ic_ir={ic_ir_calc:+.2f}  n_bars={n_bars_calc}  psr={psr_calc:.3f}")

    conn.commit()
    conn.close()
    print(f"\nBackfilled {len(needs_update)} rows.")


if __name__ == "__main__":
    main()
