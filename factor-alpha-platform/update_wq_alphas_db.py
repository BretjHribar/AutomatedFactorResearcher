"""
Update data/alphas.db evaluations table with corrected train-set metrics,
after the 2026-04-24 OHLC column-swap data fix.

Writes over sharpe_is, turnover, ic_mean, ic_ir, fitness, max_drawdown,
return_total, return_ann, n_bars for every crypto/4h alpha. Does NOT touch
train_start/train_end/test_*/val_* fields. Also appends a note flagging the
re-evaluation date in the alphas.notes column.
"""
from __future__ import annotations
import sqlite3, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.fastexpression import FastExpressionEngine

UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
MATRICES_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH = PROJECT_ROOT / "data/alphas.db"
COVERAGE_CUTOFF = 0.3
BARS_PER_YEAR = 6 * 365


def load_matrices():
    uni = pd.read_parquet(UNIVERSE_PATH)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    tickers = sorted(set(matrices["close"].columns))
    for k, v in matrices.items():
        matrices[k] = v[[t for t in tickers if t in v.columns]]
    return matrices


def signal_to_portfolio(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


def compute_stats(w, returns, start, end):
    common_idx = w.index.intersection(returns.index)
    w = w.loc[common_idx].fillna(0)
    r = returns.loc[common_idx].fillna(0)
    port = (w.shift(1) * r).sum(axis=1)
    to   = (w - w.shift(1)).abs().sum(axis=1)
    m = (port.index >= start) & (port.index <= end)
    port = port[m].dropna(); to = to.loc[port.index]

    # IC: Spearman between w_t and R_{t+1}
    w_w = w.loc[port.index]
    r_next = r.shift(-1).loc[port.index]
    ic_vals = []
    for ts in w_w.index:
        wi = w_w.loc[ts]; ri = r_next.loc[ts]
        mask = wi.notna() & ri.notna() & (wi.abs() > 1e-12)
        if mask.sum() < 10:
            continue
        try:
            ic_vals.append(wi[mask].rank().corr(ri[mask].rank()))
        except Exception:
            pass
    ic_mean = float(np.mean(ic_vals)) if ic_vals else float("nan")
    ic_std  = float(np.std(ic_vals))  if ic_vals else float("nan")
    ic_ir   = float(ic_mean / ic_std * np.sqrt(BARS_PER_YEAR)) if ic_std > 0 else float("nan")

    ann = np.sqrt(BARS_PER_YEAR)
    sharpe = float(port.mean() / (port.std(ddof=1) + 1e-12) * ann)

    cum = port.cumsum()
    mdd = float((cum - cum.cummax()).min())

    return {
        "n_bars": int(len(port)),
        "sharpe_is": sharpe,
        "fitness": sharpe,          # same convention as the DB (where fitness≈SR)
        "turnover": float(to.mean()),
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "max_drawdown": mdd,
        "return_total": float(port.sum()),
        "return_ann": float(port.mean() * BARS_PER_YEAR),
    }


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Loading matrices...", flush=True)
    matrices = load_matrices()
    engine = FastExpressionEngine(data_fields=matrices)
    returns = matrices["returns"]

    con = sqlite3.connect(str(DB_PATH))
    alphas = con.execute("""
        SELECT a.id, a.expression, e.id AS eval_id, e.train_start, e.train_end, e.sharpe_is
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id
    """).fetchall()

    print(f"[{time.strftime('%H:%M:%S')}] Updating {len(alphas)} evaluations...", flush=True)
    cur = con.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    for (aid, expr, eval_id, tstart, tend, old_sr) in alphas:
        try:
            sig = engine.evaluate(expr)
        except Exception as e:
            print(f"  alpha {aid}: EVAL FAIL {e}", flush=True)
            continue
        w = signal_to_portfolio(sig)
        stats = compute_stats(w, returns, pd.Timestamp(tstart), pd.Timestamp(tend))

        # Update evaluations row
        cur.execute("""
            UPDATE evaluations
            SET sharpe_is = ?, turnover = ?, ic_mean = ?, ic_ir = ?, fitness = ?,
                max_drawdown = ?, return_total = ?, return_ann = ?, n_bars = ?,
                evaluated_at = ?
            WHERE id = ?
        """, (
            stats["sharpe_is"], stats["turnover"], stats["ic_mean"], stats["ic_ir"],
            stats["fitness"], stats["max_drawdown"], stats["return_total"],
            stats["return_ann"], stats["n_bars"], now, eval_id,
        ))

        # Also tag the alpha with a re-evaluation note (append, don't overwrite)
        row = cur.execute("SELECT notes FROM alphas WHERE id=?", (aid,)).fetchone()
        existing = (row[0] or "").strip()
        tag = f"[re-eval 2026-04-24 post-OHLC-fix: SR_is {old_sr:.2f}->{stats['sharpe_is']:.2f}]"
        new_notes = (existing + " | " + tag).strip(" |") if existing else tag
        cur.execute("UPDATE alphas SET notes=? WHERE id=?", (new_notes, aid))

        print(f"  a{aid:>2}  SR: {old_sr:>+.2f} -> {stats['sharpe_is']:>+.2f}   "
              f"TO: {stats['turnover']:>.3f}   IC: {stats['ic_mean']:>+.4f}   "
              f"MDD: {stats['max_drawdown']:>+.3f}   RA: {stats['return_ann']*100:>+.1f}%",
              flush=True)

    con.commit()
    con.close()
    print(f"[{time.strftime('%H:%M:%S')}] DB updated. Total {time.time()-t0:.1f}s", flush=True)

    # Print top-5 updated by SR, worst-5 by SR for quick inspection
    con = sqlite3.connect(str(DB_PATH))
    top = con.execute("""
        SELECT a.id, a.name, e.sharpe_is, e.turnover, e.ic_mean, e.max_drawdown, e.return_ann, e.n_bars
        FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.asset_class='crypto' AND a.interval='4h' AND a.archived=0
        ORDER BY e.sharpe_is DESC LIMIT 5
    """).fetchall()
    print("\nTop 5 by updated IS Sharpe:")
    print(f"  {'id':>3}  {'SR':>6}  {'TO':>5}  {'IC':>+7}  {'MDD':>+6}  {'RA%':>+6}  name")
    for r in top:
        print(f"  {r[0]:>3}  {r[2]:>+6.2f}  {r[3]:>5.3f}  {r[4]:>+7.4f}  {r[5]:>+6.3f}  {r[6]*100:>+6.1f}  {r[1][:40]}")

    bot = con.execute("""
        SELECT a.id, a.name, e.sharpe_is, e.turnover, e.ic_mean, e.max_drawdown, e.return_ann
        FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.asset_class='crypto' AND a.interval='4h' AND a.archived=0
        ORDER BY e.sharpe_is ASC LIMIT 5
    """).fetchall()
    print("\nBottom 5 by updated IS Sharpe:")
    for r in bot:
        print(f"  {r[0]:>3}  {r[2]:>+6.2f}  {r[3]:>5.3f}  {r[4]:>+7.4f}  {r[5]:>+6.3f}  {r[6]*100:>+6.1f}  {r[1][:40]}")
    con.close()


if __name__ == "__main__":
    main()
