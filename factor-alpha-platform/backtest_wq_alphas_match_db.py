"""
Compare my WQ alpha recompute to what's stored in data/alphas.db evaluations table.
Runs each alpha on its exact IS window (2023-09-01 → 2025-09-01) and reports
Sharpe, turnover, IC, drawdown, return — side by side with DB values.
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


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return matrices, tickers


def signal_to_portfolio(sig: pd.DataFrame) -> pd.DataFrame:
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


def compute_stats(w: pd.DataFrame, returns: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp,
                  fee_bps: float = 0.0):
    """Return dict of Sharpe, turnover, IC (next-bar Spearman), max DD, return_ann, n_bars
    over the [start, end] window, matching the lag/convention used by most eval scripts:
      port_t = w_{t-1} · R_t
    """
    common_idx = w.index.intersection(returns.index)
    w = w.loc[common_idx].fillna(0)
    r = returns.loc[common_idx].fillna(0)

    port = (w.shift(1) * r).sum(axis=1)
    to = (w - w.shift(1)).abs().sum(axis=1)

    # Slice window
    m = (port.index >= start) & (port.index <= end)
    port = port[m]; to = to[m]
    # Drop first bar (NaN from shift)
    port = port.dropna(); to = to.loc[port.index]

    # IC = cross-sectional Spearman rank correlation between w_t and R_{t+1}
    ic_vals = []
    w_w = w.loc[port.index]
    r_next = r.shift(-1).loc[port.index]  # R_{t+1} aligned to t (where the signal was formed)
    for ts in w_w.index:
        wi = w_w.loc[ts]
        ri = r_next.loc[ts]
        mask = wi.notna() & ri.notna() & (wi.abs() > 1e-12)
        if mask.sum() < 10:
            continue
        try:
            ic_vals.append(wi[mask].rank().corr(ri[mask].rank()))
        except Exception:
            pass
    ic_mean = float(np.mean(ic_vals)) if ic_vals else float("nan")
    ic_std = float(np.std(ic_vals)) if ic_vals else float("nan")
    ic_ir = ic_mean / ic_std * np.sqrt(BARS_PER_YEAR) if ic_std > 0 else float("nan")

    net = port - to * fee_bps / 10000.0
    ann = np.sqrt(BARS_PER_YEAR)

    cum = port.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max

    return {
        "n_bars": int(len(port)),
        "sharpe_gross": float(port.mean() / (port.std(ddof=1) + 1e-12) * ann),
        "sharpe_net":   float(net.mean()  / (net.std(ddof=1)  + 1e-12) * ann),
        "turnover":     float(to.mean()),
        "ic_mean":      ic_mean,
        "ic_ir":        ic_ir,
        "max_drawdown": float(dd.min()),
        "return_total": float(port.sum()),
        "return_ann":   float(port.mean() * BARS_PER_YEAR),
        "fitness":      float(port.mean() / (port.std(ddof=1) + 1e-12) * ann),  # proxy
    }


def main():
    t0 = time.time()
    log("Loading matrices...")
    matrices, tickers = load_matrices()
    log(f"  {len(matrices)} matrices, {len(tickers)} tickers, bars={len(matrices['close'])}")

    engine = FastExpressionEngine(data_fields=matrices)
    returns = matrices["returns"]

    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute("""
        SELECT a.id, a.expression,
               e.sharpe_is, e.turnover, e.ic_mean, e.ic_ir, e.fitness,
               e.max_drawdown, e.return_total, e.return_ann, e.train_start, e.train_end, e.n_bars
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id
    """).fetchall()
    con.close()

    print()
    hdr = f"{'id':>3}  | {'DB_SR':>6} {'my_SR':>6} {'dSR':>5}  | {'DB_TO':>5} {'my_TO':>5} {'dTO':>5}  | "\
          f"{'DB_IC':>7} {'my_IC':>7}  | {'DB_DD':>6} {'my_DD':>6}  | {'DB_RA%':>6} {'my_RA%':>6}  | n"
    print(hdr); print("-" * len(hdr))

    rows_out = []
    for (aid, expr, sr_db, to_db, ic_db, icir_db, fit_db, mdd_db, rt_db, ra_db, tstart, tend, nb_db) in rows:
        try:
            sig = engine.evaluate(expr)
        except Exception as e:
            print(f"{aid:>3}  FAIL: {e}")
            continue
        w = signal_to_portfolio(sig)
        stats = compute_stats(w, returns,
                              pd.Timestamp(tstart), pd.Timestamp(tend),
                              fee_bps=0.0)
        stats["alpha_id"] = aid
        stats["sr_db"] = sr_db; stats["to_db"] = to_db; stats["ic_db"] = ic_db
        stats["mdd_db"] = mdd_db; stats["ra_db"] = ra_db; stats["n_db"] = nb_db
        rows_out.append(stats)

        d_sr = stats["sharpe_gross"] - (sr_db or 0)
        d_to = stats["turnover"] - (to_db or 0)
        print(
            f"{aid:>3}  | {sr_db or 0:>6.2f} {stats['sharpe_gross']:>+6.2f} {d_sr:>+5.2f}  | "
            f"{to_db or 0:>5.3f} {stats['turnover']:>5.3f} {d_to:>+5.3f}  | "
            f"{ic_db if ic_db is not None else 0:>+7.4f} {stats['ic_mean']:>+7.4f}  | "
            f"{mdd_db or 0:>+6.3f} {stats['max_drawdown']:>+6.3f}  | "
            f"{(ra_db or 0)*100:>+6.1f} {stats['return_ann']*100:>+6.1f}  | "
            f"{stats['n_bars']}"
        )

    df = pd.DataFrame(rows_out)
    df["d_sr"] = df["sharpe_gross"] - df["sr_db"]
    df["d_to"] = df["turnover"]     - df["to_db"]
    df["d_ic"] = df["ic_mean"]      - df["ic_db"]
    df["d_mdd"] = df["max_drawdown"] - df["mdd_db"]
    df["d_ra"]  = df["return_ann"]   - df["ra_db"]

    print("\nSummary of discrepancies (my - DB):")
    print(f"  mean |d_SR|:  {df['d_sr'].abs().mean():.2f}  max |d_SR|:  {df['d_sr'].abs().max():.2f}")
    print(f"  mean |d_TO|:  {df['d_to'].abs().mean():.3f}  max |d_TO|:  {df['d_to'].abs().max():.3f}")
    print(f"  mean |d_IC|:  {df['d_ic'].abs().mean():.4f}  max |d_IC|:  {df['d_ic'].abs().max():.4f}")
    print(f"  mean |d_MDD|: {df['d_mdd'].abs().mean():.3f}  max |d_MDD|: {df['d_mdd'].abs().max():.3f}")
    print(f"  mean |d_RA|:  {df['d_ra'].abs().mean():.3f}   max |d_RA|:  {df['d_ra'].abs().max():.3f}")

    df.to_csv(PROJECT_ROOT / "data/aipt_results/wq_alphas_db_match.csv", index=False)
    print(f"\nCSV saved to data/aipt_results/wq_alphas_db_match.csv")
    print(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
