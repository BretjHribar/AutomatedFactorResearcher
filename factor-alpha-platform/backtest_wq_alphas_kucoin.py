"""
Re-run all WQ-style alphas from data/alphas.db (crypto, 4h, KUCOIN_TOP100)
on the KuCoin 4h research matrices and compare OOS performance to AIPT P=1000.

Each alpha is evaluated independently as a simple L/S portfolio:
  signal = FastExpressionEngine.evaluate(expression)
  w_t    = cross-sectionally demean, then divide by sum(|.|)    (sum|w| = 1)
  port_t = w_t (lagged 1 bar) · R_t        # same convention as AIPT
  net_t  = port_t - turnover_t * 3bps * 2

OOS window = 2024-09-01 -> end of data.
"""
from __future__ import annotations
import sqlite3, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.operators.fastexpression import FastExpressionEngine

UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
MATRICES_DIR = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH = PROJECT_ROOT / "data/alphas.db"
OOS_START = "2024-09-01"
COVERAGE_CUTOFF = 0.3
TAKER_BPS = 3.0
BARS_PER_YEAR = 6 * 365
RESULTS_DIR = PROJECT_ROOT / "data/aipt_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    # Restrict every matrix to the same ticker set
    for k, v in matrices.items():
        matrices[k] = v[[t for t in tickers if t in v.columns]]
    return matrices, tickers


def signal_to_portfolio(sig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean + L1-normalize to gross=1."""
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    w = demean.div(gross, axis=0).fillna(0)
    return w


def backtest_signal(w: pd.DataFrame, returns: pd.DataFrame, oos_start: pd.Timestamp,
                    taker_bps: float = 3.0):
    """w_t is the position at end of bar t (used for return over t->t+1).
    Convention matching AIPT: port_return at bar t+1 = w_t · R_{t+1}."""
    # Align
    common_idx = w.index.intersection(returns.index)
    common_cols = w.columns.intersection(returns.columns)
    w = w.loc[common_idx, common_cols].fillna(0)
    r = returns.loc[common_idx, common_cols].fillna(0)

    # Lag: holdings at bar t earn return over t->t+1 which lands at bar t+1.
    # port_t+1 = w_t · r_{t+1}   ⇒  use w.shift(1) * r (index r to its bar)
    port = (w.shift(1) * r).sum(axis=1)
    turnover = (w - w.shift(1)).abs().sum(axis=1)

    # Drop first bar (no prev weights)
    port = port.iloc[1:]
    turnover = turnover.iloc[1:]
    net = port - turnover * taker_bps / 10000.0

    # OOS slice
    mask = port.index >= oos_start
    return pd.DataFrame({
        "port": port[mask], "turnover": turnover[mask], "net": net[mask]
    })


def summarize(df):
    if df.empty or df["port"].std() == 0:
        return dict(bars=0, gross_sr=0, net_sr=0, gross_cum=0, net_cum=0, avg_to=0)
    ann = np.sqrt(BARS_PER_YEAR)
    g, n = df["port"].values, df["net"].values
    return dict(
        bars=len(df),
        gross_sr=float(g.mean() / (g.std(ddof=1) + 1e-12) * ann),
        net_sr=float(n.mean() / (n.std(ddof=1) + 1e-12) * ann),
        gross_cum=float(g.sum() * 100),
        net_cum=float(n.sum() * 100),
        avg_to=float(df["turnover"].mean()),
    )


def main():
    t0 = time.time()
    log("Loading matrices...")
    matrices, tickers = load_matrices()
    log(f"  {len(matrices)} matrices, {len(tickers)} tickers, {len(matrices['close'])} bars")
    log(f"  Date range: {matrices['close'].index.min()} -> {matrices['close'].index.max()}")

    engine = FastExpressionEngine(data_fields=matrices)
    returns = matrices["returns"]
    oos_start = pd.Timestamp(OOS_START)

    log("Loading alphas from DB...")
    con = sqlite3.connect(str(DB_PATH))
    rows = con.execute("""
        SELECT a.id, a.expression, a.source, a.universe,
               e.sharpe_is, e.sharpe_oos, e.turnover, e.ic_mean
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id
    """).fetchall()
    con.close()
    log(f"  {len(rows)} crypto 4h alphas")

    results = []
    port_series = {}
    net_series = {}
    for i, (aid, expr, src, uni, sr_is, sr_oos, to_is, ic_is) in enumerate(rows):
        t1 = time.time()
        try:
            sig = engine.evaluate(expr)
        except Exception as e:
            log(f"  [{i+1:>2}/{len(rows)}] alpha {aid}: EVAL FAIL {e}")
            continue

        if not isinstance(sig, pd.DataFrame) or sig.empty:
            log(f"  [{i+1:>2}/{len(rows)}] alpha {aid}: non-DataFrame output, skip")
            continue

        w = signal_to_portfolio(sig)
        df = backtest_signal(w, returns, oos_start, TAKER_BPS)
        s = summarize(df)
        s["alpha_id"] = aid
        s["source"] = src
        s["sr_is_db"] = sr_is
        s["to_is_db"] = to_is
        s["expr_short"] = (expr[:90] + "…") if len(expr) > 90 else expr
        results.append(s)
        port_series[aid] = df["port"]
        net_series[aid] = df["net"]
        log(f"  [{i+1:>2}/{len(rows)}] a{aid:>3d}  "
            f"SR_g={s['gross_sr']:+.2f}  SR_n={s['net_sr']:+.2f}  "
            f"cum_g={s['gross_cum']:+6.1f}%  cum_n={s['net_cum']:+6.1f}%  "
            f"TO={s['avg_to']*100:5.1f}% (DB IS SR={sr_is:.2f} TO={to_is:.2f})  ({time.time()-t1:.1f}s)")

    res_df = pd.DataFrame(results).sort_values("net_sr", ascending=False)

    log(f"\n{'-'*90}")
    log("Per-alpha OOS results (2024-09-01 -> end), sorted by net Sharpe:")
    log(f"{'-'*90}")
    cols_show = ["alpha_id","gross_sr","net_sr","gross_cum","net_cum","avg_to","sr_is_db","to_is_db"]
    print(res_df[cols_show].to_string(index=False))

    # Equal-weight ensemble (demean each signal, then average)
    log(f"\n{'-'*90}")
    log("Ensemble: equal-weight average of all 18 alpha portfolios")
    all_port = pd.concat(port_series.values(), axis=1).mean(axis=1)
    # Renormalize by scaling so that avg turnover is comparable. Actually:
    # Combining N equal-weight L/S portfolios (each gross=1/N after averaging) gives combined gross<=1.
    # Compute combined turnover from summed holdings.
    all_W = None
    for aid, s in port_series.items():
        sig = engine.evaluate([e for (i, e, *_) in rows if i == aid][0])
        w = signal_to_portfolio(sig)
        w = w.loc[w.index.intersection(returns.index)]
        if all_W is None:
            all_W = w.copy()
        else:
            all_W = all_W.add(w, fill_value=0)
    all_W = all_W / len(port_series)  # equal-weight average holdings
    ens_df = backtest_signal(all_W, returns, oos_start, TAKER_BPS)
    ens_s = summarize(ens_df)
    log(f"  gross_SR={ens_s['gross_sr']:+.2f}  net_SR={ens_s['net_sr']:+.2f}  "
        f"cum_gross={ens_s['gross_cum']:+.1f}%  cum_net={ens_s['net_cum']:+.1f}%  "
        f"avg_TO={ens_s['avg_to']*100:.1f}%")

    # IC-weighted ensemble (simple: weight by gross SR, non-negative)
    sr = res_df.set_index("alpha_id")["gross_sr"]
    sr = sr.clip(lower=0.0)  # non-negative
    sr_tot = sr.sum()
    icw_W = None
    if sr_tot > 0:
        for aid, gs in sr.items():
            sig = engine.evaluate([e for (i, e, *_) in rows if i == aid][0])
            w = signal_to_portfolio(sig)
            w = w.loc[w.index.intersection(returns.index)]
            scaled = w * (gs / sr_tot)
            if icw_W is None:
                icw_W = scaled.copy()
            else:
                icw_W = icw_W.add(scaled, fill_value=0)
        icw_df = backtest_signal(icw_W, returns, oos_start, TAKER_BPS)
        icw_s = summarize(icw_df)
        log(f"  SR-weighted ensemble: gross_SR={icw_s['gross_sr']:+.2f}  net_SR={icw_s['net_sr']:+.2f}  "
            f"cum_gross={icw_s['gross_cum']:+.1f}%  cum_net={icw_s['net_cum']:+.1f}%  "
            f"avg_TO={icw_s['avg_to']*100:.1f}%")

    res_df.to_csv(RESULTS_DIR / "wq_alphas_oos.csv", index=False)

    # ── Plot: cum net curves per alpha + ensemble ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for aid, s in net_series.items():
        axes[1].plot(s.index, s.cumsum() * 100, color="gray", alpha=0.35, linewidth=0.8)
    ens_cum_n = ens_df["net"].cumsum() * 100
    axes[1].plot(ens_df.index, ens_cum_n, color="tab:red", linewidth=2.2, label="Equal-weight ensemble")

    for aid, s in port_series.items():
        axes[0].plot(s.index, s.cumsum() * 100, color="gray", alpha=0.35, linewidth=0.8)
    ens_cum_g = ens_df["port"].cumsum() * 100
    axes[0].plot(ens_df.index, ens_cum_g, color="tab:red", linewidth=2.2, label="Equal-weight ensemble")

    for ax, title in zip(axes, ["Gross cum return (0 bps)", "Net cum return (3 bps taker)"]):
        ax.set_title(title); ax.set_ylabel("Cumulative return (%)"); ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5); ax.legend(loc="upper left")

    fig.suptitle(f"WQ-style KuCoin 4h alphas OOS (2024-09-01 ->): {len(port_series)} alphas + ensemble",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wq_alphas_equity.png", dpi=150, bbox_inches="tight")
    log(f"Figure: {RESULTS_DIR / 'wq_alphas_equity.png'}")
    log(f"CSV:    {RESULTS_DIR / 'wq_alphas_oos.csv'}")
    log(f"Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
