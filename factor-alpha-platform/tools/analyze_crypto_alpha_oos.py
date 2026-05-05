"""
Investigate why KuCoin alphas don't generalize OOS.

Phase 1 — pipeline sanity check (data, universe, returns, one known alpha)
Phase 2 — per-alpha TRAIN / VAL / TEST gross SR for all 190 alphas
Phase 3 — filter sweep: keep alphas that maintain VAL/TRAIN ratio, run combiners
          on each surviving subset, report V+T net SR per (filter × combiner) cell
Phase 4 — best survivors → all-combiners → equity curve PNG

Conventions match the unified pipeline:
  delay=0 (weights × ret.shift(-1))
  gross-normalized to Σ|w|=1
  full-turnover (no /2)
  3 bps (taker+slippage) for net SR; gross SR used for alpha selection
"""
from __future__ import annotations
import sys, json, time, copy, sqlite3, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.pipeline.runner import run

CONFIG_PATH    = ROOT / "prod" / "config" / "research_crypto.json"
UNIVERSE_PATH  = ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
MATRICES_DIR   = ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH        = ROOT / "data/alphas.db"
COVERAGE_CUTOFF = 0.3
BARS_PER_YEAR  = 6 * 365
COST_BPS       = 3.0

# ──────────────────────────────────────────────────────────────────
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def signal_to_portfolio(sig):
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


def gross_sr(w, returns, start, end):
    """delay=0 gross SR. Returns (sr, n_bars, mean_per_bar)."""
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    pnl = (w_a * r_a.shift(-1)).sum(axis=1)
    m = (pnl.index >= start) & (pnl.index <= end)
    p = pnl[m].dropna()
    if len(p) < 50 or p.std(ddof=1) <= 0:
        return float("nan"), len(p), float("nan")
    sr = float(p.mean() / (p.std(ddof=1) + 1e-12) * np.sqrt(BARS_PER_YEAR))
    return sr, len(p), float(p.mean())


# ──────────────────────────────────────────────────────────────────
# Phase 1 — pipeline sanity
# ──────────────────────────────────────────────────────────────────
def phase1_sanity():
    log("PHASE 1 — pipeline sanity check")
    cfg = json.loads(CONFIG_PATH.read_text())
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end   = pd.Timestamp(cfg["splits"]["val_end"])

    uni = pd.read_parquet(UNIVERSE_PATH)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    print(f"  universe rows:   {len(uni):,}")
    print(f"  universe cols:   {len(uni.columns):,}")
    print(f"  cov>{COVERAGE_CUTOFF} valid: {len(valid):,}")

    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        if fp.parent.name == "prod": continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    tickers = sorted(set(matrices["close"].columns))
    for k, v in matrices.items():
        matrices[k] = v[[t for t in tickers if t in v.columns]]

    print(f"  matrix fields:   {len(matrices):,}")
    print(f"  matrix tickers:  {len(tickers):,}")
    print(f"  bars total:      {len(matrices['close']):,}")
    print(f"  bars span:       {matrices['close'].index.min()} -> {matrices['close'].index.max()}")

    rets = matrices["returns"]
    print(f"  returns:  mean per-cell = {float(rets.mean().mean()):+.5f}  "
          f"std per-cell = {float(rets.std().mean()):.5f}  "
          f"(expect mean~0, std~0.02-0.04 for 4h)")
    print(f"  bars in TRAIN   ({rets.index.min()} -> {train_end}): "
          f"{int(((rets.index >= rets.index.min()) & (rets.index < train_end)).sum())}")
    print(f"  bars in VAL     ({train_end} -> {val_end}): "
          f"{int(((rets.index >= train_end) & (rets.index < val_end)).sum())}")
    print(f"  bars in TEST    ({val_end} -> {rets.index.max()}): "
          f"{int(((rets.index >= val_end)).sum())}")

    # Reproduce a known alpha from DB (id=92 has DB sharpe_train +4.19)
    con = sqlite3.connect(str(DB_PATH))
    row = con.execute("SELECT expression, sharpe_train FROM alphas a JOIN evaluations e ON e.alpha_id=a.id WHERE a.id=92").fetchone()
    expr, db_sr = row
    engine = FastExpressionEngine(data_fields=matrices)
    sig = engine.evaluate(expr)
    w = signal_to_portfolio(sig)
    sr_t, n_t, _ = gross_sr(w, matrices["returns"], pd.Timestamp("2023-09-01"), train_end)
    print(f"  alpha #92 reproduce TRAIN SR (delay=0):  {sr_t:+.2f}   "
          f"(DB stored: {db_sr:+.2f},  delta={sr_t-db_sr:+.2f})")
    print()
    return matrices, tickers, train_end, val_end, cfg


# ──────────────────────────────────────────────────────────────────
# Phase 2 — per-alpha TRAIN / VAL / TEST
# ──────────────────────────────────────────────────────────────────
def phase2_per_alpha(matrices, train_end, val_end):
    log("PHASE 2 — per-alpha TRAIN / VAL / TEST gross SR (190 alphas)")
    rets = matrices["returns"]
    train_start = pd.Timestamp("2023-09-01")
    test_end    = rets.index.max()

    con = sqlite3.connect(str(DB_PATH))
    alphas = con.execute("""
        SELECT a.id, a.expression, e.sharpe_train
        FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id""").fetchall()
    print(f"  evaluating {len(alphas)} alphas...", flush=True)

    engine = FastExpressionEngine(data_fields=matrices)
    rows = []
    weights_by_id = {}
    t0 = time.time()
    for i, (aid, expr, db_sr_train) in enumerate(alphas, 1):
        try:
            sig = engine.evaluate(expr)
            w = signal_to_portfolio(sig)
            tr_sr, tr_n, _ = gross_sr(w, rets, train_start, train_end)
            va_sr, va_n, _ = gross_sr(w, rets, train_end,   val_end)
            te_sr, te_n, _ = gross_sr(w, rets, val_end,     test_end)
            vt_sr, vt_n, _ = gross_sr(w, rets, train_end,   test_end)
            # turnover (full conv) on TRAIN as a feature for selection
            to = float((w - w.shift(1)).abs().sum(axis=1).loc[train_start:train_end].mean())
            rows.append({"id": aid, "expr": expr,
                         "TRAIN_SR": tr_sr, "VAL_SR": va_sr, "TEST_SR": te_sr, "VT_SR": vt_sr,
                         "TO_train": to, "n_train": tr_n, "n_val": va_n, "n_test": te_n,
                         "DB_SR_train": db_sr_train})
            weights_by_id[aid] = w
        except Exception as e:
            print(f"  a{aid}  FAIL: {type(e).__name__}: {str(e)[:60]}", flush=True)
        if i % 30 == 0:
            print(f"    {i}/{len(alphas)}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values("VAL_SR", ascending=False).reset_index(drop=True)
    print(f"  done {time.time()-t0:.0f}s")
    print()

    print("Per-alpha summary (n=%d):" % len(df))
    for col in ["TRAIN_SR", "VAL_SR", "TEST_SR", "VT_SR"]:
        s = df[col]
        print(f"  {col:8s}  mean {s.mean():+5.2f}  median {s.median():+5.2f}  "
              f"std {s.std():4.2f}  >0: {(s>0).sum():3d}/{len(s)}  "
              f">+1: {(s>1).sum():3d}/{len(s)}")
    rho_tv = df["TRAIN_SR"].corr(df["VAL_SR"], method="spearman")
    rho_tt = df["TRAIN_SR"].corr(df["TEST_SR"], method="spearman")
    rho_vt = df["VAL_SR"].corr(df["TEST_SR"], method="spearman")
    print()
    print(f"Per-alpha rank correlations (do alphas that did well in one split do well in another?):")
    print(f"  rho(TRAIN, VAL)  = {rho_tv:+.3f}   {'(weak)' if abs(rho_tv)<0.2 else '(meaningful)'}")
    print(f"  rho(TRAIN, TEST) = {rho_tt:+.3f}   {'(weak)' if abs(rho_tt)<0.2 else '(meaningful)'}")
    print(f"  rho(VAL,   TEST) = {rho_vt:+.3f}   {'(weak)' if abs(rho_vt)<0.2 else '(meaningful)'}")
    print()
    return df, weights_by_id


# ──────────────────────────────────────────────────────────────────
# Phase 3 — filter sweep
# ──────────────────────────────────────────────────────────────────
def combine_equal(weights_by_id, ids):
    if not ids:
        return None
    ws = [weights_by_id[i] for i in ids]
    common_idx = ws[0].index
    common_cols = ws[0].columns
    for w in ws[1:]:
        common_idx = common_idx.intersection(w.index)
        common_cols = common_cols.intersection(w.columns)
    aligned = [w.loc[common_idx, common_cols] for w in ws]
    avg = sum(aligned) / len(aligned)
    g2 = avg.abs().sum(axis=1).replace(0, np.nan)
    return avg.div(g2, axis=0).fillna(0)


def split_metrics(w, returns, train_end, val_end, fee_bps):
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    pnl_g = (w_a * r_a.shift(-1)).sum(axis=1)
    to    = (w_a - w_a.shift(1)).abs().sum(axis=1)
    pnl_n = pnl_g - to * fee_bps / 10000.0
    out = {"_to": float(to.mean())}
    splits = [("TRAIN", slice(None, train_end)),
              ("VAL",   slice(train_end, val_end)),
              ("TEST",  slice(val_end, None)),
              ("V+T",   slice(train_end, None)),
              ("FULL",  slice(None, None))]
    for lab, sl in splits:
        gg, nn = pnl_g.loc[sl].dropna(), pnl_n.loc[sl].dropna()
        sr_g = float(gg.mean()/gg.std(ddof=1)*np.sqrt(BARS_PER_YEAR)) if gg.std()>0 else float("nan")
        sr_n = float(nn.mean()/nn.std(ddof=1)*np.sqrt(BARS_PER_YEAR)) if nn.std()>0 else float("nan")
        ret_n = float(nn.mean() * BARS_PER_YEAR)
        eq_n  = (1 + nn).cumprod()
        dd_n  = float((eq_n / eq_n.cummax() - 1.0).min()) if len(eq_n) else float("nan")
        out[lab] = {"SR_g": sr_g, "SR_n": sr_n, "ret_n": ret_n, "dd_n": dd_n, "n": int(len(gg))}
    return out


def phase3_filter_sweep(df, weights_by_id, returns, train_end, val_end):
    log("PHASE 3 — filter sweep (equal-weight combiner on filtered subset)")
    filters = [
        ("none (baseline 190)",    lambda d: d),
        ("VAL_SR > 0",             lambda d: d[d["VAL_SR"] > 0]),
        ("VAL_SR > 0.5",           lambda d: d[d["VAL_SR"] > 0.5]),
        ("VAL_SR > 1.0",           lambda d: d[d["VAL_SR"] > 1.0]),
        ("VAL_SR > 1.5",           lambda d: d[d["VAL_SR"] > 1.5]),
        ("VAL/TRAIN > 0.3",        lambda d: d[d["VAL_SR"] / d["TRAIN_SR"] > 0.3]),
        ("VAL/TRAIN > 0.5",        lambda d: d[d["VAL_SR"] / d["TRAIN_SR"] > 0.5]),
        ("VAL/TRAIN > 0.7",        lambda d: d[d["VAL_SR"] / d["TRAIN_SR"] > 0.7]),
        ("VAL>0.5 AND VAL/TR>0.3", lambda d: d[(d["VAL_SR"] > 0.5) & (d["VAL_SR"]/d["TRAIN_SR"] > 0.3)]),
        ("VAL>1.0 AND VAL/TR>0.3", lambda d: d[(d["VAL_SR"] > 1.0) & (d["VAL_SR"]/d["TRAIN_SR"] > 0.3)]),
        ("top 30 by VAL",          lambda d: d.nlargest(30, "VAL_SR")),
        ("top 50 by VAL",          lambda d: d.nlargest(50, "VAL_SR")),
        ("top 100 by VAL",         lambda d: d.nlargest(100, "VAL_SR")),
    ]

    print()
    print(f"{'filter':28s} {'n':>4s} | "
          f"{'TRAIN':>6s} {'VAL':>6s} {'TEST':>6s} {'V+T':>6s} {'FULL':>6s} | "
          f"{'V+T ret':>8s} {'V+T DD':>7s} | {'TO':>6s}")
    print("-" * 110)
    survivors = {}
    rows = []
    for name, f in filters:
        sub = f(df)
        ids = sub["id"].tolist()
        if len(ids) == 0:
            print(f"{name:28s} {0:>4d} | (empty)")
            continue
        w_combined = combine_equal(weights_by_id, ids)
        m = split_metrics(w_combined, returns, train_end, val_end, COST_BPS)
        rows.append({"filter": name, "n": len(ids), **{
            f"{k}_SR_n": m[k]["SR_n"] for k in ("TRAIN","VAL","TEST","V+T","FULL")},
            "VT_ret_n": m["V+T"]["ret_n"], "VT_dd_n": m["V+T"]["dd_n"], "TO": m["_to"]})
        survivors[name] = ids
        print(f"{name:28s} {len(ids):>4d} | "
              f"{m['TRAIN']['SR_n']:>+6.2f} {m['VAL']['SR_n']:>+6.2f} {m['TEST']['SR_n']:>+6.2f} "
              f"{m['V+T']['SR_n']:>+6.2f} {m['FULL']['SR_n']:>+6.2f} | "
              f"{m['V+T']['ret_n']*100:>+7.1f}% {m['V+T']['dd_n']*100:>+6.1f}% | "
              f"{m['_to']:>5.3f}")
    print()
    return pd.DataFrame(rows), survivors


# ──────────────────────────────────────────────────────────────────
# Phase 4 — best filter × all combiners + plot
# ──────────────────────────────────────────────────────────────────
def phase4_combiner_sweep_on_filter(filter_name, ids, base_cfg):
    """Run all combiners on the surviving alpha subset using the unified pipeline."""
    log(f"PHASE 4 — all combiners on filter '{filter_name}' (n={len(ids)})")
    if len(ids) < 5:
        print(f"  not enough alphas (n={len(ids)}) — skipping combiner sweep")
        return None, None

    variants = [
        ("equal",       {"name": "equal",       "params": {"max_wt": 0.10}}),
        ("billions",    {"name": "billions",    "params": {"max_wt": 0.10}}),
        ("risk_par",    {"name": "risk_par",    "params": {"max_wt": 0.10}}),
        ("ic_wt",       {"name": "ic_wt",       "params": {"max_wt": 0.10}}),
        ("sharpe_wt",   {"name": "sharpe_wt",   "params": {"max_wt": 0.10}}),
        ("adaptive",    {"name": "adaptive",    "params": {"max_wt": 0.10}}),
        ("topn_train",  {"name": "topn_train",  "params": {"top_n": min(len(ids), 100)}}),
    ]
    results, rows = {}, []
    for label, comb in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg["alpha_source"] = {**cfg["alpha_source"],
                               "filter_sql": f"id IN ({','.join(str(i) for i in ids)})"}
        cfg["combiner"] = comb
        cfg["fees"] = {**cfg["fees"], "params": {"taker_bps": float(COST_BPS), "slippage_bps": 0.0}}
        cfg["qp"] = {"enabled": False}
        t0 = time.time()
        try:
            res = run(cfg, verbose=False)
            results[label] = res
            m = res.metrics
            rows.append({"variant": label, "n": res.alpha_signals_n,
                         "TRAIN_SR": m["TRAIN"]["SR_net"], "VAL_SR": m["VAL"]["SR_net"],
                         "TEST_SR": m["TEST"]["SR_net"], "VT_SR": m["VAL+TEST"]["SR_net"],
                         "FULL_SR": m["FULL"]["SR_net"],
                         "VT_ret": m["VAL+TEST"]["ret_ann_net"]*100,
                         "VT_DD":  m["VAL+TEST"]["max_dd_net"]*100,
                         "TO":     m["_turnover_per_bar"]})
            print(f"  {label:12s} {time.time()-t0:5.0f}s  V+T_SR={m['VAL+TEST']['SR_net']:+.2f}", flush=True)
        except Exception as e:
            print(f"  {label:12s} FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
    df_c = pd.DataFrame(rows).sort_values("VT_SR", ascending=False).reset_index(drop=True)

    print()
    print(f"  -- {filter_name} (n={len(ids)})  ALL COMBINERS, sorted by V+T net SR --")
    print(f"  {'variant':12s} | {'TRAIN':>6s} {'VAL':>6s} {'TEST':>6s} {'V+T':>6s} {'FULL':>6s} | "
          f"{'V+T ret':>8s} {'V+T DD':>7s} | {'TO':>6s}")
    print(f"  " + "-" * 95)
    for _, r in df_c.iterrows():
        print(f"  {r['variant']:12s} | "
              f"{r['TRAIN_SR']:>+6.2f} {r['VAL_SR']:>+6.2f} {r['TEST_SR']:>+6.2f} "
              f"{r['VT_SR']:>+6.2f} {r['FULL_SR']:>+6.2f} | "
              f"{r['VT_ret']:>+7.1f}% {r['VT_DD']:>+6.1f}% | "
              f"{r['TO']*100:>5.2f}%")
    return df_c, results


def plot_equity(res, label, train_end, val_end, out_png):
    gross = res.gross_pnl
    net   = res.net_pnl
    eq_g = (1.0 + gross).cumprod()
    eq_n = (1.0 + net).cumprod()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(eq_g.index, eq_g.values, lw=1.0, color="#888", alpha=0.7, label="gross")
    ax.plot(eq_n.index, eq_n.values, lw=1.6, color="#0a6cb0",      label=f"net ({COST_BPS} bps)")
    ax.axvline(train_end, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.axvline(val_end,   color="k", ls="--", lw=0.6, alpha=0.5)
    xmin, xmax = eq_g.index.min(), eq_g.index.max()
    ax.axvspan(xmin,      train_end, color="#cccccc", alpha=0.18)
    ax.axvspan(train_end, val_end,   color="#fff0b3", alpha=0.30)
    ax.axvspan(val_end,   xmax,      color="#cce8ff", alpha=0.30)
    ymax = eq_g.max() * 1.02
    ax.text(xmin + (train_end - xmin)/2,    ymax, "TRAIN", ha="center", va="top", fontsize=9, color="#555")
    ax.text(train_end + (val_end - train_end)/2, ymax, "VAL", ha="center", va="top", fontsize=9, color="#a08000")
    ax.text(val_end   + (xmax    - val_end  )/2, ymax, "TEST",  ha="center", va="top", fontsize=9, color="#005599")
    m = res.metrics
    ax.set_yscale("log")
    ax.set_title(
        f"{label}  |  KuCoin 4h, delay=0, {COST_BPS} bps\n"
        f"net SR — TRAIN {m['TRAIN']['SR_net']:+.2f}  VAL {m['VAL']['SR_net']:+.2f}  "
        f"TEST {m['TEST']['SR_net']:+.2f}  V+T {m['VAL+TEST']['SR_net']:+.2f}  FULL {m['FULL']['SR_net']:+.2f}  "
        f"|  V+T ret {m['VAL+TEST']['ret_ann_net']*100:+.0f}%/yr  DD {m['VAL+TEST']['max_dd_net']*100:+.0f}%",
        fontsize=10)
    ax.set_xlabel("date"); ax.set_ylabel("equity (start = 1.0, log)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="both", alpha=0.25); ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  saved: {out_png.relative_to(ROOT)}")


# ──────────────────────────────────────────────────────────────────
def main():
    matrices, tickers, train_end, val_end, cfg = phase1_sanity()
    df_per, weights_by_id = phase2_per_alpha(matrices, train_end, val_end)
    out_csv = ROOT / "data" / "crypto_per_alpha_oos.csv"
    df_per.drop(columns="expr").to_csv(out_csv, index=False, float_format="%.3f")
    print(f"  saved per-alpha: {out_csv.relative_to(ROOT)}")
    print()

    df_filt, survivors = phase3_filter_sweep(df_per, weights_by_id, matrices["returns"],
                                             train_end, val_end)
    out_csv2 = ROOT / "data" / "crypto_filter_sweep.csv"
    df_filt.to_csv(out_csv2, index=False, float_format="%.4f")
    print(f"  saved filter sweep: {out_csv2.relative_to(ROOT)}")
    print()

    # Pick best filter by V+T net SR (excluding the 'none' baseline)
    df_filt_nb = df_filt[df_filt["filter"] != "none (baseline 190)"].copy()
    if df_filt_nb.empty or df_filt_nb["V+T_SR_n"].isna().all():
        print("No non-baseline filter survived; skipping phase 4")
        return
    best_row = df_filt_nb.loc[df_filt_nb["V+T_SR_n"].idxmax()]
    best_filter = best_row["filter"]
    best_ids    = survivors[best_filter]
    print(f">>> BEST FILTER: '{best_filter}'  n={len(best_ids)}  V+T_SR_n={best_row['V+T_SR_n']:+.2f}")
    print()

    df_c, results_c = phase4_combiner_sweep_on_filter(best_filter, best_ids, cfg)
    if results_c:
        # Plot equity curve of the top combiner on the filtered subset
        top = df_c.iloc[0]
        plot_equity(results_c[top["variant"]],
                    label=f"{top['variant']}  |  filter: {best_filter} (n={len(best_ids)})",
                    train_end=train_end, val_end=val_end,
                    out_png=ROOT / "data" / "crypto_filtered_best_d0.png")

    log("DONE")


if __name__ == "__main__":
    main()
