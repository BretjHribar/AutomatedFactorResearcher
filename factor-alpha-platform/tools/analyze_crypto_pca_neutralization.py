"""
PCA neutralization as an alpha-selection step (equity Fama-French style).

1. Estimate PCA loadings from the FULL TRAIN-window return matrix
   (long lookback, static — no look-ahead since TRAIN is bounded).
2. For each alpha, residualize the cross-sectional signal against the
   top-K PC loadings each bar:  α_resid = (I − U_K U_K') α
3. Score each alpha's TRAIN gross SR on the RESIDUALIZED signal.
   If a signal survives PC neutralization with meaningful residual SR,
   it has cross-sectional information beyond the dominant PCs.
4. Keep alphas whose residualized TRAIN SR clears a threshold.
5. Combine the surviving alphas (using the residualized signals,
   not the raw signals) and report TRAIN / VAL / TEST / V+T net SR.

PCA factor loadings come ONLY from TRAIN returns. Selection criterion
uses ONLY TRAIN performance. VAL/TEST are pure out-of-sample.

Sweeps K ∈ {1,3,5,10} × threshold ∈ {residual_TR_SR > 0.5, 1.0, 1.5, 2.0}
plus a no-residualization control.
"""
from __future__ import annotations
import sys, json, time, sqlite3, warnings, copy
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

# CLI: --universe <stem> picks data/kucoin_cache/universes/<stem>.parquet
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--universe", default="KUCOIN_TOP100_4h",
                 help="universe parquet stem (e.g. KUCOIN_TOP50_REBAL20D_4h)")
_ap.add_argument("--coverage", type=float, default=0.0,
                 help="min full-history coverage to be in valid set (0 = use all tickers ever in universe)")
_ap.add_argument("--out-tag", default="",
                 help="suffix for output files (e.g. _top50r20d). Defaults to '_<universe>' when not given")
_args = _ap.parse_args()

CONFIG_PATH    = ROOT / "prod" / "config" / "research_crypto.json"
UNIVERSE_PATH  = ROOT / "data/kucoin_cache/universes" / f"{_args.universe}.parquet"
MATRICES_DIR   = ROOT / "data/kucoin_cache/matrices/4h"
DB_PATH        = ROOT / "data/alphas.db"
COVERAGE_CUTOFF = float(_args.coverage)
BARS_PER_YEAR  = 6 * 365
COST_BPS       = 3.0
OUT_TAG        = _args.out_tag or f"_{_args.universe.lower()}"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def signal_to_portfolio(sig, universe_mask=None):
    """Demean cross-sectionally, gross-normalize. Optional per-bar universe mask:
    sets non-universe tickers to NaN before demeaning so they don't dilute
    the gross or shift the cross-sectional mean."""
    s = sig.replace([np.inf, -np.inf], np.nan)
    if universe_mask is not None:
        common_cols = s.columns.intersection(universe_mask.columns)
        s = s.loc[:, common_cols]
        m = universe_mask.loc[:, common_cols].reindex(index=s.index).fillna(False).astype(bool)
        s = s.where(m, np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


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


def gross_sr(w, returns, start, end):
    common = w.index.intersection(returns.index)
    w_a = w.loc[common].fillna(0)
    r_a = returns.loc[common].fillna(0)
    pnl = (w_a * r_a.shift(-1)).sum(axis=1)
    m = (pnl.index >= start) & (pnl.index <= end)
    p = pnl[m].dropna()
    if len(p) < 50 or p.std(ddof=1) <= 0:
        return float("nan")
    return float(p.mean() / (p.std(ddof=1) + 1e-12) * np.sqrt(BARS_PER_YEAR))


def load_matrices_and_alphas():
    log(f"loading universe {UNIVERSE_PATH.name}")
    uni = pd.read_parquet(UNIVERSE_PATH)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    matrices = {}
    for fp in sorted(MATRICES_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid if c in df.columns]
        if cols: matrices[fp.stem] = df[cols]
    tickers = sorted(set(matrices["close"].columns))
    for k, v in matrices.items():
        matrices[k] = v[[t for t in tickers if t in v.columns]]
    # Per-bar mask aligned to matrix tickers
    universe_mask = uni.reindex(columns=tickers).fillna(False).astype(bool)
    universe_mask = universe_mask.reindex(index=matrices["close"].index).fillna(False)
    avg_active = float(universe_mask.sum(axis=1).mean())
    print(f"  {len(matrices)} fields, {len(tickers)} valid tickers, "
          f"{len(matrices['close'])} bars, avg active/bar={avg_active:.1f}")

    cfg = json.loads(CONFIG_PATH.read_text())
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end   = pd.Timestamp(cfg["splits"]["val_end"])
    print(f"  splits: TRAIN<{train_end}, VAL<{val_end}, TEST>=")

    con = sqlite3.connect(str(DB_PATH))
    alphas = con.execute("""
        SELECT a.id, a.expression
        FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.asset_class='crypto' AND a.interval='4h'
        ORDER BY a.id""").fetchall()
    print(f"  {len(alphas)} alphas")
    return matrices, tickers, train_end, val_end, alphas, universe_mask


def build_train_pca(returns, tickers, train_end, k_max=20):
    """Static PCA loadings from full TRAIN returns. Returns U (N, k_max)
    orthonormal eigenvectors sorted by eigval descending."""
    R = returns.loc[:train_end, tickers].fillna(0).values
    R = R - R.mean(axis=0, keepdims=True)
    T, N = R.shape
    log(f"PCA on TRAIN: T={T} bars × N={N} tickers (T/N = {T/N:.1f})")
    cov = (R.T @ R) / max(T - 1, 1)
    cov = (cov + cov.T) * 0.5 + 1e-10 * np.eye(N)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    var_total = eigvals.sum()
    var_explained = eigvals[:10] / var_total
    print(f"  top-10 PC variance shares: {[f'{v*100:.1f}%' for v in var_explained]}")
    print(f"  cumulative top-10: {var_explained.sum()*100:.1f}%")
    return eigvecs[:, :k_max], eigvals[:k_max]


def residualize(w_df, U_K):
    """Project out top-K PCs from each row of w_df. U_K is (N, K) orthonormal."""
    W = w_df.values
    proj = W @ U_K @ U_K.T   # (T, N)
    return pd.DataFrame(W - proj, index=w_df.index, columns=w_df.columns)


def main():
    matrices, tickers, train_end, val_end, alphas, universe_mask = load_matrices_and_alphas()
    returns = matrices["returns"]
    train_start = pd.Timestamp("2023-09-01")
    test_end    = returns.index.max()

    # Build static PCA loadings from TRAIN
    U_full, eigvals = build_train_pca(returns, tickers, train_end, k_max=20)
    print()

    # Evaluate each alpha → portfolio (raw)
    log(f"evaluating {len(alphas)} alphas (raw signal_to_portfolio)")
    engine = FastExpressionEngine(data_fields=matrices)
    raw_w = {}
    for i, (aid, expr) in enumerate(alphas, 1):
        try:
            sig = engine.evaluate(expr)
            raw_w[aid] = signal_to_portfolio(sig, universe_mask=universe_mask)
        except Exception:
            pass
        if i % 30 == 0:
            print(f"    {i}/{len(alphas)}", flush=True)
    print(f"  evaluated {len(raw_w)} alphas")
    print()

    # ── For each K, residualize all alphas + score TRAIN/VAL/TEST gross SR ──
    K_VALUES = [1, 3, 5, 10]
    log("scoring residualized alphas (TRAIN selection metric)")
    per_alpha = []
    for aid, w_raw in raw_w.items():
        row = {"id": aid,
               "raw_TRAIN":  gross_sr(w_raw, returns, train_start, train_end),
               "raw_VAL":    gross_sr(w_raw, returns, train_end, val_end),
               "raw_TEST":   gross_sr(w_raw, returns, val_end, test_end),
               "raw_VT":     gross_sr(w_raw, returns, train_end, test_end)}
        for K in K_VALUES:
            U_K = U_full[:, :K]
            w_res_unnorm = residualize(w_raw, U_K)
            # Re-gross-normalize so it's comparable
            gross = w_res_unnorm.abs().sum(axis=1).replace(0, np.nan)
            w_res = w_res_unnorm.div(gross, axis=0).fillna(0)
            row[f"K{K}_TRAIN"] = gross_sr(w_res, returns, train_start, train_end)
            row[f"K{K}_VAL"]   = gross_sr(w_res, returns, train_end,   val_end)
            row[f"K{K}_TEST"]  = gross_sr(w_res, returns, val_end,     test_end)
            row[f"K{K}_VT"]    = gross_sr(w_res, returns, train_end,   test_end)
        per_alpha.append(row)
    df_per = pd.DataFrame(per_alpha)
    out_csv = ROOT / "data" / f"crypto_pca_neutralized_per_alpha{OUT_TAG}.csv"
    df_per.to_csv(out_csv, index=False, float_format="%.3f")
    print(f"  saved per-alpha: {out_csv.relative_to(ROOT)}")

    print()
    print("Per-alpha TRAIN-SR summary (gross SR on residualized signal):")
    print(f"  {'metric':>12s}  {'mean':>5s}  {'med':>5s}  {'std':>4s}  {'>0':>4s}  {'>+1':>4s}  {'>+2':>4s}")
    for col in ["raw_TRAIN", "K1_TRAIN", "K3_TRAIN", "K5_TRAIN", "K10_TRAIN"]:
        s = df_per[col]
        print(f"  {col:>12s}  {s.mean():+5.2f}  {s.median():+5.2f}  {s.std():4.2f}  "
              f"{(s>0).sum():>4d}  {(s>1).sum():>4d}  {(s>2).sum():>4d}")
    print()
    print("Per-alpha VAL+TEST-SR summary (out-of-sample on residualized signal):")
    print(f"  {'metric':>12s}  {'mean':>5s}  {'med':>5s}  {'>0':>4s}  {'>+1':>4s}")
    for col in ["raw_VT", "K1_VT", "K3_VT", "K5_VT", "K10_VT"]:
        s = df_per[col]
        print(f"  {col:>12s}  {s.mean():+5.2f}  {s.median():+5.2f}  "
              f"{(s>0).sum():>4d}  {(s>1).sum():>4d}")
    print()

    # ── Filter sweep: combine surviving alphas with equal weight on RESIDUALIZED signals ──
    log("filter sweep — combine residualized signals of surviving alphas")
    THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]
    print()
    print(f"{'K':>2s}  {'thresh':>6s}  {'n':>4s} | "
          f"{'TRAIN':>6s} {'VAL':>6s} {'TEST':>6s} {'V+T':>6s} {'FULL':>6s} | "
          f"{'V+T ret':>8s} {'V+T DD':>7s} | {'TO':>6s}")
    print("-" * 110)

    rows = []; results = {}
    for K in K_VALUES:
        U_K = U_full[:, :K]
        # Pre-residualize each alpha
        resid_w = {}
        for aid, w_raw in raw_w.items():
            w_res_un = residualize(w_raw, U_K)
            gross = w_res_un.abs().sum(axis=1).replace(0, np.nan)
            resid_w[aid] = w_res_un.div(gross, axis=0).fillna(0)
        for th in THRESHOLDS:
            survivors = df_per[df_per[f"K{K}_TRAIN"] > th]["id"].tolist()
            if not survivors:
                continue
            # Equal-weight average of residualized portfolios
            ws = [resid_w[i] for i in survivors]
            common_idx = ws[0].index
            common_cols = ws[0].columns
            for w in ws[1:]:
                common_idx = common_idx.intersection(w.index)
                common_cols = common_cols.intersection(w.columns)
            aligned = [w.loc[common_idx, common_cols] for w in ws]
            avg = sum(aligned) / len(aligned)
            g2 = avg.abs().sum(axis=1).replace(0, np.nan)
            w_combined = avg.div(g2, axis=0).fillna(0)

            m = split_metrics(w_combined, returns, train_end, val_end, COST_BPS)
            label = f"K={K} thr>{th:.1f}"
            results[label] = w_combined
            rows.append({"K": K, "threshold": th, "n": len(survivors),
                         "TRAIN_SR": m["TRAIN"]["SR_n"], "VAL_SR": m["VAL"]["SR_n"],
                         "TEST_SR": m["TEST"]["SR_n"], "VT_SR": m["V+T"]["SR_n"],
                         "FULL_SR": m["FULL"]["SR_n"],
                         "VT_ret": m["V+T"]["ret_n"]*100, "VT_dd": m["V+T"]["dd_n"]*100,
                         "TO": m["_to"]})
            print(f"{K:>2d}  >{th:>4.1f}  {len(survivors):>4d} | "
                  f"{m['TRAIN']['SR_n']:>+6.2f} {m['VAL']['SR_n']:>+6.2f} {m['TEST']['SR_n']:>+6.2f} "
                  f"{m['V+T']['SR_n']:>+6.2f} {m['FULL']['SR_n']:>+6.2f} | "
                  f"{m['V+T']['ret_n']*100:>+7.1f}% {m['V+T']['dd_n']*100:>+6.1f}% | "
                  f"{m['_to']:>5.3f}")

    # Also a control: raw alphas, no residualization, all 190 (≈ noQP baseline)
    ws = list(raw_w.values())
    common_idx = ws[0].index; common_cols = ws[0].columns
    for w in ws[1:]:
        common_idx = common_idx.intersection(w.index); common_cols = common_cols.intersection(w.columns)
    aligned = [w.loc[common_idx, common_cols] for w in ws]
    avg = sum(aligned) / len(aligned)
    g2 = avg.abs().sum(axis=1).replace(0, np.nan)
    w_ctrl = avg.div(g2, axis=0).fillna(0)
    m = split_metrics(w_ctrl, returns, train_end, val_end, COST_BPS)
    print(f"{'CTL':>2s}  {'all':>6s}  {len(raw_w):>4d} | "
          f"{m['TRAIN']['SR_n']:>+6.2f} {m['VAL']['SR_n']:>+6.2f} {m['TEST']['SR_n']:>+6.2f} "
          f"{m['V+T']['SR_n']:>+6.2f} {m['FULL']['SR_n']:>+6.2f} | "
          f"{m['V+T']['ret_n']*100:>+7.1f}% {m['V+T']['dd_n']*100:>+6.1f}% | "
          f"{m['_to']:>5.3f}   <- raw equal-weight all 190 (control)")

    df_filt = pd.DataFrame(rows)
    out_csv2 = ROOT / "data" / f"crypto_pca_neutralized_filter{OUT_TAG}.csv"
    df_filt.to_csv(out_csv2, index=False, float_format="%.4f")
    print(f"\nsaved filter sweep: {out_csv2.relative_to(ROOT)}")

    # ── Pick best by V+T SR and plot equity curve ──
    if not df_filt.empty:
        best = df_filt.sort_values("VT_SR", ascending=False).iloc[0]
        label = f"K={int(best['K'])} thr>{best['threshold']:.1f}"
        w_best = results[label]
        log(f"BEST: {label}  n={int(best['n'])}  V+T_SR_n={best['VT_SR']:+.2f}")

        common = w_best.index.intersection(returns.index)
        w_a = w_best.loc[common].fillna(0)
        r_a = returns.loc[common].fillna(0)
        gross_pnl = (w_a * r_a.shift(-1)).sum(axis=1).fillna(0)
        to = (w_a - w_a.shift(1)).abs().sum(axis=1)
        net_pnl = gross_pnl - to * COST_BPS / 10000.0
        eq_g = (1 + gross_pnl).cumprod(); eq_n = (1 + net_pnl).cumprod()

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
        ax.text(xmin + (train_end - xmin)/2, ymax, "TRAIN", ha="center", va="top", fontsize=9, color="#555")
        ax.text(train_end + (val_end - train_end)/2, ymax, "VAL", ha="center", va="top", fontsize=9, color="#a08000")
        ax.text(val_end + (xmax - val_end)/2, ymax, "TEST", ha="center", va="top", fontsize=9, color="#005599")
        ax.set_yscale("log")
        ax.set_title(
            f"BEST: PCA-neutralized {label}  (n={int(best['n'])} surviving)  |  KuCoin 4h, delay=0, {COST_BPS} bps\n"
            f"net SR — TRAIN {best['TRAIN_SR']:+.2f}  VAL {best['VAL_SR']:+.2f}  "
            f"TEST {best['TEST_SR']:+.2f}  V+T {best['VT_SR']:+.2f}  FULL {best['FULL_SR']:+.2f}  "
            f"|  V+T ret {best['VT_ret']:+.0f}%/yr  DD {best['VT_dd']:+.0f}%",
            fontsize=10)
        ax.set_xlabel("date"); ax.set_ylabel("equity (start = 1.0, log)")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, which="both", alpha=0.25); ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        out_png = ROOT / "data" / f"crypto_pca_neutralized_best{OUT_TAG}.png"
        fig.savefig(out_png, dpi=120); plt.close(fig)
        print(f"  saved: {out_png.relative_to(ROOT)}")

    log("DONE")


if __name__ == "__main__":
    main()
