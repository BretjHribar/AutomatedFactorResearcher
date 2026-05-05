"""
Universe construction deep dive — KuCoin 4h.

Hypothesis-driven experiments. For each universe variant, evaluate a fixed
benchmark alpha and measure TRAIN/VAL/TEST split metrics. Universe variants
covered:
  H1 — universe size (TOP10..TOP200)
  H2 — minimum-history requirement (no filter, 90d, 180d, 365d, 730d)
  H3 — rebalance frequency (1d, 5d, 10d, 20d, 60d, 120d)
  H4 — stablecoin / wrapped-token exclusion (name pattern filter)
  H5 — risk-model neutralization at alpha-eval time (none, beta-to-BTC residual,
       PCA-K residual on TRAIN window)

Benchmark alpha = the v10 multiplicative form found earlier:
  zscore_cs(multiply(add(add(idio_mom, parkinson_mom), vol_ratio), body))
where idio_mom = sma(ts_regression(log_returns, beta_to_btc, 120, 0, 0), 60).

Universe is computed ONCE per config; signal is computed ONCE per alpha; then
each (alpha, universe) cell takes O(bars × names) for portfolio + metric.

Outputs to experiments/results/:
  universe_h1_size.csv         (size sweep)
  universe_h2_history.csv      (history sweep)
  universe_h3_rebalance.csv    (rebalance sweep)
  universe_h4_exclusion.csv    (exclusion sweep)
  universe_h5_riskmodel.csv    (risk model sweep)
"""
from __future__ import annotations
import sys, json, time, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

# ── Paths / config ──────────────────────────────────────────────────────
MAT_DIR  = ROOT/"data/kucoin_cache/matrices/4h"
UNI_DIR  = ROOT/"data/kucoin_cache/universes"
DB_PATH  = ROOT/"data/alphas.db"
CFG      = json.load(open(ROOT/"prod/config/research_crypto.json"))
TRAIN_START = pd.Timestamp("2023-09-01")
TRAIN_END   = pd.Timestamp(CFG["splits"]["train_end"])
VAL_END     = pd.Timestamp(CFG["splits"]["val_end"])
BARS_PER_DAY = 6
BARS_PER_YEAR = BARS_PER_DAY * 365

OUT = ROOT/"experiments/results"
OUT.mkdir(parents=True, exist_ok=True)


# ── Benchmark alpha expression (v10 form — multiplicative) ──────────────
BENCHMARK_EXPR = (
  "zscore_cs(multiply(add(add("
    "ts_rank(sma(ts_regression(log_returns, beta_to_btc, 120, 0, 0), 60), 240), "
    "ts_rank(true_divide(ts_sum(log_returns, 120), df_max(parkinson_volatility_60, 0.0001)), 240)), "
    "ts_rank(sma(volume_ratio_20d, 120), 240)), "
    "ts_rank(Decay_exp(true_divide(open_close_range, df_max(high_low_range, 0.001)), 0.05), 240)))"
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Universe builders ──────────────────────────────────────────────────
def build_universe_topn_rebal(adv: pd.DataFrame, close: pd.DataFrame,
                              top_n: int,
                              rebal_bars: int,
                              min_history_days: int,
                              exclusions: list[str] = None,
                              ) -> pd.DataFrame:
    """Construct top-N by ADV with min-history filter + optional exclusions.

    rebal_bars: # of 4h bars between rebalances (e.g. 120 = every 20 days).
    min_history_days: ticker eligible only if data exists for ≥ this many days
        (with pre-existing tickers backdated to before data_start).
    exclusions: list of substring patterns; tickers containing any are excluded.
    """
    out = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    data_start = close.index[0]
    BACKDATE_BUFFER = pd.Timedelta(days=400)

    # Per-ticker first-active = first non-NaN close (backdated if at start)
    first_active = {}
    for col in close.columns:
        nonna = close[col].dropna().index
        if not len(nonna): continue
        if nonna[0] == data_start:
            first_active[col] = data_start - BACKDATE_BUFFER
        else:
            first_active[col] = nonna[0]
    fa = pd.Series(first_active)

    # Apply name-pattern exclusion
    if exclusions:
        excluded = set()
        for col in adv.columns:
            for pat in exclusions:
                if pat.upper() in col.upper():
                    excluded.add(col)
                    break
        # Mark ineligible by removing from fa
        fa = fa[~fa.index.isin(excluded)]

    rebal_idx = list(range(0, len(adv), rebal_bars))
    last_members = None
    min_days = min_history_days
    for i, b in enumerate(rebal_idx):
        next_b = rebal_idx[i+1] if i+1 < len(rebal_idx) else len(adv)
        rebal_ts = adv.index[b]
        eligible = [c for c in adv.columns
                    if c in fa and (rebal_ts - fa[c]).days >= min_days]
        adv_row = adv.iloc[b].reindex(eligible).dropna()
        if len(adv_row) < top_n:
            if last_members is None: continue
            members = last_members
        else:
            members = adv_row.nlargest(top_n).index.tolist()
            last_members = members
        out.iloc[b:next_b, out.columns.get_indexer(members)] = True
    return out


# ── Signal evaluation ──────────────────────────────────────────────────
def eval_signal(expr: str, matrices: dict) -> pd.DataFrame:
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expr)


def signal_to_portfolio(sig: pd.DataFrame,
                        universe: pd.DataFrame,
                        max_wt: float = 0.10,
                        risk_residualize: str = None,
                        residualize_args: dict = None,
                        ) -> pd.DataFrame:
    """Demean within universe, gross-normalize, optional clip.

    risk_residualize:
      None       — no residualization
      'beta_btc' — residualize signal against ticker's beta_to_btc
      'pca'      — project out top-K PCs of TRAIN-period returns
    """
    # Align columns
    common_cols = sig.columns.intersection(universe.columns)
    sig = sig[common_cols].copy()
    uni_mask = universe.reindex(index=sig.index, columns=common_cols).fillna(False).astype(bool)

    # Mask non-universe to NaN
    s = sig.where(uni_mask, np.nan).replace([np.inf, -np.inf], np.nan)

    # Cross-sectional demean
    demean = s.sub(s.mean(axis=1), axis=0)

    # Optional risk-residualization (cross-sectional, per bar)
    if risk_residualize == "pca":
        # Static PCA: compute on TRAIN returns only, project out top-K loadings
        K = residualize_args.get("k", 5)
        ret = residualize_args["returns"]
        # Align columns of returns to current `demean` columns
        cols = list(demean.columns)
        ret_train_df = ret.loc[:TRAIN_END, [c for c in cols if c in ret.columns]]
        # Fill any missing cols with 0
        ret_train_df = ret_train_df.reindex(columns=cols).fillna(0)
        R = ret_train_df.values - ret_train_df.values.mean(axis=0, keepdims=True)
        cov = R.T @ R / max(len(R) - 1, 1)
        cov = (cov + cov.T) * 0.5 + 1e-10 * np.eye(len(cols))
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        U_K = eigvecs[:, order[:K]]  # orthonormal (N, K)
        # Project out: w_resid = w - U_K U_K' w (NaN-safe by filling temporarily)
        D = demean.fillna(0).values
        proj = D @ U_K @ U_K.T
        residual = D - proj
        # Restore NaN-mask using uni_mask
        residual_df = pd.DataFrame(residual, index=demean.index, columns=demean.columns)
        demean = residual_df.where(uni_mask, np.nan)
    elif risk_residualize == "beta_btc":
        # Per-bar cross-sectional regression of signal on beta_to_btc loadings
        beta_loadings = residualize_args["beta_to_btc"][common_cols].reindex(
            index=demean.index)
        # For each row: residualize signal = signal − slope × beta
        for ts in demean.index:
            y = demean.loc[ts].values
            x = beta_loadings.loc[ts].values
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() < 5:
                continue
            yv, xv = y[valid], x[valid]
            xm = xv.mean()
            ssx = np.sum((xv - xm) ** 2)
            if ssx == 0: continue
            b = np.sum((xv - xm) * (yv - yv.mean())) / ssx
            a = yv.mean() - b * xm
            demean.loc[ts, valid] = yv - (a + b * xv)

    # Re-mask, gross-normalize, THEN clip (matches process_signal in eval_alpha.py)
    demean = demean.where(uni_mask, np.nan)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    w = demean.div(gross, axis=0)
    w = w.clip(lower=-max_wt, upper=max_wt)
    return w.fillna(0)


def split_metrics(w: pd.DataFrame, ret: pd.DataFrame, fee_bps: float = 0.0) -> dict:
    common = w.index.intersection(ret.index)
    w_a = w.loc[common].fillna(0)
    r_a = ret.loc[common].fillna(0)
    pnl_g = (w_a * r_a.shift(-1)).sum(axis=1)
    to    = (w_a - w_a.shift(1)).abs().sum(axis=1)
    pnl_n = pnl_g - to * fee_bps / 10000.0
    out = {"to_per_bar": float(to.mean())}
    splits = [("TRAIN", slice(None, TRAIN_END)),
              ("VAL",   slice(TRAIN_END, VAL_END)),
              ("TEST",  slice(VAL_END, None)),
              ("VT",    slice(TRAIN_END, None)),
              ("FULL",  slice(None, None))]
    for lab, sl in splits:
        gg, nn = pnl_g.loc[sl].dropna(), pnl_n.loc[sl].dropna()
        sr_g = float(gg.mean()/gg.std(ddof=1)*np.sqrt(BARS_PER_YEAR)) if gg.std()>0 else float("nan")
        sr_n = float(nn.mean()/nn.std(ddof=1)*np.sqrt(BARS_PER_YEAR)) if nn.std()>0 else float("nan")
        ret_n = float(nn.mean() * BARS_PER_YEAR)
        eq = (1+nn).cumprod()
        dd = float((eq/eq.cummax() - 1.0).min()) if len(eq) else float("nan")
        out[f"{lab}_SR_g"] = sr_g
        out[f"{lab}_SR_n"] = sr_n
        out[f"{lab}_ret_n"] = ret_n
        out[f"{lab}_dd_n"] = dd
        out[f"{lab}_n"] = int(len(gg))
    return out


# ── Load all matrices once ─────────────────────────────────────────────
def load_all():
    log("Loading matrices...")
    matrices = {}
    for fp in sorted(MAT_DIR.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        df = pd.read_parquet(fp)
        matrices[fp.stem] = df
    print(f"  {len(matrices)} fields, sample shape: {matrices['close'].shape}")
    return matrices


# ── Main ─────────────────────────────────────────────────────────────
def main():
    matrices = load_all()
    close = matrices["close"]
    adv = matrices["adv20"]
    rets = matrices["returns"]

    log(f"Evaluating benchmark alpha (one-time)...")
    sig = eval_signal(BENCHMARK_EXPR, matrices)
    log(f"  signal shape={sig.shape}")

    # ─────────────────────────────────────────────────────────────────
    # H1 — Universe size sweep
    # ─────────────────────────────────────────────────────────────────
    log("=== H1: universe size sweep (rebal=20d, min_hist=365d) ===")
    rows = []
    for top_n in [10, 20, 30, 50, 75, 100, 150, 200]:
        t0 = time.time()
        uni = build_universe_topn_rebal(adv, close, top_n=top_n,
                                        rebal_bars=120, min_history_days=365)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  TOP{top_n}: not enough eligible (skipped)")
            continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, rets, fee_bps=3.0)
        m["top_n"] = top_n
        m["rebal_bars"] = 120
        m["min_hist_days"] = 365
        m["avg_active"] = float(uni.sum(axis=1).mean())
        m["t_sec"] = time.time() - t0
        rows.append(m)
        print(f"  TOP{top_n:>3d} active={m['avg_active']:>5.1f} "
              f"TRAIN_SR={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} TO={m['to_per_bar']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h1_size.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h1_size.csv'}")

    # ─────────────────────────────────────────────────────────────────
    # H2 — Min-history requirement sweep (TOP30, rebal=20d)
    # ─────────────────────────────────────────────────────────────────
    log("=== H2: min-history sweep (TOP30, rebal=20d) ===")
    rows = []
    for min_days in [0, 30, 90, 180, 365, 540, 730]:
        t0 = time.time()
        uni = build_universe_topn_rebal(adv, close, top_n=30,
                                        rebal_bars=120, min_history_days=min_days)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  min_hist={min_days}d: skipped"); continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, rets, fee_bps=3.0)
        m["top_n"] = 30
        m["min_hist_days"] = min_days
        m["avg_active"] = float(uni.sum(axis=1).mean())
        m["unique_tickers"] = int(uni.any(axis=0).sum())
        m["t_sec"] = time.time() - t0
        rows.append(m)
        print(f"  min_hist={min_days:>4d}d  unique={m['unique_tickers']:>4d}  "
              f"active={m['avg_active']:>5.1f}  "
              f"TRAIN={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h2_history.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h2_history.csv'}")

    # ─────────────────────────────────────────────────────────────────
    # H3 — Rebalance frequency sweep (TOP30, min_hist=365d)
    # ─────────────────────────────────────────────────────────────────
    log("=== H3: rebalance frequency sweep (TOP30, min_hist=365d) ===")
    rows = []
    for rebal_days in [1, 5, 10, 20, 60, 120, 240]:
        t0 = time.time()
        rb = rebal_days * BARS_PER_DAY
        uni = build_universe_topn_rebal(adv, close, top_n=30,
                                        rebal_bars=rb, min_history_days=365)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  rebal={rebal_days}d: skipped"); continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, rets, fee_bps=3.0)
        m["top_n"] = 30
        m["rebal_days"] = rebal_days
        m["avg_active"] = float(uni.sum(axis=1).mean())
        diffs = uni.astype(int).diff().abs().sum(axis=1)
        m["uni_change_bars"] = int((diffs > 0).sum())
        m["avg_swap_per_rebal"] = float(diffs[diffs > 0].mean()) if (diffs > 0).any() else 0
        m["t_sec"] = time.time() - t0
        rows.append(m)
        print(f"  rebal={rebal_days:>3d}d  changes={m['uni_change_bars']:>3d}  "
              f"avg_swap={m['avg_swap_per_rebal']:>5.1f}  "
              f"TRAIN={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} TO={m['to_per_bar']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h3_rebalance.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h3_rebalance.csv'}")

    # ─────────────────────────────────────────────────────────────────
    # H4 — Stablecoin / wrapped-token exclusion (TOP30, rebal=20d)
    # ─────────────────────────────────────────────────────────────────
    log("=== H4: name-pattern exclusion (TOP30, rebal=20d, min_hist=365d) ===")
    rows = []
    exclusion_sets = {
        "no_filter":    [],
        "stables":      ["USDC", "DAI", "TUSD", "BUSD", "FDUSD"],
        "stables+wraps": ["USDC", "DAI", "TUSD", "BUSD", "FDUSD",
                          "WBTC", "WETH", "STETH", "RETH", "CBETH"],
        "memes":        ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "MEME",
                         "MOG", "MOODENG", "TRUMP", "FART"],
    }
    for name, patterns in exclusion_sets.items():
        t0 = time.time()
        uni = build_universe_topn_rebal(adv, close, top_n=30,
                                        rebal_bars=120, min_history_days=365,
                                        exclusions=patterns)
        if (uni.sum(axis=1) > 0).sum() < 100:
            print(f"  {name}: skipped"); continue
        w = signal_to_portfolio(sig, uni)
        m = split_metrics(w, rets, fee_bps=3.0)
        m["top_n"] = 30
        m["filter"] = name
        m["n_excluded"] = len(patterns)
        m["unique_tickers"] = int(uni.any(axis=0).sum())
        m["avg_active"] = float(uni.sum(axis=1).mean())
        m["t_sec"] = time.time() - t0
        rows.append(m)
        print(f"  {name:18s}  unique={m['unique_tickers']:>4d}  "
              f"TRAIN={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h4_exclusion.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h4_exclusion.csv'}")

    # ─────────────────────────────────────────────────────────────────
    # H5 — Risk-model neutralization sweep (TOP30, rebal=20d, min_hist=365d)
    # ─────────────────────────────────────────────────────────────────
    log("=== H5: risk-model neutralization (TOP30, rebal=20d, min_hist=365d) ===")
    uni = build_universe_topn_rebal(adv, close, top_n=30, rebal_bars=120,
                                    min_history_days=365)
    rows = []
    rm_configs = [
        ("none",      None,  {}),
        ("beta_btc",  "beta_btc",  {"beta_to_btc": matrices["beta_to_btc"]}),
        ("pca_K=1",   "pca", {"k": 1, "returns": rets}),
        ("pca_K=2",   "pca", {"k": 2, "returns": rets}),
        ("pca_K=3",   "pca", {"k": 3, "returns": rets}),
        ("pca_K=5",   "pca", {"k": 5, "returns": rets}),
        ("pca_K=10",  "pca", {"k": 10, "returns": rets}),
        ("pca_K=20",  "pca", {"k": 20, "returns": rets}),
    ]
    for name, mode, args in rm_configs:
        t0 = time.time()
        w = signal_to_portfolio(sig, uni, risk_residualize=mode, residualize_args=args)
        m = split_metrics(w, rets, fee_bps=3.0)
        m["risk_model"] = name
        m["t_sec"] = time.time() - t0
        rows.append(m)
        print(f"  {name:12s}  TRAIN={m['TRAIN_SR_n']:+.2f} VAL={m['VAL_SR_n']:+.2f} "
              f"TEST={m['TEST_SR_n']:+.2f} VT={m['VT_SR_n']:+.2f} "
              f"DD_VT={m['VT_dd_n']*100:+.0f}% TO={m['to_per_bar']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT/"universe_h5_riskmodel.csv", index=False, float_format="%.4f")
    log(f"  saved {OUT/'universe_h5_riskmodel.csv'}")

    log("ALL DONE")


if __name__ == "__main__":
    main()
