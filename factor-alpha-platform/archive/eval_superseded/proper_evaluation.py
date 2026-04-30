"""
Proper statistical evaluation comparing equity AIPT configurations.

For each (config, seed):
  1. Run AIPT walk-forward over the full history (no special TRAIN/TEST split)
  2. Save per-bar gross / turnover / IC series

Then for each config, aggregate across seeds and compute:
  - Sharpe in 12 non-overlapping 6-month rolling windows over 2020-04 → 2026-04
    (12 OOS Sharpe samples per config × 5 seeds = 60 estimates)
  - Block-bootstrap (5-day blocks) Sharpe distribution on the full-period series
  - Pairwise paired-difference comparisons with bootstrap CI

This characterizes BOTH seed variance AND sampling variance — the right way
to compare strategies whose TEST Sharpe estimates have ±1 SE on 14 months.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import backtest_voc_equities_neutralized as base
from backtest_voc_equities_d44_fund import RAW_FIELDS, add_extra_fundamentals

# ── Configurations to compare ─────────────────────────────────────────────────
SEEDS = [7, 13, 42, 100, 200]          # 5 seeds per config

CONFIGS = {
    "D24_arbitrary": {
        "chars": [
            "log_returns",
            "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
            "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
            "book_to_market", "earnings_yield", "free_cashflow_yield",
            "ev_to_ebitda", "ev_to_revenue",
            "roe", "roa", "gross_margin", "operating_margin", "net_margin", "asset_turnover",
            "adv20", "adv60", "dollars_traded", "cap",
            "debt_to_equity", "current_ratio",
        ],
        "signs": {},   # all +1
        "P": 2000,
    },
    "D3_block_cv": {
        "chars_file": "selected_chars_block_cv.json",
        "P": 1000,
    },
    "D11_combined_ic": {
        "chars_file": "selected_chars_combined_ic.json",
        "P": 1000,
    },
    "D44_fundamentals": {
        "chars": [
            # 24 originals + 20 fundamentals (from D44 set)
            "log_returns",
            "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
            "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
            "book_to_market", "earnings_yield", "free_cashflow_yield",
            "ev_to_ebitda", "ev_to_revenue",
            "roe", "roa", "gross_margin", "operating_margin", "net_margin", "asset_turnover",
            "adv20", "adv60", "dollars_traded", "cap",
            "debt_to_equity", "current_ratio",
            "asset_growth", "sales_growth", "eps_growth",
            "accruals", "cash_to_assets",
            "dividend_yield", "payout_ratio",
            "interest_coverage", "net_debt_to_ebitda", "quick_ratio",
            "gross_profit_to_assets", "fcf_to_revenue",
            "capex_to_revenue", "capex_to_depreciation", "shares_change_252d",
            "goodwill_to_assets", "intangibles_to_assets",
            "cf_to_debt", "ebit_to_ev", "roe_change_252d",
        ],
        "signs": {},
        "P": 2000,
    },
}

WINDOW_BARS = 126           # ~6 months
OOS_RUN_START     = "2018-01-01"   # earliest bar at which run_aipt_once emits OOS PnL
OOS_WINDOWS_START = "2020-04-01"   # earliest non-overlapping window for stats
OOS_WINDOWS_END   = "2026-04-20"
N_BOOTSTRAP = 1000          # block bootstrap iterations
BLOCK_LEN = 5               # 5-day blocks for return series
GAMMA_REF_D = 24

OUT_DIR = base.RESULTS_DIR / "proper_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading (one-shot for all configs) ──────────────────────────────────
def load_all_data(all_chars):
    """Load PIT matrices + classifications + compute extra ratios."""
    print(f"Loading PIT matrices...", flush=True)
    uni = pd.read_parquet(base.UNIVERSES_DIR / f"{base.UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > base.COVERAGE_CUTOFF].index.tolist())
    matrices = {}
    needed = set(all_chars) | set(RAW_FIELDS) | {"close"}
    for fp in sorted(base.PIT_DIR.glob("*.parquet")):
        if fp.stem not in needed:
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)
    add_extra_fundamentals(matrices)
    print(f"  T={len(common_idx)} N={len(tickers)} {len(matrices)} chars/ratios loaded", flush=True)
    with open(base.CLASSIF_PATH) as fh:
        classifications = json.load(fh)
    return matrices, tickers, common_idx, classifications


def resolve_config(cfg_name, cfg):
    """Resolve chars + signs from JSON file if needed."""
    if "chars_file" in cfg:
        sel = json.load(open(base.RESULTS_DIR / cfg["chars_file"]))
        return sel["selected"], {c: int(sel["signs"][c]) for c in sel["selected"]}, cfg["P"]
    return list(cfg["chars"]), dict(cfg.get("signs", {})), cfg["P"]


# ── Run AIPT once for a (config, seed), return per-bar PnL DataFrame ─────────
def run_aipt_once(matrices, tickers, dates, close_vals, classifications,
                  chars, signs, P, seed):
    """Apply signs, build Z, run AIPT, return per-bar DataFrame."""
    # Apply signs
    matrices_signed = dict(matrices)
    for c in chars:
        if signs.get(c, 1) < 0 and c in matrices_signed:
            matrices_signed[c] = -matrices_signed[c]

    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_RUN_START)
    start_bar = max(1, oos_start_idx - base.TRAIN_BARS - 10)
    Z_panel, D = base.build_Z_panel(matrices_signed, tickers, chars, start_bar, T_total, delay=1)

    # Patch GAMMA_GRID for this D
    gamma_scale = float(np.sqrt(GAMMA_REF_D / D))
    saved_grid = base.GAMMA_GRID
    base.GAMMA_GRID = [g * gamma_scale for g in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
    try:
        df = base.run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total,
                                            oos_start_idx, D, tickers, classifications,
                                            matrices_signed, mode="baseline", seed=seed)
    finally:
        base.GAMMA_GRID = saved_grid
    df["date"] = [dates[i] for i in df["bar_idx"]]
    return df


# ── Statistical aggregation ──────────────────────────────────────────────────
def make_windows(dates_index, start, end, window_bars):
    """Non-overlapping 126-bar windows."""
    s_idx = next(i for i, d in enumerate(dates_index) if str(d) >= start)
    e_idx = next((i for i, d in enumerate(dates_index) if str(d) >= end), len(dates_index))
    windows = []
    while s_idx + window_bars <= e_idx:
        windows.append((dates_index[s_idx], dates_index[s_idx + window_bars - 1]))
        s_idx += window_bars
    return windows


def annualized_sharpe(net_series):
    if len(net_series) < 30 or net_series.std(ddof=1) < 1e-12:
        return np.nan
    return float(net_series.mean() / net_series.std(ddof=1) * np.sqrt(252))


def block_bootstrap_sharpe(series, n_iter=N_BOOTSTRAP, block_len=BLOCK_LEN, seed=0):
    """Stationary block bootstrap; returns array of bootstrap Sharpes."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(series)
    n = len(arr)
    if n < 30:
        return np.array([np.nan])
    n_blocks = int(np.ceil(n / block_len))
    sharpes = np.empty(n_iter)
    for b in range(n_iter):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        sample = arr[idx]
        if sample.std(ddof=1) < 1e-12:
            sharpes[b] = 0.0
        else:
            sharpes[b] = sample.mean() / sample.std(ddof=1) * np.sqrt(252)
    return sharpes


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    overall_t0 = time.time()
    print("=" * 100)
    print("PROPER EVALUATION — multi-seed × walk-forward windows × block bootstrap")
    print("=" * 100)

    # Resolve all configs (chars + signs + P)
    resolved = {}
    all_chars = set()
    for name, cfg in CONFIGS.items():
        chars, signs, P = resolve_config(name, cfg)
        resolved[name] = (chars, signs, P)
        all_chars.update(chars)
        print(f"  {name}: D={len(chars)}  P={P}", flush=True)
    print(f"\n  Total unique chars across all configs: {len(all_chars)}")

    matrices, tickers, dates, classifications = load_all_data(all_chars)
    close_vals = matrices["close"].values

    # Walk-forward windows
    windows = make_windows(dates, OOS_WINDOWS_START, OOS_WINDOWS_END, WINDOW_BARS)
    print(f"\n  {len(windows)} non-overlapping {WINDOW_BARS}-bar OOS windows:")
    for i, (s, e) in enumerate(windows):
        print(f"    W{i+1:>2d}: {s.date()} → {e.date()}", flush=True)

    # ── Run all (config, seed) combos with checkpoint/resume ────────────────
    ckpt_dir = OUT_DIR / "_seed_runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    all_pnl = {}
    for cfg_name, (chars, signs, P) in resolved.items():
        for seed in SEEDS:
            ckpt = ckpt_dir / f"{cfg_name}__seed{seed}.parquet"
            if ckpt.exists():
                df = pd.read_parquet(ckpt)
                all_pnl[(cfg_name, seed)] = df
                print(f"  RESUMED {cfg_name} seed={seed}  bars={len(df)}  (cached)", flush=True)
                continue
            t0 = time.time()
            df = run_aipt_once(matrices, tickers, dates, close_vals, classifications,
                                 chars, signs, P, seed)
            df.to_parquet(ckpt)
            all_pnl[(cfg_name, seed)] = df
            print(f"  ran {cfg_name} seed={seed}  bars={len(df)}  ({(time.time()-t0)/60:.1f}min)",
                  flush=True)

    # ── Aggregate Sharpe across windows × seeds × fee levels ────────────────
    print(f"\n[Aggregation] Sharpe per (config, seed, window, fee_bps)")
    rows = []
    for (cfg_name, seed), df in all_pnl.items():
        for win_idx, (w_s, w_e) in enumerate(windows):
            sub = df[(df["date"] >= w_s) & (df["date"] <= w_e)]
            if len(sub) < 30:
                continue
            for bps in [0, 1, 3]:
                nn = sub["gross"] - sub["turnover"] * bps / 10000.0 * 2.0
                sr = annualized_sharpe(nn)
                rows.append({"config": cfg_name, "seed": seed, "window": win_idx + 1,
                              "fee_bps": bps, "n_bars": len(sub), "sr": sr,
                              "to": sub["turnover"].mean(),
                              "ic": sub["ic_p"].mean()})
    sw_df = pd.DataFrame(rows)
    sw_df.to_csv(OUT_DIR / "per_config_seed_window.csv", index=False)

    # Per-config mean ± std across (seed, window)
    print(f"\n{'='*100}\nPer-config Sharpe distribution across {len(SEEDS)} seeds × {len(windows)} windows\n{'='*100}")
    for bps in [0, 1, 3]:
        print(f"\n--- fee = {bps} bp ---")
        agg = sw_df[sw_df.fee_bps == bps].groupby("config")["sr"].agg(
            ["mean", "std", "min", "max", "count",
             lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        agg.columns = ["mean", "std", "min", "max", "n", "q25", "q75"]
        print(agg.round(2).to_string())

    # ── Block bootstrap on per-bar series, mean across seeds ────────────────
    print(f"\n{'='*100}\nBlock bootstrap (5-day blocks, {N_BOOTSTRAP} iters) — Sharpe distribution per config\n{'='*100}")
    boot_results = {}
    for cfg_name in resolved:
        # Average per-bar gross/turnover across seeds (same dates)
        seed_dfs = [all_pnl[(cfg_name, s)] for s in SEEDS]
        # Align on bar_idx
        merged = seed_dfs[0][["bar_idx", "date"]].copy()
        for s, sd in zip(SEEDS, seed_dfs):
            merged = merged.merge(sd[["bar_idx", "gross", "turnover"]],
                                  on="bar_idx", suffixes=("", f"_s{s}"))
            if "gross" in merged.columns and f"gross_s{s}" not in merged.columns:
                # First merge: rename
                pass
        # Compute mean across seeds
        gross_cols = [c for c in merged.columns if c.startswith("gross")]
        to_cols = [c for c in merged.columns if c.startswith("turnover")]
        merged["gross_avg"] = merged[gross_cols].mean(axis=1)
        merged["to_avg"] = merged[to_cols].mean(axis=1)

        # Restrict to OOS windows period
        oos_mask = (merged["date"] >= pd.Timestamp(OOS_WINDOWS_START)) & \
                   (merged["date"] <= pd.Timestamp(OOS_WINDOWS_END))
        oos = merged[oos_mask].copy()

        boot_results[cfg_name] = {}
        for bps in [0, 1, 3]:
            nn = (oos["gross_avg"] - oos["to_avg"] * bps / 10000.0 * 2.0).values
            full_sr = annualized_sharpe(pd.Series(nn))
            bs_sharpes = block_bootstrap_sharpe(nn, seed=42)
            boot_results[cfg_name][bps] = {
                "full_sr": full_sr,
                "bs_mean": float(np.nanmean(bs_sharpes)),
                "bs_std":  float(np.nanstd(bs_sharpes)),
                "bs_2_5":  float(np.nanpercentile(bs_sharpes, 2.5)),
                "bs_97_5": float(np.nanpercentile(bs_sharpes, 97.5)),
                "n_bars":  len(nn),
            }

    for bps in [0, 1, 3]:
        print(f"\n--- fee = {bps} bp ---")
        print(f"  {'config':<22} {'full_SR':>10} {'bootstrap mean':>15} {'95% CI':>22}")
        for cfg_name, br in boot_results.items():
            r = br[bps]
            print(f"  {cfg_name:<22} {r['full_sr']:>+10.2f} {r['bs_mean']:>+15.2f} "
                  f"  [{r['bs_2_5']:+.2f}, {r['bs_97_5']:+.2f}]")

    # ── Pairwise paired-diff bootstrap ──────────────────────────────────────
    print(f"\n{'='*100}\nPairwise paired-difference bootstrap (1bp fee)\n{'='*100}")
    cfg_names = list(resolved.keys())
    paired_data = {}
    for cfg_name in cfg_names:
        sd = all_pnl[(cfg_name, 42)]   # use seed=42 reference for pairing
        sd = sd[(sd["date"] >= pd.Timestamp(OOS_WINDOWS_START)) &
                (sd["date"] <= pd.Timestamp(OOS_WINDOWS_END))].copy()
        sd["net_1bp"] = sd["gross"] - sd["turnover"] * 1 / 10000.0 * 2.0
        paired_data[cfg_name] = sd.set_index("bar_idx")["net_1bp"]

    print(f"\n  {'A vs B':<40} {'mean(A-B)':>12} {'95% CI':>22} {'sig?':>6}")
    for c1, c2 in combinations(cfg_names, 2):
        a = paired_data[c1]
        b = paired_data[c2]
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).values
        # Block bootstrap on diff series
        rng = np.random.default_rng(42)
        n = len(diff); n_blocks = int(np.ceil(n / BLOCK_LEN))
        means = np.empty(N_BOOTSTRAP)
        for b_ in range(N_BOOTSTRAP):
            starts = rng.integers(0, n - BLOCK_LEN + 1, size=n_blocks)
            idx = (starts[:, None] + np.arange(BLOCK_LEN)[None, :]).ravel()[:n]
            means[b_] = diff[idx].mean()
        mean_d = float(diff.mean())
        ci_lo = float(np.percentile(means, 2.5))
        ci_hi = float(np.percentile(means, 97.5))
        sig = "yes" if (ci_lo > 0 or ci_hi < 0) else "NO"
        print(f"  {c1[:18]:<18} vs {c2[:18]:<18}  {mean_d*1e4:>+10.3f}bp/d "
              f"[{ci_lo*1e4:+.3f}, {ci_hi*1e4:+.3f}]  {sig:>6}")

    # ── Plot: equity curves with seed bands + Sharpe boxplot ────────────────
    print(f"\n[Plotting]")
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    colors = plt.cm.tab10.colors
    for i, cfg_name in enumerate(resolved):
        # Mean cum-net @ 1bp across seeds
        all_cums = []
        all_dates_ref = None
        for s in SEEDS:
            sd = all_pnl[(cfg_name, s)]
            sd = sd[(sd["date"] >= pd.Timestamp(OOS_WINDOWS_START)) &
                    (sd["date"] <= pd.Timestamp(OOS_WINDOWS_END))].copy().reset_index(drop=True)
            sd["net"] = sd["gross"] - sd["turnover"] * 1 / 10000.0 * 2.0
            all_cums.append(sd["net"].cumsum().values * 100)
            if all_dates_ref is None:
                all_dates_ref = sd["date"]
        all_cums = np.array(all_cums)
        mean_cum = all_cums.mean(axis=0)
        lo, hi = all_cums.min(axis=0), all_cums.max(axis=0)
        axes[0].plot(all_dates_ref, mean_cum, color=colors[i], linewidth=2,
                     label=f"{cfg_name}  (mean across {len(SEEDS)} seeds)")
        axes[0].fill_between(all_dates_ref, lo, hi, color=colors[i], alpha=0.15)
    axes[0].set_title(f"Cumulative net return @ 1bp  ({OOS_WINDOWS_START} → {OOS_WINDOWS_END})  "
                       f"— mean ± seed range across {len(SEEDS)} seeds",
                       fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cumulative net return (%)")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.6, alpha=0.4)

    # Sharpe boxplot per config (across seeds × windows)
    sub = sw_df[sw_df.fee_bps == 1]
    box_data = [sub[sub.config == c]["sr"].dropna().values for c in resolved]
    bp = axes[1].boxplot(box_data, labels=list(resolved.keys()), patch_artist=True,
                          showmeans=True, meanline=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[1].set_title(f"Sharpe distribution per config @ 1bp  "
                       f"(N={len(SEEDS)*len(windows)} per config: "
                       f"{len(SEEDS)} seeds × {len(windows)} 6-month windows)",
                       fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Annualized net Sharpe")
    axes[1].axhline(0, color="black", linewidth=0.6, alpha=0.4)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_png = OUT_DIR / "proper_eval_summary.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"  PNG: {out_png}")

    # Save bootstrap results
    with open(OUT_DIR / "bootstrap_results.json", "w") as fh:
        json.dump({"configs": list(resolved.keys()),
                   "seeds": SEEDS, "n_windows": len(windows),
                   "boot_results": boot_results}, fh, indent=2)

    print(f"\nDONE in {(time.time()-overall_t0)/60:.1f} min")
    print(f"Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
