"""
sweep_neutralization_ib.py — Neutralization comparison for the IB portfolio.

Approaches compared:
  Phase 1: Aggregate signal (equal-weight), neutralization applied at sim level
  Phase 2: Per-alpha individual comparison across all neutralization levels
  Phase 3: Pre-neutralize each alpha by subindustry BEFORE combining, then
            run combined signal through market/none at sim level.
            (This is the WQ-style group_neutralize() on each component)

Benchmark: market neutralization applied at sim level (status quo)
Always includes turnover in output.
"""
import os, sys, sqlite3, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIG
# ============================================================================
UNIVERSE   = "TOP2000TOP3000"
BOOKSIZE   = 20_000_000.0
MAX_WEIGHT = 0.01
DB_PATH    = "data/ib_alphas.db"

SPLITS = {
    "train": ("2016-01-01", "2023-01-01"),
    "val":   ("2023-01-01", "2024-07-01"),
    "test":  ("2024-07-01", None),
}

NEUTRALIZATIONS = ["none", "market", "sector", "industry", "subindustry"]

# IBKR Pro per-share commission rates (USD) — matches run_ib_portfolio.py
IBKR_TIERED_RATES = {
    "Tiered >100M/mo  $0.0005/sh": 0.0005,
    "Tiered 20M-100M  $0.0010/sh": 0.0010,
    "Tiered 3M-20M    $0.0015/sh": 0.0015,
    "Tiered 300k-3M   $0.0020/sh": 0.0020,
    "Tiered <=300k    $0.0035/sh": 0.0035,
}
IBKR_FIXED_RATE = 0.0050  # $0.005/share fixed

# ============================================================================
# HELPERS
# ============================================================================
def compute_ibkr_fees(matrices, universe_df):
    """Compute IBKR effective bps from median universe stock price."""
    close_in_uni = matrices["close"].where(
        universe_df.reindex(index=matrices["close"].index,
                            columns=matrices["close"].columns).fillna(False).astype(bool)
    )
    avg_price = float(close_in_uni.stack().median())
    est_daily_shares   = BOOKSIZE * 1.1 / avg_price
    est_monthly_shares = est_daily_shares * 21

    print(f"  Median universe price: ${avg_price:.2f}")
    print(f"  Est. monthly share vol: {est_monthly_shares/1e6:.1f}M shares")

    fee_schedule = {"0 bps (fee-free)": 0.0}
    for label, rate in IBKR_TIERED_RATES.items():
        bps = (rate / avg_price) * 10000
        fee_schedule[f"IBKR {label} ({bps:.2f}bps)"] = bps
    fixed_bps = (IBKR_FIXED_RATE / avg_price) * 10000
    fee_schedule[f"IBKR Fixed $0.005/sh ({fixed_bps:.2f}bps)"] = fixed_bps

    # Realistic tier based on monthly volume
    realistic_rate = None
    for (lo, hi), rate in [
        ((300_000, None), 0.0035),
        ((3_000_000, 300_000), 0.0020),
        ((20_000_000, 3_000_000), 0.0015),
        ((100_000_000, 20_000_000), 0.0010),
        ((None, 100_000_000), 0.0005),
    ]:
        if lo is None or est_monthly_shares < lo:
            realistic_rate = rate
    realistic_bps = (realistic_rate / avg_price) * 10000 if realistic_rate else list(fee_schedule.values())[-2]
    print(f"  Realistic IBKR tier: ${realistic_rate:.4f}/sh = {realistic_bps:.2f}bps")

    return fee_schedule, avg_price, est_monthly_shares, realistic_bps


def load_data():
    import eval_alpha_ib
    eval_alpha_ib.UNIVERSE  = UNIVERSE
    eval_alpha_ib.NEUTRALIZE = "market"
    matrices, universe, classifications = eval_alpha_ib.load_data("full")
    return matrices, universe, classifications


def load_alphas():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, expression FROM alphas WHERE universe = 'TOP2000TOP3000' ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def run_sim(alpha_df, matrices, universe_df, classifications, neutralization, fees_bps=0.0):
    from src.simulation.vectorized_sim_polars import simulate_vectorized_polars
    return simulate_vectorized_polars(
        alpha_df=alpha_df, returns_df=matrices["returns"], close_df=matrices["close"],
        universe_df=universe_df, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
        decay=0, delay=0, neutralization=neutralization, fees_bps=fees_bps,
        bars_per_day=1, classifications=classifications,
    )


def group_demean(signal_df, groups_series):
    """Apply group-level demeaning row-wise (like WQ group_neutralize)."""
    result = signal_df.copy()
    common_cols = signal_df.columns.intersection(groups_series.index)
    grp = groups_series.reindex(common_cols)
    for g in grp.dropna().unique():
        col_mask = (grp == g).values
        cols_in_grp = common_cols[col_mask]
        block = result[cols_in_grp]
        grp_mean = block.mean(axis=1)
        result[cols_in_grp] = block.sub(grp_mean, axis=0)
    return result


def process_signal(alpha_df, universe_df, pre_neut_groups=None):
    """
    Standard signal processing.
    If pre_neut_groups is provided, apply group demeaning BEFORE market demean.
    """
    signal = alpha_df.copy()
    uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False).astype(bool)
    signal = signal.where(uni_mask, np.nan)

    if pre_neut_groups is not None:
        signal = group_demean(signal, pre_neut_groups)

    # Market demean
    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT)
    return signal.fillna(0.0)


def build_equal_weight(signals, ref_idx, ref_cols, pre_neut_groups=None, universe_df=None):
    """Equal-weight composite. Optionally pre-neutralizes each alpha before combining."""
    stack = np.zeros((len(ref_idx), len(ref_cols)), dtype=np.float64)
    count = np.zeros_like(stack)
    for sig in signals.values():
        s = sig.reindex(index=ref_idx, columns=ref_cols)
        if pre_neut_groups is not None and universe_df is not None:
            # Apply group demeaning per-alpha before stacking
            uni_mask = universe_df.reindex(index=s.index, columns=s.columns).fillna(False).astype(bool)
            s = s.where(uni_mask, np.nan)
            s = group_demean(s, pre_neut_groups)
        vals = s.values.copy()
        mask = ~np.isnan(vals)
        stack[mask] += vals[mask]
        count[mask] += 1.0
    count[count == 0] = np.nan
    return pd.DataFrame(stack / count, index=ref_idx, columns=ref_cols)


def slice_split(df, start, end):
    return df.loc[start:end] if end else df.loc[start:]


def print_hdr(label):
    print(f"\n{'='*110}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"  {'Approach':<40} | {'Split':<6} | {'Sharpe':>8} {'AnnRet%':>8} {'Turnover':>9} {'MaxDD%':>7} {'Fitness':>7}", flush=True)
    print(f"  {'-'*95}", flush=True)


def print_row(label, split, sim, marker=""):
    print(
        f"  {label:<40} | {split:<6} | {sim.sharpe:+8.3f} {sim.returns_ann*100:+7.2f}%"
        f" {sim.turnover:9.4f} {sim.max_drawdown*100:6.2f}% {sim.fitness:7.2f}{marker}",
        flush=True
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    print("Loading data...", flush=True)
    matrices, full_universe, classifications = load_data()
    alphas = load_alphas()
    print(f"  {len(alphas)} alphas | universe tickers: {len(full_universe.columns)}", flush=True)

    print("Computing IBKR fee schedule...", flush=True)
    fee_schedule, avg_price, est_monthly_shares, realistic_bps = compute_ibkr_fees(matrices, full_universe)
    fees_bps = realistic_bps  # Use realistic tier throughout
    print(f"  Using {fees_bps:.3f}bps for all simulations", flush=True)

    # Evaluate signals
    print("Evaluating alpha signals...", flush=True)
    from eval_alpha_ib import evaluate_expression
    signals = {}
    for aid, expr in alphas:
        try:
            sig = evaluate_expression(expr, matrices)
            if sig is not None and not sig.empty:
                signals[aid] = sig
        except Exception:
            pass
    print(f"  {len(signals)}/{len(alphas)} valid", flush=True)
    if not signals:
        print("ERROR: no valid signals"); return

    # Reference index/columns for combine
    ref_idx  = list(signals.values())[0].index
    ref_cols = list(signals.values())[0].columns

    # Subindustry groups series (for pre-neutralization)
    sub_groups = classifications.get("subindustry")  # pd.Series ticker->group_id

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 1: Aggregate → sim-level neutralization sweep
    # ──────────────────────────────────────────────────────────────────────────
    print_hdr("PHASE 1: AGGREGATE equal-weight → neutralization at sim level")
    agg_raw = build_equal_weight(signals, ref_idx, ref_cols)
    agg_proc = process_signal(agg_raw, full_universe)

    all_results_p1 = {}
    for neut in NEUTRALIZATIONS:
        for split_name in args.splits:
            start, end = SPLITS[split_name]
            agg_sl  = slice_split(agg_proc, start, end)
            uni_sl  = slice_split(full_universe, start, end)
            mats_sl = {k: slice_split(v, start, end) for k, v in matrices.items()}
            try:
                sim = run_sim(agg_sl, mats_sl, uni_sl, classifications, neut, fees_bps=fees_bps)
                all_results_p1[(neut, split_name)] = sim
                marker = " ◄ BASELINE" if neut == "market" and split_name == "test" else ""
                print_row(f"post-combine neut={neut}", split_name, sim, marker)
            except Exception as e:
                print(f"  {'post-combine neut='+neut:<40} | {split_name:<6} | ERROR: {e}", flush=True)
        print(f"  {'-'*95}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: Pre-neutralize each alpha by subindustry, THEN combine
    # ──────────────────────────────────────────────────────────────────────────
    print_hdr("PHASE 3: Pre-neutralize each alpha by SUBINDUSTRY → then combine → sim-level sweep")

    if sub_groups is None:
        print("  WARNING: subindustry classifications not available — skipping Phase 3", flush=True)
    else:
        agg_pre_raw = build_equal_weight(signals, ref_idx, ref_cols,
                                         pre_neut_groups=sub_groups,
                                         universe_df=full_universe)
        # After pre-neutralizing per-alpha, apply final market demean + scale
        agg_pre_proc = process_signal(agg_pre_raw, full_universe)

        for neut in ["none", "market", "subindustry"]:
            for split_name in args.splits:
                start, end = SPLITS[split_name]
                agg_sl  = slice_split(agg_pre_proc, start, end)
                uni_sl  = slice_split(full_universe, start, end)
                mats_sl = {k: slice_split(v, start, end) for k, v in matrices.items()}
                try:
                    sim = run_sim(agg_sl, mats_sl, uni_sl, classifications, neut, fees_bps=fees_bps)
                    label = f"pre-sub→combine→sim={neut}"
                    print_row(label, split_name, sim)
                except Exception as e:
                    print(f"  ERROR {neut}/{split_name}: {e}", flush=True)
            print(f"  {'-'*95}", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # COMPARISON SUMMARY (test split only)
    # ──────────────────────────────────────────────────────────────────────────
    print_hdr("SUMMARY — Test Split Only (ranked by Sharpe)")
    rows = []

    # Phase 1 results
    for neut in NEUTRALIZATIONS:
        sim = all_results_p1.get((neut, "test"))
        if sim:
            rows.append((f"post-combine neut={neut}", sim.sharpe, sim.returns_ann*100,
                         sim.turnover, sim.max_drawdown*100, sim.fitness))

    # Phase 3 results (if ran)
    if sub_groups is not None:
        for neut in ["none", "market", "subindustry"]:
            start, end = SPLITS["test"]
            agg_sl  = slice_split(agg_pre_proc, start, end)
            uni_sl  = slice_split(full_universe, start, end)
            mats_sl = {k: slice_split(v, start, end) for k, v in matrices.items()}
            try:
                sim = run_sim(agg_sl, mats_sl, uni_sl, classifications, neut, fees_bps=fees_bps)
                rows.append((f"pre-sub→combine→sim={neut}", sim.sharpe, sim.returns_ann*100,
                             sim.turnover, sim.max_drawdown*100, sim.fitness))
            except Exception:
                pass

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"  {'Approach':<42} | {'Sharpe':>8} {'AnnRet%':>8} {'Turnover':>9} {'MaxDD%':>7} {'Fitness':>7}", flush=True)
    print(f"  {'-'*97}", flush=True)
    baseline_sr = next((r[1] for r in rows if "market" in r[0] and "post-combine" in r[0]), 0)
    for approach, sr, ret, to, dd, fit in rows:
        delta = sr - baseline_sr
        marker = " ◄ BASELINE" if "post-combine neut=market" in approach else f" (Δ{delta:+.2f})"
        print(f"  {approach:<42} | {sr:+8.3f} {ret:+7.2f}% {to:9.4f} {dd:6.2f}% {fit:7.2f}{marker}", flush=True)


if __name__ == "__main__":
    main()
