"""
eval_alpha_5m_parallel.py — In-Process Batch Alpha Evaluator (Memory-Efficient)

Problem this solves:
    Running many `python eval_alpha_5m.py --expr "..."` subprocesses causes each
    subprocess to independently load the full 5m matrix dataset, multiplied by N.
    This causes OOM crashes on Windows.

Solution:
    Load ALL matrices ONCE in this process via eval_alpha_5m._DATA_CACHE,
    then evaluate all N candidates sequentially in the SAME process.
    No inter-process data transfer. No pickling overhead. No OOM risk.
    Each candidate takes ~60-90s (same as single eval_alpha_5m.py call), but
    data is loaded only once (~17s) rather than N times.

Usage:
    # Evaluate expressions (print metrics, no save):
    python eval_alpha_5m_parallel.py --exprs "expr1" "expr2" --universe BINANCE_TOP50

    # Evaluate AND save passing alphas:
    python eval_alpha_5m_parallel.py \
        --exprs "expr1" "expr2" \
        --reasonings "Reason 1" "Reason 2" \
        --save --universe BINANCE_TOP50

    # From JSON file (list of {"expr": ..., "reasoning": ...}):
    python eval_alpha_5m_parallel.py --input candidates.json --save --universe BINANCE_TOP50

    # Dry-run (evaluate but do not write to DB):
    python eval_alpha_5m_parallel.py --exprs "..." --dry-run
"""

import sys, os, time, argparse, json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Fast IC (sampled 1/day for speed) ────────────────────────────────────────

def _compute_ic_sampled(alpha_raw, returns_pct, universe_df, bars_per_day):
    from scipy import stats as sp_stats
    signal = alpha_raw.copy()
    uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
    signal = signal.where(uni_mask, np.nan)
    signal_lagged = signal.shift(1)
    indices = signal_lagged.index[1::bars_per_day]
    ics = []
    for dt in indices:
        if dt not in returns_pct.index:
            continue
        a = signal_lagged.loc[dt]
        r = returns_pct.loc[dt]
        valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
        a_v, r_v = a[valid], r[valid]
        if len(a_v) < 10 or a_v.std() < 1e-15:
            ics.append(np.nan)
            continue
        ic, _ = sp_stats.spearmanr(a_v, r_v)
        ics.append(ic)
    return pd.Series(ics).dropna()


def eval_one(expression, ev):
    """Evaluate a single expression. Returns result dict (no DB writes)."""
    from stat_tests_5m import fama_macbeth_gate

    is_m = ev.eval_single(expression, split="train", fees_bps=0)
    if not is_m["success"]:
        return {"success": False, "error": is_m["error"]}

    _, universe = ev.load_data("train")
    returns_pct = is_m["returns_pct"]

    ic_series = _compute_ic_sampled(is_m["alpha_raw"], returns_pct, universe, ev.BARS_PER_DAY)
    ic_mean = ic_series.mean() if len(ic_series) > 0 else 0
    ic_std  = ic_series.std()  if len(ic_series) > 1 else 1
    icir    = ic_mean / ic_std if ic_std > 0 else 0

    stability = {}
    for _, _, name in ev.SUBPERIODS:
        sub = ev.eval_single(expression, split=name.lower(), fees_bps=0)
        stability[name] = sub["sharpe"] if sub["success"] else 0

    dsr = ev.deflated_sharpe_ratio(is_m["sharpe"], 1, is_m["n_bars"])

    pnl = is_m["pnl_vec"]
    pnl_s = pd.Series(pnl)
    pnl_kurt = float(pnl_s.kurtosis()) if len(pnl) > 10 else 0
    pnl_skew = float(pnl_s.skew())     if len(pnl) > 10 else 0
    rolling_sr = pnl_s.rolling(2 * ev.BARS_PER_DAY).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0
    ).dropna()
    rolling_sr_std = float(rolling_sr.std()) if len(rolling_sr) > 10 else 999

    _, fm_tstat, fm_pvalue = fama_macbeth_gate(
        is_m["alpha_raw"], returns_pct, universe, threshold=ev.MIN_FM_TSTAT
    )

    return {
        "success": True,
        "is_sharpe": is_m["sharpe"], "is_fitness": is_m["fitness"],
        "turnover": is_m["turnover"], "max_drawdown": is_m["max_drawdown"],
        "returns_ann": is_m.get("returns_ann", 0), "n_bars": is_m["n_bars"],
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "stability_h1": stability.get("H1", 0),
        "stability_h2": stability.get("H2", 0),
        "deflated_sharpe": dsr,
        "pnl_kurt": pnl_kurt, "pnl_skew": pnl_skew,
        "rolling_sr_std": rolling_sr_std,
        "fm_tstat": fm_tstat, "fm_pvalue": fm_pvalue,
        "_alpha_raw": is_m["alpha_raw"],
    }


# ─── Diversity helpers ────────────────────────────────────────────────────────

def build_diversity_cache(conn, matrices, universe_df, ev):
    rows = conn.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND universe=?",
        (ev.UNIVERSE,)
    ).fetchall()
    cache = {}
    for alpha_id, expr in rows:
        try:
            raw = ev.evaluate_expression(expr, matrices)
            if raw is None:
                continue
            processed = ev.process_signal(raw, universe_df=universe_df, max_wt=ev.MAX_WEIGHT)
            cache[alpha_id] = (expr, processed)
        except Exception:
            continue
    return cache


def check_diversity_cached(new_alpha_raw, diversity_cache, ev, universe_df):
    new_processed = ev.process_signal(new_alpha_raw, universe_df=universe_df, max_wt=ev.MAX_WEIGHT)
    for alpha_id, (expr, existing_df) in diversity_cache.items():
        common_idx  = new_processed.index.intersection(existing_df.index)
        common_cols = new_processed.columns.intersection(existing_df.columns)
        if len(common_idx) < 50 or len(common_cols) < 5:
            continue
        a = new_processed.loc[common_idx, common_cols].values.flatten()
        b = existing_df.loc[common_idx, common_cols].values.flatten()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 100:
            continue
        corr = np.corrcoef(a[mask], b[mask])[0, 1]
        if abs(corr) > ev.CORR_CUTOFF:
            print(f"  REJECTED: corr={corr:.3f} with alpha #{alpha_id} (cutoff={ev.CORR_CUTOFF})")
            print(f"            Existing: {expr[:80]}")
            return False
    return True


def save_alpha_to_db(conn, expr, reasoning, result, ev):
    c = conn.cursor()
    c.execute(
        """INSERT INTO alphas (expression, name, category, asset_class, interval, universe, source, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (expr, expr[:80], "", "crypto", ev.INTERVAL, ev.UNIVERSE, "agent1_research", reasoning),
    )
    alpha_id = c.lastrowid
    c.execute(
        """INSERT INTO evaluations (alpha_id, universe, sharpe_is, sharpe_train, return_ann,
           max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
           train_start, train_end, n_bars, evaluated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
        (
            alpha_id, ev.UNIVERSE,
            result["is_sharpe"], result["is_sharpe"],
            result.get("returns_ann", 0), result["max_drawdown"], result["turnover"],
            result["is_fitness"], result["ic_mean"], result["icir"],
            result["deflated_sharpe"], ev.TRAIN_START, ev.TRAIN_END, result.get("n_bars", 0),
        ),
    )
    conn.execute(
        "INSERT INTO trial_log (expression, universe, is_sharpe, ic_mean, saved) VALUES (?,?,?,?,1)",
        (expr, ev.UNIVERSE, result["is_sharpe"], result["ic_mean"]),
    )
    conn.commit()
    return alpha_id


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import eval_alpha_5m as ev

    parser = argparse.ArgumentParser(
        description="In-process batch 5m alpha evaluator (data loaded once, no OOM)"
    )
    parser.add_argument("--exprs",      nargs="*", default=[], help="Alpha expressions")
    parser.add_argument("--reasonings", nargs="*", default=[], help="Reasoning strings (parallel with --exprs)")
    parser.add_argument("--input",      type=str,  default=None,
                        help='JSON file: list of {"expr": ..., "reasoning": ...}')
    parser.add_argument("--save",       action="store_true", help="Save passing alphas to DB")
    parser.add_argument("--dry-run",    action="store_true", help="Evaluate but do NOT write to DB")
    parser.add_argument("--universe",   default="BINANCE_TOP100", choices=ev.ALL_UNIVERSES)
    args = parser.parse_args()

    # Apply universe config
    ev.UNIVERSE = args.universe
    ucfg = ev.UNIVERSE_CONFIG.get(ev.UNIVERSE, ev.UNIVERSE_CONFIG["BINANCE_TOP100"])
    ev.MAX_WEIGHT    = ucfg["max_weight"]
    ev.MIN_IS_SHARPE = ucfg["min_is_sharpe"]

    # Build candidate list
    candidates = []
    if args.input:
        with open(args.input) as f:
            raw = json.load(f)
        for item in raw:
            candidates.append((item["expr"], item.get("reasoning", "")))
    for i, expr in enumerate(args.exprs):
        reasoning = args.reasonings[i] if i < len(args.reasonings) else ""
        candidates.append((expr, reasoning))

    if not candidates:
        print("No candidates provided. Use --exprs or --input.")
        parser.print_help()
        return

    print(f"\n{'='*70}")
    print(f"  Batch 5m Alpha Evaluator (in-process, single data load)")
    print(f"  Universe  : {ev.UNIVERSE}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Save mode : {'YES' if args.save and not args.dry_run else 'DRY-RUN' if args.dry_run else 'NO'}")
    print(f"{'='*70}\n")

    # Load data ONCE — all evaluations reuse _DATA_CACHE
    print("  Loading 5m matrices (once for all candidates)...")
    t0 = time.time()
    matrices, universe_df = ev.load_data("train")
    print(f"  Loaded {len(matrices)} fields in {time.time()-t0:.1f}s\n")

    os.makedirs(os.path.dirname(ev.DB_PATH) or ".", exist_ok=True)
    conn = ev.get_conn()
    ev.ensure_tables(conn)

    print("  Building diversity cache...")
    t0 = time.time()
    diversity_cache = build_diversity_cache(conn, matrices, universe_df, ev)
    print(f"  Cached {len(diversity_cache)} existing alphas in {time.time()-t0:.1f}s\n")

    passed = 0
    failed = 0
    summary = []
    t_total = time.time()

    for i, (expr, reasoning) in enumerate(candidates, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(candidates)}] {reasoning or expr[:60]}")
        print(f"{'='*70}")
        t0 = time.time()

        try:
            result = eval_one(expr, ev)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            if not args.dry_run:
                conn.execute(
                    "INSERT INTO trial_log (expression, universe, is_sharpe, ic_mean, saved) VALUES (?,?,0,0,0)",
                    (expr, ev.UNIVERSE)
                )
                conn.commit()
            failed += 1
            summary.append((i, reasoning or expr[:40], "ERROR", str(e)[:80]))
            continue

        if not result["success"]:
            print(f"  ERROR: {result['error']}")
            if not args.dry_run:
                conn.execute(
                    "INSERT INTO trial_log (expression, universe, is_sharpe, ic_mean, saved) VALUES (?,?,0,0,0)",
                    (expr, ev.UNIVERSE)
                )
                conn.commit()
            failed += 1
            summary.append((i, reasoning or expr[:40], "ERROR", result["error"][:80]))
            continue

        elapsed = time.time() - t0
        both_pos = result["stability_h1"] > 0 and result["stability_h2"] > 0
        min_sub  = min(result["stability_h1"], result["stability_h2"])

        gates = [
            (result["is_sharpe"]      >= ev.MIN_IS_SHARPE,    f"SR>={ev.MIN_IS_SHARPE}: {result['is_sharpe']:+.3f}"),
            (result["turnover"]       <= ev.MAX_TURNOVER,      f"TO<={ev.MAX_TURNOVER}: {result['turnover']:.4f}"),
            (both_pos,                                         f"H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}"),
            (min_sub                  >= ev.MIN_SUB_SHARPE,   f"MinSub>={ev.MIN_SUB_SHARPE}: {min_sub:+.2f}"),
            (result["rolling_sr_std"] <= ev.MAX_ROLLING_SR_STD, f"RollSR<={ev.MAX_ROLLING_SR_STD}: {result['rolling_sr_std']:.4f}"),
            (result["pnl_skew"]       >= ev.MIN_PNL_SKEW,     f"Skew>={ev.MIN_PNL_SKEW}: {result['pnl_skew']:+.3f}"),
        ]
        all_pass = all(g[0] for g in gates)

        print(
            f"\n  IC={result['ic_mean']:+.5f} ICIR={result['icir']:.4f} "
            f"SR={result['is_sharpe']:+.3f} TO={result['turnover']:.3f} "
            f"DD={result['max_drawdown']:.3f} ({elapsed:.0f}s)"
        )
        print(
            f"  Skew={result['pnl_skew']:+.3f} Kurt={result['pnl_kurt']:.1f} "
            f"RollSR={result['rolling_sr_std']:.4f} "
            f"FM={result['fm_tstat']:+.3f} (p={result['fm_pvalue']:.4f})"
        )
        print(f"  Stability: H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}")
        for p, d in gates:
            print(f"  [{'PASS' if p else 'FAIL'}] {d}")
        print(f"  [INFO] IC (informational): {result['ic_mean']:+.5f} (ref>={ev.MIN_IC_MEAN})")
        print(f"  [INFO] Turnover (informational): {result['turnover']:.3f} (ref<={ev.MAX_TURNOVER})")
        print(f"  ALL PASS: {'YES' if all_pass else 'NO'}")

        if not args.dry_run:
            conn.execute(
                "INSERT INTO trial_log (expression, universe, is_sharpe, ic_mean, saved) VALUES (?,?,?,?,0)",
                (expr, ev.UNIVERSE, result["is_sharpe"], result["ic_mean"])
            )
            conn.commit()

        if all_pass and args.save and not args.dry_run:
            is_diverse = check_diversity_cached(result["_alpha_raw"], diversity_cache, ev, universe_df)
            if not is_diverse:
                failed += 1
                summary.append((i, reasoning or expr[:40], "REJECTED", "diversity"))
                print("  >>> REJECTED (diversity) <<<")
                continue

            existing = conn.execute(
                "SELECT id FROM alphas WHERE expression=? AND archived=0 AND universe=?",
                (expr, ev.UNIVERSE)
            ).fetchone()
            if existing:
                failed += 1
                summary.append((i, reasoning or expr[:40], "REJECTED", f"duplicate #{existing[0]}"))
                print(f"  >>> REJECTED (duplicate #{existing[0]}) <<<")
                continue

            alpha_id = save_alpha_to_db(conn, expr, reasoning, result, ev)
            new_processed = ev.process_signal(result["_alpha_raw"], universe_df=universe_df, max_wt=ev.MAX_WEIGHT)
            diversity_cache[alpha_id] = (expr, new_processed)
            passed += 1
            summary.append((i, reasoning or expr[:40], "SAVED", f"#{alpha_id} SR={result['is_sharpe']:+.3f}"))
            print(f"  SAVED as alpha #{alpha_id}")
            print("  >>> SAVED <<<")

        elif all_pass and args.dry_run:
            summary.append((i, reasoning or expr[:40], "WOULD_SAVE", f"SR={result['is_sharpe']:+.3f}"))

        elif all_pass:
            summary.append((i, reasoning or expr[:40], "PASS_NO_SAVE", f"SR={result['is_sharpe']:+.3f}"))

        else:
            failed += 1
            fail_reasons = "; ".join(d for p, d in gates if not p)
            summary.append((i, reasoning or expr[:40], "FAILED", fail_reasons))

    conn.close()

    print(f"\n{'='*70}")
    print(f"  BATCH COMPLETE: {passed} saved | {failed} failed | {len(candidates)} total")
    print(f"  Total wall time: {time.time()-t_total:.1f}s")
    print(f"{'='*70}")
    for idx, r, s, d in summary:
        print(f"    {idx:3d} [{s:12s}] {r[:60]}")
        if s not in ("SAVED", "PASS_NO_SAVE"):
            print(f"               {d[:120]}")


if __name__ == "__main__":
    main()
