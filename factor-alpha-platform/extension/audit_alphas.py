"""
Audit ALL 190 saved alphas:
  1. Pairwise correlation matrix on processed signals (TRAIN period).
     Flag any pair > CORR_CUTOFF (= 0.70).
  2. Re-evaluate every alpha through the 9 quality gates from eval_alpha.py:
       MIN_IS_SHARPE 3.0  /  MIN_FITNESS 5.0  /  MIN_IC_MEAN -0.05
       turnover <= 0.30   /  min sub-Sharpe >= 1.0  /  PnL kurtosis <= 20
       rolling SR std <= 0.05  /  PnL skew >= -0.5  /  both halves > 0
  3. Report failures (which alphas no longer pass given current data state).

Outputs:
   data/alphas_audit/pairwise_corr.csv      — full corr matrix
   data/alphas_audit/violations.csv         — pairs exceeding CORR_CUTOFF
   data/alphas_audit/gate_audit.csv         — per-alpha gate pass/fail with metrics
   data/alphas_audit/audit_summary.json     — top-line stats
"""
from __future__ import annotations
import sys, json, sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)   # eval_alpha uses relative DB/matrix paths

import eval_alpha as ea
ea.DB_PATH = str(ROOT / "data/alphas.db")  # absolute for safety
OUT_DIR = ROOT / "data/alphas_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("ALPHA AUDIT — pairwise correlation + per-alpha gate re-evaluation")
    print(f"  CORR_CUTOFF = {ea.CORR_CUTOFF}")
    print(f"  Gates: IS_SR>={ea.MIN_IS_SHARPE}, FIT>={ea.MIN_FITNESS}, IC>={ea.MIN_IC_MEAN}, "
          f"TO<={ea.MAX_TURNOVER}, SUB_SR>={ea.MIN_SUB_SHARPE}, KURT<={ea.MAX_PNL_KURTOSIS}, "
          f"ROLL_SR_STD<={ea.MAX_ROLLING_SR_STD}, SKEW>={ea.MIN_PNL_SKEW}")
    print("=" * 80)

    # ── 1. Load all alphas (KuCoin 4h) ─────────────────────────────────────
    # Patch eval_alpha to point at KuCoin (the alphas live there)
    ea.EXCHANGE = "kucoin"
    ea.UNIVERSE = "KUCOIN_TOP100"
    ea.INTERVAL = "4h"
    # Reset data cache so we load KuCoin data
    ea._DATA_CACHE.clear()

    con = sqlite3.connect(ea.DB_PATH)
    rows = con.execute(
        "SELECT id, expression FROM alphas WHERE archived=0 AND interval=? AND universe=? ORDER BY id",
        (ea.INTERVAL, ea.UNIVERSE)
    ).fetchall()
    con.close()
    print(f"\n  Loaded {len(rows)} alphas")

    # ── 2. Evaluate each on TRAIN data, run process_signal, store flat array ──
    print(f"\n[1/3] Evaluating + processing each alpha on TRAIN data...")
    matrices, universe = ea.load_data("train")
    processed = {}
    eval_failures = []
    t0 = time.time()
    for aid, expr in rows:
        try:
            raw = ea.evaluate_expression(expr, matrices)
            if raw is None or raw.empty:
                eval_failures.append((aid, "empty raw"))
                continue
            proc = ea.process_signal(raw, universe_df=universe, max_wt=ea.MAX_WEIGHT)
            processed[aid] = (expr, proc)
        except Exception as e:
            eval_failures.append((aid, f"{type(e).__name__}: {e}"))
        if len(processed) % 25 == 0:
            print(f"    {len(processed)}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  ✓ Processed {len(processed)}/{len(rows)} ({time.time()-t0:.1f}s)")
    if eval_failures:
        print(f"  ✗ Eval failures ({len(eval_failures)}):")
        for aid, err in eval_failures[:10]:
            print(f"      #{aid}: {err}")

    # ── 3. Pairwise correlation matrix on flat values ─────────────────────────
    print(f"\n[2/3] Computing pairwise correlations on processed signals...")
    aids = sorted(processed.keys())
    N = len(aids)
    # Stack all signals into (N, T*K) flattened matrix; align indices/cols
    common_idx = None
    common_cols = None
    for aid in aids:
        _, p = processed[aid]
        common_idx = p.index if common_idx is None else common_idx.intersection(p.index)
        common_cols = p.columns if common_cols is None else common_cols.intersection(p.columns)
    print(f"    common shape: {len(common_idx)} dates × {len(common_cols)} tickers", flush=True)

    flat = np.empty((N, len(common_idx) * len(common_cols)), dtype=np.float64)
    for i, aid in enumerate(aids):
        _, p = processed[aid]
        flat[i] = p.loc[common_idx, common_cols].values.flatten()

    # Mask common-NaN per pair: build correlation pairwise (skip NaN)
    print(f"    computing {N}x{N} corr (vectorized)...", flush=True)
    # Replace NaN with 0 only for finiteness check; use ma for pairwise
    fin = np.isfinite(flat)
    # For speed: use pandas DataFrame and corr() pairwise with min_periods
    df = pd.DataFrame(flat.T, columns=aids)
    corr_mat = df.corr(min_periods=1000)  # min 1000 finite obs per pair
    corr_mat.to_csv(OUT_DIR / "pairwise_corr.csv")

    # Violations
    violations = []
    for i in range(N):
        for j in range(i + 1, N):
            c = corr_mat.iloc[i, j]
            if pd.notna(c) and abs(c) > ea.CORR_CUTOFF:
                violations.append({"alpha_a": aids[i], "alpha_b": aids[j],
                                    "corr": float(c),
                                    "expr_a": processed[aids[i]][0][:80],
                                    "expr_b": processed[aids[j]][0][:80]})
    if violations:
        v_df = pd.DataFrame(violations).sort_values("corr", key=lambda x: x.abs(), ascending=False)
        v_df.to_csv(OUT_DIR / "violations.csv", index=False)
        print(f"  WARN {len(violations)} pairs exceed CORR_CUTOFF={ea.CORR_CUTOFF}")
        print(f"  Top 10 worst:")
        for _, r in v_df.head(10).iterrows():
            print(f"    #{int(r['alpha_a'])} vs #{int(r['alpha_b'])}: corr={float(r['corr']):+.3f}")
    else:
        print(f"  ✓ All pairs within |corr| <= {ea.CORR_CUTOFF}")

    # ── 4. Per-alpha gate re-evaluation ───────────────────────────────────────
    print(f"\n[3/3] Re-evaluating each alpha through all 9 gates...")
    gate_rows = []
    t0 = time.time()
    n_done = 0
    audit_conn = sqlite3.connect(ea.DB_PATH)
    for aid, (expr, proc) in processed.items():
        try:
            # Use eval_alpha's full evaluator
            result = ea.eval_full(expr, audit_conn)
            if not result.get("success"):
                gate_rows.append({"id": aid, "expression": expr, "all_pass": False,
                                   "n_gates_pass": -1, "error": result.get("error", "?")})
                continue
            min_sub = min(result['stability_h1'], result['stability_h2'])
            both_pos = result['stability_h1'] > 0 and result['stability_h2'] > 0
            gates = {
                "g_is_sharpe": result['is_sharpe'] >= ea.MIN_IS_SHARPE,
                "g_fitness":   result['is_fitness'] >= ea.MIN_FITNESS,
                "g_ic":        result['ic_mean'] >= ea.MIN_IC_MEAN,
                "g_stable":    both_pos,
                "g_turnover":  result['turnover'] <= ea.MAX_TURNOVER,
                "g_sub_sr":    min_sub >= ea.MIN_SUB_SHARPE,
                "g_kurtosis":  result['pnl_kurtosis'] <= ea.MAX_PNL_KURTOSIS,
                "g_roll_std":  result['rolling_sr_std'] <= ea.MAX_ROLLING_SR_STD,
                "g_skew":      result['pnl_skew'] >= ea.MIN_PNL_SKEW,
            }
            n_pass = sum(gates.values())
            row = {
                "id": aid, "expression": expr,
                "is_sharpe":     result['is_sharpe'],
                "is_fitness":    result['is_fitness'],
                "ic_mean":       result['ic_mean'],
                "turnover":      result['turnover'],
                "min_sub_sr":    min_sub,
                "pnl_kurtosis":  result['pnl_kurtosis'],
                "rolling_sr_std":result['rolling_sr_std'],
                "pnl_skew":      result['pnl_skew'],
                "h1_sr":         result['stability_h1'],
                "h2_sr":         result['stability_h2'],
                "n_gates_pass":  n_pass,
                "all_pass":      n_pass == 9,
                **gates,
            }
            gate_rows.append(row)
        except Exception as e:
            gate_rows.append({"id": aid, "expression": expr, "all_pass": False,
                               "n_gates_pass": -1, "error": str(e)})
        n_done += 1
        if n_done % 20 == 0:
            print(f"    {n_done}/{len(processed)}  ({time.time()-t0:.0f}s)", flush=True)

    g_df = pd.DataFrame(gate_rows)
    g_df.to_csv(OUT_DIR / "gate_audit.csv", index=False)

    pass_all = int(g_df["all_pass"].sum()) if "all_pass" in g_df else 0
    print(f"\n  Pass all 9 gates: {pass_all}/{len(g_df)}")
    if "n_gates_pass" in g_df:
        for n_pass in sorted(g_df["n_gates_pass"].unique(), reverse=True):
            count = (g_df["n_gates_pass"] == n_pass).sum()
            if n_pass == -1:
                print(f"    eval errored: {count}")
            else:
                print(f"    {n_pass}/9 gates pass: {count} alphas")

    # Per-gate failure counts
    print(f"\n  Gate failure breakdown:")
    for gate in ["g_is_sharpe","g_fitness","g_ic","g_stable","g_turnover","g_sub_sr",
                 "g_kurtosis","g_roll_std","g_skew"]:
        if gate in g_df:
            n_fail = (~g_df[gate].fillna(False).astype(bool)).sum()
            print(f"    {gate:<14} fail = {n_fail:>3}")

    # Save summary
    summary = {
        "n_alphas":           len(rows),
        "n_processed":        len(processed),
        "n_eval_failed":      len(eval_failures),
        "corr_cutoff":        ea.CORR_CUTOFF,
        "n_corr_violations":  len(violations),
        "n_pass_all_gates":   pass_all,
        "thresholds": {
            "MIN_IS_SHARPE": ea.MIN_IS_SHARPE, "MIN_FITNESS": ea.MIN_FITNESS,
            "MIN_IC_MEAN":   ea.MIN_IC_MEAN,   "MAX_TURNOVER": ea.MAX_TURNOVER,
            "MIN_SUB_SHARPE":ea.MIN_SUB_SHARPE,"MAX_PNL_KURTOSIS": ea.MAX_PNL_KURTOSIS,
            "MAX_ROLLING_SR_STD": ea.MAX_ROLLING_SR_STD, "MIN_PNL_SKEW": ea.MIN_PNL_SKEW,
        },
    }
    json.dump(summary, open(OUT_DIR / "audit_summary.json", "w"), indent=2)
    print(f"\n## DONE — outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
