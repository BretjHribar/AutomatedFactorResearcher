"""Revision 5: persistent MOC auction-pressure states.

Revisions 1-4 show the strongest component is the auction-pressure state.
Fundamental anchors created some saves but also caused high correlation or
fitness misses. This revision rejects the "anchor required" sub-hypothesis and
tests pure pressure-state observables in the same dynamic universe.

No parameter grids are run. Numeric choices are fixed:
- 20-day monthly baselines for volume/dollar/range state
- 3-day settlement-pressure reversal
- 5-day weekly drawdown location
- 3-day persistence from the prior diagnosed turnover issue
"""
from __future__ import annotations

import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import eval_alpha_ib as harness  # noqa: E402
import run_clean_workflow as base  # noqa: E402


REVISION_CSV = base.OUT_DIR / "revision_5_pressure_state_hypotheses.csv"
FIXED_PERSISTENCE_DAYS = 3


CANDIDATES = [
    (
        "pressure_add_vol_vwap_rev3",
        "rank(decay_linear(add(add(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Monthly abnormal share participation, below-VWAP close, and 3-day settlement pressure should jointly identify forced MOC supply.",
    ),
    (
        "pressure_add_dollar_vwap_rev3",
        "rank(decay_linear(add(add(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Dollar participation version of the same pressure state captures notional stress rather than share-count stress.",
    ),
    (
        "pressure_mult_vol_vwap_rev3",
        "rank(decay_linear(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Conjunctive pressure state: participation, below-VWAP location, and short reversal must all be present.",
    ),
    (
        "pressure_mult_dollar_vwap_rev3",
        "rank(decay_linear(multiply(multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Conjunctive notional-pressure version of the forced auction supply state.",
    ),
    (
        "pressure_vol_weekly_drawdown",
        "rank(decay_linear(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001)))), 3))",
        "Abnormal participation into a weak weekly location should reverse as auction supply is absorbed.",
    ),
    (
        "pressure_dollar_weekly_drawdown",
        "rank(decay_linear(multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001)))), 3))",
        "Notional participation into a weak weekly location should capture forced liquidation pressure.",
    ),
    (
        "pressure_vol_daily_close_low",
        "rank(decay_linear(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(true_divide(subtract(high, close), add(subtract(high, low), 0.001)))), 3))",
        "Abnormal participation with a close near the daily low should identify same-session auction exhaustion.",
    ),
    (
        "pressure_dollar_daily_close_low",
        "rank(decay_linear(multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(true_divide(subtract(high, close), add(subtract(high, low), 0.001)))), 3))",
        "Notional participation with a close near the daily low should identify auction exhaustion.",
    ),
    (
        "pressure_vol_intraday_down_vwap",
        "rank(decay_linear(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "A high-participation open-to-close selloff below VWAP should reverse as MOC supply clears.",
    ),
    (
        "pressure_dollar_intraday_down_vwap",
        "rank(decay_linear(multiply(multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "Notional pressure version of the high-participation intraday selloff below VWAP.",
    ),
    (
        "pressure_vol_gapdown_vwap",
        "rank(decay_linear(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(subtract(close, delay(close, 1)), add(delay(close, 1), 0.001))))), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "Abnormal participation in a close-to-close selloff below VWAP should capture forced pressure.",
    ),
    (
        "pressure_dollar_gapdown_vwap",
        "rank(decay_linear(multiply(multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(true_divide(subtract(close, delay(close, 1)), add(delay(close, 1), 0.001))))), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "Notional close-to-close selloff pressure below VWAP.",
    ),
    (
        "pressure_vol_accel_vwap_rev3",
        "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, add(sma(volume, 20), 1.0)), 3)), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Participation acceleration into below-VWAP short reversal should identify fresh forced supply.",
    ),
    (
        "pressure_dollar_accel_vwap_rev3",
        "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0)), 3)), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ts_delta(close, 3)))), 3))",
        "Dollar participation acceleration version of fresh forced supply.",
    ),
    (
        "pressure_zvol_vwap",
        "rank(decay_linear(multiply(rank(ts_zscore(volume, 20)), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "A monthly volume z-score with below-VWAP location captures unusual auction participation pressure.",
    ),
    (
        "pressure_zdollar_vwap",
        "rank(decay_linear(multiply(rank(ts_zscore(dollars_traded, 20)), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "A monthly dollar-volume z-score with below-VWAP location captures unusual notional auction pressure.",
    ),
    (
        "pressure_range_surge_vwap",
        "rank(decay_linear(multiply(rank(true_divide(subtract(high, low), add(sma(subtract(high, low), 20), 0.001))), rank(negative(true_divide(close, add(vwap, 0.001))))), 3))",
        "Range expansion with a below-VWAP close should indicate liquidation-style pressure.",
    ),
    (
        "pressure_range_surge_rev3",
        "rank(decay_linear(multiply(rank(true_divide(subtract(high, low), add(sma(subtract(high, low), 20), 0.001))), rank(negative(ts_delta(close, 3)))), 3))",
        "Range expansion with short settlement pressure should reverse as supply is absorbed.",
    ),
    (
        "pressure_vwap_reversal3_soft",
        "rank(decay_linear(add(rank(negative(true_divide(close, add(vwap, 0.001)))), rank(negative(ts_delta(close, 3)))), 3))",
        "Below-VWAP and short settlement-pressure signals should add as a pure pressure score.",
    ),
    (
        "pressure_vwap_weekly_soft",
        "rank(decay_linear(add(rank(negative(true_divide(close, add(vwap, 0.001)))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001)))), 3))",
        "Below-VWAP and weak weekly location should add as a pressure-location score.",
    ),
    (
        "pressure_vol_reversal3_soft",
        "rank(decay_linear(add(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(ts_delta(close, 3)))), 3))",
        "Participation pressure and short settlement reversal should add without requiring a VWAP term.",
    ),
    (
        "pressure_dollar_reversal3_soft",
        "rank(decay_linear(add(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(ts_delta(close, 3)))), 3))",
        "Notional participation pressure and short settlement reversal should add without requiring a VWAP term.",
    ),
]


REVISION_HYPOTHESES = [
    base.Hypothesis(
        name,
        "persistent_auction_pressure_state",
        expr,
        mechanism,
        "Pure pressure-state observable after rejecting the anchor-required sub-hypothesis; no alternate numeric parameters are tested.",
    )
    for name, expr, mechanism in CANDIDATES
]


def write_revision_hypotheses(blocklist: set[str]) -> None:
    if REVISION_CSV.exists():
        REVISION_CSV.unlink()
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        base.append_csv(
            REVISION_CSV,
            {
                "revision_order": i,
                "fixed_persistence_days": FIXED_PERSISTENCE_DAYS,
                "name": h.name,
                "family": h.family,
                "exact_blocked_before_revision": int(h.expression.strip() in blocklist),
                "expression": h.expression,
                "mechanism": h.mechanism,
                "orthogonal_reason": h.orthogonal_reason,
            },
        )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    base.configure_harness()
    base.ensure_assets()
    conn = harness.get_conn()
    blocklist = base.load_exact_blocklist()
    write_revision_hypotheses(blocklist)

    data_notes = [
        "Revision 5 was triggered because anchored variants topped out at three unique saves under the 0.70 corr gate.",
        "The revision rejects the anchor-required sub-hypothesis and tests pure persistent MOC pressure states.",
        "No parameter grids are introduced: monthly baselines are 20, settlement reversal is 3, weekly drawdown is 5, persistence is 3.",
        "All revision candidates are train-only and use subindustry neutralization.",
    ]
    base.write_research_log(conn, {"revision_5_reset": 0}, "revision_5_running", data_notes)

    print(f"Revision 5 label: {base.LABEL}", flush=True)
    print(f"Revision hypotheses: {len(REVISION_HYPOTHESES)}", flush=True)
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        if base.saved_count(conn) >= base.TARGET_SAVED:
            break
        print(f"[rev5 {i}/{len(REVISION_HYPOTHESES)}] {h.name}", flush=True)
        base.write_progress(conn, {"status": "revision_5_screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev5_exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=base.UNIVERSE)
        except Exception as exc:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev5_screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev5_screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "rev5_screen_pass" if base.screen_pass(screen) else "rev5_screen_fail"
        base.append_csv(base.TRIALS_CSV, base.row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "rev5_screen_pass":
            continue

        base.write_progress(conn, {"status": "revision_5_full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            base.append_csv(base.FULL_CSV, base.row_for(h, "rev5_full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "rev5_gate_pass" if base.gates_pass(full) else "rev5_gate_fail"
        base.append_csv(base.FULL_CSV, base.row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "rev5_gate_pass":
            continue

        reasoning = (
            f"[{base.LABEL}] Revision 5 of {base.STRATEGY_NAME}. Candidate={h.name}; "
            f"family={h.family}; train-only mechanism={h.mechanism}; "
            f"orthogonal reason={h.orthogonal_reason}; fixed dynamic universe={base.UNIVERSE}; "
            f"strict gates: train Sharpe>5, train fitness>5, turnover<=1.0, IC>0; "
            f"subindustry neutralization; exact-copy blocklist; corr<=0.70; "
            f"fixed persistence days={FIXED_PERSISTENCE_DAYS}; no parameter sweeps."
        )
        saved = harness.save_alpha(conn, h.expression, reasoning, full, base.LABEL)
        alpha_id = base.alpha_id_for_expr(conn, h.expression) if saved else None
        base.append_csv(
            base.SELECTION_CSV,
            base.row_for(h, "rev5_selection_saved" if saved else "rev5_selection_corr_reject", bool(saved), alpha_id, full),
        )
        base.write_saved_snapshot(conn)
        base.write_research_log(conn, {"revision_5_reset": 0}, "revision_5_running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={base.saved_count(conn)}/{base.TARGET_SAVED}",
            flush=True,
        )
        base.write_progress(
            conn,
            {
                "status": "revision_5_saved" if saved else "revision_5_corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    base.write_saved_snapshot(conn)
    final = "completed_target" if base.saved_count(conn) >= base.TARGET_SAVED else "revision_5_exhausted"
    base.write_progress(conn, {"status": final, "hypothesis_index": len(REVISION_HYPOTHESES)})
    base.write_research_log(conn, {"revision_5_reset": 0}, final, data_notes)
    saved_n = base.saved_count(conn)
    print(f"Revision 5 final status: {final}; saved {saved_n}/{base.TARGET_SAVED}", flush=True)
    conn.close()
    if saved_n < base.TARGET_SAVED:
        raise SystemExit(f"Only saved {saved_n} alphas; target is {base.TARGET_SAVED}.")


if __name__ == "__main__":
    main()
