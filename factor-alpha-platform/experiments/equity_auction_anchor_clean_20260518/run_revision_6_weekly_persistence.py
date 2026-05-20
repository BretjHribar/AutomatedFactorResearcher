"""Revision 6: weekly persistence for pressure states.

Revision 5 found many pressure states with Sharpe near/above 5 but fitness
limited by turnover. The user noted low-volume equity pressure should decay
over longer horizons. This revision applies one fixed weekly persistence:
decay_linear(..., 5). No alternate persistence lengths are tested.
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
import run_revision_5_pressure_states as rev5  # noqa: E402


REVISION_CSV = base.OUT_DIR / "revision_6_weekly_persistence_hypotheses.csv"
FIXED_PERSISTENCE_DAYS = 5


def to_weekly(expr: str) -> str:
    old = "rank(decay_linear("
    suffix = ", 3))"
    expr = expr.strip()
    if not (expr.startswith(old) and expr.endswith(suffix)):
        raise ValueError(f"Unexpected pressure expression: {expr}")
    core = expr[len(old) : -len(suffix)]
    return f"rank(decay_linear({core}, {FIXED_PERSISTENCE_DAYS}))"


REVISION_HYPOTHESES = [
    base.Hypothesis(
        f"persist5_{h.name}",
        "weekly_persistent_auction_pressure_state",
        to_weekly(h.expression),
        h.mechanism + " Revision 6 trades the pressure state as a fixed weekly absorption signal.",
        "Weekly persistence is a single hypothesis revision from the turnover/fitness failure; no alternate persistence lengths are tested.",
    )
    for h in rev5.REVISION_HYPOTHESES
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
        "Revision 6 was triggered because Revision 5 produced many Sharpe-positive pressure states with fitness limited by turnover.",
        "The revision applies one fixed weekly persistence, decay_linear(..., 5), based on low-volume auction-pressure absorption.",
        "No alternate persistence lengths were tested.",
        "All revision candidates are train-only and use subindustry neutralization.",
    ]
    base.write_research_log(conn, {"revision_6_reset": 0}, "revision_6_running", data_notes)

    print(f"Revision 6 label: {base.LABEL}", flush=True)
    print(f"Revision hypotheses: {len(REVISION_HYPOTHESES)}", flush=True)
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        if base.saved_count(conn) >= base.TARGET_SAVED:
            break
        print(f"[rev6 {i}/{len(REVISION_HYPOTHESES)}] {h.name}", flush=True)
        base.write_progress(conn, {"status": "revision_6_screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev6_exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=base.UNIVERSE)
        except Exception as exc:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev6_screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev6_screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "rev6_screen_pass" if base.screen_pass(screen) else "rev6_screen_fail"
        base.append_csv(base.TRIALS_CSV, base.row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "rev6_screen_pass":
            continue

        base.write_progress(conn, {"status": "revision_6_full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            base.append_csv(base.FULL_CSV, base.row_for(h, "rev6_full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "rev6_gate_pass" if base.gates_pass(full) else "rev6_gate_fail"
        base.append_csv(base.FULL_CSV, base.row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "rev6_gate_pass":
            continue

        reasoning = (
            f"[{base.LABEL}] Revision 6 of {base.STRATEGY_NAME}. Candidate={h.name}; "
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
            base.row_for(h, "rev6_selection_saved" if saved else "rev6_selection_corr_reject", bool(saved), alpha_id, full),
        )
        base.write_saved_snapshot(conn)
        base.write_research_log(conn, {"revision_6_reset": 0}, "revision_6_running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={base.saved_count(conn)}/{base.TARGET_SAVED}",
            flush=True,
        )
        base.write_progress(
            conn,
            {
                "status": "revision_6_saved" if saved else "revision_6_corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    base.write_saved_snapshot(conn)
    final = "completed_target" if base.saved_count(conn) >= base.TARGET_SAVED else "revision_6_exhausted"
    base.write_progress(conn, {"status": final, "hypothesis_index": len(REVISION_HYPOTHESES)})
    base.write_research_log(conn, {"revision_6_reset": 0}, final, data_notes)
    saved_n = base.saved_count(conn)
    print(f"Revision 6 final status: {final}; saved {saved_n}/{base.TARGET_SAVED}", flush=True)
    conn.close()
    if saved_n < base.TARGET_SAVED:
        raise SystemExit(f"Only saved {saved_n} alphas; target is {base.TARGET_SAVED}.")


if __name__ == "__main__":
    main()
