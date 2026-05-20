"""Revision 4 for the clean anchored auction workflow.

Revision 3 saved three alphas. The remaining strong candidates were mostly
rejected for correlation with the saved OCF/VWAP state or missed fitness by a
small margin. Per workflow, this revision changes the anchor observable shape:
composite accounting anchors and a soft additive score, with the same fixed
auction observables and no new numeric parameter choices.
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
import run_revision_3_observable_changes as rev3  # noqa: E402


REVISION_CSV = base.OUT_DIR / "revision_4_composite_anchor_hypotheses.csv"
FIXED_PERSISTENCE_DAYS = 3


COMPOSITE_ANCHORS = {
    "cash_roic": (
        "multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(roic))",
        "operating cash flow and capital efficiency",
    ),
    "cash_retained": (
        "multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(true_divide(retained_earnings, add(assets, 1.0))))",
        "operating cash flow and retained earnings",
    ),
    "cash_low_growth": (
        "multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0)))))",
        "operating cash flow and disciplined asset growth",
    ),
    "cash_low_debt": (
        "multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(negative(true_divide(net_debt, add(assets, 1.0)))))",
        "operating cash flow and low net debt",
    ),
    "roic_low_growth": (
        "multiply(rank(roic), rank(negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0)))))",
        "capital efficiency and disciplined asset growth",
    ),
    "roic_low_debt": (
        "multiply(rank(roic), rank(negative(true_divide(net_debt, add(assets, 1.0)))))",
        "capital efficiency and low net debt",
    ),
    "yield_roic": (
        "multiply(rank(earnings_yield_metric), rank(roic))",
        "earnings yield and capital efficiency",
    ),
    "fcf_yield_low_issuance": (
        "multiply(rank(fcf_yield_metric), rank(negative(net_stock_issuance)))",
        "free cash flow yield and low share issuance",
    ),
    "gross_profit_low_debt": (
        "multiply(rank(true_divide(gross_profit, add(assets, 1.0))), rank(negative(true_divide(net_debt, add(assets, 1.0)))))",
        "gross profitability and low net debt",
    ),
    "income_quality_roic": (
        "multiply(rank(income_quality), rank(roic))",
        "income quality and capital efficiency",
    ),
}


CANDIDATE_SPECS = [
    ("interaction", "dollar_vwap", "cash_roic"),
    ("interaction", "dollar_vwap", "cash_low_growth"),
    ("interaction", "dollar_vwap", "roic_low_growth"),
    ("interaction", "dollar_vwap", "yield_roic"),
    ("soft_add", "dollar_vwap", "cash_roic"),
    ("soft_add", "dollar_vwap", "cash_low_growth"),
    ("soft_add", "dollar_vwap", "fcf_yield_low_issuance"),
    ("interaction", "flow_daily_close_low", "cash_roic"),
    ("interaction", "flow_daily_close_low", "cash_low_growth"),
    ("interaction", "flow_daily_close_low", "roic_low_debt"),
    ("soft_add", "flow_daily_close_low", "cash_roic"),
    ("soft_add", "flow_daily_close_low", "roic_low_growth"),
    ("soft_add", "flow_daily_close_low", "gross_profit_low_debt"),
    ("interaction", "flow_weekly_drawdown", "cash_retained"),
    ("interaction", "flow_weekly_drawdown", "cash_low_growth"),
    ("interaction", "flow_weekly_drawdown", "roic_low_growth"),
    ("soft_add", "flow_weekly_drawdown", "cash_roic"),
    ("soft_add", "flow_weekly_drawdown", "cash_low_debt"),
    ("soft_add", "flow_weekly_drawdown", "income_quality_roic"),
    ("interaction", "vwap_reversal3", "cash_roic"),
    ("interaction", "vwap_reversal3", "cash_low_growth"),
    ("interaction", "vwap_reversal3", "roic_low_growth"),
    ("soft_add", "vwap_reversal3", "cash_roic"),
    ("soft_add", "vwap_reversal3", "roic_low_debt"),
    ("soft_add", "vwap_reversal3", "fcf_yield_low_issuance"),
    ("interaction", "flow_reversal3", "cash_roic"),
    ("interaction", "flow_reversal3", "roic_low_growth"),
    ("soft_add", "flow_reversal3", "cash_low_debt"),
    ("interaction", "dollar_reversal3", "cash_roic"),
    ("interaction", "dollar_reversal3", "roic_low_growth"),
    ("soft_add", "dollar_reversal3", "yield_roic"),
    ("interaction", "range_surge_vwap", "cash_roic"),
    ("interaction", "range_surge_vwap", "roic_low_debt"),
    ("soft_add", "range_surge_vwap", "cash_low_growth"),
]


def build_hypotheses() -> list[base.Hypothesis]:
    out: list[base.Hypothesis] = []
    for mode, trigger_name, anchor_name in CANDIDATE_SPECS:
        trigger_expr, trigger_desc = rev3.TRIGGERS[trigger_name]
        anchor_expr, anchor_desc = COMPOSITE_ANCHORS[anchor_name]
        if mode == "interaction":
            core = f"multiply({trigger_expr}, {anchor_expr})"
            mode_desc = "conjunctive interaction"
        elif mode == "soft_add":
            core = f"add({trigger_expr}, {anchor_expr})"
            mode_desc = "soft additive score"
        else:
            raise ValueError(mode)
        expr = f"rank(decay_linear({core}, {FIXED_PERSISTENCE_DAYS}))"
        out.append(
            base.Hypothesis(
                f"persist3_{mode}_{trigger_name}_{anchor_name}",
                f"composite_anchor_{mode}_{trigger_name}",
                expr,
                (
                    f"{trigger_desc} should reverse more under a {mode_desc} with {anchor_desc}. "
                    "This changes anchor shape after correlation rejection, not numeric parameters."
                ),
                (
                    f"Composite anchor {anchor_name} changes the conditioning observable; "
                    f"trigger {trigger_name} and persistence remain fixed."
                ),
            )
        )
    return out


REVISION_HYPOTHESES = build_hypotheses()


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
        "Revision 4 was triggered because Revision 3 saved three alphas, with remaining strong candidates mostly corr-rejected or just below fitness.",
        "The revision changes anchor shape through composite fundamental anchors and soft additive scoring.",
        "No new numeric parameters are introduced; persistence remains 3, volume/dollar/range baselines remain 20, reversal remains 3, weekly drawdown remains 5.",
        "All revision candidates are train-only and use subindustry neutralization.",
    ]
    base.write_research_log(conn, {"revision_4_reset": 0}, "revision_4_running", data_notes)

    print(f"Revision 4 label: {base.LABEL}", flush=True)
    print(f"Revision hypotheses: {len(REVISION_HYPOTHESES)}", flush=True)
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        if base.saved_count(conn) >= base.TARGET_SAVED:
            break
        print(f"[rev4 {i}/{len(REVISION_HYPOTHESES)}] {h.name}", flush=True)
        base.write_progress(conn, {"status": "revision_4_screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev4_exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=base.UNIVERSE)
        except Exception as exc:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev4_screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev4_screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "rev4_screen_pass" if base.screen_pass(screen) else "rev4_screen_fail"
        base.append_csv(base.TRIALS_CSV, base.row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "rev4_screen_pass":
            continue

        base.write_progress(conn, {"status": "revision_4_full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            base.append_csv(base.FULL_CSV, base.row_for(h, "rev4_full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "rev4_gate_pass" if base.gates_pass(full) else "rev4_gate_fail"
        base.append_csv(base.FULL_CSV, base.row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "rev4_gate_pass":
            continue

        reasoning = (
            f"[{base.LABEL}] Revision 4 of {base.STRATEGY_NAME}. Candidate={h.name}; "
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
            base.row_for(h, "rev4_selection_saved" if saved else "rev4_selection_corr_reject", bool(saved), alpha_id, full),
        )
        base.write_saved_snapshot(conn)
        base.write_research_log(conn, {"revision_4_reset": 0}, "revision_4_running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={base.saved_count(conn)}/{base.TARGET_SAVED}",
            flush=True,
        )
        base.write_progress(
            conn,
            {
                "status": "revision_4_saved" if saved else "revision_4_corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    base.write_saved_snapshot(conn)
    final = "completed_target" if base.saved_count(conn) >= base.TARGET_SAVED else "revision_4_exhausted"
    base.write_progress(conn, {"status": final, "hypothesis_index": len(REVISION_HYPOTHESES)})
    base.write_research_log(conn, {"revision_4_reset": 0}, final, data_notes)
    saved_n = base.saved_count(conn)
    print(f"Revision 4 final status: {final}; saved {saved_n}/{base.TARGET_SAVED}", flush=True)
    conn.close()
    if saved_n < base.TARGET_SAVED:
        raise SystemExit(f"Only saved {saved_n} alphas; target is {base.TARGET_SAVED}.")


if __name__ == "__main__":
    main()
