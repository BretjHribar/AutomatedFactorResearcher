"""Revision 1 for the clean anchored auction workflow.

Result from the first block:
fundamental anchors plus below-VWAP close improved fitness, but the trigger
was not strong enough to clear Sharpe > 5. The revised hypothesis adds one
fixed participation-confirmation observable: volume / 20-day average volume.

This is not a parameter sweep: the 20-day participation denominator is the
standard monthly liquidity proxy already used elsewhere in the project, and
each expression tests one anchor once.
"""
from __future__ import annotations

import csv
import sqlite3
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


REVISION_CSV = base.OUT_DIR / "revision_1_flow_confirmed_hypotheses.csv"


REVISION_HYPOTHESES = [
    base.Hypothesis(
        "flow_ocf_assets_vwap_dislocation",
        "flow_confirmed_cash_conversion_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(cashflow_op, add(assets, 1.0)))))",
        "A below-VWAP close should reverse more when abnormal participation confirms forced auction supply and operating cash flow supplies a valuation anchor.",
        "Adds one fixed flow-confirmation observable to the first failed cash-conversion hypothesis.",
    ),
    base.Hypothesis(
        "flow_fcf_assets_vwap_dislocation",
        "flow_confirmed_cash_conversion_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(free_cashflow, add(assets, 1.0)))))",
        "Auction supply confirmed by abnormal participation should mean-revert more in firms producing free cash flow.",
        "Uses free-cash-flow backing rather than another price/volume lookback.",
    ),
    base.Hypothesis(
        "flow_fcf_yield_vwap_dislocation",
        "flow_confirmed_valuation_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(fcf_yield_metric)))",
        "High FCF yield should make high-participation weak closes more likely to be liquidity dislocations.",
        "Yield anchor is independent of the participation and VWAP terms.",
    ),
    base.Hypothesis(
        "flow_earnings_yield_vwap_dislocation",
        "flow_confirmed_valuation_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(earnings_yield_metric)))",
        "Earnings-backed names should recover better when a high-participation close prints below VWAP.",
        "Uses an accounting valuation anchor, not a tuned auction parameter.",
    ),
    base.Hypothesis(
        "flow_gross_profit_assets_vwap_dislocation",
        "flow_confirmed_profitability_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(gross_profit, add(assets, 1.0)))))",
        "Profitable small caps should reverse forced high-volume auction markdowns more reliably.",
        "Gross profitability is separate from flow and same-day price location.",
    ),
    base.Hypothesis(
        "flow_roic_vwap_dislocation",
        "flow_confirmed_profitability_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(roic)))",
        "Capital-efficient firms should have stronger next-bar reversal after confirmed auction supply.",
        "ROIC is a capital-efficiency anchor, not a price/volume variant.",
    ),
    base.Hypothesis(
        "flow_net_income_assets_vwap_dislocation",
        "flow_confirmed_profitability_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(net_income, add(assets, 1.0)))))",
        "Asset-scaled profitability should make high-participation weak closes less likely to be fundamental repricing.",
        "Bottom-line profitability is an independent accounting anchor.",
    ),
    base.Hypothesis(
        "flow_retained_earnings_assets_vwap_dislocation",
        "flow_confirmed_balance_sheet_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(retained_earnings, add(assets, 1.0)))))",
        "Seasoned firms with retained earnings should absorb forced auction supply better.",
        "Retained earnings measures capital history, not intraday flow.",
    ),
    base.Hypothesis(
        "flow_low_net_debt_assets_vwap_dislocation",
        "flow_confirmed_balance_sheet_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(true_divide(net_debt, add(assets, 1.0))))))",
        "Low leverage should improve reversal odds after a forced below-VWAP auction close.",
        "Debt burden is a balance-sheet risk filter independent of the trigger.",
    ),
    base.Hypothesis(
        "flow_cash_assets_vwap_dislocation",
        "flow_confirmed_balance_sheet_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(true_divide(cash, add(assets, 1.0)))))",
        "Cash-rich firms should rebound better from high-participation auction markdowns.",
        "Cash intensity is a separate balance-sheet anchor.",
    ),
    base.Hypothesis(
        "flow_low_debt_to_equity_vwap_dislocation",
        "flow_confirmed_balance_sheet_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(debt_to_equity))))",
        "Lower accounting leverage should filter high-participation weak closes toward transitory pressure.",
        "Leverage ratio is independent from the MOC dislocation.",
    ),
    base.Hypothesis(
        "flow_low_intangibles_assets_vwap_dislocation",
        "flow_confirmed_balance_sheet_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(intangibles_to_assets))))",
        "Tangible balance sheets should provide stronger support after confirmed auction supply.",
        "Tangibility is a balance-sheet anchor rather than a trigger tweak.",
    ),
    base.Hypothesis(
        "flow_low_inventory_assets_vwap_dislocation",
        "flow_confirmed_operating_efficiency_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(true_divide(inventory, add(assets, 1.0))))))",
        "Lower inventory burden should remove weak closes that are more likely operating-stress information.",
        "Inventory burden is an operating-condition filter, not a market microstructure variant.",
    ),
    base.Hypothesis(
        "flow_low_capex_to_revenue_vwap_dislocation",
        "flow_confirmed_investment_discipline_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(capex_to_revenue))))",
        "Asset-light firms should recover better from forced high-volume below-VWAP closes.",
        "Investment intensity is a distinct fundamental conditioner.",
    ),
    base.Hypothesis(
        "flow_low_asset_growth_vwap_dislocation",
        "flow_confirmed_investment_discipline_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0))))))",
        "Disciplined asset growth should make confirmed auction pressure more transitory.",
        "Annual asset growth is a separate accounting observable.",
    ),
    base.Hypothesis(
        "flow_low_net_stock_issuance_vwap_dislocation",
        "flow_confirmed_capital_discipline_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(net_stock_issuance))))",
        "Low share issuance should reduce structural supply pressure after high-volume weak closes.",
        "Issuance discipline is separate from auction participation.",
    ),
    base.Hypothesis(
        "flow_income_quality_vwap_dislocation",
        "flow_confirmed_cash_conversion_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(income_quality)))",
        "Higher income quality should make a high-participation weak close more likely to reverse.",
        "Accrual/cash quality is independent from the MOC trigger.",
    ),
    base.Hypothesis(
        "flow_low_ev_sales_vwap_dislocation",
        "flow_confirmed_valuation_anchor",
        "rank(multiply(multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001))))), rank(negative(ev_to_sales))))",
        "Low EV/sales should anchor reversals after confirmed auction supply.",
        "Enterprise-value valuation is a distinct accounting/market-value conditioner.",
    ),
]


def write_revision_hypotheses(blocklist: set[str]) -> None:
    if REVISION_CSV.exists():
        REVISION_CSV.unlink()
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        base.append_csv(
            REVISION_CSV,
            {
                "revision_order": i,
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
        "Revision 1 was triggered because all fundamental-only candidates failed Sharpe > 5 while several had fitness > 5.",
        "The revision adds one fixed participation-confirmation observable: volume divided by 20-day average volume.",
        "The 20-day denominator is the standard monthly liquidity proxy; no alternate volume windows were tested.",
        "All revision candidates are still train-only and use subindustry neutralization.",
    ]
    base.write_research_log(conn, {"revision_1_reset": 0}, "revision_1_running", data_notes)

    print(f"Revision 1 label: {base.LABEL}", flush=True)
    print(f"Revision hypotheses: {len(REVISION_HYPOTHESES)}", flush=True)
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        if base.saved_count(conn) >= base.TARGET_SAVED:
            break
        print(f"[rev1 {i}/{len(REVISION_HYPOTHESES)}] {h.name}", flush=True)
        base.write_progress(conn, {"status": "revision_1_screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev1_exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=base.UNIVERSE)
        except Exception as exc:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev1_screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev1_screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "rev1_screen_pass" if base.screen_pass(screen) else "rev1_screen_fail"
        base.append_csv(base.TRIALS_CSV, base.row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "rev1_screen_pass":
            continue

        base.write_progress(conn, {"status": "revision_1_full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            base.append_csv(base.FULL_CSV, base.row_for(h, "rev1_full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "rev1_gate_pass" if base.gates_pass(full) else "rev1_gate_fail"
        base.append_csv(base.FULL_CSV, base.row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "rev1_gate_pass":
            continue

        reasoning = (
            f"[{base.LABEL}] Revision 1 of {base.STRATEGY_NAME}. Candidate={h.name}; "
            f"family={h.family}; train-only mechanism={h.mechanism}; "
            f"orthogonal reason={h.orthogonal_reason}; fixed dynamic universe={base.UNIVERSE}; "
            f"strict gates: train Sharpe>5, train fitness>5, turnover<=1.0, IC>0; "
            f"subindustry neutralization; exact-copy blocklist; corr<=0.70; no parameter sweeps."
        )
        saved = harness.save_alpha(conn, h.expression, reasoning, full, base.LABEL)
        alpha_id = base.alpha_id_for_expr(conn, h.expression) if saved else None
        base.append_csv(
            base.SELECTION_CSV,
            base.row_for(h, "rev1_selection_saved" if saved else "rev1_selection_corr_reject", bool(saved), alpha_id, full),
        )
        base.write_saved_snapshot(conn)
        base.write_research_log(conn, {"revision_1_reset": 0}, "revision_1_running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={base.saved_count(conn)}/{base.TARGET_SAVED}",
            flush=True,
        )
        base.write_progress(
            conn,
            {
                "status": "revision_1_saved" if saved else "revision_1_corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    base.write_saved_snapshot(conn)
    final = "completed_target" if base.saved_count(conn) >= base.TARGET_SAVED else "revision_1_exhausted"
    base.write_progress(conn, {"status": final, "hypothesis_index": len(REVISION_HYPOTHESES)})
    base.write_research_log(conn, {"revision_1_reset": 0}, final, data_notes)
    print(f"Revision 1 final status: {final}; saved {base.saved_count(conn)}/{base.TARGET_SAVED}", flush=True)
    saved_n = base.saved_count(conn)
    conn.close()
    if saved_n < base.TARGET_SAVED:
        raise SystemExit(f"Only saved {saved_n} alphas; target is {base.TARGET_SAVED}.")


if __name__ == "__main__":
    main()
