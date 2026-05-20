"""Revision 3 for the clean anchored auction workflow.

Revision 2 produced one saved alpha. The next passing candidates were either
too correlated to that saved alpha or missed the fitness gate. The workflow
instruction for this case is to change the observable, not to sweep parameters.

This revision keeps all numeric choices fixed and tests one explicitly named
auction-pressure observable at a time:
- 20-day volume/dollar/range baselines as standard monthly liquidity references
- 3-day reversal as short settlement pressure
- 5-day drawdown as a weekly pressure location
- 3-day persistence from Revision 2, not reswept
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


REVISION_CSV = base.OUT_DIR / "revision_3_observable_change_hypotheses.csv"
FIXED_PERSISTENCE_DAYS = 3


ANCHORS = {
    "ocf_assets": (
        "true_divide(cashflow_op, add(assets, 1.0))",
        "operating cash flow to assets",
    ),
    "fcf_assets": (
        "true_divide(free_cashflow, add(assets, 1.0))",
        "free cash flow to assets",
    ),
    "roic": ("roic", "capital efficiency"),
    "gross_profit_assets": (
        "true_divide(gross_profit, add(assets, 1.0))",
        "gross profitability to assets",
    ),
    "retained_earnings_assets": (
        "true_divide(retained_earnings, add(assets, 1.0))",
        "retained earnings to assets",
    ),
    "low_net_debt_assets": (
        "negative(true_divide(net_debt, add(assets, 1.0)))",
        "low net debt to assets",
    ),
    "low_asset_growth": (
        "negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0)))",
        "low annual asset growth",
    ),
    "low_net_stock_issuance": (
        "negative(net_stock_issuance)",
        "low net stock issuance",
    ),
    "earnings_yield": ("earnings_yield_metric", "earnings yield"),
    "fcf_yield": ("fcf_yield_metric", "free cash flow yield"),
    "low_capex_to_revenue": ("negative(capex_to_revenue)", "low capex to revenue"),
    "income_quality": ("income_quality", "income quality"),
}


TRIGGERS = {
    "flow_reversal3": (
        "multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(ts_delta(close, 3))))",
        "abnormal participation combined with short settlement-pressure reversal",
    ),
    "flow_weekly_drawdown": (
        "multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(true_divide(subtract(ts_max(high, 5), close), add(subtract(ts_max(high, 5), ts_min(low, 5)), 0.001))))",
        "abnormal participation into a weak weekly price location",
    ),
    "flow_daily_close_low": (
        "multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(true_divide(subtract(high, close), add(subtract(high, low), 0.001))))",
        "abnormal participation into a weak same-day close location",
    ),
    "flow_intraday_down": (
        "multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001)))))",
        "abnormal participation into an open-to-close selloff",
    ),
    "dollar_vwap": (
        "multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(true_divide(close, add(vwap, 0.001)))))",
        "abnormal dollar participation into a below-VWAP close",
    ),
    "dollar_reversal3": (
        "multiply(rank(true_divide(dollars_traded, add(sma(dollars_traded, 20), 1.0))), rank(negative(ts_delta(close, 3))))",
        "abnormal dollar participation with short settlement-pressure reversal",
    ),
    "range_surge_vwap": (
        "multiply(rank(true_divide(subtract(high, low), add(sma(subtract(high, low), 20), 0.001))), rank(negative(true_divide(close, add(vwap, 0.001)))))",
        "range expansion into a below-VWAP close",
    ),
    "range_surge_reversal3": (
        "multiply(rank(true_divide(subtract(high, low), add(sma(subtract(high, low), 20), 0.001))), rank(negative(ts_delta(close, 3))))",
        "range expansion with short settlement-pressure reversal",
    ),
    "gapdown_flow": (
        "multiply(rank(true_divide(volume, add(sma(volume, 20), 1.0))), rank(negative(true_divide(subtract(close, delay(close, 1)), add(delay(close, 1), 0.001)))))",
        "abnormal participation into a previous-close selloff",
    ),
    "vwap_reversal3": (
        "multiply(rank(negative(true_divide(close, add(vwap, 0.001)))), rank(negative(ts_delta(close, 3))))",
        "below-VWAP close confirmed by short settlement-pressure reversal",
    ),
}


CANDIDATE_SPECS = [
    ("flow_reversal3", "ocf_assets"),
    ("flow_reversal3", "roic"),
    ("flow_reversal3", "retained_earnings_assets"),
    ("flow_reversal3", "low_net_debt_assets"),
    ("flow_reversal3", "low_asset_growth"),
    ("flow_weekly_drawdown", "ocf_assets"),
    ("flow_weekly_drawdown", "roic"),
    ("flow_weekly_drawdown", "gross_profit_assets"),
    ("flow_weekly_drawdown", "retained_earnings_assets"),
    ("flow_weekly_drawdown", "low_net_debt_assets"),
    ("flow_daily_close_low", "ocf_assets"),
    ("flow_daily_close_low", "roic"),
    ("flow_daily_close_low", "gross_profit_assets"),
    ("flow_daily_close_low", "low_asset_growth"),
    ("flow_intraday_down", "ocf_assets"),
    ("flow_intraday_down", "roic"),
    ("flow_intraday_down", "retained_earnings_assets"),
    ("flow_intraday_down", "low_net_debt_assets"),
    ("dollar_vwap", "ocf_assets"),
    ("dollar_vwap", "roic"),
    ("dollar_vwap", "low_asset_growth"),
    ("dollar_reversal3", "ocf_assets"),
    ("dollar_reversal3", "roic"),
    ("dollar_reversal3", "low_net_stock_issuance"),
    ("range_surge_vwap", "ocf_assets"),
    ("range_surge_vwap", "roic"),
    ("range_surge_vwap", "fcf_yield"),
    ("range_surge_reversal3", "ocf_assets"),
    ("range_surge_reversal3", "roic"),
    ("range_surge_reversal3", "income_quality"),
    ("gapdown_flow", "ocf_assets"),
    ("gapdown_flow", "roic"),
    ("gapdown_flow", "low_capex_to_revenue"),
    ("vwap_reversal3", "ocf_assets"),
    ("vwap_reversal3", "roic"),
    ("vwap_reversal3", "low_asset_growth"),
]


def build_hypotheses() -> list[base.Hypothesis]:
    out: list[base.Hypothesis] = []
    for trigger_name, anchor_name in CANDIDATE_SPECS:
        trigger_expr, trigger_desc = TRIGGERS[trigger_name]
        anchor_expr, anchor_desc = ANCHORS[anchor_name]
        expr = f"rank(decay_linear(multiply({trigger_expr}, rank({anchor_expr})), {FIXED_PERSISTENCE_DAYS}))"
        out.append(
            base.Hypothesis(
                f"persist3_{trigger_name}_{anchor_name}",
                f"observable_change_{trigger_name}",
                expr,
                (
                    f"{trigger_desc} should reverse more when conditioned on {anchor_desc}. "
                    "This tests a changed auction observable after Revision 2 correlation rejection."
                ),
                (
                    f"The auction observable is {trigger_name}; the anchor is {anchor_name}. "
                    "No alternate numeric parameters are tested."
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
        "Revision 3 was triggered because Revision 2 saved one alpha and then hit either corr rejection or fitness misses.",
        "The revision changes the auction-pressure observable instead of changing numeric parameters.",
        "Fixed numeric choices: 20-day monthly baselines, 3-day reversal, 5-day weekly drawdown, and 3-day persistence.",
        "The `decay_linear` operator was optimized to an equivalent vectorized implementation after the library callback path bottlenecked correlation checks.",
        "All revision candidates are train-only and use subindustry neutralization.",
    ]
    base.write_research_log(conn, {"revision_3_reset": 0}, "revision_3_running", data_notes)

    print(f"Revision 3 label: {base.LABEL}", flush=True)
    print(f"Revision hypotheses: {len(REVISION_HYPOTHESES)}", flush=True)
    for i, h in enumerate(REVISION_HYPOTHESES, 1):
        if base.saved_count(conn) >= base.TARGET_SAVED:
            break
        print(f"[rev3 {i}/{len(REVISION_HYPOTHESES)}] {h.name}", flush=True)
        base.write_progress(conn, {"status": "revision_3_screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev3_exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=base.UNIVERSE)
        except Exception as exc:
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev3_screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            base.append_csv(base.TRIALS_CSV, base.row_for(h, "rev3_screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "rev3_screen_pass" if base.screen_pass(screen) else "rev3_screen_fail"
        base.append_csv(base.TRIALS_CSV, base.row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "rev3_screen_pass":
            continue

        base.write_progress(conn, {"status": "revision_3_full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            base.append_csv(base.FULL_CSV, base.row_for(h, "rev3_full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "rev3_gate_pass" if base.gates_pass(full) else "rev3_gate_fail"
        base.append_csv(base.FULL_CSV, base.row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "rev3_gate_pass":
            continue

        reasoning = (
            f"[{base.LABEL}] Revision 3 of {base.STRATEGY_NAME}. Candidate={h.name}; "
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
            base.row_for(h, "rev3_selection_saved" if saved else "rev3_selection_corr_reject", bool(saved), alpha_id, full),
        )
        base.write_saved_snapshot(conn)
        base.write_research_log(conn, {"revision_3_reset": 0}, "revision_3_running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={base.saved_count(conn)}/{base.TARGET_SAVED}",
            flush=True,
        )
        base.write_progress(
            conn,
            {
                "status": "revision_3_saved" if saved else "revision_3_corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    base.write_saved_snapshot(conn)
    final = "completed_target" if base.saved_count(conn) >= base.TARGET_SAVED else "revision_3_exhausted"
    base.write_progress(conn, {"status": final, "hypothesis_index": len(REVISION_HYPOTHESES)})
    base.write_research_log(conn, {"revision_3_reset": 0}, final, data_notes)
    saved_n = base.saved_count(conn)
    print(f"Revision 3 final status: {final}; saved {saved_n}/{base.TARGET_SAVED}", flush=True)
    conn.close()
    if saved_n < base.TARGET_SAVED:
        raise SystemExit(f"Only saved {saved_n} alphas; target is {base.TARGET_SAVED}.")


if __name__ == "__main__":
    main()
