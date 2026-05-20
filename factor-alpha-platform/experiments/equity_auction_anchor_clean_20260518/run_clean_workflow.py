"""Clean train-only alpha workflow for anchored auction dislocation research.

This run is intentionally boring in the right way:
- one fixed dynamic universe, chosen before any alpha result is read
- one pre-registered expression per economic hypothesis
- train-only evaluation and persistence through eval_alpha_ib
- subindustry neutralization and DB correlation gate from the project harness
- no parameter sweeps
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_alpha_ib as harness  # noqa: E402
from seed_alphas_ib import SEED_ALPHAS  # noqa: E402


LABEL = "AUCT_ANCHOR_MCAP90_550_D0_CLEAN_S5_F5_20260518"
STRATEGY_NAME = "Anchored MOC auction-dislocation reversal"
UNIVERSE = "AUCT_ANCHOR_MCAP90M_550M_DAILY"
TARGET_SAVED = 10

SOURCE_UNIVERSE = (
    ROOT
    / "experiments"
    / "equity_auction_exhaustion_mcap90_550_20260517"
    / "universes"
    / "AEXH_MCAP90M_550M_DAILY.parquet"
)
SOURCE_UNIVERSE_MANIFEST = SOURCE_UNIVERSE.with_name("universe_manifest.json")
SOURCE_MATRICES = ROOT / "data" / "fmp_cache" / "matrices"

OUT_DIR = EXP_DIR / "outputs"
TRIALS_CSV = OUT_DIR / "train_trials.csv"
FULL_CSV = OUT_DIR / "full_gate_trials.csv"
SELECTION_CSV = OUT_DIR / "selection_save_phase.csv"
SAVED_CSV = OUT_DIR / "saved_alphas.csv"
PREREG_CSV = OUT_DIR / "pre_registered_hypotheses.csv"
FIELD_COVERAGE_CSV = OUT_DIR / "field_coverage_train.csv"
MANIFEST_JSON = OUT_DIR / "discovery_manifest.json"
PROGRESS_JSON = OUT_DIR / "progress_state.json"
LOG_MD = EXP_DIR / "research_log.md"

PRIOR_NONCOMPLIANT_LABELS = (
    "AEXH_MCAP90M_550M_DAILY_D0_S5_F5_20260517",
)

REQUIRED_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "dollars_traded",
    "adv20",
    "adv60",
    "market_cap",
    "subindustry",
    "assets",
    "cashflow_op",
    "free_cashflow",
    "gross_profit",
    "net_income",
    "revenue",
    "sales",
    "ebit",
    "ebitda",
    "enterprise_value",
    "cash",
    "working_capital",
    "inventory",
    "intangibles",
    "liabilities",
    "debt",
    "net_debt",
    "capex",
    "ppe_net",
    "retained_earnings",
    "current_ratio",
    "debt_to_equity",
    "income_quality",
    "roic",
    "roa",
    "roe",
    "return_equity",
    "earnings_yield_metric",
    "fcf_yield_metric",
    "gross_profit_field",
    "pe_ratio",
    "pb_ratio",
    "ev_to_sales",
    "ev_to_fcf_metric",
    "dividend_yield",
    "net_stock_issuance",
    "capex_to_revenue",
    "intangibles_to_assets",
]


@dataclass(frozen=True)
class Hypothesis:
    name: str
    family: str
    expression: str
    mechanism: str
    orthogonal_reason: str


HYPOTHESES = [
    Hypothesis(
        "ocf_assets_vwap_dislocation",
        "cash_conversion_anchor",
        "rank(multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "If a small-cap closes weak versus VWAP, same-day forced auction supply should mean-revert more when operating cash flow gives a valuation anchor.",
        "Uses operating cash conversion rather than price/volume state as the conditioning variable.",
    ),
    Hypothesis(
        "fcf_assets_vwap_dislocation",
        "cash_conversion_anchor",
        "rank(multiply(rank(true_divide(free_cashflow, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Free-cash-flow-backed names should absorb temporary auction pressure faster than cash-consuming peers.",
        "Free cash flow is a different accounting observable from the volume/VWAP production family.",
    ),
    Hypothesis(
        "fcf_yield_vwap_dislocation",
        "valuation_anchor",
        "rank(multiply(rank(fcf_yield_metric), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "High FCF yield gives a cross-sectional anchor for a close below VWAP to reverse on the next bar.",
        "Uses a yield metric, not a lookback/window variant of the auction trigger.",
    ),
    Hypothesis(
        "earnings_yield_vwap_dislocation",
        "valuation_anchor",
        "rank(multiply(rank(earnings_yield_metric), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Earnings-backed auction dislocations should be less likely to represent permanent information.",
        "Earnings yield is independent from flow imbalance and from raw price reversal.",
    ),
    Hypothesis(
        "gross_profit_assets_vwap_dislocation",
        "profitability_anchor",
        "rank(multiply(rank(true_divide(gross_profit, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "High gross profitability should make closing-auction markdowns more likely to be liquidity-driven than fundamental.",
        "Profitability anchor uses income statement scale, not the auction state itself.",
    ),
    Hypothesis(
        "roic_vwap_dislocation",
        "profitability_anchor",
        "rank(multiply(rank(roic), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Capital-efficient firms should have stronger buyer response after an auction close below VWAP.",
        "ROIC captures capital efficiency rather than value or volume.",
    ),
    Hypothesis(
        "net_income_assets_vwap_dislocation",
        "profitability_anchor",
        "rank(multiply(rank(true_divide(net_income, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Positive asset-scaled profitability should stabilize auction-driven price dislocations.",
        "Uses a bottom-line profitability anchor with separate accounting coverage.",
    ),
    Hypothesis(
        "low_net_debt_assets_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(negative(true_divide(net_debt, add(assets, 1.0)))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Auction weakness should reverse more in less levered small caps because balance-sheet risk is lower.",
        "Debt burden is independent from the intraday auction dislocation.",
    ),
    Hypothesis(
        "cash_assets_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(true_divide(cash, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Cash-rich small caps should mean-revert better after weak auction prints because financing stress is lower.",
        "Cash intensity is a balance-sheet anchor, not a price/volume transform.",
    ),
    Hypothesis(
        "working_capital_assets_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(true_divide(working_capital, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Positive working-capital cushion should separate liquidity pressure from fundamental distress after a weak close.",
        "Working-capital exposure is separate from quality/value production expressions.",
    ),
    Hypothesis(
        "low_debt_to_equity_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(negative(debt_to_equity)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Lower balance-sheet leverage should increase the reversal odds of a close below VWAP.",
        "Leverage ratio is an external risk anchor and not a tuned auction parameter.",
    ),
    Hypothesis(
        "retained_earnings_assets_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(true_divide(retained_earnings, add(assets, 1.0))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Accumulated retained earnings should indicate a more seasoned business where auction pressure is more transitory.",
        "Retained earnings is a capital-history anchor, not a flow or range variant.",
    ),
    Hypothesis(
        "low_intangibles_assets_vwap_dislocation",
        "balance_sheet_anchor",
        "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Tangible-balance-sheet firms should provide a harder anchor after a weak closing auction.",
        "Asset tangibility is orthogonal to raw reversal and volume participation.",
    ),
    Hypothesis(
        "low_inventory_assets_vwap_dislocation",
        "operating_efficiency_anchor",
        "rank(multiply(rank(negative(true_divide(inventory, add(assets, 1.0)))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Lower inventory intensity should avoid stale-inventory distress names where weak auctions are information-heavy.",
        "Inventory burden is an operating-efficiency anchor, not a price/volume parameter.",
    ),
    Hypothesis(
        "low_capex_to_revenue_vwap_dislocation",
        "investment_discipline_anchor",
        "rank(multiply(rank(negative(capex_to_revenue)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Asset-light firms should recover better after auction pressure because forced reinvestment needs are lower.",
        "Capex intensity is a fundamental conditioner, not a smoothing/window change.",
    ),
    Hypothesis(
        "low_asset_growth_vwap_dislocation",
        "investment_discipline_anchor",
        "rank(multiply(rank(negative(true_divide(ts_delta(assets, 252), add(delay(assets, 252), 1.0)))), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Low asset growth should identify disciplined balance sheets where closing auction weakness is more liquidity-driven.",
        "One annual accounting change is a separate observable, not a parameter sweep.",
    ),
    Hypothesis(
        "low_net_stock_issuance_vwap_dislocation",
        "capital_discipline_anchor",
        "rank(multiply(rank(negative(net_stock_issuance)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Low dilution should improve reversal odds after weak auction prints because supply pressure is less structurally persistent.",
        "Share issuance is a capital-markets anchor independent of same-day flow.",
    ),
    Hypothesis(
        "income_quality_vwap_dislocation",
        "cash_conversion_anchor",
        "rank(multiply(rank(income_quality), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Higher income quality should make a below-VWAP close more likely to be temporary liquidity pressure.",
        "Accrual/cash-conversion quality is not a price or volume transform.",
    ),
    Hypothesis(
        "low_ev_sales_vwap_dislocation",
        "valuation_anchor",
        "rank(multiply(rank(negative(ev_to_sales)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Low EV/sales should provide a simple valuation anchor for closing-auction dislocation reversal.",
        "Enterprise-value sales valuation is separate from the auction trigger.",
    ),
    Hypothesis(
        "low_ev_fcf_vwap_dislocation",
        "valuation_anchor",
        "rank(multiply(rank(negative(ev_to_fcf_metric)), rank(decay_exp(negative(true_divide(close, add(vwap, 0.001))), 0.08))))",
        "Low EV/FCF should identify names where a weak auction close has stronger valuation support.",
        "Enterprise-value cash-flow valuation is a separate fundamental lens.",
    ),
    Hypothesis(
        "ocf_assets_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(true_divide(cashflow_op, add(assets, 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "An intraday down close should reverse more when operating cash flow provides fundamental support.",
        "Uses open-to-close capitulation instead of VWAP dislocation.",
    ),
    Hypothesis(
        "fcf_yield_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(fcf_yield_metric), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Open-to-close selling pressure should be less persistent in high FCF-yield names.",
        "Changes the auction observable to intraday return while keeping the valuation anchor fixed.",
    ),
    Hypothesis(
        "gross_profit_assets_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(true_divide(gross_profit, add(assets, 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "High gross-profit names should bounce better after weak intraday closes in the auction band.",
        "Profitability anchor plus intraday move is structurally different from volume/VWAP alphas.",
    ),
    Hypothesis(
        "low_net_debt_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(negative(true_divide(net_debt, add(assets, 1.0)))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Less levered firms should be more resilient after open-to-close pressure.",
        "Debt-risk conditioning is independent from same-day auction flow.",
    ),
    Hypothesis(
        "cash_assets_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(true_divide(cash, add(assets, 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Cash-rich names should mean-revert more after an intraday selloff into the close.",
        "Cash buffer is a balance-sheet anchor rather than a parameter choice.",
    ),
    Hypothesis(
        "low_capex_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(negative(capex_to_revenue)), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Asset-light firms should rebound better after intraday auction-band pressure.",
        "Investment intensity is an independent conditioner.",
    ),
    Hypothesis(
        "low_inventory_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(negative(true_divide(inventory, add(assets, 1.0)))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Lower inventory burden should avoid names where weak closes reflect operating impairment.",
        "Inventory efficiency is not a tuned lookback or volume transform.",
    ),
    Hypothesis(
        "retained_earnings_intraday_down_close",
        "intraday_capitulation_anchor",
        "rank(multiply(rank(true_divide(retained_earnings, add(assets, 1.0))), rank(negative(true_divide(subtract(close, open), add(open, 0.001))))))",
        "Seasoned retained-earnings firms should have stronger reversal after intraday selling pressure.",
        "Capital-history anchor is orthogonal to raw auction pressure.",
    ),
]


def configure_harness() -> None:
    harness.MATRICES_DIR = EXP_DIR / "matrices_minimal"
    harness.UNIVERSES_DIR = EXP_DIR / "universes"
    harness.UNIVERSE = UNIVERSE
    harness.TRAIN_START = "2016-01-01"
    harness.TRAIN_END = "2023-01-01"
    harness.VAL_START = "2023-01-01"
    harness.VAL_END = "2024-07-01"
    harness.TEST_START = "2024-07-01"
    harness.COVERAGE_CUTOFF = 0.0
    harness.MAX_WEIGHT = 0.02
    harness.NEUTRALIZE = "subindustry"
    harness.CORR_CUTOFF = 0.70
    harness.MIN_IS_SHARPE = 5.0
    harness.MIN_FITNESS = 5.0
    harness.MIN_IC_MEAN = 0.0
    harness.MAX_TURNOVER = 1.0
    harness.MAX_PNL_KURTOSIS = 25
    harness.MAX_ROLLING_SR_STD = 1.50
    harness.MIN_PNL_SKEW = -1.00
    harness.DELAY = 0
    harness.DECAY = 0
    harness.FEES_BPS = 0.0
    harness._DATA_CACHE.clear()
    if hasattr(harness, "_DIVERSITY_CACHE"):
        harness._DIVERSITY_CACHE.clear()


def ensure_assets() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "universes").mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "matrices_minimal").mkdir(parents=True, exist_ok=True)

    if not SOURCE_UNIVERSE.exists():
        raise FileNotFoundError(f"Source universe not found: {SOURCE_UNIVERSE}")
    target_universe = EXP_DIR / "universes" / f"{UNIVERSE}.parquet"
    if not target_universe.exists():
        shutil.copy2(SOURCE_UNIVERSE, target_universe)
    if SOURCE_UNIVERSE_MANIFEST.exists():
        shutil.copy2(SOURCE_UNIVERSE_MANIFEST, EXP_DIR / "universes" / "source_universe_manifest.json")

    missing = []
    for field in REQUIRED_FIELDS:
        src = SOURCE_MATRICES / f"{field}.parquet"
        dst = EXP_DIR / "matrices_minimal" / f"{field}.parquet"
        if not src.exists():
            missing.append(field)
            continue
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    if missing:
        raise FileNotFoundError(f"Missing required matrix fields: {missing}")


def load_exact_blocklist() -> set[str]:
    block = {a["expr"].strip() for a in SEED_ALPHAS}
    for db_rel in ("data/alpha_results.db", "data/ib_alphas.db"):
        path = ROOT / db_rel
        if not path.exists():
            continue
        con = sqlite3.connect(path)
        try:
            block.update(r[0].strip() for r in con.execute("SELECT expression FROM alphas"))
        finally:
            con.close()
    for csv_path in (ROOT / "experiments").glob("equity_*/outputs/saved_alphas.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    expr = (row.get("expression") or "").strip()
                    if expr:
                        block.add(expr)
        except Exception:
            continue
    return block


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def row_for(h: Hypothesis, status: str, saved: bool, alpha_id: int | None, m: dict | None, err: str = "") -> dict:
    row = {
        "name": h.name,
        "family": h.family,
        "status": status,
        "saved": int(saved),
        "alpha_id": alpha_id or "",
        "expression": h.expression,
        "mechanism": h.mechanism,
        "orthogonal_reason": h.orthogonal_reason,
        "error": err,
    }
    for key in (
        "sharpe",
        "fitness",
        "is_sharpe",
        "is_fitness",
        "turnover",
        "ic_mean",
        "icir",
        "stability_h1",
        "stability_h2",
        "pnl_kurtosis",
        "pnl_skew",
        "rolling_sr_std",
        "returns_ann",
        "max_drawdown",
        "deflated_sharpe",
        "elapsed",
    ):
        row[key] = "" if m is None else m.get(key, "")
    return row


def alpha_id_for_expr(conn: sqlite3.Connection, expr: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM alphas WHERE expression=? AND category=? AND archived=0",
        (expr, LABEL),
    ).fetchone()
    return int(row[0]) if row else None


def saved_count(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM alphas WHERE archived=0 AND category=?",
            (LABEL,),
        ).fetchone()[0]
    )


def delete_label(conn: sqlite3.Connection, label: str) -> int:
    ids = [r[0] for r in conn.execute("SELECT id FROM alphas WHERE category=?", (label,))]
    if not ids:
        return 0
    q = ",".join("?" for _ in ids)
    conn.execute(f"DELETE FROM evaluations WHERE alpha_id IN ({q})", ids)
    conn.execute(f"DELETE FROM alphas WHERE id IN ({q})", ids)
    conn.commit()
    return len(ids)


def reset_clean_labels(conn: sqlite3.Connection) -> dict[str, int]:
    deleted = {LABEL: delete_label(conn, LABEL)}
    for label in PRIOR_NONCOMPLIANT_LABELS:
        deleted[label] = delete_label(conn, label)
    return deleted


def write_preregistered(blocklist: set[str]) -> None:
    if PREREG_CSV.exists():
        PREREG_CSV.unlink()
    for i, h in enumerate(HYPOTHESES, 1):
        append_csv(
            PREREG_CSV,
            {
                "order": i,
                "name": h.name,
                "family": h.family,
                "exact_blocked_before_run": int(h.expression.strip() in blocklist),
                "expression": h.expression,
                "mechanism": h.mechanism,
                "orthogonal_reason": h.orthogonal_reason,
            },
        )


def write_field_coverage() -> None:
    import pandas as pd

    if FIELD_COVERAGE_CSV.exists():
        FIELD_COVERAGE_CSV.unlink()
    universe = pd.read_parquet(EXP_DIR / "universes" / f"{UNIVERSE}.parquet")
    universe.index = pd.to_datetime(universe.index)
    universe = universe.loc[harness.TRAIN_START:harness.TRAIN_END]
    active = universe.astype(bool)
    for field in REQUIRED_FIELDS:
        fp = EXP_DIR / "matrices_minimal" / f"{field}.parquet"
        df = pd.read_parquet(fp)
        df.index = pd.to_datetime(df.index)
        cols = [c for c in active.columns if c in df.columns]
        if not cols:
            coverage = 0.0
        else:
            aligned = df.reindex(index=active.index, columns=cols)
            mask = active[cols]
            denom = int(mask.values.sum())
            coverage = float(aligned.where(mask).notna().values.sum() / denom) if denom else 0.0
        append_csv(FIELD_COVERAGE_CSV, {"field": field, "train_universe_coverage": coverage})


def screen_pass(m: dict) -> bool:
    return (
        m.get("success", False)
        and m["sharpe"] > harness.MIN_IS_SHARPE
        and m["fitness"] > harness.MIN_FITNESS
        and m["turnover"] <= harness.MAX_TURNOVER
    )


def gates_pass(m: dict) -> bool:
    return (
        m.get("success", False)
        and m["is_sharpe"] > harness.MIN_IS_SHARPE
        and m["is_fitness"] > harness.MIN_FITNESS
        and m["ic_mean"] > harness.MIN_IC_MEAN
        and m["turnover"] <= harness.MAX_TURNOVER
        and m["pnl_kurtosis"] <= harness.MAX_PNL_KURTOSIS
        and m["rolling_sr_std"] <= harness.MAX_ROLLING_SR_STD
        and m["pnl_skew"] >= harness.MIN_PNL_SKEW
    )


def write_saved_snapshot(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT a.id, a.expression, a.notes, e.sharpe_is, e.fitness, e.turnover,
               e.ic_mean, e.ic_ir, e.sharpe_h1, e.sharpe_h2,
               e.pnl_kurtosis, e.pnl_skew, e.rolling_sr_std
        FROM alphas a JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived=0 AND a.category=?
        ORDER BY a.id
        """,
        (LABEL,),
    ).fetchall()
    headers = [
        "alpha_id",
        "expression",
        "notes",
        "sharpe_is",
        "fitness",
        "turnover",
        "ic_mean",
        "ic_ir",
        "sharpe_h1",
        "sharpe_h2",
        "pnl_kurtosis",
        "pnl_skew",
        "rolling_sr_std",
    ]
    SAVED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SAVED_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_progress(conn: sqlite3.Connection, extra: dict | None = None) -> None:
    state = {
        "label": LABEL,
        "target_saved": TARGET_SAVED,
        "saved_count": saved_count(conn),
        "updated_at_epoch": time.time(),
    }
    if extra:
        state.update(extra)
    PROGRESS_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


def write_research_log(
    conn: sqlite3.Connection,
    deleted_labels: dict[str, int],
    final_status: str,
    data_notes: list[str],
) -> None:
    saved_n = saved_count(conn)
    rows = conn.execute(
        """
        SELECT a.id, e.sharpe_is, e.fitness, e.turnover, e.ic_mean, e.sharpe_h1, e.sharpe_h2, a.expression
        FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
        WHERE a.archived=0 AND a.category=?
        ORDER BY a.id
        """,
        (LABEL,),
    ).fetchall()
    lines = [
        f"# {STRATEGY_NAME}",
        "",
        f"Label: `{LABEL}`",
        f"Status: {final_status}",
        "",
        "Fixed universe:",
        f"- `{UNIVERSE}` copied before the run from the dynamic PIT cap-band universe.",
        "- Rules: market cap 90MM-550MM, price 1.5-80, requires PIT membership, VWAP, volume, and subindustry.",
        "- The universe is daily dynamic, not a static train-date membership snapshot.",
        "",
        "Pre-registered strategy hypothesis:",
        "- Small-cap MOC closing-auction dislocations should reverse when the weak close is paired with an independent accounting/valuation/liquidity anchor.",
        "- The auction trigger supplies timing; the anchor filters cases where the dislocation is more likely liquidity pressure than permanent information.",
        "- Each alpha changes the anchor or the auction observable once. There are no window/decay/lookback grids.",
        "",
        "Workflow controls:",
        "- Train-only discovery: 2016-01-01 through 2023-01-01.",
        "- No validation/test data is read by this script.",
        "- Existing libraries used: `eval_alpha_ib` loading, expression engine, simulator, preprocessing, persistence, and DB diversity check.",
        "- Neutralization: subindustry through the shared preprocessing path.",
        "- Gates: train Sharpe > 5, train fitness > 5, turnover <= 1.0, IC > 0, PnL kurtosis <= 25, rolling SR std <= 1.50, skew >= -1.00.",
        "- Correlation cutoff: 0.70 through the IB DB save gate.",
        "- Exact expression blocklist includes seed alphas, `data/alpha_results.db`, `data/ib_alphas.db`, and prior experiment saved-alpha CSVs.",
        "",
        "DB cleanup:",
    ]
    for label, n in deleted_labels.items():
        lines.append(f"- Deleted {n} prior rows for `{label}` before this clean run.")
    lines.extend(
        [
            "",
            "Passing alphas saved:",
        ]
    )
    if rows:
        for row in rows:
            alpha_id, sr, fit, turnover, ic, h1, h2, expr = row
            lines.append(
                f"- #{alpha_id}: SR={sr:+.3f}, Fit={fit:.3f}, TO={turnover:.3f}, "
                f"IC={ic:+.4f}, H1={h1:+.3f}, H2={h2:+.3f}, expr=`{expr}`"
            )
    else:
        lines.append("- None yet.")
    lines.extend(
        [
            "",
            f"Saved count: {saved_n}/{TARGET_SAVED}",
            "",
            "Data/workflow notes:",
        ]
    )
    for note in data_notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "- Individual alpha discovery remains fee-free by project convention; portfolio validation must apply IB/MOC fees with impact_bps=0.0.",
            "- The harness recomputes returns from close and clips absolute daily returns above 50 percent.",
        ]
    )
    LOG_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    configure_harness()
    ensure_assets()

    for path in (TRIALS_CSV, FULL_CSV, SELECTION_CSV, SAVED_CSV, MANIFEST_JSON, PROGRESS_JSON):
        if path.exists():
            path.unlink()

    conn = harness.get_conn()
    deleted_labels = reset_clean_labels(conn)
    blocklist = load_exact_blocklist()
    write_preregistered(blocklist)
    write_field_coverage()

    data_notes = [
        "Subindustry labels are consumed through the existing matrix/classification loader.",
        "The minimal matrix folder is hardlinked from source parquet matrices; values are not transformed.",
        "No candidate expression is generated by a parameter loop.",
    ]

    manifest = {
        "label": LABEL,
        "strategy": STRATEGY_NAME,
        "universe": UNIVERSE,
        "db_path": harness.DB_PATH,
        "target_saved": TARGET_SAVED,
        "train_start": harness.TRAIN_START,
        "train_end": harness.TRAIN_END,
        "neutralize": harness.NEUTRALIZE,
        "corr_cutoff": harness.CORR_CUTOFF,
        "max_weight": harness.MAX_WEIGHT,
        "fees_bps_individual_alpha": harness.FEES_BPS,
        "hypothesis_count": len(HYPOTHESES),
        "no_param_sweeps": True,
        "gates": {
            "min_is_sharpe_strict_gt": harness.MIN_IS_SHARPE,
            "min_fitness_strict_gt": harness.MIN_FITNESS,
            "max_turnover": harness.MAX_TURNOVER,
            "min_ic_mean_strict_gt": harness.MIN_IC_MEAN,
            "max_pnl_kurtosis": harness.MAX_PNL_KURTOSIS,
            "max_rolling_sr_std": harness.MAX_ROLLING_SR_STD,
            "min_pnl_skew": harness.MIN_PNL_SKEW,
        },
        "deleted_prior_labels": deleted_labels,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Clean workflow label: {LABEL}", flush=True)
    print(f"Pre-registered hypotheses: {len(HYPOTHESES)}", flush=True)
    print(f"Deleted prior non-compliant rows: {deleted_labels}", flush=True)
    write_progress(conn, {"status": "started", "hypothesis_index": 0})
    write_research_log(conn, deleted_labels, "running", data_notes)

    for i, h in enumerate(HYPOTHESES, 1):
        if saved_count(conn) >= TARGET_SAVED:
            break

        print(f"[{i}/{len(HYPOTHESES)}] {h.name}", flush=True)
        write_progress(conn, {"status": "screening", "hypothesis_index": i, "hypothesis": h.name})

        if h.expression.strip() in blocklist:
            append_csv(TRIALS_CSV, row_for(h, "exact_blocked", False, None, None, "Exact expression exists in blocklist"))
            print("  exact_blocked", flush=True)
            continue

        try:
            screen = harness.eval_single(h.expression, split="train", universe_name=UNIVERSE)
        except Exception as exc:
            append_csv(TRIALS_CSV, row_for(h, "screen_error", False, None, None, str(exc)))
            print(f"  screen_error {exc}", flush=True)
            continue

        harness.log_trial(conn, h.expression, screen.get("sharpe", 0) if isinstance(screen, dict) else 0, saved=False)
        if not screen.get("success", False):
            append_csv(TRIALS_CSV, row_for(h, "screen_error", False, None, screen, screen.get("error", "")))
            print(f"  screen_error {screen.get('error', '')}", flush=True)
            continue

        status = "screen_pass" if screen_pass(screen) else "screen_fail"
        append_csv(TRIALS_CSV, row_for(h, status, False, None, screen))
        print(
            f"  {status} SR={screen['sharpe']:+.3f} Fit={screen['fitness']:.3f} "
            f"TO={screen['turnover']:.3f}",
            flush=True,
        )
        if status != "screen_pass":
            write_progress(conn, {"status": "screen_fail", "hypothesis_index": i, "hypothesis": h.name})
            continue

        write_progress(conn, {"status": "full_gate", "hypothesis_index": i, "hypothesis": h.name})
        try:
            full = harness.eval_full(h.expression, conn)
        except Exception as exc:
            append_csv(FULL_CSV, row_for(h, "full_error", False, None, None, str(exc)))
            print(f"  full_error {exc}", flush=True)
            continue

        gate_status = "gate_pass" if gates_pass(full) else "gate_fail"
        append_csv(FULL_CSV, row_for(h, gate_status, False, None, full, full.get("error", "")))
        print(
            f"  {gate_status} SR={full.get('is_sharpe', 0):+.3f} "
            f"Fit={full.get('is_fitness', 0):.3f} TO={full.get('turnover', 0):.3f} "
            f"IC={full.get('ic_mean', 0):+.4f} H1={full.get('stability_h1', 0):+.3f} "
            f"H2={full.get('stability_h2', 0):+.3f}",
            flush=True,
        )
        if gate_status != "gate_pass":
            write_progress(conn, {"status": "gate_fail", "hypothesis_index": i, "hypothesis": h.name})
            continue

        reasoning = (
            f"[{LABEL}] {STRATEGY_NAME}. Candidate={h.name}; family={h.family}; "
            f"train-only mechanism={h.mechanism}; orthogonal reason={h.orthogonal_reason}; "
            f"fixed dynamic universe={UNIVERSE}; strict gates: train Sharpe>5, train fitness>5, "
            f"turnover<=1.0, IC>0; subindustry neutralization; exact-copy blocklist; corr<=0.70; "
            f"no parameter sweeps."
        )
        saved = harness.save_alpha(conn, h.expression, reasoning, full, LABEL)
        alpha_id = alpha_id_for_expr(conn, h.expression) if saved else None
        append_csv(
            SELECTION_CSV,
            row_for(h, "selection_saved" if saved else "selection_corr_reject", bool(saved), alpha_id, full),
        )
        write_saved_snapshot(conn)
        write_research_log(conn, deleted_labels, "running", data_notes)
        print(
            f"  {'saved' if saved else 'corr_reject'} id={alpha_id or ''} "
            f"saved_total={saved_count(conn)}/{TARGET_SAVED}",
            flush=True,
        )
        write_progress(
            conn,
            {
                "status": "saved" if saved else "corr_reject",
                "hypothesis_index": i,
                "hypothesis": h.name,
                "alpha_id": alpha_id,
            },
        )

    write_saved_snapshot(conn)
    final = "completed_target" if saved_count(conn) >= TARGET_SAVED else "exhausted_preregistered_hypotheses"
    write_progress(conn, {"status": final, "hypothesis_index": len(HYPOTHESES)})
    write_research_log(conn, deleted_labels, final, data_notes)
    saved_n = saved_count(conn)
    print(f"Final status: {final}; saved {saved_n}/{TARGET_SAVED}", flush=True)
    conn.close()
    if final != "completed_target":
        raise SystemExit(f"Only saved {saved_n} alphas; target is {TARGET_SAVED}.")


if __name__ == "__main__":
    main()
