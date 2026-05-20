"""Apply the production IB MOC share-rounding/min-order execution layer to AUCT.

Production path in prod/moc_trader.py:
  1. Convert target weights to rounded integer target shares.
  2. Compute order diffs versus current IB shares.
  3. Skip order diffs with abs(qty) * price < CFG["execution"]["min_order_value"].

This script applies that same recursive held-share logic to the saved AUCT
validation-selected QP weights. It does not change alpha research or QP
settings; it only evaluates the executable-position layer.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.netting import build_child_orders
from src.strategies.registry import _weights_to_targets, parse_strategy_config


BARS_PER_YEAR = 252
BOOK = 500_000.0
START = "2025-04-01"
END = "2026-05-14"
WEIGHT_LABEL = "ic_wt_qp_styleppca_s10_l2_k50"
STRATEGY_ID = "auct_research_exec_filter"


def _ann_sr(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1 or clean.std() <= 0:
        return float("nan")
    return float(clean.mean() / clean.std() * math.sqrt(BARS_PER_YEAR))


def _max_dd(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    eq = (1.0 + clean).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _load_weights() -> pd.DataFrame:
    path = OUT_DIR / "test_selected_strategy_weights.parquet"
    weights = pd.read_parquet(path)
    if isinstance(weights.index, pd.MultiIndex):
        weights = weights.xs(WEIGHT_LABEL, level="series")
    weights.index = pd.to_datetime(weights.index)
    return weights.sort_index()


def _load_close(columns: pd.Index, index: pd.Index) -> pd.DataFrame:
    close = pd.read_parquet(ROOT / "data/fmp_cache/matrices/close.parquet")
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    return close.reindex(index=index, columns=columns)


def _simulate_prod_execution(
    target_weights: pd.DataFrame,
    close: pd.DataFrame,
    *,
    book: float,
    min_order_value: float,
    commission_per_share: float,
    per_order_min: float,
    sec_fee_per_dollar: float,
    borrow_bps_annual: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return executable weights, orders, and per-day execution stats."""
    tickers = list(target_weights.columns)
    strategy_config = parse_strategy_config({
        "strategy_id": STRATEGY_ID,
        "name": "AUCT Research Execution Filter",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": "equity:ib:paper:stock:MOC:close",
        },
        "portfolio": {"book": book},
        "execution": {"order_type": "MOC", "min_order_value": min_order_value},
        "signal": {},
        "metadata": {"family": "auct_research"},
    })
    current_positions: dict[tuple[str, str], float] = {}
    exec_weights = []
    order_rows = []
    stat_rows = []

    for dt in target_weights.index:
        price = close.loc[dt].replace(0, np.nan)
        clean_weights = target_weights.loc[dt].replace([np.inf, -np.inf], np.nan).dropna().to_dict()
        targets = _weights_to_targets(
            strategy_config,
            weights=clean_weights,
            prices=price,
            signal_time=str(dt.date()),
            metadata={"source_weight_label": WEIGHT_LABEL},
        )
        latest_prices = {str(symbol): float(px) for symbol, px in price.dropna().items() if float(px) > 0}
        raw_children = build_child_orders(
            batch_id=f"auct_{dt:%Y%m%d}_raw",
            targets=targets,
            current_positions=current_positions,
            min_order_value=0.0,
            strategy_routes={STRATEGY_ID: strategy_config.route},
            latest_prices=latest_prices,
        )
        kept_children = build_child_orders(
            batch_id=f"auct_{dt:%Y%m%d}",
            targets=targets,
            current_positions=current_positions,
            min_order_value={STRATEGY_ID: float(min_order_value)},
            strategy_routes={STRATEGY_ID: strategy_config.route},
            latest_prices=latest_prices,
        )

        for child in kept_children:
            key = (child.strategy_id, child.symbol)
            next_qty = float(current_positions.get(key, 0.0) or 0.0) + float(child.delta_qty)
            if abs(next_qty) < 1e-9:
                current_positions.pop(key, None)
            else:
                current_positions[key] = next_qty

        current_share_series = pd.Series(
            {
                symbol: qty
                for (strategy_id, symbol), qty in current_positions.items()
                if strategy_id == STRATEGY_ID
            },
            dtype=float,
        ).reindex(tickers).fillna(0.0)
        exec_w = (current_share_series * price.fillna(0.0)) / book
        exec_weights.append(exec_w.rename(dt))

        raw_notional = float(sum(child.notional for child in raw_children))
        kept_notional = float(sum(child.notional for child in kept_children))
        order_count = len(kept_children)
        skipped_count = len(raw_children) - len(kept_children)

        if order_count:
            commission = float(
                sum(max(abs(child.delta_qty) * commission_per_share, per_order_min) for child in kept_children)
            )
        else:
            commission = 0.0
        sell_value = float(sum(child.notional for child in kept_children if child.delta_qty < 0))
        sec_fee = sell_value * sec_fee_per_dollar
        short_dollars = float((-current_share_series.where(current_share_series < 0, 0)).mul(price.fillna(0.0)).sum())
        borrow = short_dollars * (borrow_bps_annual / 1e4) / BARS_PER_YEAR
        total_cost = commission + sec_fee + borrow

        stat_rows.append({
            "date": dt,
            "raw_order_count": len(raw_children),
            "kept_order_count": order_count,
            "skipped_order_count": skipped_count,
            "raw_turnover": raw_notional / book,
            "executed_turnover": kept_notional / book,
            "skipped_turnover": (raw_notional - kept_notional) / book,
            "gross_weight_l1": float(exec_w.abs().sum()),
            "net_weight": float(exec_w.sum()),
            "commission": commission,
            "sec_fee": sec_fee,
            "borrow": borrow,
            "cost": total_cost / book,
        })

        if kept_children:
            frame = pd.DataFrame({
                "date": dt,
                "ticker": [child.symbol for child in kept_children],
                "shares": [child.delta_qty for child in kept_children],
                "price": [child.reference_price for child in kept_children],
            })
            frame["order_value"] = frame["shares"].abs() * frame["price"]
            order_rows.append(frame)

    weights = pd.DataFrame(exec_weights)
    orders = pd.concat(order_rows, ignore_index=True) if order_rows else pd.DataFrame(
        columns=["date", "ticker", "shares", "price", "order_value"]
    )
    stats = pd.DataFrame(stat_rows).set_index("date")
    return weights, orders, stats


def _metrics(label: str, gross: pd.Series, cost: pd.Series, net: pd.Series, turnover: pd.Series) -> dict:
    g = gross.loc[START:END]
    c = cost.loc[START:END]
    n = net.loc[START:END]
    t = turnover.loc[START:END]
    return {
        "series": label,
        "start": START,
        "end": END,
        "n_bars": int(n.dropna().shape[0]),
        "SR_gross": _ann_sr(g),
        "SR_net": _ann_sr(n),
        "ret_ann_gross": float(g.mean() * BARS_PER_YEAR),
        "ret_ann_net": float(n.mean() * BARS_PER_YEAR),
        "cost_ann": float(c.mean() * BARS_PER_YEAR),
        "turnover_mean": float(t.mean()),
        "turnover_median": float(t.median()),
        "turnover_min": float(t.min()),
        "turnover_max": float(t.max()),
        "max_dd_net": _max_dd(n),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strategy_cfg = json.loads((ROOT / "prod/config/strategy.json").read_text())
    research_cfg = json.loads((ROOT / "prod/config/research_equity.json").read_text())
    min_order_value = float(strategy_cfg["execution"].get("min_order_value", 0.0))
    fee_params = research_cfg["fees"]["params"]

    target_weights = _load_weights()
    close = _load_close(target_weights.columns, target_weights.index)
    ret = close.pct_change(fill_method=None)

    exec_weights, orders, stats = _simulate_prod_execution(
        target_weights,
        close,
        book=BOOK,
        min_order_value=min_order_value,
        commission_per_share=float(fee_params.get("commission_per_share", 0.0045)),
        per_order_min=float(fee_params.get("per_order_min", 0.35)),
        sec_fee_per_dollar=float(fee_params.get("sec_fee_per_dollar", 27.80e-6)),
        borrow_bps_annual=float(fee_params.get("borrow_bps_annual", 50.0)),
    )

    gross = (exec_weights * ret.shift(-1)).sum(axis=1).fillna(0.0)
    cost = stats["cost"].reindex(gross.index).fillna(0.0)
    net = gross - cost

    continuous_returns = pd.read_parquet(OUT_DIR / "test_selected_strategy_returns.parquet")
    continuous = {
        "gross": continuous_returns[f"{WEIGHT_LABEL}_gross"],
        "cost": continuous_returns[f"{WEIGHT_LABEL}_cost"],
        "net": continuous_returns[f"{WEIGHT_LABEL}_net"],
        "turnover": continuous_returns[f"{WEIGHT_LABEL}_turnover"],
    }

    rows = [
        _metrics(
            "auct_continuous_qp_before_prod_exec_filter",
            continuous["gross"],
            continuous["cost"],
            continuous["net"],
            continuous["turnover"],
        ),
        _metrics(
            f"auct_qp_prod_share_round_min_order_{int(min_order_value)}",
            gross,
            cost,
            net,
            stats["executed_turnover"],
        ),
    ]
    summary = pd.DataFrame(rows)
    turnover_stats = stats.loc[START:END].agg({
        "raw_order_count": ["mean", "median", "min", "max"],
        "kept_order_count": ["mean", "median", "min", "max"],
        "skipped_order_count": ["mean", "median", "min", "max"],
        "raw_turnover": ["mean", "median", "min", "max"],
        "executed_turnover": ["mean", "median", "min", "max"],
        "skipped_turnover": ["mean", "median", "min", "max"],
        "gross_weight_l1": ["mean", "median", "min", "max"],
    }).T

    summary.to_csv(OUT_DIR / "auct_prod_execution_filter_summary.csv", index=False)
    stats.to_csv(OUT_DIR / "auct_prod_execution_filter_daily_stats.csv")
    orders.to_parquet(OUT_DIR / "auct_prod_execution_filter_orders.parquet", index=False)
    exec_weights.to_parquet(OUT_DIR / "auct_prod_execution_filter_weights.parquet")
    pd.DataFrame({"gross": gross, "cost": cost, "net": net, "turnover": stats["executed_turnover"]}).to_parquet(
        OUT_DIR / "auct_prod_execution_filter_returns.parquet"
    )
    turnover_stats.to_csv(OUT_DIR / "auct_prod_execution_filter_turnover_stats.csv")

    print("=== production execution filter ===", flush=True)
    print(f"min_order_value=${min_order_value:,.0f}", flush=True)
    print(summary.to_markdown(index=False, floatfmt=".4f"), flush=True)
    print("\n=== daily execution stats, production TEST ===", flush=True)
    print(turnover_stats.to_markdown(floatfmt=".4f"), flush=True)


if __name__ == "__main__":
    main()
