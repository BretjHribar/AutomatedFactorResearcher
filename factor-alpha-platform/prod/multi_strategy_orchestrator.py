"""Multi-strategy execution orchestrator with internal netting.

Default mode is shadow: compute targets, child orders, internal crosses, and
residual net orders, then persist them without touching broker positions.

Use `--mode paper-sim` to apply strategy-level virtual fills at the reference
price so the monitor can build per-strategy paper PnL curves. Actual IB paper
submission is intentionally guarded and should only be enabled after shadow
reconciliation has been reviewed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.ledger import ExecutionLedger
from src.execution.netting import allocate_external_fill, build_child_orders, net_child_orders
from src.strategies.registry import (
    StaleStrategySignal,
    build_targets,
    load_latest_price_map,
    load_strategy_configs,
)


DEFAULT_CONFIG_DIR = PROJECT_ROOT / "prod" / "config" / "strategies"
DEFAULT_LEDGER_DB = PROJECT_ROOT / "data" / "execution_ledger.db"


def run(
    *,
    mode: str = "shadow",
    config_dir: Path = DEFAULT_CONFIG_DIR,
    ledger_db: Path = DEFAULT_LEDGER_DB,
    strategy_ids: set[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"shadow", "paper-sim", "ib-paper"}:
        raise ValueError(f"unsupported mode {mode!r}")

    configs = [
        cfg for cfg in load_strategy_configs(config_dir)
        if cfg.enabled and (strategy_ids is None or cfg.strategy_id in strategy_ids)
    ]
    ledger = ExecutionLedger(ledger_db)
    batch_id = ledger.start_batch(
        mode=mode,
        metadata={
            "config_dir": str(config_dir),
            "strategies": [c.strategy_id for c in configs],
        },
    )

    try:
        targets = []
        latest_prices: dict[str, float] = {}
        active_configs = []
        skipped_strategies: list[dict[str, str]] = []
        for cfg in configs:
            try:
                cfg_targets = build_targets(cfg, root=PROJECT_ROOT)
            except StaleStrategySignal as exc:
                skipped_strategies.append({
                    "strategy_id": cfg.strategy_id,
                    "reason": str(exc),
                })
                continue
            targets.extend(cfg_targets)
            latest_prices.update(load_latest_price_map(cfg, root=PROJECT_ROOT))
            active_configs.append(cfg)
        ledger.record_targets(batch_id, targets)

        current_positions = ledger.current_positions(c.strategy_id for c in active_configs)
        child_orders = build_child_orders(
            batch_id=batch_id,
            targets=targets,
            current_positions=current_positions,
            min_order_value={c.strategy_id: c.min_order_value for c in active_configs},
            strategy_routes={c.strategy_id: c.route for c in active_configs},
            latest_prices=latest_prices,
        )
        netted = net_child_orders(batch_id=batch_id, child_orders=child_orders)
        ledger.record_netting_result(netted)

        route_by_strategy = {cfg.strategy_id: cfg.route for cfg in active_configs}
        simulated_external_allocations = []
        if mode == "paper-sim":
            ledger.mark_to_market(prices=latest_prices, source="paper_sim_pre_trade_mark")
            for net_order in netted.net_orders:
                simulated_external_allocations.extend(
                    allocate_external_fill(
                        batch_id=batch_id,
                        net_order=net_order,
                        child_orders=netted.child_orders,
                        filled_qty=net_order.quantity,
                        fill_price=net_order.reference_price,
                        fee=0.0,
                    )
                )
            ledger.record_allocations(simulated_external_allocations)
            ledger.apply_allocations_to_positions(
                [*netted.internal_allocations, *simulated_external_allocations],
                route_by_strategy=route_by_strategy,
            )
            ledger.mark_to_market(prices=latest_prices, source="paper_sim_post_trade_mark")
        elif mode == "ib-paper":
            _submit_ib_paper_guarded(netted.net_orders)

        summary = {
            "batch_id": batch_id,
            "mode": mode,
            "strategies": [c.strategy_id for c in active_configs],
            "skipped_strategies": skipped_strategies,
            "n_targets": len(targets),
            "n_child_orders": len(netted.child_orders),
            "n_internal_crosses": len(netted.internal_crosses),
            "n_net_orders": len(netted.net_orders),
            "gross_child_notional": round(netted.gross_child_notional, 2),
            "crossed_notional": round(netted.crossed_notional, 2),
            "external_notional": round(netted.external_notional, 2),
            "compression_ratio": round(netted.compression_ratio, 4),
        }
        ledger.finish_batch(batch_id, status="completed", metadata=summary)
        return summary
    except Exception as exc:
        ledger.finish_batch(
            batch_id,
            status="failed",
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def _submit_ib_paper_guarded(net_orders) -> None:
    """Submit residual IB paper MOC orders through the existing IB wrapper."""
    if os.environ.get("ALLOW_MULTI_STRATEGY_IB_PAPER_ORDERS") != "1":
        raise RuntimeError(
            "IB paper submission blocked; set ALLOW_MULTI_STRATEGY_IB_PAPER_ORDERS=1 after shadow review"
        )
    order_diffs = {
        order.symbol: int(round(order.quantity))
        for order in net_orders
        if order.venue == "ib" and order.account == "paper" and abs(order.quantity) >= 1
    }
    if not order_diffs:
        return
    from prod.moc_trader import IBConnection, IB_HOST, IB_PORT_PAPER, IB_CLIENT_ID_ORDER_ENTRY

    conn = IBConnection(host=IB_HOST, port=IB_PORT_PAPER, client_id=IB_CLIENT_ID_ORDER_ENTRY)
    try:
        if not conn.connect():
            raise RuntimeError("could not connect to IB paper gateway")
        conn.submit_moc_orders(order_diffs)
    finally:
        conn.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-strategy netting orchestrator")
    parser.add_argument("--mode", choices=["shadow", "paper-sim", "ib-paper"], default="shadow")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--ledger-db", type=Path, default=DEFAULT_LEDGER_DB)
    parser.add_argument("--strategy-id", action="append", default=None,
                        help="Run only this strategy_id; can be passed multiple times")
    args = parser.parse_args()
    summary = run(
        mode=args.mode,
        config_dir=args.config_dir,
        ledger_db=args.ledger_db,
        strategy_ids=set(args.strategy_id) if args.strategy_id else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
