"""Internal crossing and broker-order netting for multi-strategy execution.

The netting boundary is intentionally explicit: only child orders in the same
broker bucket can cross. A bucket should encode asset type, venue, account,
instrument type, order style, and execution window, for example:

    equity:ib:paper:stock:MOC:close

Strategies keep their own virtual books. The broker only sees the residual
net order after internal crosses, while fill allocation pushes executions back
to the originating strategies for PnL attribution.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class StrategyTarget:
    strategy_id: str
    asset_type: str
    venue: str
    account: str
    symbol: str
    target_qty: float
    target_notional: float
    price: float
    weight: float | None = None
    order_type: str = "MOC"
    bucket: str | None = None
    signal_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def execution_bucket(self) -> str:
        if self.bucket:
            return self.bucket
        return f"{self.asset_type}:{self.venue}:{self.account}:{self.order_type}"


@dataclass(frozen=True)
class ChildOrder:
    child_order_id: str
    batch_id: str
    strategy_id: str
    asset_type: str
    venue: str
    account: str
    bucket: str
    symbol: str
    delta_qty: float
    reference_price: float
    order_type: str = "MOC"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "BUY" if self.delta_qty > 0 else "SELL"

    @property
    def abs_qty(self) -> float:
        return abs(float(self.delta_qty))

    @property
    def notional(self) -> float:
        return self.abs_qty * float(self.reference_price)


@dataclass(frozen=True)
class InternalCross:
    cross_id: str
    batch_id: str
    bucket: str
    symbol: str
    buy_strategy_id: str
    sell_strategy_id: str
    quantity: float
    price: float

    @property
    def notional(self) -> float:
        return abs(float(self.quantity)) * float(self.price)


@dataclass(frozen=True)
class NetOrder:
    net_order_id: str
    batch_id: str
    asset_type: str
    venue: str
    account: str
    bucket: str
    symbol: str
    quantity: float
    reference_price: float
    order_type: str = "MOC"
    child_order_ids: tuple[str, ...] = ()
    residual_child_qty: dict[str, float] = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"

    @property
    def abs_qty(self) -> float:
        return abs(float(self.quantity))

    @property
    def notional(self) -> float:
        return self.abs_qty * float(self.reference_price)


@dataclass(frozen=True)
class FillAllocation:
    allocation_id: str
    batch_id: str
    strategy_id: str
    bucket: str
    symbol: str
    quantity: float
    price: float
    source: str
    source_id: str
    fee: float = 0.0

    @property
    def notional(self) -> float:
        return abs(float(self.quantity)) * float(self.price)


@dataclass(frozen=True)
class NettingResult:
    batch_id: str
    child_orders: list[ChildOrder]
    internal_crosses: list[InternalCross]
    net_orders: list[NetOrder]
    internal_allocations: list[FillAllocation]

    @property
    def gross_child_notional(self) -> float:
        return float(sum(o.notional for o in self.child_orders))

    @property
    def external_notional(self) -> float:
        return float(sum(o.notional for o in self.net_orders))

    @property
    def crossed_notional(self) -> float:
        return float(sum(c.notional for c in self.internal_crosses))

    @property
    def compression_ratio(self) -> float:
        if self.gross_child_notional <= 0:
            return 0.0
        return 1.0 - (self.external_notional / self.gross_child_notional)


def build_child_orders(
    *,
    batch_id: str,
    targets: list[StrategyTarget],
    current_positions: dict[tuple[str, str], float],
    min_order_value: float | dict[str, float] = 0.0,
    strategy_routes: dict[str, dict[str, str]] | None = None,
    latest_prices: dict[str, float] | None = None,
) -> list[ChildOrder]:
    """Convert target positions into strategy-level child deltas.

    `current_positions` is keyed by `(strategy_id, symbol)`. Any existing
    strategy position not present in targets is flattened when a route and
    current price are available.
    """
    target_by_strategy_symbol = {(t.strategy_id, t.symbol): t for t in targets}
    all_keys = set(target_by_strategy_symbol) | set(current_positions)
    child_orders: list[ChildOrder] = []
    strategy_routes = strategy_routes or {}
    latest_prices = latest_prices or {}

    # Use the target metadata for route/price. For flatten-only rows without a
    # current target, inherit the last known target for that strategy if present.
    route_by_strategy: dict[str, StrategyTarget] = {}
    price_by_symbol: dict[str, float] = {}
    for target in targets:
        route_by_strategy.setdefault(target.strategy_id, target)
        if target.price > 0:
            price_by_symbol[target.symbol] = float(target.price)

    for strategy_id, symbol in sorted(all_keys):
        target = target_by_strategy_symbol.get((strategy_id, symbol))
        current_qty = float(current_positions.get((strategy_id, symbol), 0.0) or 0.0)
        target_qty = float(target.target_qty if target is not None else 0.0)
        delta = target_qty - current_qty
        if abs(delta) < 1e-9:
            continue

        route = target or route_by_strategy.get(strategy_id)
        route_override = strategy_routes.get(strategy_id, {})
        if route is None and not route_override:
            continue
        price = float(
            (
                target.price
                if target is not None
                else price_by_symbol.get(symbol) or latest_prices.get(symbol) or 0.0
            )
            or 0.0
        )
        if price <= 0:
            continue
        threshold = _min_order_value(min_order_value, strategy_id)
        if abs(delta) * price < threshold:
            continue

        child_orders.append(
            ChildOrder(
                child_order_id=new_id("child"),
                batch_id=batch_id,
                strategy_id=strategy_id,
                asset_type=_route_attr(route, route_override, "asset_type"),
                venue=_route_attr(route, route_override, "venue"),
                account=_route_attr(route, route_override, "account"),
                bucket=route.execution_bucket() if route is not None else str(route_override["bucket"]),
                symbol=symbol,
                delta_qty=float(delta),
                reference_price=price,
                order_type=_route_attr(route, route_override, "order_type", default="MOC"),
                metadata=dict(route.metadata) if route is not None else {},
            )
        )
    return child_orders


def net_child_orders(*, batch_id: str, child_orders: list[ChildOrder]) -> NettingResult:
    """Cross opposing child deltas and produce residual broker orders."""
    groups: dict[tuple[str, str], list[ChildOrder]] = defaultdict(list)
    for order in child_orders:
        groups[(order.bucket, order.symbol)].append(order)

    crosses: list[InternalCross] = []
    internal_allocations: list[FillAllocation] = []
    net_orders: list[NetOrder] = []

    for (bucket, symbol), orders in sorted(groups.items()):
        buys = [{"order": o, "remaining": o.abs_qty} for o in orders if o.delta_qty > 0]
        sells = [{"order": o, "remaining": o.abs_qty} for o in orders if o.delta_qty < 0]
        price = _group_reference_price(orders)

        bi = 0
        si = 0
        while bi < len(buys) and si < len(sells):
            qty = min(buys[bi]["remaining"], sells[si]["remaining"])
            if qty > 1e-9:
                buy_order = buys[bi]["order"]
                sell_order = sells[si]["order"]
                cross_id = new_id("cross")
                crosses.append(
                    InternalCross(
                        cross_id=cross_id,
                        batch_id=batch_id,
                        bucket=bucket,
                        symbol=symbol,
                        buy_strategy_id=buy_order.strategy_id,
                        sell_strategy_id=sell_order.strategy_id,
                        quantity=float(qty),
                        price=price,
                    )
                )
                internal_allocations.extend([
                    FillAllocation(
                        allocation_id=new_id("alloc"),
                        batch_id=batch_id,
                        strategy_id=buy_order.strategy_id,
                        bucket=bucket,
                        symbol=symbol,
                        quantity=float(qty),
                        price=price,
                        source="internal_cross",
                        source_id=cross_id,
                    ),
                    FillAllocation(
                        allocation_id=new_id("alloc"),
                        batch_id=batch_id,
                        strategy_id=sell_order.strategy_id,
                        bucket=bucket,
                        symbol=symbol,
                        quantity=-float(qty),
                        price=price,
                        source="internal_cross",
                        source_id=cross_id,
                    ),
                ])
                buys[bi]["remaining"] -= qty
                sells[si]["remaining"] -= qty
            if buys[bi]["remaining"] <= 1e-9:
                bi += 1
            if sells[si]["remaining"] <= 1e-9:
                si += 1

        residual_orders: list[ChildOrder] = []
        net_qty = 0.0
        for row in buys:
            if row["remaining"] > 1e-9:
                residual_orders.append(row["order"])
                net_qty += float(row["remaining"])
        for row in sells:
            if row["remaining"] > 1e-9:
                residual_orders.append(row["order"])
                net_qty -= float(row["remaining"])

        if abs(net_qty) > 1e-9:
            exemplar = residual_orders[0]
            net_orders.append(
                NetOrder(
                    net_order_id=new_id("net"),
                    batch_id=batch_id,
                    asset_type=exemplar.asset_type,
                    venue=exemplar.venue,
                    account=exemplar.account,
                    bucket=bucket,
                    symbol=symbol,
                    quantity=float(net_qty),
                    reference_price=price,
                    order_type=exemplar.order_type,
                    child_order_ids=tuple(o.child_order_id for o in residual_orders),
                    residual_child_qty={
                        row["order"].child_order_id: float(row["remaining"])
                        for row in buys + sells
                        if row["remaining"] > 1e-9
                    },
                )
            )

    return NettingResult(
        batch_id=batch_id,
        child_orders=child_orders,
        internal_crosses=crosses,
        net_orders=net_orders,
        internal_allocations=internal_allocations,
    )


def allocate_external_fill(
    *,
    batch_id: str,
    net_order: NetOrder,
    child_orders: list[ChildOrder],
    filled_qty: float,
    fill_price: float,
    fee: float = 0.0,
) -> list[FillAllocation]:
    """Allocate one broker fill back to same-side residual child orders."""
    residual_qty = dict(net_order.residual_child_qty or {})
    same_side = [
        o for o in child_orders
        if o.bucket == net_order.bucket
        and o.symbol == net_order.symbol
        and (o.delta_qty > 0) == (net_order.quantity > 0)
        and (not residual_qty or residual_qty.get(o.child_order_id, 0.0) > 0)
    ]
    total_child_qty = sum(float(residual_qty.get(o.child_order_id, abs(o.delta_qty))) for o in same_side)
    if total_child_qty <= 0 or abs(filled_qty) <= 0:
        return []

    sign = 1.0 if net_order.quantity > 0 else -1.0
    abs_fill = abs(float(filled_qty))
    allocations: list[FillAllocation] = []
    allocated_qty = 0.0
    allocated_fee = 0.0
    for idx, child in enumerate(same_side):
        if idx == len(same_side) - 1:
            qty_abs = max(0.0, abs_fill - allocated_qty)
            fee_part = max(0.0, float(fee) - allocated_fee)
        else:
            share = float(residual_qty.get(child.child_order_id, abs(child.delta_qty))) / total_child_qty
            qty_abs = abs_fill * share
            fee_part = float(fee) * share
            allocated_qty += qty_abs
            allocated_fee += fee_part
        if qty_abs <= 1e-9:
            continue
        allocations.append(
            FillAllocation(
                allocation_id=new_id("alloc"),
                batch_id=batch_id,
                strategy_id=child.strategy_id,
                bucket=child.bucket,
                symbol=child.symbol,
                quantity=sign * qty_abs,
                price=float(fill_price),
                source="external_fill",
                source_id=net_order.net_order_id,
                fee=float(fee_part),
            )
        )
    return allocations


def _group_reference_price(orders: list[ChildOrder]) -> float:
    notionals = [o.notional for o in orders if o.reference_price > 0]
    qtys = [o.abs_qty for o in orders if o.reference_price > 0]
    if sum(qtys) <= 0:
        return 0.0
    return float(sum(notionals) / sum(qtys))


def _min_order_value(value: float | dict[str, float], strategy_id: str) -> float:
    if isinstance(value, dict):
        return float(value.get(strategy_id, 0.0) or 0.0)
    return float(value or 0.0)


def _route_attr(
    route: StrategyTarget | None,
    fallback: dict[str, str],
    name: str,
    *,
    default: str = "unknown",
) -> str:
    if route is not None:
        return str(getattr(route, name))
    return str(fallback.get(name) or default)
