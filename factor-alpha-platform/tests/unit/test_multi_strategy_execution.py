from __future__ import annotations

import json

import pandas as pd
import pytest

from src.execution.ledger import ExecutionLedger
from src.execution.netting import (
    FillAllocation,
    NetOrder,
    StrategyTarget,
    allocate_external_fill,
    build_child_orders,
    net_child_orders,
)
from src.strategies.registry import (
    StaleStrategySignal,
    build_targets,
    load_latest_price_map,
    load_strategy_configs,
    parse_strategy_config,
)


DEFAULT_BUCKET = "equity:ib:paper:stock:MOC:close"


def _target(
    strategy_id: str,
    symbol: str,
    qty: float,
    price: float = 10.0,
    bucket: str = DEFAULT_BUCKET,
    venue: str = "ib",
    account: str = "paper",
) -> StrategyTarget:
    return StrategyTarget(
        strategy_id=strategy_id,
        asset_type="equity",
        venue=venue,
        account=account,
        bucket=bucket,
        symbol=symbol,
        target_qty=qty,
        target_notional=qty * price,
        price=price,
        weight=None,
        order_type="MOC",
    )


def _write_close(root, prices: dict[str, float], date: str = "2026-05-12") -> str:
    path = root / "data/fmp_cache/matrices/close.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prices, index=pd.to_datetime([date])).to_parquet(path)
    return "data/fmp_cache/matrices/close.parquet"


def _write_artifact_strategy(
    root,
    config_dir,
    strategy_id: str,
    weights: dict[str, float],
    *,
    name: str | None = None,
    book: float = 100000.0,
    min_order_value: float = 0.0,
    enabled: bool = True,
    bucket: str = DEFAULT_BUCKET,
    signal_date: str = "2026-05-12",
    max_signal_lag_days: int | None = None,
    stale_action: str | None = None,
) -> None:
    weights_rel = f"artifacts/{strategy_id}.weights_tail.parquet"
    weights_path = root / weights_rel
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weights, index=pd.to_datetime([signal_date])).to_parquet(weights_path)
    signal = {
        "adapter": "aipt_weights_tail_artifact",
        "weights_path": weights_rel,
        "price_matrix": "data/fmp_cache/matrices/close.parquet",
    }
    if max_signal_lag_days is not None:
        signal["max_signal_lag_days"] = max_signal_lag_days
    if stale_action is not None:
        signal["stale_action"] = stale_action
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{strategy_id}.json").write_text(json.dumps({
        "strategy_id": strategy_id,
        "name": name or strategy_id,
        "enabled": enabled,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": bucket,
        },
        "portfolio": {"book": book},
        "execution": {"order_type": "MOC", "min_order_value": min_order_value},
        "signal": signal,
        "metadata": {
            "family": "test",
            "description": f"Test strategy {strategy_id}",
            "research": {"reported_val_test_net_sharpe": 1.23},
        },
    }), encoding="utf-8")


def test_netting_crosses_between_strategies_and_allocates_residual_fill() -> None:
    batch_id = "batch_test"
    targets = [_target("long_a", "XYZ", 100), _target("short_b", "XYZ", -40)]
    children = build_child_orders(batch_id=batch_id, targets=targets, current_positions={})

    result = net_child_orders(batch_id=batch_id, child_orders=children)

    assert len(result.internal_crosses) == 1
    assert result.internal_crosses[0].quantity == 40
    assert len(result.net_orders) == 1
    net = result.net_orders[0]
    assert net.quantity == 60
    assert round(result.compression_ratio, 6) == round(1.0 - 600.0 / 1400.0, 6)

    internal_by_strategy = {a.strategy_id: a.quantity for a in result.internal_allocations}
    assert internal_by_strategy == {"long_a": 40, "short_b": -40}

    external = allocate_external_fill(
        batch_id=batch_id,
        net_order=net,
        child_orders=children,
        filled_qty=net.quantity,
        fill_price=10.0,
    )
    assert [(a.strategy_id, a.quantity) for a in external] == [("long_a", 60.0)]


def test_external_fill_uses_residual_child_quantities_after_partial_cross() -> None:
    batch_id = "batch_test"
    targets = [
        _target("long_a", "XYZ", 100),
        _target("long_c", "XYZ", 50),
        _target("short_b", "XYZ", -90),
    ]
    children = build_child_orders(batch_id=batch_id, targets=targets, current_positions={})

    result = net_child_orders(batch_id=batch_id, child_orders=children)
    net = result.net_orders[0]

    assert net.quantity == 60
    residual_by_strategy = {
        child.strategy_id: net.residual_child_qty[child.child_order_id]
        for child in children
        if child.child_order_id in net.residual_child_qty
    }
    assert residual_by_strategy == {"long_a": 10.0, "long_c": 50.0}

    external = allocate_external_fill(
        batch_id=batch_id,
        net_order=net,
        child_orders=children,
        filled_qty=net.quantity,
        fill_price=10.0,
        fee=6.0,
    )
    assert [(a.strategy_id, a.quantity, a.fee) for a in external] == [
        ("long_a", 10.0, 1.0),
        ("long_c", 50.0, 5.0),
    ]


def test_netting_isolated_by_bucket_and_symbol() -> None:
    other_bucket = "equity:kucoin:paper:spot:MOC:close"
    targets = [
        _target("ib_long", "XYZ", 50, bucket=DEFAULT_BUCKET),
        _target("kucoin_short", "XYZ", -50, bucket=other_bucket, venue="kucoin"),
        _target("ib_long_other_symbol", "AAA", 10, bucket=DEFAULT_BUCKET),
        _target("ib_short_other_symbol", "BBB", -10, bucket=DEFAULT_BUCKET),
    ]
    children = build_child_orders(batch_id="batch_test", targets=targets, current_positions={})

    result = net_child_orders(batch_id="batch_test", child_orders=children)

    assert result.internal_crosses == []
    assert round(result.compression_ratio, 6) == 0.0
    assert sorted((o.bucket, o.symbol, o.quantity) for o in result.net_orders) == [
        (DEFAULT_BUCKET, "AAA", 10.0),
        (DEFAULT_BUCKET, "BBB", -10.0),
        (DEFAULT_BUCKET, "XYZ", 50.0),
        (other_bucket, "XYZ", -50.0),
    ]


def test_child_order_min_value_can_be_strategy_specific() -> None:
    targets = [
        _target("strict_strategy", "AAA", 10, price=10.0),
        _target("loose_strategy", "BBB", 10, price=10.0),
    ]

    children = build_child_orders(
        batch_id="batch_test",
        targets=targets,
        current_positions={},
        min_order_value={"strict_strategy": 200.0, "loose_strategy": 50.0},
    )

    assert [(child.strategy_id, child.symbol, child.notional) for child in children] == [
        ("loose_strategy", "BBB", 100.0),
    ]


def test_external_sell_fill_allocates_negative_quantities_and_fees() -> None:
    targets = [
        _target("seller_a", "XYZ", -100),
        _target("seller_c", "XYZ", -50),
        _target("buyer_b", "XYZ", 40),
    ]
    children = build_child_orders(batch_id="batch_test", targets=targets, current_positions={})
    result = net_child_orders(batch_id="batch_test", child_orders=children)
    net = result.net_orders[0]

    assert net.quantity == -110
    residual_by_strategy = {
        child.strategy_id: net.residual_child_qty[child.child_order_id]
        for child in children
        if child.child_order_id in net.residual_child_qty
    }
    assert residual_by_strategy == {"seller_a": 60.0, "seller_c": 50.0}

    external = allocate_external_fill(
        batch_id="batch_test",
        net_order=net,
        child_orders=children,
        filled_qty=55.0,
        fill_price=10.0,
        fee=11.0,
    )

    assert [a.strategy_id for a in external] == ["seller_a", "seller_c"]
    assert [a.quantity for a in external] == pytest.approx([-30.0, -25.0])
    assert [a.fee for a in external] == pytest.approx([6.0, 5.0])


def test_child_orders_use_current_positions_for_deltas_and_skip_noops() -> None:
    targets = [
        _target("strategy_a", "AAA", 100, price=20.0),
        _target("strategy_a", "BBB", 25, price=30.0),
    ]
    children = build_child_orders(
        batch_id="batch_test",
        targets=targets,
        current_positions={
            ("strategy_a", "AAA"): 40.0,
            ("strategy_a", "BBB"): 25.0,
        },
    )

    assert [(child.symbol, child.delta_qty, child.reference_price) for child in children] == [
        ("AAA", 60.0, 20.0),
    ]


def test_child_orders_flatten_stale_positions_with_strategy_route_and_latest_price() -> None:
    children = build_child_orders(
        batch_id="batch_test",
        targets=[],
        current_positions={("strategy_x", "OLD"): 25.0},
        min_order_value={"strategy_x": 100.0},
        strategy_routes={
            "strategy_x": {
                "asset_type": "equity",
                "venue": "ib",
                "account": "paper",
                "bucket": "equity:ib:paper:stock:MOC:close",
                "order_type": "MOC",
            }
        },
        latest_prices={"OLD": 8.0},
    )

    assert len(children) == 1
    child = children[0]
    assert child.strategy_id == "strategy_x"
    assert child.symbol == "OLD"
    assert child.delta_qty == -25.0
    assert child.reference_price == 8.0
    assert child.bucket == "equity:ib:paper:stock:MOC:close"


def test_ledger_attributes_positions_fees_and_mtm_by_strategy(tmp_path) -> None:
    db_path = tmp_path / "ledger.db"
    ledger = ExecutionLedger(db_path)
    batch_id = ledger.start_batch(mode="paper-sim", batch_id="batch_test")
    targets = [_target("long_a", "XYZ", 10), _target("short_b", "XYZ", -4)]
    children = build_child_orders(batch_id=batch_id, targets=targets, current_positions={})
    result = net_child_orders(batch_id=batch_id, child_orders=children)
    external = allocate_external_fill(
        batch_id=batch_id,
        net_order=result.net_orders[0],
        child_orders=children,
        filled_qty=result.net_orders[0].quantity,
        fill_price=10.0,
        fee=3.0,
    )

    ledger.record_targets(batch_id, targets)
    ledger.record_netting_result(result)
    ledger.record_allocations(external)
    ledger.apply_allocations_to_positions(
        [*result.internal_allocations, *external],
        route_by_strategy={
            "long_a": {"asset_type": "equity", "venue": "ib", "account": "paper"},
            "short_b": {"asset_type": "equity", "venue": "ib", "account": "paper"},
        },
        timestamp_utc="2026-05-13T20:00:00+00:00",
    )

    assert ledger.current_positions() == {("long_a", "XYZ"): 10.0, ("short_b", "XYZ"): -4.0}

    marks = ledger.mark_to_market(
        prices={"XYZ": 12.0},
        timestamp_utc="2026-05-14T20:00:00+00:00",
        source="test_mark",
    )
    by_strategy = {row["strategy_id"]: row for row in marks}
    assert by_strategy["long_a"]["pnl"] == 20.0
    assert by_strategy["short_b"]["pnl"] == -8.0

    summaries = {row["strategy_id"]: row for row in ledger.strategy_summaries()}
    assert summaries["long_a"]["cumulative_pnl"] == 17.0
    assert summaries["short_b"]["cumulative_pnl"] == -8.0


def test_ledger_current_position_filter_and_flatten_delete(tmp_path) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.db")
    route_by_strategy = {
        "strategy_a": {"asset_type": "equity", "venue": "ib", "account": "paper"},
        "strategy_b": {"asset_type": "equity", "venue": "ib", "account": "paper"},
    }
    ledger.apply_allocations_to_positions(
        [
            FillAllocation("alloc_a1", "batch_test", "strategy_a", DEFAULT_BUCKET, "AAA", 10, 20, "external_fill", "net_a"),
            FillAllocation("alloc_b1", "batch_test", "strategy_b", DEFAULT_BUCKET, "BBB", -5, 30, "external_fill", "net_b"),
        ],
        route_by_strategy=route_by_strategy,
        timestamp_utc="2026-05-13T20:00:00+00:00",
    )

    assert ledger.current_positions({"strategy_a"}) == {("strategy_a", "AAA"): 10.0}
    assert ledger.current_positions({"strategy_b"}) == {("strategy_b", "BBB"): -5.0}

    ledger.apply_allocations_to_positions(
        [FillAllocation("alloc_a2", "batch_test", "strategy_a", DEFAULT_BUCKET, "AAA", -10, 21, "external_fill", "net_a2")],
        route_by_strategy=route_by_strategy,
        timestamp_utc="2026-05-14T20:00:00+00:00",
    )

    assert ledger.current_positions() == {("strategy_b", "BBB"): -5.0}


def test_ledger_reports_netting_stats_and_curve_downsample_keeps_latest(tmp_path) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.db")
    batch_id = ledger.start_batch(mode="paper-sim", batch_id="batch_test")
    targets = [_target("long_a", "XYZ", 10), _target("short_b", "XYZ", -4)]
    children = build_child_orders(batch_id=batch_id, targets=targets, current_positions={})
    result = net_child_orders(batch_id=batch_id, child_orders=children)
    external = allocate_external_fill(
        batch_id=batch_id,
        net_order=result.net_orders[0],
        child_orders=children,
        filled_qty=result.net_orders[0].quantity,
        fill_price=10.0,
    )

    ledger.record_netting_result(result)
    ledger.record_allocations(external)
    ledger.finish_batch(batch_id, status="completed", metadata={})
    ledger.apply_allocations_to_positions(
        [*result.internal_allocations, *external],
        route_by_strategy={
            "long_a": {"asset_type": "equity", "venue": "ib", "account": "paper"},
            "short_b": {"asset_type": "equity", "venue": "ib", "account": "paper"},
        },
        timestamp_utc="2026-05-13T20:00:00+00:00",
    )
    for day, price in [("2026-05-14", 11.0), ("2026-05-15", 12.0), ("2026-05-18", 13.0)]:
        ledger.mark_to_market(
            prices={"XYZ": price},
            timestamp_utc=f"{day}T20:00:00+00:00",
            source="test_mark",
        )

    stats = ledger.latest_netting_stats()
    assert stats["status"] == "completed"
    assert stats["n_child_orders"] == 2
    assert stats["n_internal_crosses"] == 1
    assert stats["n_net_orders"] == 1
    assert stats["child_notional"] == 140.0
    assert stats["crossed_notional"] == 40.0
    assert stats["net_notional"] == 60.0
    assert round(stats["compression_ratio"], 6) == round(1.0 - 60.0 / 140.0, 6)

    curves = ledger.strategy_curves(max_points=1)
    assert curves["long_a"] == [{"timestamp": "2026-05-18T20:00:00+00:00", "pnl": 30.0}]
    assert curves["short_b"] == [{"timestamp": "2026-05-18T20:00:00+00:00", "pnl": -12.0}]


def test_ledger_uses_bucket_route_fallback_for_unknown_strategy_route(tmp_path) -> None:
    ledger = ExecutionLedger(tmp_path / "ledger.db")
    ledger.apply_allocations_to_positions(
        [FillAllocation("alloc_1", "batch_test", "mystery", "crypto:kucoin:paper:spot:MOC:close", "BTC-USDT", 2, 100, "external_fill", "net_1")],
        route_by_strategy={},
        timestamp_utc="2026-05-13T20:00:00+00:00",
    )

    summary = ledger.strategy_summaries()[0]
    assert summary["strategy_id"] == "mystery"
    assert summary["asset_type"] == "crypto"
    assert summary["venue"] == "kucoin"
    assert summary["account"] == "paper"
    assert summary["gross_exposure"] == 200.0


def test_highest_sharpe_aipt_strategy_config_is_registered() -> None:
    configs = {
        cfg.strategy_id: cfg
        for cfg in load_strategy_configs("prod/config/strategies")
    }
    cfg = configs["aipt_smallcap_d0_prox_l1_tau5"]
    research = cfg.metadata["research"]

    assert cfg.enabled is True
    assert cfg.venue == "ib"
    assert cfg.account == "paper"
    assert cfg.order_type == "MOC"
    assert cfg.broker_bucket == "equity:ib:paper:stock:MOC:close"
    assert cfg.signal["adapter"] == "aipt_weights_tail_artifact"
    assert "tau5" in cfg.signal["weights_path"]
    assert research["scenario"] == "equity_smallcap_d0"
    assert research["layer"] == "prox_l1_gross1_cap_fee"
    assert research["tau"] == 5.0
    assert research["reported_val_test_net_sharpe"] == pytest.approx(4.630583461607214)


def test_aipt_artifact_adapter_builds_targets_from_latest_weights_and_close(tmp_path) -> None:
    weights_path = tmp_path / "experiments/results/aipt/weights_tail.parquet"
    close_path = tmp_path / "data/fmp_cache/matrices/close.parquet"
    weights_path.parent.mkdir(parents=True)
    close_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {"AAA": [0.10], "BBB": [-0.20]},
        index=pd.to_datetime(["2026-05-12"]),
    ).to_parquet(weights_path)
    pd.DataFrame(
        {"AAA": [100.0], "BBB": [50.0]},
        index=pd.to_datetime(["2026-05-12"]),
    ).to_parquet(close_path)
    config = parse_strategy_config({
        "strategy_id": "aipt_smallcap_d0_prox_l1_tau5",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": "equity:ib:paper:stock:MOC:close",
        },
        "portfolio": {"book": 100000},
        "execution": {"order_type": "MOC"},
        "signal": {
            "adapter": "aipt_weights_tail_artifact",
            "weights_path": "experiments/results/aipt/weights_tail.parquet",
            "price_matrix": "data/fmp_cache/matrices/close.parquet",
        },
        "metadata": {"research": {"reported_val_test_net_sharpe": 4.77}},
    })

    targets = build_targets(config, root=tmp_path)

    assert [(t.symbol, t.target_qty, t.target_notional) for t in targets] == [
        ("AAA", 100.0, 10000.0),
        ("BBB", -400.0, -20000.0),
    ]
    assert all(t.order_type == "MOC" for t in targets)
    assert load_latest_price_map(config, root=tmp_path) == {"AAA": 100.0, "BBB": 50.0}


def test_orchestrator_shadow_nets_overlapping_strategy_orders(tmp_path, monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch

    config_dir = tmp_path / "strategies"
    ledger_db = tmp_path / "ledger.db"
    _write_close(tmp_path, {"AAA": [100.0], "BBB": [50.0]})
    _write_artifact_strategy(tmp_path, config_dir, "strategy_a", {"AAA": [0.10], "BBB": [0.05]}, name="Strategy A")
    _write_artifact_strategy(tmp_path, config_dir, "strategy_b", {"AAA": [-0.04]}, name="Strategy B")
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    summary = orch.run(mode="shadow", config_dir=config_dir, ledger_db=ledger_db)

    assert summary["strategies"] == ["strategy_a", "strategy_b"]
    assert summary["n_targets"] == 3
    assert summary["n_child_orders"] == 3
    assert summary["n_internal_crosses"] == 1
    assert summary["n_net_orders"] == 2
    assert summary["gross_child_notional"] == 19000.0
    assert summary["crossed_notional"] == 4000.0
    assert summary["external_notional"] == 11000.0
    assert summary["compression_ratio"] == round(1.0 - 11000.0 / 19000.0, 4)

    ledger = ExecutionLedger(ledger_db)
    assert ledger.current_positions() == {}
    stats = ledger.latest_netting_stats()
    assert stats["status"] == "completed"
    assert stats["n_internal_crosses"] == 1


def test_orchestrator_paper_sim_marks_existing_positions_before_rebalance(tmp_path, monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch

    config_dir = tmp_path / "strategies"
    ledger_db = tmp_path / "ledger.db"
    _write_close(tmp_path, {"AAA": [100.0], "BBB": [50.0]})
    _write_artifact_strategy(tmp_path, config_dir, "strategy_a", {"AAA": [0.10], "BBB": [0.05]}, name="Strategy A")
    _write_artifact_strategy(tmp_path, config_dir, "strategy_b", {"AAA": [-0.04]}, name="Strategy B")
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    first = orch.run(mode="paper-sim", config_dir=config_dir, ledger_db=ledger_db)
    assert first["n_internal_crosses"] == 1

    ledger = ExecutionLedger(ledger_db)
    assert ledger.current_positions() == {
        ("strategy_a", "AAA"): 100.0,
        ("strategy_a", "BBB"): 100.0,
        ("strategy_b", "AAA"): -40.0,
    }

    _write_close(tmp_path, {"AAA": [110.0], "BBB": [55.0]}, date="2026-05-13")
    second = orch.run(mode="paper-sim", config_dir=config_dir, ledger_db=ledger_db)

    assert second["n_child_orders"] == 3
    summaries = {row["strategy_id"]: row for row in ledger.strategy_summaries()}
    assert summaries["strategy_a"]["cumulative_pnl"] == 1500.0
    assert summaries["strategy_b"]["cumulative_pnl"] == -400.0
    assert ledger.current_positions() == {
        ("strategy_a", "AAA"): 91.0,
        ("strategy_a", "BBB"): 91.0,
        ("strategy_b", "AAA"): -36.0,
    }


def test_orchestrator_respects_enabled_flag_and_strategy_id_filter(tmp_path, monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch

    config_dir = tmp_path / "strategies"
    _write_close(tmp_path, {"AAA": [100.0], "BBB": [50.0], "CCC": [25.0]})
    _write_artifact_strategy(tmp_path, config_dir, "strategy_a", {"AAA": [0.10]})
    _write_artifact_strategy(tmp_path, config_dir, "strategy_b", {"BBB": [0.20]})
    _write_artifact_strategy(tmp_path, config_dir, "disabled_strategy", {"CCC": [0.30]}, enabled=False)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    summary = orch.run(
        mode="shadow",
        config_dir=config_dir,
        ledger_db=tmp_path / "ledger.db",
        strategy_ids={"strategy_b", "disabled_strategy"},
    )

    assert summary["strategies"] == ["strategy_b"]
    assert summary["n_targets"] == 1
    assert summary["n_child_orders"] == 1


def test_ib_paper_submission_requires_explicit_env(monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch

    monkeypatch.delenv("ALLOW_MULTI_STRATEGY_IB_PAPER_ORDERS", raising=False)
    order = NetOrder(
        net_order_id="net_1",
        batch_id="batch_test",
        asset_type="equity",
        venue="ib",
        account="paper",
        bucket=DEFAULT_BUCKET,
        symbol="AAA",
        quantity=10,
        reference_price=100,
    )

    with pytest.raises(RuntimeError, match="IB paper submission blocked"):
        orch._submit_ib_paper_guarded([order])


def test_dashboard_payload_reads_strategy_groups_from_ledger(tmp_path, monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch
    from prod.stats import ops_dashboard as dash

    config_dir = tmp_path / "strategies"
    ledger_db = tmp_path / "ledger.db"
    _write_close(tmp_path, {"AAA": [100.0]})
    _write_artifact_strategy(tmp_path, config_dir, "strategy_a", {"AAA": [0.10]}, name="Strategy A")
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    orch.run(mode="paper-sim", config_dir=config_dir, ledger_db=ledger_db)
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dash, "MULTI_STRATEGY_CONFIG_DIR", config_dir)
    monkeypatch.setattr(dash, "EXECUTION_LEDGER_DB", ledger_db)

    payload = dash._multi_strategy_payload()
    section = dash._multi_strategy_section(payload)

    assert payload["status"] == "ok"
    assert payload["netting"]["status"] == "completed"
    assert payload["groups"]["equity"][0]["strategy_id"] == "strategy_a"
    assert payload["groups"]["equity"][0]["gross_exposure"] == 10000.0
    assert "Strategies By Asset" in section
    assert "Strategy A" in section
    assert "Latest batch" in section


def test_dashboard_multi_strategy_payload_does_not_create_missing_ledger(tmp_path, monkeypatch) -> None:
    from prod.stats import ops_dashboard as dash

    config_dir = tmp_path / "strategies"
    config_dir.mkdir()
    (config_dir / "aipt.json").write_text(json.dumps({
        "strategy_id": "aipt_smallcap_d0_prox_l1_tau5",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": "equity:ib:paper:stock:MOC:close",
        },
        "portfolio": {"book": 500000},
        "execution": {"order_type": "MOC"},
        "signal": {"adapter": "aipt_weights_tail_artifact"},
        "metadata": {
            "family": "aipt",
            "research": {"reported_val_test_net_sharpe": 4.77},
        },
    }), encoding="utf-8")
    ledger_path = tmp_path / "missing" / "execution_ledger.db"

    monkeypatch.setattr(dash, "MULTI_STRATEGY_CONFIG_DIR", config_dir)
    monkeypatch.setattr(dash, "EXECUTION_LEDGER_DB", ledger_path)

    payload = dash._multi_strategy_payload()

    assert not ledger_path.exists()
    assert payload["status"] == "ok"
    assert payload["groups"]["equity"][0]["strategy_id"] == "aipt_smallcap_d0_prox_l1_tau5"


def test_stale_aipt_artifact_is_not_routeable(tmp_path) -> None:
    config_dir = tmp_path / "strategies"
    _write_close(tmp_path, {"AAA": [100.0]}, date="2026-05-14")
    _write_artifact_strategy(
        tmp_path,
        config_dir,
        "aipt_stale",
        {"AAA": [0.10]},
        signal_date="2026-04-24",
        max_signal_lag_days=1,
        stale_action="skip_strategy",
    )
    cfg = load_strategy_configs(config_dir)[0]

    with pytest.raises(StaleStrategySignal, match="signal is stale"):
        build_targets(cfg, root=tmp_path)


def test_json_aipt_weight_artifact_builds_targets(tmp_path) -> None:
    config_dir = tmp_path / "strategies"
    _write_close(tmp_path, {"AAA": [100.0]}, date="2026-05-14")
    artifact = tmp_path / "prod/config/strategies/aipt_weights.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({
        "signal_time": "2026-05-14",
        "weights": {"AAA": 0.25},
    }), encoding="utf-8")
    config_dir.mkdir()
    (config_dir / "aipt.json").write_text(json.dumps({
        "strategy_id": "aipt_json",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": DEFAULT_BUCKET,
        },
        "portfolio": {"book": 100000},
        "execution": {"order_type": "MOC"},
        "signal": {
            "adapter": "aipt_weights_tail_artifact",
            "weights_path": "prod/config/strategies/aipt_weights.json",
            "price_matrix": "data/fmp_cache/matrices/close.parquet",
            "max_signal_lag_days": 1,
        },
    }), encoding="utf-8")
    cfg = load_strategy_configs(config_dir)[0]

    targets = build_targets(cfg, root=tmp_path)

    assert len(targets) == 1
    assert targets[0].symbol == "AAA"
    assert targets[0].target_qty == 250.0


def test_orchestrator_skips_stale_strategy_without_net_orders(tmp_path, monkeypatch) -> None:
    from prod import multi_strategy_orchestrator as orch

    config_dir = tmp_path / "strategies"
    ledger_db = tmp_path / "ledger.db"
    _write_close(tmp_path, {"AAA": [100.0]}, date="2026-05-14")
    _write_artifact_strategy(
        tmp_path,
        config_dir,
        "aipt_stale",
        {"AAA": [0.10]},
        signal_date="2026-04-24",
        max_signal_lag_days=1,
        stale_action="skip_strategy",
    )
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)

    summary = orch.run(mode="paper-sim", config_dir=config_dir, ledger_db=ledger_db)

    assert summary["strategies"] == []
    assert summary["skipped_strategies"][0]["strategy_id"] == "aipt_stale"
    assert summary["n_child_orders"] == 0
    assert summary["n_net_orders"] == 0


def test_dashboard_marks_stale_aipt_shadow_excluded_from_current_totals(tmp_path, monkeypatch) -> None:
    from prod.stats import ops_dashboard as dash

    config_dir = tmp_path / "strategies"
    _write_close(tmp_path, {"AAA": [100.0]}, date="2026-05-14")
    _write_artifact_strategy(
        tmp_path,
        config_dir,
        "aipt_smallcap_d0_prox_l1_tau5",
        {"AAA": [0.10]},
        signal_date="2026-04-24",
        max_signal_lag_days=1,
        stale_action="skip_strategy",
    )
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dash, "MULTI_STRATEGY_CONFIG_DIR", config_dir)
    monkeypatch.setattr(dash, "EXECUTION_LEDGER_DB", tmp_path / "missing_ledger.db")
    monkeypatch.setattr(dash, "TRADE_LOG_DIRS", {"ib": tmp_path / "logs/trades"})
    monkeypatch.setattr(dash, "RECON_DIR", tmp_path / "logs/reconciliation")

    payload = dash._multi_strategy_payload(performance=[])
    row = payload["groups"]["equity"][0]

    assert row["truth_status"] == "warn"
    assert row["scope"] == "stale_shadow"
    assert row["gross_exposure"] is None
    assert row["cumulative_pnl"] is None
    assert payload["totals"]["n_current"] == 0
    assert payload["totals"]["gross_exposure"] is None


def test_dashboard_ib_strategy_row_uses_post_recon_actuals(tmp_path, monkeypatch) -> None:
    from prod.stats import ops_dashboard as dash

    config_dir = tmp_path / "strategies"
    config_dir.mkdir()
    (tmp_path / "data/fmp_cache/matrices").mkdir(parents=True, exist_ok=True)
    close = pd.DataFrame(
        {"AAA": [10.0, 11.0], "BBB": [20.0, 19.0]},
        index=pd.to_datetime(["2026-05-13", "2026-05-14"]),
    )
    close.to_parquet(tmp_path / "data/fmp_cache/matrices/close.parquet")
    (config_dir / "ib.json").write_text(json.dumps({
        "strategy_id": "ib_moc_equity",
        "name": "IB Closing Auction L/S Equity",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": DEFAULT_BUCKET,
        },
        "portfolio": {"book": 500000},
        "execution": {"order_type": "MOC"},
        "signal": {
            "adapter": "pipeline_latest_signal",
            "config_path": "prod/config/research_equity.json",
            "price_matrix": "data/fmp_cache/matrices/close.parquet",
        },
        "metadata": {"family": "prod_equity"},
    }), encoding="utf-8")
    trade_dir = tmp_path / "logs/trades"
    recon_dir = tmp_path / "logs/reconciliation"
    trade_dir.mkdir(parents=True)
    recon_dir.mkdir(parents=True)
    (trade_dir / "trade_2026-05-14.json").write_text(json.dumps({
        "date": "2026-05-14",
        "mode": "live",
        "signal_date": "2026-05-14",
        "timestamp": "2026-05-14T19:41:00+00:00",
        "current_positions": {"AAA": 10, "BBB": -5},
        "target_portfolio": {"AAA": 12, "BBB": -6},
    }), encoding="utf-8")
    (recon_dir / "equity_2026-05-14.json").write_text(json.dumps({
        "date": "2026-05-14",
        "timestamp": "2026-05-14T20:30:00+00:00",
        "status": "reconciled",
        "total_commission": 1.5,
        "fill_rate_qty": 1.0,
        "n_orders_intent": 2,
        "n_orders_with_fills": 2,
        "fills": [
            {"symbol": "AAA", "filled_qty": 2, "avg_price": 11.0, "commission": 0.5},
            {"symbol": "BBB", "filled_qty": -1, "avg_price": 19.0, "commission": 1.0},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dash, "MULTI_STRATEGY_CONFIG_DIR", config_dir)
    monkeypatch.setattr(dash, "EXECUTION_LEDGER_DB", tmp_path / "missing_ledger.db")
    monkeypatch.setattr(dash, "TRADE_LOG_DIRS", {"ib": trade_dir})
    monkeypatch.setattr(dash, "RECON_DIR", recon_dir)

    payload = dash._multi_strategy_payload(performance=[{
        "exchange": "ib",
        "live_recon_curve": [{"timestamp": "2026-05-14", "pnl": 100.0}],
        "live_recon_total_pnl": 100.0,
    }])
    row = payload["groups"]["equity"][0]

    assert row["truth_status"] == "ok"
    assert row["scope"] == "actual_post_recon"
    assert row["cumulative_pnl"] == 98.5
    assert row["gross_exposure"] == 246.0
    assert row["net_exposure"] == 18.0
    assert row["n_positions"] == 2
    assert payload["totals"]["n_current"] == 1
    assert payload["totals"]["gross_exposure"] == 246.0
