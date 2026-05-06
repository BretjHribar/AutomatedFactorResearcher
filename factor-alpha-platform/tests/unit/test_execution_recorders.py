from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.execution.recorders import (
    run_crypto_paper_recorder,
    run_ib_paper_moc,
    summarize_trade_log,
    validate_crypto_paper_config,
    validate_ib_paper_config,
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _strategy_config(port_paper: int = 4002, port_live: int = 4001) -> dict:
    return {
        "strategy": {"name": "IB Closing Auction L/S Equity"},
        "ibkr": {
            "host": "127.0.0.1",
            "port_paper": port_paper,
            "port_live": port_live,
            "paper_account": "DUQ372830",
            "client_id": 10,
            "client_id_order_entry": 10,
        },
        "paths": {"trade_logs": "prod/logs/trades"},
    }


def _crypto_config(exchange: str) -> dict:
    return {
        "exchange": exchange,
        "strategy": {"name": f"{exchange.title()} Futures L/S Crypto", "universe": "TOP100"},
        "account": {"target_gmv": 100000},
        "execution": {"paper_mode": True},
        "paths": {
            "trade_logs": f"prod/logs/{exchange}/trades",
            "performance_logs": f"prod/logs/{exchange}/performance",
            "matrices": f"data/{exchange}_cache/matrices/4h/prod",
        },
    }


def test_validate_ib_paper_config_accepts_known_paper_gateway():
    guard = validate_ib_paper_config(_strategy_config())

    assert guard["paper_port"] == 4002
    assert guard["paper_account"].startswith("DU")


def test_validate_ib_paper_config_prefers_env_host_and_port(monkeypatch):
    monkeypatch.setenv("IB_HOST", "host.docker.internal")
    monkeypatch.setenv("IB_PORT", "7497")

    guard = validate_ib_paper_config(_strategy_config())

    assert guard["host"] == "host.docker.internal"
    assert guard["paper_port"] == 7497


def test_validate_ib_paper_config_accepts_docker_gateway_paper_port(monkeypatch):
    monkeypatch.setenv("IB_HOST", "ib-gateway")
    monkeypatch.setenv("IB_PORT", "4004")

    guard = validate_ib_paper_config(_strategy_config())

    assert guard["host"] == "ib-gateway"
    assert guard["paper_port"] == 4004


def test_validate_ib_paper_config_rejects_live_port_collision():
    with pytest.raises(ValueError, match="paper port equals live port"):
        validate_ib_paper_config(_strategy_config(port_paper=4001, port_live=4001))


def test_run_ib_paper_moc_blocked_writes_summary_without_subprocess(tmp_path):
    _write_json(tmp_path / "prod/config/strategy.json", _strategy_config())

    result = run_ib_paper_moc(root=tmp_path, allow_orders=False)

    assert result.status == "blocked"
    assert result.command == []
    assert result.summary_path
    assert Path(result.summary_path).exists()
    assert "ALLOW_IB_PAPER_ORDERS" in result.message


def test_validate_crypto_paper_config_rejects_non_paper_mode():
    cfg = _crypto_config("binance")
    cfg["execution"]["paper_mode"] = False

    with pytest.raises(ValueError, match="paper_mode must be true"):
        validate_crypto_paper_config(cfg, "binance")


def test_run_crypto_paper_recorder_detects_new_trade_log(tmp_path):
    exchange = "binance"
    _write_json(tmp_path / f"prod/config/{exchange}.json", _crypto_config(exchange))
    trade_dir = tmp_path / "prod/logs/binance/trades"
    trade_dir.mkdir(parents=True)

    def fake_runner(command, cwd, capture_output, text, timeout, env):
        trade_log = Path(cwd) / "prod/logs/binance/trades/trade_2026-05-05T20-00-00.json"
        _write_json(
            trade_log,
            {
                "exchange": "binance",
                "timestamp": "2026-05-05T20:00:00",
                "bar_time": "2026-05-05 20:00:00",
                "mode": "PAPER",
                "orders": {"n_orders": 2, "diffs": {"BTCUSDT": 1000, "ETHUSDT": -1000}},
                "fills": {"BTCUSDT": {}, "ETHUSDT": {}},
                "portfolio": {"gmv": 2000},
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="paper recorder complete", stderr="")

    result = run_crypto_paper_recorder(exchange, root=tmp_path, runner=fake_runner)

    assert result.status == "completed"
    assert result.trade_log_path
    assert result.metadata["produced_new_log"] is True
    assert result.metadata["trade_summary"]["n_orders"] == 2


def test_summarize_trade_log_handles_ib_order_records(tmp_path):
    path = tmp_path / "trade.json"
    _write_json(
        path,
        {
            "mode": "live",
            "signal_date": "2026-05-05",
            "order_records": [
                {"symbol": "AAA", "status": "Filled"},
                {"symbol": "BBB", "status": "FAILED: Inactive"},
            ],
            "portfolio_summary": {"n_long": 1, "n_short": 1},
        },
    )

    summary = summarize_trade_log(path)

    assert summary["n_orders"] == 2
    assert summary["order_status_counts"] == {"Filled": 1, "FAILED: Inactive": 1}
