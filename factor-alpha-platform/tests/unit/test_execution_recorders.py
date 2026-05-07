from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.execution.recorders import (
    run_crypto_paper_recorder,
    run_ib_paper_moc,
    summarize_trade_log,
    validate_crypto_paper_config,
    validate_ib_paper_config,
)


class FakePopen:
    """Drop-in replacement for subprocess.Popen used by the new _run_subprocess.

    Writes whatever the side-effect callable produces to the stdout file handle,
    then exits with the requested returncode. Supports timeout simulation.
    """

    def __init__(self, command, cwd, stdout, stderr, env, on_run=None,
                 returncode=0, raise_timeout=False, hangs=False, **_kwargs):
        self.args = command
        self._cwd = cwd
        self._stdout_fh = stdout
        self._stderr_fh = stderr
        self._env = env
        self._on_run = on_run
        self._returncode = returncode
        self._raise_timeout = raise_timeout
        self._hangs = hangs
        self._terminated = False
        self.pid = 99999

    def wait(self, timeout=None):
        if self._raise_timeout:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout)
        if self._on_run is not None:
            self._on_run(self.args, self._cwd, self._stdout_fh, self._stderr_fh, self._env)
        self._stdout_fh.flush()
        self._stderr_fh.flush()
        return self._returncode

    def poll(self):
        return None if not self._terminated else -1

    def send_signal(self, sig):  # noqa: D401
        self._terminated = True

    def kill(self):
        self._terminated = True


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

    def on_run(command, cwd, stdout_fh, stderr_fh, env):
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
        stdout_fh.write(b"paper recorder complete\n")

    def popen_factory(*args, **kwargs):
        return FakePopen(*args, on_run=on_run, returncode=0, **kwargs)

    with patch("src.execution.recorders.subprocess.Popen", side_effect=popen_factory):
        result = run_crypto_paper_recorder(exchange, root=tmp_path)

    assert result.status == "completed"
    assert result.trade_log_path
    assert result.metadata["produced_new_log"] is True
    assert result.metadata["trade_summary"]["n_orders"] == 2


def test_run_crypto_paper_recorder_kills_child_on_timeout(tmp_path):
    """A subprocess timeout must terminate the process tree, not just leak."""
    exchange = "kucoin"
    _write_json(tmp_path / f"prod/config/{exchange}.json", _crypto_config(exchange))
    (tmp_path / f"prod/logs/{exchange}/trades").mkdir(parents=True)

    fake_proc = {"instance": None}

    def popen_factory(*args, **kwargs):
        proc = FakePopen(*args, raise_timeout=True, **kwargs)
        fake_proc["instance"] = proc
        return proc

    with patch("src.execution.recorders.subprocess.Popen", side_effect=popen_factory):
        result = run_crypto_paper_recorder(exchange, root=tmp_path, timeout_sec=1)

    assert result.status == "failed"
    assert "timed out" in result.message
    assert fake_proc["instance"]._terminated is True


def test_run_ib_paper_moc_invokes_cancel_all_on_timeout(tmp_path):
    """IB execution timeout must best-effort fire cancel_all to drop in-flight orders."""
    _write_json(tmp_path / "prod/config/strategy.json", _strategy_config())
    cancel_calls = []

    def popen_factory(*args, **kwargs):
        return FakePopen(*args, raise_timeout=True, **kwargs)

    def fake_subprocess_run(cmd, **kwargs):
        cancel_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="cancelled", stderr="")

    with patch("src.execution.recorders.subprocess.Popen", side_effect=popen_factory), \
         patch("src.execution.recorders.subprocess.run", side_effect=fake_subprocess_run):
        result = run_ib_paper_moc(root=tmp_path, allow_orders=True, timeout_sec=1)

    assert result.status == "failed"
    assert any("cancel_all.py" in str(c) for c in cancel_calls), "cancel_all.py must run on IB timeout"


def test_write_execution_summary_is_atomic(tmp_path):
    """Writes go through .tmp + os.replace so a crash mid-write can't corrupt the summary."""
    from src.execution.recorders import ExecutionResult, write_execution_summary

    result = ExecutionResult(
        strategy_id="test", venue="ib_equity", mode="paper",
        status="completed", started_at_utc="x", ended_at_utc="y",
        elapsed_sec=0.0, run_id="abc",
    )
    path = write_execution_summary(result, root=tmp_path)
    # No leftover .tmp file
    assert not list(path.parent.glob("*.tmp"))
    assert path.exists()
    assert json.loads(path.read_text())["run_id"] == "abc"


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
