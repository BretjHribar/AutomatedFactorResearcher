from __future__ import annotations

from src.execution.ib_gateway_health import (
    IBGatewayStatus,
    ib_gateway_status_to_check_payload,
    resolve_ib_gateway_endpoint,
)


def _strategy_config() -> dict:
    return {"ibkr": {"host": "127.0.0.1", "port_paper": 4002}}


def test_resolve_ib_gateway_endpoint_prefers_env(monkeypatch):
    monkeypatch.setenv("IB_HOST", "ib-gateway")
    monkeypatch.setenv("IB_PORT", "4004")

    assert resolve_ib_gateway_endpoint(_strategy_config()) == ("ib-gateway", 4004)


def test_ib_gateway_status_to_check_payload_pass():
    status = IBGatewayStatus(
        host="ib-gateway",
        port=4004,
        connected=True,
        mode="ib_insync",
        checked_at_utc="2026-05-05T20:00:00+00:00",
        elapsed_sec=0.25,
        timeout_sec=8.0,
        client_id=19,
        accounts=["DUQ372830"],
        message="ok",
    )

    payload = ib_gateway_status_to_check_payload(status)

    assert payload["name"] == "ib_gateway_connectivity"
    assert payload["status"] == "pass"
    assert payload["severity"] == "critical"
    assert payload["metadata"]["accounts_count"] == 1


def test_ib_gateway_status_to_check_payload_fail():
    payload = ib_gateway_status_to_check_payload({
        "host": "ib-gateway",
        "port": 4004,
        "connected": False,
        "mode": "ib_insync",
        "checked_at_utc": "2026-05-05T20:00:00+00:00",
        "elapsed_sec": 8.0,
        "timeout_sec": 8.0,
        "client_id": 19,
        "accounts": [],
        "message": "timeout",
        "error_type": "TimeoutError",
    })

    assert payload["status"] == "fail"
    assert payload["message"] == "timeout"
    assert payload["metadata"]["error_type"] == "TimeoutError"


def test_ib_gateway_status_to_check_payload_includes_paper_flags():
    status = IBGatewayStatus(
        host="ib-gateway", port=4004, connected=True, mode="ib_insync",
        checked_at_utc="2026-05-05T20:00:00+00:00", elapsed_sec=0.25, timeout_sec=8.0,
        client_id=19, accounts=["DUQ372830"], message="ok",
        paper_only=True, has_live_account=False,
    )
    payload = ib_gateway_status_to_check_payload(status)
    assert payload["metadata"]["paper_only"] is True
    assert payload["metadata"]["has_live_account"] is False


def test_ib_gateway_check_refuses_to_pass_when_live_account_present(monkeypatch):
    """A gateway that returns a live account must be reported as not connected,
    even though the TCP/API handshake itself succeeded."""
    from src.execution import ib_gateway_health

    class FakeIB:
        def __init__(self):
            self._connected = False
        def connect(self, host, port, clientId, timeout):
            self._connected = True
        def isConnected(self):
            return self._connected
        def disconnect(self):
            self._connected = False
        def managedAccounts(self):
            return ["U1234567"]  # live account

    fake_module = type("FakeModule", (), {"IB": FakeIB})
    monkeypatch.setitem(__import__("sys").modules, "ib_insync", fake_module)
    monkeypatch.delenv("IB_ALLOW_LIVE_TRADING", raising=False)
    monkeypatch.delenv("IB_GATEWAY_HEALTH_MODE", raising=False)

    status = ib_gateway_health.check_ib_gateway(_strategy_config())

    assert status.connected is False
    assert status.has_live_account is True
    assert "non-paper" in status.message
