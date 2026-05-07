"""IB Gateway connectivity checks for local, Docker, and cloud deployments."""
from __future__ import annotations

import datetime as dt
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IBGatewayStatus:
    host: str
    port: int
    connected: bool
    mode: str
    checked_at_utc: str
    elapsed_sec: float
    timeout_sec: float
    client_id: int | None = None
    accounts: list[str] = field(default_factory=list)
    message: str = ""
    error_type: str | None = None
    paper_only: bool = False
    has_live_account: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_ib_gateway_endpoint(strategy_config: dict[str, Any]) -> tuple[str, int]:
    ibkr = strategy_config.get("ibkr", {})
    host = os.environ.get("IB_HOST", ibkr.get("host", "127.0.0.1"))
    port = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", ibkr.get("port_paper", 4002))))
    return host, port


def _mask_account(account: str) -> str:
    if len(account) <= 6:
        return "***"
    return f"{account[:2]}***{account[-4:]}"


def check_ib_gateway(
    strategy_config: dict[str, Any],
    *,
    mode: str | None = None,
    timeout_sec: float | None = None,
    client_id: int | None = None,
) -> IBGatewayStatus:
    """Probe the configured IB Gateway.

    `ib_insync` mode performs a real API handshake and account list request.
    `tcp` mode only verifies that the TCP port accepts a connection.
    """
    host, port = resolve_ib_gateway_endpoint(strategy_config)
    probe_mode = (mode or os.environ.get("IB_GATEWAY_HEALTH_MODE", "ib_insync")).lower()
    timeout = float(timeout_sec or os.environ.get("IB_GATEWAY_HEALTH_TIMEOUT_SEC", "8"))
    health_client_id = int(client_id or os.environ.get("IB_CLIENT_ID_HEALTHCHECK", "19"))
    started = time.monotonic()

    try:
        if probe_mode == "tcp":
            with socket.create_connection((host, port), timeout=timeout):
                pass
            accounts: list[str] = []
        else:
            from ib_insync import IB

            ib = IB()
            try:
                ib.connect(host, port, clientId=health_client_id, timeout=timeout)
                accounts = list(ib.managedAccounts())
            finally:
                if ib.isConnected():
                    ib.disconnect()
        elapsed = round(time.monotonic() - started, 3)

        # Paper safety: tcp probes can't see accounts at all (false positive risk),
        # and ib_insync probes must show only DU* paper accounts unless explicitly
        # opted into live trading via env var.
        allow_live = os.environ.get("IB_ALLOW_LIVE_TRADING") == "1"
        masked_accounts = [_mask_account(a) for a in accounts]
        has_live_account = bool(accounts) and not all(a.startswith("DU") for a in accounts)
        paper_only = bool(accounts) and all(a.startswith("DU") for a in accounts)

        if probe_mode == "tcp":
            detail = (
                f"IB Gateway TCP reachable at {host}:{port} (no account verification — "
                f"do not use 'tcp' mode in production; set IB_GATEWAY_HEALTH_MODE=ib_insync)"
            )
            connected = True
        elif has_live_account and not allow_live:
            detail = (
                f"IB Gateway at {host}:{port} exposed a non-paper account "
                f"(accounts={masked_accounts}); refusing to mark as connected. "
                f"Set IB_ALLOW_LIVE_TRADING=1 if this is intentional."
            )
            connected = False
        elif not accounts:
            detail = (
                f"IB Gateway handshake at {host}:{port} returned no managed accounts; "
                f"gateway login may be incomplete."
            )
            connected = False
        else:
            detail = f"IB Gateway reachable at {host}:{port} using {probe_mode}; "
            detail += f"accounts={', '.join(masked_accounts)} (paper_only={paper_only})"
            connected = True

        return IBGatewayStatus(
            host=host,
            port=port,
            connected=connected,
            mode=probe_mode,
            checked_at_utc=utc_now_iso(),
            elapsed_sec=elapsed,
            timeout_sec=timeout,
            client_id=health_client_id if probe_mode != "tcp" else None,
            accounts=accounts,
            message=detail,
            paper_only=paper_only,
            has_live_account=has_live_account,
        )
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 3)
        return IBGatewayStatus(
            host=host,
            port=port,
            connected=False,
            mode=probe_mode,
            checked_at_utc=utc_now_iso(),
            elapsed_sec=elapsed,
            timeout_sec=timeout,
            client_id=health_client_id if probe_mode != "tcp" else None,
            message=f"IB Gateway unreachable at {host}:{port}: {exc}",
            error_type=type(exc).__name__,
        )


def ib_gateway_status_to_check_payload(status: IBGatewayStatus | dict[str, Any]) -> dict[str, Any]:
    data = status.to_dict() if isinstance(status, IBGatewayStatus) else dict(status)
    return {
        "name": "ib_gateway_connectivity",
        "status": "pass" if data.get("connected") else "fail",
        "severity": "critical",
        "message": data.get("message") or "IB Gateway status unavailable",
        "value": data.get("elapsed_sec"),
        "threshold": f"{data.get('timeout_sec')}s timeout",
        "metadata": {
            "host": data.get("host"),
            "port": data.get("port"),
            "mode": data.get("mode"),
            "client_id": data.get("client_id"),
            "accounts_count": len(data.get("accounts") or []),
            "paper_only": data.get("paper_only", False),
            "has_live_account": data.get("has_live_account", False),
            "error_type": data.get("error_type"),
            "checked_at_utc": data.get("checked_at_utc"),
        },
    }
