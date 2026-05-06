"""Execution wrappers with durable audit records.

The production scripts remain the source of truth for order generation and
paper simulation. This module gives Dagster a safe wrapper around them: every
attempt gets stdout/stderr logs, a structured summary, config guardrails, and a
machine-readable status.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_LOG_DIR = Path("prod/logs/execution")


@dataclass(frozen=True)
class ExecutionResult:
    strategy_id: str
    venue: str
    mode: str
    status: str
    started_at_utc: str
    ended_at_utc: str
    elapsed_sec: float
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    run_id: str | None = None
    summary_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    trade_log_path: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_execution_summary(result: ExecutionResult, root: Path = PROJECT_ROOT) -> Path:
    log_dir = root / EXECUTION_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = result.run_id or f"{result.venue}_{result.mode}_{utc_stamp()}"
    path = log_dir / f"{run_id}.json"
    payload = result.to_dict() | {"summary_path": str(path)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _finalize_result(result: ExecutionResult, root: Path) -> ExecutionResult:
    summary_path = write_execution_summary(result, root)
    return ExecutionResult(**(result.to_dict() | {"summary_path": str(summary_path)}))


def latest_json_log(log_dir: Path, pattern: str = "trade_*.json") -> Path | None:
    if not log_dir.exists():
        return None
    candidates = [p for p in log_dir.glob(pattern) if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def file_signature(path: Path | None) -> tuple[str, float, int] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return str(path), float(stat.st_mtime), int(stat.st_size)


def summarize_trade_log(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {"exists": False, "path": str(p)}
    try:
        data = load_json(p)
    except Exception as exc:
        return {"exists": True, "path": str(p), "read_error": str(exc)}

    orders = data.get("order_records") or data.get("orders") or {}
    fills = data.get("fills") or {}
    status_counts: dict[str, int] = {}
    if isinstance(orders, list):
        for order in orders:
            status = str(order.get("status", "UNKNOWN"))
            status_counts[status] = status_counts.get(status, 0) + 1
        n_orders = len(orders)
    elif isinstance(orders, dict):
        diffs = orders.get("diffs", orders)
        n_orders = int(orders.get("n_orders", len(diffs) if isinstance(diffs, dict) else 0))
    else:
        n_orders = 0

    return {
        "exists": True,
        "path": str(p),
        "exchange": data.get("exchange", data.get("market", "ib_equity")),
        "mode": data.get("mode"),
        "timestamp": data.get("timestamp"),
        "bar_time": data.get("bar_time"),
        "signal_date": data.get("signal_date"),
        "n_orders": n_orders,
        "n_fills": len(fills) if isinstance(fills, dict) else 0,
        "order_status_counts": status_counts,
        "portfolio": data.get("portfolio") or data.get("portfolio_summary") or {},
        "costs": data.get("costs") or {},
    }


def _run_subprocess(
    *,
    command: list[str],
    root: Path,
    run_id: str,
    venue: str,
    mode: str,
    strategy_id: str,
    timeout_sec: int,
    env: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[ExecutionResult, str, str]:
    started = utc_now()
    log_dir = root / EXECUTION_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_id}.stdout.log"
    stderr_path = log_dir / f"{run_id}.stderr.log"
    proc_env = os.environ.copy()
    proc_env.update(env or {})
    proc_env.setdefault("PYTHONUTF8", "1")
    proc_env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        completed = runner(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=proc_env,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        ended = utc_now()
        status = "completed" if completed.returncode == 0 else "failed"
        message = "subprocess completed" if completed.returncode == 0 else f"subprocess failed with returncode {completed.returncode}"
        result = ExecutionResult(
            strategy_id=strategy_id,
            venue=venue,
            mode=mode,
            status=status,
            started_at_utc=started.isoformat(),
            ended_at_utc=ended.isoformat(),
            elapsed_sec=round((ended - started).total_seconds(), 3),
            command=command,
            returncode=int(completed.returncode),
            run_id=run_id,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            message=message,
        )
        return result, stdout, stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        ended = utc_now()
        result = ExecutionResult(
            strategy_id=strategy_id,
            venue=venue,
            mode=mode,
            status="failed",
            started_at_utc=started.isoformat(),
            ended_at_utc=ended.isoformat(),
            elapsed_sec=round((ended - started).total_seconds(), 3),
            command=command,
            returncode=None,
            run_id=run_id,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            message=f"subprocess timed out after {timeout_sec}s",
            metadata={"exception": "TimeoutExpired"},
        )
        return result, stdout, stderr
    except Exception as exc:
        ended = utc_now()
        stderr = traceback.format_exc()
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        result = ExecutionResult(
            strategy_id=strategy_id,
            venue=venue,
            mode=mode,
            status="failed",
            started_at_utc=started.isoformat(),
            ended_at_utc=ended.isoformat(),
            elapsed_sec=round((ended - started).total_seconds(), 3),
            command=command,
            returncode=None,
            run_id=run_id,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            message=str(exc),
            metadata={"exception": type(exc).__name__},
        )
        return result, "", stderr


def validate_ib_paper_config(strategy_config: dict[str, Any]) -> dict[str, Any]:
    ibkr = strategy_config["ibkr"]
    host = os.environ.get("IB_HOST", ibkr["host"])
    paper_port = int(os.environ.get("IB_PORT_PAPER", os.environ.get("IB_PORT", ibkr["port_paper"])))
    live_port = int(os.environ.get("IB_PORT_LIVE", ibkr.get("port_live", 0)))
    if paper_port == live_port:
        raise ValueError(f"IB paper port equals live port ({paper_port}); refusing execution")
    if paper_port not in {4002, 4004, 7497, 7499}:
        raise ValueError(
            f"IB paper port {paper_port} is not a known paper gateway/TWS port "
            "(4002 Gateway host, 4004 Gateway Docker, 7497 TWS host, 7499 TWS Docker)"
        )
    paper_account = str(ibkr.get("paper_account", ""))
    if paper_account and not paper_account.startswith("DU"):
        raise ValueError(f"IB paper_account must start with DU; got {paper_account!r}")
    return {
        "host": host,
        "paper_port": paper_port,
        "live_port": live_port,
        "paper_account": paper_account,
        "client_id_order_entry": int(ibkr.get("client_id_order_entry", ibkr.get("client_id", 10))),
    }


def run_ib_paper_moc(
    *,
    root: str | Path = PROJECT_ROOT,
    allow_orders: bool = False,
    force: bool = False,
    timeout_sec: int = 900,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ExecutionResult:
    """Run the IB MOC trader against the configured paper gateway only."""
    root = Path(root)
    cfg = load_json(root / "prod/config/strategy.json")
    guard = validate_ib_paper_config(cfg)
    run_id = f"ib_paper_moc_{utc_stamp()}"
    started = utc_now()
    if not allow_orders:
        ended = utc_now()
        result = ExecutionResult(
            strategy_id=cfg["strategy"]["name"],
            venue="ib_equity",
            mode="paper_live_gateway",
            status="blocked",
            started_at_utc=started.isoformat(),
            ended_at_utc=ended.isoformat(),
            elapsed_sec=round((ended - started).total_seconds(), 3),
            run_id=run_id,
            message="IB paper order submission blocked; set ALLOW_IB_PAPER_ORDERS=1 to enable",
            metadata={"guard": guard},
        )
        return _finalize_result(result, root)

    trade_dir = root / cfg["paths"]["trade_logs"]
    today_log = trade_dir / f"trade_{dt.date.today().isoformat()}.json"
    before_sig = file_signature(today_log)
    command = [
        sys.executable,
        "prod/moc_trader.py",
        "--live",
        "--port",
        str(guard["paper_port"]),
    ]
    if force:
        command.append("--force")

    result, stdout, stderr = _run_subprocess(
        command=command,
        root=root,
        run_id=run_id,
        venue="ib_equity",
        mode="paper_live_gateway",
        strategy_id=cfg["strategy"]["name"],
        timeout_sec=timeout_sec,
        runner=runner,
    )
    after_sig = file_signature(today_log)
    trade_log_path = str(today_log) if after_sig and after_sig != before_sig else None
    trade_summary = summarize_trade_log(trade_log_path)
    metadata = result.metadata | {
        "guard": guard,
        "trade_log_changed": bool(trade_log_path),
        "trade_summary": trade_summary,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    final_status = result.status
    if result.status == "completed" and not trade_log_path:
        final_status = "skipped"
        metadata["skip_reason"] = "no new or modified IB trade log detected"
    final = ExecutionResult(**(
        result.to_dict()
        | {"status": final_status, "trade_log_path": trade_log_path, "metadata": metadata}
    ))
    return _finalize_result(final, root)


def load_exchange_config(root: Path, exchange: str) -> dict[str, Any]:
    if exchange not in {"kucoin", "binance"}:
        raise ValueError(f"unsupported exchange {exchange!r}")
    return load_json(root / "prod/config" / f"{exchange}.json")


def validate_crypto_paper_config(cfg: dict[str, Any], exchange: str) -> dict[str, Any]:
    if cfg.get("exchange") != exchange:
        raise ValueError(f"expected exchange={exchange!r}; got {cfg.get('exchange')!r}")
    if not bool(cfg.get("execution", {}).get("paper_mode", False)):
        raise ValueError(f"{exchange} execution.paper_mode must be true for recorder runs")
    return {
        "exchange": exchange,
        "paper_mode": True,
        "target_gmv": cfg.get("account", {}).get("target_gmv"),
        "trade_logs": cfg.get("paths", {}).get("trade_logs"),
        "performance_logs": cfg.get("paths", {}).get("performance_logs"),
    }


def run_crypto_paper_recorder(
    exchange: str,
    *,
    root: str | Path = PROJECT_ROOT,
    timeout_sec: int = 900,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> ExecutionResult:
    """Run the KuCoin/Binance paper recorder and summarize the produced log."""
    root = Path(root)
    cfg = load_exchange_config(root, exchange)
    guard = validate_crypto_paper_config(cfg, exchange)
    run_id = f"{exchange}_paper_recorder_{utc_stamp()}"
    trade_dir = root / cfg["paths"]["trade_logs"]
    before_latest = latest_json_log(trade_dir)
    before_sig = file_signature(before_latest)
    command = [sys.executable, f"prod/{exchange}_trader.py"]

    result, stdout, stderr = _run_subprocess(
        command=command,
        root=root,
        run_id=run_id,
        venue=exchange,
        mode="paper_recorder",
        strategy_id=cfg["strategy"]["name"],
        timeout_sec=timeout_sec,
        runner=runner,
    )

    after_latest = latest_json_log(trade_dir)
    after_sig = file_signature(after_latest)
    produced_new_log = bool(after_latest and after_sig != before_sig)
    trade_log_path = str(after_latest) if produced_new_log else None
    trade_summary = summarize_trade_log(trade_log_path)
    final_status = result.status
    if result.status == "completed" and not produced_new_log:
        if "already processed" in stdout or "already processed" in stderr:
            final_status = "skipped"
        else:
            final_status = "completed_no_trade_log"

    metadata = result.metadata | {
        "guard": guard,
        "produced_new_log": produced_new_log,
        "trade_summary": trade_summary,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }
    final = ExecutionResult(**(
        result.to_dict()
        | {"status": final_status, "trade_log_path": trade_log_path, "metadata": metadata}
    ))
    return _finalize_result(final, root)
