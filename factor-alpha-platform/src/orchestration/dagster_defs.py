"""Dagster asset graph for production/research orchestration.

These assets are thin wrappers around the existing shared code. Dagster should
own orchestration, lineage, checks, schedules, and observability; the strategy
logic should stay in `src.pipeline.*`, `src.data.*`, and `prod/live_bar.py`.
"""
from __future__ import annotations

import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

import dagster as dg

from src.data.equity_refresh import refresh_equity_eod_cache
from src.data.integrity import run_crypto_integrity, run_equity_integrity
from src.execution.ib_gateway_health import check_ib_gateway, ib_gateway_status_to_check_payload
from src.execution.recorders import run_crypto_paper_recorder, run_ib_paper_moc
from src.monitoring.state_store import (
    StrategyState,
    latest_checks,
    open_alerts,
    record_check_results,
    strategy_states,
    sync_alerts_from_latest_checks,
    upsert_strategy_state,
)
from src.orchestration.paths import PROJECT_ROOT, PlatformPaths, active_universe_tickers, load_json
from src.pipeline.signal_service import latest_signal_snapshot
from src.runtime.decision import stable_config_hash


class PlatformPathsResource(dg.ConfigurableResource):
    """Runtime path configuration for local, paper, and cloud-box deployments."""

    root: str = str(PROJECT_ROOT)
    strategy_config_rel: str = "prod/config/strategy.json"
    research_equity_config_rel: str = "prod/config/research_equity.json"
    research_crypto_config_rel: str = "prod/config/research_crypto.json"
    state_db_rel: str = "data/prod_state.db"

    def platform_paths(self) -> PlatformPaths:
        return PlatformPaths(
            root=Path(self.root),
            strategy_config_rel=self.strategy_config_rel,
            research_equity_config_rel=self.research_equity_config_rel,
            research_crypto_config_rel=self.research_crypto_config_rel,
            state_db_rel=self.state_db_rel,
        )


class ExecutionControlsResource(dg.ConfigurableResource):
    """Guardrails for assets that can create paper orders or paper fills."""

    allow_ib_paper_orders: bool = os.environ.get("ALLOW_IB_PAPER_ORDERS", "0") == "1"
    force_ib_after_deadline: bool = os.environ.get("FORCE_IB_PAPER_MOC", "0") == "1"
    ib_timeout_sec: int = int(os.environ.get("IB_PAPER_EXEC_TIMEOUT_SEC", "1200"))
    crypto_timeout_sec: int = int(os.environ.get("CRYPTO_PAPER_EXEC_TIMEOUT_SEC", "900"))


def _check_payloads(results: list[Any]) -> list[dict[str, Any]]:
    return [asdict(r) if hasattr(r, "__dataclass_fields__") else dict(r) for r in results]


def _failure_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("status") == "fail")


def _warning_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if r.get("status") == "warn")


def _live_bar_module(root: Path):
    prod_dir = root / "prod"
    if str(prod_dir) not in sys.path:
        sys.path.insert(0, str(prod_dir))
    import live_bar

    return live_bar


def _crypto_universe_rel(exchange: str, cfg: dict[str, Any]) -> str:
    universe = cfg["strategy"]["universe"]
    return f"data/{exchange}_cache/universes/{exchange.upper()}_{universe}_4h.parquet"


def _latest_execution_summaries(root: Path, limit: int = 20) -> list[dict[str, Any]]:
    log_dir = root / "prod/logs/execution"
    if not log_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = load_json(path)
            rows.append({
                "path": str(path),
                "venue": payload.get("venue"),
                "mode": payload.get("mode"),
                "status": payload.get("status"),
                "started_at_utc": payload.get("started_at_utc"),
                "elapsed_sec": payload.get("elapsed_sec"),
                "message": payload.get("message"),
                "trade_log_path": payload.get("trade_log_path"),
            })
        except Exception as exc:
            rows.append({"path": str(path), "status": "unreadable", "message": str(exc)})
    return rows


@dg.asset(group_name="config")
def production_strategy_config(context, paths: PlatformPathsResource) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    cfg = load_json(platform_paths.strategy_config)
    cfg_hash = stable_config_hash(cfg)
    context.add_output_metadata({
        "strategy": cfg["strategy"]["name"],
        "version": cfg["strategy"]["version"],
        "universe": cfg["strategy"]["universe"],
        "min_alpha_sharpe": cfg["strategy"].get("min_alpha_sharpe"),
        "config_hash": cfg_hash,
    })
    return cfg | {"_config_hash": cfg_hash}


@dg.asset(group_name="data")
def equity_eod_data_refresh_result(
    context,
    paths: PlatformPathsResource,
    production_strategy_config: dict[str, Any],
) -> dict[str, Any]:
    """Refresh the FMP EOD equity cache and verify the required NYSE bar.

    Hourly EOD schedule runs recheck recent vendor bars to catch late fills and
    revisions. Intraday preflight/signal/execution jobs only repair if the
    historical cache is behind the required bar, keeping the MOC window lean
    when the hourly refresh has done its job.
    """
    platform_paths = paths.platform_paths()
    job_name = getattr(context, "job_name", "") or ""
    recheck_recent = job_name == "equity_eod_data_refresh_job"
    result = refresh_equity_eod_cache(
        root=platform_paths.root,
        universe_name=production_strategy_config["strategy"]["universe"],
        recheck_recent=recheck_recent,
        max_workers=int(os.environ.get("EQUITY_EOD_REFRESH_WORKERS", "5")),
        overlap_days=int(os.environ.get("EQUITY_EOD_REFRESH_OVERLAP_DAYS", "7")),
        min_active_coverage=float(os.environ.get("EQUITY_EOD_MIN_ACTIVE_COVERAGE", "0.99")),
    )
    coverage = result.get("active_coverage") or {}
    context.add_output_metadata({
        "status": result["status"],
        "message": result["message"],
        "expected_bar_date": result["expected_bar_date"],
        "matrix_end_before": result.get("matrix_end_before") or "",
        "matrix_end_after": result.get("matrix_end_after") or "",
        "latest_price_date_seen": result.get("latest_price_date_seen") or "",
        "symbols_checked": result.get("symbols_checked", 0),
        "symbols_updated": result.get("symbols_updated", 0),
        "symbols_failed": result.get("symbols_failed", 0),
        "rebuilt": result.get("rebuilt", False),
        "active_coverage": coverage.get("coverage", 0.0),
        "active_missing_count": coverage.get("missing_count", 0),
        "elapsed_sec": result.get("elapsed_sec", 0.0),
        "recheck_recent": recheck_recent,
    })
    return result


@dg.asset(group_name="integrity")
def ib_gateway_connectivity_status(
    context,
    production_strategy_config: dict[str, Any],
) -> dict[str, Any]:
    """Verify the configured IB Gateway is reachable before equity execution windows."""
    status = check_ib_gateway(production_strategy_config)
    payload = status.to_dict()
    context.add_output_metadata({
        "connected": payload["connected"],
        "host": payload["host"],
        "port": payload["port"],
        "mode": payload["mode"],
        "elapsed_sec": payload["elapsed_sec"],
        "message": payload["message"],
    })
    return payload


@dg.asset(group_name="integrity")
def equity_integrity_results(
    context,
    paths: PlatformPathsResource,
    production_strategy_config: dict[str, Any],
    equity_eod_data_refresh_result: dict[str, Any],
) -> list[dict[str, Any]]:
    platform_paths = paths.platform_paths()
    universe_name = production_strategy_config["strategy"]["universe"]
    db_name = Path(production_strategy_config["paths"]["db"]).name
    payloads = _check_payloads(
        run_equity_integrity(platform_paths.root, universe_name=universe_name, db_name=db_name)
    )
    context.add_output_metadata({
        "checks": len(payloads),
        "failures": _failure_count(payloads),
        "warnings": _warning_count(payloads),
        "eod_refresh_status": equity_eod_data_refresh_result.get("status", ""),
        "eod_matrix_end_after": equity_eod_data_refresh_result.get("matrix_end_after") or "",
    })
    return payloads


@dg.asset(group_name="integrity")
def kucoin_integrity_results(context, paths: PlatformPathsResource) -> list[dict[str, Any]]:
    platform_paths = paths.platform_paths()
    cfg = load_json(platform_paths.root / "prod/config/kucoin.json")
    payloads = _check_payloads(run_crypto_integrity(
        platform_paths.root,
        matrices_rel=cfg["paths"]["matrices"],
        universe_rel=_crypto_universe_rel("kucoin", cfg),
    ))
    context.add_output_metadata({
        "exchange": "kucoin",
        "checks": len(payloads),
        "failures": _failure_count(payloads),
        "warnings": _warning_count(payloads),
    })
    return payloads


@dg.asset(group_name="integrity")
def binance_integrity_results(context, paths: PlatformPathsResource) -> list[dict[str, Any]]:
    platform_paths = paths.platform_paths()
    cfg = load_json(platform_paths.root / "prod/config/binance.json")
    payloads = _check_payloads(run_crypto_integrity(
        platform_paths.root,
        matrices_rel=cfg["paths"]["matrices"],
        universe_rel=_crypto_universe_rel("binance", cfg),
    ))
    context.add_output_metadata({
        "exchange": "binance",
        "checks": len(payloads),
        "failures": _failure_count(payloads),
        "warnings": _warning_count(payloads),
    })
    return payloads


@dg.asset(group_name="integrity")
def crypto_integrity_results(
    context,
    kucoin_integrity_results: list[dict[str, Any]],
    binance_integrity_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads = []
    for exchange, results in (
        ("kucoin", kucoin_integrity_results),
        ("binance", binance_integrity_results),
    ):
        for result in results:
            payload = dict(result)
            payload["metadata"] = dict(payload.get("metadata") or {}) | {"exchange": exchange}
            payload["name"] = f"{exchange}_{payload['name']}"
            payloads.append(payload)
    context.add_output_metadata({
        "checks": len(payloads),
        "failures": _failure_count(payloads),
        "warnings": _warning_count(payloads),
    })
    return payloads


@dg.asset(group_name="integrity")
def equity_integrity_state_write(
    paths: PlatformPathsResource,
    equity_integrity_results: list[dict[str, Any]],
    ib_gateway_connectivity_status: dict[str, Any],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    run_id = f"equity-integrity-{uuid.uuid4().hex[:12]}"
    payloads = equity_integrity_results + [
        ib_gateway_status_to_check_payload(ib_gateway_connectivity_status)
    ]
    record_check_results(
        payloads,
        run_id=run_id,
        market="equity",
        db_path=platform_paths.state_db,
    )
    return {"run_id": run_id, "rows": len(payloads)}


@dg.asset(group_name="integrity")
def crypto_integrity_state_write(
    paths: PlatformPathsResource,
    crypto_integrity_results: list[dict[str, Any]],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    run_id = f"crypto-integrity-{uuid.uuid4().hex[:12]}"
    record_check_results(
        crypto_integrity_results,
        run_id=run_id,
        market="crypto",
        db_path=platform_paths.state_db,
    )
    return {"run_id": run_id, "rows": len(crypto_integrity_results)}


@dg.asset(group_name="integrity")
def integrity_alert_sync(
    paths: PlatformPathsResource,
    equity_integrity_state_write: dict[str, Any],
    crypto_integrity_state_write: dict[str, Any],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    raised = sync_alerts_from_latest_checks(platform_paths.state_db)
    return {
        "raised_or_open": raised,
        "equity_run_id": equity_integrity_state_write["run_id"],
        "crypto_run_id": crypto_integrity_state_write["run_id"],
    }


@dg.asset(group_name="live")
def live_equity_quote_snapshot(
    context,
    paths: PlatformPathsResource,
    production_strategy_config: dict[str, Any],
) -> dict[str, Any]:
    """Append one FMP quote snapshot for the active equity universe."""
    platform_paths = paths.platform_paths()
    tickers = active_universe_tickers(production_strategy_config, platform_paths)
    live_bar = _live_bar_module(platform_paths.root)
    quotes = live_bar.fetch_fmp_live_quotes(tickers)
    snapshot_path = live_bar.save_fmp_quote_snapshot(quotes)
    coverage = len(quotes) / max(len(tickers), 1)
    context.add_output_metadata({
        "active_tickers": len(tickers),
        "quotes": len(quotes),
        "coverage": coverage,
        "snapshot_path": str(snapshot_path) if snapshot_path else "",
    })
    return {
        "active_tickers": len(tickers),
        "quotes": len(quotes),
        "coverage": coverage,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
    }


@dg.asset(group_name="live")
def live_quote_tape_summary(
    context,
    paths: PlatformPathsResource,
    live_equity_quote_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Summarize today's local quote tape for the dashboard and checks."""
    platform_paths = paths.platform_paths()
    live_bar = _live_bar_module(platform_paths.root)
    path = Path(live_equity_quote_snapshot["snapshot_path"] or live_bar.LIVE_QUOTE_TAPE_DIR)
    if path.is_dir():
        path = path / f"fmp_quotes_{pd.Timestamp.utcnow().date().isoformat()}.csv"
    if not path.exists():
        summary = {"exists": False, "path": str(path), "rows": 0, "symbols": 0, "snapshots": 0}
    else:
        tape = pd.read_csv(path)
        summary = {
            "exists": True,
            "path": str(path),
            "rows": int(len(tape)),
            "symbols": int(tape["symbol"].nunique()) if "symbol" in tape else 0,
            "snapshots": int(tape["captured_at_utc"].nunique()) if "captured_at_utc" in tape else 0,
        }
    context.add_output_metadata(summary)
    return summary


@dg.asset(group_name="research")
def research_equity_signal_snapshot(
    context,
    paths: PlatformPathsResource,
    equity_eod_data_refresh_result: dict[str, Any],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    snapshot = latest_signal_snapshot(platform_paths.research_equity_config, root=platform_paths.root)
    payload = asdict(snapshot)
    context.add_output_metadata({
        "strategy": payload["strategy"],
        "market": payload["market"],
        "signal_date": payload["signal_date"],
        "alpha_signals_n": payload["alpha_signals_n"],
        "n_positions": payload["n_positions"],
        "gross_exposure": payload["gross_exposure"],
        "eod_refresh_status": equity_eod_data_refresh_result.get("status", ""),
        "eod_matrix_end_after": equity_eod_data_refresh_result.get("matrix_end_after") or "",
    })
    return payload


@dg.asset(group_name="research")
def research_crypto_signal_snapshot(context, paths: PlatformPathsResource) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    snapshot = latest_signal_snapshot(platform_paths.research_crypto_config, root=platform_paths.root)
    payload = asdict(snapshot)
    context.add_output_metadata({
        "strategy": payload["strategy"],
        "market": payload["market"],
        "signal_date": payload["signal_date"],
        "alpha_signals_n": payload["alpha_signals_n"],
        "n_positions": payload["n_positions"],
        "gross_exposure": payload["gross_exposure"],
    })
    return payload


@dg.asset(group_name="monitoring")
def equity_strategy_state_write(
    paths: PlatformPathsResource,
    research_equity_signal_snapshot: dict[str, Any],
    production_strategy_config: dict[str, Any],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    state = StrategyState(
        strategy_id=research_equity_signal_snapshot["strategy"],
        market="equity",
        mode="paper",
        status="signal_ready",
        last_data_bar=research_equity_signal_snapshot["signal_date"],
        last_signal_bar=research_equity_signal_snapshot["signal_date"],
        config_hash=production_strategy_config["_config_hash"],
        gross_exposure=float(research_equity_signal_snapshot["gross_exposure"]),
        net_exposure=float(research_equity_signal_snapshot["net_exposure"]),
        n_positions=int(research_equity_signal_snapshot["n_positions"]),
        metrics_json=research_equity_signal_snapshot["metrics"],
    )
    upsert_strategy_state(state, db_path=platform_paths.state_db)
    return {"strategy_id": state.strategy_id, "status": state.status}


@dg.asset(group_name="monitoring")
def ops_dashboard_state(paths: PlatformPathsResource, integrity_alert_sync: dict[str, Any]) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    return {
        "strategies": strategy_states(platform_paths.state_db),
        "checks": latest_checks(db_path=platform_paths.state_db),
        "alerts": open_alerts(platform_paths.state_db),
        "executions": _latest_execution_summaries(platform_paths.root),
        "alert_sync": integrity_alert_sync,
    }


@dg.asset(group_name="execution")
def ib_paper_moc_execution_result(
    context,
    paths: PlatformPathsResource,
    execution_controls: ExecutionControlsResource,
    production_strategy_config: dict[str, Any],
    ib_gateway_connectivity_status: dict[str, Any],
    equity_integrity_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Submit the equity MOC workflow to the configured IB paper gateway.

    The wrapper refuses non-paper ports/accounts and writes a structured
    execution record even when disabled, skipped, or failed.
    """
    failures = _failure_count(equity_integrity_results)
    if failures:
        raise dg.Failure(
            "Refusing IB paper execution because equity integrity checks failed.",
            metadata={"failures": failures},
        )
    if execution_controls.allow_ib_paper_orders and not ib_gateway_connectivity_status.get("connected"):
        raise dg.Failure(
            "Refusing IB paper execution because the configured IB Gateway is unreachable.",
            metadata={
                "host": ib_gateway_connectivity_status.get("host"),
                "port": ib_gateway_connectivity_status.get("port"),
                "message": ib_gateway_connectivity_status.get("message"),
            },
        )
    platform_paths = paths.platform_paths()
    result = run_ib_paper_moc(
        root=platform_paths.root,
        allow_orders=execution_controls.allow_ib_paper_orders,
        force=execution_controls.force_ib_after_deadline,
        timeout_sec=execution_controls.ib_timeout_sec,
    )
    payload = result.to_dict()
    context.add_output_metadata({
        "status": payload["status"],
        "venue": payload["venue"],
        "elapsed_sec": payload["elapsed_sec"],
        "summary_path": payload.get("summary_path") or "",
        "trade_log_path": payload.get("trade_log_path") or "",
        "allow_ib_paper_orders": execution_controls.allow_ib_paper_orders,
        "config_universe": production_strategy_config["strategy"]["universe"],
    })
    if payload["status"] == "failed":
        raise dg.Failure(
            payload.get("message") or "IB paper execution failed",
            metadata={
                "summary_path": payload.get("summary_path") or "",
                "stdout_path": payload.get("stdout_path") or "",
                "stderr_path": payload.get("stderr_path") or "",
            },
        )
    return payload


@dg.asset(group_name="execution")
def kucoin_paper_execution_record(
    context,
    paths: PlatformPathsResource,
    execution_controls: ExecutionControlsResource,
    kucoin_integrity_results: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = _failure_count(kucoin_integrity_results)
    if failures:
        raise dg.Failure(
            "Refusing KuCoin paper recorder because crypto integrity checks failed.",
            metadata={"failures": failures},
        )
    platform_paths = paths.platform_paths()
    result = run_crypto_paper_recorder(
        "kucoin",
        root=platform_paths.root,
        timeout_sec=execution_controls.crypto_timeout_sec,
    )
    payload = result.to_dict()
    context.add_output_metadata({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "summary_path": payload.get("summary_path") or "",
        "trade_log_path": payload.get("trade_log_path") or "",
    })
    if payload["status"] == "failed":
        raise dg.Failure(
            payload.get("message") or "KuCoin paper recorder failed",
            metadata={
                "summary_path": payload.get("summary_path") or "",
                "stdout_path": payload.get("stdout_path") or "",
                "stderr_path": payload.get("stderr_path") or "",
            },
        )
    return payload


@dg.asset(group_name="execution")
def binance_paper_execution_record(
    context,
    paths: PlatformPathsResource,
    execution_controls: ExecutionControlsResource,
    binance_integrity_results: list[dict[str, Any]],
) -> dict[str, Any]:
    platform_paths = paths.platform_paths()
    cfg = load_json(platform_paths.root / "prod/config/binance.json")
    execution_cfg = cfg.get("execution") or {}
    if execution_cfg.get("enabled") is False:
        payload = {
            "venue": "binance",
            "mode": "paper_recorder",
            "status": "disabled",
            "elapsed_sec": 0.0,
            "message": execution_cfg.get("disabled_reason") or "Binance paper execution disabled in config.",
            "summary_path": None,
            "trade_log_path": None,
        }
        context.add_output_metadata({
            "status": payload["status"],
            "message": payload["message"],
        })
        return payload

    failures = _failure_count(binance_integrity_results)
    if failures:
        raise dg.Failure(
            "Refusing Binance paper recorder because crypto integrity checks failed.",
            metadata={"failures": failures},
        )
    result = run_crypto_paper_recorder(
        "binance",
        root=platform_paths.root,
        timeout_sec=execution_controls.crypto_timeout_sec,
    )
    payload = result.to_dict()
    context.add_output_metadata({
        "status": payload["status"],
        "elapsed_sec": payload["elapsed_sec"],
        "summary_path": payload.get("summary_path") or "",
        "trade_log_path": payload.get("trade_log_path") or "",
    })
    if payload["status"] == "failed":
        raise dg.Failure(
            payload.get("message") or "Binance paper recorder failed",
            metadata={
                "summary_path": payload.get("summary_path") or "",
                "stdout_path": payload.get("stdout_path") or "",
                "stderr_path": payload.get("stderr_path") or "",
            },
        )
    return payload


@dg.asset_check(asset=production_strategy_config)
def production_config_guard(production_strategy_config: dict[str, Any]) -> dg.AssetCheckResult:
    strategy = production_strategy_config["strategy"]
    passed = (
        strategy.get("universe") == "MCAP_100M_500M"
        and float(strategy.get("min_alpha_sharpe", 0.0)) >= 5.0
    )
    return dg.AssetCheckResult(
        passed=passed,
        metadata={
            "universe": strategy.get("universe"),
            "min_alpha_sharpe": strategy.get("min_alpha_sharpe"),
        },
        description="Production equity config must stay on the midcap Sharpe>5 model.",
    )


@dg.asset_check(asset=equity_eod_data_refresh_result)
def equity_eod_refresh_expected_bar_loaded(equity_eod_data_refresh_result: dict[str, Any]) -> dg.AssetCheckResult:
    status = equity_eod_data_refresh_result.get("status")
    coverage = equity_eod_data_refresh_result.get("active_coverage") or {}
    return dg.AssetCheckResult(
        passed=status in {"completed", "up_to_date"},
        metadata={
            "status": status,
            "message": equity_eod_data_refresh_result.get("message", ""),
            "expected_bar_date": equity_eod_data_refresh_result.get("expected_bar_date", ""),
            "matrix_end_after": equity_eod_data_refresh_result.get("matrix_end_after") or "",
            "active_coverage": coverage.get("coverage", 0.0),
            "active_missing_count": coverage.get("missing_count", 0),
            "symbols_updated": equity_eod_data_refresh_result.get("symbols_updated", 0),
            "symbols_failed": equity_eod_data_refresh_result.get("symbols_failed", 0),
        },
        description="FMP EOD cache must load the expected NYSE bar with active-universe coverage.",
    )


@dg.asset_check(asset=equity_integrity_results)
def equity_integrity_passes(equity_integrity_results: list[dict[str, Any]]) -> dg.AssetCheckResult:
    failures = _failure_count(equity_integrity_results)
    return dg.AssetCheckResult(
        passed=failures == 0,
        metadata={"failures": failures, "warnings": _warning_count(equity_integrity_results)},
    )


@dg.asset_check(asset=ib_gateway_connectivity_status)
def ib_gateway_connectivity_passes(ib_gateway_connectivity_status: dict[str, Any]) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(
        passed=bool(ib_gateway_connectivity_status.get("connected")),
        metadata={
            "host": ib_gateway_connectivity_status.get("host"),
            "port": ib_gateway_connectivity_status.get("port"),
            "mode": ib_gateway_connectivity_status.get("mode"),
            "elapsed_sec": ib_gateway_connectivity_status.get("elapsed_sec"),
            "message": ib_gateway_connectivity_status.get("message"),
        },
        description="IB Gateway must accept an API handshake before equity execution.",
    )


@dg.asset_check(asset=crypto_integrity_results)
def crypto_integrity_passes(crypto_integrity_results: list[dict[str, Any]]) -> dg.AssetCheckResult:
    failures = _failure_count(crypto_integrity_results)
    return dg.AssetCheckResult(
        passed=failures == 0,
        metadata={"failures": failures, "warnings": _warning_count(crypto_integrity_results)},
    )


@dg.asset_check(asset=kucoin_integrity_results)
def kucoin_integrity_passes(kucoin_integrity_results: list[dict[str, Any]]) -> dg.AssetCheckResult:
    failures = _failure_count(kucoin_integrity_results)
    return dg.AssetCheckResult(
        passed=failures == 0,
        metadata={"failures": failures, "warnings": _warning_count(kucoin_integrity_results)},
    )


@dg.asset_check(asset=binance_integrity_results)
def binance_integrity_passes(binance_integrity_results: list[dict[str, Any]]) -> dg.AssetCheckResult:
    failures = _failure_count(binance_integrity_results)
    return dg.AssetCheckResult(
        passed=failures == 0,
        metadata={"failures": failures, "warnings": _warning_count(binance_integrity_results)},
    )


@dg.asset_check(asset=live_equity_quote_snapshot)
def live_quote_snapshot_coverage(live_equity_quote_snapshot: dict[str, Any]) -> dg.AssetCheckResult:
    coverage = float(live_equity_quote_snapshot.get("coverage") or 0.0)
    return dg.AssetCheckResult(
        passed=coverage >= 0.95,
        metadata={"coverage": coverage, "quotes": live_equity_quote_snapshot.get("quotes", 0)},
        description="FMP quote collector should cover at least 95% of active equity names.",
    )


@dg.asset_check(asset=research_equity_signal_snapshot)
def research_equity_signal_nonempty(research_equity_signal_snapshot: dict[str, Any]) -> dg.AssetCheckResult:
    return dg.AssetCheckResult(
        passed=(
            int(research_equity_signal_snapshot.get("alpha_signals_n") or 0) >= 1
            and int(research_equity_signal_snapshot.get("n_positions") or 0) >= 1
        ),
        metadata={
            "alpha_signals_n": research_equity_signal_snapshot.get("alpha_signals_n"),
            "n_positions": research_equity_signal_snapshot.get("n_positions"),
        },
    )


@dg.asset_check(asset=ib_paper_moc_execution_result)
def ib_paper_execution_not_failed(ib_paper_moc_execution_result: dict[str, Any]) -> dg.AssetCheckResult:
    status = ib_paper_moc_execution_result.get("status")
    return dg.AssetCheckResult(
        passed=status in {"completed", "skipped", "blocked"},
        metadata={
            "status": status,
            "summary_path": ib_paper_moc_execution_result.get("summary_path") or "",
            "trade_log_path": ib_paper_moc_execution_result.get("trade_log_path") or "",
        },
        description="IB paper execution wrapper must not fail. Blocked means the safety flag is off.",
    )


@dg.asset_check(asset=kucoin_paper_execution_record)
def kucoin_paper_execution_not_failed(kucoin_paper_execution_record: dict[str, Any]) -> dg.AssetCheckResult:
    status = kucoin_paper_execution_record.get("status")
    return dg.AssetCheckResult(
        passed=status in {"completed", "completed_no_trade_log", "skipped"},
        metadata={
            "status": status,
            "summary_path": kucoin_paper_execution_record.get("summary_path") or "",
            "trade_log_path": kucoin_paper_execution_record.get("trade_log_path") or "",
        },
    )


@dg.asset_check(asset=binance_paper_execution_record)
def binance_paper_execution_not_failed(binance_paper_execution_record: dict[str, Any]) -> dg.AssetCheckResult:
    status = binance_paper_execution_record.get("status")
    return dg.AssetCheckResult(
        passed=status in {"completed", "completed_no_trade_log", "skipped", "disabled"},
        metadata={
            "status": status,
            "summary_path": binance_paper_execution_record.get("summary_path") or "",
            "trade_log_path": binance_paper_execution_record.get("trade_log_path") or "",
        },
    )


equity_integrity_job = dg.define_asset_job(
    "equity_integrity_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        equity_eod_data_refresh_result,
        ib_gateway_connectivity_status,
        equity_integrity_results,
        equity_integrity_state_write,
    ),
)

crypto_integrity_job = dg.define_asset_job(
    "crypto_integrity_job",
    selection=dg.AssetSelection.assets(
        kucoin_integrity_results,
        binance_integrity_results,
        crypto_integrity_results,
        crypto_integrity_state_write,
    ),
)

all_integrity_job = dg.define_asset_job(
    "all_integrity_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        equity_eod_data_refresh_result,
        ib_gateway_connectivity_status,
        equity_integrity_results,
        kucoin_integrity_results,
        binance_integrity_results,
        crypto_integrity_results,
        equity_integrity_state_write,
        crypto_integrity_state_write,
        integrity_alert_sync,
        ops_dashboard_state,
    ),
)

live_quote_collector_job = dg.define_asset_job(
    "live_quote_collector_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        live_equity_quote_snapshot,
        live_quote_tape_summary,
    ),
)

research_signal_job = dg.define_asset_job(
    "research_signal_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        equity_eod_data_refresh_result,
        research_equity_signal_snapshot,
        equity_strategy_state_write,
    ),
)

crypto_research_signal_job = dg.define_asset_job(
    "crypto_research_signal_job",
    selection=dg.AssetSelection.assets(research_crypto_signal_snapshot),
)

ib_paper_moc_execution_job = dg.define_asset_job(
    "ib_paper_moc_execution_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        equity_eod_data_refresh_result,
        ib_gateway_connectivity_status,
        equity_integrity_results,
        ib_paper_moc_execution_result,
    ),
)

equity_eod_data_refresh_job = dg.define_asset_job(
    "equity_eod_data_refresh_job",
    selection=dg.AssetSelection.assets(
        production_strategy_config,
        equity_eod_data_refresh_result,
    ),
    tags={"factor_alpha/eod_refresh": "equity_eod"},
)

crypto_paper_execution_job = dg.define_asset_job(
    "crypto_paper_execution_job",
    selection=dg.AssetSelection.assets(
        kucoin_integrity_results,
        binance_integrity_results,
        kucoin_paper_execution_record,
        binance_paper_execution_record,
    ),
)

kucoin_paper_execution_job = dg.define_asset_job(
    "kucoin_paper_execution_job",
    selection=dg.AssetSelection.assets(
        kucoin_integrity_results,
        kucoin_paper_execution_record,
    ),
)

binance_paper_execution_job = dg.define_asset_job(
    "binance_paper_execution_job",
    selection=dg.AssetSelection.assets(
        binance_integrity_results,
        binance_paper_execution_record,
    ),
)


schedules = [
    dg.ScheduleDefinition(
        name="equity_eod_refresh_hourly_after_close_et",
        job=equity_eod_data_refresh_job,
        cron_schedule="30 16-23 * * 1-5",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="equity_eod_refresh_hourly_overnight_catchup_et",
        job=equity_eod_data_refresh_job,
        cron_schedule="30 0-8 * * 2-6",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="equity_preflight_1515_et",
        job=equity_integrity_job,
        cron_schedule="15 15 * * 1-5",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="live_quote_collector_5m_regular_session",
        job=live_quote_collector_job,
        cron_schedule="*/5 9-15 * * 1-5",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="equity_research_signal_1530_et",
        job=research_signal_job,
        cron_schedule="30 15 * * 1-5",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="ib_paper_moc_execution_1538_et",
        job=ib_paper_moc_execution_job,
        cron_schedule="38 15 * * 1-5",
        execution_timezone="America/New_York",
        default_status=dg.DefaultScheduleStatus.STOPPED,
    ),
    dg.ScheduleDefinition(
        name="post_close_integrity_1610_et",
        job=all_integrity_job,
        cron_schedule="10 16 * * 1-5",
        execution_timezone="America/New_York",
    ),
    dg.ScheduleDefinition(
        name="crypto_research_signal_4h",
        job=crypto_research_signal_job,
        cron_schedule="2 */4 * * *",
        execution_timezone="UTC",
    ),
    dg.ScheduleDefinition(
        name="crypto_paper_execution_4h_utc",
        job=kucoin_paper_execution_job,
        cron_schedule="3 */4 * * *",
        execution_timezone="UTC",
        default_status=dg.DefaultScheduleStatus.STOPPED,
    ),
]


defs = dg.Definitions(
    assets=[
        production_strategy_config,
        equity_eod_data_refresh_result,
        ib_gateway_connectivity_status,
        equity_integrity_results,
        kucoin_integrity_results,
        binance_integrity_results,
        crypto_integrity_results,
        equity_integrity_state_write,
        crypto_integrity_state_write,
        integrity_alert_sync,
        live_equity_quote_snapshot,
        live_quote_tape_summary,
        research_equity_signal_snapshot,
        research_crypto_signal_snapshot,
        equity_strategy_state_write,
        ops_dashboard_state,
        ib_paper_moc_execution_result,
        kucoin_paper_execution_record,
        binance_paper_execution_record,
    ],
    asset_checks=[
        production_config_guard,
        equity_eod_refresh_expected_bar_loaded,
        equity_integrity_passes,
        ib_gateway_connectivity_passes,
        kucoin_integrity_passes,
        binance_integrity_passes,
        crypto_integrity_passes,
        live_quote_snapshot_coverage,
        research_equity_signal_nonempty,
        ib_paper_execution_not_failed,
        kucoin_paper_execution_not_failed,
        binance_paper_execution_not_failed,
    ],
    jobs=[
        equity_eod_data_refresh_job,
        equity_integrity_job,
        crypto_integrity_job,
        all_integrity_job,
        live_quote_collector_job,
        research_signal_job,
        crypto_research_signal_job,
        ib_paper_moc_execution_job,
        crypto_paper_execution_job,
        kucoin_paper_execution_job,
        binance_paper_execution_job,
    ],
    schedules=schedules,
    resources={
        "paths": PlatformPathsResource(),
        "execution_controls": ExecutionControlsResource(),
    },
)
