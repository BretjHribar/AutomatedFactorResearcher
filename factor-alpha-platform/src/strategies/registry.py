"""Load strategy configs and produce target positions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.execution.netting import StrategyTarget
from src.pipeline.signal_service import latest_signal_snapshot


class StaleStrategySignal(RuntimeError):
    """Raised when a strategy signal is older than its configured data source."""


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    name: str
    enabled: bool
    asset_type: str
    venue: str
    account: str
    order_type: str
    broker_bucket: str
    book: float
    min_order_value: float
    signal: dict[str, Any]
    metadata: dict[str, Any]
    path: Path | None = None

    @property
    def route(self) -> dict[str, str]:
        return {
            "asset_type": self.asset_type,
            "venue": self.venue,
            "account": self.account,
            "bucket": self.broker_bucket,
            "order_type": self.order_type,
        }


def load_strategy_configs(config_dir: str | Path) -> list[StrategyConfig]:
    config_dir = Path(config_dir)
    configs: list[StrategyConfig] = []
    if not config_dir.exists():
        return configs
    for path in sorted(config_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        configs.append(parse_strategy_config(raw, path=path))
    return configs


def parse_strategy_config(raw: dict[str, Any], *, path: Path | None = None) -> StrategyConfig:
    execution = raw.get("execution") or {}
    portfolio = raw.get("portfolio") or {}
    route = raw.get("route") or {}
    strategy_id = str(raw["strategy_id"])
    asset_type = str(route.get("asset_type") or raw.get("asset_type") or "unknown")
    venue = str(route.get("venue") or raw.get("venue") or "unknown")
    account = str(route.get("account") or raw.get("account") or "paper")
    order_type = str(execution.get("order_type") or "MOC")
    bucket = str(route.get("bucket") or f"{asset_type}:{venue}:{account}:{order_type}")
    return StrategyConfig(
        strategy_id=strategy_id,
        name=str(raw.get("name") or strategy_id),
        enabled=bool(raw.get("enabled", True)),
        asset_type=asset_type,
        venue=venue,
        account=account,
        order_type=order_type,
        broker_bucket=bucket,
        book=float(portfolio.get("book") or raw.get("book") or 1.0),
        min_order_value=float(execution.get("min_order_value") or 0.0),
        signal=dict(raw.get("signal") or {}),
        metadata=dict(raw.get("metadata") or {}),
        path=path,
    )


def build_targets(config: StrategyConfig, *, root: str | Path) -> list[StrategyTarget]:
    adapter = str(config.signal.get("adapter") or "")
    if adapter == "pipeline_latest_signal":
        return _pipeline_latest_targets(config, root=Path(root))
    if adapter == "aipt_weights_tail_artifact":
        return _artifact_weight_targets(config, root=Path(root))
    raise ValueError(f"unsupported strategy adapter {adapter!r} for {config.strategy_id}")


def load_latest_price_map(config: StrategyConfig, *, root: str | Path) -> dict[str, float]:
    price_matrix = config.signal.get("price_matrix")
    if not price_matrix:
        return {}
    close = _latest_close(Path(root), str(price_matrix))
    return {str(symbol): float(price) for symbol, price in close.dropna().items() if float(price) > 0}


def _pipeline_latest_targets(config: StrategyConfig, *, root: Path) -> list[StrategyTarget]:
    signal_cfg = config.signal
    snapshot = latest_signal_snapshot(
        root / signal_cfg["config_path"],
        root=root,
        max_lookback_bars=signal_cfg.get("max_lookback_bars"),
        min_abs_weight=float(signal_cfg.get("min_abs_weight") or 0.0),
        exclude_tickers=list(signal_cfg.get("exclude_tickers") or []),
    )
    close = _latest_close(root, signal_cfg["price_matrix"])
    return _weights_to_targets(
        config,
        weights=snapshot.weights,
        prices=close,
        signal_time=snapshot.signal_date,
        metadata={
            "adapter": "pipeline_latest_signal",
            "source_config": signal_cfg["config_path"],
            "alpha_signals_n": snapshot.alpha_signals_n,
            "universe_size": snapshot.universe_size,
            "metrics": snapshot.metrics,
        },
    )


def _artifact_weight_targets(config: StrategyConfig, *, root: Path) -> list[StrategyTarget]:
    signal_cfg = config.signal
    weights_path = root / signal_cfg["weights_path"]
    if not weights_path.exists():
        raise FileNotFoundError(f"AIPT weights artifact not found: {weights_path}")
    weights_row, signal_time = _read_weight_artifact(weights_path)
    _check_artifact_freshness(config, root=root, signal_time=signal_time)
    weights_row = weights_row.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if "min_abs_weight" in signal_cfg:
        weights_row = weights_row[weights_row.abs() >= float(signal_cfg["min_abs_weight"])]
    close = _latest_close(root, signal_cfg["price_matrix"])
    metadata = {
        "adapter": "aipt_weights_tail_artifact",
        "weights_path": signal_cfg["weights_path"],
        "artifact_signal_time": signal_time,
        "research": config.metadata.get("research", {}),
        "production_note": (
            "Artifact-backed AIPT strategy for initial IB-paper rollout. "
            "Next step is replacing this with a live daily AIPT signal generator."
        ),
    }
    return _weights_to_targets(
        config,
        weights={str(k): float(v) for k, v in weights_row.items()},
        prices=close,
        signal_time=signal_time,
        metadata=metadata,
    )


def _read_weight_artifact(path: Path) -> tuple[pd.Series, str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        weights = payload.get("weights") or {}
        signal_time = str(payload.get("signal_time") or "")
        if not weights:
            raise ValueError(f"AIPT weights artifact is empty: {path}")
        if not signal_time:
            raise ValueError(f"AIPT weights artifact has no signal_time: {path}")
        return pd.Series({str(k): float(v) for k, v in weights.items()}, dtype=float), signal_time
    weights_df = pd.read_parquet(path)
    if weights_df.empty:
        raise ValueError(f"AIPT weights artifact is empty: {path}")
    return weights_df.iloc[-1], str(weights_df.index[-1])


def _check_artifact_freshness(config: StrategyConfig, *, root: Path, signal_time: str) -> None:
    signal_cfg = config.signal
    max_lag_raw = signal_cfg.get("max_signal_lag_days")
    if max_lag_raw is None:
        return
    price_matrix = signal_cfg.get("price_matrix")
    if not price_matrix:
        return
    price_path = root / str(price_matrix)
    if not price_path.exists():
        return
    close = pd.read_parquet(price_path)
    if close.empty:
        return
    data_ts = pd.Timestamp(close.index[-1])
    signal_ts = pd.Timestamp(signal_time)
    max_lag_days = int(max_lag_raw)
    lag_days = (data_ts.normalize() - signal_ts.normalize()).days
    if lag_days <= max_lag_days:
        return
    action = str(signal_cfg.get("stale_action") or "raise").lower()
    message = (
        f"{config.strategy_id} signal is stale: signal={signal_ts.date()} "
        f"data={data_ts.date()} lag_days={lag_days} max={max_lag_days}"
    )
    if action in {"skip", "skip_strategy"}:
        raise StaleStrategySignal(message)
    raise RuntimeError(message)


def _weights_to_targets(
    config: StrategyConfig,
    *,
    weights: dict[str, float],
    prices: pd.Series,
    signal_time: str,
    metadata: dict[str, Any],
) -> list[StrategyTarget]:
    targets: list[StrategyTarget] = []
    for symbol, weight in sorted(weights.items()):
        weight = float(weight)
        if abs(weight) <= 1e-12:
            continue
        price = float(prices.get(symbol, float("nan")))
        if not price or price <= 0 or pd.isna(price):
            continue
        target_notional = weight * config.book
        target_qty = round(target_notional / price)
        if abs(target_qty) <= 0:
            continue
        targets.append(
            StrategyTarget(
                strategy_id=config.strategy_id,
                asset_type=config.asset_type,
                venue=config.venue,
                account=config.account,
                bucket=config.broker_bucket,
                symbol=symbol,
                target_qty=float(target_qty),
                target_notional=float(target_qty * price),
                price=price,
                weight=weight,
                order_type=config.order_type,
                signal_time=signal_time,
                metadata=metadata | {"strategy_name": config.name},
            )
        )
    return targets


def _latest_close(root: Path, rel_path: str) -> pd.Series:
    path = root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"price matrix not found: {path}")
    close = pd.read_parquet(path)
    if close.empty:
        raise ValueError(f"price matrix is empty: {path}")
    return close.iloc[-1]
