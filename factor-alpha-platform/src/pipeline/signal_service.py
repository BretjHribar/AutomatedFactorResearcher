"""Shared signal-generation surface for research and production.

Production traders should consume this module instead of re-implementing alpha
loading, expression evaluation, preprocessing, combination, and QP handling.
The first public contract is intentionally small: build the same weights that
the config-driven research runner builds, then expose the latest target row in
a serializable form.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.runner import PipelineResult, run


@dataclass(frozen=True)
class LatestSignalSnapshot:
    strategy: str
    market: str
    signal_date: str
    config_notes: list[str]
    alpha_signals_n: int
    universe_size: int
    gross_exposure: float
    net_exposure: float
    n_positions: int
    weights: dict[str, float]
    metrics: dict[str, dict[str, float]]


def build_pipeline_weights(config: str | Path | dict, *, root: Path | None = None,
                           verbose: bool = False) -> PipelineResult:
    """Run the canonical config-driven pipeline and return full weights."""
    return run(config, root=root, verbose=verbose)


def latest_signal_snapshot(config: str | Path | dict, *, root: Path | None = None,
                           verbose: bool = False, min_abs_weight: float = 0.0) -> LatestSignalSnapshot:
    """Return the latest target-weight row from the canonical pipeline."""
    result = build_pipeline_weights(config, root=root, verbose=verbose)
    weights = result.weights.iloc[-1].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if min_abs_weight > 0:
        weights = weights[weights.abs() >= min_abs_weight]

    cfg = result.config
    market = str(cfg.get("market", "unknown"))
    strategy = str(
        cfg.get("strategy", {}).get("name")
        or f"{market}_{cfg.get('interval', 'unknown')}"
    )
    return LatestSignalSnapshot(
        strategy=strategy,
        market=market,
        signal_date=str(result.weights.index[-1]),
        config_notes=list(result.notes),
        alpha_signals_n=result.alpha_signals_n,
        universe_size=result.universe_size,
        gross_exposure=float(weights.abs().sum()),
        net_exposure=float(weights.sum()),
        n_positions=int((weights != 0).sum()),
        weights={str(k): float(v) for k, v in weights.items()},
        metrics=result.metrics,
    )


def write_latest_signal(config: str | Path | dict, output_path: str | Path, *,
                        root: Path | None = None, verbose: bool = False,
                        min_abs_weight: float = 0.0) -> LatestSignalSnapshot:
    """Build and write the latest signal snapshot as JSON."""
    snapshot = latest_signal_snapshot(
        config, root=root, verbose=verbose, min_abs_weight=min_abs_weight
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(snapshot), indent=2), encoding="utf-8")
    return snapshot


def record_strategy_state(snapshot: LatestSignalSnapshot, *, market: str,
                          mode: str = "paper", status: str = "signal_ready",
                          state_db: str | Path = "data/prod_state.db") -> None:
    """Upsert dashboard strategy state from a signal snapshot."""
    from src.monitoring.state_store import StrategyState, upsert_strategy_state

    upsert_strategy_state(
        StrategyState(
            strategy_id=snapshot.strategy,
            market=market,
            mode=mode,
            status=status,
            last_data_bar=snapshot.signal_date,
            last_signal_bar=snapshot.signal_date,
            gross_exposure=snapshot.gross_exposure,
            net_exposure=snapshot.net_exposure,
            n_positions=snapshot.n_positions,
            metrics_json=snapshot.metrics,
        ),
        db_path=state_db,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical strategy signal snapshot.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-abs-weight", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--write-state", action="store_true")
    parser.add_argument("--state-db", type=Path, default=Path("data/prod_state.db"))
    parser.add_argument("--mode", default="paper")
    parser.add_argument("--status", default="signal_ready")
    args = parser.parse_args()

    snapshot = latest_signal_snapshot(
        args.config,
        verbose=args.verbose,
        min_abs_weight=args.min_abs_weight,
    )
    payload: dict[str, Any] = asdict(snapshot)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    if args.write_state:
        record_strategy_state(
            snapshot,
            market=snapshot.market,
            mode=args.mode,
            status=args.status,
            state_db=args.state_db,
        )
        print(f"updated {args.state_db}")


if __name__ == "__main__":
    main()
