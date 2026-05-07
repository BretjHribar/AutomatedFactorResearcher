"""Shared signal-generation surface for research and production.

Production traders should consume this module instead of re-implementing alpha
loading, expression evaluation, preprocessing, combination, and QP handling.
The first public contract is intentionally small: build the same weights that
the config-driven research runner builds, then expose the latest target row in
a serializable form.

Incremental signal compute
--------------------------
A scheduled signal recomputation does NOT need the full backtest history; it
only needs the latest target weights row. To bound recompute cost the signal
service slices the input data to the last `max_lookback_bars` bars before the
runner evaluates alphas, combines, and runs the QP. The output of `iloc[-1]`
of the bounded compute is byte-exact equal to the full-history compute as
long as `max_lookback_bars` exceeds the longest lookback used by any alpha,
combiner, or risk model.

Lookback guidance:
- `ts_*(window=N)` operators need exactly N bars to produce a non-NaN value.
- Chained operators stack: `ts_zscore(sma(x, 120), 240)` needs 120+240=360.
- `Decay_exp(x, alpha=a)` is exponentially weighted with EWM(alpha=|a|,
  adjust=True). The truncation error at the last bar is approximately
  (1-|a|)^K — the weight that would have been carried by bars before the
  slice. For fp64-clean equivalence (atol=1e-10) you need
  K >= log(1e-10) / log(1-|a|). Worked points:
    α=0.10 → K≥220   (≈ 2e-10 tail)
    α=0.05 → K≥450   (≈ 9e-11 tail)
    α=0.02 → K≥1140  (≈ 9e-11 tail)
  For α=0.05 K=600 gives ≈ 2e-14 (fp64-clean with margin).
  For α=0.02 K=1500 gives ≈ 8e-14 (fp64-clean with margin).
- Combiners with a `lookback` parameter need that many bars.
- Risk models need `factor_window` (default 126) of return history.

The repo uses `combiner_topn_train` for crypto (no extra lookback) and
`combiner_equal` for equity (no extra lookback). The binding constraints are
therefore (a) the slowest Decay_exp alpha, and (b) the QP `factor_window`.

Production defaults written into the configs:
- equity (daily): `max_lookback_bars=400` (>252 = 1y) covers all current
  alphas; `factor_window=126`, longest single ts_* window is 240.
- crypto (4h): `max_lookback_bars=1500` covers `Decay_exp(α=0.02)` and
  `factor_window=126` for ipca.

Use `verify_incremental_signal_matches` (or `tools/verify_incremental_signal.py`)
to sanity check the bound before deploying any new alpha or combiner.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.runner import PipelineResult, merge_overrides, run


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
    max_lookback_bars: int | None = None


def build_pipeline_weights(config: str | Path | dict, *, root: Path | None = None,
                           verbose: bool = False) -> PipelineResult:
    """Run the canonical config-driven pipeline and return full weights."""
    return run(config, root=root, verbose=verbose)


def _load_config(config: str | Path | dict) -> dict:
    if isinstance(config, (str, Path)):
        return json.loads(Path(config).read_text(encoding="utf-8"))
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"config must be path or dict, got {type(config).__name__}")


def latest_signal_snapshot(
    config: str | Path | dict,
    *,
    root: Path | None = None,
    verbose: bool = False,
    min_abs_weight: float = 0.0,
    max_lookback_bars: int | None = None,
) -> LatestSignalSnapshot:
    """Return the latest target-weight row from the canonical pipeline.

    If `max_lookback_bars` is provided (or `data.max_lookback_bars` is set in
    the config), the runner slices the date index to the last N bars before
    Stage 1. This is intended for scheduled (every-tick) signal recomputes.
    For byte-exact equivalence to a full rerun, ensure N is at least the
    longest lookback used by any alpha, combiner, or risk model. See
    `verify_incremental_signal_matches`.
    """
    cfg = _load_config(config)
    if max_lookback_bars is not None:
        cfg = merge_overrides(cfg, {"data": {"max_lookback_bars": int(max_lookback_bars)}})
    result = run(cfg, root=root, verbose=verbose)
    weights = result.weights.iloc[-1].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if min_abs_weight > 0:
        weights = weights[weights.abs() >= min_abs_weight]

    res_cfg = result.config
    market = str(res_cfg.get("market", "unknown"))
    strategy = str(
        res_cfg.get("strategy", {}).get("name")
        or f"{market}_{res_cfg.get('interval', 'unknown')}"
    )
    effective_max_lookback = res_cfg.get("data", {}).get("max_lookback_bars")
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
        max_lookback_bars=int(effective_max_lookback) if effective_max_lookback is not None else None,
    )


def verify_incremental_signal_matches(
    config: str | Path | dict,
    *,
    max_lookback_bars: int,
    root: Path | None = None,
    atol: float = 1e-10,
    rtol: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the pipeline both with full history and with a bounded slice; assert the
    last weights row matches.

    Use atol=1e-10 for "fp64-clean" verification. If the test fails, raise the
    `max_lookback_bars` until `Decay_exp` and the longest chained `ts_*` window
    are fully covered.

    Returns a dict with diagnostic stats; raises AssertionError on mismatch.
    """
    cfg = _load_config(config)
    full_cfg = merge_overrides(cfg, {"data": {"max_lookback_bars": None}})
    full_cfg["data"].pop("max_lookback_bars", None)  # treat None as absent
    incr_cfg = merge_overrides(cfg, {"data": {"max_lookback_bars": int(max_lookback_bars)}})

    full = run(full_cfg, root=root, verbose=verbose)
    incr = run(incr_cfg, root=root, verbose=verbose)

    full_last = full.weights.iloc[-1]
    incr_last = incr.weights.iloc[-1]

    # Last row must align on the same timestamp.
    assert full.weights.index[-1] == incr.weights.index[-1], (
        f"signal_date mismatch: full={full.weights.index[-1]} incr={incr.weights.index[-1]}"
    )
    # Same column set.
    assert list(full_last.index) == list(incr_last.index), (
        "ticker column ordering mismatch between full and incremental runs"
    )

    diff = (full_last - incr_last).abs()
    max_abs = float(diff.max() if len(diff) else 0.0)
    rel = (full_last - incr_last).abs() / full_last.abs().replace(0, np.nan)
    max_rel = float(rel.max() if len(rel) else 0.0) if not rel.dropna().empty else 0.0

    stats = {
        "max_lookback_bars": int(max_lookback_bars),
        "signal_date": str(full.weights.index[-1]),
        "n_tickers": int(len(full_last)),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "atol": float(atol),
        "rtol": float(rtol),
        "passed": bool(max_abs <= atol and (rtol == 0 or max_rel <= rtol)),
    }
    if not stats["passed"]:
        worst = (full_last - incr_last).abs().nlargest(5)
        stats["worst_offenders"] = {str(k): float(v) for k, v in worst.items()}
        raise AssertionError(
            f"Incremental compute does not match full rerun within tolerance "
            f"(atol={atol}, rtol={rtol}). max_abs_diff={max_abs:.3e}, "
            f"max_rel_diff={max_rel:.3e}. Worst tickers: {stats['worst_offenders']}. "
            f"Increase data.max_lookback_bars."
        )
    return stats


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
