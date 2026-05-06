"""Decision-time contracts shared by research replay, live trading, and Dagster.

The goal is to make a production decision replayable: the same strategy code
should be able to consume the same decision dataset in live mode and research
mode, with explicit data/version/provenance metadata.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal


VwapSource = Literal["fmp_intraday", "quote_tape", "ib_stream", "typical_price"]
ACCEPTED_INTRADAY_VWAP_SOURCES = frozenset({"fmp_intraday", "quote_tape", "ib_stream"})
ALL_VWAP_SOURCES = ACCEPTED_INTRADAY_VWAP_SOURCES | {"typical_price"}


@dataclass(frozen=True)
class DecisionCheck:
    name: str
    status: Literal["pass", "warn", "fail"]
    severity: Literal["info", "warning", "critical"]
    message: str
    value: Any = None
    threshold: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VwapCoverage:
    active_count: int
    covered_count: int
    fallback_count: int
    missing_count: int
    coverage: float
    by_source: dict[str, int]
    fallback_symbols: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)

    def passes(self, *, min_coverage: float, max_fallback_count: int) -> bool:
        return self.coverage >= min_coverage and self.fallback_count <= max_fallback_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_config_hash(config: dict[str, Any]) -> str:
    """Hash config by semantic JSON content rather than dict insertion order."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_sha(root: str | Path) -> str | None:
    """Return the current git SHA if available, without failing live runs."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except Exception:
        return None


def summarize_vwap_sources(
    active_tickers: Iterable[str],
    vwap_sources: dict[str, str] | None,
) -> VwapCoverage:
    """Summarize how many active names have real intraday VWAP provenance."""
    active = sorted({str(t) for t in active_tickers if t})
    sources = {str(k): str(v) for k, v in (vwap_sources or {}).items()}
    by_source = {source: 0 for source in sorted(ALL_VWAP_SOURCES)}
    fallback_symbols: list[str] = []
    missing_symbols: list[str] = []
    covered = 0

    for sym in active:
        source = sources.get(sym)
        if source in ACCEPTED_INTRADAY_VWAP_SOURCES:
            covered += 1
            by_source[source] = by_source.get(source, 0) + 1
        elif source == "typical_price":
            by_source[source] = by_source.get(source, 0) + 1
            fallback_symbols.append(sym)
        else:
            missing_symbols.append(sym)

    active_count = len(active)
    coverage = covered / active_count if active_count else 1.0
    return VwapCoverage(
        active_count=active_count,
        covered_count=covered,
        fallback_count=len(fallback_symbols),
        missing_count=len(missing_symbols),
        coverage=float(coverage),
        by_source={k: int(v) for k, v in by_source.items() if v},
        fallback_symbols=fallback_symbols,
        missing_symbols=missing_symbols,
    )


@dataclass(frozen=True)
class DecisionDataset:
    """Serializable manifest for one strategy decision."""

    strategy_id: str
    market: str
    signal_date: str
    decision_time_utc: str
    data_last_bar: str
    active_tickers: list[str]
    alpha_ids: list[int | str] = field(default_factory=list)
    config_hash: str | None = None
    git_sha: str | None = None
    vwap_sources: dict[str, str] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def vwap_coverage(self) -> VwapCoverage:
        return summarize_vwap_sources(self.active_tickers, self.vwap_sources)

    def validate(
        self,
        *,
        min_active_tickers: int = 10,
        min_alpha_count: int = 1,
        min_vwap_coverage: float = 0.95,
        max_vwap_fallback_count: int = 0,
    ) -> list[DecisionCheck]:
        checks: list[DecisionCheck] = []
        active_count = len(set(self.active_tickers))
        if active_count >= min_active_tickers:
            checks.append(DecisionCheck(
                "active_universe_size",
                "pass",
                "info",
                "Active strategy universe is populated.",
                value=active_count,
                threshold=f">={min_active_tickers}",
            ))
        else:
            checks.append(DecisionCheck(
                "active_universe_size",
                "fail",
                "critical",
                "Active strategy universe is too small for a production decision.",
                value=active_count,
                threshold=f">={min_active_tickers}",
            ))

        alpha_count = len(set(self.alpha_ids))
        if alpha_count >= min_alpha_count:
            checks.append(DecisionCheck(
                "alpha_count",
                "pass",
                "info",
                "Alpha set is populated.",
                value=alpha_count,
                threshold=f">={min_alpha_count}",
            ))
        else:
            checks.append(DecisionCheck(
                "alpha_count",
                "fail",
                "critical",
                "No alpha identifiers are attached to the decision dataset.",
                value=alpha_count,
                threshold=f">={min_alpha_count}",
            ))

        coverage = self.vwap_coverage()
        if coverage.passes(
            min_coverage=min_vwap_coverage,
            max_fallback_count=max_vwap_fallback_count,
        ):
            checks.append(DecisionCheck(
                "live_vwap_coverage",
                "pass",
                "info",
                "Active names have sufficient intraday VWAP provenance.",
                value=f"{coverage.coverage:.2%}",
                threshold=f">={min_vwap_coverage:.2%}; fallback<={max_vwap_fallback_count}",
                metadata=coverage.to_dict(),
            ))
        else:
            checks.append(DecisionCheck(
                "live_vwap_coverage",
                "fail",
                "critical",
                "Active names do not have sufficient intraday VWAP provenance.",
                value=f"{coverage.coverage:.2%}",
                threshold=f">={min_vwap_coverage:.2%}; fallback<={max_vwap_fallback_count}",
                metadata=coverage.to_dict(),
            ))
        return checks

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionDataset":
        return cls(**payload)


def write_decision_dataset(dataset: DecisionDataset, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return out


def read_decision_dataset(path: str | Path) -> DecisionDataset:
    return DecisionDataset.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

