"""Path/config helpers for orchestration code.

This module intentionally has no Dagster dependency so it can be unit-tested
and reused by scripts before the cloud orchestration stack is installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlatformPaths:
    root: Path = PROJECT_ROOT
    strategy_config_rel: str = "prod/config/strategy.json"
    research_equity_config_rel: str = "prod/config/research_equity.json"
    research_crypto_config_rel: str = "prod/config/research_crypto.json"
    state_db_rel: str = "data/prod_state.db"

    def resolve(self, rel_or_abs: str | Path) -> Path:
        path = Path(rel_or_abs)
        return path if path.is_absolute() else self.root / path

    @property
    def strategy_config(self) -> Path:
        return self.resolve(self.strategy_config_rel)

    @property
    def research_equity_config(self) -> Path:
        return self.resolve(self.research_equity_config_rel)

    @property
    def research_crypto_config(self) -> Path:
        return self.resolve(self.research_crypto_config_rel)

    @property
    def state_db(self) -> Path:
        return self.resolve(self.state_db_rel)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_strategy_config(paths: PlatformPaths | None = None) -> dict[str, Any]:
    paths = paths or PlatformPaths()
    return load_json(paths.strategy_config)


def active_universe_tickers(strategy_config: dict[str, Any], paths: PlatformPaths | None = None) -> list[str]:
    """Load the latest active strategy universe from the configured parquet."""
    paths = paths or PlatformPaths()
    universe_name = strategy_config["strategy"]["universe"]
    universes_dir = paths.resolve(strategy_config["paths"]["universes"])
    universe_path = universes_dir / f"{universe_name}.parquet"
    df = pd.read_parquet(universe_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    latest = df.iloc[-1].fillna(False).astype(bool)
    return sorted(str(sym) for sym, active in latest.items() if bool(active))

