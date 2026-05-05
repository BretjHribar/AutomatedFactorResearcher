"""Reusable data-integrity checks for research, production, and dashboards."""
from __future__ import annotations

import argparse
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    severity: str
    message: str
    value: Any = None
    threshold: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _ok(name: str, message: str, *, value: Any = None, threshold: Any = None,
        severity: str = "info", metadata: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, "pass", severity, message, value, threshold, metadata or {})


def _fail(name: str, message: str, *, value: Any = None, threshold: Any = None,
          severity: str = "critical", metadata: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, "fail", severity, message, value, threshold, metadata or {})


def _warn(name: str, message: str, *, value: Any = None, threshold: Any = None,
          metadata: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(name, "warn", "warning", message, value, threshold, metadata or {})


def _load_matrix(matrices_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_parquet(matrices_dir / f"{name}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def expected_nyse_dates(first: date, last: date) -> set[date]:
    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=str(first), end_date=str(last))
        return {d.date() for d in schedule.index}
    except Exception:
        return {d.date() for d in pd.bdate_range(first, last)}


def previous_nyse_trading_day(today: date | None = None) -> date:
    today = today or datetime.now().date()
    expected = expected_nyse_dates(today - timedelta(days=10), today)
    prior = sorted(d for d in expected if d < today)
    if prior:
        return prior[-1]
    probe = today
    while True:
        probe -= timedelta(days=1)
        if probe.weekday() < 5:
            return probe


def check_calendar(close: pd.DataFrame) -> list[CheckResult]:
    first = close.index.min().date()
    last = close.index.max().date()
    cached = {d.date() for d in close.index}
    expected = expected_nyse_dates(first, last)
    missing = sorted(expected - cached)
    extra = sorted(cached - expected)
    results: list[CheckResult] = []
    if missing:
        results.append(_fail(
            "calendar_missing_sessions",
            f"{len(missing)} NYSE trading sessions are missing from close.parquet.",
            value=len(missing),
            threshold=0,
            metadata={"first_missing": [str(d) for d in missing[:10]]},
        ))
    else:
        results.append(_ok("calendar_missing_sessions", "No NYSE trading sessions are missing.", value=0))
    if len(extra) > 5:
        results.append(_fail(
            "calendar_extra_sessions",
            f"{len(extra)} cached dates are not NYSE sessions.",
            value=len(extra),
            threshold="<=5",
            metadata={"first_extra": [str(d) for d in extra[:10]]},
        ))
    else:
        results.append(_ok("calendar_extra_sessions", "Cached dates match the NYSE calendar tolerance.", value=len(extra)))

    expected_last = previous_nyse_trading_day()
    if last < expected_last:
        results.append(_fail(
            "latest_bar_freshness",
            f"close.parquet ends {last}; expected at least {expected_last}.",
            value=str(last),
            threshold=str(expected_last),
        ))
    else:
        results.append(_ok(
            "latest_bar_freshness",
            f"close.parquet is fresh through {last}.",
            value=str(last),
            threshold=str(expected_last),
        ))
    return results


def check_classifications(root: Path) -> list[CheckResult]:
    matrices_dir = root / "data" / "fmp_cache" / "matrices"
    sector = _load_matrix(matrices_dir, "sector").iloc[-1]
    industry = _load_matrix(matrices_dir, "industry").iloc[-1]
    subindustry = _load_matrix(matrices_dir, "subindustry").iloc[-1]
    n_sector = int(sector.nunique())
    n_industry = int(industry.nunique())
    n_subindustry = int(subindustry.nunique())
    same_pct = float((industry == subindustry).mean() * 100)
    if not (8 <= n_sector <= 15 and n_industry > n_sector and n_subindustry > n_industry and same_pct < 20):
        return [_fail(
            "classification_hierarchy",
            "Classification hierarchy is degenerate or duplicated.",
            value=f"{n_sector}/{n_industry}/{n_subindustry}; same={same_pct:.1f}%",
            threshold="8-15 sectors; industry>sector; subindustry>industry; same<20%",
        )]
    return [_ok(
        "classification_hierarchy",
        "Classification hierarchy is granular and non-duplicated.",
        value=f"{n_sector}/{n_industry}/{n_subindustry}; same={same_pct:.1f}%",
    )]


def check_ohlc(matrices_dir: Path) -> list[CheckResult]:
    o = _load_matrix(matrices_dir, "open")
    h = _load_matrix(matrices_dir, "high")
    l = _load_matrix(matrices_dir, "low")
    c = _load_matrix(matrices_dir, "close")
    common = sorted(set(o.columns) & set(h.columns) & set(l.columns) & set(c.columns))
    o, h, l, c = o[common], h[common], l[common], c[common]
    low_gt_high = int((l > h + 0.01).sum().sum())
    total = int(o.notna().sum().sum())
    open_bad = int(((o < l - 0.01) | (o > h + 0.01)).sum().sum())
    close_bad = int(((c < l - 0.01) | (c > h + 0.01)).sum().sum())
    bad_pct = ((open_bad + close_bad) / max(total * 2, 1)) * 100
    if low_gt_high >= 100 or bad_pct >= 1.0:
        return [_fail(
            "ohlc_consistency",
            "OHLC bars violate low/high bounds above tolerance.",
            value=f"low>high={low_gt_high}; bad_pct={bad_pct:.3f}",
            threshold="low>high<100 and bad_pct<1%",
        )]
    return [_ok(
        "ohlc_consistency",
        "OHLC bars are within configured tolerance.",
        value=f"low>high={low_gt_high}; bad_pct={bad_pct:.3f}",
    )]


def check_universe(universes_dir: Path, name: str) -> list[CheckResult]:
    path = universes_dir / f"{name}.parquet"
    if not path.exists():
        return [_warn("universe_exists", f"{name}.parquet does not exist.", value=name)]
    df = pd.read_parquet(path)
    avg_members = float(df.sum(axis=1).mean())
    last_members = int(df.iloc[-1].sum())
    if avg_members <= 0 or last_members <= 0:
        return [_fail(
            "universe_membership",
            f"{name} has no active members.",
            value=f"avg={avg_members:.0f}; last={last_members}",
            threshold=">0",
        )]
    return [_ok(
        "universe_membership",
        f"{name} has active membership.",
        value=f"avg={avg_members:.0f}; last={last_members}",
    )]


def check_alpha_database(db_path: Path) -> list[CheckResult]:
    if not db_path.exists() or db_path.stat().st_size == 0:
        return [_fail("alpha_database_exists", f"{db_path} is missing or empty.", value=str(db_path))]
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        active_equity = 0
        active_crypto = 0
        if "alphas" in tables:
            active_equity = conn.execute("SELECT COUNT(*) FROM alphas WHERE archived=0").fetchone()[0]
        if "alphas_crypto" in tables:
            active_crypto = conn.execute("SELECT COUNT(*) FROM alphas_crypto WHERE archived=0").fetchone()[0]
    if active_equity + active_crypto == 0:
        return [_fail("alpha_database_active_alphas", "No active alphas are available.", value=0, threshold=">0")]
    return [_ok(
        "alpha_database_active_alphas",
        "Active alpha rows are available.",
        value=f"equity={active_equity}; crypto={active_crypto}",
    )]


def run_equity_integrity(root: Path = PROJECT_ROOT, *,
                         universe_name: str = "MCAP_100M_500M",
                         db_name: str = "alpha_results.db") -> list[CheckResult]:
    matrices_dir = root / "data" / "fmp_cache" / "matrices"
    universes_dir = root / "data" / "fmp_cache" / "universes"
    results: list[CheckResult] = []
    close = _load_matrix(matrices_dir, "close")
    results.extend(check_calendar(close))
    results.extend(check_classifications(root))
    results.extend(check_ohlc(matrices_dir))
    results.extend(check_universe(universes_dir, universe_name))
    results.extend(check_alpha_database(root / "data" / db_name))
    return results


def run_crypto_integrity(root: Path = PROJECT_ROOT, *,
                         matrices_rel: str = "data/kucoin_cache/matrices/4h",
                         universe_rel: str = "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet",
                         db_name: str = "alpha_results.db") -> list[CheckResult]:
    matrices_dir = root / matrices_rel
    results: list[CheckResult] = []
    close = _load_matrix(matrices_dir, "close")
    latest = close.index.max()
    stale_hours = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(latest)).total_seconds() / 3600
    if stale_hours > 24:
        results.append(_warn("crypto_latest_bar_freshness", f"Latest crypto bar is {stale_hours:.1f} hours old.", value=f"{stale_hours:.1f}", threshold="<=24h"))
    else:
        results.append(_ok("crypto_latest_bar_freshness", "Latest crypto bar is fresh.", value=f"{stale_hours:.1f}", threshold="<=24h"))
    results.extend(check_ohlc(matrices_dir))
    universe_path = root / universe_rel
    if universe_path.exists():
        df = pd.read_parquet(universe_path)
        results.append(_ok("crypto_universe_exists", "Crypto universe parquet exists.", value=str(df.shape)))
    else:
        results.append(_fail("crypto_universe_exists", "Crypto universe parquet is missing.", value=str(universe_path)))
    results.extend(check_alpha_database(root / "data" / db_name))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured data-integrity checks.")
    parser.add_argument("--market", choices=["equity", "crypto", "all"], default="all")
    parser.add_argument("--write-db", action="store_true")
    parser.add_argument("--state-db", type=Path, default=Path("data/prod_state.db"))
    args = parser.parse_args()

    markets = ["equity", "crypto"] if args.market == "all" else [args.market]
    run_id = f"integrity-{uuid.uuid4().hex[:12]}"
    any_fail = False
    for market in markets:
        results = run_equity_integrity() if market == "equity" else run_crypto_integrity()
        if args.write_db:
            from src.monitoring.state_store import record_check_results

            record_check_results(results, run_id=run_id, market=market, db_path=args.state_db)
        print(f"\n[{market}]")
        for result in results:
            print(f"  {result.status.upper():4s} {result.name}: {result.message} ({result.value})")
            any_fail = any_fail or result.status == "fail"
    raise SystemExit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
