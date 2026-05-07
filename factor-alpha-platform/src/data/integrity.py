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
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def _utc_naive_timestamp(ts: pd.Timestamp | datetime | str | None = None) -> pd.Timestamp:
    if ts is None:
        return pd.Timestamp.now("UTC").tz_localize(None)
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out
    return out.tz_convert("UTC").tz_localize(None)


def _expected_completed_bar_start(
    now_ts: pd.Timestamp | datetime | str | None = None,
    *,
    freq_hours: int = 4,
    grace_minutes: int = 6,
) -> pd.Timestamp:
    """Return the candle-open timestamp the local cache should contain."""
    now = _utc_naive_timestamp(now_ts)
    current_boundary = now.floor(f"{freq_hours}h")
    expected = current_boundary - pd.Timedelta(hours=freq_hours)
    if now - current_boundary < pd.Timedelta(minutes=grace_minutes):
        expected -= pd.Timedelta(hours=freq_hours)
    return expected


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


def check_crypto_latest_freshness(
    close: pd.DataFrame,
    *,
    now_ts: pd.Timestamp | datetime | str | None = None,
    freq_hours: int = 4,
    grace_minutes: int = 6,
) -> list[CheckResult]:
    if close.empty:
        return [_fail("crypto_latest_bar_freshness", "Crypto close matrix is empty.", value="empty")]
    latest = pd.Timestamp(close.index.max())
    now = _utc_naive_timestamp(now_ts)
    expected = _expected_completed_bar_start(now, freq_hours=freq_hours, grace_minutes=grace_minutes)
    stale_hours = (now - latest).total_seconds() / 3600
    metadata = {
        "latest_bar": str(latest),
        "expected_bar": str(expected),
        "stale_hours": round(stale_hours, 3),
        "freq_hours": freq_hours,
        "grace_minutes": grace_minutes,
    }
    if latest < expected:
        return [_fail(
            "crypto_latest_bar_freshness",
            f"Latest crypto bar is {latest}; expected at least {expected}.",
            value=str(latest),
            threshold=str(expected),
            metadata=metadata,
        )]
    return [_ok(
        "crypto_latest_bar_freshness",
        "Latest crypto bar is at or after the expected closed 4h bar.",
        value=str(latest),
        threshold=str(expected),
        metadata=metadata,
    )]


def check_crypto_bar_index(
    close: pd.DataFrame,
    *,
    freq_hours: int = 4,
    recent_lookback_hours: int = 48,
) -> list[CheckResult]:
    if close.empty:
        return [_fail("crypto_bar_index_continuity", "Crypto close matrix is empty.", value="empty")]

    idx = pd.DatetimeIndex(close.index)
    duplicate_count = int(idx.duplicated().sum())
    if duplicate_count:
        return [_fail(
            "crypto_bar_index_continuity",
            "Crypto close matrix contains duplicate bar timestamps.",
            value=f"duplicates={duplicate_count}",
            threshold="0",
        )]
    if not idx.is_monotonic_increasing:
        return [_fail(
            "crypto_bar_index_continuity",
            "Crypto close matrix index is not monotonic increasing.",
            value="not_monotonic",
            threshold="monotonic",
        )]

    latest = pd.Timestamp(idx.max())
    recent = idx[idx >= latest - pd.Timedelta(hours=recent_lookback_hours)]
    recent_gaps: list[str] = []
    if len(recent) >= 2:
        diffs = pd.Series(recent).diff().dropna()
        expected_delta = pd.Timedelta(hours=freq_hours)
        gap_locs = diffs[diffs > expected_delta]
        recent_gaps = [str(pd.Timestamp(recent[int(i)])) for i in gap_locs.index[:10]]

    if recent_gaps:
        return [_fail(
            "crypto_bar_index_continuity",
            f"{len(recent_gaps)} recent 4h bars are missing from the close matrix.",
            value=f"recent_gaps={len(recent_gaps)}",
            threshold="0 recent gaps",
            metadata={"first_gap_ends": recent_gaps, "lookback_hours": recent_lookback_hours},
        )]
    return [_ok(
        "crypto_bar_index_continuity",
        "Crypto close matrix has no duplicate bars and no recent 4h gaps.",
        value=f"duplicates=0; recent_gaps=0",
        threshold="0",
        metadata={"lookback_hours": recent_lookback_hours},
    )]


def check_crypto_latest_coverage(
    matrices_dir: Path,
    *,
    latest: pd.Timestamp,
    min_latest_coverage: float = 0.90,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> list[CheckResult]:
    coverage: dict[str, float] = {}
    missing_fields: list[str] = []
    for field_name in fields:
        path = matrices_dir / f"{field_name}.parquet"
        if not path.exists():
            missing_fields.append(field_name)
            continue
        mat = _load_matrix(matrices_dir, field_name)
        if latest not in mat.index or mat.shape[1] == 0:
            coverage[field_name] = 0.0
            continue
        row = mat.loc[latest]
        coverage[field_name] = float(row.notna().sum() / max(len(row), 1))

    if missing_fields:
        return [_fail(
            "crypto_latest_coverage",
            "Required crypto matrix fields are missing.",
            value=",".join(missing_fields),
            threshold="all required fields present",
            metadata={"missing_fields": missing_fields},
        )]

    min_field = min(coverage, key=coverage.get)
    min_value = coverage[min_field]
    metadata = {"coverage_by_field": {k: round(v, 4) for k, v in coverage.items()}, "latest_bar": str(latest)}
    if min_value < min_latest_coverage:
        return [_fail(
            "crypto_latest_coverage",
            "Latest crypto bar has too many missing OHLCV values.",
            value=f"{min_field}={min_value:.1%}",
            threshold=f">={min_latest_coverage:.0%}",
            metadata=metadata,
        )]
    return [_ok(
        "crypto_latest_coverage",
        "Latest crypto bar has sufficient OHLCV coverage.",
        value=f"{min_field}={min_value:.1%}",
        threshold=f">={min_latest_coverage:.0%}",
        metadata=metadata,
    )]


def check_crypto_universe(
    universe_path: Path,
    *,
    close: pd.DataFrame,
    expected_members: int | None = 100,
    min_member_fraction: float = 0.90,
) -> list[CheckResult]:
    if not universe_path.exists():
        return [_fail("crypto_universe_exists", "Crypto universe parquet is missing.", value=str(universe_path))]

    results: list[CheckResult] = []
    df = pd.read_parquet(universe_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    if df.empty:
        return [_fail("crypto_universe_exists", "Crypto universe parquet is empty.", value=str(universe_path))]

    results.append(_ok("crypto_universe_exists", "Crypto universe parquet exists.", value=str(df.shape)))

    close_latest = pd.Timestamp(close.index.max())
    universe_latest = pd.Timestamp(df.index.max())
    metadata = {"close_latest": str(close_latest), "universe_latest": str(universe_latest)}
    if universe_latest < close_latest:
        results.append(_fail(
            "crypto_universe_current",
            "Crypto universe is stale relative to the production close matrix.",
            value=str(universe_latest),
            threshold=str(close_latest),
            metadata=metadata,
        ))
    else:
        results.append(_ok(
            "crypto_universe_current",
            "Crypto universe is aligned with the production close matrix.",
            value=str(universe_latest),
            threshold=str(close_latest),
            metadata=metadata,
        ))

    last_members = df.loc[universe_latest].fillna(False).astype(bool)
    member_count = int(last_members.sum())
    unknown_members = sorted(set(last_members[last_members].index) - set(close.columns))
    min_members = int((expected_members or 1) * min_member_fraction) if expected_members else 1
    member_metadata = {
        "expected_members": expected_members,
        "min_member_fraction": min_member_fraction,
        "unknown_members": unknown_members[:10],
    }
    if unknown_members:
        results.append(_fail(
            "crypto_universe_membership",
            "Crypto universe contains symbols missing from the close matrix.",
            value=f"unknown_members={len(unknown_members)}",
            threshold="0",
            metadata=member_metadata,
        ))
    elif member_count < min_members:
        results.append(_fail(
            "crypto_universe_membership",
            "Crypto universe has too few active members on the latest bar.",
            value=str(member_count),
            threshold=f">={min_members}",
            metadata=member_metadata,
        ))
    else:
        results.append(_ok(
            "crypto_universe_membership",
            "Crypto universe has active members aligned with the close matrix.",
            value=str(member_count),
            threshold=f">={min_members}",
            metadata=member_metadata,
        ))
    return results


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
                         db_name: str = "alpha_results.db",
                         expected_universe_size: int | None = 100,
                         now_ts: pd.Timestamp | datetime | str | None = None,
                         freq_hours: int = 4) -> list[CheckResult]:
    matrices_dir = root / matrices_rel
    results: list[CheckResult] = []
    close = _load_matrix(matrices_dir, "close")
    latest = pd.Timestamp(close.index.max())
    results.extend(check_crypto_latest_freshness(close, now_ts=now_ts, freq_hours=freq_hours))
    results.extend(check_crypto_bar_index(close, freq_hours=freq_hours))
    results.extend(check_crypto_latest_coverage(matrices_dir, latest=latest))
    results.extend(check_ohlc(matrices_dir))
    results.extend(check_crypto_universe(
        root / universe_rel,
        close=close,
        expected_members=expected_universe_size,
    ))
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
