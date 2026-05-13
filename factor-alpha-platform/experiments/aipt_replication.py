"""
AIPT random-feature SDF replication and execution-cost extension.

Replicates the empirical core of Didisheim, Ke, Kelly, and Malamud,
"APT or AIPT? The Surprising Dominance of Large Factor Models", using the
platform's equity and KuCoin matrices:

  1. Cross-sectionally rank-standardize point-in-time characteristics.
  2. Generate random Fourier / tanh / ReLU features.
  3. Convert each random feature into a characteristic-managed portfolio.
  4. Estimate the SDF as the rolling ridge tangency portfolio of those factors.
  5. Convert the SDF weight function back to tradable asset weights and apply
     realized execution costs.

Lookahead convention
--------------------
For a signal date t, the realized factor return row F[t] uses the return after
date t. Therefore the rolling fit at t uses only rows strictly known by t:

  delay=0: train rows < t
  delay=1: train rows < t-1

The datasource projection option also uses only the initial training window.
No validation/test metric is used for feature/source selection.

Usage examples:
  python experiments/aipt_replication.py --preset quick
  python experiments/aipt_replication.py --preset extended
  python experiments/aipt_replication.py --scenario equity_smallcap_d0 --p-grid 64,256,1024
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.fees import cost_bps_taker, cost_per_share_ib


GAMMA_GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
_MARKET_DATA_CACHE: dict[tuple[str, tuple[str, ...], bool], tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, list[str]]] = {}

EQUITY_MATRICES_DIR = (
    "data/fmp_cache/matrices_pit_v2"
    if (ROOT / "data/fmp_cache/matrices_pit_v2").exists()
    else "data/fmp_cache/matrices_pit"
    if (ROOT / "data/fmp_cache/matrices_pit").exists()
    else "data/fmp_cache/matrices"
)
EQUITY_SMALLCAP_UNIVERSE = (
    "experiments/data/aipt_universes/MCAP_100M_500M_PITV2.parquet"
    if (ROOT / "experiments/data/aipt_universes/MCAP_100M_500M_PITV2.parquet").exists()
    else "data/fmp_cache/universes/MCAP_100M_500M.parquet"
)
EQUITY_TOP1000_UNIVERSE = (
    "experiments/data/aipt_universes/TOP1000_ADV60_PITV2.parquet"
    if (ROOT / "experiments/data/aipt_universes/TOP1000_ADV60_PITV2.parquet").exists()
    else "data/fmp_cache/universes/TOP1000.parquet"
)
EQUITY_TOP3000_UNIVERSE = (
    "experiments/data/aipt_universes/TOP3000_ADV60_PITV2.parquet"
    if (ROOT / "experiments/data/aipt_universes/TOP3000_ADV60_PITV2.parquet").exists()
    else "data/fmp_cache/universes/TOP3000.parquet"
)


EQUITY_PRICE_LIQUIDITY = [
    "close",
    "open",
    "high",
    "low",
    "vwap",
    "volume",
    "dollars_traded",
    "adv20",
    "adv60",
    "returns",
    "high_low_range",
    "open_close_range",
    "close_position_in_range",
    "vwap_deviation",
    "turnover",
    "volume_ratio_20d",
    "volume_momentum_1",
    "volume_momentum_5_20",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "historical_volatility_20",
    "historical_volatility_60",
    "parkinson_volatility_20",
    "parkinson_volatility_60",
]

EQUITY_VALUE_QUALITY = [
    "market_cap",
    "book_to_market",
    "pb_ratio",
    "earnings_yield",
    "earnings_yield_metric",
    "free_cashflow_yield",
    "fcf_yield_metric",
    "ev_to_ebitda",
    "ev_to_revenue",
    "ev_to_sales",
    "roe",
    "roa",
    "gross_profit",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "asset_turnover",
    "total_assets",
    "assets",
    "current_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "capex",
    "rd_expense",
]

EQUITY_FUNDAMENTAL_FLOWS = [
    "revenue",
    "net_income",
    "operating_income",
    "ebit",
    "ebitda",
    "gross_profit",
    "free_cashflow",
    "operating_cashflow",
    "interest_expense",
    "dividends_paid",
    "depreciation_amortization",
]

EQUITY_FUNDAMENTAL_PERSHARE = [
    "eps",
    "eps_diluted",
    "fcf_per_share",
    "bookvalue_ps",
    "tangible_book_per_share",
    "shares_out",
    "shares_out_diluted",
]

EQUITY_FUNDAMENTAL_BALANCE = [
    "cash",
    "inventory",
    "goodwill",
    "intangibles",
    "total_assets",
    "total_current_assets",
    "total_current_liabilities",
    "total_liabilities",
    "total_equity",
    "total_stockholders_equity",
    "total_debt",
    "long_term_debt",
    "short_term_debt",
    "net_debt",
    "enterprise_value",
]

EQUITY_FUNDAMENTAL_RATIOS = [
    "market_cap",
    "cap",
    "book_to_market",
    "earnings_yield",
    "free_cashflow_yield",
    "ev_to_ebitda",
    "ev_to_revenue",
    "roe",
    "roa",
    "return_equity",
    "return_assets",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "asset_turnover",
    "current_ratio",
    "debt_to_equity",
    "debt_to_assets",
    "capex",
]

# Fundamental_full: all fundamental-class fields we actually have in PIT v2.
# This is the expanded source set for AIPT runs targeting fundamental signal.
EQUITY_FUNDAMENTAL_FULL = list(
    dict.fromkeys(
        EQUITY_FUNDAMENTAL_RATIOS
        + EQUITY_FUNDAMENTAL_FLOWS
        + EQUITY_FUNDAMENTAL_PERSHARE
        + EQUITY_FUNDAMENTAL_BALANCE
    )
)

EQUITY_SOURCE_SETS = {
    "price": EQUITY_PRICE_LIQUIDITY,
    "fundamental": EQUITY_VALUE_QUALITY,
    "fundamental_full": EQUITY_FUNDAMENTAL_FULL,
    "all": list(dict.fromkeys(EQUITY_PRICE_LIQUIDITY + EQUITY_VALUE_QUALITY)),
    "all_full": list(dict.fromkeys(EQUITY_PRICE_LIQUIDITY + EQUITY_FUNDAMENTAL_FULL)),
}


CRYPTO_PRICE = [
    "close",
    "open",
    "high",
    "low",
    "vwap",
    "returns",
    "high_low_range",
    "open_close_range",
    "close_position_in_range",
    "upper_shadow",
    "lower_shadow",
    "historical_volatility_20",
    "historical_volatility_60",
    "parkinson_volatility_20",
    "parkinson_volatility_60",
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
]

CRYPTO_LIQUIDITY = [
    "volume",
    "quote_volume",
    "dollars_traded",
    "turnover",
    "adv20",
    "adv60",
    "volume_ratio_20d",
    "volume_momentum_1",
    "volume_momentum_5_20",
]

CRYPTO_BTC = ["beta_to_btc", "overnight_gap", "vwap_deviation"]

CRYPTO_SOURCE_SETS = {
    "price": CRYPTO_PRICE,
    "liquidity": CRYPTO_LIQUIDITY,
    "btc": CRYPTO_BTC,
    "all": list(dict.fromkeys(CRYPTO_PRICE + CRYPTO_LIQUIDITY + CRYPTO_BTC)),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    market: str
    matrices_dir: str
    universe_path: str
    source_sets: dict[str, list[str]]
    default_source_set: str
    train_window: int
    rebalance_every: int
    bars_per_year: int
    delay: int
    max_names: int
    selection_field: str
    book: float
    max_weight: float
    split_train_end: str
    split_val_end: str
    fee_model: str
    fee_params: dict[str, float]
    min_names: int = 20
    start: str | None = None
    end: str | None = None


SCENARIOS = {
    "equity_smallcap_d0": Scenario(
        name="equity_smallcap_d0",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_SMALLCAP_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=0,
        max_names=650,
        selection_field="adv60",
        book=500_000.0,
        max_weight=0.02,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.5,
            "borrow_bps_annual": 50.0,
        },
        start="2018-01-01",
    ),
    "equity_smallcap_d1": Scenario(
        name="equity_smallcap_d1",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_SMALLCAP_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=650,
        selection_field="adv60",
        book=500_000.0,
        max_weight=0.02,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.5,
            "borrow_bps_annual": 50.0,
        },
        start="2018-01-01",
    ),
    "equity_top1000_d0": Scenario(
        name="equity_top1000_d0",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_TOP1000_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=0,
        max_names=800,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.01,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_top1000_d1": Scenario(
        name="equity_top1000_d1",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_TOP1000_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=800,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.01,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_top3000_d0": Scenario(
        name="equity_top3000_d0",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_TOP3000_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=0,
        max_names=3000,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.005,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_top3000_d1": Scenario(
        name="equity_top3000_d1",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path=EQUITY_TOP3000_UNIVERSE,
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="all",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=3000,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.005,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_top1000_d1_rebal20d": Scenario(
        name="equity_top1000_d1_rebal20d",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path="experiments/data/aipt_universes/TOP1000_REBAL20D.parquet",
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="fundamental_full",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=1000,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.01,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_top3000_d1_rebal20d": Scenario(
        name="equity_top3000_d1_rebal20d",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path="experiments/data/aipt_universes/TOP3000_REBAL20D.parquet",
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="fundamental_full",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=3000,
        selection_field="adv60",
        book=2_000_000.0,
        max_weight=0.005,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.35,
            "borrow_bps_annual": 35.0,
        },
        start="2018-01-01",
    ),
    "equity_midcap_d1_rebal20d": Scenario(
        name="equity_midcap_d1_rebal20d",
        market="equity",
        matrices_dir=EQUITY_MATRICES_DIR,
        universe_path="experiments/data/aipt_universes/MIDCAP_500M_5B_REBAL20D.parquet",
        source_sets=EQUITY_SOURCE_SETS,
        default_source_set="fundamental_full",
        train_window=252,
        rebalance_every=5,
        bars_per_year=252,
        delay=1,
        max_names=1737,
        selection_field="adv60",
        book=1_000_000.0,
        max_weight=0.015,
        split_train_end="2024-01-01",
        split_val_end="2025-04-01",
        fee_model="per_share_ib",
        fee_params={
            "commission_per_share": 0.0045,
            "per_order_min": 0.35,
            "sec_fee_per_dollar": 27.80e-6,
            "sell_fraction": 0.50,
            "impact_bps": 0.40,
            "borrow_bps_annual": 40.0,
        },
        start="2018-01-01",
    ),
    "kucoin_top100": Scenario(
        name="kucoin_top100",
        market="crypto",
        matrices_dir="data/kucoin_cache/matrices/4h",
        universe_path="data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet",
        source_sets=CRYPTO_SOURCE_SETS,
        default_source_set="all",
        train_window=360,
        rebalance_every=6,
        bars_per_year=2190,
        delay=0,
        max_names=90,
        selection_field="adv60",
        book=100_000.0,
        max_weight=0.10,
        split_train_end="2026-01-01",
        split_val_end="2026-03-01",
        fee_model="bps_taker",
        fee_params={"taker_bps": 2.5, "slippage_bps": 1.0},
        min_names=15,
        start="2025-09-01",
    ),
}


@dataclass(frozen=True)
class ExperimentSpec:
    scenario: str
    source_set: str
    n_features: int
    ridge_z: float
    activation: str
    seed: int
    cost_tau: float
    projected_sources: bool
    project_top_k: int


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True).sub(0.5)


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    read_columns = columns
    if columns is not None:
        available = set(pq.ParquetFile(path).schema.names)
        read_columns = [c for c in columns if c in available]
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            df = pd.read_parquet(path, columns=read_columns)
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(0.25 * (attempt + 1))
    else:
        raise RuntimeError(f"failed reading parquet {path}") from last_exc
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
    return df.sort_index()


def _slice_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def _select_initial_names(scenario: Scenario, root: Path, uni: pd.DataFrame) -> list[str]:
    """Freeze the tradable column set from information available at the first fit date."""
    field_path = root / scenario.matrices_dir / f"{scenario.selection_field}.parquet"
    selector = _read_parquet(field_path)
    selector = _slice_dates(selector, scenario.start, scenario.end)
    selector = selector.reindex(index=uni.index, columns=uni.columns)
    counts = uni.sum(axis=1)
    candidates = np.flatnonzero((np.arange(len(uni.index)) >= scenario.train_window) & (counts.values >= scenario.min_names))
    fit_pos = int(candidates[0]) if len(candidates) else min(max(scenario.train_window, 1), len(uni.index) - 1)
    sel_dt = uni.index[fit_pos]
    active = uni.loc[sel_dt].astype(bool)
    row = selector.loc[sel_dt].where(active)
    row = row.replace([np.inf, -np.inf], np.nan).dropna()
    if row.empty:
        # Fallback is still point-in-time: active names on the initial fit date.
        return active[active].index[: scenario.max_names].tolist()
    return row.sort_values(ascending=False).head(scenario.max_names).index.tolist()


def load_market_data(
    scenario: Scenario,
    fields: list[str],
    *,
    root: Path = ROOT,
    dynamic_universe: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, list[str]]:
    cache_key = (scenario.name, tuple(fields), bool(dynamic_universe))
    if cache_key in _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE[cache_key]

    uni = _read_parquet(root / scenario.universe_path)
    if uni.dtypes.iloc[0] != bool:
        uni = uni.astype(bool)
    uni = _slice_dates(uni, scenario.start, scenario.end)
    if dynamic_universe:
        tickers = uni.columns.tolist()
        if scenario.max_names > 0 and scenario.selection_field:
            selector_path = root / scenario.matrices_dir / f"{scenario.selection_field}.parquet"
            selector = _read_parquet(selector_path, columns=tickers)
            selector = _slice_dates(selector, scenario.start, scenario.end)
            selector = selector.reindex(index=uni.index, columns=tickers)
            ranks = selector.where(uni).rank(axis=1, ascending=False, method="first")
            uni = (uni & ranks.le(scenario.max_names)).fillna(False).astype(bool)
        tickers = uni.columns[uni.any(axis=0)].tolist()
    else:
        tickers = _select_initial_names(scenario, root, uni)
    uni = uni.reindex(columns=tickers).fillna(False).astype(bool)

    required = sorted(set(fields + ["close", scenario.selection_field]))
    if scenario.delay >= 1:
        required.append("open")
    mats: dict[str, pd.DataFrame] = {}
    for field in sorted(set(required)):
        path = root / scenario.matrices_dir / f"{field}.parquet"
        if not path.exists():
            continue
        df = _read_parquet(path, columns=tickers)
        df = _slice_dates(df, scenario.start, scenario.end)
        mats[field] = df.reindex(index=uni.index, columns=tickers)

    available_fields = [f for f in fields if f in mats]
    if "close" not in mats:
        raise RuntimeError(f"{scenario.name}: close.parquet not loaded")
    if scenario.delay >= 1 and "open" not in mats:
        raise RuntimeError(f"{scenario.name}: delay=1 requires open.parquet")
    result = (uni, mats, mats["close"], available_fields)
    _MARKET_DATA_CACHE[cache_key] = result
    return result


def make_forward_returns(mats: dict[str, pd.DataFrame], delay: int) -> pd.DataFrame:
    close = mats["close"]
    if delay == 0:
        return close.pct_change(fill_method=None).shift(-1)
    open_ = mats["open"]
    return open_.pct_change(fill_method=None).shift(-(delay + 1))


def make_characteristic_tensor(
    uni: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    fields: list[str],
    *,
    max_missing_frac: float = 0.30,
    field_scales: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    dates, tickers = uni.index, uni.columns
    arr = np.full((len(dates), len(tickers), len(fields)), np.nan, dtype=np.float32)
    used: list[str] = []
    for j, field in enumerate(fields):
        if field not in mats:
            continue
        ranked = _rank_cs(mats[field].reindex(index=dates, columns=tickers).where(uni))
        scale = 1.0 if field_scales is None else float(field_scales.get(field, 1.0))
        arr[:, :, j] = (ranked.values * scale).astype(np.float32)
        used.append(field)
    if not used:
        raise RuntimeError("no usable fields for characteristic tensor")
    arr = arr[:, :, : len(used)]
    finite = np.isfinite(arr)
    min_present = max(1, int(math.ceil((1.0 - max_missing_frac) * len(used))))
    active = (finite.sum(axis=2) >= min_present) & uni.values.astype(bool)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, active, used


def make_random_params(
    n_inputs: int,
    n_features: int,
    activation: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if activation == "sincos":
        if n_features % 2 != 0:
            raise ValueError("sincos n_features must be even")
        n_cols = n_features // 2
    else:
        n_cols = n_features
    w = rng.standard_normal((n_inputs, n_cols)).astype(np.float64)
    gamma = rng.choice(GAMMA_GRID, size=n_cols).astype(np.float64)
    return w, gamma


def random_features_for_date(
    x_t: np.ndarray,
    active_t: np.ndarray,
    w: np.ndarray,
    gamma: np.ndarray,
    activation: str,
    *,
    demean_features: bool = True,
) -> np.ndarray:
    z = x_t.astype(np.float64, copy=False) @ w
    z *= gamma[None, :]
    if activation == "sincos":
        s = np.empty((x_t.shape[0], w.shape[1] * 2), dtype=np.float64)
        s[:, 0::2] = np.sin(z)
        s[:, 1::2] = np.cos(z)
    elif activation == "tanh":
        s = np.tanh(z)
    elif activation == "relu":
        s = np.maximum(z, 0.0)
    else:
        raise ValueError(f"unknown activation {activation!r}")

    s[~active_t, :] = 0.0
    if demean_features and active_t.sum() > 1:
        means = s[active_t, :].mean(axis=0, keepdims=True)
        s[active_t, :] -= means
    s[~np.isfinite(s)] = 0.0
    return s


def build_factor_returns(
    x: np.ndarray,
    active: np.ndarray,
    fwd_ret: pd.DataFrame,
    w: np.ndarray,
    gamma: np.ndarray,
    activation: str,
    *,
    min_names: int,
) -> np.ndarray:
    ret_np = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    n_dates = x.shape[0]
    n_features = w.shape[1] * 2 if activation == "sincos" else w.shape[1]
    out = np.zeros((n_dates, n_features), dtype=np.float64)
    for t in range(n_dates):
        n_active = int(active[t].sum())
        if n_active < min_names:
            continue
        s = random_features_for_date(x[t], active[t], w, gamma, activation)
        out[t] = (s.T @ ret_np[t]) / math.sqrt(n_active)
    return out


def _fit_lambda(
    f_train: np.ndarray,
    ridge_z: float,
    *,
    cost_diag: np.ndarray | None = None,
    carry: np.ndarray | None = None,
) -> np.ndarray:
    f_train = np.asarray(f_train, dtype=np.float64)
    f_train = np.nan_to_num(f_train, nan=0.0, posinf=0.0, neginf=0.0)
    n_obs, n_features = f_train.shape
    mu = f_train.mean(axis=0)
    if carry is not None:
        mu = mu + carry
    if cost_diag is not None:
        gram = (f_train.T @ f_train) / max(n_obs, 1)
        diag = ridge_z + np.maximum(cost_diag, 0.0)
        gram.flat[:: n_features + 1] += diag
        try:
            return np.linalg.solve(gram, mu)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(gram) @ mu

    if n_features > n_obs:
        # Dual ridge: (F'F/T + zI)^-1 F'1/T = A'(AA' + zI)^-1 y.
        a = f_train / math.sqrt(max(n_obs, 1))
        y = np.ones(n_obs, dtype=np.float64) / math.sqrt(max(n_obs, 1))
        k = a @ a.T
        k.flat[:: n_obs + 1] += ridge_z
        try:
            dual = np.linalg.solve(k, y)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(k) @ y
        return a.T @ dual

    gram = (f_train.T @ f_train) / max(n_obs, 1)
    gram.flat[:: n_features + 1] += ridge_z
    try:
        return np.linalg.solve(gram, mu)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ mu


def _normalise_portfolio(raw: np.ndarray, active_t: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.zeros_like(raw, dtype=np.float64)
    mask = active_t & np.isfinite(raw)
    if mask.sum() < 2:
        return w
    vals = raw[mask].astype(np.float64, copy=True)
    vals -= vals.mean()
    gross = np.abs(vals).sum()
    if gross <= 1e-12:
        return w
    vals /= gross
    vals = np.clip(vals, -max_weight, max_weight)
    gross = np.abs(vals).sum()
    if gross > 1e-12:
        vals /= gross
    vals = np.clip(vals, -max_weight, max_weight)
    w[mask] = vals
    return w


def _cost_kernel_terms(
    s_t: np.ndarray,
    active_t: np.ndarray,
    w_prev: np.ndarray,
    cost_vec: np.ndarray,
    cost_tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cost_tau <= 0 or active_t.sum() < 2:
        n_features = s_t.shape[1]
        return np.zeros(n_features), np.zeros(n_features)
    n_active = float(active_t.sum())
    c = np.nan_to_num(cost_vec, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    s = s_t[active_t]
    c_active = c[active_t]
    diag = cost_tau * ((s * s) * c_active[:, None]).sum(axis=0) / n_active
    carry = cost_tau * (s.T @ (c_active * w_prev[active_t])) / math.sqrt(n_active)
    return diag, carry


def _cost_vector(scenario: Scenario, close_row: pd.Series, adv_row: pd.Series | None) -> np.ndarray:
    price = close_row.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    if scenario.fee_model == "bps_taker":
        base = (scenario.fee_params["taker_bps"] + scenario.fee_params["slippage_bps"]) / 1e4
    else:
        base = scenario.fee_params.get("impact_bps", 0.5) / 1e4
        base += scenario.fee_params.get("commission_per_share", 0.0045) / np.maximum(price, 0.01)
    if adv_row is not None:
        adv = adv_row.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
        participation = (scenario.book * scenario.max_weight) / np.maximum(adv, 1.0)
        base = base + 0.5 / 1e4 * np.sqrt(np.maximum(participation, 0.0))
    return np.full_like(price, base, dtype=np.float64) if np.isscalar(base) else base


def run_rolling_sdf(
    scenario: Scenario,
    spec: ExperimentSpec,
    x: np.ndarray,
    active: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: list[str],
    close: pd.DataFrame,
    adv: pd.DataFrame | None,
    fwd_ret: pd.DataFrame,
    factor_returns: np.ndarray,
    random_w: np.ndarray,
    gamma: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    sdf_ret = pd.Series(0.0, index=dates, dtype=float)
    w_prev = np.zeros(len(tickers), dtype=np.float64)
    lambda_prev = np.zeros(factor_returns.shape[1], dtype=np.float64)

    last_fit = -10**9
    for t in range(len(dates)):
        train_end = t - scenario.delay
        train_start = train_end - scenario.train_window
        if train_start < 0 or train_end <= train_start:
            weights.iloc[t] = w_prev
            continue

        refit = (t - last_fit) >= scenario.rebalance_every
        if refit:
            f_train = factor_returns[train_start:train_end]
            s_t = random_features_for_date(x[t], active[t], random_w, gamma, spec.activation)
            adv_row = adv.iloc[t] if adv is not None else None
            c_vec = _cost_vector(scenario, close.iloc[t], adv_row)
            c_diag, carry = _cost_kernel_terms(s_t, active[t], w_prev, c_vec, spec.cost_tau)
            lambda_prev = _fit_lambda(
                f_train,
                spec.ridge_z,
                cost_diag=c_diag if spec.cost_tau > 0 else None,
                carry=carry if spec.cost_tau > 0 else None,
            )
            n_active = max(int(active[t].sum()), 1)
            raw = (s_t @ lambda_prev) / math.sqrt(n_active)
            w_prev = _normalise_portfolio(raw, active[t], scenario.max_weight)
            last_fit = t

        weights.iloc[t] = w_prev
        sdf_ret.iloc[t] = float(np.nan_to_num(factor_returns[t]) @ lambda_prev)

    return weights, sdf_ret, pd.Series(lambda_prev, dtype=float)


def _metrics_one(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"n_bars": 0, "SR": float("nan"), "ret_ann": float("nan"), "max_dd": float("nan")}
    sd = x.std()
    sr = (x.mean() / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan")
    ret_ann = x.mean() * bars_per_year
    eq = (1.0 + x).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return {"n_bars": int(len(x)), "SR": float(sr), "ret_ann": float(ret_ann), "max_dd": float(dd)}


def split_metrics(
    gross: pd.Series,
    net: pd.Series,
    weights: pd.DataFrame,
    *,
    scenario: Scenario,
) -> dict[str, dict[str, float]]:
    splits = {
        "TRAIN": slice(None, scenario.split_train_end),
        "VAL": slice(scenario.split_train_end, scenario.split_val_end),
        "TEST": slice(scenario.split_val_end, None),
        "VAL+TEST": slice(scenario.split_train_end, None),
        "FULL": slice(None, None),
    }
    out: dict[str, dict[str, float]] = {}
    turnover = weights.diff().abs().sum(axis=1)
    for label, sl in splits.items():
        gm = _metrics_one(gross.loc[sl], scenario.bars_per_year)
        nm = _metrics_one(net.loc[sl], scenario.bars_per_year)
        out[label] = {
            "n_bars": nm["n_bars"],
            "SR_gross": gm["SR"],
            "SR_net": nm["SR"],
            "ret_ann_gross": gm["ret_ann"],
            "ret_ann_net": nm["ret_ann"],
            "max_dd_net": nm["max_dd"],
            "turnover_per_bar": float(turnover.loc[sl].mean()),
        }
    return out


def compute_hjd(sdf_ret: pd.Series, factor_returns: np.ndarray, dates: pd.DatetimeIndex, start: str) -> float:
    p = factor_returns.shape[1]
    if p > 512:
        return float("nan")
    mask = dates >= pd.Timestamp(start)
    f = np.nan_to_num(factor_returns[mask], nan=0.0, posinf=0.0, neginf=0.0)
    if f.shape[0] < 10:
        return float("nan")
    m = 1.0 - sdf_ret.loc[mask].values.astype(np.float64)
    e = (m[:, None] * f).mean(axis=0)
    second = (f.T @ f) / max(f.shape[0], 1)
    try:
        inv = np.linalg.pinv(second)
        return float(e @ inv @ e)
    except np.linalg.LinAlgError:
        return float("nan")


def realised_costs(scenario: Scenario, weights: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    if scenario.fee_model == "bps_taker":
        return cost_bps_taker(weights, **scenario.fee_params)
    if scenario.fee_model == "per_share_ib":
        return cost_per_share_ib(
            weights,
            close,
            scenario.book,
            bars_per_year=scenario.bars_per_year,
            **scenario.fee_params,
        )
    raise ValueError(f"unknown fee_model {scenario.fee_model!r}")


def project_datasources(
    scenario: Scenario,
    uni: pd.DataFrame,
    mats: dict[str, pd.DataFrame],
    fields: list[str],
    fwd_ret: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Train-only feature/source projection via no-neutrality univariate managed-portfolio SR."""
    scores: dict[str, float] = {}
    scales: dict[str, float] = {}
    window_end = min(scenario.train_window, len(uni.index) - scenario.delay - 2)
    if window_end < 30:
        return fields, {f: 1.0 for f in fields}, scores
    ret_np = fwd_ret.iloc[:window_end].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    active = uni.iloc[:window_end].values.astype(bool)
    adv = mats.get(scenario.selection_field)
    for field in fields:
        ranked = _rank_cs(mats[field].iloc[:window_end].where(uni.iloc[:window_end]))
        vals = ranked.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        pnl = []
        for t in range(window_end):
            mask = active[t]
            if mask.sum() < scenario.min_names:
                pnl.append(0.0)
                continue
            w = vals[t].copy()
            w[~mask] = 0.0
            gross = np.abs(w[mask]).sum()
            if gross > 1e-12:
                w /= gross
            pnl.append(float(np.dot(w, ret_np.iloc[t].values)))
        ser = pd.Series(pnl)
        sd = ser.std()
        sr = ser.mean() / sd * math.sqrt(scenario.bars_per_year) if sd > 0 else 0.0
        coverage = float(np.isfinite(mats[field].iloc[:window_end].where(uni.iloc[:window_end]).values).mean())
        cost_penalty = 1.0
        if adv is not None:
            avg_adv = adv.iloc[:window_end].where(uni.iloc[:window_end]).median(axis=1).median()
            if np.isfinite(avg_adv) and avg_adv > 0:
                cost_penalty = 1.0 + math.sqrt((scenario.book * scenario.max_weight) / avg_adv)
        score = max(sr, 0.0) * math.sqrt(max(coverage, 1e-6)) / cost_penalty
        scores[field] = float(score)

    selected = [k for k, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v > 0]
    if top_k > 0:
        selected = selected[:top_k]
    if len(selected) < min(4, len(fields)):
        selected = fields[: min(len(fields), max(4, top_k))]
    max_score = max([scores.get(f, 0.0) for f in selected] + [1e-12])
    for f in selected:
        scales[f] = math.sqrt(0.25 + 0.75 * scores.get(f, 0.0) / max_score)
    return selected, scales, scores


def run_one(spec: ExperimentSpec, *, out_dir: Path, root: Path = ROOT) -> dict:
    scenario = SCENARIOS[spec.scenario]
    fields = scenario.source_sets[spec.source_set]
    print(
        f"[run] {spec.scenario} source={spec.source_set}"
        f"{'_projected' if spec.projected_sources else ''} P={spec.n_features}"
        f" z={spec.ridge_z:g} act={spec.activation} seed={spec.seed} cost_tau={spec.cost_tau:g}",
        flush=True,
    )
    t0 = time.time()
    uni, mats, close, available_fields = load_market_data(scenario, fields, root=root)
    fwd_ret = make_forward_returns(mats, scenario.delay)

    selected_fields = available_fields
    field_scales = {f: 1.0 for f in selected_fields}
    projection_scores: dict[str, float] = {}
    if spec.projected_sources:
        selected_fields, field_scales, projection_scores = project_datasources(
            scenario,
            uni,
            mats,
            available_fields,
            fwd_ret,
            top_k=spec.project_top_k,
        )

    x, active, used_fields = make_characteristic_tensor(
        uni,
        mats,
        selected_fields,
        field_scales=field_scales,
    )
    dates = uni.index
    tickers = uni.columns.tolist()
    random_w, gamma = make_random_params(len(used_fields), spec.n_features, spec.activation, spec.seed)
    factor_returns = build_factor_returns(
        x,
        active,
        fwd_ret,
        random_w,
        gamma,
        spec.activation,
        min_names=scenario.min_names,
    )
    adv = mats.get(scenario.selection_field)
    weights, sdf_ret, _lambda_last = run_rolling_sdf(
        scenario,
        spec,
        x,
        active,
        dates,
        tickers,
        close,
        adv,
        fwd_ret,
        factor_returns,
        random_w,
        gamma,
    )

    gross = (weights * fwd_ret.reindex(index=dates, columns=tickers).fillna(0.0)).sum(axis=1)
    cost = realised_costs(scenario, weights, close.reindex(index=dates, columns=tickers))
    net = gross - cost
    metrics = split_metrics(gross, net, weights, scenario=scenario)
    sdf_metrics = _metrics_one(sdf_ret.loc[sdf_ret.index >= dates[min(scenario.train_window, len(dates) - 1)]], scenario.bars_per_year)
    hjd = compute_hjd(sdf_ret, factor_returns, dates, scenario.split_val_end)
    elapsed = time.time() - t0

    result = {
        "spec": asdict(spec),
        "scenario": asdict(scenario),
        "n_dates": len(dates),
        "n_names": len(tickers),
        "n_fields": len(used_fields),
        "used_fields": used_fields,
        "field_scales": field_scales,
        "projection_scores": projection_scores,
        "metrics": metrics,
        "sdf_raw": {
            "SR": sdf_metrics["SR"],
            "ret_ann": sdf_metrics["ret_ann"],
            "hjd_test": hjd,
        },
        "mean_cost_per_bar": float(cost.mean()),
        "elapsed_sec": elapsed,
        "lookahead_audit": {
            "signal_index": "features at t",
            "delay": scenario.delay,
            "fit_uses_factor_rows": f"< t-{scenario.delay}",
            "fwd_return": "close[t+1]/close[t]-1 for delay=0; open[t+2]/open[t+1]-1 for delay=1",
            "source_projection": "initial training window only" if spec.projected_sources else "fixed ex ante source set",
            "no_global_scalers": True,
            "no_bfill": True,
        },
    }

    tag = (
        f"{spec.scenario}__{spec.source_set}"
        f"{'__proj' if spec.projected_sources else ''}"
        f"__P{spec.n_features}__z{spec.ridge_z:g}__{spec.activation}"
        f"__seed{spec.seed}__ct{spec.cost_tau:g}"
    ).replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    weights.tail(500).to_parquet(out_dir / f"{tag}.weights_tail.parquet")
    print(
        f"      VAL SR_n={metrics['VAL']['SR_net']:+.2f} "
        f"TEST SR_n={metrics['TEST']['SR_net']:+.2f} "
        f"V+T SR_n={metrics['VAL+TEST']['SR_net']:+.2f} "
        f"rawSDF={sdf_metrics['SR']:+.2f} "
        f"to={metrics['FULL']['turnover_per_bar']*100:.2f}% "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return result


def specs_for_preset(args: argparse.Namespace) -> list[ExperimentSpec]:
    scenarios = args.scenarios or []
    if args.scenario:
        scenarios.append(args.scenario)
    if not scenarios:
        if args.preset == "quick":
            scenarios = ["equity_smallcap_d0", "equity_smallcap_d1", "kucoin_top100"]
        elif args.preset == "extended":
            scenarios = [
                "equity_smallcap_d0",
                "equity_smallcap_d1",
                "equity_top1000_d0",
                "equity_top1000_d1",
                "kucoin_top100",
            ]
        else:
            scenarios = ["equity_smallcap_d0", "equity_smallcap_d1", "kucoin_top100"]

    p_grid = _parse_csv_ints(args.p_grid)
    z_grid = _parse_csv_floats(args.z_grid)
    activations = [x.strip() for x in args.activations.split(",") if x.strip()]
    seeds = _parse_csv_ints(args.seeds)
    cost_taus = _parse_csv_floats(args.cost_taus)

    specs: list[ExperimentSpec] = []
    for sc_name in scenarios:
        scenario = SCENARIOS[sc_name]
        source_sets = [x.strip() for x in args.source_sets.split(",") if x.strip()]
        if not source_sets:
            source_sets = [scenario.default_source_set]
        for source_set in source_sets:
            if source_set == "default":
                source_set = scenario.default_source_set
            if source_set not in scenario.source_sets:
                continue
            for p in p_grid:
                for z in z_grid:
                    for act in activations:
                        if act == "sincos" and p % 2 != 0:
                            continue
                        for seed in seeds:
                            for cost_tau in cost_taus:
                                specs.append(
                                    ExperimentSpec(
                                        scenario=sc_name,
                                        source_set=source_set,
                                        n_features=p,
                                        ridge_z=z,
                                        activation=act,
                                        seed=seed,
                                        cost_tau=cost_tau,
                                        projected_sources=False,
                                        project_top_k=args.project_top_k,
                                    )
                                )
                                if args.include_projected and source_set == scenario.default_source_set:
                                    specs.append(
                                        ExperimentSpec(
                                            scenario=sc_name,
                                            source_set=source_set,
                                            n_features=p,
                                            ridge_z=z,
                                            activation=act,
                                            seed=seed,
                                            cost_tau=cost_tau,
                                            projected_sources=True,
                                            project_top_k=args.project_top_k,
                                        )
                                    )
    if args.limit:
        specs = specs[: args.limit]
    return specs


def write_summary(results: list[dict], out_dir: Path) -> Path:
    summary_path = out_dir / "aipt_summary.csv"
    rows = []
    for r in results:
        spec = r["spec"]
        base = {
            "scenario": spec["scenario"],
            "source_set": spec["source_set"],
            "projected": spec["projected_sources"],
            "P": spec["n_features"],
            "z": spec["ridge_z"],
            "activation": spec["activation"],
            "seed": spec["seed"],
            "cost_tau": spec["cost_tau"],
            "n_names": r["n_names"],
            "n_fields": r["n_fields"],
            "elapsed_sec": r["elapsed_sec"],
            "sdf_raw_SR": r["sdf_raw"]["SR"],
            "hjd_test": r["sdf_raw"]["hjd_test"],
        }
        for split in ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]:
            m = r["metrics"][split]
            row = dict(base)
            row["split"] = split
            row.update(m)
            rows.append(row)
    if rows:
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "aipt_summary.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return summary_path


def write_run_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_replication.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "mode": "tradable_constrained_cost_diagnostic",
        "notes": "Diagnostic runner that converts the SDF to asset weights with normalization/caps/costs.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "extended"], default="quick")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--source-sets", default="default")
    parser.add_argument("--p-grid", default="64,256,1024")
    parser.add_argument("--z-grid", default="0.001,0.01,0.1")
    parser.add_argument("--activations", default="sincos")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--cost-taus", default="0,25")
    parser.add_argument("--include-projected", action="store_true")
    parser.add_argument("--project-top-k", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/results/aipt_replication")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = ROOT / args.out_dir
    write_run_manifest(args, out_dir)
    specs = specs_for_preset(args)
    print(f"[setup] running {len(specs)} experiment cells -> {out_dir.relative_to(ROOT)}", flush=True)
    results = []
    failures = []
    for spec in specs:
        try:
            results.append(run_one(spec, out_dir=out_dir))
            write_summary(results, out_dir)
        except Exception as exc:
            failures.append({"spec": asdict(spec), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {spec}: {type(exc).__name__}: {exc}", flush=True)
            (out_dir / "aipt_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary_path = write_summary(results, out_dir)
    print(f"[done] results={len(results)} failures={len(failures)} summary={summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
