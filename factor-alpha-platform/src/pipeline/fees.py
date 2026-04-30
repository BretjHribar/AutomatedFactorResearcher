"""
Realized cost models for the unified pipeline.

Two interchangeable cost functions, both returning a per-bar cost Series in
fraction-of-book units (multiply by book to get $ cost).

  per_share_ib  — IB MOC + impact + SEC fee + borrow.  Used for equities.
  bps_taker     — flat (taker_bps + slippage_bps) × |Δw|.  Used for crypto perp
                  futures (highest-tier MM, no per-order min, no SEC fee, no
                  borrow / funding).

Both produce the same shape, so the runner can swap them by config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cost_per_share_ib(
    w: pd.DataFrame,
    close: pd.DataFrame,
    book: float,
    *,
    commission_per_share: float = 0.0045,
    per_order_min: float = 0.35,
    sec_fee_per_dollar: float = 27.80e-6,
    sell_fraction: float = 0.50,
    impact_bps: float = 0.5,
    borrow_bps_annual: float = 50.0,
    bars_per_year: int = 252,
) -> pd.Series:
    """Realistic per-share IB MOC fee model (equity)."""
    pos = w * book
    trd = pos.diff().abs()
    safe = close.where(close > 0)
    shares = trd / safe
    pn_comm = (shares * commission_per_share).clip(lower=0)
    has = trd > 1.0
    pn_comm = pn_comm.where(~has, np.maximum(pn_comm, per_order_min)).where(has, 0)
    cost = (pn_comm.sum(axis=1)
            + (trd * sec_fee_per_dollar * sell_fraction).sum(axis=1)
            + (trd * impact_bps / 1e4).sum(axis=1)
            + (-pos.clip(upper=0)).sum(axis=1) * (borrow_bps_annual / 1e4) / float(bars_per_year)
            ) / book
    return cost


def cost_bps_taker(
    w: pd.DataFrame,
    *,
    taker_bps: float = 2.5,
    slippage_bps: float = 1.0,
    **_unused,
) -> pd.Series:
    """Linear bps cost model (crypto perp futures, MM-tier).

    cost_per_bar = (taker_bps + slippage_bps) × Σ|Δw_i| / 1e4
    Each unit of |Δw| is one leg; round-trip is naturally captured.
    """
    total_bps = float(taker_bps) + float(slippage_bps)
    trd = w.diff().abs().sum(axis=1)
    return trd * (total_bps / 1e4)


def make_cost_fn(model: str, params: dict, *, bars_per_year: int):
    """Resolve config → callable.

    Returns a function `f(w, close=...) -> per-bar cost Series`.
    `close` is required only for `per_share_ib`.
    """
    if model == "per_share_ib":
        def f(w, close, book):
            return cost_per_share_ib(w, close, book, bars_per_year=bars_per_year, **params)
        return f
    if model == "bps_taker":
        def f(w, close, book):
            return cost_bps_taker(w, **params)
        return f
    raise ValueError(f"unknown fee model {model!r}")
