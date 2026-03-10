"""
backtest_engine.py — Binary option backtester for Polymarket crypto candle contracts.

Simulates Tier 1 (structural alpha) trading:
- Enter at candle open with a directional bet based on signal
- Settle at candle close: win $1 or lose $0
- Model Polymarket taker fees
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from config import BLENDED_TAKER_FEE, polymarket_taker_fee, INITIAL_CAPITAL, BASE_TRADE_SIZE


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_fees: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_edge: float = 0.0
    pnl_series: pd.Series = field(default_factory=pd.Series)
    cumulative_pnl: pd.Series = field(default_factory=pd.Series)
    daily_pnl: pd.Series = field(default_factory=pd.Series)
    trades_per_day: float = 0.0
    annual_return: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0


def run_backtest(
    signal: pd.Series,
    target: pd.Series,
    entry_price: float = 0.50,
    fee_model: str = "blended",  # "blended" or "dynamic"
    trade_size: float = BASE_TRADE_SIZE,
    threshold: float = 0.0,
    signal_is_probability: bool = False,
    bars_per_day: float = 288,  # 5m = 288, 15m = 96, 1h = 24
) -> BacktestResult:
    """
    Run a backtest of a binary option trading strategy.

    Parameters:
    -----------
    signal : pd.Series
        Signal values. If signal_is_probability=False, positive = predict UP,
        negative = predict DOWN. Magnitude = conviction.
        If signal_is_probability=True, values are P(UP) in [0, 1].
    target : pd.Series
        Actual outcomes: 1 = UP, 0 = DOWN.
    entry_price : float
        Assumed entry price for YES token (default 0.50 for pre-candle entry).
    fee_model : str
        "blended" uses fixed fee, "dynamic" uses probability-dependent fee.
    trade_size : float
        USDC per trade.
    threshold : float
        Minimum signal magnitude to trade.
    signal_is_probability : bool
        If True, signal values are P(UP) in [0, 1].
    bars_per_day : float
        Number of bars per day for annualization.
    """
    # Align signal and target
    common_idx = signal.dropna().index.intersection(target.dropna().index)
    if len(common_idx) == 0:
        return BacktestResult()

    sig = signal.loc[common_idx].copy()
    tgt = target.loc[common_idx].copy()

    # Convert signal to direction and conviction
    if signal_is_probability:
        prob_up = sig.values
        direction = np.where(prob_up > 0.5, 1, np.where(prob_up < 0.5, -1, 0))
        conviction = np.abs(prob_up - 0.5)
    else:
        direction = np.sign(sig.values)
        # Normalize conviction to [0, 1] range using sigmoid
        conviction = 1.0 / (1.0 + np.exp(-np.abs(sig.values) * 5))
        prob_up = np.where(direction > 0, 0.5 + conviction * 0.4, 0.5 - conviction * 0.4)

    outcomes = tgt.values
    n = len(outcomes)

    # Apply threshold filter
    trade_mask = conviction > threshold if not signal_is_probability else np.abs(prob_up - 0.5) > threshold

    # Compute per-trade PnL
    pnl = np.zeros(n)
    fees = np.zeros(n)
    traded = np.zeros(n, dtype=bool)

    for i in range(n):
        if not trade_mask[i] or direction[i] == 0:
            continue

        traded[i] = True

        # Determine entry price (for Tier 1, near 50%)
        p_entry = entry_price

        # Fee calculation
        if fee_model == "dynamic":
            fee = polymarket_taker_fee(p_entry) * trade_size
        else:
            fee = BLENDED_TAKER_FEE * trade_size

        fees[i] = fee

        if direction[i] > 0:  # Bet UP (buy YES)
            if outcomes[i] == 1:  # Correct
                pnl[i] = (1.0 - p_entry) * trade_size - fee
            else:  # Wrong
                pnl[i] = -p_entry * trade_size - fee
        else:  # Bet DOWN (buy NO)
            if outcomes[i] == 0:  # Correct
                pnl[i] = (1.0 - (1.0 - p_entry)) * trade_size - fee
            else:  # Wrong
                pnl[i] = -(1.0 - p_entry) * trade_size - fee

    # Build result
    pnl_series = pd.Series(pnl, index=common_idx)
    traded_pnl = pnl[traded]

    if traded.sum() == 0:
        return BacktestResult()

    wins = (traded_pnl > 0).sum()
    losses = (traded_pnl <= 0).sum()
    total_trades = traded.sum()

    # Daily aggregation for Sharpe
    pnl_df = pd.DataFrame({"pnl": pnl}, index=common_idx)
    daily_pnl = pnl_df.resample("1D").sum()["pnl"]
    daily_pnl = daily_pnl[daily_pnl != 0]  # Remove non-trading days

    # Sharpe ratio
    if len(daily_pnl) > 5:
        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std * np.sqrt(365)) if daily_std > 0 else 0.0
    else:
        sharpe = 0.0

    # Cumulative PnL and drawdown
    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    # Profit factor
    gross_wins = traded_pnl[traded_pnl > 0].sum()
    gross_losses = abs(traded_pnl[traded_pnl <= 0].sum())
    profit_factor = gross_wins / max(gross_losses, 1e-10)

    # Time span
    days = (common_idx[-1] - common_idx[0]).days
    trades_per_day = total_trades / max(days, 1)

    # Annual return
    annual_return = daily_pnl.sum() / max(days, 1) * 365

    result = BacktestResult(
        total_trades=int(total_trades),
        wins=int(wins),
        losses=int(losses),
        win_rate=wins / max(total_trades, 1),
        gross_pnl=traded_pnl.sum() + fees[traded].sum(),
        net_pnl=traded_pnl.sum(),
        total_fees=fees[traded].sum(),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        avg_edge=traded_pnl.mean(),
        pnl_series=pnl_series,
        cumulative_pnl=cumulative,
        daily_pnl=daily_pnl,
        trades_per_day=trades_per_day,
        annual_return=annual_return,
        calmar_ratio=annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        profit_factor=profit_factor,
    )
    return result


def run_combined_backtest(
    signals_df: pd.DataFrame,
    weights: np.ndarray,
    target: pd.Series,
    **kwargs,
) -> BacktestResult:
    """
    Run backtest with a weighted combination of signals.
    """
    # Normalize weights
    w = weights / max(np.abs(weights).sum(), 1e-10)

    # Align columns
    valid_cols = [c for c in signals_df.columns if c in signals_df.columns]
    if len(valid_cols) != len(w):
        w = w[:len(valid_cols)]

    # Combine signals
    combined = (signals_df[valid_cols].values * w[np.newaxis, :]).sum(axis=1)
    combined_series = pd.Series(combined, index=signals_df.index)

    return run_backtest(combined_series, target, **kwargs)


def evaluate_signal(
    signal: pd.Series,
    target: pd.Series,
    bars_per_day: float = 288,
    label: str = "",
) -> Dict:
    """
    Quick evaluation of a single signal. Returns a dict of metrics.
    """
    result = run_backtest(signal, target, bars_per_day=bars_per_day)
    return {
        "name": label,
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "sharpe": result.sharpe,
        "net_pnl": result.net_pnl,
        "max_dd": result.max_drawdown,
        "avg_edge": result.avg_edge,
        "profit_factor": result.profit_factor,
        "trades_per_day": result.trades_per_day,
    }


# ============================================================================
# POLYMARKET HISTORICAL DATA BACKTEST
# ============================================================================

def run_backtest_with_polymarket(
    signal: pd.Series,
    target: pd.Series,
    coin: str,
    interval: str,
    trade_size: float = BASE_TRADE_SIZE,
    threshold: float = 0.0,
    signal_is_probability: bool = False,
    bars_per_day: float = 96,
) -> BacktestResult:
    """
    Run a backtest using REAL Polymarket historical data for entry prices and fees.

    Instead of assuming a fixed 50¢ entry, this uses actual YES/NO prices
    and spreads from the Polymarket CLOB history database.

    Parameters:
    -----------
    signal : pd.Series
        Signal values (same as run_backtest).
    target : pd.Series
        Actual outcomes: 1=UP, 0=DOWN.
    coin : str
        'btc', 'eth', or 'sol'
    interval : str
        '5m', '15m', or '1h'
    trade_size : float
        USD per trade.
    threshold : float
        Minimum edge to trade.
    signal_is_probability : bool
        If True, signal is P(UP) in [0,1].
    bars_per_day : float
        Bars per day for annualization.
    """
    try:
        from fetch_polymarket_history import load_polymarket_history, PM_DB_PATH
    except ImportError:
        print("Warning: fetch_polymarket_history not available, falling back to simulated")
        return run_backtest(signal, target, entry_price=0.50, fee_model="dynamic",
                           trade_size=trade_size, threshold=threshold,
                           signal_is_probability=signal_is_probability,
                           bars_per_day=bars_per_day)

    # Load Polymarket historical data
    pm_data = load_polymarket_history(coin, interval)
    if not pm_data:
        print(f"Warning: No Polymarket history for {coin} {interval}, falling back to simulated")
        return run_backtest(signal, target, entry_price=0.50, fee_model="dynamic",
                           trade_size=trade_size, threshold=threshold,
                           signal_is_probability=signal_is_probability,
                           bars_per_day=bars_per_day)

    # Build a lookup: timestamp → PM data
    pm_lookup = {}
    for row in pm_data:
        ts = row["candle_end_ts"]
        pm_lookup[ts] = row

    print(f"  Loaded {len(pm_data)} Polymarket historical contracts for {coin.upper()} {interval}")
    print(f"  Date range: {pm_data[0]['candle_end_dt'][:10]} to {pm_data[-1]['candle_end_dt'][:10]}")

    # Align signal and target
    common_idx = signal.dropna().index.intersection(target.dropna().index)
    if len(common_idx) == 0:
        return BacktestResult()

    sig = signal.loc[common_idx].copy()
    tgt = target.loc[common_idx].copy()

    # Convert signal
    if signal_is_probability:
        prob_up = sig.values
        direction = np.where(prob_up > 0.5, 1, np.where(prob_up < 0.5, -1, 0))
        conviction = np.abs(prob_up - 0.5)
    else:
        direction = np.sign(sig.values)
        conviction = 1.0 / (1.0 + np.exp(-np.abs(sig.values) * 5))
        prob_up = np.where(direction > 0, 0.5 + conviction * 0.4, 0.5 - conviction * 0.4)

    outcomes = tgt.values
    n = len(outcomes)

    trade_mask = conviction > threshold if not signal_is_probability else np.abs(prob_up - 0.5) > threshold

    pnl = np.zeros(n)
    fees = np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    pm_used = 0
    sim_used = 0

    from polymarket_api import INTERVAL_SECONDS, compute_polymarket_fee

    for i in range(n):
        if not trade_mask[i] or direction[i] == 0:
            continue

        traded[i] = True

        # Try to match this bar to a Polymarket contract
        bar_ts = int(common_idx[i].timestamp())
        secs = INTERVAL_SECONDS.get(interval, 900)
        candle_end = ((bar_ts // secs) + 1) * secs

        pm_row = pm_lookup.get(candle_end)

        if pm_row and pm_row.get("best_ask", 0) > 0:
            # USE REAL POLYMARKET PRICES
            if direction[i] > 0:
                p_entry = pm_row["best_ask"]  # HIT the ask to buy YES
            else:
                p_entry = 1.0 - pm_row["best_bid"]  # Effective NO price
                if p_entry <= 0 or p_entry >= 1:
                    p_entry = 0.50
            fee = compute_polymarket_fee(p_entry) * trade_size
            pm_used += 1
        else:
            # Fallback to simulated
            p_entry = 0.50
            fee = BLENDED_TAKER_FEE * trade_size
            sim_used += 1

        fees[i] = fee

        if direction[i] > 0:  # Bet UP
            if outcomes[i] == 1:
                shares = trade_size / p_entry
                pnl[i] = shares * (1.0 - p_entry) - fee
            else:
                pnl[i] = -trade_size - fee
        else:  # Bet DOWN
            no_price = 1.0 - p_entry
            if no_price <= 0:
                no_price = 0.50
            if outcomes[i] == 0:
                shares = trade_size / no_price
                pnl[i] = shares * (1.0 - no_price) - fee
            else:
                pnl[i] = -trade_size - fee

    print(f"  Polymarket data used for {pm_used}/{pm_used+sim_used} trades ({sim_used} fallback)")

    # Build result (same logic as run_backtest)
    pnl_series = pd.Series(pnl, index=common_idx)
    traded_pnl = pnl[traded]

    if traded.sum() == 0:
        return BacktestResult()

    wins = (traded_pnl > 0).sum()
    losses = (traded_pnl <= 0).sum()
    total_trades = traded.sum()

    pnl_df = pd.DataFrame({"pnl": pnl}, index=common_idx)
    daily_pnl = pnl_df.resample("1D").sum()["pnl"]
    daily_pnl = daily_pnl[daily_pnl != 0]

    if len(daily_pnl) > 5:
        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std * np.sqrt(365)) if daily_std > 0 else 0.0
    else:
        sharpe = 0.0

    cumulative = pnl_series.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

    gross_wins = traded_pnl[traded_pnl > 0].sum()
    gross_losses = abs(traded_pnl[traded_pnl <= 0].sum())
    profit_factor = gross_wins / max(gross_losses, 1e-10)

    days = (common_idx[-1] - common_idx[0]).days
    trades_per_day = total_trades / max(days, 1)
    annual_return = daily_pnl.sum() / max(days, 1) * 365

    result = BacktestResult(
        total_trades=int(total_trades),
        wins=int(wins),
        losses=int(losses),
        win_rate=wins / max(total_trades, 1),
        gross_pnl=traded_pnl.sum() + fees[traded].sum(),
        net_pnl=traded_pnl.sum(),
        total_fees=fees[traded].sum(),
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        avg_edge=traded_pnl.mean(),
        pnl_series=pnl_series,
        cumulative_pnl=cumulative,
        daily_pnl=daily_pnl,
        trades_per_day=trades_per_day,
        annual_return=annual_return,
        calmar_ratio=annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0,
        profit_factor=profit_factor,
    )
    return result
