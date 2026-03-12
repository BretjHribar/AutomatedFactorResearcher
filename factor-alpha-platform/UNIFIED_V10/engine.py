"""
Unified Streaming Signal Engine — V10

DESIGN PRINCIPLE (Dubno/Goldman):
  "The system should not be able to tell which world it is in.
   Backtesting on recorded data should produce the same results as the live run."
  "Even in backtesting, data is only fed to the system one at a time."
  "No function can query external stats."

This engine processes ONE BAR AT A TIME through on_bar().
The SAME code path is used for backtesting and live trading.
No .shift() tricks. No vectorized lookahead. No separate code paths.

SIGNAL TIMING (critical):
  - on_bar(bar_T) is called when bar_T is FULLY CLOSED
  - Alpha values are computed from data through close[T]
  - The direction decision uses alpha[T] (current bar)
  - This direction will be HELD from close[T] to close[T+1]
  - PnL from previous direction is realized: prev_dir * return[T]
  - Fee charged on direction change at T

PNL ACCOUNTING:
  At on_bar(T):
    realized_pnl = prev_direction * (close[T]/close[T-1] - 1)
    fee = FEE_FRAC * |new_direction - prev_direction|
    net_pnl = realized_pnl - fee

FACTOR RETURNS (for adaptive weights):
  factor_return[T] = sign(alpha_value[T-1]) * return[T]
  Uses PREVIOUS bar's alpha (stored in self._prev_alpha_values)
  This is the same lag as the vectorized backtest's .shift(1)

CRITICAL: Alpha and factor return computation happens on EVERY bar,
  even during warmup. Only the trade DIRECTION output is gated.
  This ensures the weight history matches the vectorized version exactly.
"""
import numpy as np
import pandas as pd
from collections import deque

from .config import FEE_FRAC


class StreamingEngine:
    """Unified streaming signal engine.

    Processes one 1H bar at a time. Maintains all internal state.
    Same code path for backtest and live.

    Usage:
        engine = StreamingEngine(config)
        for bar in historical_bars:
            result = engine.on_bar(bar)
    """

    def __init__(self, selected_alphas, lookback=120, phl=1,
                 buffer_size=2000):
        """
        Args:
            selected_alphas: list of alpha column names to use
            lookback: rolling window for adaptive weight computation
            phl: position halflife for weight smoothing (1 = no smoothing)
            buffer_size: max bars to keep in rolling buffer
        """
        self.selected_alphas = list(selected_alphas)
        self.lookback = lookback
        self.phl = phl
        self.buffer_size = buffer_size

        # Cached DataFrame — avoids rebuilding from list every bar
        self._cached_df = None

        # Rolling buffer of OHLCV bars (list of dicts)
        self._bars = []

        # State tracking
        self._bar_count = 0
        self._prev_alpha_values = {}   # Alpha values from PREVIOUS bar
        self._prev_direction = 0       # Direction held entering this bar
        self._prev_close = None        # Previous bar's close price
        self._cumulative_pnl = 0.0     # Running total PnL (fractional)

        # Factor returns history (deque with max length = lookback)
        self._factor_returns = deque(maxlen=lookback)

        # EWM state for weight smoothing (phl > 1)
        self._weights_ewm = None
        self._ewm_alpha = 2.0 / (phl + 1) if phl > 1 else None

    def on_bar(self, bar):
        """Process a single closed 1H bar. THE ONLY ENTRY POINT.

        Args:
            bar: dict with keys: datetime, open, high, low, close,
                 volume, quote_volume, taker_buy_volume, taker_buy_quote_volume

        Returns:
            dict with:
                direction: +1 (long), -1 (short), 0 (flat/warmup)
                signal_value: combined signal strength
                realized_pnl: PnL earned from PREVIOUS direction * current return
                fee: fee charged for direction change
                net_pnl: realized_pnl - fee
                cumulative_pnl: running total
                bar_count: number of bars processed
        """
        self._bar_count += 1
        self._bars.append(dict(bar))  # Defensive copy

        # Trim buffer to max size
        if len(self._bars) > self.buffer_size:
            self._bars = self._bars[-self.buffer_size:]
            self._cached_df = None  # Force rebuild after trim

        current_close = bar['close']

        # --- Step 1: Compute return from previous bar ---
        current_return = 0.0
        if self._prev_close is not None:
            current_return = current_close / self._prev_close - 1

        # --- Step 2: Compute realized PnL from previous direction ---
        realized_pnl = self._prev_direction * current_return

        # --- Step 3: ALWAYS compute alpha values (even during warmup) ---
        # This ensures factor return history matches vectorized exactly.
        current_alphas = self._compute_alphas()

        # --- Step 4: ALWAYS update factor returns (even during warmup) ---
        # CRITICAL: NaN alpha → NaN factor return (excluded from mean),
        # matching vectorized rolling.mean() which skips NaN.
        # Skip bar_count <= 2: in vectorized, pct_change() produces NaN at
        # index 0, and shift(1) on alpha produces NaN at index 1. So the
        # first valid factor return is at index 2 (bar_count=3).
        if (self._prev_alpha_values and self._prev_close is not None
                and self._bar_count >= 3):
            fr_row = {}
            for name in self.selected_alphas:
                prev_val = self._prev_alpha_values.get(name, np.nan)
                if prev_val is None or (isinstance(prev_val, float) and np.isnan(prev_val)):
                    fr_row[name] = np.nan  # Will be excluded from mean
                else:
                    fr_row[name] = np.sign(prev_val) * current_return
            self._factor_returns.append(fr_row)

        # --- Step 5: Save current alphas for next bar ---
        self._prev_alpha_values = current_alphas.copy()

        # --- Step 6: Compute weights and direction ---
        # Gate: need at least min(100, lookback) factor returns
        min_history = min(100, self.lookback)
        if len(self._factor_returns) < min_history:
            # During warmup: direction=0, accumulate state
            net_pnl = realized_pnl  # no fee during warmup
            self._cumulative_pnl += net_pnl
            self._prev_close = current_close
            return {
                'direction': 0,
                'signal_value': 0.0,
                'realized_pnl': realized_pnl,
                'fee': 0.0,
                'net_pnl': net_pnl,
                'cumulative_pnl': self._cumulative_pnl,
                'bar_count': self._bar_count,
                'timestamp': bar.get('datetime', self._bar_count),
                'close_price': current_close,
                'reason': 'warmup',
                'weights': None,
                'alphas': current_alphas,
            }

        # --- Step 7: Compute adaptive weights ---
        weights = self._compute_weights()

        # --- Step 8: Compute combined signal ---
        combined = 0.0
        for name in self.selected_alphas:
            alpha_val = current_alphas.get(name, 0.0)
            weight_val = weights.get(name, 0.0)
            # NaN alpha or NaN weight contributes 0 to combined
            if alpha_val is None or (isinstance(alpha_val, float) and np.isnan(alpha_val)):
                alpha_val = 0.0
            if weight_val is None or (isinstance(weight_val, float) and np.isnan(weight_val)):
                weight_val = 0.0
            combined += alpha_val * weight_val

        # NaN or zero combined → flat
        if np.isnan(combined):
            direction = 0
        else:
            direction = int(np.sign(combined))

        # --- Step 9: Compute fee for direction change ---
        fee = FEE_FRAC * abs(direction - self._prev_direction)

        # --- Step 10: Update state ---
        net_pnl = realized_pnl - fee
        self._cumulative_pnl += net_pnl
        self._prev_direction = direction
        self._prev_close = current_close

        return {
            'direction': direction,
            'signal_value': combined,
            'realized_pnl': realized_pnl,
            'fee': fee,
            'net_pnl': net_pnl,
            'cumulative_pnl': self._cumulative_pnl,
            'bar_count': self._bar_count,
            'timestamp': bar.get('datetime', self._bar_count),
            'close_price': current_close,
            'reason': 'signal',
            'weights': weights,
            'alphas': current_alphas,
        }

    def _build_df(self):
        """Build or update cached DataFrame from bar buffer.

        Optimization: append new row instead of rebuilding from scratch.
        Only rebuild after buffer trim or on first call.
        """
        if self._cached_df is not None and len(self._bars) == len(self._cached_df) + 1:
            # Fast path: just append the latest bar
            new_bar = self._bars[-1]
            new_row = pd.DataFrame([new_bar])
            if 'datetime' in new_row.columns:
                new_row = new_row.set_index('datetime')
            for col in ['open', 'high', 'low', 'close', 'volume',
                         'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
                if col in new_row.columns:
                    new_row[col] = pd.to_numeric(new_row[col], errors='coerce')
            self._cached_df = pd.concat([self._cached_df, new_row])
            return self._cached_df

        # Full rebuild (first call or after buffer trim)
        df_1h = pd.DataFrame(self._bars)
        if 'datetime' in df_1h.columns:
            df_1h = df_1h.set_index('datetime').sort_index()
        else:
            df_1h.index = pd.RangeIndex(len(df_1h))

        for col in ['open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
            if col in df_1h.columns:
                df_1h[col] = pd.to_numeric(df_1h[col], errors='coerce')

        self._cached_df = df_1h
        return df_1h

    def _compute_alphas(self):
        """Compute all alpha values from the internal buffer.

        Uses the SAME alpha functions as the vectorized backtest,
        applied to a DataFrame built from the buffer.
        Returns only the LAST row's values (no lookahead possible).
        """
        df_1h = self._build_df()

        # Compute 1H alphas using the SAME functions as vectorized backtest
        from .alphas import build_1h_alphas
        all_alphas = build_1h_alphas(df_1h)

        # HTF signals (resample from buffer)
        if len(df_1h) >= 50 and isinstance(df_1h.index, pd.DatetimeIndex):
            from .alphas import build_htf_signals
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                   'volume': 'sum'}
            for col in ['quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']:
                if col in df_1h.columns:
                    agg[col] = 'sum'

            for freq, prefix in [('2h', 'h2'), ('4h', 'h4'),
                                  ('8h', 'h8'), ('12h', 'h12')]:
                try:
                    df_htf = df_1h.resample(freq).agg(agg).dropna()
                    if len(df_htf) >= 5:
                        htf_alphas = build_htf_signals(df_htf, df_1h, prefix, shift_n=1)
                        all_alphas.update(htf_alphas)
                except Exception:
                    pass

        # Extract only selected alphas, last value only
        # CRITICAL: preserve NaN — do NOT convert to 0. NaN alpha means
        # the alpha couldn't be computed (insufficient data). NaN propagates
        # to factor returns and is excluded from weight computation,
        # matching vectorized rolling.mean() behavior.
        result = {}
        for name in self.selected_alphas:
            if name in all_alphas:
                series = all_alphas[name]
                val = series.iloc[-1] if len(series) > 0 else np.nan
                result[name] = float(val)  # Preserve NaN
            else:
                result[name] = np.nan

        return result

    def _compute_weights(self):
        """Compute adaptive net weights from factor return history.

        Uses EXACTLY the same computation as the vectorized backtest:
            rolling_er = factor_returns.rolling(lookback, min_periods=min(100, lookback)).mean()
            weights = rolling_er.clip(lower=0)
            weights_norm = weights / weights.sum()

        This ensures identical weight values between streaming and vectorized.
        """
        if len(self._factor_returns) == 0:
            return {name: 0.0 for name in self.selected_alphas}

        # Build DataFrame from factor returns history
        fr_df = pd.DataFrame(list(self._factor_returns))

        # Use rolling mean with SAME parameters as vectorized backtest
        # The deque has at most `lookback` entries, so rolling(lookback)
        # on the deque is equivalent to rolling on the last `lookback` entries
        # of the full factor return history.
        min_p = min(100, self.lookback)
        rolling_er = fr_df.rolling(self.lookback, min_periods=min_p).mean()

        # Take the LAST row of the rolling mean (current weights)
        mean_fr = rolling_er.iloc[-1]

        # Only positive-return alphas get weight (NaN → 0)
        weights = mean_fr.clip(lower=0).fillna(0)
        wsum = weights.sum()

        if wsum == 0 or np.isnan(wsum):
            return {name: 0.0 for name in self.selected_alphas}

        weights_norm = weights / wsum

        # Apply EWM smoothing if phl > 1
        if self.phl > 1 and self._weights_ewm is not None:
            for name in weights_norm.index:
                old = self._weights_ewm.get(name, 0.0)
                new = weights_norm.get(name, 0.0)
                self._weights_ewm[name] = (
                    self._ewm_alpha * new + (1 - self._ewm_alpha) * old)
            ewm_vals = pd.Series(self._weights_ewm)
            ewm_sum = ewm_vals.sum()
            if ewm_sum > 0:
                weights_norm = ewm_vals / ewm_sum
        elif self.phl > 1:
            self._weights_ewm = {name: weights_norm.get(name, 0.0)
                                  for name in self.selected_alphas}

        return {name: float(weights_norm.get(name, 0.0))
                for name in self.selected_alphas}

    def get_state(self):
        """Return serializable state for inspection/debugging."""
        return {
            'bar_count': self._bar_count,
            'prev_direction': self._prev_direction,
            'prev_close': self._prev_close,
            'cumulative_pnl': self._cumulative_pnl,
            'n_factor_returns': len(self._factor_returns),
            'prev_alpha_values': dict(self._prev_alpha_values),
            'buffer_size': len(self._bars),
        }

    def reset(self):
        """Reset all state. Used between walk-forward folds."""
        self._bars.clear()
        self._bar_count = 0
        self._prev_alpha_values = {}
        self._prev_direction = 0
        self._prev_close = None
        self._cumulative_pnl = 0.0
        self._factor_returns.clear()
        self._weights_ewm = None
