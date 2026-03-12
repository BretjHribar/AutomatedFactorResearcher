"""
signal_engine.py — Computes V9b alpha signals and adaptive net direction (live mode).

CRITICAL: In live mode, NO .shift(1) is applied to the alpha matrix.
The shift is implicit because we only compute signals from closed bars.
HTF signals STILL use .shift(1) on the HTF data (matches backtest).
"""
import sys, os
import numpy as np
import pandas as pd
import json

# Import the production alpha library from FINAL_v2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FINAL_v2', 'scripts'))
from univariate_hf_v9b_mtf import (
    build_1h_alphas, build_htf_signals, build_cross_asset_signals,
    safe_div, ts_zscore, ts_sum, ema, stddev, decay_exp, correlation,
    sma, ts_min, ts_max, delta,
)
from config import FEE_FRAC, PARAMS_FILE


class SignalEngine:
    """Computes V9b signals for a single symbol in live mode."""

    def __init__(self, symbol, frozen_config):
        """
        Args:
            symbol: e.g. 'BTCUSDT'
            frozen_config: dict with keys:
                - selected_alphas: list of alpha column names
                - lookback: int (rolling window for adaptive weights)
                - phl: int (position halflife smoothing)
        """
        self.symbol = symbol
        self.selected_alphas = frozen_config['selected_alphas']
        self.lookback = frozen_config['lookback']
        self.phl = frozen_config['phl']

        # Running state for adaptive weights
        self.factor_returns_history = []  # list of dicts {alpha_name: directional_return}
        self.prev_direction = 0
        self.bar_count = 0
        self._warmed_up = False
        self._prev_alpha_values = {}  # BUG FIX: track previous bar's alphas for lagged factor returns

    def warmup_from_history(self, df_1h, df_2h, df_4h, df_8h, df_12h, all_1h_data):
        """Pre-seed factor returns from historical bars so adaptive weights
        are immediately available on the first live signal.
        
        This replays the last `lookback` 1H bars to build up the
        factor_returns_history, exactly matching what the backtest does.
        """
        if self._warmed_up:
            return
        
        n_bars = len(df_1h)
        if n_bars < 200:
            return
        
        # Build all alpha signals over full history
        alphas_1h = build_1h_alphas(df_1h)
        a2h = build_htf_signals(df_2h, df_1h, 'h2', shift_n=1)
        a4h = build_htf_signals(df_4h, df_1h, 'h4', shift_n=1)
        a8h = build_htf_signals(df_8h, df_1h, 'h8', shift_n=1)
        a12h = build_htf_signals(df_12h, df_1h, 'h12', shift_n=1)
        across = build_cross_asset_signals(all_1h_data, self.symbol, df_1h)
        
        all_alphas = {**alphas_1h, **a2h, **a4h, **a8h, **a12h, **across}
        available = [a for a in self.selected_alphas if a in all_alphas]
        
        if len(available) < 2:
            return
        
        # Build alpha DataFrame (no shift — see BUG FIX below)
        alpha_df = pd.DataFrame({name: all_alphas[name] for name in available},
                                index=df_1h.index)
        
        returns = df_1h['close'].pct_change()
        
        # Replay last `lookback` bars to seed factor returns
        # BUG FIX: Use alpha[i-1] not alpha[i] to match backtest's .shift(1)
        # Backtest: factor_return[t] = sign(alpha_shifted[t]) * return[t]
        #   where alpha_shifted[t] = alpha[t-1] due to .shift(1)
        # So we must use alpha[i-1] here too.
        warmup_start = max(1, n_bars - self.lookback - 1)  # start at 1 so i-1 >= 0
        for i in range(warmup_start + 1, n_bars):
            ret_val = returns.iloc[i]
            if np.isnan(ret_val):
                continue
            fr_row = {}
            for name in available:
                sig_val = alpha_df[name].iloc[i - 1]  # BUG FIX: use PREVIOUS bar's alpha
                if np.isnan(sig_val):
                    sig_val = 0.0
                direction_i = np.sign(sig_val)
                fr_row[name] = direction_i * ret_val
            self.factor_returns_history.append(fr_row)
        
        # Trim to lookback
        if len(self.factor_returns_history) > self.lookback:
            self.factor_returns_history = self.factor_returns_history[-self.lookback:]
        
        self._warmed_up = True
        print(f"  [{self.symbol}] Warmup complete: {len(self.factor_returns_history)} "
              f"factor return history bars seeded")

    def compute_signal(self, df_1h, df_2h, df_4h, df_8h, df_12h, all_1h_data):
        """Compute the current trading signal from closed bar data.

        CRITICAL: In live mode:
          - df_1h, df_2h, etc. contain ONLY fully closed bars
          - We do NOT apply .shift(1) to the alpha matrix
          - HTF signals use .shift(1) on the HTF data (inside build_htf_signals)

        Args:
            df_1h: DataFrame of closed 1H bars
            df_2h, df_4h, df_8h, df_12h: higher-TF DataFrames
            all_1h_data: dict {symbol: df_1h} for cross-asset signals

        Returns:
            (direction, signal_value, diagnostics)
            direction: +1 (long), -1 (short), 0 (flat/warmup)
        """
        self.bar_count += 1

        if len(df_1h) < 100:
            return 0, 0.0, {'reason': 'warmup', 'n_bars': len(df_1h)}

        # Build all alpha signals (NO shift in live mode)
        alphas_1h = build_1h_alphas(df_1h)
        a2h = build_htf_signals(df_2h, df_1h, 'h2', shift_n=1)
        a4h = build_htf_signals(df_4h, df_1h, 'h4', shift_n=1)
        a8h = build_htf_signals(df_8h, df_1h, 'h8', shift_n=1)
        a12h = build_htf_signals(df_12h, df_1h, 'h12', shift_n=1)
        across = build_cross_asset_signals(all_1h_data, self.symbol, df_1h)

        all_alphas = {**alphas_1h, **a2h, **a4h, **a8h, **a12h, **across}

        # Get latest values for selected alphas
        available = [a for a in self.selected_alphas if a in all_alphas]
        if len(available) < 2:
            return 0, 0.0, {'reason': 'insufficient_alphas', 'available': len(available)}

        # Get latest alpha values (last row = most recent closed bar)
        latest_values = {}
        for name in available:
            series = all_alphas[name]
            val = series.iloc[-1] if not series.empty else np.nan
            latest_values[name] = val if not np.isnan(val) else 0.0

        # Current 1H return (for adaptive weight update)
        close = df_1h['close']
        if len(close) >= 2:
            current_return = close.iloc[-1] / close.iloc[-2] - 1
        else:
            current_return = 0.0

        # Update factor returns history
        # BUG FIX: Use PREVIOUS bar's alpha values for factor return computation
        # This matches backtest: factor_return[t] = sign(alpha[t-1]) * return[t]
        if self._prev_alpha_values:  # Skip first bar (no previous values)
            fr_row = {}
            for name in available:
                prev_val = self._prev_alpha_values.get(name, 0.0)
                direction_i = np.sign(prev_val)
                fr_row[name] = direction_i * current_return
            self.factor_returns_history.append(fr_row)
        
        # Save current values for next bar's factor return computation
        self._prev_alpha_values = latest_values.copy()

        # Keep only lookback window
        if len(self.factor_returns_history) > self.lookback:
            self.factor_returns_history = self.factor_returns_history[-self.lookback:]

        # Compute adaptive weights from factor returns history
        if len(self.factor_returns_history) < min(100, self.lookback):
            return 0, 0.0, {'reason': 'weight_warmup',
                            'n_history': len(self.factor_returns_history)}

        fr_df = pd.DataFrame(self.factor_returns_history)
        mean_fr = fr_df.mean()  # Average factor return per alpha
        weights = mean_fr.clip(lower=0)  # Only positive-return alphas
        wsum = weights.sum()

        if wsum == 0:
            return 0, 0.0, {'reason': 'all_negative_weights'}

        weights_norm = weights / wsum

        # Combine signals
        combined = sum(latest_values.get(name, 0) * weights_norm.get(name, 0)
                       for name in available)
        direction = int(np.sign(combined))

        # Diagnostics
        top_5 = weights_norm.nlargest(5)
        diagnostics = {
            'combined_signal': float(combined),
            'n_active_alphas': int((weights_norm > 0).sum()),
            'n_history': len(self.factor_returns_history),
            'top_alphas': {k: float(v) for k, v in top_5.items()},
            'top_signals': {k: float(latest_values.get(k, 0)) for k in top_5.index},
        }

        return direction, float(combined), diagnostics

    @staticmethod
    def load_frozen_params(params_file=None):
        """Load frozen parameters from JSON file."""
        path = params_file or PARAMS_FILE
        with open(path) as f:
            return json.load(f)
