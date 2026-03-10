"""
live_paper_trade_v2.py — V2 Live paper trader with OPTIMIZED configs.

Changes from V1:
  - phl=1 (no position smoothing — each bar is an independent binary bet)
  - Shorter lookbacks for BTC/SOL (1440 vs 5760 — faster adaptation)
  - corr=0.90 for BTC (fewer, more orthogonal alphas)
  - Kelly-based trade sizing ($500/trade = 0.25x Kelly for ~2% edge)
  - Separate state files (paper_state_5m_v2.json) for concurrent running

Backtest improvement: Combined Sharpe 12.97 → 17.89 (holdout, after fees)

Usage:
    python live_paper_trade_v2.py              # Run with all 3 assets
    python live_paper_trade_v2.py --symbol ETH # ETH only
    python live_paper_trade_v2.py --stats      # Show summary
"""
import sys, os, json, time, asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (SYMBOLS, SYMBOL_NAMES, DATA_DIR, BASE_TRADE_SIZE, BLENDED_TAKER_FEE)
from polymarket_api import (
    PolymarketClient, CandleContract, Orderbook,
    get_current_candle_end, get_next_candle_end, compute_polymarket_fee
)

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "V2"  # Differentiator tag for state/logging
INTERVAL = "5m"
BINANCE_WS = "wss://stream.binance.com:9443/ws"
MAX_HISTORY = 500          # Bars to keep in rolling buffer
PAPER_TRADE_LOG = Path(__file__).parent / "paper_trades_5m_v2.jsonl"
PAPER_STATE_FILE = Path(__file__).parent / "paper_state_5m_v2.json"

# V2 OPTIMIZED configs: phl=1 (no smoothing), shorter lookbacks
# Holdout SR: BTC=10.24, ETH=14.49, SOL=10.21 (combined=17.89)
CONFIGS = {
    "BTCUSDT": {"corr_cutoff": 0.90, "max_alphas": 10, "lookback": 1440, "phl": 1},
    "ETHUSDT": {"corr_cutoff": 0.80, "max_alphas": 12, "lookback": 5760, "phl": 1},
    "SOLUSDT": {"corr_cutoff": 0.80, "max_alphas": 15, "lookback": 1440, "phl": 1},
}
FEE_PER_TRADE_BPS = 50     # For adaptive weight computation
BARS_PER_DAY = 288

# Trade sizing: fractional Kelly
# Edge estimate: ~2% (52% WR on 50/50 payoff at ~$0.50 entry)
# Full Kelly = edge / odds = 0.02 / 1.0 = 2% of capital = ~$1,000 on $50k
# We use 0.5x Kelly = $500 per trade for risk management
V2_TRADE_SIZE = 500

BINANCE_TO_PM = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol"}

# ============================================================================
# ALPHA PRIMITIVES (identical to iterate_5m_v2.py)
# ============================================================================

def sma(s, w): return s.rolling(w, min_periods=1).mean()
def ema(s, w): return s.ewm(halflife=w, min_periods=1).mean()
def stddev(s, w): return s.rolling(w, min_periods=2).std()
def ts_zscore(s, w):
    m = s.rolling(w, min_periods=2).mean()
    sd = s.rolling(w, min_periods=2).std()
    return (s - m) / sd.replace(0, np.nan)
def delta(s, p): return s - s.shift(p)
def ts_sum(s, w): return s.rolling(w, min_periods=1).sum()
def safe_div(a, b):
    r = a / b; return r.replace([np.inf, -np.inf], 0).fillna(0)


def build_alpha_signals(df, live_mode=False):
    """Build all alpha signals from OHLCV.
    
    In backtest mode (live_mode=False): shift(1) prevents lookahead bias.
    In live mode (live_mode=True): NO shift — the natural event flow already
    provides the 1-bar lag (compute after bar close, bet on next bar).
    
    EXACT same alpha library as iterate_5m_v2.py."""
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    opn = df["open"]
    qv = df["quote_volume"]
    taker_buy = df["taker_buy_base"]
    trades = df["trades"].astype(float)

    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)

    alphas = {}

    # Mean reversion
    for w in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)

    # Cumulative return reversal
    for w in [3, 4, 5, 6, 8, 10, 12, 15, 20, 24]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)

    # Normalized delta
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))

    # VWAP z-score
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)

    # Close-VWAP deviation
    dev = close - vwap
    for w in [5, 10, 20]:
        alphas[f"cvdev_{w}"] = -ts_zscore(dev, w)

    # OBV momentum
    obv = (np.sign(ret) * volume).cumsum()
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)

    # Acceleration
    for w in [5, 8, 12]:
        alphas[f"accel_{w}"] = -ts_zscore(delta(sma(ret, w), w), w * 2)

    # Volume z-score signed by return
    for w in [5, 10, 20]:
        alphas[f"vol_signed_{w}"] = ts_zscore(volume, w) * np.sign(ret)

    # Taker buy ratio
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)

    # Delta taker
    for w in [3, 5, 10]:
        alphas[f"dtaker_{w}"] = delta(taker_ratio, w)

    # Close position in range
    close_pos = safe_div(close - low, high - low)
    for w in [5, 10, 20]:
        alphas[f"cpos_{w}"] = -ts_zscore(close_pos, w)

    # Body
    body = close - opn
    for w in [5, 10]:
        alphas[f"body_{w}"] = -ts_zscore(body, w)

    # Range expansion
    hl = safe_div(high - low, close)
    for w in [10, 20]:
        alphas[f"rng_{w}"] = -ts_zscore(hl, w)

    # Trade intensity
    intensity = safe_div(trades, volume)
    for w in [10, 20]:
        alphas[f"intens_{w}"] = ts_zscore(intensity, w)

    # EMA mean reversion
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w * 2)

    # High-low midpoint reversion
    mid = (high + low) / 2
    for w in [10, 20]:
        alphas[f"mid_mr_{w}"] = -ts_zscore(close - mid, w)

    # Volume-weighted price reversion
    for w in [10, 20]:
        alphas[f"vw_mr_{w}"] = -ts_zscore(close - vwap, w) * ts_zscore(volume, w).clip(-2, 2)

    # NEW: EMA of log returns (reversal) — HO SR=17.00
    log_ret = np.log(close / close.shift(1))
    for hl in [3, 5, 8]:
        alphas[f"ema_ret_{hl}"] = -ema(log_ret, hl)

    # NEW: Volatility-conditioned MR — HO SR=16.58
    vol_z = ts_zscore(stddev(close, 10), 50)
    for w in [8, 10, 15, 20]:
        alphas[f"hivol_mr_{w}"] = -ts_zscore(close, w) * vol_z.clip(0, 3)

    # NEW: VWAP deviation / ATR — HO SR=16.21
    atr = sma(high - low, 14)
    for w in [5, 10, 15]:
        alphas[f"vwap_atr_{w}"] = -safe_div(close - sma(vwap, w), atr)

    # NEW: Bollinger Band position — HO SR=15.95
    for w in [10, 15, 20]:
        bb_mid = sma(close, w)
        bb_band = 2 * stddev(close, w)
        alphas[f"bb_{w}"] = -safe_div(close - bb_mid, bb_band)

    alpha_df = pd.DataFrame(alphas, index=df.index)
    if not live_mode:
        alpha_df = alpha_df.shift(1)  # Backtest: prevent lookahead
    # Live mode: no shift — we already compute AFTER bar close
    return alpha_df


# ============================================================================
# ALPHA SELECTION (done once from historical data at startup)
# ============================================================================

def evaluate_alpha_nofee(signal, target):
    """Evaluate an alpha with no fees — IC + Sharpe."""
    binary_return = 2.0 * (target.astype(float) - 0.5)
    common = signal.dropna().index.intersection(binary_return.dropna().index)
    if len(common) < 500:
        return None
    sig = signal.loc[common]
    ret = binary_return.loc[common]

    ic_series = sig.rolling(BARS_PER_DAY, min_periods=100).corr(ret)
    ic_mean = ic_series.dropna().mean()

    direction = np.sign(sig)
    nofee_pnl = direction * ret
    daily_pnl = nofee_pnl.resample("1D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]
    nofee_sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(365)
                    if len(daily_pnl) > 10 and daily_pnl.std() > 0 else 0.0)
    return {"ic_mean": ic_mean, "nofee_sharpe": nofee_sharpe}


def select_alphas_from_history(symbol):
    """Load historical data at startup, discover + select orthogonal alphas."""
    cfg = CONFIGS[symbol]
    path = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
    if not path.exists():
        print(f"  WARNING: No historical data for {symbol}")
        return [], None

    df = pd.read_parquet(path)
    # Use ALL available data for alpha selection (most recent = best)
    alpha_matrix = build_alpha_signals(df)
    target = (df["close"] >= df["open"]).astype(int)

    results = []
    for col in alpha_matrix.columns:
        m = evaluate_alpha_nofee(alpha_matrix[col], target)
        if m is None:
            continue
        results.append({"name": col, **m})

    results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)

    # Orthogonal selection
    selected = []
    for r in results:
        if r["name"] not in alpha_matrix.columns:
            continue
        if r["ic_mean"] < 0.005:
            continue
        sig = alpha_matrix[r["name"]]
        too_corr = False
        for sel in selected:
            if abs(sig.corr(alpha_matrix[sel["name"]])) > cfg["corr_cutoff"]:
                too_corr = True
                break
        if not too_corr:
            selected.append(r)
        if len(selected) >= cfg["max_alphas"]:
            break

    alpha_names = [s["name"] for s in selected]
    return alpha_names, selected


# ============================================================================
# KLINE BUFFER
# ============================================================================

class KlineBuffer:
    """Rolling buffer of 5m klines for signal computation."""
    def __init__(self, symbol, max_size=MAX_HISTORY):
        self.symbol = symbol
        self.max_size = max_size
        self.klines = deque(maxlen=max_size)
        self._df_cache = None
        self._cache_valid = False

    def add_kline(self, kline):
        self.klines.append({
            "open_time": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline["q"]),
            "trades": int(kline["n"]),
            "taker_buy_base": float(kline["V"]),
            "taker_buy_quote": float(kline["Q"]),
        })
        self._cache_valid = False

    def seed_from_parquet(self, path, n_bars=400):
        if not path.exists():
            return
        df = pd.read_parquet(path)
        recent = df.tail(n_bars)
        for _, row in recent.iterrows():
            self.klines.append({
                "open_time": row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp(row.name),
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"], "quote_volume": row["quote_volume"],
                "trades": int(row["trades"]),
                "taker_buy_base": row["taker_buy_base"],
                "taker_buy_quote": row["taker_buy_quote"],
            })
        self._cache_valid = False
        print(f"  Seeded {self.symbol} with {len(self.klines)} bars")

    def to_dataframe(self):
        if self._cache_valid and self._df_cache is not None:
            return self._df_cache
        if len(self.klines) < 60:
            return pd.DataFrame()
        records = list(self.klines)
        df = pd.DataFrame(records)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)
        self._df_cache = df
        self._cache_valid = True
        return df


# ============================================================================
# ADAPTIVE NET SIGNAL ENGINE
# ============================================================================

class AdaptiveNetEngine:
    """
    Implements the Adaptive Net Factor Returns strategy per-asset.

    On each new bar:
      1. Recomputes all alpha signals from the rolling buffer
      2. Computes per-alpha directional PnL (with fee estimate)
      3. Weights alphas by rolling mean of net returns (positive ER only)
      4. Returns the combined signal direction (UP/DOWN/FLAT)
    """
    def __init__(self, symbol, alpha_names, cfg):
        self.symbol = symbol
        self.alpha_names = alpha_names
        self.cfg = cfg
        self.lookback = cfg["lookback"]
        self.phl = cfg["phl"]
        self.fee_per = FEE_PER_TRADE_BPS / 10000.0

        # Rolling state
        self.alpha_history = []  # list of {name: value} dicts
        self.outcome_history = []  # list of +1/-1
        self.bar_count = 0
        self.is_warmup = True  # True during historical warmup

    def update(self, df):
        """Given the current kline DataFrame, compute the signal for the NEXT bar.
        Returns (direction, signal_value, weights_dict)."""
        if len(df) < 60:
            return 0, 0.0, {}

        # Build alpha signals — live_mode=True means NO shift(1)
        # (the 1-bar lag is inherent: we compute AFTER bar close, bet on NEXT bar)
        alpha_matrix = build_alpha_signals(df, live_mode=True)
        latest_alphas = {}
        for col in self.alpha_names:
            if col in alpha_matrix.columns:
                val = alpha_matrix[col].iloc[-1]
                latest_alphas[col] = val if not np.isnan(val) else 0.0

        if not latest_alphas:
            return 0, 0.0, {}

        # Record outcome of the PREVIOUS bar (for weight update)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-1]
            prev_open = df["open"].iloc[-1]
            prev_outcome = 1 if prev_close >= prev_open else 0
            binary_ret = 2.0 * (prev_outcome - 0.5)  # +1 or -1

            # Record what each alpha WOULD HAVE predicted for this bar
            if self.bar_count > 0 and len(self.alpha_history) > 0:
                prev_alphas = self.alpha_history[-1]
                outcome_entry = {}
                for name, val in prev_alphas.items():
                    d = np.sign(val)
                    net_ret = d * binary_ret - self.fee_per
                    outcome_entry[name] = net_ret
                self.outcome_history.append(outcome_entry)

        # Store current alpha values for next bar's outcome tracking
        self.alpha_history.append(latest_alphas)
        self.bar_count += 1

        # Keep memory bounded
        max_mem = self.lookback + 100
        if len(self.alpha_history) > max_mem:
            self.alpha_history = self.alpha_history[-max_mem:]
        if len(self.outcome_history) > max_mem:
            self.outcome_history = self.outcome_history[-max_mem:]

        # During warmup (historical seeding), only accumulate outcome history
        # Don't return a trading signal — we're just building the ER estimates
        if self.is_warmup:
            return 0, 0.0, {}

        # Compute rolling net expected return per alpha
        if len(self.outcome_history) < 200:
            # Not enough adaptive history — use equal-weight on raw signals
            # NO smoothing — take the direction the alphas vote THIS bar
            combined = sum(latest_alphas.values())
            weights = {k: 1.0/len(latest_alphas) for k in latest_alphas}
            direction = np.sign(combined)
            return int(direction), combined, weights

        # Build rolling ER from outcome history
        window = min(self.lookback, len(self.outcome_history))
        recent = self.outcome_history[-window:]

        er = {}
        for name in self.alpha_names:
            returns = [entry.get(name, 0.0) for entry in recent if name in entry]
            if len(returns) > 50:
                er[name] = np.mean(returns)
            else:
                er[name] = 0.0

        # Only positive-ER alphas get weight
        positive_er = {k: v for k, v in er.items() if v > 0}
        if not positive_er:
            # All alphas negative ER — use equal weight on top 3
            top3 = sorted(er.items(), key=lambda x: x[1], reverse=True)[:3]
            positive_er = {k: max(v, 0.001) for k, v in top3}

        total_er = sum(positive_er.values())
        weights = {k: v / total_er for k, v in positive_er.items()}

        # Weighted combination — use RAW signal, no position smoothing
        # With binary options there's no partial position to smooth.
        # Each bar is an independent UP/DOWN bet.
        combined = sum(latest_alphas.get(k, 0) * w for k, w in weights.items())
        direction = np.sign(combined)
        return int(direction), combined, weights


# ============================================================================
# PAPER TRADING STATE
# ============================================================================

class PaperTrader:
    def __init__(self, pm_client):
        self.pm_client = pm_client
        self.capital = 50_000.0
        self.trades = []
        self.pending = {}
        self.pnl_by_symbol = {}
        self.load_state()

    def load_state(self):
        if PAPER_STATE_FILE.exists():
            try:
                with open(PAPER_STATE_FILE) as f:
                    state = json.load(f)
                self.capital = state.get("capital", 50_000.0)
                self.trades = state.get("trades", [])
                self.pnl_by_symbol = state.get("pnl_by_symbol", {})
                print(f"  Loaded state: ${self.capital:,.0f} capital, {len(self.trades)} trades")
            except Exception:
                pass

    def save_state(self):
        state = {
            "capital": self.capital,
            "trades": self.trades[-2000:],
            "pnl_by_symbol": self.pnl_by_symbol,
        }
        with open(PAPER_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def open_position(self, symbol, candle_open_time, direction,
                      signal_value, weights, contract=None):
        """Open a paper position with real Polymarket data."""
        pm_data = {}
        entry_price = 0.50
        trade_size = V2_TRADE_SIZE
        fee = BLENDED_TAKER_FEE * trade_size

        if contract and contract.yes_token_id:
            try:
                if direction == "UP":
                    book = self.pm_client.get_orderbook(contract, side="yes")
                    entry_price = book.best_ask if book.best_ask < 1.0 else 0.50
                    action, token = "HIT_ASK", "YES"
                else:
                    book = self.pm_client.get_orderbook(contract, side="no")
                    entry_price = book.best_ask if book.best_ask < 1.0 else 0.50
                    action, token = "HIT_ASK", "NO"

                fee_rate = compute_polymarket_fee(entry_price)
                fee = fee_rate * trade_size

                pm_data = {
                    "pm_slug": contract.slug,
                    "pm_question": contract.question,
                    "entry_price": entry_price,
                    "spread": book.spread,
                    "action": action,
                    "token": token,
                    "fee_rate": fee_rate,
                    "source": "POLYMARKET_LIVE",
                    "time_remaining": contract.time_remaining,
                }
            except Exception as e:
                pm_data = {"source": "SIMULATED", "error": str(e)}
        else:
            pm_data = {"source": "SIMULATED"}

        trade_id = f"{symbol}_5m_{candle_open_time}"
        trade = {
            "id": trade_id,
            "symbol": symbol,
            "name": SYMBOL_NAMES.get(symbol, symbol),
            "direction": direction,
            "signal": signal_value,
            "entry_price": entry_price,
            "fee": fee,
            "top_weights": dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]),
            "status": "OPEN",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            **pm_data,
        }
        self.pending[trade_id] = trade

        with open(PAPER_TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "OPEN", **trade}, default=str) + "\n")
        return trade

    def close_position(self, trade_id, outcome):
        """Close with actual candle outcome (1=UP, 0=DOWN)."""
        if trade_id not in self.pending:
            return None

        trade = self.pending.pop(trade_id)
        entry_price = trade["entry_price"]
        fee = trade["fee"]
        direction = trade["direction"]

        trade_size = V2_TRADE_SIZE
        if direction == "UP":
            if outcome == 1:
                pnl = (1.0 - entry_price) * trade_size - fee
                trade["result"] = "WIN"
            else:
                pnl = -entry_price * trade_size - fee
                trade["result"] = "LOSS"
        else:
            if outcome == 0:
                pnl = entry_price * trade_size - fee
                trade["result"] = "WIN"
            else:
                pnl = -(1.0 - entry_price) * trade_size - fee
                trade["result"] = "LOSS"

        trade["pnl"] = pnl
        trade["outcome"] = "UP" if outcome == 1 else "DOWN"
        trade["status"] = "CLOSED"
        trade["closed_at"] = datetime.now(timezone.utc).isoformat()

        self.capital += pnl
        self.trades.append(trade)

        sym = trade["symbol"]
        self.pnl_by_symbol[sym] = self.pnl_by_symbol.get(sym, 0) + pnl

        self.save_state()

        with open(PAPER_TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "CLOSE", **trade}, default=str) + "\n")
        return trade

    def get_stats(self):
        if not self.trades:
            return {}
        recent = self.trades[-100:]
        wins = sum(1 for t in recent if t.get("result") == "WIN")
        total = len(recent)
        total_pnl = sum(t.get("pnl", 0) for t in recent)
        return {
            "capital": self.capital,
            "total_trades": len(self.trades),
            "recent_100_wr": wins / max(total, 1),
            "recent_100_pnl": total_pnl,
            "pending": len(self.pending),
            "pnl_by_symbol": self.pnl_by_symbol,
        }


# ============================================================================
# LIVE FEED
# ============================================================================

async def run_live_feed(symbols_filter=None):
    active_symbols = [s for s in SYMBOLS if not symbols_filter or SYMBOL_NAMES[s] in symbols_filter]

    print(f"\n{'='*72}")
    print(f"  POLYMARKET 5m CANDLE — ADAPTIVE NET FACTOR RETURNS [V2]")
    print(f"  Live Paper Trading with OPTIMIZED Configs")
    print(f"{'='*72}")
    print(f"  Symbols:  {', '.join(SYMBOL_NAMES[s] for s in active_symbols)}")
    print(f"  Interval: {INTERVAL}")
    print(f"  Strategy: Adaptive Net (phl=1, Kelly-sized)")
    print(f"  Trade Size: ${V2_TRADE_SIZE}/contract (0.5x Kelly)")
    print(f"  State: {PAPER_STATE_FILE.name}")
    print()

    # Initialize Polymarket client
    pm_client = PolymarketClient()
    print(f"  ✓ Polymarket API initialized")

    # Test connectivity
    try:
        contracts = pm_client.discover_all_current_contracts(INTERVAL)
        for c in (contracts or []):
            book = pm_client.get_orderbook(c, side="yes")
            print(f"    {c.coin.upper()} {INTERVAL}: Bid/Ask ${book.best_bid:.3f}/${book.best_ask:.3f} "
                  f"Spread={book.spread:.3f}")
    except Exception as e:
        print(f"  ⚠ Contract discovery: {e}")

    # Initialize signal engines (load historical data, select alphas)
    engines = {}
    for symbol in active_symbols:
        name = SYMBOL_NAMES[symbol]
        cfg = CONFIGS[symbol]
        print(f"\n  Initializing {name} engine...")
        alpha_names, selected = select_alphas_from_history(symbol)
        if not alpha_names:
            print(f"    ⚠ No alphas found for {name}, using defaults")
            alpha_names = [f"mr_{w}" for w in [10, 15, 20]] + [f"logrev_{w}" for w in [10, 20]]
            selected = [{"name": n} for n in alpha_names]
        engines[symbol] = AdaptiveNetEngine(symbol, alpha_names, cfg)
        print(f"    Selected {len(alpha_names)} alphas: {alpha_names[:5]}...")

    # Initialize buffers (seed with historical data)
    buffers = {}
    for symbol in active_symbols:
        buf = KlineBuffer(symbol)
        buf.seed_from_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet", n_bars=400)
        buffers[symbol] = buf

        # Warm up the engine with historical data
        df = buf.to_dataframe()
        if len(df) > 100:
            engine = engines[symbol]
            print(f"  Warming up {SYMBOL_NAMES[symbol]} engine with {len(df)} bars...")
            for i in range(50, len(df)):
                sub_df = df.iloc[:i]
                engine.update(sub_df)
            # Switch from warmup to live mode
            engine.is_warmup = False
            print(f"    ✓ Engine warm, "
                  f"outcome_history={len(engine.outcome_history)} bars")

    # Paper trader
    trader = PaperTrader(pm_client)

    # WebSocket
    streams = [f"{s.lower()}@kline_{INTERVAL}" for s in active_symbols]
    ws_url = f"{BINANCE_WS}/{'/'.join(streams)}"
    last_trade_id = {s: None for s in active_symbols}

    print(f"\n  Connecting to Binance WebSocket...")
    print(f"  Streams: {', '.join(streams)}")
    print(f"  Press Ctrl+C to stop\n")
    print(f"  {'Time':<10} {'Asset':<5} {'Dir':>4} {'Signal':>8} {'Entry$':>8} "
          f"{'Result':>8} {'PnL':>10} {'Capital':>12} {'Source':<10}")
    print(f"  {'-'*90}")

    while True:
        try:
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                print(f"  ✓ Connected to Binance WebSocket")

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        kline = data.get("k", {})
                        symbol = kline.get("s", "")
                        is_closed = kline.get("x", False)

                        if symbol not in active_symbols or not is_closed:
                            continue

                        name = SYMBOL_NAMES[symbol]
                        pm_coin = BINANCE_TO_PM.get(symbol, "")
                        now_str = datetime.now().strftime('%H:%M:%S')

                        # 1) Close any pending position from the PREVIOUS candle
                        prev_id = last_trade_id[symbol]
                        if prev_id and prev_id in trader.pending:
                            outcome = 1 if float(kline["c"]) >= float(kline["o"]) else 0
                            closed = trader.close_position(prev_id, outcome)
                            if closed:
                                r = "✅ WIN " if closed["result"] == "WIN" else "❌ LOSS"
                                print(f"  {now_str:<10} {name:<5} {'':>4} {'':>8} "
                                      f"{'':>8} {r:>8} ${closed['pnl']:>+9,.0f} "
                                      f"${trader.capital:>11,.0f}")

                        # 2) Add the closed candle to the buffer
                        buffers[symbol].add_kline(kline)

                        # 3) Compute signal for the NEXT candle
                        df = buffers[symbol].to_dataframe()
                        if len(df) < 60:
                            continue

                        direction, signal_val, weights = engines[symbol].update(df)

                        if direction == 0:
                            last_trade_id[symbol] = None
                            continue

                        # 4) Discover the NEXT Polymarket contract
                        contract = None
                        if pm_coin:
                            try:
                                next_end = get_next_candle_end(INTERVAL)
                                contract = pm_client.discover_contract(pm_coin, INTERVAL, next_end)
                            except Exception:
                                pass

                        # 5) Open position
                        dir_str = "UP" if direction > 0 else "DOWN"
                        trade = trader.open_position(
                            symbol=symbol,
                            candle_open_time=kline["T"],
                            direction=dir_str,
                            signal_value=signal_val,
                            weights=weights,
                            contract=contract,
                        )
                        last_trade_id[symbol] = trade["id"]

                        arrow = "🟢 ↑" if dir_str == "UP" else "🔴 ↓"
                        src = trade.get("source", "SIM")[:8]
                        entry = trade.get("entry_price", 0.50)
                        print(f"  {now_str:<10} {name:<5} {arrow:>4} "
                              f"{signal_val:>+8.4f} ${entry:>7.3f} "
                              f"{'OPEN':>8} {'':>10} "
                              f"{'':>12} {src:<10}")

                        if contract:
                            print(f"    └─ {contract.question}")

                        # 6) Periodic stats
                        stats = trader.get_stats()
                        n = stats.get("total_trades", 0)
                        if n > 0 and n % 15 == 0:
                            print(f"\n  ═══ STATS ({n} trades) ═══")
                            print(f"  Capital: ${stats['capital']:,.0f}  "
                                  f"WR: {stats['recent_100_wr']:.1%}  "
                                  f"Recent PnL: ${stats['recent_100_pnl']:+,.0f}")
                            for sym, pnl in stats.get("pnl_by_symbol", {}).items():
                                print(f"    {sym}: ${pnl:+,.0f}")
                            print()

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"  Error: {e}")
                        import traceback
                        traceback.print_exc()

        except websockets.exceptions.ConnectionClosed:
            print(f"\n  Connection lost, reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"\n  WebSocket error: {e}, reconnecting in 10s...")
            await asyncio.sleep(10)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket 5m Candle — Adaptive Net Paper Trading")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Trade single asset: BTC, ETH, or SOL")
    parser.add_argument("--stats", action="store_true", help="Show current stats")
    args = parser.parse_args()

    if args.stats:
        pm_client = PolymarketClient()
        trader = PaperTrader(pm_client)
        stats = trader.get_stats()
        print(json.dumps(stats, indent=2))
        return

    symbols_filter = None
    if args.symbol:
        symbols_filter = [args.symbol.upper()]

    try:
        asyncio.run(run_live_feed(symbols_filter))
    except KeyboardInterrupt:
        print("\n\nPaper trading stopped.")


if __name__ == "__main__":
    main()
