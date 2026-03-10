"""
live_trade_real.py — REAL MONEY live trading on Polymarket 5m candle contracts.

FIXES from V1:
  1. Contract discovery: slug timestamp = candle START, not end.
     Use (now // 300) * 300 to get the contract starting NOW.
  2. Outcome check: Query Polymarket resolution via Gamma API instead of
     using Binance OHLCV (Chainlink vs Binance disagree ~3% of the time).
  3. Auto-redemption: Redeem winning conditional tokens back to USDC on-chain
     after every resolved contract.

Trade size: $5/trade (~1x Kelly at 52% WR with $194 bankroll)
"""
import sys, os, time, json, asyncio, traceback
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SYMBOLS, SYMBOL_NAMES, DATA_DIR
from polymarket_api import (
    PolymarketClient, CandleContract, Orderbook,
    get_current_candle_end, get_next_candle_end, compute_polymarket_fee
)

try:
    import websockets
except ImportError:
    print("pip install websockets"); sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

import requests as http_requests
try:
    from web3 import Web3
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False
    print("  ⚠ web3 not installed — auto-redemption disabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "REAL_V2"
INTERVAL = "5m"
INTERVAL_SECONDS = 300
BINANCE_WS = "wss://stream.binance.com:9443/ws"
MAX_HISTORY = 500

TRADE_LOG   = Path(__file__).parent / "real_trades.jsonl"
STATE_FILE  = Path(__file__).parent / "real_state.json"
BOOK_LOG    = Path(__file__).parent / "real_book_snapshots.jsonl"
FILL_LOG    = Path(__file__).parent / "real_fill_quality.csv"

# V2-optimized configs
CONFIGS = {
    "BTCUSDT": {"corr_cutoff": 0.90, "max_alphas": 10, "lookback": 1440, "phl": 1},
    "ETHUSDT": {"corr_cutoff": 0.80, "max_alphas": 12, "lookback": 5760, "phl": 1},
    "SOLUSDT": {"corr_cutoff": 0.80, "max_alphas": 15, "lookback": 1440, "phl": 1},
}

TRADE_SIZE_USD = 1.0
MAX_TRADE_SIZE = 5.0
MIN_CAPITAL = 150.0
STARTING_CAPITAL = 194.0   # Updated after initial losses

FEE_PER_TRADE_BPS = 50
BARS_PER_DAY = 288

BINANCE_TO_PM = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol"}

TRADE_DELAY_SECONDS = 0  # Fire immediately when signal generated

# On-chain addresses for auto-redemption
POLYGON_RPC = "https://polygon-bor-rpc.publicnode.com"
CT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# ============================================================================
# ALPHA PRIMITIVES (identical to paper trader)
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
    close = df["close"]; volume = df["volume"]; high = df["high"]; low = df["low"]
    opn = df["open"]; taker_buy = df["taker_buy_base"]
    qv = df["quote_volume"]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vwap = safe_div(qv, volume)
    taker_ratio = safe_div(taker_buy, volume)
    taker_sell = volume - taker_buy
    taker_imbalance = safe_div(taker_buy - taker_sell, volume)
    obv = (np.sign(ret) * volume).cumsum()
    alphas = {}
    for w in [5, 8, 10, 12, 15, 20, 24, 30, 36, 48]:
        alphas[f"mr_{w}"] = -ts_zscore(close, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"logrev_{w}"] = -ts_sum(log_ret, w)
    for w in [3, 5, 8, 10, 12, 15, 20]:
        alphas[f"dstd_{w}"] = -safe_div(delta(close, w), stddev(close, w))
    for w in [5, 10, 15, 20, 30]:
        alphas[f"vwap_mr_{w}"] = -ts_zscore(vwap, w)
    for w in [5, 10, 20]:
        alphas[f"ema_mr_{w}"] = -(close - ema(close, w)) / stddev(close, w*2)
    for w in [10, 20, 30]:
        alphas[f"obv_{w}"] = -ts_zscore(obv, w)
    for w in [10, 20]:
        alphas[f"vp_div_{w}"] = ts_zscore(volume, w) - ts_zscore(close, w)
    for w in [5, 10, 20]:
        alphas[f"tbr_{w}"] = ts_zscore(taker_ratio, w)
        alphas[f"timb_{w}"] = ts_zscore(taker_imbalance, w)
    alpha_df = pd.DataFrame(alphas, index=df.index)
    if not live_mode:
        alpha_df = alpha_df.shift(1)
    return alpha_df


# ============================================================================
# ALPHA SELECTION
# ============================================================================

def evaluate_alpha_nofee(signal, target):
    common = signal.dropna().index.intersection(target.dropna().index)
    if len(common) < 500: return None
    s, t = signal.loc[common], target.loc[common]
    direction = np.sign(s)
    correct = (direction == (2*t - 1))
    wr = correct.mean()
    if wr < 0.505: return None
    daily = (direction * (2*t.astype(float) - 1)).resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 20 or daily.std() == 0: return None
    sharpe = daily.mean() / daily.std() * np.sqrt(365)
    ic = s.corr(t.astype(float))
    return {"nofee_sharpe": sharpe, "win_rate": wr, "ic": ic}


def select_alphas_from_history(symbol):
    cfg = CONFIGS[symbol]
    parquet = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
    if not parquet.exists(): return [], []
    df = pd.read_parquet(parquet)
    train_df = df.loc["2024-03-01":"2025-05-01"]
    if len(train_df) < 10000: return [], []
    target = (train_df["close"] >= train_df["open"]).astype(int)
    alpha_matrix = build_alpha_signals(train_df, live_mode=False)
    results = []
    for col in alpha_matrix.columns:
        m = evaluate_alpha_nofee(alpha_matrix[col], target)
        if m: results.append({"name": col, **m})
    results.sort(key=lambda x: x["nofee_sharpe"], reverse=True)
    selected = []
    for r in results:
        sig = alpha_matrix[r["name"]]
        too_corr = False
        for sel in selected:
            if abs(sig.corr(alpha_matrix[sel["name"]])) > cfg["corr_cutoff"]:
                too_corr = True; break
        if not too_corr:
            selected.append(r)
        if len(selected) >= cfg["max_alphas"]: break
    return [s["name"] for s in selected], selected


# ============================================================================
# KLINE BUFFER
# ============================================================================

class KlineBuffer:
    def __init__(self, symbol, max_size=MAX_HISTORY):
        self.symbol = symbol; self.max_size = max_size
        self.klines = deque(maxlen=max_size)
        self._df_cache = None; self._cache_valid = False

    def add_kline(self, kline):
        bar = {
            "open_time": pd.Timestamp(kline["t"], unit="ms", tz="UTC"),
            "open": float(kline["o"]), "high": float(kline["h"]),
            "low": float(kline["l"]), "close": float(kline["c"]),
            "volume": float(kline["v"]), "close_time": int(kline["T"]),
            "quote_volume": float(kline["q"]), "trades": int(kline["n"]),
            "taker_buy_base": float(kline["V"]),
            "taker_buy_quote": float(kline["Q"]),
        }
        self.klines.append(bar); self._cache_valid = False

    def seed_from_parquet(self, path, n_bars=400):
        df = pd.read_parquet(path)
        for _, row in df.tail(n_bars).iterrows():
            self.klines.append({
                "open_time": row.name, "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"], "volume": row["volume"],
                "quote_volume": row["quote_volume"], "trades": row["trades"],
                "taker_buy_base": row["taker_buy_base"],
            })
        self._cache_valid = False

    def to_dataframe(self):
        if self._cache_valid: return self._df_cache
        if not self.klines: return pd.DataFrame()
        df = pd.DataFrame(list(self.klines))
        df.index = pd.DatetimeIndex(df["open_time"])
        for c in ["open","high","low","close","volume","quote_volume","taker_buy_base"]:
            if c in df.columns: df[c] = df[c].astype(float)
        if "trades" in df.columns: df["trades"] = df["trades"].astype(float)
        self._df_cache = df; self._cache_valid = True
        return df


# ============================================================================
# ADAPTIVE NET SIGNAL ENGINE
# ============================================================================

class AdaptiveNetEngine:
    def __init__(self, symbol, alpha_names, cfg):
        self.symbol = symbol; self.alpha_names = alpha_names; self.cfg = cfg
        self.lookback = cfg["lookback"]; self.phl = cfg["phl"]
        self.outcome_history = []; self.bar_count = 0; self.is_warmup = True

    def update(self, df):
        self.bar_count += 1
        alpha_df = build_alpha_signals(df, live_mode=True)
        cols = [c for c in self.alpha_names if c in alpha_df.columns]
        if len(cols) < 2: return 0, 0.0, {}
        X = alpha_df[cols]
        latest = X.iloc[-1]
        if latest.isna().all(): return 0, 0.0, {}
        target = (df["close"] >= df["open"]).astype(int)
        # Signal on bar N predicts bar N+1's outcome, so shift target
        y = 2.0 * (target.shift(-1).astype(float) - 0.5)
        fee_per = FEE_PER_TRADE_BPS / 10000.0
        fr = pd.DataFrame(index=X.index, columns=cols, dtype=float)
        for col in cols:
            d = np.sign(X[col].values)
            fr[col] = d * y.values - fee_per
        lb = min(self.lookback, len(fr))
        rer = fr.rolling(lb, min_periods=min(200, lb)).mean()
        w = rer.clip(lower=0)
        ws = w.sum(axis=1).replace(0, np.nan)
        wn = w.div(ws, axis=0).fillna(0)
        if self.phl > 1:
            wn = wn.ewm(halflife=self.phl, min_periods=1).mean()
            ws2 = wn.sum(axis=1).replace(0, np.nan)
            wn = wn.div(ws2, axis=0).fillna(0)
        combined = (X * wn).sum(axis=1)
        signal_val = combined.iloc[-1]
        direction = int(np.sign(signal_val)) if not np.isnan(signal_val) else 0
        weights = {c: float(wn[c].iloc[-1]) for c in cols if wn[c].iloc[-1] > 0.01}
        return direction, float(signal_val), weights


# ============================================================================
# CONTRACT DISCOVERY (FIXED: slug timestamp = candle START)
# ============================================================================

def get_current_contract_start_ts():
    """Get the slug timestamp for the contract whose 5m window is happening NOW.

    Polymarket slug format: coin-updown-5m-{START_TIMESTAMP}
    E.g. at 15:07 UTC, the active contract started at 15:05 → ts=15:05.
    """
    now = int(time.time())
    return (now // INTERVAL_SECONDS) * INTERVAL_SECONDS


# ============================================================================
# POLYMARKET RESOLUTION CHECK (via Gamma API, not Binance)
# ============================================================================

def check_polymarket_resolution(slug):
    """Check how a Polymarket contract resolved. Returns 'UP', 'DOWN', or None."""
    try:
        resp = http_requests.get(
            f"https://gamma-api.polymarket.com/markets",
            params={"slug": slug}, timeout=5
        )
        markets = resp.json()
        if not markets:
            return None
        m = markets[0]
        if not m.get("closed"):
            return None
        prices = json.loads(m.get("outcomePrices", '["0.5","0.5"]'))
        up_price = float(prices[0])
        if up_price == 1.0:
            return "UP"
        elif up_price == 0.0:
            return "DOWN"
        return None  # Not yet resolved
    except Exception:
        return None


# ============================================================================
# AUTO-REDEMPTION (on-chain via CTF contract)
# ============================================================================

def try_redeem_position(condition_id):
    """Try to redeem a resolved position on-chain. Returns USDC gained."""
    if not HAS_WEB3:
        return 0.0

    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC, request_kwargs={"timeout": 15}))
        if not w3.is_connected():
            return 0.0

        pk = os.getenv("POLYGON_PRIVATE_KEY")
        acct = w3.eth.account.from_key(pk)

        CT = Web3.to_checksum_address(CT_ADDRESS)
        USDC = Web3.to_checksum_address(USDC_ADDRESS)

        CT_ABI = [
            {"constant": False, "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"}
            ], "name": "redeemPositions", "outputs": [], "type": "function"},
            {"constant": True, "inputs": [
                {"name": "conditionId", "type": "bytes32"}
            ], "name": "payoutDenominator", "outputs": [
                {"name": "", "type": "uint256"}
            ], "type": "function"},
        ]

        USDC_ABI = [
            {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
             "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}],
             "type": "function"},
        ]

        ct = w3.eth.contract(address=CT, abi=CT_ABI)
        usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_ADDRESS), abi=USDC_ABI)

        # Check if resolved on-chain
        cond_bytes = Web3.to_bytes(hexstr=condition_id)
        payout_denom = ct.functions.payoutDenominator(cond_bytes).call()
        if payout_denom == 0:
            return 0.0  # Not resolved on-chain yet

        # Get USDC before
        bal_before = usdc.functions.balanceOf(acct.address).call()

        # Redeem
        nonce = w3.eth.get_transaction_count(acct.address)
        gas_price = w3.eth.gas_price

        tx = ct.functions.redeemPositions(
            Web3.to_checksum_address(USDC_ADDRESS),
            bytes(32),  # parentCollectionId
            cond_bytes,
            [1, 2]  # Both YES and NO index sets
        ).build_transaction({
            "from": acct.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": 200000,
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

        if receipt["status"] == 1:
            bal_after = usdc.functions.balanceOf(acct.address).call()
            gained = (bal_after - bal_before) / 1e6
            return gained
        return 0.0

    except Exception as e:
        print(f"    Redeem error: {e}")
        return 0.0


# ============================================================================
# BOOK SNAPSHOT + ORDER EXECUTION
# ============================================================================

def snapshot_book(pm_client, contract, side="yes"):
    token_id = contract.yes_token_id if side == "yes" else contract.no_token_id
    url = f"https://clob.polymarket.com/book?token_id={token_id}"
    try:
        data = http_requests.get(url, timeout=5).json()
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        bids = sorted(data.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
        tob_ask = float(asks[0]["price"]) if asks else 0
        tob_ask_sz = float(asks[0]["size"]) if asks else 0
        tob_bid = float(bids[0]["price"]) if bids else 0
        tob_bid_sz = float(bids[0]["size"]) if bids else 0
        depth_3c = sum(float(a["price"])*float(a["size"]) for a in asks if float(a["price"]) <= tob_ask + 0.03)
        return {
            "side": side,
            "tob_ask": tob_ask, "tob_ask_size": tob_ask_sz,
            "tob_bid": tob_bid, "tob_bid_size": tob_bid_sz,
            "spread": tob_ask - tob_bid if tob_ask and tob_bid else 0,
            "depth_3c_usd": depth_3c,
            "levels": [(float(a["price"]), float(a["size"])) for a in asks[:5]],
        }
    except Exception as e:
        return {"error": str(e)}


def execute_market_order(clob_client, token_id, size_usd, neg_risk=False):
    t0 = time.time()
    try:
        mo = MarketOrderArgs(token_id=token_id, amount=size_usd, side=BUY)
        if neg_risk:
            signed = clob_client.create_market_order(mo, options={"neg_risk": True})
        else:
            signed = clob_client.create_market_order(mo)
        resp = clob_client.post_order(signed, OrderType.FOK)
        latency_ms = (time.time() - t0) * 1000
        # Check if order was actually matched
        success = resp.get("success", False) or resp.get("status") == "matched"
        return {"success": success, "response": resp, "latency_ms": latency_ms}
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "traceback": traceback.format_exc(),
            "latency_ms": (time.time() - t0) * 1000,
        }


# ============================================================================
# REAL TRADER
# ============================================================================

class RealTrader:
    def __init__(self, clob_client, pm_client):
        self.clob = clob_client
        self.pm = pm_client
        self.capital = STARTING_CAPITAL
        self.trades = []
        self.pending = {}  # trade_id -> trade dict
        self.total_real_pnl = 0.0
        self.consecutive_errors = 0
        self.redeemed_conditions = set()  # Track what we've already redeemed
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.capital = state.get("capital", STARTING_CAPITAL)
                self.trades = state.get("trades", [])
                self.total_real_pnl = state.get("total_real_pnl", 0.0)
                self.redeemed_conditions = set(state.get("redeemed_conditions", []))
                print(f"  Loaded state: ${self.capital:,.2f} capital, {len(self.trades)} trades, "
                      f"PnL=${self.total_real_pnl:+,.2f}")
            except Exception:
                pass

    def save_state(self):
        state = {
            "capital": self.capital,
            "trades": self.trades[-2000:],
            "total_real_pnl": self.total_real_pnl,
            "redeemed_conditions": list(self.redeemed_conditions),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def should_stop(self):
        if self.capital < MIN_CAPITAL:
            print(f"\n  CAPITAL BELOW ${MIN_CAPITAL} -- AUTO-STOPPING")
            return True
        if self.consecutive_errors >= 5:
            print(f"\n  {self.consecutive_errors} CONSECUTIVE ERRORS -- PAUSING")
            return True
        return False

    def open_position(self, symbol, candle_open_time, direction,
                      signal_value, weights, contract=None):
        trade_start = time.time()
        now_ts = datetime.now(timezone.utc).isoformat()
        trade_id = f"{symbol}_5m_{candle_open_time}"

        if not contract or not contract.yes_token_id:
            print(f"    [!] No contract found, skipping trade")
            return None

        # Determine token
        if direction == "UP":
            token_side, token_id = "yes", contract.yes_token_id
        else:
            token_side, token_id = "no", contract.no_token_id

        # Book snapshot BEFORE
        pre_book = snapshot_book(self.pm, contract, side=token_side)

        # Execute
        size = min(TRADE_SIZE_USD, MAX_TRADE_SIZE, self.capital * 0.05)
        try:
            neg_risk = self.clob.get_neg_risk(token_id)
        except:
            neg_risk = False

        fill = execute_market_order(self.clob, token_id, size, neg_risk=neg_risk)

        # Book snapshot AFTER
        post_book = snapshot_book(self.pm, contract, side=token_side)

        expected_price = pre_book.get("tob_ask", 0.50) if pre_book.get("tob_ask") else 0.50
        order_accepted = fill.get("success", False)
        fill_response = fill.get("response", {})
        fee_rate = compute_polymarket_fee(expected_price)
        fee_usd = fee_rate * size

        # Extract actual fill info from response
        taking_amount = float(fill_response.get("takingAmount", 0)) if isinstance(fill_response, dict) else 0
        making_amount = float(fill_response.get("makingAmount", 0)) if isinstance(fill_response, dict) else 0
        actual_price = making_amount / taking_amount if taking_amount > 0 else expected_price

        trade = {
            "id": trade_id,
            "version": VERSION,
            "symbol": symbol,
            "coin": BINANCE_TO_PM.get(symbol, ""),
            "direction": direction,
            "signal": signal_value,
            "top_weights": dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]),
            "size_usd": size,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "shares_received": taking_amount,
            "fee_rate": fee_rate,
            "fee_usd": fee_usd,
            "token_side": token_side,
            "token_id_short": token_id[:20] + "...",
            "full_token_id": token_id,
            "order_accepted": order_accepted,
            "fill_response": str(fill_response)[:500],
            "api_latency_ms": fill.get("latency_ms", 0),
            "fill_error": fill.get("error", None),
            "pre_book": pre_book,
            "post_book": post_book,
            "slug": contract.slug,
            "question": contract.question,
            "condition_id": contract.condition_id,
            "time_remaining_s": contract.time_remaining,
            "time_since_roll_s": INTERVAL_SECONDS - contract.time_remaining,
            "status": "OPEN" if order_accepted else "FAILED",
            "opened_at": now_ts,
        }

        if order_accepted:
            self.pending[trade_id] = trade
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1

        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "OPEN", **trade}, default=str) + "\n")
        with open(BOOK_LOG, "a") as f:
            f.write(json.dumps({
                "event": "TRADE", "time": now_ts, "coin": trade["coin"],
                "direction": direction, "pre": pre_book, "post": post_book,
            }, default=str) + "\n")

        return trade

    def close_position(self, trade_id, pm_outcome):
        """Close position using Polymarket resolution (not Binance).
        pm_outcome: 'UP' or 'DOWN' from check_polymarket_resolution()
        """
        if trade_id not in self.pending:
            return None

        trade = self.pending.pop(trade_id)
        if not trade.get("order_accepted", False):
            return None

        direction = trade["direction"]
        size = trade["size_usd"]
        entry_price = trade["actual_price"] or trade["expected_price"]
        fee = trade["fee_usd"]
        shares = trade.get("shares_received", size / entry_price)

        # Binary payoff based on POLYMARKET resolution
        won = (direction == pm_outcome)

        if won:
            # Shares pay $1 each, cost was entry_price per share
            pnl = shares * (1.0 - entry_price) - fee  # Approximation
            trade["result"] = "WIN"
        else:
            # Shares worth $0
            pnl = -(size + fee)
            trade["result"] = "LOSS"

        trade["pnl"] = pnl
        trade["pm_outcome"] = pm_outcome
        trade["status"] = "CLOSED"
        trade["closed_at"] = datetime.now(timezone.utc).isoformat()

        self.capital += pnl
        self.total_real_pnl += pnl
        self.trades.append(trade)
        self.save_state()

        with open(TRADE_LOG, "a") as f:
            f.write(json.dumps({"event": "CLOSE", **trade}, default=str) + "\n")

        # Fill quality CSV
        if not FILL_LOG.exists():
            with open(FILL_LOG, "w") as f:
                f.write("time,coin,direction,expected_price,actual_price,size,fee,result,pnl,"
                        "tob_ask,spread,depth_3c,latency_ms,pm_outcome\n")
        with open(FILL_LOG, "a") as f:
            pre = trade.get("pre_book", {})
            f.write(f"{trade['opened_at']},{trade['coin']},{trade['direction']},"
                    f"{trade['expected_price']:.3f},{entry_price:.3f},{size:.2f},"
                    f"{fee:.3f},{trade['result']},{pnl:.2f},"
                    f"{pre.get('tob_ask',0):.3f},{pre.get('spread',0):.3f},"
                    f"{pre.get('depth_3c_usd',0):.1f},{trade.get('api_latency_ms',0):.0f},"
                    f"{pm_outcome}\n")

        # Auto-redeem if WIN
        if won and trade.get("condition_id"):
            cond = trade["condition_id"]
            if cond not in self.redeemed_conditions:
                print(f"    Redeeming {cond[:16]}...")
                gained = try_redeem_position(cond)
                if gained > 0:
                    print(f"    Redeemed +${gained:.2f} USDC")
                    self.redeemed_conditions.add(cond)
                    self.save_state()
                else:
                    print(f"    Redeem pending (not yet settled on-chain)")

        return trade

    def try_redeem_all_pending(self):
        """Try to redeem any previously unresolved winning positions."""
        for trade in self.trades:
            if trade.get("result") != "WIN":
                continue
            cond = trade.get("condition_id", "")
            if not cond or cond in self.redeemed_conditions:
                continue
            gained = try_redeem_position(cond)
            if gained > 0:
                print(f"    [REDEEM] {trade['slug']}: +${gained:.2f} USDC")
                self.redeemed_conditions.add(cond)
        self.save_state()

    def get_stats(self):
        if not self.trades: return {}
        recent = self.trades[-100:]
        wins = sum(1 for t in recent if t.get("result") == "WIN")
        total = len(recent)
        return {
            "capital": self.capital,
            "total_trades": len(self.trades),
            "recent_wr": wins / max(total, 1),
            "recent_pnl": sum(t.get("pnl", 0) for t in recent),
            "all_time_pnl": self.total_real_pnl,
            "pending": len(self.pending),
        }


def update_parquet_data():
    """Download latest candles from Binance and update parquet files.
    Called at startup to ensure data is current before trading."""
    print("  [DATA] Checking data freshness...")
    for symbol in SYMBOLS:
        parquet_path = DATA_DIR / f"{symbol}_{INTERVAL}.parquet"
        if not parquet_path.exists():
            print(f"    [!] {symbol}: parquet not found!")
            continue

        old_df = pd.read_parquet(parquet_path)
        last_bar = old_df.index[-1]
        now = pd.Timestamp.now(tz="UTC")
        staleness = (now - last_bar).total_seconds() / 60

        if staleness > 10:  # More than 10 minutes stale
            print(f"    {SYMBOL_NAMES[symbol]}: stale by {staleness:.0f}min, updating...")
            try:
                url = "https://api.binance.com/api/v3/klines"
                resp = http_requests.get(url, params={
                    "symbol": symbol, "interval": INTERVAL, "limit": 1000
                }, timeout=10)
                data = resp.json()
                rows = []
                for k in data:
                    rows.append({
                        "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                        "open": float(k[1]), "high": float(k[2]),
                        "low": float(k[3]), "close": float(k[4]),
                        "volume": float(k[5]), "close_time": int(k[6]),
                        "quote_volume": float(k[7]), "trades": int(k[8]),
                        "taker_buy_base": float(k[9]),
                        "taker_buy_quote": float(k[10]),
                    })
                new_df = pd.DataFrame(rows)
                new_df.index = pd.DatetimeIndex(new_df["open_time"])
                new_df = new_df.drop(columns=["open_time"])

                combined = pd.concat([old_df, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined.to_parquet(parquet_path)

                new_last = combined.index[-1]
                gap_check = combined.index.to_series().diff().dt.total_seconds()
                max_gap = gap_check.max()
                n_gaps = (gap_check > 310).sum()  # > 5min + 10s tolerance

                print(f"    {SYMBOL_NAMES[symbol]}: updated to {new_last} "
                      f"({len(combined)} bars, {n_gaps} gaps, max_gap={max_gap:.0f}s)")
            except Exception as e:
                print(f"    {SYMBOL_NAMES[symbol]}: update failed: {e}")
        else:
            print(f"    {SYMBOL_NAMES[symbol]}: fresh ({staleness:.0f}min old, {len(old_df)} bars)")

    print("  [DATA] Check complete\n")


async def run_live_feed():
    active_symbols = list(SYMBOLS)

    print(f"\n{'='*72}")
    print(f"  POLYMARKET 5m CANDLE -- REAL MONEY TRADING V2")
    print(f"  Trade Size: ${TRADE_SIZE_USD}/trade")
    print(f"  Fixes: Correct contract alignment + PM resolution + auto-redeem")
    print(f"{'='*72}")
    print(f"  Symbols:  {', '.join(SYMBOL_NAMES[s] for s in active_symbols)}")
    print()

    # Step 0: Ensure data is fresh
    update_parquet_data()

    # CLOB client
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=os.getenv("POLYGON_PRIVATE_KEY"),
        chain_id=137, signature_type=0,
        funder=os.getenv("POLYGON_WALLET_ADDRESS"),
    )
    creds = clob.create_or_derive_api_creds()
    clob.set_api_creds(creds)
    print(f"  [OK] CLOB API authenticated")

    pm_client = PolymarketClient()
    print(f"  [OK] Polymarket API initialized")

    # Test
    try:
        contracts = pm_client.discover_all_current_contracts(INTERVAL)
        for c in (contracts or []):
            book = pm_client.get_orderbook(c, side="yes")
            print(f"    {c.coin.upper()}: Bid/Ask ${book.best_bid:.3f}/${book.best_ask:.3f}")
    except Exception as e:
        print(f"  [!] Contract test: {e}")

    # Engines
    engines = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        print(f"\n  Initializing {SYMBOL_NAMES[symbol]} engine...")
        alpha_names, _ = select_alphas_from_history(symbol)
        if not alpha_names:
            alpha_names = [f"mr_{w}" for w in [10, 15, 20]] + [f"logrev_{w}" for w in [10, 20]]
        engines[symbol] = AdaptiveNetEngine(symbol, alpha_names, cfg)
        print(f"    Selected {len(alpha_names)} alphas: {alpha_names[:5]}...")

    # Buffers — seed with enough bars for the full lookback window
    buffers = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        n_seed = max(cfg["lookback"] + 200, 2000)  # ETH needs 5760+200
        buf = KlineBuffer(symbol, max_size=n_seed + 500)
        buf.seed_from_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet", n_bars=n_seed)
        buffers[symbol] = buf
        df = buf.to_dataframe()
        if len(df) > 100:
            engine = engines[symbol]
            for i in range(50, len(df)):
                engine.update(df.iloc[:i])
            engine.is_warmup = False
            print(f"    [OK] {SYMBOL_NAMES[symbol]} warm ({len(df)} bars, lookback={cfg['lookback']})")

    # Trader
    trader = RealTrader(clob, pm_client)

    # Try to redeem any old pending wins
    print("\n  Checking for unredeemed positions...")
    trader.try_redeem_all_pending()

    # WebSocket
    streams = [f"{s.lower()}@kline_{INTERVAL}" for s in active_symbols]
    ws_url = f"{BINANCE_WS}/{'/'.join(streams)}"
    last_trade_id = {s: None for s in active_symbols}
    last_trade_slug = {s: None for s in active_symbols}

    print(f"\n  Connecting to Binance WebSocket...")
    print(f"  Trade delay: {TRADE_DELAY_SECONDS}s after bar close")
    print(f"\n  {'Time':<10} {'Asset':<5} {'Dir':>4} {'Signal':>8} {'Entry$':>8} "
          f"{'Result':>8} {'PnL':>10} {'Capital':>12}")
    print(f"  {'-'*80}")

    while True:
        if trader.should_stop():
            print("\n  TRADING STOPPED -- check logs")
            break

        try:
            async with websockets.connect(ws_url, ping_interval=20) as ws:
                print(f"  [OK] Connected to Binance WebSocket")

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

                        # 1) Close pending position — check POLYMARKET resolution
                        prev_id = last_trade_id[symbol]
                        prev_slug = last_trade_slug[symbol]
                        if prev_id and prev_id in trader.pending and prev_slug:
                            # Query Polymarket for the actual resolution
                            pm_outcome = check_polymarket_resolution(prev_slug)
                            if pm_outcome is None:
                                # Not resolved yet — use Binance as fallback
                                # (matches ~97% of the time)
                                binance_up = float(kline["c"]) >= float(kline["o"])
                                pm_outcome = "UP" if binance_up else "DOWN"
                                outcome_source = "BINANCE_FALLBACK"
                            else:
                                outcome_source = "POLYMARKET"

                            closed = trader.close_position(prev_id, pm_outcome)
                            if closed:
                                r = "WIN " if closed["result"] == "WIN" else "LOSS"
                                print(f"  {now_str:<10} {name:<5} {'':>4} {'':>8} "
                                      f"{'':>8} {r:>8} ${closed['pnl']:>+9.2f} "
                                      f"${trader.capital:>11.2f} [{outcome_source}]")

                        # 2) Add bar
                        buffers[symbol].add_kline(kline)

                        # 3) Signal
                        df = buffers[symbol].to_dataframe()
                        if len(df) < 60:
                            continue
                        direction, signal_val, weights = engines[symbol].update(df)
                        if direction == 0:
                            last_trade_id[symbol] = None
                            last_trade_slug[symbol] = None
                            continue

                        # 4) Discover contract — FIXED: use candle START timestamp
                        #    No delay — fire immediately for best fill price
                        contract = None
                        if pm_coin:
                            try:
                                # The contract we want to bet on has its 5m window
                                # starting NOW (after the delay)
                                start_ts = get_current_contract_start_ts()
                                contract = pm_client.discover_contract(
                                    pm_coin, INTERVAL, start_ts
                                )
                            except Exception as e:
                                print(f"    [!] Contract discovery failed: {e}")

                        # 6) Trade
                        dir_str = "UP" if direction > 0 else "DOWN"
                        trade = trader.open_position(
                            symbol=symbol,
                            candle_open_time=kline["T"],
                            direction=dir_str,
                            signal_value=signal_val,
                            weights=weights,
                            contract=contract,
                        )
                        if trade:
                            last_trade_id[symbol] = trade["id"]
                            last_trade_slug[symbol] = trade.get("slug")
                            arrow = "UP" if dir_str == "UP" else "DN"
                            status = "LIVE" if trade["order_accepted"] else "FAIL"
                            entry = trade.get("expected_price", 0.50)
                            latency = trade.get("api_latency_ms", 0)
                            shares = trade.get("shares_received", 0)
                            print(f"  {now_str:<10} {name:<5} {arrow:>4} "
                                  f"{signal_val:>+8.4f} ${entry:>7.3f} "
                                  f"{status:>8} {'':>10} "
                                  f"{'':>12} [{latency:.0f}ms {shares:.1f}sh]")
                            if contract:
                                print(f"    -> {contract.question}")
                        else:
                            last_trade_id[symbol] = None
                            last_trade_slug[symbol] = None

                        # 7) Stats
                        stats = trader.get_stats()
                        n = stats.get("total_trades", 0)
                        if n > 0 and n % 15 == 0:
                            print(f"\n  === REAL STATS ({n} trades) ===")
                            print(f"  Capital: ${stats['capital']:.2f}  "
                                  f"WR: {stats['recent_wr']:.1%}  "
                                  f"PnL: ${stats['all_time_pnl']:+.2f}")
                            print()

                        # 8) Periodic redemption attempts (every 10 trades)
                        if n > 0 and n % 10 == 0:
                            trader.try_redeem_all_pending()

                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"  Error: {e}")
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
    parser = argparse.ArgumentParser(description="Polymarket 5m -- REAL MONEY V2")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--redeem", action="store_true", help="Redeem all pending wins")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stats:
        pm = PolymarketClient()
        trader = RealTrader(ClobClient("https://clob.polymarket.com"), pm)
        print(json.dumps(trader.get_stats(), indent=2))
        return

    if args.redeem:
        pm = PolymarketClient()
        trader = RealTrader(ClobClient("https://clob.polymarket.com"), pm)
        print("Attempting to redeem all pending wins...")
        trader.try_redeem_all_pending()
        return

    pk = os.getenv("POLYGON_PRIVATE_KEY")
    addr = os.getenv("POLYGON_WALLET_ADDRESS")
    if not pk or not addr:
        print("ERROR: Set POLYGON_PRIVATE_KEY and POLYGON_WALLET_ADDRESS in .env")
        sys.exit(1)

    print(f"  Wallet: {addr}")
    print(f"  Trade size: ${TRADE_SIZE_USD}")
    print(f"  Safety stop: ${MIN_CAPITAL}")

    if args.dry_run:
        print("\n  DRY RUN mode")
        return

    try:
        asyncio.run(run_live_feed())
    except KeyboardInterrupt:
        print("\n\n  Trading stopped (Ctrl+C). State saved.")


if __name__ == "__main__":
    main()
