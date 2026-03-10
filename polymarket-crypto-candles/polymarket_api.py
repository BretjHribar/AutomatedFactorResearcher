"""
polymarket_api.py — Polymarket CLOB + Gamma API client for crypto candle contracts.
Handles market discovery, orderbook reads, price history, and contract resolution.
"""
import time
import json
import requests
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Slug pattern:  {coin}-updown-{interval}-{unix_timestamp}
# 1h pattern:    {asset}-up-or-down-{month}-{day}-{time}-et
COINS = ["btc", "eth", "sol"]
INTERVALS = ["5m", "15m"]  # 1h uses different slug pattern
INTERVAL_SECONDS = {"5m": 300, "15m": 900, "1h": 3600}

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december"
}

COIN_NAMES_1H = {"btc": "bitcoin", "eth": "ethereum", "sol": "solana"}


@dataclass
class OrderbookLevel:
    price: float
    size: float


@dataclass
class Orderbook:
    bids: List[OrderbookLevel] = field(default_factory=list)
    asks: List[OrderbookLevel] = field(default_factory=list)
    timestamp: int = 0

    @property
    def best_bid(self) -> float:
        return max((l.price for l in self.bids), default=0.0)

    @property
    def best_ask(self) -> float:
        return min((l.price for l in self.asks), default=1.0)

    @property
    def mid(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb > 0 and ba < 1:
            return (bb + ba) / 2
        return 0.50

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def taker_buy_price(self, size: float = 0) -> float:
        """Price to buy YES shares (hit the ask)."""
        if not self.asks:
            return 1.0
        if size <= 0:
            return self.best_ask
        # Walk up the ask ladder
        remaining = size
        cost = 0.0
        for level in sorted(self.asks, key=lambda l: l.price):
            fill = min(remaining, level.size)
            cost += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        if remaining > 0:
            return 1.0  # not enough liquidity
        return cost / size

    def taker_sell_price(self, size: float = 0) -> float:
        """Price to sell YES shares (lift the bid) = effectively buy NO."""
        if not self.bids:
            return 0.0
        if size <= 0:
            return self.best_bid
        remaining = size
        proceeds = 0.0
        for level in sorted(self.bids, key=lambda l: -l.price):
            fill = min(remaining, level.size)
            proceeds += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        if remaining > 0:
            return 0.0
        return proceeds / size


@dataclass
class CandleContract:
    """Represents a single Polymarket crypto candle contract."""
    coin: str           # "btc", "eth", "sol"
    interval: str       # "5m", "15m", "1h"
    candle_end_ts: int  # Unix timestamp when candle ends
    slug: str
    condition_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""
    question: str = ""
    yes_price: float = 0.50
    no_price: float = 0.50
    volume: float = 0.0
    active: bool = True
    resolved: bool = False
    outcome: Optional[str] = None  # "Up" or "Down" after resolution

    @property
    def candle_start_ts(self) -> int:
        return self.candle_end_ts - INTERVAL_SECONDS[self.interval]

    @property
    def time_remaining(self) -> float:
        """Seconds until candle closes."""
        return max(0, self.candle_end_ts - time.time())

    @property
    def candle_progress(self) -> float:
        """0.0 = just started, 1.0 = expired."""
        duration = INTERVAL_SECONDS[self.interval]
        elapsed = time.time() - self.candle_start_ts
        return min(1.0, max(0.0, elapsed / duration))


def build_slug(coin: str, interval: str, candle_end_ts: int) -> str:
    """Build the Polymarket slug for a candle contract."""
    if interval in ("5m", "15m"):
        return f"{coin}-updown-{interval}-{candle_end_ts}"
    elif interval == "1h":
        dt = datetime.fromtimestamp(candle_end_ts, tz=timezone(timedelta(hours=-5)))  # ET
        coin_name = COIN_NAMES_1H.get(coin, coin)
        month = MONTH_NAMES[dt.month]
        day = dt.day
        hour = dt.hour
        ampm = "am" if hour < 12 else "pm"
        display_hour = hour % 12
        if display_hour == 0:
            display_hour = 12
        return f"{coin_name}-up-or-down-{month}-{day}-{display_hour}{ampm}-et"
    return f"{coin}-updown-{interval}-{candle_end_ts}"


def get_current_candle_end(interval: str) -> int:
    """Get the unix timestamp for the end of the CURRENT candle."""
    now = int(time.time())
    secs = INTERVAL_SECONDS[interval]
    return ((now // secs) + 1) * secs


def get_next_candle_end(interval: str) -> int:
    """Get the unix timestamp for the end of the NEXT candle."""
    now = int(time.time())
    secs = INTERVAL_SECONDS[interval]
    return ((now // secs) + 2) * secs


class PolymarketClient:
    """Client for Polymarket Gamma + CLOB APIs."""

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketCandleTrader/1.0"
        })
        self._market_cache: Dict[str, CandleContract] = {}

    def discover_contract(self, coin: str, interval: str,
                          candle_end_ts: Optional[int] = None) -> Optional[CandleContract]:
        """Discover a candle contract by coin, interval, and optional candle end timestamp."""
        if candle_end_ts is None:
            candle_end_ts = get_current_candle_end(interval)

        slug = build_slug(coin, interval, candle_end_ts)

        # Check cache
        if slug in self._market_cache:
            cached = self._market_cache[slug]
            if cached.condition_id:
                return cached

        try:
            r = self.session.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": slug},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()

            if not data:
                logger.warning(f"No market found for slug: {slug}")
                return None

            market = data[0]
            token_ids = json.loads(market.get("clobTokenIds", "[]"))
            prices = json.loads(market.get("outcomePrices", "[0.5, 0.5]"))

            contract = CandleContract(
                coin=coin,
                interval=interval,
                candle_end_ts=candle_end_ts,
                slug=slug,
                condition_id=market.get("conditionId", ""),
                yes_token_id=token_ids[0] if len(token_ids) > 0 else "",
                no_token_id=token_ids[1] if len(token_ids) > 1 else "",
                question=market.get("question", ""),
                yes_price=float(prices[0]) if len(prices) > 0 else 0.5,
                no_price=float(prices[1]) if len(prices) > 1 else 0.5,
                volume=float(market.get("volumeNum", 0)),
                active=market.get("active", True),
                resolved=market.get("closed", False),
            )

            self._market_cache[slug] = contract
            return contract

        except Exception as e:
            logger.error(f"Error discovering contract {slug}: {e}")
            return None

    def get_orderbook(self, contract: CandleContract,
                      side: str = "yes") -> Orderbook:
        """Get the CLOB orderbook for a contract's YES or NO token."""
        token_id = contract.yes_token_id if side == "yes" else contract.no_token_id
        if not token_id:
            return Orderbook()

        try:
            r = self.session.get(
                f"{CLOB_BASE}/book",
                params={"token_id": token_id},
                timeout=10
            )
            r.raise_for_status()
            data = r.json()

            bids = [OrderbookLevel(float(l["price"]), float(l["size"]))
                    for l in data.get("bids", [])]
            asks = [OrderbookLevel(float(l["price"]), float(l["size"]))
                    for l in data.get("asks", [])]
            ts = int(data.get("timestamp", 0))

            return Orderbook(bids=bids, asks=asks, timestamp=ts)

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return Orderbook()

    def get_price_history(self, condition_id: str,
                          interval: str = "1h",
                          fidelity: int = 60) -> List[Dict]:
        """Get historical price data for a market.

        interval: '1d', '1w', '1m' (month), 'max'
        fidelity: minutes between data points (min 10 for 1m range)
        """
        try:
            r = self.session.get(
                f"{CLOB_BASE}/prices-history",
                params={
                    "market": condition_id,
                    "interval": interval,
                    "fidelity": max(fidelity, 10),
                },
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                logger.error(f"Price history error: {data['error']}")
                return []
            return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Error fetching price history: {e}")
            return []

    def discover_all_current_contracts(self, interval: str,
                                       coins: Optional[List[str]] = None
                                       ) -> List[CandleContract]:
        """Discover all current candle contracts for an interval."""
        coins = coins or COINS
        candle_end = get_current_candle_end(interval)
        contracts = []
        for coin in coins:
            c = self.discover_contract(coin, interval, candle_end)
            if c:
                contracts.append(c)
        return contracts

    def get_resolved_contracts(self, coin: str, interval: str,
                                count: int = 20) -> List[CandleContract]:
        """Get recently resolved candle contracts for backtesting with real PM data."""
        contracts = []
        now = int(time.time())
        secs = INTERVAL_SECONDS[interval]
        # Go backwards from the most recently closed candle
        last_closed = (now // secs) * secs
        for i in range(count):
            ts = last_closed - (i * secs)
            c = self.discover_contract(coin, interval, ts)
            if c:
                contracts.append(c)
            time.sleep(0.1)  # Rate limiting
        return contracts

    def simulate_taker_fill(self, contract: CandleContract,
                             direction: str, size_usd: float
                             ) -> Dict:
        """Simulate a taker fill against the real orderbook.

        direction: 'yes' (bet UP) or 'no' (bet DOWN)
        size_usd: dollar amount to trade

        Returns dict with fill details:
          - side: 'HIT' (buying, crossing the ask) or 'LIFT' (selling, lifting the bid)
          - price: average fill price
          - shares: number of shares acquired
          - cost_usd: total cost in USD
          - max_payout: payout if correct ($1 per share)
          - fee: estimated Polymarket fee
        """
        book = self.get_orderbook(contract, side="yes")

        if direction == "yes":
            # Buying YES = HITTING the ask (taker buy)
            fill_price = book.taker_buy_price(size_usd / book.best_ask if book.best_ask > 0 else 0)
            shares = size_usd / fill_price if fill_price > 0 else 0
            side = "HIT"
        else:
            # Buying NO = effectively selling YES = LIFTING the bid
            # In Polymarket CLOB, buying NO at price p is same as selling YES at (1-p)
            no_book = self.get_orderbook(contract, side="no")
            fill_price = no_book.taker_buy_price(size_usd / no_book.best_ask if no_book.best_ask > 0 else 0)
            shares = size_usd / fill_price if fill_price > 0 else 0
            side = "HIT"  # Hitting the NO ask

        # Polymarket fee: ~2% * p * (1-p) at the traded price
        fee_rate = 0.02 * fill_price * (1 - fill_price)
        fee = size_usd * fee_rate

        return {
            "side": side,
            "direction": direction,
            "price": fill_price,
            "shares": shares,
            "cost_usd": size_usd,
            "max_payout": shares * 1.0,
            "fee": fee,
            "fee_rate": fee_rate,
            "spread": book.spread,
            "best_bid": book.best_bid,
            "best_ask": book.best_ask,
            "mid": book.mid,
        }


def compute_polymarket_fee(probability: float) -> float:
    """Compute Polymarket taker fee given a probability level.
    Fee = 2% * p * (1-p), peaking at 0.5% at p=0.5.
    """
    return 0.02 * probability * (1 - probability)


# ============================================================================
# HISTORICAL DATA COLLECTION
# ============================================================================

def fetch_historical_polymarket_data(coin: str, interval: str,
                                      hours_back: int = 24,
                                      client: Optional[PolymarketClient] = None
                                      ) -> List[Dict]:
    """Fetch historical contract data by iterating through past candle timestamps.

    Returns list of dicts with:
      - candle_end_ts, slug, condition_id
      - yes_price, no_price (final prices)
      - volume
      - resolved, outcome
    """
    client = client or PolymarketClient()
    secs = INTERVAL_SECONDS[interval]
    now = int(time.time())
    last_closed = (now // secs) * secs

    results = []
    num_candles = (hours_back * 3600) // secs

    for i in range(num_candles):
        ts = last_closed - (i * secs)
        slug = build_slug(coin, interval, ts)

        try:
            r = client.session.get(
                f"{GAMMA_BASE}/markets",
                params={"slug": slug},
                timeout=10
            )
            data = r.json()
            if data:
                m = data[0]
                prices = json.loads(m.get("outcomePrices", "[0.5, 0.5]"))
                results.append({
                    "candle_end_ts": ts,
                    "candle_end_dt": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                    "coin": coin,
                    "interval": interval,
                    "slug": slug,
                    "condition_id": m.get("conditionId", ""),
                    "yes_price": float(prices[0]) if prices else 0.5,
                    "no_price": float(prices[1]) if len(prices) > 1 else 0.5,
                    "volume": float(m.get("volumeNum", 0)),
                    "closed": m.get("closed", False),
                    "active": m.get("active", True),
                })
        except Exception as e:
            logger.warning(f"Error fetching {slug}: {e}")

        time.sleep(0.05)  # Rate limit: 20 req/sec

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = PolymarketClient()

    print("=" * 70)
    print("  POLYMARKET CRYPTO CANDLE API TEST")
    print("=" * 70)

    # Discover current contracts
    for interval in ["15m"]:
        contracts = client.discover_all_current_contracts(interval)
        for c in contracts:
            print(f"\n{c.coin.upper()} {c.interval}:")
            print(f"  Question: {c.question}")
            print(f"  Slug: {c.slug}")
            print(f"  YES price: ${c.yes_price:.3f}  NO price: ${c.no_price:.3f}")
            print(f"  Progress: {c.candle_progress:.0%}  Remaining: {c.time_remaining:.0f}s")

            # Get orderbook
            book = client.get_orderbook(c, side="yes")
            print(f"  Book: bid=${book.best_bid:.3f} / ask=${book.best_ask:.3f}  spread={book.spread:.3f}")
            print(f"  Depth: {len(book.bids)} bid levels, {len(book.asks)} ask levels")

            # Simulate a $250 taker buy
            fill = client.simulate_taker_fill(c, "yes", 250.0)
            print(f"  Sim $250 YES buy: price=${fill['price']:.4f}  "
                  f"shares={fill['shares']:.1f}  fee=${fill['fee']:.2f}")

    print("\nDone!")
