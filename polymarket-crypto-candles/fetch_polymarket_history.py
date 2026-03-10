"""
fetch_polymarket_history.py — Fetch historical Polymarket crypto candle contract data.

Downloads resolved contract data (prices, volumes, outcomes) by iterating
through past candle timestamps and querying the Gamma API.

Saves to SQLite for use in the backtester.

Usage:
    python fetch_polymarket_history.py                      # Last 24h, 15m
    python fetch_polymarket_history.py --hours 168 --interval 15m   # 1 week
    python fetch_polymarket_history.py --hours 720 --interval 5m    # 30 days
"""
import sys, os, json, time, sqlite3, argparse
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polymarket_api import (
    GAMMA_BASE, CLOB_BASE, COINS, INTERVAL_SECONDS,
    build_slug, PolymarketClient
)

DB_DIR = Path(__file__).parent / "data"
PM_DB_PATH = DB_DIR / "polymarket_history.db"


def init_db():
    """Create the Polymarket history database."""
    DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(PM_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candle_contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            interval TEXT NOT NULL,
            candle_end_ts INTEGER NOT NULL,
            candle_end_dt TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            condition_id TEXT,
            yes_token_id TEXT,
            no_token_id TEXT,
            question TEXT,
            yes_price REAL,
            no_price REAL,
            volume REAL,
            closed INTEGER DEFAULT 0,
            outcome TEXT,
            best_bid REAL,
            best_ask REAL,
            spread REAL,
            liquidity REAL,
            fee_rate REAL,
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_candle_coin_interval_ts
        ON candle_contracts(coin, interval, candle_end_ts)
    """)
    conn.commit()
    return conn


def fetch_single_contract(session, coin, interval, candle_end_ts):
    """Fetch a single contract's data from the Gamma API."""
    slug = build_slug(coin, interval, candle_end_ts)

    try:
        r = session.get(
            f"{GAMMA_BASE}/markets",
            params={"slug": slug},
            timeout=15
        )
        if r.status_code != 200:
            return None

        data = r.json()
        if not data:
            return None

        m = data[0]
        prices = json.loads(m.get("outcomePrices", "[0.5, 0.5]"))
        token_ids = json.loads(m.get("clobTokenIds", "[]"))

        # Determine outcome from prices (resolved contracts have extreme prices)
        yes_p = float(prices[0]) if prices else 0.5
        no_p = float(prices[1]) if len(prices) > 1 else 0.5
        closed = m.get("closed", False)

        outcome = None
        if closed:
            if yes_p > 0.95:
                outcome = "Up"
            elif no_p > 0.95:
                outcome = "Down"

        # Fee at entry price (would have been near 0.50)
        fee_rate = 0.02 * 0.5 * 0.5  # 0.5% at p=0.50

        return {
            "coin": coin,
            "interval": interval,
            "candle_end_ts": candle_end_ts,
            "candle_end_dt": datetime.fromtimestamp(candle_end_ts, tz=timezone.utc).isoformat(),
            "slug": slug,
            "condition_id": m.get("conditionId", ""),
            "yes_token_id": token_ids[0] if len(token_ids) > 0 else "",
            "no_token_id": token_ids[1] if len(token_ids) > 1 else "",
            "question": m.get("question", ""),
            "yes_price": yes_p,
            "no_price": no_p,
            "volume": float(m.get("volumeNum", 0)),
            "closed": 1 if closed else 0,
            "outcome": outcome,
            "best_bid": float(m.get("bestBid", 0)),
            "best_ask": float(m.get("bestAsk", 0)),
            "spread": float(m.get("spread", 0)),
            "liquidity": float(m.get("liquidityNum", 0)),
            "fee_rate": fee_rate,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        return None


def fetch_history(coins=None, interval="15m", hours_back=24,
                  max_workers=5, progress=True):
    """Fetch historical Polymarket data and save to DB."""
    coins = coins or COINS
    secs = INTERVAL_SECONDS[interval]
    now = int(time.time())
    last_closed = (now // secs) * secs

    num_candles = (hours_back * 3600) // secs
    total = num_candles * len(coins)

    conn = init_db()

    # Check what we already have
    existing = set()
    for row in conn.execute("SELECT slug FROM candle_contracts"):
        existing.add(row[0])

    print(f"{'='*60}")
    print(f"POLYMARKET HISTORICAL DATA FETCHER")
    print(f"{'='*60}")
    print(f"Coins: {', '.join(c.upper() for c in coins)}")
    print(f"Interval: {interval}")
    print(f"Period: last {hours_back} hours ({num_candles} candles per coin)")
    print(f"Total contracts: {total}")
    print(f"Already in DB: {len(existing)}")
    print()

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "PolymarketHistoryFetcher/1.0"
    })

    fetched = 0
    skipped = 0
    errors = 0

    # Build task list
    tasks = []
    for coin in coins:
        for i in range(num_candles):
            ts = last_closed - (i * secs)
            slug = build_slug(coin, interval, ts)
            if slug in existing:
                skipped += 1
                continue
            tasks.append((coin, interval, ts))

    print(f"New contracts to fetch: {len(tasks)} (skipping {skipped} already in DB)")

    # Fetch with rate limiting
    batch_size = 10
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_single_contract, session, coin, intv, ts): (coin, intv, ts)
                for coin, intv, ts in batch
            }

            for future in as_completed(futures):
                result = future.result()
                if result:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO candle_contracts
                            (coin, interval, candle_end_ts, candle_end_dt, slug,
                             condition_id, yes_token_id, no_token_id, question,
                             yes_price, no_price, volume, closed, outcome,
                             best_bid, best_ask, spread, liquidity,
                             fee_rate, fetched_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            result["coin"], result["interval"],
                            result["candle_end_ts"], result["candle_end_dt"],
                            result["slug"], result["condition_id"],
                            result["yes_token_id"], result["no_token_id"],
                            result["question"],
                            result["yes_price"], result["no_price"],
                            result["volume"], result["closed"],
                            result["outcome"],
                            result["best_bid"], result["best_ask"],
                            result["spread"], result["liquidity"],
                            result["fee_rate"], result["fetched_at"],
                        ))
                        fetched += 1
                    except sqlite3.IntegrityError:
                        skipped += 1
                else:
                    errors += 1

        conn.commit()

        if progress:
            done = batch_start + len(batch)
            pct = done / max(len(tasks), 1) * 100
            print(f"\r  Progress: {done}/{len(tasks)} ({pct:.0f}%) "
                  f"| Fetched: {fetched} | Errors: {errors}", end="", flush=True)

        # Rate limit: ~10 requests per second
        time.sleep(1.0)

    conn.commit()
    print(f"\n\nDone! Fetched: {fetched}, Skipped: {skipped}, Errors: {errors}")

    # Summary
    total_rows = conn.execute("SELECT COUNT(*) FROM candle_contracts").fetchone()[0]
    resolved = conn.execute("SELECT COUNT(*) FROM candle_contracts WHERE closed=1").fetchone()[0]
    print(f"\nDatabase: {PM_DB_PATH}")
    print(f"Total records: {total_rows}")
    print(f"Resolved (closed): {resolved}")

    # Per-coin stats
    for coin in coins:
        count = conn.execute(
            "SELECT COUNT(*) FROM candle_contracts WHERE coin=? AND interval=?",
            (coin, interval)
        ).fetchone()[0]
        up = conn.execute(
            "SELECT COUNT(*) FROM candle_contracts WHERE coin=? AND interval=? AND outcome='Up'",
            (coin, interval)
        ).fetchone()[0]
        down = conn.execute(
            "SELECT COUNT(*) FROM candle_contracts WHERE coin=? AND interval=? AND outcome='Down'",
            (coin, interval)
        ).fetchone()[0]
        print(f"  {coin.upper()}: {count} contracts ({up} Up, {down} Down, "
              f"{count-up-down} {'pending' if count-up-down > 0 else ''})")

    conn.close()


def load_polymarket_history(coin: str, interval: str) -> list:
    """Load historical Polymarket data from the SQLite DB.

    Returns list of dicts with keys:
        candle_end_ts, yes_price, no_price, volume, outcome, spread, etc.
    """
    if not PM_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(PM_DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM candle_contracts
        WHERE coin=? AND interval=? AND closed=1
        ORDER BY candle_end_ts ASC
    """, (coin, interval)).fetchall()
    conn.close()

    return [dict(r) for r in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Polymarket historical data")
    parser.add_argument("--hours", type=int, default=48,
                        help="Hours of history to fetch (default: 48)")
    parser.add_argument("--interval", type=str, default="15m",
                        choices=["5m", "15m", "1h"],
                        help="Candle interval (default: 15m)")
    parser.add_argument("--coins", type=str, nargs="+", default=None,
                        help="Coins to fetch (default: btc eth sol)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent workers (default: 5)")
    args = parser.parse_args()

    fetch_history(
        coins=args.coins,
        interval=args.interval,
        hours_back=args.hours,
        max_workers=args.workers,
    )
