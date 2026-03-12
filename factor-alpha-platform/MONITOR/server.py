"""
Live monitoring server for V9b Paper Trader.

Reads signals.jsonl from PAPER_TRADER/logs, fetches live Binance prices,
computes mark-to-market PnL, and serves a dashboard.

Completely independent of the paper trader process.

Usage:
    python server.py
    Then open http://localhost:8877
"""
import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import urllib.request
import os

SIGNALS_FILE = Path(__file__).parent.parent / "PAPER_TRADER" / "logs" / "signals.jsonl"
TRADES_FILE = Path(__file__).parent.parent / "PAPER_TRADER" / "logs" / "trades.jsonl"
LOG_FILE = Path(__file__).parent.parent / "PAPER_TRADER" / "logs" / "paper_trader.log"
PORT = 8877

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
SYMBOL_COLORS = {
    'BTCUSDT': '#F7931A', 'ETHUSDT': '#627EEA', 'SOLUSDT': '#9945FF',
    'BNBUSDT': '#F3BA2F', 'DOGEUSDT': '#C2A633',
    'XRPUSDT': '#00AAE4', 'ADAUSDT': '#0033AD', 'AVAXUSDT': '#E84142',
    'LINKUSDT': '#2A5ADA', 'LTCUSDT': '#BFBBBB',
}

# Cache for live prices
_price_cache = {}
_price_lock = threading.Lock()


def fetch_live_prices():
    """Fetch latest prices from Binance Futures REST API."""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        with _price_lock:
            for item in data:
                _price_cache[item['symbol']] = float(item['price'])
    except Exception as e:
        print(f"Price fetch error: {e}")


def price_updater():
    """Background thread to update prices every 3 seconds."""
    while True:
        fetch_live_prices()
        time.sleep(3)


def read_signals():
    """Read all signals from the JSONL file."""
    signals = []
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        signals.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return signals


def read_trades():
    """Read all trades from the JSONL file."""
    trades = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return trades


def compute_mtm_state():
    """Compute mark-to-market state for all positions."""
    signals = read_signals()
    trades = read_trades()

    # Get current positions per symbol
    positions = {}
    signal_history = {}

    for sig in signals:
        sym = sig['symbol']
        if sym not in signal_history:
            signal_history[sym] = []
        signal_history[sym].append(sig)

        positions[sym] = {
            'direction': sig['direction'],
            'signal_value': sig.get('signal_value', 0),
            'entry_price': None,
            'cumulative_pnl_bps': sig.get('cumulative_pnl_bps', 0),
            'timestamp': sig['timestamp'],
            'close_price': sig['close_price'],
        }

    # Find entry prices from trades
    for trade in trades:
        sym = trade['symbol']
        if sym in positions and trade['new_pos'] != 0:
            positions[sym]['entry_price'] = trade['close_price']

    # Mark-to-market with live prices
    with _price_lock:
        live_prices = dict(_price_cache)

    mtm_positions = {}
    for sym, pos in positions.items():
        live_price = live_prices.get(sym, pos['close_price'])
        entry = pos['entry_price'] or pos['close_price']
        direction = pos['direction']

        # Unrealized PnL
        if direction == 1:
            unrealized_bps = (live_price / entry - 1) * 10000
        elif direction == -1:
            unrealized_bps = (1 - live_price / entry) * 10000
        else:
            unrealized_bps = 0

        # Total = realized (from signal log) + unrealized
        # The cumulative_pnl_bps in signals includes all REALIZED trades
        # We need to add current unrealized
        realized_bps = pos['cumulative_pnl_bps']
        if direction != 0:
            # Subtract the entry fee that was already counted
            total_bps = realized_bps + unrealized_bps
        else:
            total_bps = realized_bps

        mtm_positions[sym] = {
            'direction': direction,
            'dir_str': 'LONG' if direction > 0 else 'SHORT' if direction < 0 else 'FLAT',
            'signal_value': pos['signal_value'],
            'entry_price': entry,
            'live_price': live_price,
            'realized_bps': round(realized_bps, 2),
            'unrealized_bps': round(unrealized_bps, 2),
            'total_bps': round(total_bps, 2),
            'last_signal_time': pos['timestamp'],
        }

    # Build equity curve data
    equity_curves = {}
    for sym, hist in signal_history.items():
        curve = []
        for sig in hist:
            curve.append({
                'time': sig['timestamp'],
                'realized_bps': sig.get('cumulative_pnl_bps', 0),
            })
        equity_curves[sym] = curve

    # Portfolio equity curve (average of all symbols)
    all_times = sorted(set(
        sig['timestamp'] for sigs in signal_history.values() for sig in sigs
    ))
    portfolio_curve = []
    for t in all_times:
        vals = []
        for sym in signal_history:
            # Find latest signal at or before time t
            for sig in reversed(signal_history[sym]):
                if sig['timestamp'] <= t:
                    vals.append(sig.get('cumulative_pnl_bps', 0))
                    break
        if vals:
            portfolio_curve.append({
                'time': t,
                'realized_bps': round(sum(vals) / len(vals), 2),
            })

    return {
        'positions': mtm_positions,
        'equity_curves': equity_curves,
        'portfolio_curve': portfolio_curve,
        'live_prices': {s: live_prices.get(s) for s in SYMBOLS},
        'n_signals': len(signals),
        'n_trades': len(trades),
        'server_time': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve dashboard HTML and API endpoints."""

    def do_GET(self):
        if self.path == '/api/state':
            state = compute_mtm_state()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html_path = Path(__file__).parent / 'index.html'
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logs


if __name__ == '__main__':
    # Start price updater thread
    price_thread = threading.Thread(target=price_updater, daemon=True)
    price_thread.start()

    # Initial price fetch
    fetch_live_prices()
    time.sleep(1)

    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    print(f"V9b Monitor running at http://localhost:{PORT}")
    print(f"Reading signals from: {SIGNALS_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
