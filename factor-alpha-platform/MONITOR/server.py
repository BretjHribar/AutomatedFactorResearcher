"""
Live monitoring server for V10 Unified Paper Trader.

Reads signals from UNIFIED_V10/logs/signals/{SYM}.json,
fetches live Binance prices, computes mark-to-market PnL.

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

SIGNALS_DIR = Path(__file__).parent.parent / "UNIFIED_V10" / "logs" / "signals"
LOG_FILE = Path(__file__).parent.parent / "UNIFIED_V10" / "logs" / "paper_trader.log"
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

# MTM time series (in-memory, appended every ~5s)
_mtm_history = []  # list of {time: iso, symbols: {sym: total_bps}, portfolio: avg_bps}
_mtm_lock = threading.Lock()
MAX_MTM_POINTS = 10000  # ~14 hours at 5s intervals


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


def mtm_snapshotter():
    """Background thread to capture MTM snapshots every 5 seconds."""
    time.sleep(5)  # Wait for initial data
    while True:
        try:
            snapshot = compute_mtm_snapshot()
            if snapshot:
                with _mtm_lock:
                    _mtm_history.append(snapshot)
                    if len(_mtm_history) > MAX_MTM_POINTS:
                        _mtm_history.pop(0)
        except Exception as e:
            print(f"MTM snapshot error: {e}")
        time.sleep(5)


def compute_mtm_snapshot():
    """Compute current MTM PnL for all positions."""
    signals = read_signals()
    trades = read_trades()
    if not signals:
        return None

    with _price_lock:
        live_prices = dict(_price_cache)

    if not live_prices:
        return None

    # Get current state per symbol
    positions = {}
    for sig in signals:
        sym = sig['symbol']
        positions[sym] = {
            'direction': sig['direction'],
            'close_price': sig['close_price'],
            'cumulative_pnl_bps': sig.get('cumulative_pnl_bps', 0),
        }

    # Find entry prices from trades
    for trade in trades:
        sym = trade['symbol']
        if sym in positions and trade['new_pos'] != 0:
            positions[sym]['entry_price'] = trade['close_price']

    # Compute MTM
    sym_mtm = {}
    for sym, pos in positions.items():
        live_price = live_prices.get(sym, pos['close_price'])
        entry = pos.get('entry_price', pos['close_price'])
        direction = pos['direction']
        if direction == 1:
            unrealized = (live_price / entry - 1) * 10000
        elif direction == -1:
            unrealized = (1 - live_price / entry) * 10000
        else:
            unrealized = 0
        sym_mtm[sym] = round(pos['cumulative_pnl_bps'] + unrealized, 2)

    if not sym_mtm:
        return None

    return {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
        'symbols': sym_mtm,
        'portfolio': round(sum(sym_mtm.values()) / len(sym_mtm), 2),
        'portfolio_sum': round(sum(sym_mtm.values()), 2),
    }


def read_signals():
    """Read all signals from UNIFIED_V10 per-symbol JSON files."""
    signals = []
    if SIGNALS_DIR.exists():
        for sig_file in sorted(SIGNALS_DIR.glob('*.json')):
            try:
                data = json.loads(sig_file.read_text())
                if isinstance(data, list):
                    signals.extend(data)
                elif isinstance(data, dict):
                    signals.append(data)
            except (json.JSONDecodeError, Exception):
                pass
    # Sort by timestamp
    signals.sort(key=lambda s: s.get('timestamp', ''))
    return signals


def read_trades():
    """Read trades — in V10, trades are part of signal events (direction changes)."""
    signals = read_signals()
    trades = []
    prev_dir = {}
    for sig in signals:
        sym = sig['symbol']
        old_dir = prev_dir.get(sym, 0)
        new_dir = sig['direction']
        if old_dir != new_dir:
            trades.append({
                'symbol': sym,
                'timestamp': sig['timestamp'],
                'old_pos': old_dir,
                'new_pos': new_dir,
                'close_price': sig['close_price'],
            })
        prev_dir[sym] = new_dir
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
            'cumulative_pnl_bps': sig.get('cumulative_pnl', sig.get('cumulative_pnl_bps', 0)) * (10000 if 'cumulative_pnl' in sig else 1),
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
                cum_bps = sig.get('cumulative_pnl', sig.get('cumulative_pnl_bps', 0))
                if 'cumulative_pnl' in sig:
                    cum_bps = cum_bps * 10000  # Convert fractional to bps
                curve.append({
                    'time': sig['timestamp'],
                    'realized_bps': cum_bps,
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
                    cum = sig.get('cumulative_pnl', sig.get('cumulative_pnl_bps', 0))
                    if 'cumulative_pnl' in sig:
                        cum = cum * 10000
                    vals.append(cum)
                    break
        if vals:
            portfolio_curve.append({
                'time': t,
                'realized_bps': round(sum(vals) / len(vals), 2),
            })

    # Get MTM history for live charts
    with _mtm_lock:
        mtm_series = list(_mtm_history)

    return {
        'positions': mtm_positions,
        'equity_curves': equity_curves,
        'portfolio_curve': portfolio_curve,
        'mtm_series': mtm_series,
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

    # Start MTM snapshotter thread
    mtm_thread = threading.Thread(target=mtm_snapshotter, daemon=True)
    mtm_thread.start()

    # Initial price fetch
    fetch_live_prices()
    time.sleep(1)

    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    print(f"V10 Monitor running at http://localhost:{PORT}")
    print(f"Reading signals from: {SIGNALS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
