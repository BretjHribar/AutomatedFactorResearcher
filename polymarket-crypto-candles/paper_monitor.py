"""Paper-trading monitor — tracks LGB + MR model predictions vs actual outcomes.
No real money. Just logs predictions and checks accuracy."""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Import engines from live_trade_real
from live_trade_real import (
    LGBEngine, MREnsembleEngine, build_mr_ensemble, build_lgb_features,
    CONFIGS, KlineBuffer, BINANCE_TO_PM, SYMBOL_NAMES,
)

DATA_DIR = Path(__file__).parent / "data"
INTERVAL = "5m"
LOG_FILE = Path(__file__).parent / "paper_trades_v4.jsonl"

active_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def main():
    print("=" * 70)
    print("  PAPER MONITOR V4 — NO REAL TRADES")
    print("  Tracks LGB + MR predictions vs Binance candle outcomes")
    print("=" * 70)

    # Initialize engines
    engines = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        engine_type = cfg.get("engine", "mr")
        if engine_type == "lgb":
            engines[symbol] = LGBEngine(symbol, cfg)
            print(f"  {SYMBOL_NAMES[symbol]}: LGB (thresh={cfg['prob_threshold']})")
        else:
            engines[symbol] = MREnsembleEngine(symbol, cfg)
            print(f"  {SYMBOL_NAMES[symbol]}: MR ensemble (p{cfg['pctile']})")

    # Seed buffers
    buffers = {}
    for symbol in active_symbols:
        cfg = CONFIGS[symbol]
        engine_type = cfg.get("engine", "mr")
        n_seed = 10000 if engine_type == "lgb" else 2000
        buf = KlineBuffer(symbol, max_size=n_seed + 500)
        buf.seed_from_parquet(DATA_DIR / f"{symbol}_{INTERVAL}.parquet", n_bars=n_seed)
        buffers[symbol] = buf
        df = buf.to_dataframe()
        if len(df) > 100:
            engines[symbol].update(df)
            print(f"    Warmed {symbol} ({len(df)} bars)")

    # Stats
    results = {s: {"wins": 0, "losses": 0, "skips": 0, "predictions": []} for s in active_symbols}
    pending = {}  # symbol -> (direction, signal, proba, candle_open_time)

    print(f"\n  Monitoring started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Logging to {LOG_FILE}")
    print("-" * 70)

    import websockets

    streams = [f"{s.lower()}@kline_{INTERVAL}" for s in active_symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    async with websockets.connect(url, ping_interval=30) as ws:
        print("  [OK] Connected to Binance WebSocket")

        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                kl = msg.get("data", {}).get("k", {})
                if not kl or not kl.get("x"):  # Only process closed candles
                    continue

                sym = kl["s"]
                if sym not in active_symbols:
                    continue

                name = SYMBOL_NAMES[sym]
                candle_open = kl["t"]
                close_price = float(kl["c"])
                open_price = float(kl["o"])
                candle_up = close_price >= open_price

                # Check pending prediction for this symbol
                if sym in pending:
                    pred_dir, pred_sig, pred_proba, pred_time = pending[sym]
                    actual = "UP" if candle_up else "DOWN"
                    predicted = "UP" if pred_dir > 0 else "DOWN"
                    correct = (predicted == actual)

                    if correct:
                        results[sym]["wins"] += 1
                    else:
                        results[sym]["losses"] += 1

                    w = results[sym]["wins"]
                    l = results[sym]["losses"]
                    wr = w / (w + l) * 100 if (w + l) > 0 else 0

                    marker = "✓" if correct else "✗"
                    print(f"  {marker} {name:4s} predicted {predicted:4s}, actual {actual:4s} "
                          f"(p={pred_proba:.3f}) | WR: {w}/{w+l} = {wr:.1f}%")

                    # Log
                    with open(LOG_FILE, "a") as f:
                        f.write(json.dumps({
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "symbol": sym,
                            "predicted": predicted,
                            "actual": actual,
                            "correct": correct,
                            "proba": pred_proba,
                            "signal": pred_sig,
                            "wr": wr,
                            "n": w + l,
                        }) + "\n")

                    del pending[sym]

                # Update buffer with new candle
                buffers[sym].add_kline(kl)
                df = buffers[sym].to_dataframe()

                # Generate new prediction for NEXT candle
                direction, signal_val, info = engines[sym].update(df)
                proba = info.get("proba", 0) if isinstance(info, dict) else 0
                filtered = info.get("filtered", True) if isinstance(info, dict) else True

                if direction != 0 and not filtered:
                    pending[sym] = (direction, signal_val, proba, candle_open)
                    dir_str = "UP" if direction > 0 else "DOWN"
                    cfg = CONFIGS[sym]
                    etype = cfg.get("engine", "mr")
                    p_str = f" p={proba:.3f}" if etype == "lgb" else f" sig={signal_val:.2f}"
                    print(f"  → {name:4s} predicts {dir_str:4s} for next bar{p_str}")
                else:
                    results[sym]["skips"] += 1

                # Print summary every 12 bars (1 hour)
                total_bars = sum(results[s]["wins"] + results[s]["losses"] + results[s]["skips"]
                                 for s in active_symbols)
                if total_bars > 0 and total_bars % 36 == 0:  # Every ~12 bars per coin
                    print(f"\n  {'='*60}")
                    print(f"  SUMMARY after {total_bars} bars:")
                    for s in active_symbols:
                        r = results[s]
                        n = r["wins"] + r["losses"]
                        wr = r["wins"] / n * 100 if n > 0 else 0
                        trade_rate = n / (n + r["skips"]) * 100 if (n + r["skips"]) > 0 else 0
                        print(f"    {SYMBOL_NAMES[s]:4s}: {r['wins']}W {r['losses']}L = {wr:.1f}% WR "
                              f"({trade_rate:.0f}% trade rate, {r['skips']} skipped)")
                    total_w = sum(results[s]["wins"] for s in active_symbols)
                    total_l = sum(results[s]["losses"] for s in active_symbols)
                    total_wr = total_w / (total_w + total_l) * 100 if (total_w + total_l) > 0 else 0
                    print(f"    COMBINED: {total_w}W {total_l}L = {total_wr:.1f}% WR")
                    print(f"  {'='*60}\n")

            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("  [!] WebSocket disconnected, reconnecting...")
                await asyncio.sleep(5)
                break
            except Exception as e:
                print(f"  [!] Error: {e}")
                await asyncio.sleep(1)


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n  Stopped.")
            break
        except Exception as e:
            print(f"  [!] Fatal: {e}, restarting in 10s...")
            time.sleep(10)
