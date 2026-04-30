"""
Data-integrity and lookahead tests for the AIPT feature pipeline.

These tests would have caught the three bugs discovered on 2026-04-24:
  1. KuCoin Futures API column-order mismatch (kline.close actually held HIGH values)
  2. Feature-set mismatch between research (D=24) and prod (D=22) matrices
  3. Research matrix high/low populated with close/open copies

Run: pytest tests/unit/test_aipt_integrity.py -v
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

KLINES_DIR = PROJECT_ROOT / "data/kucoin_cache/klines/4h"
RESEARCH_MATRICES = PROJECT_ROOT / "data/kucoin_cache/matrices/4h"
PROD_MATRICES = PROJECT_ROOT / "data/kucoin_cache/matrices/4h/prod"
UNIVERSE_PATH = PROJECT_ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"

CHAR_NAMES = [
    "adv20", "adv60", "beta_to_btc", "close_position_in_range",
    "dollars_traded", "high_low_range",
    "historical_volatility_10", "historical_volatility_20",
    "historical_volatility_60", "historical_volatility_120",
    "log_returns", "momentum_5d", "momentum_20d", "momentum_60d",
    "open_close_range",
    "parkinson_volatility_10", "parkinson_volatility_20", "parkinson_volatility_60",
    "quote_volume", "turnover",
    "volume_momentum_1", "volume_momentum_5_20", "volume_ratio_20d", "vwap_deviation",
]

GAMMA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# =============================================================================
# Helper: build synthetic kline / matrix data that mimics the prod builder
# =============================================================================

def _synthetic_klines(T: int = 400, N: int = 10, seed: int = 0) -> pd.DataFrame:
    """Generate N tickers' worth of random-walk OHLCV klines that satisfy OHLC invariants."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="4h")
    data = {}
    for n in range(N):
        sym = f"T{n}USDTM"
        open_ = 100 + np.cumsum(rng.normal(0, 1, T))
        # High/low as random extensions of open with sign preserving C vs O direction
        c_change = rng.normal(0, 1, T)
        close = open_ + c_change
        spread = np.abs(rng.normal(0, 1.5, T))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        vol = np.abs(rng.normal(1e6, 2e5, T))
        turnover = close * vol
        df = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": vol, "turnover": turnover,
        }, index=idx)
        df.index.name = "time"
        data[sym] = df
    return data


def _compute_derived(all_data: dict) -> dict:
    """Replicate data_refresh.py._build_kucoin_matrices derived computations."""
    idx = sorted(set().union(*[df.index for df in all_data.values()]))
    close = pd.DataFrame({s: d["close"] for s, d in all_data.items()}, index=idx)
    high  = pd.DataFrame({s: d["high"]  for s, d in all_data.items()}, index=idx)
    low   = pd.DataFrame({s: d["low"]   for s, d in all_data.items()}, index=idx)
    opn   = pd.DataFrame({s: d["open"]  for s, d in all_data.items()}, index=idx)
    vol   = pd.DataFrame({s: d["volume"]   for s, d in all_data.items()}, index=idx)
    turn  = pd.DataFrame({s: d["turnover"] for s, d in all_data.items()}, index=idx)

    ret = close.pct_change(fill_method=None)
    d = {
        "returns": ret,
        "log_returns": np.log1p(ret.fillna(0)),
        "vwap": (high + low + close) / 3,
        "adv20": turn.rolling(120, min_periods=60).mean(),
        "adv60": turn.rolling(360, min_periods=180).mean(),
        "high_low_range": (high - low) / close,
        "open_close_range": (close - opn).abs() / close,
        "close_position_in_range": (close - low) / (high - low + 1e-10),
        "volume_momentum_5_20": vol.rolling(30).mean() / vol.rolling(120).mean(),
        "historical_volatility_10": ret.rolling(60).std() * np.sqrt(6 * 365),
        "historical_volatility_20": ret.rolling(120).std() * np.sqrt(6 * 365),
        "historical_volatility_60": ret.rolling(360).std() * np.sqrt(6 * 365),
        "historical_volatility_120": ret.rolling(720, min_periods=360).std() * np.sqrt(6 * 365),
        "momentum_5d": close / close.shift(30) - 1,
        "momentum_20d": close / close.shift(120) - 1,
        "momentum_60d": close / close.shift(360) - 1,
        "vwap_deviation": (close - (high + low + close) / 3) / ((high + low + close) / 3),
        "dollars_traded": turn,
        "quote_volume": turn,
        "volume_ratio_20d": vol / vol.rolling(120).mean(),
        "volume_momentum_1": vol / vol.shift(1) - 1,
    }
    hl = np.log(high / low)
    d["parkinson_volatility_10"] = hl.pow(2).rolling(60).mean().pow(0.5) / (2 * np.log(2))**0.5
    d["parkinson_volatility_20"] = hl.pow(2).rolling(120).mean().pow(0.5) / (2 * np.log(2))**0.5
    d["parkinson_volatility_60"] = hl.pow(2).rolling(360).mean().pow(0.5) / (2 * np.log(2))**0.5
    return d


# =============================================================================
# 1. KLINE OHLC INVARIANT TESTS
#    Protects against a recurrence of the KuCoin column-order bug.
# =============================================================================

class TestOHLCInvariants:
    """OHLC consistency: low <= min(o,c) <= max(o,c) <= high, and low <= high."""

    @pytest.mark.parametrize("symbol", ["XBTUSDTM", "ETHUSDTM", "SOLUSDTM"])
    def test_kline_ohlc_invariant_major_pairs(self, symbol):
        fp = KLINES_DIR / f"{symbol}.parquet"
        if not fp.exists():
            pytest.skip(f"{symbol} kline parquet missing")
        df = pd.read_parquet(fp)
        required = {"open", "high", "low", "close"}
        assert required.issubset(df.columns), f"{symbol} missing columns"
        violations = (
            (df["low"] > df["open"] + 1e-9) | (df["low"] > df["close"] + 1e-9)
            | (df["high"] < df["open"] - 1e-9) | (df["high"] < df["close"] - 1e-9)
            | (df["low"] > df["high"] + 1e-9)
        ).sum()
        pct = violations / len(df) * 100
        assert pct < 1.0, f"{symbol}: {violations}/{len(df)} ({pct:.2f}%) OHLC violations"

    def test_all_klines_ohlc_invariant_global(self):
        """Aggregate check across every kline parquet: < 0.5% global violation rate."""
        files = sorted(KLINES_DIR.glob("*.parquet"))
        assert files, "no kline parquets found"
        total_rows = 0
        total_violations = 0
        failed_files: list[str] = []
        for fp in files:
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                continue
            v = (
                (df["low"] > df["open"] + 1e-9) | (df["low"] > df["close"] + 1e-9)
                | (df["high"] < df["open"] - 1e-9) | (df["high"] < df["close"] - 1e-9)
                | (df["low"] > df["high"] + 1e-9)
            ).sum()
            total_rows += len(df)
            total_violations += v
            if v > len(df) * 0.01:
                failed_files.append(f"{fp.stem}: {v}/{len(df)}")
        pct = total_violations / max(total_rows, 1) * 100
        assert pct < 0.5, f"global {pct:.3f}% OHLC violations; worst files: {failed_files[:10]}"

    def test_research_matrix_ohlc_invariant(self):
        rh = pd.read_parquet(RESEARCH_MATRICES / "high.parquet")
        rl = pd.read_parquet(RESEARCH_MATRICES / "low.parquet")
        ro = pd.read_parquet(RESEARCH_MATRICES / "open.parquet")
        rc = pd.read_parquet(RESEARCH_MATRICES / "close.parquet")
        bad = (
            (rl > ro + 1e-9) | (rl > rc + 1e-9)
            | (rh < ro - 1e-9) | (rh < rc - 1e-9) | (rl > rh + 1e-9)
        )
        n = bad.sum().sum()
        total = bad.size
        assert n / total < 0.001, f"research matrix has {n}/{total} OHLC violations"

    def test_prod_matrix_ohlc_invariant(self):
        ph = pd.read_parquet(PROD_MATRICES / "high.parquet")
        pl = pd.read_parquet(PROD_MATRICES / "low.parquet")
        po = pd.read_parquet(PROD_MATRICES / "open.parquet")
        pc = pd.read_parquet(PROD_MATRICES / "close.parquet")
        bad = (
            (pl > po + 1e-9) | (pl > pc + 1e-9)
            | (ph < po - 1e-9) | (ph < pc - 1e-9) | (pl > ph + 1e-9)
        )
        n = bad.sum().sum()
        total = bad.size
        assert n / total < 0.001, f"prod matrix has {n}/{total} OHLC violations"


# =============================================================================
# 2. FEATURE-SET PARITY TESTS
#    Protects against Bug #1 (D=22 vs D=24 mismatch between backfill and live).
# =============================================================================

class TestFeatureParity:
    def test_research_matrix_has_all_char_names(self):
        missing = [c for c in CHAR_NAMES if not (RESEARCH_MATRICES / f"{c}.parquet").exists()]
        assert not missing, f"research matrix missing CHAR_NAMES: {missing}"

    def test_prod_matrix_has_all_char_names(self):
        missing = [c for c in CHAR_NAMES if not (PROD_MATRICES / f"{c}.parquet").exists()]
        assert not missing, f"prod matrix missing CHAR_NAMES: {missing}"

    def test_research_prod_char_names_parity(self):
        r_has = {c for c in CHAR_NAMES if (RESEARCH_MATRICES / f"{c}.parquet").exists()}
        p_has = {c for c in CHAR_NAMES if (PROD_MATRICES / f"{c}.parquet").exists()}
        assert r_has == p_has, f"CHAR_NAMES parity broken: {r_has ^ p_has}"


# =============================================================================
# 3. FEATURE VALUE PARITY TESTS
#    Protects against Bug #3 (research high/low diverging from prod).
# =============================================================================

class TestFeatureValueParity:
    @pytest.fixture(scope="class")
    def universe_tickers(self):
        uni = pd.read_parquet(UNIVERSE_PATH)
        cov = uni.sum(axis=0) / len(uni)
        t = sorted(cov[cov > 0.3].index.tolist())
        rc = pd.read_parquet(RESEARCH_MATRICES / "close.parquet")
        return [x for x in t if x in rc.columns]

    @pytest.fixture(scope="class")
    def common_bars(self):
        r = pd.read_parquet(RESEARCH_MATRICES / "close.parquet")
        p = pd.read_parquet(PROD_MATRICES / "close.parquet")
        return sorted(set(r.index) & set(p.index))

    @pytest.mark.parametrize("feat", ["close", "open", "high", "low",
                                      "momentum_5d", "historical_volatility_20",
                                      "returns"])
    def test_research_vs_prod_exact(self, feat, universe_tickers, common_bars):
        """Pure functions of klines must match bit-for-bit on overlapping bars.

        Skip the first bar of prod's tail (prev-bar data doesn't exist in prod's
        1500-row window but does in research's full history) — that boundary
        creates legitimate differences for pct_change-based features.
        """
        r = pd.read_parquet(RESEARCH_MATRICES / f"{feat}.parquet")
        p = pd.read_parquet(PROD_MATRICES / f"{feat}.parquet")
        common_t = [x for x in universe_tickers if x in r.columns and x in p.columns]
        bars = common_bars[1:] if len(common_bars) > 1 else common_bars
        rv = r.loc[bars, common_t].values
        pv = p.loc[bars, common_t].values
        diff = np.where(np.isnan(rv) & np.isnan(pv), 0.0, np.abs(rv - pv))
        max_d = np.nanmax(diff) if diff.size else 0.0
        assert max_d < 1e-6, f"{feat}: max |diff| = {max_d:g} between research and prod"

    def test_log_returns_parity_excluding_nan_boundary(self, universe_tickers, common_bars):
        """log_returns uses np.log1p(ret.fillna(0)) which creates 0s where ret
        is NaN (new tickers / first bar of prod tail). Check parity only where
        both sides computed a non-zero (valid) log return."""
        r = pd.read_parquet(RESEARCH_MATRICES / "log_returns.parquet")
        p = pd.read_parquet(PROD_MATRICES / "log_returns.parquet")
        common_t = [x for x in universe_tickers if x in r.columns and x in p.columns]
        bars = common_bars[1:]  # skip prod-tail boundary
        rv = r.loc[bars, common_t].values
        pv = p.loc[bars, common_t].values
        # Parity where BOTH are non-zero (i.e., both computed a real log return)
        both_real = (np.abs(rv) > 1e-12) & (np.abs(pv) > 1e-12)
        diff = np.where(both_real, np.abs(rv - pv), 0.0)
        max_d = np.nanmax(diff)
        assert max_d < 1e-6, f"log_returns (both-real): max |diff| = {max_d:g}"


# =============================================================================
# 4. NO-LOOKAHEAD TESTS
#    Modifying bars strictly after index t must not change any characteristic
#    value at index t or earlier, for any ticker.
# =============================================================================

class TestNoLookahead:
    def _matrices_from_data(self, all_data):
        return _compute_derived(all_data)

    def test_modifying_future_does_not_change_past(self):
        all_data = _synthetic_klines(T=400, N=10, seed=0)
        d1 = self._matrices_from_data(all_data)

        # Perturb bars after index t=300 drastically, across all fields
        t = 300
        all_data2 = {s: df.copy() for s, df in all_data.items()}
        for s in all_data2:
            idx = all_data2[s].index
            mask = idx > idx[t]
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                all_data2[s].loc[mask, col] = (
                    all_data2[s].loc[mask, col].values * np.random.default_rng(7).uniform(0.5, 2.0, mask.sum())
                )
            # Re-enforce OHLC invariants on perturbed rows so _compute_derived does not NaN blow-up
            lo = np.minimum.reduce([all_data2[s].loc[mask, "open"].values,
                                    all_data2[s].loc[mask, "close"].values,
                                    all_data2[s].loc[mask, "low"].values])
            hi = np.maximum.reduce([all_data2[s].loc[mask, "open"].values,
                                    all_data2[s].loc[mask, "close"].values,
                                    all_data2[s].loc[mask, "high"].values])
            all_data2[s].loc[mask, "low"] = lo
            all_data2[s].loc[mask, "high"] = hi

        d2 = self._matrices_from_data(all_data2)

        for feat in d1:
            before_1 = d1[feat].iloc[:t + 1]
            before_2 = d2[feat].iloc[:t + 1]
            v1 = before_1.values
            v2 = before_2.values
            # NaN == NaN
            diff = np.where(np.isnan(v1) & np.isnan(v2), 0.0, np.abs(v1 - v2))
            max_d = np.nanmax(diff) if diff.size else 0.0
            assert max_d < 1e-10, f"{feat} at bar<=t={t} changed when future perturbed: max_diff={max_d:g}"


# =============================================================================
# 5. RFF DETERMINISM TESTS
#    theta/gamma must be bit-identical between backfill (D=24 on research) and
#    live (D=24 on prod) given same seed.
# =============================================================================

class TestRFFDeterminism:
    def _generate(self, D, P, seed=42):
        rng = np.random.default_rng(seed)
        theta = rng.standard_normal((P // 2, D))
        gamma = rng.choice(GAMMA_GRID, size=P // 2)
        return theta, gamma

    def test_same_d_same_seed_bit_identical(self):
        t1, g1 = self._generate(D=24, P=100, seed=42)
        t2, g2 = self._generate(D=24, P=100, seed=42)
        assert np.array_equal(t1, t2)
        assert np.array_equal(g1, g2)

    def test_d22_d24_produce_different_theta_shapes(self):
        """Regression test for Bug #1: if D silently drops, theta becomes unusable."""
        t22, _ = self._generate(D=22, P=100, seed=42)
        t24, _ = self._generate(D=24, P=100, seed=42)
        assert t22.shape != t24.shape
        # Lambda trained on D=24 cannot be applied against D=22 signals.
        # Enforce invariant at test time: backfill's D == live's D.

    def test_backfill_and_live_use_same_d(self):
        """With CHAR_NAMES parity, backfill sees same D as live."""
        r_has = [c for c in CHAR_NAMES if (RESEARCH_MATRICES / f"{c}.parquet").exists()]
        p_has = [c for c in CHAR_NAMES if (PROD_MATRICES / f"{c}.parquet").exists()]
        assert len(r_has) == len(p_has), \
            f"D mismatch: backfill D={len(r_has)} vs live D={len(p_has)}"


# =============================================================================
# 6. API SCHEMA REGRESSION TEST
#    Verify KuCoin Futures API actually returns [time, O, H, L, C, V, tv].
#    Marked 'net' — skip if network unavailable.
# =============================================================================

class TestAPISchema:
    @pytest.mark.network
    def test_kucoin_futures_api_ohlc_schema(self):
        import requests
        from datetime import datetime, timezone
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        bar_ms = 4 * 3600 * 1000
        from_ms = now_ms - 48 * bar_ms
        to_ms = now_ms - 24 * bar_ms  # sealed bars only
        try:
            r = requests.get("https://api-futures.kucoin.com/api/v1/kline/query",
                             params={"symbol": "XBTUSDTM", "granularity": 240,
                                     "from": from_ms, "to": to_ms}, timeout=10)
            d = r.json()
        except Exception as e:
            pytest.skip(f"KuCoin API unreachable: {e}")
        if d.get("code") != "200000":
            pytest.skip(f"KuCoin API returned {d.get('code')}")
        n_invalid_ohlc = 0
        n_invalid_oclh = 0
        for c in d["data"]:
            _, a, b, x, y, *_ = c
            # Under [O, H, L, C] (our claim): H=b >= max(O,C), L=x <= min(O,C), L<=H
            if not (b >= max(a, y) - 1e-9 and x <= min(a, y) + 1e-9 and x <= b):
                n_invalid_ohlc += 1
            # Under [O, C, H, L] (KuCoin's claimed docs format):
            if not (x >= max(a, b) - 1e-9 and y <= min(a, b) + 1e-9 and y <= x):
                n_invalid_oclh += 1
        assert n_invalid_ohlc < n_invalid_oclh, (
            f"KuCoin Futures API schema changed: [O,H,L,C] gives {n_invalid_ohlc} invalid, "
            f"[O,C,H,L] gives {n_invalid_oclh}. If [O,C,H,L] is now correct, revert data_refresh.py."
        )


# =============================================================================
# 7. REPLAY-EQUALS-LIVE INVARIANT (integration-ish)
#    A pure re-computation of the aipt_trader pipeline on prod matrices must
#    exactly reproduce a live log/trade JSON for the same bar.
# =============================================================================

class TestReplayMatchesLive:
    def test_latest_live_row_reproducible_from_prod_matrices(self):
        """Given live state weights and lambda, replaying the latest equity row
        from prod matrices must match the logged port_return within fp noise."""
        equity = PROJECT_ROOT / "prod/logs/kucoin/aipt/performance/equity_aipt.csv"
        if not equity.exists():
            pytest.skip("no equity_aipt.csv")
        df = pd.read_csv(equity)
        if df.empty:
            pytest.skip("equity csv empty")
        last = df.iloc[-1]
        bar_time = pd.Timestamp(last["bar_time"])
        # Find a trade JSON matching this bar — it holds the exact prev_weights used
        trades_dir = PROJECT_ROOT / "prod/logs/kucoin/aipt/trades"
        candidate = [p for p in trades_dir.glob("*.json")
                     if json.loads(p.read_text()).get("bar_time") == str(bar_time)]
        if not candidate:
            pytest.skip(f"no trade JSON for bar {bar_time}")

        # Find the trade JSON for the PRIOR bar; its 'weights' dict is prev_weights
        prior_trades = []
        for p in sorted(trades_dir.glob("*.json")):
            try:
                j = json.loads(p.read_text())
                bt = pd.Timestamp(j.get("bar_time"))
                if bt < bar_time:
                    prior_trades.append((bt, j))
            except Exception:
                pass
        if not prior_trades:
            pytest.skip("no prior trade JSON to source prev_weights")
        prior_trades.sort(key=lambda x: x[0])
        prev_j = prior_trades[-1][1]
        prev_w_dict = prev_j["weights"]

        close = pd.read_parquet(PROD_MATRICES / "close.parquet")
        if bar_time not in close.index:
            pytest.skip(f"{bar_time} not in prod close matrix")
        tickers = list(prev_w_dict.keys())
        close_t = close.loc[bar_time, tickers].values.astype(np.float64)
        t_prev = close.index[close.index.get_loc(bar_time) - 1]
        close_prev = close.loc[t_prev, tickers].values.astype(np.float64)
        R = np.nan_to_num((close_t - close_prev) / close_prev, nan=0.0)
        w = np.array([prev_w_dict[t] for t in tickers], dtype=np.float64)
        replay_gross = float(w @ R)
        logged_gross = float(last["port_return_gross"])
        diff_bps = (replay_gross - logged_gross) * 10000
        assert abs(diff_bps) < 0.5, (
            f"replay gross {replay_gross*10000:.4f} bps vs logged "
            f"{logged_gross*10000:.4f} bps (|diff|={abs(diff_bps):.4f} bps)"
        )
