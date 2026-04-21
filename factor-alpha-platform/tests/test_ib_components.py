"""
test_ib_components.py — Tests for IB Closing Auction Trading System.

Tests:
    1. Band Universe Builder (universe_band.py)
    2. IB Cost Model (ib_cost_model.py)
    3. Seed Alphas (seed_alphas_ib.py)
    4. Alpha Evaluator config (eval_alpha_ib.py)

Run:
    python -m pytest tests/test_ib_components.py -v
    python tests/test_ib_components.py  # standalone
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# 1. BAND UNIVERSE BUILDER TESTS
# ============================================================================

class TestBandUniverseBuilder:
    """Test band universe construction logic."""

    def _create_mock_universes(self, tmpdir):
        """Create mock TOP2000 and TOP3000 universe parquets."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        tickers = [f"SYM{i:04d}" for i in range(3500)]

        # TOP2000: first 2000 tickers are True
        top2000 = pd.DataFrame(False, index=dates, columns=tickers)
        top2000.iloc[:, :2000] = True

        # TOP3000: first 3000 tickers are True
        top3000 = pd.DataFrame(False, index=dates, columns=tickers)
        top3000.iloc[:, :3000] = True

        # TOP1500
        top1500 = pd.DataFrame(False, index=dates, columns=tickers)
        top1500.iloc[:, :1500] = True

        # TOP2500
        top2500 = pd.DataFrame(False, index=dates, columns=tickers)
        top2500.iloc[:, :2500] = True

        # TOP3500
        top3500 = pd.DataFrame(False, index=dates, columns=tickers)
        top3500.iloc[:, :3500] = True

        univ_dir = Path(tmpdir) / "universes"
        univ_dir.mkdir(parents=True, exist_ok=True)

        top1500.to_parquet(univ_dir / "TOP1500.parquet")
        top2000.to_parquet(univ_dir / "TOP2000.parquet")
        top2500.to_parquet(univ_dir / "TOP2500.parquet")
        top3000.to_parquet(univ_dir / "TOP3000.parquet")
        top3500.to_parquet(univ_dir / "TOP3500.parquet")

        return univ_dir, dates, tickers

    def test_band_universe_set_difference(self, tmp_path):
        """Band universe = TOP_upper \\ TOP_lower (set difference)."""
        from src.data.universe_band import build_band_universe

        univ_dir, dates, tickers = self._create_mock_universes(tmp_path)

        band = build_band_universe(2000, 3000, universes_dir=univ_dir, save=False)

        assert band is not None
        # Should contain tickers 2000-2999 (indices 2000 through 2999)
        # These are in TOP3000 but NOT in TOP2000
        for i in range(2000, 3000):
            ticker = f"SYM{i:04d}"
            assert band[ticker].all(), f"{ticker} should be in band"

        # Should NOT contain tickers 0-1999 (they're in TOP2000)
        for i in range(0, 2000):
            ticker = f"SYM{i:04d}"
            assert not band[ticker].any(), f"{ticker} should NOT be in band"

        # Should NOT contain tickers 3000+ (they're not in TOP3000)
        for i in range(3000, 3500):
            ticker = f"SYM{i:04d}"
            assert not band[ticker].any(), f"{ticker} should NOT be in band"

    def test_band_member_count(self, tmp_path):
        """Band should have ~1000 members (upper - lower)."""
        from src.data.universe_band import build_band_universe

        univ_dir, _, _ = self._create_mock_universes(tmp_path)

        band = build_band_universe(2000, 3000, universes_dir=univ_dir, save=False)
        avg_members = band.sum(axis=1).mean()
        assert avg_members == 1000, f"Expected 1000 members, got {avg_members}"

    def test_band_saves_parquet(self, tmp_path):
        """Band universe should save to parquet when save=True."""
        from src.data.universe_band import build_band_universe

        univ_dir, _, _ = self._create_mock_universes(tmp_path)

        build_band_universe(2000, 3000, universes_dir=univ_dir, save=True)
        assert (univ_dir / "TOP2000TOP3000.parquet").exists()

    def test_missing_universe_returns_none(self, tmp_path):
        """Should return None when required universe files don't exist."""
        from src.data.universe_band import build_band_universe

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = build_band_universe(2000, 3000, universes_dir=empty_dir, save=False)
        assert result is None

    def test_all_band_universes(self, tmp_path):
        """All standard band universes should build successfully."""
        from src.data.universe_band import build_all_band_universes

        univ_dir, _, _ = self._create_mock_universes(tmp_path)

        results = build_all_band_universes(universes_dir=univ_dir)
        assert "TOP2000TOP3000" in results
        assert "TOP1500TOP2500" in results
        assert "TOP2500TOP3500" in results

        # Each band should have 1000 members
        for name, df in results.items():
            avg = df.sum(axis=1).mean()
            assert avg == 1000, f"{name} has {avg} members, expected 1000"

    def test_validate_band_universe(self, tmp_path):
        """Validation should report correct statistics."""
        from src.data.universe_band import build_band_universe, validate_band_universe

        univ_dir, _, _ = self._create_mock_universes(tmp_path)
        build_band_universe(2000, 3000, universes_dir=univ_dir, save=True)

        v = validate_band_universe("TOP2000TOP3000", universes_dir=univ_dir)
        assert v["exists"] is True
        assert v["valid"] is True
        assert v["avg_members"] == 1000
        assert v["n_dates"] == 100


# ============================================================================
# 2. IB COST MODEL TESTS
# ============================================================================

class TestIBCostModel:
    """Test IBKR Pro Tiered commission model."""

    def test_minimum_commission(self):
        """$0.35 minimum should apply for small share counts."""
        from src.simulation.ib_cost_model import ib_commission

        # 10 shares at $25 = $0.035 calculated, but $0.35 minimum
        assert ib_commission(10, 25.0) == 0.35

    def test_per_share_rate(self):
        """Per-share rate should apply when > minimum."""
        from src.simulation.ib_cost_model import ib_commission

        # 200 shares at $25: 200 * 0.0035 = $0.70 > $0.35 min
        assert ib_commission(200, 25.0) == 0.70

    def test_max_cap(self):
        """Commission should cap at 1% of trade value."""
        from src.simulation.ib_cost_model import ib_commission

        # 1000 shares at $1 = $1000 value, 1% = $10
        # Per share: 1000 * 0.0035 = $3.50
        # Cap: min($3.50, $10) = $3.50 (cap doesn't bind here)
        assert ib_commission(1000, 1.0) == 3.50

        # But for very cheap stocks:
        # 100 shares at $0.10 = $10 value, 1% = $0.10
        # Per share: 100 * 0.0035 = $0.35
        # Cap: min($0.35, $0.10) = $0.10
        # But min($0.10, $0.35 min) -> $0.35 (min dominates)
        result = ib_commission(100, 0.10)
        assert result == 0.35  # min dominates

    def test_zero_shares(self):
        """Zero shares should return zero commission."""
        from src.simulation.ib_cost_model import ib_commission
        assert ib_commission(0, 25.0) == 0.0

    def test_no_spread(self):
        """Total cost should NOT include any spread component."""
        from src.simulation.ib_cost_model import total_trade_cost

        result = total_trade_cost(32, 25.0, is_sell=False)
        # Total should be commission + regulatory fees only
        assert result.total_cost == result.commission + result.regulatory_fees
        # No hidden spread

    def test_sell_side_fees_higher(self):
        """Sell-side should have higher regulatory fees (SEC + FINRA)."""
        from src.simulation.ib_cost_model import regulatory_fees

        buy_fees = regulatory_fees(100, 25.0, is_sell=False)
        sell_fees = regulatory_fees(100, 25.0, is_sell=True)
        assert sell_fees > buy_fees  # SEC + FINRA only on sells

    def test_buy_side_exchange_only(self):
        """Buy-side regulatory fees should be exchange fee only."""
        from src.simulation.ib_cost_model import regulatory_fees, EXCHANGE_FEE_RATE

        buy_fees = regulatory_fees(100, 25.0, is_sell=False)
        expected = 100 * EXCHANGE_FEE_RATE
        assert abs(buy_fees - expected) < 0.01

    def test_typical_trade_cost_bps(self):
        """Typical trade at our position size should be <10 bps."""
        from src.simulation.ib_cost_model import total_trade_cost

        # $3,667 position at $25 = 147 shares (our 4x, 120 position config)
        result = total_trade_cost(147, 25.0, is_sell=False)
        assert result.cost_bps < 10, f"Cost too high: {result.cost_bps} bps"
        # Should be around 1-2 bps for buy side

    def test_daily_cost_estimate(self):
        """Daily cost estimate should be reasonable."""
        from src.simulation.ib_cost_model import estimate_daily_costs

        costs = estimate_daily_costs(
            n_positions=120,
            avg_position_size=3667,
            avg_price=25.0,
            daily_turnover=0.25,
        )
        assert costs["n_trades"] == 60  # 120 * 0.25 * 2
        assert costs["daily_total"] > 0
        assert costs["daily_bps"] < 5  # should be ~2 bps
        assert costs["annual_pct_of_book"] < 5  # should be ~2%

    def test_higher_tier_rates(self):
        """Higher monthly volumes should get lower per-share rates."""
        from src.simulation.ib_cost_model import ib_commission

        # At 500K monthly shares, rate drops to $0.0020
        low_vol = ib_commission(200, 25.0, monthly_shares=100_000)   # $0.0035 tier
        high_vol = ib_commission(200, 25.0, monthly_shares=500_000)  # $0.0020 tier
        assert high_vol < low_vol


# ============================================================================
# 3. SEED ALPHAS TESTS
# ============================================================================

class TestSeedAlphas:
    """Test seed alpha library."""

    def test_seed_count(self):
        """Should have at least 15 seed alphas."""
        from seed_alphas_ib import SEED_ALPHAS
        assert len(SEED_ALPHAS) >= 15

    def test_seed_structure(self):
        """Each seed alpha should have required fields."""
        from seed_alphas_ib import SEED_ALPHAS

        for alpha in SEED_ALPHAS:
            assert "expr" in alpha, f"Missing 'expr' in: {alpha}"
            assert "name" in alpha, f"Missing 'name' in: {alpha}"
            assert "category" in alpha, f"Missing 'category' in: {alpha}"
            assert "reasoning" in alpha, f"Missing 'reasoning' in: {alpha}"
            assert len(alpha["expr"]) > 0
            assert len(alpha["name"]) > 0
            assert len(alpha["reasoning"]) > 10  # meaningful reasoning

    def test_seed_categories(self):
        """Seed alphas should cover multiple categories."""
        from seed_alphas_ib import SEED_ALPHAS

        categories = set(a["category"] for a in SEED_ALPHAS)
        assert "candle" in categories
        assert "reversal" in categories
        assert "momentum" in categories
        assert "volume" in categories
        assert "volatility" in categories
        assert "composite" in categories

    def test_unique_names(self):
        """All seed alpha names should be unique."""
        from seed_alphas_ib import SEED_ALPHAS

        names = [a["name"] for a in SEED_ALPHAS]
        assert len(names) == len(set(names)), f"Duplicate names found: {names}"

    def test_unique_expressions(self):
        """All seed alpha expressions should be unique."""
        from seed_alphas_ib import SEED_ALPHAS

        exprs = [a["expr"] for a in SEED_ALPHAS]
        assert len(exprs) == len(set(exprs)), "Duplicate expressions found"

    def test_get_seed_expressions(self):
        """Helper should return flat list of expressions."""
        from seed_alphas_ib import get_seed_expressions

        exprs = get_seed_expressions()
        assert isinstance(exprs, list)
        assert len(exprs) >= 15
        assert all(isinstance(e, str) for e in exprs)


# ============================================================================
# 4. EVALUATOR CONFIGURATION TESTS
# ============================================================================

class TestEvalAlphaIBConfig:
    """Test that eval_alpha_ib.py has correct configuration."""

    def test_delay_zero(self):
        """Evaluator must use delay=0 for closing auction."""
        import eval_alpha_ib
        assert eval_alpha_ib.DELAY == 0

    def test_fee_free(self):
        """Individual alpha evaluation must be fee-free."""
        import eval_alpha_ib
        assert eval_alpha_ib.FEES_BPS == 0.0

    def test_universe(self):
        """Default universe should be TOP2000TOP3000."""
        import eval_alpha_ib
        assert eval_alpha_ib.UNIVERSE == "TOP2000TOP3000"

    def test_market_neutralization(self):
        """Should use market-level neutralization (sector groups too small in band)."""
        import eval_alpha_ib
        assert eval_alpha_ib.NEUTRALIZE == "market"

    def test_train_period(self):
        """Train period should be 7 years (2016-2022)."""
        import eval_alpha_ib
        assert eval_alpha_ib.TRAIN_START == "2016-01-01"
        assert eval_alpha_ib.TRAIN_END == "2023-01-01"

    def test_separate_db(self):
        """Should use separate DB from equity research."""
        import eval_alpha_ib
        assert eval_alpha_ib.DB_PATH == "data/ib_alphas.db"
        assert eval_alpha_ib.DB_PATH != "data/alpha_results.db"

    def test_db_schema_creation(self):
        """DB schema should create without errors."""
        import eval_alpha_ib
        import sqlite3

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            eval_alpha_ib._ensure_schema(conn)

            # Verify tables exist
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "alphas" in table_names
            assert "evaluations" in table_names
            assert "trial_log" in table_names

            # Verify alphas table has IB-specific columns
            cols = conn.execute("PRAGMA table_info(alphas)").fetchall()
            col_names = [c[1] for c in cols]
            assert "universe" in col_names
            assert "delay" in col_names
            assert "neutralize" in col_names
            assert "asset_class" in col_names

            # Verify evaluations has IB-specific columns
            cols = conn.execute("PRAGMA table_info(evaluations)").fetchall()
            col_names = [c[1] for c in cols]
            assert "sharpe_h1" in col_names
            assert "sharpe_h2" in col_names
            assert "pnl_kurtosis" in col_names
            assert "universe" in col_names

            conn.close()
        finally:
            os.unlink(db_path)

    def test_quality_gates_reasonable(self):
        """Quality gate thresholds should be reasonable for delay-0 small-caps."""
        import eval_alpha_ib

        # Delay-0 should have higher turnover tolerance
        assert eval_alpha_ib.MAX_TURNOVER >= 0.40
        # Small-caps have fatter tails
        assert eval_alpha_ib.MAX_PNL_KURTOSIS >= 20
        # Lower coverage threshold for band universe
        assert eval_alpha_ib.COVERAGE_CUTOFF < 0.5


# ============================================================================
# 5. BULK DOWNLOAD UNIVERSE SIZES TEST
# ============================================================================

class TestBulkDownloadUniverse:
    """Test that bulk_download has the extended universe sizes."""

    def test_extended_sizes(self):
        """UNIVERSE_SIZES should include TOP1500, TOP2500, TOP3500."""
        from src.data.bulk_download import UNIVERSE_SIZES

        assert "TOP1500" in UNIVERSE_SIZES
        assert "TOP2000" in UNIVERSE_SIZES
        assert "TOP2500" in UNIVERSE_SIZES
        assert "TOP3000" in UNIVERSE_SIZES
        assert "TOP3500" in UNIVERSE_SIZES

        assert UNIVERSE_SIZES["TOP1500"] == 1500
        assert UNIVERSE_SIZES["TOP2500"] == 2500
        assert UNIVERSE_SIZES["TOP3500"] == 3500


# ============================================================================
# RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
