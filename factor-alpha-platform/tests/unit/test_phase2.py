"""
Phase 2 tests — fast unit tests only.
Slow tests (GP evolution, full evaluation pipeline) are marked @pytest.mark.slow.
Run with: pytest -m "not slow" for fast tests only.
"""

import asyncio
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_db():
    return os.path.join(tempfile.mkdtemp(), "t.db")


# ===========================================================================
# Vectorized Sim (no synthetic generator — raw DataFrames)
# ===========================================================================

class TestVectorizedSim:

    @pytest.fixture(autouse=True)
    def _data(self):
        np.random.seed(42)
        d, n = 30, 5
        dates = pd.bdate_range("2020-01-02", periods=d)
        tks = [f"T{i}" for i in range(n)]
        self.close = pd.DataFrame(
            np.cumsum(np.random.randn(d, n) * 0.02, axis=0) + 10,
            index=dates, columns=tks)
        self.fwd = self.close.pct_change().shift(-1)

    def test_basic(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = self.close.rank(axis=1, pct=True)
        r = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, booksize=1e6)
        assert not math.isnan(r.sharpe)
        assert len(r.daily_pnl) > 0

    def test_zero_alpha(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = pd.DataFrame(0.0, index=self.close.index, columns=self.close.columns)
        r = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, booksize=1e6)
        assert r.total_pnl == 0.0

    def test_market_neutral(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = self.close.rank(axis=1, pct=True)
        r = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd,
                                neutralization="market", booksize=1e6)
        assert abs(r.positions.sum(axis=1).dropna().mean()) < 0.05

    def test_delay_matters(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = self.close.rank(axis=1, pct=True)
        r0 = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, delay=0, booksize=1e6)
        r1 = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, delay=1, booksize=1e6)
        assert r0.sharpe != r1.sharpe

    def test_turnover_bounded(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = self.close.diff(5).rank(axis=1, pct=True)
        r = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, booksize=1e6)
        assert 0 <= r.turnover < 2.0

    def test_fees_reduce_pnl(self):
        from src.simulation.vectorized_sim import simulate_vectorized
        alpha = self.close.diff(3).rank(axis=1, pct=True)
        r0 = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, fees_bps=0, booksize=1e6)
        r1 = simulate_vectorized(alpha_df=alpha, returns_df=self.fwd, fees_bps=50, booksize=1e6)
        assert r1.total_pnl <= r0.total_pnl


# ===========================================================================
# Alpha Database (pure SQLite, no data generation)
# ===========================================================================

class TestAlphaDatabase:

    def test_create(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db); assert os.path.exists(tmp_db); db.close()

    def test_insert_retrieve(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        aid = db.insert_alpha(expression="rank(close)", reasoning="test")
        assert db.get_alpha(aid)["expression"] == "rank(close)"
        db.close()

    def test_duplicate(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        assert db.insert_alpha(expression="rank(close)") == db.insert_alpha(expression="rank(close)")
        db.close()

    def test_evaluation(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        aid = db.insert_alpha(expression="rank(close)")
        assert db.insert_evaluation(alpha_id=aid, sharpe=1.5) > 0
        db.close()

    def test_top_alphas(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        for i in range(5):
            aid = db.insert_alpha(expression=f"rank(d{i})")
            db.insert_evaluation(alpha_id=aid, sharpe=float(i), fitness=float(i)/2)
        top = db.get_top_alphas(metric="sharpe", limit=3)
        assert len(top) == 3 and top[0]["sharpe"] >= top[1]["sharpe"]
        db.close()

    def test_run(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        assert db.create_run(strategy="test") > 0
        db.close()

    def test_history(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        for i in range(3):
            aid = db.insert_alpha(expression=f"rank(x{i})")
            db.insert_evaluation(alpha_id=aid, sharpe=float(i))
        assert all("code" in x for x in db.get_history_for_prompt(limit=2))
        db.close()

    def test_stats(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        aid = db.insert_alpha(expression="rank(close)")
        db.insert_evaluation(alpha_id=aid, sharpe=1.5)
        assert db.get_stats()["top_sharpe"] == 1.5
        db.close()

    def test_exists(self, tmp_db):
        from src.data.alpha_database import AlphaDatabase
        db = AlphaDatabase(tmp_db)
        db.insert_alpha(expression="rank(close)")
        assert db.alpha_exists("rank(close)") and not db.alpha_exists("rank(open)")
        db.close()


# ===========================================================================
# Field Catalog (pure data, no IO)
# ===========================================================================

class TestFieldCatalog:

    def test_populated(self):
        from src.data.field_catalog import ALL_FIELDS, FIELD_BY_NAME
        assert len(ALL_FIELDS) > 50 and "close" in FIELD_BY_NAME

    def test_by_group(self):
        from src.data.field_catalog import get_fields_by_group
        assert len(get_fields_by_group("MARKET DATA")) > 5

    def test_format_prompt(self):
        from src.data.field_catalog import format_fields_for_prompt
        assert "close" in format_fields_for_prompt(groups=["MARKET DATA"])

    def test_grouping(self):
        from src.data.field_catalog import GROUPING_FIELD_NAMES
        assert "industry" in GROUPING_FIELD_NAMES and "close" not in GROUPING_FIELD_NAMES


# ===========================================================================
# LLM Agent Parsing (pure string logic, no data)
# ===========================================================================

class TestLLMAgentParsing:

    def test_extract_marker(self):
        from src.agent.research_agent import _extract_alpha_code
        assert _extract_alpha_code("ALPHA: rank(close)\nRE") == "rank(close)"

    def test_extract_bare(self):
        from src.agent.research_agent import _extract_alpha_code
        assert "rank(" in _extract_alpha_code("Try:\nrank(ts_delta(close, 60)) * rank(volume)")

    def test_clean_spacing(self):
        from src.agent.research_agent import _clean_expression
        assert "* rank(volume)" in _clean_expression("rank(close) rank(volume)")

    def test_extract_reasoning(self):
        from src.agent.research_agent import _extract_reasoning
        assert _extract_reasoning("REASONING: momentum signal") == "momentum signal"

    def test_extract_params(self):
        from src.agent.research_agent import _extract_parameters
        p = _extract_parameters('PARAMETERS_JSON: {"decay": 5, "delay": 2}')
        assert p["decay"] == 5

    def test_validate_valid(self):
        from src.agent.research_agent import _validate_syntax
        assert _validate_syntax("rank(ts_delta(close, 60))")

    def test_validate_invalid(self):
        from src.agent.research_agent import _validate_syntax
        assert not _validate_syntax("") and not _validate_syntax("rank(close")

    def test_stub_provider(self):
        from src.agent.research_agent import StubLLMProvider
        stub = StubLLMProvider()
        loop = asyncio.new_event_loop()
        assert "ALPHA:" in loop.run_until_complete(stub.generate("test"))
        loop.close()


# ===========================================================================
# New Vectorized Operators (no data generation)
# ===========================================================================

class TestNewOps:

    @pytest.fixture(autouse=True)
    def _df(self):
        np.random.seed(42)
        self.df = pd.DataFrame(np.random.randn(30, 4) * 0.05 + 1.0,
                               index=pd.bdate_range("2020-01-02", periods=30),
                               columns=["A", "B", "C", "D"])

    def test_normalize(self):
        from src.operators.vectorized import normalize
        r = normalize(self.df)
        assert r.max().max() <= 1.001 and r.min().min() >= -1.001

    def test_ts_backfill(self):
        from src.operators.vectorized import ts_backfill
        df = self.df.copy(); df.iloc[5:8, 0] = np.nan
        assert not np.isnan(ts_backfill(df, 5).iloc[5, 0])

    def test_ts_scale(self):
        from src.operators.vectorized import ts_scale
        assert ts_scale(self.df, 10).shape == self.df.shape

    def test_reverse(self):
        from src.operators.vectorized import reverse
        pd.testing.assert_frame_equal(reverse(self.df), -self.df)

    def test_is_nan(self):
        from src.operators.vectorized import is_nan
        df = self.df.copy(); df.iloc[0, 0] = np.nan
        assert is_nan(df).iloc[0, 0] == 1.0

    def test_bucket(self):
        from src.operators.vectorized import bucket
        assert bucket(self.df, 4).shape == self.df.shape

    def test_vec_avg(self):
        from src.operators.vectorized import vec_avg
        pd.testing.assert_series_equal(vec_avg(self.df)["A"], self.df.mean(axis=1), check_names=False)

    def test_vec_sum(self):
        from src.operators.vectorized import vec_sum
        assert vec_sum(self.df).shape == self.df.shape

    def test_group_zscore(self):
        from src.operators.vectorized import group_zscore
        assert group_zscore(self.df, pd.Series({"A": "t", "B": "t", "C": "f", "D": "f"})).shape == self.df.shape

    def test_group_scale(self):
        from src.operators.vectorized import group_scale
        assert group_scale(self.df, pd.Series({"A": "t", "B": "t", "C": "f", "D": "f"})).shape == self.df.shape

    def test_days_from_last_change(self):
        from src.operators.vectorized import days_from_last_change
        r = days_from_last_change(pd.DataFrame({"A": [1, 1, 1, 2, 2], "B": [1, 2, 3, 3, 4]}))
        assert r["A"].iloc[3] == 0

    def test_ts_decay_linear_alias(self):
        from src.operators.vectorized import ts_decay_linear, Decay_lin
        pd.testing.assert_frame_equal(ts_decay_linear(self.df, 5), Decay_lin(self.df, 5))


# ===========================================================================
# FastExpression new ops (small inline data)
# ===========================================================================

class TestFastExprNewOps:

    @pytest.fixture(autouse=True)
    def _engine(self):
        from src.operators.fastexpression import FastExpressionEngine
        np.random.seed(42)
        dates = pd.bdate_range("2020-01-02", periods=30)
        tks = ["A", "B", "C", "D"]
        self.engine = FastExpressionEngine(
            data_fields={
                "close": pd.DataFrame(np.cumsum(np.random.randn(30, 4) * 0.02, axis=0) + 100,
                                      index=dates, columns=tks),
                "volume": pd.DataFrame(np.abs(np.random.randn(30, 4) * 1e6 + 5e6),
                                       index=dates, columns=tks),
            },
            groups={"subindustry": pd.Series({"A": "t", "B": "t", "C": "f", "D": "f"})},
        )

    def test_normalize(self):
        assert isinstance(self.engine.evaluate("normalize(close)"), pd.DataFrame)

    def test_reverse(self):
        assert isinstance(self.engine.evaluate("reverse(close)"), pd.DataFrame)

    def test_is_nan(self):
        assert self.engine.evaluate("is_nan(close)").sum().sum() == 0

    def test_ts_scale(self):
        assert isinstance(self.engine.evaluate("ts_scale(close, 10)"), pd.DataFrame)

    def test_ts_decay_linear(self):
        assert isinstance(self.engine.evaluate("ts_decay_linear(close, 5)"), pd.DataFrame)

    def test_vec_avg(self):
        assert isinstance(self.engine.evaluate("vec_avg(close)"), pd.DataFrame)

    def test_group_zscore(self):
        assert isinstance(self.engine.evaluate("group_zscore(close, subindustry)"), pd.DataFrame)

    def test_group_scale(self):
        assert isinstance(self.engine.evaluate("group_scale(close, subindustry)"), pd.DataFrame)


# ===========================================================================
# EvaluationResult (pure dataclass, no IO)
# ===========================================================================

class TestEvaluationResult:

    def test_quality_checks(self):
        from src.evaluation.pipeline import EvaluationResult
        r = EvaluationResult(expression="test", success=True)
        r.sharpe, r.fitness, r.turnover = 2.0, 1.5, 0.1
        r.max_drawdown, r.returns_ann, r.margin_bps = -0.05, 0.15, 5.0
        assert r.compute_quality_checks() > 0

    def test_to_dict(self):
        from src.evaluation.pipeline import EvaluationResult
        r = EvaluationResult(expression="rank(x)", success=True, sharpe=1.0)
        d = r.to_dict()
        assert d["expression"] == "rank(x)" and d["sharpe"] == 1.0


# ===========================================================================
# SLOW TESTS: need synthetic data generation (skip by default)
# ===========================================================================

@pytest.mark.slow
class TestEvaluationPipelineSlow:

    def test_simple(self, tmp_db):
        from src.evaluation.pipeline import AlphaEvaluator
        ev = AlphaEvaluator.from_synthetic(n_stocks=5, n_days=30, seed=42, db_path=tmp_db)
        r = ev.evaluate("rank(close)", store=False)
        assert r.success and r.sharpe is not None

    def test_invalid(self, tmp_db):
        from src.evaluation.pipeline import AlphaEvaluator
        ev = AlphaEvaluator.from_synthetic(n_stocks=5, n_days=30, seed=42, db_path=tmp_db)
        r = ev.evaluate("rank(NONEXISTENT)", store=False)
        assert not r.success

    def test_db_integration(self, tmp_db):
        from src.evaluation.pipeline import AlphaEvaluator
        ev = AlphaEvaluator.from_synthetic(n_stocks=5, n_days=30, seed=42, db_path=tmp_db)
        ev.evaluate("rank(close)", store=True)
        assert ev.get_stats()["total_alphas"] == 1


@pytest.mark.slow
class TestGPEngineSlow:

    def test_build_pset(self):
        from src.agent.gp_engine import build_primitive_set
        pset = build_primitive_set(["close", "volume"])
        assert len(pset.terminals) > 0

    def test_create(self):
        from src.agent.gp_engine import GPAlphaEngine, GPConfig
        e = GPAlphaEngine.from_synthetic(n_stocks=5, n_days=30, seed=42,
                                         config=GPConfig(population_size=5, n_generations=1))
        assert len(e.feature_names) > 0

    def test_short_run(self):
        from src.agent.gp_engine import GPAlphaEngine, GPConfig
        e = GPAlphaEngine.from_synthetic(n_stocks=5, n_days=30, seed=42,
                                         config=GPConfig(population_size=10, n_generations=1, max_tree_depth=3))
        assert e.run(n_generations=1, population_size=10, verbose=False)["trials_evaluated"] > 0


@pytest.mark.slow
class TestAgentIntegrationSlow:

    def test_generate_stub(self, tmp_db):
        from src.agent.research_agent import AlphaResearchAgent, StubLLMProvider
        from src.evaluation.pipeline import AlphaEvaluator
        ev = AlphaEvaluator.from_synthetic(n_stocks=5, n_days=30, seed=42, db_path=tmp_db)
        agent = AlphaResearchAgent(evaluator=ev, llm=StubLLMProvider())
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(agent.generate_alpha(strategy="test"))
        loop.close()
        assert r['success']
