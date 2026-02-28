"""
FastAPI backend for the Alpha Research Platform.

Serves the GUI and exposes REST + WebSocket endpoints for
alpha evaluation, campaign management, and database queries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialised at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Mutable singleton holding the loaded evaluator, DB, etc."""
    def __init__(self):
        self.ctx = None
        self.engine = None
        self.evaluator = None
        self.db = None
        self.fmp_key = ""
        self.ready = False
        self.ws_clients: list[WebSocket] = []

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load data context and evaluator."""
    fmp_key = os.environ.get("FMP_API_KEY", "")
    db_path = os.environ.get("ALPHA_DB", "data/alpha.db")
    cache_dir = os.environ.get("FMP_CACHE", "data/fmp_cache")
    symbols_env = os.environ.get("SYMBOLS", "")
    start_date = os.environ.get("START_DATE", "2022-01-01")

    from src.data.alpha_database import AlphaDatabase

    state.db = AlphaDatabase(db_path)
    state.fmp_key = fmp_key

    # Always try to load pre-built matrices first (they don't need an API key)
    matrices_dir = Path(cache_dir) / "matrices"
    meta_path = Path(cache_dir) / "metadata.json"

    if matrices_dir.exists() and meta_path.exists():
        try:
            import json
            meta = json.load(open(meta_path))
            logger.info(f"Loading pre-built matrices: {meta['n_tickers']} tickers × {meta['n_days']} days")

            from src.operators.fastexpression import create_engine_from_context
            from src.evaluation.pipeline import AlphaEvaluator
            from src.data.context_research import InMemoryDataContext

            ctx = InMemoryDataContext()

            # Load all matrix parquets
            for pfile in sorted(matrices_dir.glob("*.parquet")):
                field_name = pfile.stem
                df = pd.read_parquet(pfile)
                ctx._price_matrices[field_name] = df

            # Set trading days + tickers from close matrix
            if "close" in ctx._price_matrices:
                close = ctx._price_matrices["close"]
                ctx._trading_days = sorted(close.index.tolist())

            # Load classifications
            cls_path = Path(cache_dir) / "classifications.json"
            if cls_path.exists():
                ctx._classifications = json.load(open(cls_path))
                logger.info(f"Classifications loaded: {len(ctx._classifications)} symbols")

            # Load universe memberships
            uni_dir = Path(cache_dir) / "universes"
            if uni_dir.exists():
                for ufile in uni_dir.glob("*.json"):
                    uni_name = ufile.stem  # e.g., TOP3000
                    ctx._universes[uni_name] = json.load(open(ufile))
                logger.info(f"Universes loaded: {list(ctx._universes.keys())}")

            state.ctx = ctx
            state.engine = create_engine_from_context(ctx)
            state.evaluator = AlphaEvaluator(
                engine=state.engine, ctx=state.ctx, db=state.db
            )
            state.ready = True
            state.data_source = "FMP"
            state.n_tickers = meta["n_tickers"]
            state.n_days = meta["n_days"]
            logger.info(f"FMP data ready: {meta['n_tickers']} tickers × {meta['n_days']} days")
        except Exception as e:
            logger.error(f"Failed to load pre-built matrices: {e}")
            traceback.print_exc()

    # Fallback: use FMPDataLoader (slow, per-symbol) — only if matrices failed AND key exists
    if not state.ready and fmp_key:
        from src.operators.fastexpression import create_engine_from_context
        from src.evaluation.pipeline import AlphaEvaluator
        from src.data.fmp_loader import FMPDataLoader, DEFAULT_US_SYMBOLS
        symbols = symbols_env.split(",") if symbols_env else DEFAULT_US_SYMBOLS[:30]
        loader = FMPDataLoader(api_key=fmp_key, cache_dir=cache_dir)
        try:
            state.ctx = loader.build_context(symbols=symbols, start_date=start_date)
            state.engine = create_engine_from_context(state.ctx)
            state.evaluator = AlphaEvaluator(
                engine=state.engine, ctx=state.ctx, db=state.db
            )
            state.ready = True
            logger.info("Data loaded via FMPDataLoader, evaluator ready.")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    if not state.ready:
        # Synthetic fallback
        from src.evaluation.pipeline import AlphaEvaluator
        ev = AlphaEvaluator.from_synthetic(n_stocks=50, n_days=300, db_path=db_path)
        state.evaluator = ev
        state.engine = ev.engine
        state.ctx = ev.ctx
        state.db = ev.db
        state.ready = True
        logger.info("Using synthetic data (no FMP_API_KEY or matrices).")

    yield
    if state.db:
        state.db.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Alpha Research Platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the GUI (static files)
GUI_DIR = Path(__file__).resolve().parent.parent.parent / "gui"
if GUI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(GUI_DIR)), name="gui")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    expression: str
    params: dict | None = None
    delay: int = 1
    decay: int = 0
    neutralization: str = "market"
    booksize: float = 10_000_000

class BatchEvalRequest(BaseModel):
    expressions: list[str]
    params: dict | None = None

class GPRunRequest(BaseModel):
    generations: int = 50
    population: int = 200
    max_depth: int = 6
    seed: int | None = None
    sharpe_cutoff: float = 0.5    # Minimum Sharpe to record an alpha
    corr_cutoff: float = 0.7      # Maximum correlation allowed between alphas
    min_turnover: float = 0.01    # Minimum turnover filter
    max_turnover: float = 0.7     # Maximum turnover filter
    neutralization: str = "market"

class LLMRunRequest(BaseModel):
    trials: int = 20
    strategy: str = "momentum+value"
    api_key: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Serve the main GUI page."""
    index_path = GUI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "GUI not found. Place index.html in gui/"})


@app.get("/api/status")
async def api_status():
    data_source = getattr(state, "data_source", "synthetic" if not state.fmp_key else "fmp")
    n_tickers = getattr(state, "n_tickers", 0)
    n_days = getattr(state, "n_days", 0)

    # Fallback to computing from matrices if not set
    if n_tickers == 0 and state.ctx and "close" in state.ctx._price_matrices:
        close = state.ctx._price_matrices["close"]
        n_tickers = len(close.columns)
        n_days = len(close.index)

    return {
        "ready": state.ready,
        "data_source": data_source,
        "n_tickers": n_tickers,
        "n_days": n_days,
        "n_fields": len(state.ctx._price_matrices) if state.ctx else 0,
    }


@app.post("/api/evaluate")
async def api_evaluate(req: EvalRequest):
    """Evaluate a single alpha expression with IS/OOS split."""
    if not state.ready:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    params = req.params or {}
    params.update({
        "delay": req.delay,
        "decay": req.decay,
        "neutralization": req.neutralization,
        "booksize": req.booksize,
    })

    result = state.evaluator.evaluate(
        expression=req.expression,
        params=params,
        store=True,
    )

    resp = result.to_dict()
    if result.sim_result:
        sim = result.sim_result
        resp["daily_pnl"] = sim.daily_pnl.tolist()
        resp["cumulative_pnl"] = sim.cumulative_pnl.tolist()
        resp["pnl_dates"] = [str(d) for d in sim.daily_pnl.index]
        resp["daily_returns"] = sim.daily_returns.tolist()
        resp["annualized_return"] = float(sim.returns_ann)
        resp["coverage"] = float(
            sim.positions.notna().sum(axis=1).mean() / max(sim.positions.shape[1], 1)
        )

        # Year-by-year stats
        try:
            yearly = _compute_yearly_stats(sim)
            resp["yearly_stats"] = yearly
        except Exception as e:
            logger.warning(f"Yearly stats failed: {e}")

        # OOS evaluation
        try:
            from src.simulation.oos import fixed_split_oos
            if result.alpha_df is not None and state.evaluator._forward_returns is not None:
                oos_result = fixed_split_oos(
                    alpha_df=result.alpha_df,
                    returns_df=state.evaluator._forward_returns,
                    close_df=state.evaluator._close_df,
                    groups=state.evaluator._groups if req.neutralization == "group" else None,
                    is_ratio=0.7,
                    booksize=req.booksize,
                    decay=req.decay,
                    delay=req.delay,
                    neutralization=req.neutralization,
                )
                resp["oos"] = oos_result.to_dict()
        except Exception as e:
            logger.warning(f"OOS evaluation failed: {e}")

    # Sanitize numpy types for JSON serialization
    resp = _sanitize_for_json(resp)

    await state.broadcast({"type": "eval_result", "data": resp})
    return JSONResponse(content=resp)


def _sanitize_for_json(obj):
    """Recursively convert numpy types to Python builtins for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    return obj


def _compute_yearly_stats(sim) -> list[dict]:
    """Compute year-by-year stats from a VectorizedSimResult."""
    pnl = sim.daily_pnl
    dates = pnl.index
    equity = 10_000_000.0  # booksize / 2

    years = {}
    for i, d in enumerate(dates):
        yr = d.year if hasattr(d, 'year') else pd.Timestamp(d).year
        if yr not in years:
            years[yr] = {"pnl": [], "turnover": [], "indices": []}
        years[yr]["pnl"].append(float(pnl.iloc[i]))
        if i < len(sim.daily_turnover):
            years[yr]["turnover"].append(float(sim.daily_turnover.iloc[i]))
        years[yr]["indices"].append(i)

    results = []
    for yr in sorted(years.keys()):
        yr_pnl = np.array(years[yr]["pnl"])
        yr_turn = np.array(years[yr]["turnover"]) if years[yr]["turnover"] else np.array([0.0])
        yr_ret = yr_pnl / equity

        n = len(yr_pnl)
        if n < 5:
            continue

        mean_ret = yr_ret.mean()
        std_ret = yr_ret.std(ddof=1) if n > 1 else 1e-8
        sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        ann_return = float(mean_ret * 252)
        turnover = float(yr_turn.mean())

        # Drawdown
        cum = np.cumsum(yr_pnl)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak)
        max_dd = float(dd.min() / equity) if equity > 0 else 0.0

        # Fitness
        if turnover > 0.001:
            fitness = sharpe * np.sqrt(abs(ann_return * 100) / max(turnover, 0.125))
        else:
            fitness = 0.0

        # Margin
        total_traded = sum(abs(p) for p in yr_pnl)
        margin = float(sum(yr_pnl) / total_traded * 10000) if total_traded > 0 else 0.0

        # Position counts
        pos_df = sim.positions
        yr_dates = dates[years[yr]["indices"]]
        valid_pos = pos_df.loc[yr_dates[yr_dates.isin(pos_df.index)]] if len(yr_dates) > 0 else pd.DataFrame()
        long_count = int((valid_pos > 0).sum(axis=1).mean()) if len(valid_pos) > 0 else 0
        short_count = int((valid_pos < 0).sum(axis=1).mean()) if len(valid_pos) > 0 else 0

        results.append({
            "year": yr,
            "sharpe": round(sharpe, 2),
            "turnover": round(turnover, 4),
            "fitness": round(fitness, 2),
            "returns": round(ann_return, 4),
            "drawdown": round(max_dd, 4),
            "margin_bps": round(margin, 2),
            "long_count": long_count,
            "short_count": short_count,
        })
    return results


@app.post("/api/evaluate/batch")
async def api_evaluate_batch(req: BatchEvalRequest):
    if not state.ready:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    results = []
    for expr in req.expressions:
        r = state.evaluator.evaluate(expression=expr, params=req.params, store=True)
        results.append(r.to_dict())
    return {"results": results}


@app.get("/api/alphas")
async def api_list_alphas(
    limit: int = Query(50, ge=1, le=500),
    metric: str = Query("sharpe"),
):
    """Get top alphas from the database."""
    if not state.db:
        return {"alphas": []}
    return {"alphas": state.db.get_top_alphas(metric=metric, limit=limit)}


@app.get("/api/alphas/{alpha_id}")
async def api_get_alpha(alpha_id: int):
    if not state.db:
        return JSONResponse({"error": "No DB"}, status_code=503)
    alpha = state.db.get_alpha(alpha_id)
    if not alpha:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return alpha


@app.get("/api/stats")
async def api_stats():
    if not state.db:
        return {}
    return state.db.get_stats()


@app.get("/api/history")
async def api_history(limit: int = Query(30)):
    if not state.db:
        return {"history": []}
    return {"history": state.db.get_history_for_prompt(limit=limit)}


@app.get("/api/fields")
async def api_fields():
    """Get available data fields."""
    from src.data.field_catalog import ALL_FIELDS
    return {
        "fields": [
            {"name": f.name, "group": f.group, "description": f.description}
            for f in ALL_FIELDS
        ]
    }


@app.get("/api/operators")
async def api_operators():
    """Get available operators."""
    from src.agent.research_agent import OPERATOR_CATALOG
    return {"catalog": OPERATOR_CATALOG}


# ---------------------------------------------------------------------------
# Portfolio Optimization
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    alpha_ids: list[int] | None = None  # If None, use top N by Sharpe
    top_n: int = 10
    method: str = "max_sharpe"  # equal_weight, risk_parity, max_sharpe, min_variance
    booksize: float = 20_000_000.0
    compare_all: bool = False  # If True, run all methods and return comparison


@app.post("/api/portfolio/optimize")
async def api_portfolio_optimize(req: PortfolioRequest):
    """Optimize a portfolio of multiple alphas."""
    if not state.ready:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    from src.portfolio.optimizer import PortfolioOptimizer

    optimizer = PortfolioOptimizer(booksize=req.booksize)

    # Get alpha expressions to combine
    if req.alpha_ids:
        alphas = [state.db.get_alpha(aid) for aid in req.alpha_ids]
        alphas = [a for a in alphas if a]
    else:
        top_alphas = state.db.get_top_alphas(metric="sharpe", limit=req.top_n)
        alphas = top_alphas

    if not alphas:
        return JSONResponse({"error": "No alphas to optimize"}, status_code=400)

    # Evaluate each alpha and add to optimizer
    alpha_info = []  # Track names and expressions
    for i, alpha in enumerate(alphas):
        expr = alpha.get("expression", alpha.get("alpha_expression", ""))
        if not expr:
            continue
        try:
            result = state.evaluator.evaluate(expression=expr, store=False)
            if result.success and result.sim_result:
                name = f"alpha_{i+1}"
                optimizer.add_from_sim_result(name, result.sim_result)
                alpha_info.append({
                    "name": name,
                    "expression": expr,
                    "sharpe": float(result.sharpe or 0),
                    "id": alpha.get("id", i),
                })
        except Exception:
            continue

    if optimizer.n_alphas == 0:
        return JSONResponse({"error": "No valid alphas could be evaluated"}, status_code=400)

    try:
        logger.info(f"Portfolio optimize: compare_all={req.compare_all}, method={req.method}, n_alphas={optimizer.n_alphas}")
        if req.compare_all:
            # Run all methods and return comparison
            all_results = optimizer.optimize_all()
            logger.info(f"optimize_all returned {len(all_results)} methods: {list(all_results.keys())}")
            resp = {
                "compare_all": True,
                "alpha_info": alpha_info,
                "methods": {},
            }
            for method_name, result in all_results.items():
                resp["methods"][method_name] = _sanitize_for_json(result.to_dict())
            return JSONResponse(content=resp)
        else:
            result = optimizer.optimize(method=req.method)
            resp = _sanitize_for_json(result.to_dict())
            resp["alpha_info"] = alpha_info
            return JSONResponse(content=resp)
    except Exception as e:
        logger.exception(f"Portfolio optimization error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# GP campaign (runs in background, pushes via WS)
# ---------------------------------------------------------------------------

@app.post("/api/gp/run")
async def api_gp_run(req: GPRunRequest):
    if not state.ready:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    asyncio.create_task(_run_gp(req))
    return {"status": "started", "generations": req.generations, "population": req.population}


async def _run_gp(req: GPRunRequest):
    from src.agent.gp_engine import GPAlphaEngine, GPConfig

    config = GPConfig(
        population_size=req.population,
        n_generations=req.generations,
        max_tree_depth=req.max_depth,
        fitness_cutoff=req.sharpe_cutoff,
        corr_cutoff=req.corr_cutoff,
        min_turnover=req.min_turnover,
        max_turnover=req.max_turnover,
        neutralization=req.neutralization,
    )

    engine = GPAlphaEngine.from_context(state.ctx, config=config, db=state.db)

    await state.broadcast({"type": "gp_start", "data": {
        "generations": req.generations, "population": req.population,
        "sharpe_cutoff": req.sharpe_cutoff, "corr_cutoff": req.corr_cutoff,
    }})

    try:
        results = await asyncio.to_thread(
            engine.run,
            n_generations=req.generations,
            population_size=req.population,
            seed=req.seed,
            verbose=False,
        )

        await state.broadcast({"type": "gp_complete", "data": {
            "best_expression": results.get("best_expression", ""),
            "best_fitness": results.get("best_fitness", 0),
            "trials": results.get("trials_evaluated", 0),
            "best_alphas": results.get("best_alphas", [])[:20],
        }})
    except Exception as e:
        logger.exception(f"GP campaign error: {e}")
        await state.broadcast({"type": "gp_error", "data": {"error": str(e)}})


# ---------------------------------------------------------------------------
# LLM campaign
# ---------------------------------------------------------------------------

@app.post("/api/llm/run")
async def api_llm_run(req: LLMRunRequest):
    if not state.ready:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    asyncio.create_task(_run_llm(req))
    return {"status": "started", "trials": req.trials}


async def _run_llm(req: LLMRunRequest):
    from src.agent.research_agent import AlphaResearchAgent, GeminiProvider, StubLLMProvider

    gemini_key = req.api_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        llm = GeminiProvider(api_key=gemini_key)
    else:
        llm = StubLLMProvider()

    agent = AlphaResearchAgent(evaluator=state.evaluator, llm=llm)

    await state.broadcast({"type": "llm_start", "data": {"trials": req.trials}})

    try:
        results = await agent.run_campaign(
            n_trials=req.trials,
            strategy=req.strategy,
        )
        await state.broadcast({"type": "llm_complete", "data": results})
    except Exception as e:
        await state.broadcast({"type": "llm_error", "data": {"error": str(e)}})


# ---------------------------------------------------------------------------
# WebSocket for live updates
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.ws_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Client can send pings or commands here
            if data == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
