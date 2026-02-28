"""
CLI entry point for the Factor Alpha Research Platform.

Usage:
    # Evaluate a single expression
    python -m src.cli eval "rank(ts_delta(close, 20))"

    # Run GP evolution
    python -m src.cli gp --generations 50 --population 200

    # Run LLM research agent
    python -m src.cli llm --trials 20 --api-key YOUR_GEMINI_KEY

    # Load data from FMP
    python -m src.cli load --symbols AAPL,MSFT,GOOGL --start 2022-01-01

    # Start the web server
    python -m src.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import pandas as pd


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_fmp_key(args) -> str:
    return getattr(args, "fmp_key", None) or os.environ.get("FMP_API_KEY", "")


def cmd_load(args):
    """Load data from FMP into cache."""
    from src.data.fmp_loader import FMPDataLoader, DEFAULT_US_SYMBOLS

    key = get_fmp_key(args)
    if not key:
        print("ERROR: --fmp-key or FMP_API_KEY env var required")
        sys.exit(1)

    symbols = args.symbols.split(",") if args.symbols else DEFAULT_US_SYMBOLS
    loader = FMPDataLoader(api_key=key, cache_dir=args.cache_dir)
    ctx = loader.build_context(
        symbols=symbols,
        start_date=args.start,
        include_fundamentals=args.fundamentals,
    )
    print(f"\nDone. {loader.get_request_count()} API calls, "
          f"{loader.remaining_requests()} remaining today (approx).")


def cmd_eval(args):
    """Evaluate one or more alpha expressions."""
    from src.data.fmp_loader import FMPDataLoader, DEFAULT_US_SYMBOLS
    from src.operators.fastexpression import create_engine_from_context
    from src.evaluation.pipeline import AlphaEvaluator
    from src.data.alpha_database import AlphaDatabase

    key = get_fmp_key(args)
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_US_SYMBOLS[:20]

    if key:
        loader = FMPDataLoader(api_key=key, cache_dir=args.cache_dir)
        ctx = loader.build_context(symbols=symbols, start_date=args.start)
    else:
        from src.evaluation.pipeline import AlphaEvaluator
        evaluator = AlphaEvaluator.from_synthetic(n_stocks=50, n_days=500)
        print("(No FMP key -- using synthetic data)")
        engine = evaluator.engine
        ctx = evaluator.ctx
        db = evaluator.db

    if key:
        engine = create_engine_from_context(ctx)
        db = AlphaDatabase(args.db)
        evaluator = AlphaEvaluator(engine=engine, ctx=ctx, db=db)

    expressions = args.expression if isinstance(args.expression, list) else [args.expression]
    print(f"\nEvaluating {len(expressions)} expression(s)...")
    print("=" * 80)

    for expr in expressions:
        r = evaluator.evaluate(expr, store=True)
        if r.success:
            print(f"  Sharpe:    {r.sharpe:+.4f}")
            print(f"  Fitness:   {r.fitness:+.4f}")
            print(f"  Turnover:  {r.turnover:.4f}")
            print(f"  MaxDD:     {r.max_drawdown:+.4f}")
            print(f"  Margin:    {r.margin_bps:.2f} bps")
            print(f"  Checks:    {r.passed_checks}/8")
        else:
            print(f"  FAILED: {r.error}")
        print(f"  Expr: {expr}")
        print()


def cmd_gp(args):
    """Run GP evolutionary search."""
    from src.data.fmp_loader import FMPDataLoader, DEFAULT_US_SYMBOLS
    from src.agent.gp_engine import GPAlphaEngine, GPConfig
    from src.data.alpha_database import AlphaDatabase
    from src.data.context_research import InMemoryDataContext

    key = get_fmp_key(args)
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_US_SYMBOLS[:30]

    config = GPConfig(
        population_size=args.population,
        n_generations=args.generations,
        max_tree_depth=args.depth,
        booksize=args.booksize,
    )

    if key:
        loader = FMPDataLoader(api_key=key, cache_dir=args.cache_dir)
        ctx = loader.build_context(symbols=symbols, start_date=args.start)
        db = AlphaDatabase(args.db)
        engine = GPAlphaEngine.from_context(ctx, config=config, db=db)
    else:
        print("(No FMP key -- using synthetic data)")
        engine = GPAlphaEngine.from_synthetic(
            n_stocks=50, n_days=300, config=config
        )

    print(f"\nStarting GP evolution: pop={args.population}, gen={args.generations}")
    print("=" * 80)

    results = engine.run(
        n_generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print(f"Best expression: {results['best_expression']}")
    print(f"Best fitness:    {results['best_fitness']:.4f}")
    print(f"Trials:          {results['trials_evaluated']}")

    if results.get("best_alphas"):
        print(f"\nTop alphas found:")
        for i, a in enumerate(results["best_alphas"][:10]):
            print(f"  {i+1}. F={a['fitness']:+.4f} | {a['expression'][:70]}")


def cmd_llm(args):
    """Run LLM research agent."""
    from src.data.fmp_loader import FMPDataLoader, DEFAULT_US_SYMBOLS
    from src.operators.fastexpression import create_engine_from_context
    from src.evaluation.pipeline import AlphaEvaluator
    from src.data.alpha_database import AlphaDatabase
    from src.agent.research_agent import AlphaResearchAgent, GeminiProvider, StubLLMProvider

    key = get_fmp_key(args)
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_US_SYMBOLS[:30]

    if key:
        loader = FMPDataLoader(api_key=key, cache_dir=args.cache_dir)
        ctx = loader.build_context(symbols=symbols, start_date=args.start)
        engine = create_engine_from_context(ctx)
        db = AlphaDatabase(args.db)
        evaluator = AlphaEvaluator(engine=engine, ctx=ctx, db=db)
    else:
        evaluator = AlphaEvaluator.from_synthetic(n_stocks=50, n_days=300)
        print("(No FMP key -- using synthetic data)")

    gemini_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        llm = GeminiProvider(api_key=gemini_key)
    else:
        print("(No Gemini key -- using stub LLM)")
        llm = StubLLMProvider()

    agent = AlphaResearchAgent(evaluator=evaluator, llm=llm)

    print(f"\nStarting LLM research: {args.trials} trials")
    print("=" * 80)

    loop = asyncio.new_event_loop()
    results = loop.run_until_complete(
        agent.run_campaign(n_trials=args.trials, strategy=args.strategy)
    )
    loop.close()

    print("\n" + "=" * 80)
    print(f"Completed {results.get('total', 0)} trials")
    print(f"Successful: {results.get('successful', 0)}")
    if results.get("best"):
        best = results["best"]
        print(f"Best Sharpe: {best.get('sharpe', 0):.4f}")
        print(f"Best expr: {best.get('expression', 'N/A')}")


def cmd_serve(args):
    """Start the web server."""
    print(f"Starting server on http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")

    # Import here to avoid loading uvicorn at CLI parse time
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="alpha",
        description="Factor Alpha Research Platform",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    # Shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--fmp-key", help="FMP API key")
    shared.add_argument("--cache-dir", default="data/fmp_cache")
    shared.add_argument("--db", default="data/alpha.db", help="SQLite DB path")
    shared.add_argument("--start", default="2022-01-01", help="Start date")
    shared.add_argument("--symbols", default="", help="Comma-separated symbols")

    sub = parser.add_subparsers(dest="command")

    # load
    p_load = sub.add_parser("load", parents=[shared], help="Load FMP data")
    p_load.add_argument("--fundamentals", action="store_true")

    # eval
    p_eval = sub.add_parser("eval", parents=[shared], help="Evaluate expression(s)")
    p_eval.add_argument("expression", nargs="+", help="Alpha expression(s)")

    # gp
    p_gp = sub.add_parser("gp", parents=[shared], help="Run GP evolution")
    p_gp.add_argument("--generations", type=int, default=50)
    p_gp.add_argument("--population", type=int, default=200)
    p_gp.add_argument("--depth", type=int, default=6)
    p_gp.add_argument("--booksize", type=float, default=10_000_000)
    p_gp.add_argument("--seed", type=int, default=None)

    # llm
    p_llm = sub.add_parser("llm", parents=[shared], help="Run LLM agent")
    p_llm.add_argument("--api-key", help="Gemini API key")
    p_llm.add_argument("--trials", type=int, default=20)
    p_llm.add_argument("--strategy", default="momentum+value", help="Strategy hint")

    # serve
    p_serve = sub.add_parser("serve", help="Start web server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "load": cmd_load,
        "eval": cmd_eval,
        "gp": cmd_gp,
        "llm": cmd_llm,
        "serve": cmd_serve,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
