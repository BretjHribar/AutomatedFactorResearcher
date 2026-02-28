"""
GP Alpha Runner — TOP1000 universe with fundamentals + subindustry neutralization.

Loads pre-built matrices, trims to TOP1000 tickers only, runs GP.
"""
import sys
import os
import logging

import numpy as np
import pandas as pd

from src.data.context_research import InMemoryDataContext
from src.data.alpha_database import AlphaDatabase
from src.agent.gp_engine import GPAlphaEngine, GPConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ── Load universe to find which tickers to keep ───────────────────────
print("Loading TOP1000 universe...")
universe_df = pd.read_parquet("data/fmp_cache/universes/TOP1000.parquet")

# Find tickers that are in TOP1000 on at least 50% of days
ticker_coverage = universe_df.sum(axis=0) / len(universe_df)
top1000_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())
print(f"  TOP1000 tickers (>30% coverage): {len(top1000_tickers)}")

# ── Load only needed matrices, trimmed to TOP1000 tickers ────────────
print("Loading matrices from cache (trimmed to TOP1000)...")
import json

matrices_dir = "data/fmp_cache/matrices"
classifications = {}
cls_path = "data/fmp_cache/classifications.json"
if os.path.exists(cls_path):
    with open(cls_path) as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in top1000_tickers}

# Load and trim matrices
matrices = {}
for fname in os.listdir(matrices_dir):
    if not fname.endswith(".parquet") or fname.startswith("_"):
        continue
    field = fname.replace(".parquet", "")
    df = pd.read_parquet(os.path.join(matrices_dir, fname))
    # Trim to TOP1000 tickers only
    valid_cols = [c for c in top1000_tickers if c in df.columns]
    matrices[field] = df[valid_cols]

print(f"  Loaded {len(matrices)} fields, trimmed to {len(top1000_tickers)} tickers")
sample_shape = matrices["close"].shape
print(f"  Matrix shape: {sample_shape[0]} days × {sample_shape[1]} tickers")

# ── Apply universe mask (set non-member to NaN per day) ──────────────
print("Applying universe mask...")
for field, mat in matrices.items():
    if isinstance(mat, pd.DataFrame):
        common_cols = mat.columns.intersection(universe_df.columns)
        common_idx = mat.index.intersection(universe_df.index)
        if len(common_cols) > 0 and len(common_idx) > 0:
            mask = universe_df.loc[common_idx, common_cols]
            matrices[field] = mat.loc[common_idx, common_cols].where(mask)

# ── Build context ────────────────────────────────────────────────────
ctx = InMemoryDataContext()
ctx.load_from_matrices(matrices, classifications)
print(f"  Context: {len(ctx._price_matrices)} fields, {len(ctx._classifications)} classifications")

# ── Configure GP ─────────────────────────────────────────────────────
config = GPConfig(
    population_size=200,
    n_generations=50,
    max_tree_depth=6,
    booksize=20_000_000.0,          # $20MM
    neutralization="subindustry",     # GICS sub-industry neutralization
    delay=1,
    fees_bps=0.0,
    corr_cutoff=0.7,
    lookback_range=40,
    include_advanced=True,
)

db = AlphaDatabase("data/alpha.db")

# ── Build engine ─────────────────────────────────────────────────────
engine = GPAlphaEngine.from_context(ctx, config=config, db=db)

if engine.classifications is not None:
    subind = engine.classifications.get("subindustry")
    if subind is not None:
        n_groups = subind.nunique()
        print(f"  GICS sub-industry groups: {n_groups} unique, {len(subind)} tickers")
    ind = engine.classifications.get("industry")
    if ind is not None:
        print(f"  GICS industry groups: {ind.nunique()} unique")
    sec = engine.classifications.get("sector")
    if sec is not None:
        print(f"  GICS sector groups: {sec.nunique()} unique")
else:
    print("  WARNING: No GICS classifications loaded!")

print(f"  Features ({len(engine.feature_names)}): {engine.feature_names}")
print(f"  Book size: ${config.booksize:,.0f}")
print(f"  Neutralization: {config.neutralization}")
print(f"  Delay: {config.delay}, Fees: {config.fees_bps} bps")

# ── Run GP ───────────────────────────────────────────────────────────
results = engine.run(
    n_generations=config.n_generations,
    population_size=config.population_size,
    verbose=True,
)

print(f"\nBest expression: {results['best_expression']}")
print(f"Best fitness:    {results['best_fitness']:.4f}")
print(f"Trials:          {results['trials_evaluated']}")

if results.get("best_alphas"):
    print(f"\nTop alphas found:")
    top = sorted(results["best_alphas"], key=lambda x: x["fitness"], reverse=True)[:10]
    for i, a in enumerate(top, 1):
        print(f"  {i}. Sharpe={a['sharpe']:+.4f} Fitness={a['fitness']:+.4f} "
              f"T/O={a['turnover']:.3f} | {a['expression'][:70]}")
