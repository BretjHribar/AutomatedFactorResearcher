"""
Continue GP pipeline from existing database — faster settings.
Keeps existing 42 alphas, runs more rounds with smaller/faster GP until we hit target.
Then runs optimizer and OOS evaluation.
"""
import sys, os, json, time, logging, sqlite3
import numpy as np
import pandas as pd

from src.data.context_research import InMemoryDataContext
from src.data.alpha_database import AlphaDatabase
from src.agent.gp_engine import GPAlphaEngine, GPConfig
from src.simulation.vectorized_sim import simulate_vectorized
from src.simulation.oos import fixed_split_oos
from src.portfolio.optimizer import PortfolioOptimizer
from src.operators.fastexpression import FastExpressionEngine

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ══════════════════════════════════════════════════════════════════════
IS_START = "2017-01-01"
IS_END   = "2022-12-31"
OOS_START = "2023-01-01"
SPLIT_DATE = "2023-01-01"
UNIVERSE = "TOP3000"
TARGET_ALPHAS = 25
MIN_SHARPE = 1.0          # Lowered from 1.25 — we already have 11 above 1.0
DELAY = 1
NEUTRALIZATION = "subindustry"
BOOKSIZE = 20_000_000.0
DB_PATH = "data/alpha_gp_pipeline.db"
MATRICES_DIR = "data/fmp_cache/matrices"
CLS_PATH = "data/fmp_cache/classifications.json"

# ══════════════════════════════════════════════════════════════════════
# Load existing qualifying alphas from DB
# ══════════════════════════════════════════════════════════════════════
db = AlphaDatabase(DB_PATH)
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute('''
    SELECT a.expression, e.sharpe, e.fitness, e.turnover, e.returns_ann, e.max_drawdown, e.margin_bps
    FROM alphas a JOIN evaluations e ON a.alpha_id = e.alpha_id
    WHERE e.sharpe >= ?
    ORDER BY e.sharpe DESC
''', (MIN_SHARPE,))
existing_qualifying = [
    {"expression": r[0], "sharpe": r[1], "fitness": r[2], "turnover": r[3],
     "returns_ann": r[4] or 0, "max_drawdown": r[5] or 0, "margin_bps": r[6] or 0}
    for r in cur.fetchall()
]
conn.close()

print(f"  Existing qualifying alphas (Sharpe > {MIN_SHARPE}): {len(existing_qualifying)}")
for i, a in enumerate(existing_qualifying, 1):
    print(f"    {i:2d}. Sharpe={a['sharpe']:+.3f} | {a['expression'][:70]}")

qualifying_alphas = list(existing_qualifying)
still_needed = TARGET_ALPHAS - len(qualifying_alphas)
print(f"\n  Need {still_needed} more alphas to reach target of {TARGET_ALPHAS}")

if still_needed <= 0:
    print(f"  ✅ Already have {len(qualifying_alphas)} qualifying alphas!")
else:
    # ══════════════════════════════════════════════════════════════════════
    # Load data (same as before)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Loading data...")
    print("=" * 70)
    
    universe_df = pd.read_parquet(f"data/fmp_cache/universes/{UNIVERSE}.parquet")
    universe_is = universe_df.loc[IS_START:IS_END]
    ticker_coverage = universe_is.sum(axis=0) / len(universe_is)
    universe_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())
    print(f"  {UNIVERSE}: {len(universe_tickers)} tickers")
    
    classifications = {}
    if os.path.exists(CLS_PATH):
        with open(CLS_PATH) as f:
            all_cls = json.load(f)
        classifications = {k: v for k, v in all_cls.items() if k in universe_tickers}
    
    # Load ONLY the GP feature set fields to save memory/time
    from src.agent.gp_engine import GPAlphaEngine as _GPE
    gp_fields = set(_GPE.GP_FEATURE_SET + ["close", "open", "volume", "returns", "log_returns"])
    
    matrices_is = {}
    for fname in sorted(os.listdir(MATRICES_DIR)):
        if not fname.endswith(".parquet") or fname.startswith("_"):
            continue
        field = fname.replace(".parquet", "")
        if field not in gp_fields:
            continue
        df = pd.read_parquet(os.path.join(MATRICES_DIR, fname))
        valid_cols = [c for c in universe_tickers if c in df.columns]
        if len(valid_cols) > 0:
            trimmed = df[valid_cols].loc[IS_START:IS_END]
            if len(trimmed) > 0:
                matrices_is[field] = trimmed
    
    # Apply universe mask
    for field, mat in matrices_is.items():
        if isinstance(mat, pd.DataFrame) and mat.shape[1] > 1:
            common_cols = mat.columns.intersection(universe_is.columns)
            common_idx = mat.index.intersection(universe_is.index)
            if len(common_cols) > 0 and len(common_idx) > 0:
                mask = universe_is.loc[common_idx, common_cols]
                matrices_is[field] = mat.loc[common_idx, common_cols].where(mask)
    
    print(f"  Loaded {len(matrices_is)} fields, shape: {matrices_is['close'].shape}")
    
    ctx = InMemoryDataContext()
    ctx.load_from_matrices(matrices_is, classifications)
    
    # ── Faster GP config ──
    config = GPConfig(
        population_size=200,     # smaller pop
        n_generations=25,        # fewer gens — each round ~30 min
        max_tree_depth=5,        # slightly shallower trees (faster eval)
        booksize=BOOKSIZE,
        neutralization=NEUTRALIZATION,
        delay=DELAY,
        fees_bps=0.0,
        corr_cutoff=0.6,
        lookback_range=40,
        include_advanced=True,
        fitness_cutoff=0.3,
    )
    
    round_num = 1  # Start counting from R1 since we already had R1 from before
    while len(qualifying_alphas) < TARGET_ALPHAS:
        round_num += 1
        print(f"\n  ── GP Round {round_num} ({len(qualifying_alphas)}/{TARGET_ALPHAS} qualifying) ──")
        
        engine = GPAlphaEngine.from_context(ctx, config=config, db=db)
        results = engine.run(
            n_generations=config.n_generations,
            population_size=config.population_size,
            seed=round_num * 137,
            verbose=False,
        )
        
        new_this_round = 0
        for alpha in results.get("best_alphas", []):
            if alpha["sharpe"] >= MIN_SHARPE:
                expr = alpha["expression"]
                if expr not in [a["expression"] for a in qualifying_alphas]:
                    qualifying_alphas.append(alpha)
                    new_this_round += 1
                    print(f"    ✅ #{len(qualifying_alphas)}: Sharpe={alpha['sharpe']:+.3f} "
                          f"T/O={alpha['turnover']:.3f} | {expr[:70]}")
        
        print(f"    +{new_this_round} new qualifying this round → {len(qualifying_alphas)}/{TARGET_ALPHAS}")
        
        if round_num > 20:
            print(f"\n  ⚠️ Max rounds reached. Continuing with {len(qualifying_alphas)} alphas.")
            break

print(f"\n  🎯 Total qualifying alphas: {len(qualifying_alphas)}")

# ══════════════════════════════════════════════════════════════════════
# OOS Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  OOS Evaluation (IS: 2017-2022, OOS: 2023+)")
print("=" * 70)

# Load full data
universe_df = pd.read_parquet(f"data/fmp_cache/universes/{UNIVERSE}.parquet")
universe_is_full = universe_df.loc[IS_START:IS_END]
ticker_coverage = universe_is_full.sum(axis=0) / len(universe_is_full)
universe_tickers = sorted(ticker_coverage[ticker_coverage > 0.3].index.tolist())

if not os.path.exists(CLS_PATH):
    classifications = {}
else:
    with open(CLS_PATH) as f:
        all_cls = json.load(f)
    classifications = {k: v for k, v in all_cls.items() if k in universe_tickers}

matrices_full = {}
for fname in sorted(os.listdir(MATRICES_DIR)):
    if not fname.endswith(".parquet") or fname.startswith("_"):
        continue
    field = fname.replace(".parquet", "")
    df = pd.read_parquet(os.path.join(MATRICES_DIR, fname))
    valid_cols = [c for c in universe_tickers if c in df.columns]
    if len(valid_cols) > 0:
        matrices_full[field] = df[valid_cols]

# Apply mask
for field, mat in matrices_full.items():
    if isinstance(mat, pd.DataFrame) and mat.shape[1] > 1:
        common_cols = mat.columns.intersection(universe_df.columns)
        common_idx = mat.index.intersection(universe_df.index)
        if len(common_cols) > 0 and len(common_idx) > 0:
            mask = universe_df.loc[common_idx, common_cols]
            matrices_full[field] = mat.loc[common_idx, common_cols].where(mask)

cls_series = {}
for level in ["sector", "industry", "subindustry"]:
    mapping = {}
    for sym, cls_data in classifications.items():
        if isinstance(cls_data, dict):
            mapping[sym] = cls_data.get(level, "Unknown")
    if mapping:
        cls_series[level] = pd.Series(mapping)

full_returns = matrices_full["close"].pct_change().shift(-1)
full_close = matrices_full["close"]
full_open = matrices_full.get("open")

expr_engine = FastExpressionEngine(data_fields=matrices_full)
for group_name, group_series in cls_series.items():
    expr_engine.add_group(group_name, group_series)

oos_results = []
for i, alpha in enumerate(qualifying_alphas, 1):
    expr = alpha["expression"]
    try:
        alpha_df = expr_engine.evaluate(expr)
        oos_result = fixed_split_oos(
            alpha_df=alpha_df, returns_df=full_returns,
            close_df=full_close, open_df=full_open,
            classifications=cls_series, split_date=SPLIT_DATE,
            booksize=BOOKSIZE, delay=DELAY, neutralization=NEUTRALIZATION,
        )
        oos_results.append({
            "expression": expr,
            "is_sharpe": oos_result.is_sharpe,
            "oos_sharpe": oos_result.oos_sharpe,
            "sharpe_decay": oos_result.sharpe_decay,
            "is_consistent": oos_result.is_consistent,
            "is_sim": oos_result.is_sim,
            "oos_sim": oos_result.oos_sim,
        })
        s = "✅" if oos_result.oos_sharpe > 0.5 else "⚠️" if oos_result.oos_sharpe > 0 else "❌"
        print(f"  {s} {i:2d}: IS={oos_result.is_sharpe:+.2f}  OOS={oos_result.oos_sharpe:+.2f}  "
              f"decay={oos_result.sharpe_decay:.0%}  | {expr[:55]}")
    except Exception as e:
        print(f"  ❌ {i:2d}: FAILED — {e}")
        oos_results.append(None)

valid_oos = [(i,r) for i,r in enumerate(oos_results) if r and r["oos_sharpe"]>0]
print(f"\n  Positive OOS: {len(valid_oos)}/{len(qualifying_alphas)}")

# ══════════════════════════════════════════════════════════════════════
# Portfolio Optimization
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Portfolio Optimization (IS period)")
print("=" * 70)

optimizer = PortfolioOptimizer(booksize=BOOKSIZE)
for i, result in enumerate(oos_results):
    if result is None or result["is_sim"] is None:
        continue
    optimizer.add_alpha(
        name=f"alpha_{i+1}",
        daily_pnl=result["is_sim"].daily_pnl,
        sharpe=result["is_sim"].sharpe,
        expression=qualifying_alphas[i]["expression"],
    )

print(f"  Added {optimizer.n_alphas} alphas")

if optimizer.n_alphas >= 2:
    all_results = optimizer.optimize_all()
    print(f"\n  {'Method':<20} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
    print(f"  {'-'*48}")
    for method_name, r in all_results.items():
        print(f"  {method_name:<20} {r.sharpe:>+8.2f} {r.returns_ann:>9.1%} {r.max_drawdown:>7.1%}")
    
    best_method = max(all_results, key=lambda m: all_results[m].sharpe)
    best = all_results[best_method]
    print(f"\n  🏆 Best: {best_method} (IS Sharpe={best.sharpe:.2f})")
    
    # Weights
    print(f"\n  Weights:")
    for name, w in sorted(best.weights.items(), key=lambda x: -x[1]):
        if w > 0.01:
            idx = int(name.split("_")[1]) - 1
            e = qualifying_alphas[idx]["expression"][:50] if idx < len(qualifying_alphas) else "?"
            print(f"    {name}: {w:.1%}  | {e}")
    
    # OOS portfolio
    print("\n" + "=" * 70)
    print("  OOS Portfolio")
    print("=" * 70)
    
    opt_oos = PortfolioOptimizer(booksize=BOOKSIZE)
    for i, result in enumerate(oos_results):
        if result is None or result["oos_sim"] is None:
            continue
        opt_oos.add_alpha(
            name=f"alpha_{i+1}",
            daily_pnl=result["oos_sim"].daily_pnl,
            sharpe=result["oos_sim"].sharpe,
            expression=qualifying_alphas[i]["expression"],
        )
    
    if opt_oos.n_alphas >= 2:
        oos_port = opt_oos._evaluate_portfolio(best.weights, f"{best_method}_oos")
        print(f"    IS Sharpe:  {best.sharpe:+.2f}")
        print(f"    OOS Sharpe: {oos_port.sharpe:+.2f}")
        print(f"    Decay:      {(1-oos_port.sharpe/best.sharpe) if best.sharpe else 0:.0%}")
        print(f"    OOS Return: {oos_port.returns_ann:+.1%}")
        print(f"    OOS MaxDD:  {oos_port.max_drawdown:.1%}")

# Save
results_path = "data/gp_pipeline_results.json"
summary = {"alphas": []}
for i, (alpha, oos) in enumerate(zip(qualifying_alphas, oos_results)):
    entry = {"expression": alpha["expression"], "is_sharpe": alpha["sharpe"]}
    if oos:
        entry["oos_sharpe"] = oos["oos_sharpe"]
        entry["sharpe_decay"] = oos["sharpe_decay"]
    summary["alphas"].append(entry)
with open(results_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

db.close()
print(f"\n  ✅ Pipeline complete! Results → {results_path}")
