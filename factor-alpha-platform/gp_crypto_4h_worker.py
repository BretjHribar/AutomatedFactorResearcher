"""
gp_crypto_4h_worker.py — Genetic Programming alpha discovery for 4h Binance crypto.

Runs a continuous GP loop, inserting qualifying alphas into data/alphas.db
immediately as they are found — never waits for the run to end.

Multiple instances can run simultaneously with different --seed values.
All instances share the same SQLite DB (WAL mode) and diversity check.

Usage:
    # Single instance:
    cd factor-alpha-platform
    python gp_crypto_4h_worker.py

    # Multiple parallel instances (different terminals / tmux panes):
    python gp_crypto_4h_worker.py --seed 1 --max-depth 6
    python gp_crypto_4h_worker.py --seed 2 --max-depth 5
    python gp_crypto_4h_worker.py --seed 3 --max-depth 7 --pop-size 200

Two-stage evaluation:
    Stage 1 (every individual):  DEAP compiled function on DataFrames → inline Sharpe.
                                  Fast, no string parsing overhead.
    Stage 2 (candidates > threshold): eval_full() — full quality gates including
                                  sub-period stability, IC, PnL kurtosis/skew.
                                  Saves to DB immediately on pass.

Quality gates (identical to eval_alpha.py):
    IS Sharpe >= 3.0   Fitness >= 5.0      Turnover <= 0.30
    H1 Sharpe >= 1.0   H2 Sharpe >= 1.0   IC mean >= -0.05
    PnL kurtosis <= 20 PnL skew >= -0.5   Rolling SR std <= 0.05
"""

from __future__ import annotations

import argparse
import math
import operator
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

# ── Project root on sys.path so relative imports work regardless of cwd ────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import eval_alpha                          # access to globals (UNIVERSE, gates, etc.)
from eval_alpha import (
    load_data,
    eval_full,
    save_alpha,
    get_conn,
    ensure_trial_log,
    log_trial,
    process_signal,
    MAX_WEIGHT,
    BARS_PER_DAY,
)
from src.operators import vectorized as ops

DF = pd.core.frame.DataFrame

# ── Pre-filter (looser than quality gates to avoid false negatives) ─────────────
PREFILTER_SHARPE   = 2.5    # min quick Sharpe before running slow eval_full()
PREFILTER_TURNOVER = 0.38   # max turnover before even trying full eval

# ── 4h crypto lookbacks (6=1d, 12=2d, 30=5d, 60=10d, 120=20d, 240=40d bars) ────
CRYPTO_INT_LOOKBACKS = [3, 6, 12, 18, 24, 30, 36, 48, 60, 90, 120, 180, 240]

# Decay_exp takes float alpha but DEAP typed GP can't handle float nodes without
# a float-returning primitive. Wrap it: int n (1–19) → alpha = n * 0.05.
def _decay_exp_int(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return ops.Decay_exp(df, max(0.01, min(0.99, n * 0.05)))


# ──────────────────────────────────────────────────────────────────────────────
# DEAP setup
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_creator() -> None:
    """Create DEAP fitness/individual classes — safe to call multiple times."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def gen_safe(pset, min_: int, max_: int, type_=None) -> list:
    """
    Like DEAP's genGrow but never tries to expand types that have no primitives
    (e.g. int terminals). Fixes 'Cannot choose from an empty sequence' when the
    pset has mixed DF+int types but no int-returning primitives.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while stack:
        depth, cur_type = stack.pop()
        prims = pset.primitives[cur_type]
        terms = pset.terminals[cur_type]
        # Use a terminal when: at max depth, OR this type has no primitives at all
        if depth >= height or not prims:
            expr.append(random.choice(terms))
        else:
            # genGrow: sometimes terminate early (except at root)
            if depth > 0 and random.random() < pset.terminalRatio:
                expr.append(random.choice(terms))
            else:
                prim = random.choice(prims)
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
    return expr


def build_pset(feature_names: list[str], max_depth: int) -> gp.PrimitiveSetTyped:
    """Build typed primitive set using the same operator names as FastExpressionEngine."""
    pset = gp.PrimitiveSetTyped("alpha", [DF] * len(feature_names), DF)

    for x in CRYPTO_INT_LOOKBACKS:
        pset.addTerminal(x, int)

    # (DF, DF) → DF
    pset.addPrimitive(ops.add,         [DF, DF], DF, name="add")
    pset.addPrimitive(ops.subtract,    [DF, DF], DF, name="subtract")
    pset.addPrimitive(ops.multiply,    [DF, DF], DF, name="multiply")
    pset.addPrimitive(ops.true_divide, [DF, DF], DF, name="true_divide")
    pset.addPrimitive(ops.df_max,      [DF, DF], DF, name="df_max")
    pset.addPrimitive(ops.df_min,      [DF, DF], DF, name="df_min")

    # (DF) → DF
    pset.addPrimitive(ops.rank,      [DF], DF, name="rank")
    pset.addPrimitive(ops.Abs,       [DF], DF, name="Abs")
    pset.addPrimitive(ops.negative,  [DF], DF, name="negative")
    pset.addPrimitive(ops.log,       [DF], DF, name="log")
    pset.addPrimitive(ops.sqrt,      [DF], DF, name="sqrt")
    pset.addPrimitive(ops.square,    [DF], DF, name="square")
    pset.addPrimitive(ops.s_log_1p,  [DF], DF, name="s_log_1p")
    pset.addPrimitive(ops.zscore_cs, [DF], DF, name="zscore_cs")
    pset.addPrimitive(ops.normalize, [DF], DF, name="normalize")
    pset.addPrimitive(ops.Sign,      [DF], DF, name="Sign")

    # (DF, int) → DF
    pset.addPrimitive(ops.ts_zscore,   [DF, int], DF, name="ts_zscore")
    pset.addPrimitive(ops.ts_rank,     [DF, int], DF, name="ts_rank")
    pset.addPrimitive(ops.sma,         [DF, int], DF, name="sma")
    pset.addPrimitive(ops.stddev,      [DF, int], DF, name="stddev")
    pset.addPrimitive(ops.ts_min,      [DF, int], DF, name="ts_min")
    pset.addPrimitive(ops.ts_max,      [DF, int], DF, name="ts_max")
    pset.addPrimitive(ops.ts_sum,      [DF, int], DF, name="ts_sum")
    pset.addPrimitive(ops.delta,       [DF, int], DF, name="delta")
    pset.addPrimitive(ops.ts_skewness, [DF, int], DF, name="ts_skewness")
    pset.addPrimitive(ops.ts_kurtosis, [DF, int], DF, name="ts_kurtosis")
    pset.addPrimitive(ops.delay,       [DF, int], DF, name="delay")
    pset.addPrimitive(ops.Decay_lin,   [DF, int], DF, name="Decay_lin")

    # (DF, int) → DF  — Decay_exp wrapped to use int so pset stays pure DF+int
    pset.addPrimitive(_decay_exp_int, [DF, int], DF, name="Decay_exp")

    # (DF, DF, int) → DF  — two-input time series
    pset.addPrimitive(ops.correlation, [DF, DF, int], DF, name="correlation")

    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    return pset


def build_toolbox(pset: gp.PrimitiveSetTyped, max_depth: int) -> base.Toolbox:
    tb = base.Toolbox()
    tb.register("expr",       gen_safe, pset=pset, min_=1, max_=3)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile",    gp.compile, pset=pset)
    tb.register("select",     tools.selTournament, tournsize=7)
    tb.register("mate",       gp.cxOnePoint)
    tb.register("expr_mut",   gen_safe, pset=pset, min_=0, max_=2)
    tb.register("mutate",     gp.mutUniform, expr=tb.expr_mut, pset=pset)
    # Hard tree-depth ceiling on crossover and mutation
    tb.decorate("mate",   gp.staticLimit(operator.attrgetter("height"), max_depth))
    tb.decorate("mutate", gp.staticLimit(operator.attrgetter("height"), max_depth))
    return tb


# ──────────────────────────────────────────────────────────────────────────────
# Worker — holds per-process evaluation state
# ──────────────────────────────────────────────────────────────────────────────

class GPWorker:
    """
    Encapsulates all mutable state for one GP worker process.

    evaluate() is registered as the DEAP fitness function and implements
    the two-stage pipeline: fast inline Sharpe → full quality gates.
    Passing alphas are inserted to the DB immediately.
    """

    _DEDUP_LIMIT = 50_000    # clear seen-set when it grows beyond this

    def __init__(
        self,
        feature_names: list[str],
        feature_dfs:   list[pd.DataFrame],
        universe_df:   pd.DataFrame,
        returns_pct:   pd.DataFrame,
        conn,
        toolbox:       base.Toolbox,
        quick_threshold: float,
        verbose:       bool,
    ) -> None:
        self.feature_names   = feature_names
        self.feature_dfs     = feature_dfs
        self.universe_df     = universe_df
        self.returns_pct     = returns_pct
        self.conn            = conn
        self.toolbox         = toolbox
        self.quick_threshold = quick_threshold
        self.verbose         = verbose

        self.trial_counter = 0
        self.n_full_evals  = 0
        self.n_saved       = 0
        self._seen: set[str] = set()

        # Annualised Sharpe multiplier: sqrt(bars_per_day * 365)
        self._sr_scale = math.sqrt(BARS_PER_DAY * 365)

    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 — fast inline evaluation (DEAP compiled tree, no string parsing)
    # ──────────────────────────────────────────────────────────────────────

    def _fast_eval(self, individual) -> tuple[float, float]:
        """
        Returns (quick_sharpe, turnover).
        quick_sharpe=0.0 signals that the individual should be discarded.
        """
        try:
            func      = self.toolbox.compile(expr=individual)
            alpha_raw = func(*self.feature_dfs)
        except Exception:
            return 0.0, 999.0

        if not isinstance(alpha_raw, pd.DataFrame) or alpha_raw.empty:
            return 0.0, 999.0

        try:
            alpha_raw = alpha_raw.replace([np.inf, -np.inf], np.nan)
            alpha_df  = process_signal(alpha_raw,
                                       universe_df=self.universe_df,
                                       max_wt=MAX_WEIGHT)

            turnover = float(alpha_df.diff().abs().sum(axis=1).mean())
            if not (1e-6 < turnover < PREFILTER_TURNOVER):
                return 0.0, turnover

            # PnL: position[t] × return[t+1]  (close-to-close forward return)
            pnl = (alpha_df * self.returns_pct.shift(-1)).sum(axis=1).dropna()
            if len(pnl) < 100:
                return 0.0, turnover

            sharpe = pnl.mean() / pnl.std() * self._sr_scale
            return (float(sharpe) if math.isfinite(sharpe) else 0.0), turnover

        except Exception:
            return 0.0, 999.0

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 — full quality gate (eval_full + save)
    # ──────────────────────────────────────────────────────────────────────

    def _full_eval_and_save(self, expr_str: str, quick_sharpe: float) -> float:
        """
        Run eval_full() and save if all gates pass.
        Returns the IS Sharpe from the full eval (used as DEAP fitness).
        """
        self.n_full_evals += 1
        print(f"\n  >>> Full eval #{self.n_full_evals} "
              f"[trial {self.trial_counter}] "
              f"quick_sr={quick_sharpe:.2f}: {expr_str[:80]}", flush=True)
        t0 = time.time()

        try:
            result = eval_full(expr_str, self.conn)
        except Exception as e:
            print(f"  eval_full exception: {e}", flush=True)
            return quick_sharpe

        if not result["success"]:
            print(f"  FAILED: {result['error']}", flush=True)
            log_trial(self.conn, expr_str, 0.0, saved=False)
            return 0.0

        log_trial(self.conn, expr_str, result["is_sharpe"], saved=False)

        min_sub  = min(result["stability_h1"], result["stability_h2"])
        gates = {
            f"SR>={eval_alpha.MIN_IS_SHARPE}":
                result["is_sharpe"]       >= eval_alpha.MIN_IS_SHARPE,
            f"Fit>={eval_alpha.MIN_FITNESS}":
                result["is_fitness"]      >= eval_alpha.MIN_FITNESS,
            f"TO<={eval_alpha.MAX_TURNOVER}":
                result["turnover"]        <= eval_alpha.MAX_TURNOVER,
            f"IC>={eval_alpha.MIN_IC_MEAN}":
                result["ic_mean"]         >= eval_alpha.MIN_IC_MEAN,
            "H1>0":
                result["stability_h1"]    > 0,
            "H2>0":
                result["stability_h2"]    > 0,
            f"minSub>={eval_alpha.MIN_SUB_SHARPE}":
                min_sub                   >= eval_alpha.MIN_SUB_SHARPE,
            f"Kurt<={eval_alpha.MAX_PNL_KURTOSIS}":
                result["pnl_kurtosis"]    <= eval_alpha.MAX_PNL_KURTOSIS,
            f"RollSRstd<={eval_alpha.MAX_ROLLING_SR_STD}":
                result["rolling_sr_std"]  <= eval_alpha.MAX_ROLLING_SR_STD,
            f"Skew>={eval_alpha.MIN_PNL_SKEW}":
                result["pnl_skew"]        >= eval_alpha.MIN_PNL_SKEW,
        }
        all_pass = all(gates.values())
        failed   = [k for k, v in gates.items() if not v]

        elapsed = time.time() - t0
        summary = (f"SR={result['is_sharpe']:+.2f} Fit={result['is_fitness']:.2f} "
                   f"TO={result['turnover']:.3f} "
                   f"H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f} "
                   f"Kurt={result['pnl_kurtosis']:.1f} Skew={result['pnl_skew']:+.2f}")
        verdict = "ALL PASS" if all_pass else f"FAIL [{', '.join(failed)}]"
        print(f"  [{elapsed:.1f}s] {summary} -> {verdict}", flush=True)

        if all_pass:
            saved = save_alpha(self.conn, expr_str,
                               "GP genetic programming discovery", result)
            if saved:
                self.n_saved += 1
                print(f"\n  *** INSERTED alpha #{self.n_saved}: {expr_str}\n", flush=True)
            else:
                print(f"  diversity check failed - not inserted", flush=True)

            # Penalise slightly to encourage the population to explore new territory
            return result["is_sharpe"] * 0.5

        return max(0.0, result["is_sharpe"])

    # ──────────────────────────────────────────────────────────────────────
    # DEAP fitness function
    # ──────────────────────────────────────────────────────────────────────

    def evaluate(self, individual) -> tuple[float]:
        """Called by DEAP for every individual. Returns (fitness,) tuple."""
        self.trial_counter += 1

        # Stage 1
        quick_sharpe, turnover = self._fast_eval(individual)

        if self.verbose and self.trial_counter % 100 == 0:
            expr_str = str(individual)
            print(f"  [t={self.trial_counter}] "
                  f"qs={quick_sharpe:.2f} to={turnover:.3f} "
                  f"| {expr_str[:55]}", flush=True)

        if quick_sharpe < self.quick_threshold:
            return (max(0.0, quick_sharpe),)

        # Dedup: skip if we've already run a full eval for this exact expression
        expr_str = str(individual)
        if expr_str in self._seen:
            return (max(0.0, quick_sharpe),)
        if len(self._seen) > self._DEDUP_LIMIT:
            self._seen.clear()   # flush to avoid unbounded growth
        self._seen.add(expr_str)

        # Stage 2 — full quality gate
        fitness = self._full_eval_and_save(expr_str, quick_sharpe)
        return (fitness,)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous GP alpha discovery for 4h crypto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--universe",        default="BINANCE_TOP50",
                        help="Data universe (BINANCE_TOP50, BINANCE_TOP100, KUCOIN_TOP50, ...)")
    parser.add_argument("--seed",            type=int, default=None,
                        help="RNG seed. Default: os.getpid()+time — unique per instance")
    parser.add_argument("--max-depth",       type=int, default=4,
                        help="Hard tree-depth ceiling enforced on mate and mutate")
    parser.add_argument("--pop-size",        type=int, default=300,
                        help="Number of individuals in the population")
    parser.add_argument("--cxpb",            type=float, default=0.7,
                        help="Crossover probability per generation step")
    parser.add_argument("--mutpb",           type=float, default=0.1,
                        help="Mutation probability per generation step")
    parser.add_argument("--quick-threshold", type=float, default=2.5,
                        help="Min fast Sharpe before running slow eval_full()")
    parser.add_argument("--verbose",         action="store_true",
                        help="Print every-100-trial progress line")
    parser.add_argument("--generations",     type=int, default=None,
                        help="Stop after N generations (default: run indefinitely)")
    args = parser.parse_args()

    # Unique seed per instance when not specified
    seed = (args.seed if args.seed is not None
            else (os.getpid() * 1000 + int(time.time())) % (2 ** 31))
    random.seed(seed)
    np.random.seed(seed)

    # ── Configure eval_alpha module globals for this universe ────────────────
    eval_alpha.UNIVERSE = args.universe
    if args.universe.upper().startswith("KUCOIN"):
        eval_alpha.EXCHANGE    = "kucoin"
        eval_alpha.TRAIN_START = "2023-09-01"
        eval_alpha.TRAIN_END   = "2025-09-01"
        eval_alpha.SUBPERIODS  = [
            ("2023-09-01", "2024-09-01", "H1"),
            ("2024-09-01", "2025-09-01", "H2"),
        ]
    eval_alpha._DATA_CACHE.clear()   # ensure fresh load with new settings

    print(f"\n{'='*64}", flush=True)
    print(f"  GP Crypto 4h Worker  |  pid={os.getpid()}", flush=True)
    print(f"  universe={args.universe}  seed={seed}  max_depth={args.max_depth}", flush=True)
    print(f"  pop_size={args.pop_size}  cxpb={args.cxpb}  mutpb={args.mutpb}", flush=True)
    print(f"  quick_threshold={args.quick_threshold}  "
          f"verbose={args.verbose}", flush=True)
    print(f"{'='*64}", flush=True)

    # ── Load train data (cached after first call) ────────────────────────────
    print("\nLoading train data...", flush=True)
    matrices, universe_df = load_data("train")

    # Drop fields that are near-perfect linear transforms of others
    SKIP_FIELDS = {"log_returns"}
    feature_names = sorted(k for k in matrices if k not in SKIP_FIELDS)
    feature_dfs   = [matrices[k] for k in feature_names]
    close         = matrices["close"]
    returns_pct   = close.pct_change()

    print(f"  {len(feature_names)} features:", flush=True)
    for i in range(0, len(feature_names), 8):
        print(f"    {', '.join(feature_names[i:i+8])}", flush=True)

    # ── DEAP objects ─────────────────────────────────────────────────────────
    _ensure_creator()
    pset    = build_pset(feature_names, args.max_depth)
    toolbox = build_toolbox(pset, args.max_depth)

    # ── SQLite connection — WAL mode allows concurrent readers + 1 writer ────
    os.makedirs(os.path.join(_HERE, "data"), exist_ok=True)
    conn = get_conn()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # wait up to 30 s on write lock
    ensure_trial_log(conn)
    conn.commit()

    # ── Worker ───────────────────────────────────────────────────────────────
    worker = GPWorker(
        feature_names    = feature_names,
        feature_dfs      = feature_dfs,
        universe_df      = universe_df,
        returns_pct      = returns_pct,
        conn             = conn,
        toolbox          = toolbox,
        quick_threshold  = args.quick_threshold,
        verbose          = args.verbose,
    )
    toolbox.register("evaluate", worker.evaluate)

    # ── Statistics ───────────────────────────────────────────────────────────
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(10)

    n_gens = args.generations or 10_000_000   # effectively infinite

    print(f"\nPre-filter:    quick_sharpe >= {args.quick_threshold}  "
          f"turnover < {PREFILTER_TURNOVER}", flush=True)
    print(f"Quality gates: IS Sharpe >= {eval_alpha.MIN_IS_SHARPE}  "
          f"Fitness >= {eval_alpha.MIN_FITNESS}  "
          f"Turnover <= {eval_alpha.MAX_TURNOVER}", flush=True)
    print(f"\nStarting GP evolution - press Ctrl+C to stop cleanly.\n", flush=True)

    t_start = time.time()
    pop = toolbox.population(n=args.pop_size)

    try:
        pop, _log = algorithms.eaSimple(
            pop, toolbox,
            cxpb     = args.cxpb,
            mutpb    = args.mutpb,
            ngen     = n_gens,
            stats    = stats,
            halloffame = hof,
            verbose  = True,
        )
    except KeyboardInterrupt:
        elapsed = (time.time() - t_start) / 60
        print(f"\n[Interrupted - {worker.trial_counter} trials in {elapsed:.1f} min]", flush=True)

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}", flush=True)
    print(f"  Trials evaluated : {worker.trial_counter}", flush=True)
    print(f"  Full evals run   : {worker.n_full_evals}", flush=True)
    print(f"  Alphas inserted  : {worker.n_saved}", flush=True)
    if hof:
        print(f"  Best individual  : {str(hof[0])[:100]}", flush=True)
    print(f"{'='*64}", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
