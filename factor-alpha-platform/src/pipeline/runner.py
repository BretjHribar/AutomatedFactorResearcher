"""
Unified config-driven research pipeline.

Single entry point: `run(config_path) -> PipelineResult`.

Pipeline stages (every stage is config-driven):
    0. load matrices + universe per `data`
    1. load alpha expressions per `alpha_source` (+ optional `train_sharpes`)
    2. preprocess each alpha per `preprocessing` (apply_preprocess)
    3. combine per `combiner` (combiners.py)
    4. post-combiner re-normalize per `post_combiner`
    5. (optional) QP per `risk_model` + `qp` (src.portfolio.qp.run_walkforward)
    6. fees per `fees` (src.pipeline.fees)
    7. metrics per `splits` + `annualization`

Two markets supported today: equity (daily, fundamentals + style factors) and
crypto (4h, no fundamentals). The same JSON shape covers both — see
prod/config/research_equity.json and prod/config/research_crypto.json.
"""
from __future__ import annotations
import json, sqlite3, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root for `src.*` imports when invoked as a script
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.preprocessing import apply_preprocess
from src.portfolio.combiners import (
    combiner_equal, combiner_adaptive, combiner_risk_parity, combiner_billions,
    combiner_ic_weighted, combiner_sharpe_weighted, combiner_topn_sharpe,
    combiner_topn_train,
)
from src.portfolio.risk_model import (
    build_diagonal, build_pca, build_style, build_style_pca, build_style_factors,
)
from src.portfolio.qp import run_walkforward
from src.pipeline.fees import make_cost_fn


@dataclass
class PipelineResult:
    config: dict
    alpha_signals_n: int
    universe_size: int
    n_bars: int
    weights: pd.DataFrame
    gross_pnl: pd.Series
    cost: pd.Series
    net_pnl: pd.Series
    metrics: Dict[str, Dict[str, float]]
    elapsed_sec: float
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 0 — data
# ---------------------------------------------------------------------------

def _load_universe_and_matrices(cfg: dict, *, root: Path):
    data = cfg["data"]
    matrices_dir = root / data["matrices_dir"]
    uni = pd.read_parquet(root / data["universe_path"])
    if uni.dtypes.iloc[0] != bool:
        uni = uni.astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)

    uf = data["universe_filter"]
    if uf["method"] == "coverage":
        cov = uni.sum(axis=0) / len(uni)
        valid = sorted(cov[cov > float(uf["threshold"])].index.tolist())
    else:
        raise ValueError(f"unknown universe_filter method {uf['method']!r}")
    uni = uni[valid]
    dates = uni.index
    tickers = uni.columns.tolist()

    mats = {}
    for fp in sorted(matrices_dir.glob("*.parquet")):
        if fp.stem.startswith("_"):
            continue
        # Skip nested subdirs (e.g. matrices/4h/prod) when the cfg points at
        # the parent dir.
        if fp.parent != matrices_dir:
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc:
            mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(
                index=dates, columns=tickers)

    close = mats["close"]
    rsource = data.get("returns_source", "compute_from_close")
    if rsource == "compute_from_close":
        ret = close.pct_change(fill_method=None)
    elif rsource.startswith("matrix:"):
        ret = mats[rsource.split(":", 1)[1]].reindex(index=dates, columns=tickers)
    else:
        raise ValueError(f"unknown returns_source {rsource!r}")

    classifications = None
    if cfg.get("preprocessing", {}).get("demean_method") == "subindustry":
        sub_field = cfg["preprocessing"].get("subindustry_field", "subindustry")
        path = matrices_dir / f"{sub_field}.parquet"
        classifications = pd.read_parquet(path).iloc[-1].reindex(tickers)

    return uni, dates, tickers, mats, close, ret, classifications


# ---------------------------------------------------------------------------
# Stage 1 — alpha source
# ---------------------------------------------------------------------------

def _load_alphas(cfg: dict, *, root: Path):
    src = cfg["alpha_source"]
    db = sqlite3.connect(root / src["db_path"])
    table = src["table"]
    where = src.get("filter_sql", "1=1")
    rows = db.execute(f'SELECT id, expression FROM "{table}" WHERE {where}').fetchall()

    train_sharpes = {}
    sh_table = src.get("train_sharpe_table")
    sh_col = src.get("train_sharpe_column")
    if sh_table and sh_col:
        for r in db.execute(
            f'SELECT alpha_id, "{sh_col}" FROM "{sh_table}" '
            f'WHERE "{sh_col}" IS NOT NULL').fetchall():
            train_sharpes[r[0]] = float(r[1])
    db.close()
    return rows, train_sharpes


# ---------------------------------------------------------------------------
# Stages 2-3-4 — preprocess + combine + post-process
# ---------------------------------------------------------------------------

_COMBINERS = {
    "equal":       combiner_equal,
    "adaptive":    combiner_adaptive,
    "risk_par":    combiner_risk_parity,
    "billions":    combiner_billions,
    "ic_wt":       combiner_ic_weighted,
    "sharpe_wt":   combiner_sharpe_weighted,
    "topn_sharpe": combiner_topn_sharpe,
    "topn_train":  combiner_topn_train,
}


def _build_combined(cfg, alpha_signals, train_sharpes, mats, uni, ret):
    cb = cfg["combiner"]
    name = cb["name"]
    params = dict(cb.get("params", {}))
    if name not in _COMBINERS:
        raise ValueError(f"unknown combiner {name!r}")
    if name == "topn_train":
        params["train_sharpes"] = train_sharpes
    return _COMBINERS[name](alpha_signals, mats, uni, ret, **params)


def _post_combiner(combined, cfg, dates, tickers):
    pc = cfg.get("post_combiner", {})
    out = combined.reindex(index=dates, columns=tickers).fillna(0)
    if pc.get("renormalize_l1", False):
        gross = out.abs().sum(axis=1).replace(0, np.nan)
        out = out.div(gross, axis=0)
    clip = pc.get("clip_max_w")
    if clip is not None:
        out = out.clip(lower=-float(clip), upper=float(clip))
    return out.fillna(0)


# ---------------------------------------------------------------------------
# Stage 5 — risk model + QP
# ---------------------------------------------------------------------------

def _build_risk_model_fn(name, params, mats, dates, tickers):
    if name == "diagonal":
        def fn(i, idx, vol_today, ret_mat, factor_window):
            return build_diagonal(vol_today)
        return fn

    n_pca = int(params.get("n_pca_factors", 5))
    if name == "pca":
        def fn(i, idx, vol_today, ret_mat, factor_window):
            if i < factor_window + 1:
                return build_diagonal(vol_today)
            R = ret_mat[i - factor_window:i, idx]
            R = np.where(np.isfinite(R), R, 0.0)
            return build_pca(R, n_pca)
        return fn

    # style or style+pca — need precomputed style stack
    style_factors = build_style_factors(mats)
    factor_names = list(style_factors.keys())
    n_dates = len(dates); n_tickers = len(tickers); n_factors = len(factor_names)
    B_stack = np.full((n_dates, n_tickers, n_factors), np.nan, dtype=np.float32)
    for k, fname in enumerate(factor_names):
        df = style_factors[fname].reindex(index=dates, columns=tickers)
        B_stack[:, :, k] = df.values.astype(np.float32)

    if name == "style":
        def fn(i, idx, vol_today, ret_mat, factor_window):
            if i < factor_window + 1:
                return build_diagonal(vol_today)
            R = ret_mat[i - factor_window:i, idx]
            R = np.where(np.isfinite(R), R, 0.0)
            B = B_stack[i, idx, :]
            ok = np.all(np.isfinite(B), axis=0)
            if ok.sum() == 0:
                return build_pca(R, n_pca)
            return build_style(R, B[:, ok])
        return fn
    if name == "style+pca":
        def fn(i, idx, vol_today, ret_mat, factor_window):
            if i < factor_window + 1:
                return build_diagonal(vol_today)
            R = ret_mat[i - factor_window:i, idx]
            R = np.where(np.isfinite(R), R, 0.0)
            B = B_stack[i, idx, :]
            ok = np.all(np.isfinite(B), axis=0)
            if ok.sum() == 0:
                return build_pca(R, n_pca)
            return build_style_pca(R, B[:, ok], n_pca)
        return fn
    raise ValueError(f"unknown risk_model {name!r}")


# ---------------------------------------------------------------------------
# Stage 7 — metrics
# ---------------------------------------------------------------------------

def _split_metrics(g, n, w, *, train_end, val_end, bars_per_year):
    ann = float(np.sqrt(bars_per_year))
    to = w.diff().abs().sum(axis=1).mean() / 2
    out = {"_turnover_per_bar": float(to)}
    splits = [
        ("TRAIN",    slice(None, train_end)),
        ("VAL",      slice(train_end, val_end)),
        ("TEST",     slice(val_end, None)),
        ("VAL+TEST", slice(train_end, None)),
        ("FULL",     slice(None, None)),
    ]
    for lab, sl in splits:
        gg = g.loc[sl]; nn = n.loc[sl]
        sr_g = gg.mean() / gg.std() * ann if gg.std() > 0 else float("nan")
        sr_n = nn.mean() / nn.std() * ann if nn.std() > 0 else float("nan")
        ret_g = gg.mean() * bars_per_year
        ret_n = nn.mean() * bars_per_year
        # Max drawdown — peak-to-trough on the cumulative compounding curve.
        eq_g = (1.0 + gg).cumprod()
        eq_n = (1.0 + nn).cumprod()
        mdd_g = float((eq_g / eq_g.cummax() - 1.0).min()) if len(eq_g) else float("nan")
        mdd_n = float((eq_n / eq_n.cummax() - 1.0).min()) if len(eq_n) else float("nan")
        out[lab] = {
            "n_bars":  int(len(gg)),
            "SR_gross": float(sr_g),
            "SR_net":   float(sr_n),
            "ret_ann_gross": float(ret_g),
            "ret_ann_net":   float(ret_n),
            "max_dd_gross":  mdd_g,
            "max_dd_net":    mdd_n,
        }
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def merge_overrides(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of base, returning a new dict.

    Dict values are merged recursively; lists/scalars are replaced.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_overrides(out[k], v)
        else:
            out[k] = v
    return out


def run(config: str | Path | dict, *, root: Optional[Path] = None,
        verbose: bool = True) -> PipelineResult:
    t0 = time.time()
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        cfg = json.loads(config_path.read_text())
        notes = [f"config={config_path}"]
    elif isinstance(config, dict):
        cfg = config
        notes = ["config=<inline dict>"]
    else:
        raise TypeError(f"config must be path or dict, got {type(config).__name__}")
    if root is None:
        root = ROOT
    bars_per_year = int(cfg["annualization"]["bars_per_year"])

    # Stage 0
    uni, dates, tickers, mats, close, ret, classifications = \
        _load_universe_and_matrices(cfg, root=root)
    if verbose:
        print(f"[0] {len(tickers)} tickers, {len(dates)} bars  ({dates[0]} -> {dates[-1]})", flush=True)

    # Stage 1
    rows, train_sharpes = _load_alphas(cfg, root=root)
    if verbose:
        print(f"[1] {len(rows)} alpha expressions, {len(train_sharpes)} train_sharpes", flush=True)

    engine = FastExpressionEngine(data_fields=mats)
    pre = cfg["preprocessing"]
    alpha_signals = {}
    n_skip = 0
    for aid, expr in rows:
        try:
            sig = engine.evaluate(expr).reindex(index=dates, columns=tickers)
            alpha_signals[aid] = apply_preprocess(
                sig,
                universe_mask=bool(pre.get("universe_mask", False)),
                universe=uni if pre.get("universe_mask", False) else None,
                demean_method=pre.get("demean_method", "cross_section"),
                classifications=classifications,
                normalize=pre.get("normalize", "l1"),
                clip_max_w=pre.get("clip_max_w"),
            )
        except Exception as e:
            n_skip += 1
            if n_skip <= 5 and verbose:
                print(f"  skip a{aid}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    if n_skip and verbose:
        print(f"  ({n_skip} alphas failed eval; using {len(alpha_signals)})", flush=True)

    # Stage 3 — combine
    combined = _build_combined(cfg, alpha_signals, train_sharpes, mats, uni, ret)
    # Stage 4 — post-combiner
    weights = _post_combiner(combined, cfg, dates, tickers)

    # Stage 5 — optional QP
    qp_cfg = cfg.get("qp", {})
    if qp_cfg.get("enabled", False):
        rm = cfg["risk_model"]
        rfn = _build_risk_model_fn(rm["name"], rm.get("params", {}), mats, dates, tickers)
        adv_cap = qp_cfg.get("adv_cap")
        adv_df = mats[adv_cap["adv_field"]] if adv_cap else None
        weights = run_walkforward(
            weights, close, ret, uni, rfn,
            lambda_risk=float(qp_cfg.get("lambda_risk", 5.0)),
            kappa_tc=float(qp_cfg.get("kappa_tc", 30.0)),
            max_w=float(qp_cfg.get("max_w", 0.02)),
            commission_per_share=float(qp_cfg.get("commission_per_share", 0.0)),
            impact_bps=float(qp_cfg.get("impact_bps", 0.0)),
            vol_window=int(rm.get("params", {}).get("vol_window", 60)),
            factor_window=int(rm.get("params", {}).get("factor_window", 126)),
            dollar_neutral=bool(qp_cfg.get("dollar_neutral", True)),
            max_gross_leverage=float(qp_cfg.get("max_gross_leverage", 1.0)),
            adv=adv_df,
            book=float(cfg.get("book", 0)) if adv_cap else None,
            moc_frac=float(adv_cap.get("moc_frac", 0.10)) if adv_cap else 0.10,
            max_moc_participation=float(adv_cap.get("max_part", 0.30)) if adv_cap else 0.30,
            label="qp", verbose=verbose,
        )

    # Stage 6 — fees + PnL
    book = float(cfg.get("book", 1.0))
    cost_fn = make_cost_fn(cfg["fees"]["model"], cfg["fees"]["params"],
                            bars_per_year=bars_per_year)
    cost = cost_fn(weights, close, book)
    # Backtest convention: PnL at bar t+1 from weights at end of bar t.
    gross = (weights * ret.shift(-1)).sum(axis=1).fillna(0)
    net = gross - cost

    # Stage 7 — metrics
    splits = cfg["splits"]
    metrics = _split_metrics(gross, net, weights,
                              train_end=splits["train_end"],
                              val_end=splits["val_end"],
                              bars_per_year=bars_per_year)
    if verbose:
        print(f"[7] FULL  SR_g={metrics['FULL']['SR_gross']:+.2f}  SR_n={metrics['FULL']['SR_net']:+.2f}  "
              f"ret_n={metrics['FULL']['ret_ann_net']*100:+.1f}%/yr", flush=True)

    elapsed = time.time() - t0
    return PipelineResult(
        config=cfg, alpha_signals_n=len(alpha_signals),
        universe_size=len(tickers), n_bars=len(dates),
        weights=weights, gross_pnl=gross, cost=cost, net_pnl=net,
        metrics=metrics, elapsed_sec=elapsed, notes=notes,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    run(args.config)
