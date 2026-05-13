"""Stepwise tradability layers for the AIPT SDF.

Run this only after the unconstrained SDF matches the paper. The layers are
ordered so we can see exactly where Sharpe is lost:

  raw_sdf              frictionless paper SDF return, no costs
  gross1              daily no-neutrality gross-normalized asset weights
  gross1_fee          gross1 plus realized execution costs
  gross1_cap          gross1 plus per-name cap, no costs
  gross1_cap_fee      gross1_cap plus realized execution costs
  kernel_gross1_cap_fee
                       full quadratic one-step cost kernel, then gross1_cap_fee
  qp_gross1_cap_fee   project-native name-level QP execution, no neutrality
  kernel_qp_gross1_cap_fee
                       cost-aware lambda fit plus project-native QP execution

None of these layers subtracts the cross-sectional mean or enforces dollar
neutrality.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aipt_replication import (
    SCENARIOS,
    _cost_vector,
    _fit_lambda,
    _parse_csv_floats,
    _parse_csv_ints,
    load_market_data,
    make_characteristic_tensor,
    make_forward_returns,
    make_random_params,
    random_features_for_date,
    realised_costs,
)
from experiments.aipt_unconstrained import build_factor_returns_unconstrained
from src.portfolio.qp import solve_qp
from src.portfolio.risk_model import build_diagonal, build_style_factors


LAYERS = [
    "raw_sdf",
    "gross1",
    "gross1_fee",
    "gross1_cap",
    "gross1_cap_fee",
    "kernel_gross1_cap_fee",
    "qp_gross1_cap_fee",
    "kernel_qp_gross1_cap_fee",
    # L1 proximal trade gate. Cost-aware weight-space soft-threshold around w_prev.
    # Decouples SDF fit (information) from trade decision (cost). Dimensionally
    # matches realized L1 fees, unlike kernel_*. Knob: cost_tau scales per-name
    # threshold = cost_tau * c_i in fraction-of-book units.
    "prox_l1_gross1_cap_fee",
    # Risk-neutralized target: residualize SDF target weights against Barra-style
    # factors (market_beta, size, value, momentum, profitability, low_vol, growth,
    # leverage) + industry dummies cross-sectionally each day, then gross-norm +
    # L1 prox + cap + fee. Removes common-factor exposure so the AIPT signal
    # operates on idiosyncratic risk only.
    "neutral_prox_l1_gross1_cap_fee",
]


def _layer_uses_kernel(layer: str) -> bool:
    return layer.startswith("kernel_")


def _layer_uses_prox(layer: str) -> bool:
    return "prox_l1" in layer


def _layer_uses_neutral(layer: str) -> bool:
    return layer.startswith("neutral_")


def _build_risk_loading_stack(
    mats: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    tickers: list[str],
    classifications: dict | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pre-compute (n_dates, n_tickers, n_factors) loadings tensor and the
    (n_tickers, n_industries) industry-dummy matrix using the project's
    `src.portfolio.risk_model.build_style_factors`.

    Returns (B_stack, industry_dummies, factor_names).
      B_stack: float32 (T, N, K_style) cross-sectional style loadings per bar
      industry_dummies: float32 (N, K_ind) one-hot industry membership
      factor_names: list[str]
    """
    style_factors = build_style_factors(mats)
    factor_names = list(style_factors.keys())
    n_dates = len(dates)
    n_tickers = len(tickers)
    K = len(factor_names)
    B = np.full((n_dates, n_tickers, K), np.nan, dtype=np.float32)
    for k, fname in enumerate(factor_names):
        df = style_factors[fname].reindex(index=dates, columns=tickers)
        B[:, :, k] = df.values.astype(np.float32)

    # Industry dummies — drop one to avoid singularity (industry_0 absorbed into intercept).
    if classifications is None:
        return B, np.zeros((n_tickers, 0), dtype=np.float32), factor_names
    levels = [classifications.get(t, "Unknown") for t in tickers]
    groups = sorted({g for g in levels if g != "Unknown"})
    if len(groups) > 1:
        groups = groups[1:]  # drop one for singularity
    dummies = np.zeros((n_tickers, len(groups)), dtype=np.float32)
    for k, g in enumerate(groups):
        for i, t in enumerate(tickers):
            if classifications.get(t, "Unknown") == g:
                dummies[i, k] = 1.0
    return B, dummies, factor_names


def _neutralize_target_weights(
    w_target: np.ndarray,
    active_t: np.ndarray,
    B_t: np.ndarray,
    industry_dummies: np.ndarray,
) -> np.ndarray:
    """Cross-sectional regression of w_target on style+industry loadings.

    Returns the residual: w_target − X β̂.

    NaN-safe: rows with insufficient cross-sectional coverage return w_target
    unmodified. The risk loadings themselves are masked to the active set.
    """
    if int(active_t.sum()) < 50:
        return w_target
    # Build design matrix (active_names, n_factors+n_industries+1[intercept]).
    cols = [np.ones((int(active_t.sum()), 1), dtype=np.float64)]
    # Style factors: only keep columns that are finite for all active names.
    Ba = B_t[active_t].astype(np.float64)
    ok_style = np.all(np.isfinite(Ba), axis=0)
    if ok_style.any():
        Ba_clean = Ba[:, ok_style]
        cols.append(Ba_clean)
    # Industry dummies (already binary, no NaN risk).
    if industry_dummies.shape[1] > 0:
        cols.append(industry_dummies[active_t].astype(np.float64))
    X = np.hstack(cols)
    y = w_target[active_t].astype(np.float64)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return w_target
    resid = y - X @ beta
    out = w_target.astype(np.float64, copy=True)
    out[active_t] = resid
    return out


def _apply_l1_proximal_cost_gate(
    w_target: np.ndarray,
    w_prev: np.ndarray,
    active_t: np.ndarray,
    c_vec: np.ndarray,
    cost_tau: float,
    max_weight: float,
) -> np.ndarray:
    """L1 proximal trade gate around w_prev.

    Solves, per name independently:
        min_{Δw_i}  0.5 (w_prev_i + Δw_i - w_target_i)^2 + cost_tau * c_i * |Δw_i|

    Closed form: Δw_i = sign(w_target_i - w_prev_i) * max(0, |w_target_i - w_prev_i| - cost_tau * c_i).

    Trades smaller than the cost threshold are zeroed out; large trades are
    shrunk by the per-name threshold. This is the standard soft-threshold /
    Lasso proximal operator. Matches L1 realized-fee geometry directly.

    After thresholding, re-normalize to gross 1 and re-clip to max_weight to
    preserve the gross1_cap layer's invariants.
    """
    if cost_tau <= 0 or int(active_t.sum()) < 2:
        return w_target
    mask = active_t & np.isfinite(w_target)
    if not mask.any():
        return w_target
    c = np.nan_to_num(c_vec, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    threshold = cost_tau * c
    delta_target = w_target - w_prev
    sign = np.sign(delta_target)
    abs_delta = np.abs(delta_target)
    shrunk = np.maximum(abs_delta - threshold, 0.0)
    delta_new = sign * shrunk
    w_new = w_prev + delta_new
    w_new = np.where(mask, w_new, 0.0)
    gross = float(np.abs(w_new[mask]).sum())
    if gross > 1e-12:
        w_new[mask] = w_new[mask] / gross
    if max_weight > 0:
        w_new[mask] = np.clip(w_new[mask], -max_weight, max_weight)
        gross = float(np.abs(w_new[mask]).sum())
        if gross > 1e-12:
            w_new[mask] = w_new[mask] / gross
    return w_new


def _layer_uses_qp(layer: str) -> bool:
    return layer.startswith("qp_") or "_qp_" in layer


def _layer_uses_fee(layer: str) -> bool:
    return layer.endswith("_fee")


def _layer_uses_tau(layer: str) -> bool:
    return _layer_uses_kernel(layer) or _layer_uses_qp(layer) or _layer_uses_prox(layer)


@dataclass(frozen=True)
class StepSpec:
    scenario: str
    source_set: str
    n_features: int
    ridge_z: float
    activation: str
    seed: int
    layer: str
    max_weight: float
    cost_tau: float
    turnover_cap: float
    blend: float
    qp_alpha_scale: float
    qp_risk_lambda: float


def _metrics_one(x: pd.Series, bars_per_year: int) -> dict[str, float]:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return {"n_bars": int(len(x)), "SR": float("nan"), "ret_ann": float("nan"), "vol_ann": float("nan"), "max_dd": float("nan")}
    sd = float(x.std())
    mu = float(x.mean())
    eq = (1.0 + x).cumprod()
    return {
        "n_bars": int(len(x)),
        "SR": (mu / sd) * math.sqrt(bars_per_year) if sd > 0 else float("nan"),
        "ret_ann": mu * bars_per_year,
        "vol_ann": sd * math.sqrt(bars_per_year),
        "max_dd": float((eq / eq.cummax() - 1.0).min()),
    }


def _scale_no_neutral(
    raw: np.ndarray,
    active_t: np.ndarray,
    *,
    target_gross: float,
    max_weight: float,
) -> np.ndarray:
    w = np.zeros_like(raw, dtype=np.float64)
    mask = active_t & np.isfinite(raw)
    if mask.sum() < 2:
        return w
    vals = raw[mask].astype(np.float64, copy=True)
    gross = float(np.abs(vals).sum())
    if gross <= 1e-12:
        return w
    vals *= target_gross / gross
    if max_weight > 0:
        vals = np.clip(vals, -max_weight, max_weight)
        gross = float(np.abs(vals).sum())
        if gross > 1e-12 and gross < target_gross:
            vals *= min(target_gross / gross, 1.0 / max(np.max(np.abs(vals)) / max_weight, 1.0))
        vals = np.clip(vals, -max_weight, max_weight)
    w[mask] = vals
    return w


def _apply_turnover_control(target: np.ndarray, prev: np.ndarray, turnover_cap: float, blend: float) -> np.ndarray:
    """Move from prev toward target with optional blend and L1 turnover cap."""
    blend = min(max(float(blend), 0.0), 1.0)
    w = prev + blend * (target - prev)
    if turnover_cap > 0:
        delta = w - prev
        turnover = float(np.abs(delta).sum())
        if turnover > turnover_cap > 1e-12:
            w = prev + delta * (turnover_cap / turnover)
    return w


def _fit_lambda_full_cost(
    f_train: np.ndarray,
    ridge_z: float,
    s_t: np.ndarray,
    active_t: np.ndarray,
    w_prev: np.ndarray,
    c_vec: np.ndarray,
    cost_tau: float,
) -> np.ndarray:
    if cost_tau <= 0 or int(active_t.sum()) < 2:
        return _fit_lambda(f_train, ridge_z)

    f_train = np.nan_to_num(np.asarray(f_train, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n_obs, n_features = f_train.shape
    mu = f_train.mean(axis=0)
    gram = (f_train.T @ f_train) / max(n_obs, 1)
    gram.flat[:: n_features + 1] += ridge_z

    n_active = float(active_t.sum())
    a = s_t[active_t].astype(np.float64, copy=False) / math.sqrt(n_active)
    c = np.nan_to_num(c_vec[active_t], nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    gram += cost_tau * (a.T @ (c[:, None] * a))
    mu = mu + cost_tau * (a.T @ (c * w_prev[active_t]))

    try:
        return np.linalg.solve(gram, mu)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ mu


def _qp_weight_cap(
    scenario_name: str,
    active_t: np.ndarray,
    adv_row: pd.Series | None,
    tickers: list[str],
    max_weight: float,
) -> np.ndarray | float:
    scenario = SCENARIOS[scenario_name]
    if adv_row is None or scenario.book <= 0:
        return max_weight
    adv = adv_row.reindex(tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    adv_active = adv[active_t]
    if adv_active.size == 0 or not np.isfinite(adv_active).any():
        return max_weight
    capacity_cap = 0.10 * 0.30 * adv_active / float(scenario.book)
    capacity_cap = np.where((capacity_cap > 0) & np.isfinite(capacity_cap), capacity_cap, 0.0)
    return np.minimum(max_weight, capacity_cap)


def _solve_project_qp(
    *,
    spec: StepSpec,
    raw: np.ndarray,
    active_t: np.ndarray,
    w_prev: np.ndarray,
    close_row: pd.Series,
    ret_train: np.ndarray,
    adv_row: pd.Series | None,
    tickers: list[str],
) -> tuple[np.ndarray | None, bool]:
    """Project raw AIPT target into the project-native name-level QP."""
    scenario = SCENARIOS[spec.scenario]
    mask = active_t & np.isfinite(raw)
    if int(mask.sum()) < scenario.min_names:
        return None, False
    alpha = np.nan_to_num(raw[mask], nan=0.0, posinf=0.0, neginf=0.0) * spec.qp_alpha_scale
    if np.all(np.abs(alpha) < 1e-14):
        return None, False
    prices = close_row.reindex(tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float64)
    price_s = np.maximum(prices[mask], 0.01)
    wp_s = w_prev[mask]

    train = np.nan_to_num(ret_train[:, mask], nan=0.0, posinf=0.0, neginf=0.0)
    if train.shape[0] >= 20:
        vol = np.nanstd(train, axis=0, ddof=1)
    else:
        vol = np.full(mask.sum(), 0.02, dtype=np.float64)
    vol = np.maximum(np.nan_to_num(vol, nan=0.02, posinf=0.02, neginf=0.02), 1e-4)
    l_list, s2 = build_diagonal(vol)

    cap = _qp_weight_cap(spec.scenario, mask, adv_row, tickers, spec.max_weight)
    if np.ndim(cap) > 0 and np.all(np.asarray(cap) <= 0):
        return None, False

    if scenario.fee_model == "bps_taker":
        commission = 0.0
        impact_bps = scenario.fee_params.get("taker_bps", 0.0) + scenario.fee_params.get("slippage_bps", 0.0)
    else:
        commission = scenario.fee_params.get("commission_per_share", 0.0045)
        impact_bps = scenario.fee_params.get("impact_bps", 0.5)

    sol = solve_qp(
        alpha,
        wp_s,
        price_s,
        l_list,
        s2,
        lambda_risk=spec.qp_risk_lambda,
        kappa_tc=spec.cost_tau,
        max_w=cap,
        commission_per_share=commission,
        impact_bps=impact_bps,
        dollar_neutral=False,
        max_gross_leverage=1.0,
    )
    if sol is None or not np.all(np.isfinite(sol)):
        return None, False
    out = np.zeros_like(raw, dtype=np.float64)
    out[mask] = np.asarray(sol, dtype=np.float64)
    return out, True


def _split_metrics(
    scenario_name: str,
    sdf: pd.Series,
    gross: pd.Series,
    net: pd.Series,
    costs: pd.Series,
    weights: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    scenario = SCENARIOS[scenario_name]
    splits = {
        "TRAIN": slice(None, scenario.split_train_end),
        "VAL": slice(scenario.split_train_end, scenario.split_val_end),
        "TEST": slice(scenario.split_val_end, None),
        "VAL+TEST": slice(scenario.split_train_end, None),
        "FULL": slice(None, None),
    }
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross_exp = weights.abs().sum(axis=1)
    net_exp = weights.sum(axis=1)
    short_gross = weights.clip(upper=0.0).abs().sum(axis=1)
    max_abs = weights.abs().max(axis=1)
    out: dict[str, dict[str, float]] = {}
    for label, sl in splits.items():
        sm = _metrics_one(sdf.loc[sl], scenario.bars_per_year)
        gm = _metrics_one(gross.loc[sl], scenario.bars_per_year)
        nm = _metrics_one(net.loc[sl], scenario.bars_per_year)
        out[label] = {
            "n_bars": nm["n_bars"],
            "SR_sdf": sm["SR"],
            "SR_gross": gm["SR"],
            "SR_net": nm["SR"],
            "ret_ann_net": nm["ret_ann"],
            "vol_ann_net": nm["vol_ann"],
            "max_dd_net": nm["max_dd"],
            "cost_ann": float(costs.loc[sl].mean() * scenario.bars_per_year),
            "turnover_per_bar": float(turnover.loc[sl].mean()),
            "gross_exposure": float(gross_exp.loc[sl].mean()),
            "net_exposure": float(net_exp.loc[sl].mean()),
            "short_gross": float(short_gross.loc[sl].mean()),
            "max_abs_weight": float(max_abs.loc[sl].mean()),
        }
    return out


def run_one(spec: StepSpec, out_dir: Path) -> dict:
    scenario = SCENARIOS[spec.scenario]
    fields = scenario.source_sets[spec.source_set]
    print(
        f"[step] {spec.scenario} {spec.layer} P={spec.n_features} z={spec.ridge_z:g} "
        f"seed={spec.seed} maxw={spec.max_weight:g} tau={spec.cost_tau:g} "
        f"turncap={spec.turnover_cap:g} blend={spec.blend:g} "
        f"qpscale={spec.qp_alpha_scale:g} qprisk={spec.qp_risk_lambda:g}",
        flush=True,
    )
    t0 = time.time()
    uni, mats, close, available_fields = load_market_data(scenario, fields, root=ROOT)
    fwd_ret = make_forward_returns(mats, scenario.delay)
    x, active, used_fields = make_characteristic_tensor(uni, mats, available_fields)
    random_w, gamma = make_random_params(len(used_fields), spec.n_features, spec.activation, spec.seed)
    factors = build_factor_returns_unconstrained(
        x,
        active,
        fwd_ret,
        random_w,
        gamma,
        spec.activation,
        min_names=scenario.min_names,
        demean_features=False,
    )

    dates = uni.index
    tickers = uni.columns.tolist()
    ret_np = fwd_ret.reindex(index=dates, columns=tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    sdf_ret = pd.Series(0.0, index=dates, dtype=float)
    gross_ret = pd.Series(0.0, index=dates, dtype=float)
    lam = np.zeros(factors.shape[1], dtype=np.float64)
    w_prev = np.zeros(len(tickers), dtype=np.float64)
    last_fit = -10**9
    adv = mats.get(scenario.selection_field)
    qp_solves = 0
    qp_failures = 0

    # Precompute risk loadings if the layer needs neutralization. Done once
    # because build_style_factors does its own per-date z-scoring inside.
    B_stack = None
    industry_dummies = None
    factor_names: list[str] = []
    if _layer_uses_neutral(spec.layer):
        # Load FMP industry classifications for industry dummies; project's
        # neutralize() uses subindustry by default — match that here.
        cls_path = ROOT / "data/fmp_cache/classifications.json"
        if cls_path.exists():
            raw_cls = json.loads(cls_path.read_text(encoding="utf-8"))
            classifications = {t: str(raw_cls.get(t, {}).get("subindustry", "Unknown")) for t in tickers}
        else:
            classifications = None
        B_stack, industry_dummies, factor_names = _build_risk_loading_stack(
            mats, dates, tickers, classifications
        )
        print(
            f"  risk loadings: K_style={len(factor_names)} K_industry={industry_dummies.shape[1]}",
            flush=True,
        )

    for t in range(len(dates)):
        train_end = t - scenario.delay
        train_start = train_end - scenario.train_window
        if train_start < 0 or train_end <= train_start:
            weights.iloc[t] = w_prev
            continue

        s_t = random_features_for_date(
            x[t],
            active[t],
            random_w,
            gamma,
            spec.activation,
            demean_features=False,
        )
        if (t - last_fit) >= scenario.rebalance_every:
            f_train = factors[train_start:train_end]
            if _layer_uses_kernel(spec.layer) and spec.cost_tau > 0:
                adv_row = adv.iloc[t] if adv is not None else None
                c_vec = _cost_vector(scenario, close.iloc[t], adv_row)
                lam = _fit_lambda_full_cost(f_train, spec.ridge_z, s_t, active[t], w_prev, c_vec, spec.cost_tau)
            else:
                lam = _fit_lambda(f_train, spec.ridge_z)
            last_fit = t

        n_active = max(int(active[t].sum()), 1)
        raw = (s_t @ lam) / math.sqrt(n_active)
        if spec.layer == "raw_sdf":
            w_t = np.where(active[t] & np.isfinite(raw), raw, 0.0)
        elif _layer_uses_qp(spec.layer):
            adv_row = adv.iloc[t] if adv is not None else None
            ret_train = ret_np[train_start:train_end]
            qp_w, ok = _solve_project_qp(
                spec=spec,
                raw=raw,
                active_t=active[t],
                w_prev=w_prev,
                close_row=close.iloc[t],
                ret_train=ret_train,
                adv_row=adv_row,
                tickers=tickers,
            )
            if ok and qp_w is not None:
                w_t = qp_w
                qp_solves += 1
            else:
                w_t = _scale_no_neutral(raw, active[t], target_gross=1.0, max_weight=spec.max_weight)
                qp_failures += 1
        elif _layer_uses_prox(spec.layer):
            # Optional risk neutralization (style + industry residualization) of
            # the raw SDF target BEFORE normalization. Removes common-factor
            # exposure so the AIPT signal operates on idiosyncratic risk only.
            if _layer_uses_neutral(spec.layer) and B_stack is not None:
                raw_neutral = _neutralize_target_weights(
                    raw, active[t], B_stack[t], industry_dummies,
                )
            else:
                raw_neutral = raw
            # L1 proximal trade gate: standard gross-norm + cap target, then
            # soft-threshold each name's trade by tau * per-name cost rate.
            w_target = _scale_no_neutral(raw_neutral, active[t], target_gross=1.0, max_weight=spec.max_weight)
            adv_row = adv.iloc[t] if adv is not None else None
            c_vec = _cost_vector(scenario, close.iloc[t], adv_row)
            w_t = _apply_l1_proximal_cost_gate(
                w_target, w_prev, active[t], c_vec, spec.cost_tau, spec.max_weight
            )
        elif "cap" in spec.layer:
            w_t = _scale_no_neutral(raw, active[t], target_gross=1.0, max_weight=spec.max_weight)
        else:
            w_t = _scale_no_neutral(raw, active[t], target_gross=1.0, max_weight=0.0)
        if spec.layer != "raw_sdf":
            w_t = _apply_turnover_control(w_t, w_prev, spec.turnover_cap, spec.blend)

        weights.iloc[t] = w_t
        w_prev = w_t
        sdf_ret.iloc[t] = float(np.nan_to_num(factors[t]) @ lam)
        gross_ret.iloc[t] = float(np.dot(w_t, ret_np[t]))

    if _layer_uses_fee(spec.layer):
        costs = realised_costs(scenario, weights, close.reindex(index=dates, columns=tickers))
    else:
        costs = pd.Series(0.0, index=dates, dtype=float)
    net_ret = gross_ret - costs
    metrics = _split_metrics(spec.scenario, sdf_ret, gross_ret, net_ret, costs, weights)
    elapsed = time.time() - t0

    result = {
        "spec": asdict(spec),
        "scenario": asdict(scenario),
        "n_dates": len(dates),
        "n_names": len(tickers),
        "n_fields": len(used_fields),
        "qp_solves": qp_solves,
        "qp_failures": qp_failures,
        "used_fields": used_fields,
        "metrics": metrics,
        "elapsed_sec": elapsed,
        "lookahead_audit": {
            "features": "S(Z[t]) only",
            "factor_return": "S(Z[t])' R[t+1] / sqrt(N[t])",
            "fit_uses_factor_rows": f"< t-{scenario.delay}",
            "weights": "daily S(Z[t]) lambda[t] updates; lambda refit cadence is separate",
            "dollar_neutral": False,
            "fees": _layer_uses_fee(spec.layer),
            "cost_kernel": _layer_uses_kernel(spec.layer),
            "project_native_qp": _layer_uses_qp(spec.layer),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"{spec.scenario}__{spec.layer}__P{spec.n_features}__z{spec.ridge_z:g}"
        f"__seed{spec.seed}__maxw{spec.max_weight:g}__tau{spec.cost_tau:g}"
        f"__turncap{spec.turnover_cap:g}__blend{spec.blend:g}"
        f"__qpscale{spec.qp_alpha_scale:g}__qprisk{spec.qp_risk_lambda:g}"
    ).replace(".", "p")
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    weights.tail(300).to_parquet(out_dir / f"{tag}.weights_tail.parquet")
    pd.DataFrame({"sdf": sdf_ret, "gross": gross_ret, "cost": costs, "net": net_ret}).to_parquet(
        out_dir / f"{tag}.returns.parquet"
    )
    print(
        f"       SR_net VAL={metrics['VAL']['SR_net']:+.2f} TEST={metrics['TEST']['SR_net']:+.2f} "
        f"V+T={metrics['VAL+TEST']['SR_net']:+.2f} "
        f"to={metrics['VAL+TEST']['turnover_per_bar']*100:.1f}% "
        f"gross={metrics['VAL+TEST']['gross_exposure']:.2f} {elapsed:.1f}s",
        flush=True,
    )
    return result


def make_specs(args: argparse.Namespace) -> list[StepSpec]:
    scenarios = args.scenarios or []
    if args.scenario:
        scenarios.append(args.scenario)
    if not scenarios:
        scenarios = ["equity_smallcap_d0", "equity_smallcap_d1", "kucoin_top100"]
    p_grid = _parse_csv_ints(args.p_grid)
    z_grid = _parse_csv_floats(args.z_grid)
    seeds = _parse_csv_ints(args.seeds)
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    cost_taus = _parse_csv_floats(args.cost_taus)
    turnover_caps = _parse_csv_floats(args.turnover_caps)
    blends = _parse_csv_floats(args.blends)
    qp_alpha_scales = _parse_csv_floats(args.qp_alpha_scales)
    qp_risk_lambdas = _parse_csv_floats(args.qp_risk_lambdas)
    specs: list[StepSpec] = []
    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        source_sets = [x.strip() for x in args.source_sets.split(",") if x.strip()]
        if not source_sets or source_sets == ["default"]:
            source_sets = [scenario.default_source_set]
        for source_set in source_sets:
            for p in p_grid:
                for z in z_grid:
                    for seed in seeds:
                        for layer in layers:
                            if layer not in LAYERS:
                                continue
                            taus = cost_taus if _layer_uses_tau(layer) else [0.0]
                            scales = qp_alpha_scales if _layer_uses_qp(layer) else [1.0]
                            risk_lambdas = qp_risk_lambdas if _layer_uses_qp(layer) else [0.0]
                            for tau in taus:
                                for turnover_cap in turnover_caps:
                                    for blend in blends:
                                        for qp_alpha_scale in scales:
                                            for qp_risk_lambda in risk_lambdas:
                                                specs.append(
                                                    StepSpec(
                                                        scenario=scenario_name,
                                                        source_set=source_set,
                                                        n_features=p,
                                                        ridge_z=z,
                                                        activation=args.activation,
                                                        seed=seed,
                                                        layer=layer,
                                                        max_weight=args.max_weight if args.max_weight > 0 else scenario.max_weight,
                                                        cost_tau=tau,
                                                        turnover_cap=turnover_cap,
                                                        blend=blend,
                                                        qp_alpha_scale=qp_alpha_scale,
                                                        qp_risk_lambda=qp_risk_lambda,
                                                    )
                                                )
    if args.limit:
        specs = specs[: args.limit]
    return specs


def write_summary(results: list[dict], out_dir: Path) -> Path:
    rows = []
    for r in results:
        spec = r["spec"]
        base = {
            "scenario": spec["scenario"],
            "source_set": spec["source_set"],
            "P": spec["n_features"],
            "z": spec["ridge_z"],
            "activation": spec["activation"],
            "seed": spec["seed"],
            "layer": spec["layer"],
            "max_weight": spec["max_weight"],
            "cost_tau": spec["cost_tau"],
            "turnover_cap": spec["turnover_cap"],
            "blend": spec["blend"],
            "qp_alpha_scale": spec["qp_alpha_scale"],
            "qp_risk_lambda": spec["qp_risk_lambda"],
            "qp_solves": r.get("qp_solves", 0),
            "qp_failures": r.get("qp_failures", 0),
            "n_names": r["n_names"],
            "n_fields": r["n_fields"],
            "elapsed_sec": r["elapsed_sec"],
        }
        for split, metrics in r["metrics"].items():
            row = dict(base)
            row["split"] = split
            row.update(metrics)
            rows.append(row)
    out = out_dir / "aipt_stepwise_summary.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "aipt_stepwise_summary.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return out


def write_run_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "experiments/aipt_stepwise_constraints.py",
        "started_at": datetime.now().isoformat(),
        "cwd": str(ROOT),
        "argv": sys.argv,
        "args": vars(args),
        "mode": "stepwise_tradability_layers",
        "notes": "No dollar neutrality. Layers isolate raw SDF, gross/cap scaling, fees, local cost kernel, turnover controls, and project-native QP execution.",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--source-sets", default="default")
    p.add_argument("--p-grid", default="256")
    p.add_argument("--z-grid", default="0.001")
    p.add_argument("--activation", default="sincos")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--layers", default="raw_sdf,gross1,gross1_fee,gross1_cap,gross1_cap_fee,kernel_gross1_cap_fee")
    p.add_argument("--cost-taus", default="0.1,1,10")
    p.add_argument("--turnover-caps", default="0")
    p.add_argument("--blends", default="1")
    p.add_argument("--qp-alpha-scales", default="1")
    p.add_argument("--qp-risk-lambdas", default="5")
    p.add_argument("--max-weight", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default="experiments/results/aipt_stepwise")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    out_dir = ROOT / args.out_dir
    write_run_manifest(args, out_dir)
    specs = make_specs(args)
    print(f"[setup] stepwise cells={len(specs)} -> {out_dir.relative_to(ROOT)}", flush=True)
    results: list[dict] = []
    failures: list[dict] = []
    for spec in specs:
        try:
            results.append(run_one(spec, out_dir))
            write_summary(results, out_dir)
        except Exception as exc:
            failures.append({"spec": asdict(spec), "error": f"{type(exc).__name__}: {exc}"})
            print(f"[fail] {spec}: {type(exc).__name__}: {exc}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "aipt_stepwise_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary = write_summary(results, out_dir)
    print(f"[done] results={len(results)} failures={len(failures)} summary={summary.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
