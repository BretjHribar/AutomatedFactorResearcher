"""Replay AUCT QP with production-style executable share-state feedback.

The normal research QP walk-forward uses the previous continuous QP weight
vector as w_prev. The live MOC path converts weights to integer shares, skips
orders below the configured min_order_value, then carries the actual shares
forward. This script keeps the project-native QP/risk model and shared
execution netting library, but feeds the executable held-share weights into
the next day's QP.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = EXP_DIR / "outputs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.equity_auction_anchor_clean_20260518.run_qp_tuning_original_alphas import (  # noqa: E402
    BOOK,
    QP_START,
    _build_targets,
)
from experiments.equity_auction_anchor_clean_20260518.run_test_selected_strategy import (  # noqa: E402
    PROD_TEST_START,
    RISK_MODEL_NAME,
    RISK_MODEL_PARAMS,
    SELECTED_QP,
    TEST_START,
    _base_full_auct_config,
    _prod_reference,
    _aipt_reference,
)
from src.execution.netting import build_child_orders  # noqa: E402
from src.pipeline.fees import make_cost_fn  # noqa: E402
from src.pipeline.runner import _build_risk_model_fn  # noqa: E402
from src.portfolio.qp import solve_qp  # noqa: E402
from src.strategies.registry import _weights_to_targets, parse_strategy_config  # noqa: E402


BARS_PER_YEAR = 252
LABEL = "auct_ic_wt_qp_exec_feedback_minord200"
STRATEGY_ID = "auct_exec_feedback"


def _ann_sr(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1 or clean.std() <= 0:
        return float("nan")
    return float(clean.mean() / clean.std() * math.sqrt(BARS_PER_YEAR))


def _max_dd(s: pd.Series) -> float:
    clean = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return float("nan")
    eq = (1.0 + clean).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _metrics(
    *,
    label: str,
    gross: pd.Series,
    cost: pd.Series,
    net: pd.Series,
    turnover: pd.Series | None,
    start: str,
    end: str,
) -> dict:
    g = gross.loc[start:end]
    c = cost.loc[start:end]
    n = net.loc[start:end]
    t = turnover.loc[start:end] if turnover is not None else None
    return {
        "series": label,
        "start": start,
        "end": end,
        "n_bars": int(n.replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
        "SR_gross": _ann_sr(g),
        "SR_net": _ann_sr(n),
        "ret_ann_net": float(n.mean() * BARS_PER_YEAR),
        "vol_ann_net": float(n.std() * math.sqrt(BARS_PER_YEAR)),
        "max_dd_net": _max_dd(n),
        "cost_ann": float(c.mean() * BARS_PER_YEAR),
        "turnover": float(t.mean()) if t is not None else float("nan"),
    }


def _corr(a: pd.Series, b: pd.Series, start: str, end: str) -> float:
    aa = a.loc[start:end].replace([np.inf, -np.inf], np.nan)
    bb = b.loc[start:end].replace([np.inf, -np.inf], np.nan)
    idx = aa.dropna().index.intersection(bb.dropna().index)
    if len(idx) < 20:
        return float("nan")
    return float(aa.loc[idx].corr(bb.loc[idx]))


def _strategy_config(min_order_value: float):
    return parse_strategy_config({
        "strategy_id": STRATEGY_ID,
        "name": "AUCT Execution Feedback QP",
        "enabled": True,
        "route": {
            "asset_type": "equity",
            "venue": "ib",
            "account": "paper",
            "bucket": "equity:ib:paper:stock:MOC:close",
        },
        "portfolio": {"book": BOOK},
        "execution": {"order_type": "MOC", "min_order_value": min_order_value},
        "signal": {},
        "metadata": {"family": "auct_research"},
    })


def _current_share_series(
    current_positions: dict[tuple[str, str], float],
    tickers: list[str],
) -> pd.Series:
    return pd.Series(
        {
            symbol: qty
            for (strategy_id, symbol), qty in current_positions.items()
            if strategy_id == STRATEGY_ID
        },
        dtype=float,
    ).reindex(tickers).fillna(0.0)


def _run_exec_feedback_qp(
    alpha: pd.DataFrame,
    close: pd.DataFrame,
    ret: pd.DataFrame,
    uni: pd.DataFrame,
    risk_fn,
    *,
    min_order_value: float,
    fee_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = alpha.index
    tickers = list(alpha.columns)
    n_names = len(tickers)
    vol = ret.rolling(RISK_MODEL_PARAMS["vol_window"], min_periods=20).std().shift(1).bfill().fillna(0.02)
    ret_mat = ret.values
    config = _strategy_config(min_order_value)
    current_positions: dict[tuple[str, str], float] = {}

    exec_weight_rows = []
    target_weight_rows = []
    order_rows = []
    stat_rows = []
    n_solves = 0
    n_fails = 0
    t0 = time.time()

    for i, dt in enumerate(dates):
        price_row = close.iloc[i].replace(0, np.nan)
        current_shares = _current_share_series(current_positions, tickers)
        w_prev = ((current_shares * price_row.fillna(0.0)) / BOOK).fillna(0.0).values.astype(float)

        a = alpha.iloc[i].values
        sig = vol.iloc[i].values
        active = (~np.isnan(close.iloc[i].values)) & uni.iloc[i].values & (sig > 0)

        if active.sum() >= 10:
            idx = np.where(active)[0]
            try:
                l_list, s2 = risk_fn(i, idx, sig[idx], ret_mat, RISK_MODEL_PARAMS["factor_window"])
                sol = solve_qp(
                    a[idx],
                    w_prev[idx],
                    close.iloc[i].values[idx],
                    l_list,
                    s2,
                    lambda_risk=SELECTED_QP["lambda_risk"],
                    kappa_tc=SELECTED_QP["kappa_tc"],
                    max_w=0.02,
                    commission_per_share=float(fee_params.get("commission_per_share", 0.0045)),
                    impact_bps=float(fee_params.get("impact_bps", 0.0)),
                    dollar_neutral=True,
                    max_gross_leverage=1.0,
                )
                if sol is None:
                    n_fails += 1
                    desired_w = w_prev.copy()
                else:
                    n_solves += 1
                    desired_w = np.zeros(n_names)
                    desired_w[idx] = sol
            except Exception:
                n_fails += 1
                desired_w = w_prev.copy()
        else:
            desired_w = w_prev.copy()

        desired = pd.Series(desired_w, index=tickers).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target_weight_rows.append(desired.rename(dt))

        targets = _weights_to_targets(
            config,
            weights=desired.to_dict(),
            prices=price_row,
            signal_time=str(dt.date()),
            metadata={"source": LABEL},
        )
        latest_prices = {str(symbol): float(px) for symbol, px in price_row.dropna().items() if float(px) > 0}
        raw_children = build_child_orders(
            batch_id=f"{STRATEGY_ID}_{dt:%Y%m%d}_raw",
            targets=targets,
            current_positions=current_positions,
            min_order_value=0.0,
            strategy_routes={STRATEGY_ID: config.route},
            latest_prices=latest_prices,
        )
        kept_children = build_child_orders(
            batch_id=f"{STRATEGY_ID}_{dt:%Y%m%d}",
            targets=targets,
            current_positions=current_positions,
            min_order_value={STRATEGY_ID: min_order_value},
            strategy_routes={STRATEGY_ID: config.route},
            latest_prices=latest_prices,
        )

        for child in kept_children:
            key = (child.strategy_id, child.symbol)
            next_qty = float(current_positions.get(key, 0.0) or 0.0) + float(child.delta_qty)
            if abs(next_qty) < 1e-9:
                current_positions.pop(key, None)
            else:
                current_positions[key] = next_qty

        current_shares = _current_share_series(current_positions, tickers)
        exec_w = ((current_shares * price_row.fillna(0.0)) / BOOK).fillna(0.0)
        exec_weight_rows.append(exec_w.rename(dt))

        order_count = len(kept_children)
        raw_notional = float(sum(child.notional for child in raw_children))
        kept_notional = float(sum(child.notional for child in kept_children))
        commission = float(
            sum(max(abs(child.delta_qty) * float(fee_params.get("commission_per_share", 0.0045)),
                    float(fee_params.get("per_order_min", 0.35)))
                for child in kept_children)
        )
        sell_value = float(sum(child.notional for child in kept_children if child.delta_qty < 0))
        sec_fee = sell_value * float(fee_params.get("sec_fee_per_dollar", 27.80e-6))
        short_dollars = float((-current_shares.where(current_shares < 0, 0)).mul(price_row.fillna(0.0)).sum())
        borrow = short_dollars * (float(fee_params.get("borrow_bps_annual", 50.0)) / 1e4) / BARS_PER_YEAR
        total_cost = commission + sec_fee + borrow

        stat_rows.append({
            "date": dt,
            "raw_order_count": len(raw_children),
            "kept_order_count": order_count,
            "skipped_order_count": len(raw_children) - order_count,
            "raw_turnover": raw_notional / BOOK,
            "executed_turnover": kept_notional / BOOK,
            "skipped_turnover": (raw_notional - kept_notional) / BOOK,
            "gross_weight_l1": float(exec_w.abs().sum()),
            "net_weight": float(exec_w.sum()),
            "commission": commission,
            "sec_fee": sec_fee,
            "borrow": borrow,
            "cost": total_cost / BOOK,
        })

        if kept_children:
            frame = pd.DataFrame({
                "date": dt,
                "ticker": [child.symbol for child in kept_children],
                "shares": [child.delta_qty for child in kept_children],
                "price": [child.reference_price for child in kept_children],
            })
            frame["order_value"] = frame["shares"].abs() * frame["price"]
            order_rows.append(frame)

        if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(dates):
            elapsed = time.time() - t0
            print(
                f"  [{LABEL}] {i + 1}/{len(dates)} bars "
                f"solves={n_solves} fails={n_fails} elapsed={elapsed:.0f}s",
                flush=True,
            )

    exec_weights = pd.DataFrame(exec_weight_rows)
    target_weights = pd.DataFrame(target_weight_rows)
    stats = pd.DataFrame(stat_rows).set_index("date")
    orders = pd.concat(order_rows, ignore_index=True) if order_rows else pd.DataFrame(
        columns=["date", "ticker", "shares", "price", "order_value"]
    )
    return exec_weights, target_weights, stats, orders


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _base_full_auct_config()
    fee_params = dict(base["fees"]["params"])
    fee_fn = make_cost_fn(base["fees"]["model"], fee_params, bars_per_year=BARS_PER_YEAR)
    min_order_value = float(json.loads((ROOT / "prod/config/strategy.json").read_text())["execution"]["min_order_value"])

    print("=== building full AUCT target for execution-feedback QP ===", flush=True)
    targets, uni, _mats, close, ret, dates, tickers = _build_targets(base)
    print(f"=== building {RISK_MODEL_NAME} risk model fn ===", flush=True)
    risk_fn = _build_risk_model_fn(RISK_MODEL_NAME, RISK_MODEL_PARAMS, _mats, dates, tickers)

    qslice = slice(QP_START, None)
    alpha = targets["ic_wt"].loc[qslice] * float(SELECTED_QP["alpha_scale"])
    close_q = close.loc[qslice]
    ret_q = ret.loc[qslice]
    uni_q = uni.loc[qslice]

    exec_weights, target_weights, stats, orders = _run_exec_feedback_qp(
        alpha,
        close_q,
        ret_q,
        uni_q,
        risk_fn,
        min_order_value=min_order_value,
        fee_params=fee_params,
    )

    gross = (exec_weights * ret_q.shift(-1)).sum(axis=1).fillna(0.0)
    cost = stats["cost"].reindex(gross.index).fillna(0.0)
    net = gross - cost
    turnover = stats["executed_turnover"].reindex(gross.index).fillna(0.0)

    continuous = pd.read_parquet(OUT_DIR / "test_selected_strategy_returns.parquet")
    prod = _prod_reference()
    aipt_gross, aipt_cost, aipt_net = _aipt_reference()

    common_end = min(
        net.dropna().index.max(),
        continuous["ic_wt_qp_styleppca_s10_l2_k50_net"].dropna().index.max(),
        prod.net_pnl.dropna().index.max(),
        aipt_net.dropna().index.max(),
    )
    end = str(pd.Timestamp(common_end).date())

    rows = [
        _metrics(
            label="paper_ib_moc_equity",
            gross=prod.gross_pnl,
            cost=prod.cost,
            net=prod.net_pnl,
            turnover=prod.weights.diff().abs().sum(axis=1).fillna(0.0),
            start=TEST_START,
            end=end,
        ),
        _metrics(
            label="paper_aipt_smallcap_d0",
            gross=aipt_gross,
            cost=aipt_cost,
            net=aipt_net,
            turnover=None,
            start=TEST_START,
            end=end,
        ),
        _metrics(
            label="auct_continuous_qp",
            gross=continuous["ic_wt_qp_styleppca_s10_l2_k50_gross"],
            cost=continuous["ic_wt_qp_styleppca_s10_l2_k50_cost"],
            net=continuous["ic_wt_qp_styleppca_s10_l2_k50_net"],
            turnover=continuous["ic_wt_qp_styleppca_s10_l2_k50_turnover"],
            start=TEST_START,
            end=end,
        ),
        _metrics(
            label=LABEL,
            gross=gross,
            cost=cost,
            net=net,
            turnover=turnover,
            start=TEST_START,
            end=end,
        ),
    ]
    prod_rows = [
        _metrics(
            label="auct_continuous_qp",
            gross=continuous["ic_wt_qp_styleppca_s10_l2_k50_gross"],
            cost=continuous["ic_wt_qp_styleppca_s10_l2_k50_cost"],
            net=continuous["ic_wt_qp_styleppca_s10_l2_k50_net"],
            turnover=continuous["ic_wt_qp_styleppca_s10_l2_k50_turnover"],
            start=PROD_TEST_START,
            end=end,
        ),
        _metrics(
            label=LABEL,
            gross=gross,
            cost=cost,
            net=net,
            turnover=turnover,
            start=PROD_TEST_START,
            end=end,
        ),
    ]
    summary = pd.DataFrame(rows)
    prod_window = pd.DataFrame(prod_rows)
    summary["corr_vs_paper_ib_moc"] = [
        _corr(prod.net_pnl, prod.net_pnl, TEST_START, end),
        _corr(aipt_net, prod.net_pnl, TEST_START, end),
        _corr(continuous["ic_wt_qp_styleppca_s10_l2_k50_net"], prod.net_pnl, TEST_START, end),
        _corr(net, prod.net_pnl, TEST_START, end),
    ]
    summary["corr_vs_paper_aipt"] = [
        _corr(prod.net_pnl, aipt_net, TEST_START, end),
        _corr(aipt_net, aipt_net, TEST_START, end),
        _corr(continuous["ic_wt_qp_styleppca_s10_l2_k50_net"], aipt_net, TEST_START, end),
        _corr(net, aipt_net, TEST_START, end),
    ]

    returns = pd.DataFrame({
        f"{LABEL}_gross": gross,
        f"{LABEL}_cost": cost,
        f"{LABEL}_net": net,
        f"{LABEL}_turnover": turnover,
    })
    exec_weights.to_parquet(OUT_DIR / "auct_exec_feedback_qp_weights.parquet")
    target_weights.to_parquet(OUT_DIR / "auct_exec_feedback_qp_target_weights.parquet")
    returns.to_parquet(OUT_DIR / "auct_exec_feedback_qp_returns.parquet")
    stats.to_csv(OUT_DIR / "auct_exec_feedback_qp_daily_stats.csv")
    orders.to_parquet(OUT_DIR / "auct_exec_feedback_qp_orders.parquet", index=False)
    summary.to_csv(OUT_DIR / "auct_exec_feedback_qp_summary.csv", index=False)
    prod_window.to_csv(OUT_DIR / "auct_exec_feedback_qp_prod_window_summary.csv", index=False)

    print("\n=== EXECUTION-FEEDBACK QP SUMMARY ===", flush=True)
    print(summary.to_markdown(index=False, floatfmt=".4f"), flush=True)
    print("\n=== PROD TEST WINDOW ===", flush=True)
    print(prod_window.to_markdown(index=False, floatfmt=".4f"), flush=True)
    print(f"\nSaved {OUT_DIR / 'auct_exec_feedback_qp_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
