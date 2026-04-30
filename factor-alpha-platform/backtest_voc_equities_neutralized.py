"""
Equities AIPT with PIT-correct data + neutralization comparison.

Reads from data/fmp_cache/matrices_pit/ (rebuilt with filingDate forward-fill).

Compares 5 modes (all use same forecast pipeline, differ in how the predicted
signal is neutralized before normalization):

  baseline      — no neutralization (raw cross-sectional signal)
  market        — demean across all stocks
  industry      — demean within each 3-digit SIC industry
  subindustry   — demean within each 4-digit SIC subindustry
  risk_model    — regression residuals against FactorRiskModel loadings
                  (sector dummies + size/value/momentum/vol/leverage)

Reports gross SR, net SR, IC, R², turnover for each mode + VAL/TEST split.
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from src.pipeline.isichenko import FactorRiskModel

# ── Config ───────────────────────────────────────────────────────────────────
PIT_DIR        = PROJECT_ROOT / "data/fmp_cache/matrices_pit"
UNIVERSES_DIR  = PROJECT_ROOT / "data/fmp_cache/universes"
CLASSIF_PATH   = PROJECT_ROOT / "data/fmp_cache/classifications.json"
RESULTS_DIR    = PROJECT_ROOT / "data/aipt_results"
LOG_CSV        = RESULTS_DIR / "voc_equities_neutralized.csv"

UNIVERSE_NAME    = "TOP2000"
BARS_PER_YEAR    = 252
TRAIN_BARS       = 1500
MIN_TRAIN_BARS   = 500
REBAL_EVERY      = 5
OOS_START        = "2024-01-01"
COVERAGE_CUTOFF  = 0.5
Z_RIDGE          = 1e-3
SEED             = 42
GAMMA_GRID       = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAKER_BPS_GRID   = [0.0, 1.0, 3.0]   # report net SR at 0/1/3 bps aggregate
P                = 1000

CHAR_NAMES = [
    "log_returns",
    "historical_volatility_20", "historical_volatility_60", "historical_volatility_120",
    "parkinson_volatility_20", "parkinson_volatility_60", "parkinson_volatility_120",
    "book_to_market", "earnings_yield", "free_cashflow_yield",
    "ev_to_ebitda", "ev_to_revenue",
    "roe", "roa", "gross_margin", "operating_margin", "net_margin",
    "asset_turnover",
    "adv20", "adv60", "dollars_traded", "cap",
    "debt_to_equity", "current_ratio",
]

NEUTRALIZATIONS = ["baseline", "market", "industry", "subindustry", "risk_model"]


def load_data():
    """Load matrices from PIT dir + classifications."""
    print(f"Loading PIT matrices from {PIT_DIR}...", flush=True)
    if not PIT_DIR.exists():
        raise FileNotFoundError(f"PIT matrices not built yet — run rebuild_pit_matrices.py")

    uni = pd.read_parquet(UNIVERSES_DIR / f"{UNIVERSE_NAME}.parquet")
    cov = uni.sum(axis=0) / len(uni)
    valid_tickers = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    print(f"  {UNIVERSE_NAME} tickers passing {COVERAGE_CUTOFF*100:.0f}% coverage: {len(valid_tickers)}")

    matrices = {}
    for name in CHAR_NAMES + ["close"]:
        fp = PIT_DIR / f"{name}.parquet"
        if not fp.exists():
            print(f"  WARN: missing {name}.parquet")
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[name] = df[cols]

    common_idx = matrices["close"].index
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(index=common_idx)
    tickers = sorted(set(matrices["close"].columns))
    for k in list(matrices):
        matrices[k] = matrices[k].reindex(columns=tickers)

    close_vals = matrices["close"].values
    dates = matrices["close"].index
    chars = [c for c in CHAR_NAMES if c in matrices]
    print(f"  Loaded {len(chars)} chars, T={len(dates)}, N={len(tickers)}")

    # Classifications
    with open(CLASSIF_PATH) as fh:
        classifications = json.load(fh)
    print(f"  Classifications: {len(classifications)} tickers (will use for industry / subindustry / risk model)")
    return matrices, tickers, dates, close_vals, chars, classifications


def build_Z_panel(matrices, tickers, chars, start, end, delay=1):
    N = len(tickers)
    D = len(chars)
    panel = {}
    for t in range(start, end):
        z_idx = t - delay
        if z_idx < 0:
            continue
        Z = np.full((N, D), np.nan)
        for j, cn in enumerate(chars):
            Z[:, j] = matrices[cn].iloc[z_idx].reindex(tickers).values.astype(np.float64)
        for j in range(D):
            col = Z[:, j]
            ok = ~np.isnan(col)
            if ok.sum() < 3:
                Z[:, j] = 0.0
                continue
            r = rankdata(col[ok], method="average") / ok.sum() - 0.5
            Z[ok, j] = r
            Z[~ok, j] = 0.0
        panel[t] = Z
    return panel, D


# ─────────────────────────────────────────────────────────────────────────────
# Neutralization functions (operate on a per-bar predicted signal vector pred)
# ─────────────────────────────────────────────────────────────────────────────

def neutralize_market(pred):
    return pred - np.nanmean(pred)


def neutralize_group(pred, group_ids):
    """pred and group_ids are aligned arrays; demean within each group."""
    out = pred.copy()
    if group_ids is None or len(group_ids) != len(pred):
        return out
    df = pd.DataFrame({"p": pred, "g": group_ids})
    means = df.groupby("g")["p"].transform("mean")
    return (df["p"] - means).values


def neutralize_risk_model(pred, B):
    """Regress pred ~ B (intercept + factors) and return residuals.
    B: (N, K) loading matrix. NaNs handled by zero-imputation."""
    if B is None or B.shape[0] != len(pred):
        return pred
    valid = np.isfinite(pred) & ~np.isnan(B).any(axis=1)
    if valid.sum() < 10:
        return pred
    y = pred[valid]
    X = np.column_stack([np.ones(valid.sum()), B[valid]])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        return pred
    out = pred.copy()
    out[valid] = residuals
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Risk-model loadings builder (rebuilt per rebalance to capture changing chars)
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_loadings(tickers, classifications, matrices, t_idx):
    """Build a (N, K) loading matrix for the risk-model regression.
    Sector dummies + standardized style factors (size, value, momentum, volatility, leverage)."""
    N = len(tickers)

    # Sector dummies
    sectors = []
    sec_map = {}
    for sym in tickers:
        s = classifications.get(sym, {}).get("sector", "Unknown")
        if s not in sec_map:
            sec_map[s] = len(sec_map)
        sectors.append(sec_map[s])
    n_sec = len(sec_map)
    sec_dum = np.zeros((N, n_sec))
    for i, s in enumerate(sectors):
        sec_dum[i, s] = 1.0

    def zscore_safe(arr):
        a = np.array(arr, dtype=float)
        ok = np.isfinite(a)
        if ok.sum() < 10: return np.zeros_like(a)
        mu = a[ok].mean(); sd = a[ok].std()
        if sd < 1e-10: return np.zeros_like(a)
        out = np.zeros_like(a)
        out[ok] = (a[ok] - mu) / sd
        return out

    # Style factors at time t_idx (use chars from matrices)
    def get_row(name):
        if name not in matrices:
            return np.zeros(N)
        return matrices[name].iloc[t_idx].reindex(tickers).values

    size = zscore_safe(np.log(np.maximum(get_row("cap"), 1e-6)))
    value = zscore_safe(get_row("book_to_market"))
    mom = zscore_safe(get_row("momentum_252d") if "momentum_252d" in matrices else get_row("log_returns"))
    vol = zscore_safe(get_row("historical_volatility_60"))
    lev = zscore_safe(get_row("debt_to_equity"))
    style = np.column_stack([size, value, mom, vol, lev])

    B = np.column_stack([sec_dum, style])
    return B


# ─────────────────────────────────────────────────────────────────────────────
# Backtest with neutralization mode
# ─────────────────────────────────────────────────────────────────────────────

def run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                              tickers, classifications, matrices,
                              mode="baseline", seed=SEED, ridge=Z_RIDGE, alpha_ewma=1.0):
    rng = np.random.default_rng(seed)
    n_pairs = P // 2
    theta = rng.standard_normal((n_pairs, D))
    gamma = rng.choice(GAMMA_GRID, size=n_pairs)

    # Pre-compute group IDs (stable per-ticker)
    industries = np.array([classifications.get(t, {}).get("industry", "?") for t in tickers])
    subindustries = np.array([classifications.get(t, {}).get("subindustry", "?") for t in tickers])

    fr_history, lambda_hat = [], None
    bars_since_rebal = REBAL_EVERY
    prev_w, sm_w = None, None
    rows = []

    for t in range(start_bar, T_total - 1):
        if t not in Z_panel:
            continue
        Z_t = Z_panel[t]
        proj = (Z_t @ theta.T) * gamma[None, :]
        S_t = np.empty((Z_t.shape[0], P))
        S_t[:, 0::2] = np.sin(proj)
        S_t[:, 1::2] = np.cos(proj)

        R_t1 = (close_vals[t + 1] - close_vals[t]) / close_vals[t]
        R_t1 = np.nan_to_num(R_t1, nan=0.0)

        valid = (~np.isnan(Z_t).any(axis=1)
                 & ~np.isnan(close_vals[t]) & ~np.isnan(close_vals[t + 1]))
        N_t = int(valid.sum())
        if N_t < 5:
            continue

        S_v, R_v = S_t[valid], R_t1[valid]
        F_t1 = (1.0 / np.sqrt(N_t)) * (S_v.T @ R_v)
        fr_history.append((t + 1, F_t1))

        if t + 1 < oos_start_idx:
            continue

        if bars_since_rebal >= REBAL_EVERY or lambda_hat is None:
            cutoff_low = (t + 1) - TRAIN_BARS
            train = [fr for (idx, fr) in fr_history if cutoff_low <= idx < (t + 1)]
            if len(train) < MIN_TRAIN_BARS:
                continue
            F_train = np.vstack(train)
            T_tr, P_tr = F_train.shape
            FF = (F_train.T @ F_train) / T_tr
            A = ridge * np.eye(P_tr) + FF
            lambda_hat = np.linalg.solve(A, F_train.mean(axis=0))
            bars_since_rebal = 0

        # Predicted signal — full-N vector for neutralization compatibility
        pred_full = np.full(Z_t.shape[0], np.nan)
        pred_full[valid] = (1.0 / np.sqrt(N_t)) * (S_v @ lambda_hat)

        # ── NEUTRALIZE ────────────────────────────────────────────────────
        if mode == "baseline":
            pred_neut = pred_full.copy()
        elif mode == "market":
            pred_neut = pred_full.copy()
            pred_neut[valid] = neutralize_market(pred_full[valid])
        elif mode == "industry":
            pred_neut = pred_full.copy()
            pred_neut[valid] = neutralize_group(pred_full[valid], industries[valid])
        elif mode == "subindustry":
            pred_neut = pred_full.copy()
            pred_neut[valid] = neutralize_group(pred_full[valid], subindustries[valid])
        elif mode == "risk_model":
            B_full = build_risk_loadings(tickers, classifications, matrices, t)
            pred_neut = pred_full.copy()
            pred_neut[valid] = neutralize_risk_model(pred_full[valid], B_full[valid])
        else:
            raise ValueError(mode)

        # IC + R² on the neutralized predictor
        pred_v = pred_neut[valid]
        if pred_v.std() > 1e-12 and R_v.std() > 1e-12:
            ic_p = float(np.corrcoef(pred_v, R_v)[0, 1])
            ic_s = float(np.corrcoef(rankdata(pred_v), rankdata(R_v))[0, 1])
            r2 = ic_p ** 2
        else:
            ic_p = ic_s = r2 = 0.0

        # Build raw weight vector and normalize
        raw_w = np.zeros(Z_t.shape[0])
        raw_w[valid] = pred_v
        sm_w = raw_w.copy() if sm_w is None else (1 - alpha_ewma) * sm_w + alpha_ewma * raw_w
        sm_abs = np.abs(sm_w).sum()
        if sm_abs < 1e-12:
            bars_since_rebal += 1
            continue
        w_norm = sm_w / sm_abs

        port_ret = float(w_norm @ R_t1)
        to = float(np.abs(w_norm - prev_w).sum() / 2.0) if prev_w is not None else 0.0
        prev_w = w_norm.copy()
        bars_since_rebal += 1

        rows.append({"bar_idx": t + 1, "gross": port_ret, "turnover": to,
                     "ic_p": ic_p, "ic_s": ic_s, "r2": r2})

    df = pd.DataFrame(rows)
    for bps in TAKER_BPS_GRID:
        df[f"net_{bps:g}bps"] = df["gross"] - df["turnover"] * bps / 10000.0 * 2.0
    return df


def split_metrics(df, dates, oos_start_idx):
    """Return dict with stats per (split, fee) combo."""
    if df.empty: return {}
    df = df.copy()
    df["date"] = [dates[i] for i in df["bar_idx"]]
    is_oos = df["bar_idx"] >= oos_start_idx
    df_oos = df[is_oos].reset_index(drop=True)
    n = len(df_oos)
    split = n // 2
    splits = {"full": df_oos, "val": df_oos.iloc[:split], "test": df_oos.iloc[split:]}
    out = {}
    ann = np.sqrt(BARS_PER_YEAR)
    for tag, sub in splits.items():
        if len(sub) < 30: continue
        g = sub["gross"].values
        out[f"{tag}_sr_g"] = g.mean() / g.std(ddof=1) * ann if g.std() > 1e-12 else 0.0
        out[f"{tag}_to"]   = sub["turnover"].mean()
        out[f"{tag}_ic_p"] = sub["ic_p"].mean()
        out[f"{tag}_ir_p"] = sub["ic_p"].mean() / sub["ic_p"].std(ddof=1) * ann if sub["ic_p"].std() > 1e-12 else 0.0
        out[f"{tag}_r2"]   = sub["r2"].mean()
        for bps in TAKER_BPS_GRID:
            col = f"net_{bps:g}bps"
            nn = sub[col].values
            if nn.std(ddof=1) < 1e-12:
                continue
            out[f"{tag}_sr_n_{bps:g}bps"] = nn.mean() / nn.std(ddof=1) * ann
            out[f"{tag}_ncum_{bps:g}bps"] = nn.sum() * 100
    return out


def main():
    overall_t0 = time.time()
    print("=" * 100)
    print(f"EQUITIES AIPT WITH PIT DATA + NEUTRALIZATION COMPARISON  (P={P})")
    print("=" * 100)

    matrices, tickers, dates, close_vals, chars, classifications = load_data()
    T_total = len(dates)
    oos_start_idx = next(i for i, d in enumerate(dates) if str(d) >= OOS_START)
    start_bar = max(1, oos_start_idx - TRAIN_BARS - 10)
    print(f"  N={len(tickers)} T={T_total} OOS_start={oos_start_idx} chars={len(chars)}")
    print(f"  OOS dates: {dates[oos_start_idx]} -> {dates[-1]}")

    print(f"\nBuilding Z panel for {T_total - start_bar} bars (DELAY=1)...")
    t1 = time.time()
    Z_panel, D = build_Z_panel(matrices, tickers, chars, start_bar, T_total, delay=1)
    print(f"  Z panel D={D} built in {time.time()-t1:.1f}s")

    all_rows = []
    for mode in NEUTRALIZATIONS:
        print(f"\n--- mode = {mode} ---", flush=True)
        t0 = time.time()
        df = run_with_neutralization(P, Z_panel, close_vals, start_bar, T_total, oos_start_idx, D,
                                       tickers, classifications, matrices, mode=mode)
        m = split_metrics(df, dates, oos_start_idx)
        m["mode"] = mode
        m["minutes"] = (time.time() - t0) / 60
        all_rows.append(m)
        pd.DataFrame(all_rows).to_csv(LOG_CSV, index=False)
        print(f"  {mode:<14}  bars={len(df)}  TO={m.get('full_to',0)*100:5.1f}%  "
              f"IC={m.get('full_ic_p',0):+.4f}  IR={m.get('full_ir_p',0):+.2f}  "
              f"R²={m.get('full_r2',0):.4f}  ({(time.time()-t0)/60:.1f}min)", flush=True)
        # Per-fee breakdown
        for bps in TAKER_BPS_GRID:
            print(f"    fee={bps:g}bps  "
                  f"FULL: gSR={m.get('full_sr_g',0):+.2f} nSR={m.get(f'full_sr_n_{bps:g}bps',0):+.2f} "
                  f"ncum={m.get(f'full_ncum_{bps:g}bps',0):+.1f}%  |  "
                  f"VAL nSR={m.get(f'val_sr_n_{bps:g}bps',0):+.2f}  "
                  f"TEST nSR={m.get(f'test_sr_n_{bps:g}bps',0):+.2f}", flush=True)

    print(f"\n{'='*100}")
    print(f"DONE in {(time.time()-overall_t0)/60:.1f} min")
    print(f"CSV: {LOG_CSV}")


if __name__ == "__main__":
    main()
