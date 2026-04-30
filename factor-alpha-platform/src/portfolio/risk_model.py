"""
Multi-factor risk model — both cross-sectional alpha neutralization
AND covariance-decomposition builders for the QP optimizer.

Cov-decomposition builders (used by src/portfolio/qp.py):
  build_diagonal   : Σ ≈ diag(σ²)
  build_pca        : Σ ≈ B_pca B_pca' + diag(s²)
  build_style      : Σ ≈ B_style Σ_F B_style' + diag(s²)
  build_style_pca  : Σ ≈ B_style Σ_F B_style' + B_pca B_pca' + diag(s²)
Each returns (L_list, s²) where L_list is a list of (N, K_k) matrices and
the QP risk term is ½λ (Σ_k ||L_k' w||² + s²·w²).

Cross-sectional neutralization (style factors):

Standard Barra-ish style factors built from already-cached fundamental and
price matrices:

  Style factors (z-scored cross-sectionally each day):
    - market_beta   : 252d rolling regression of returns vs equal-weight universe
                      return; the slope is the factor exposure.
    - size          : log(market_cap)  (or log(close * shares_out) if cap missing)
    - value         : average of z-rank(book_to_market) and z-rank(earnings_yield)
                      and z-rank(free_cashflow_yield).
    - momentum_12_1 : ts_delta(close, 252) − ts_delta(close, 21) (12-1 skip)
    - profitability : z-rank(true_divide(gross_profit, assets))
                      (Novy-Marx GP/A; falls back to operating_margin)
    - low_vol       : −z(historical_volatility_60)
    - growth        : z-rank(ts_delta(s_log_1p(revenue), 252))
    - leverage      : z-rank(true_divide(net_debt, total_equity))

  Industry factors (one-hot dummies):
    - subindustry classifications (already loaded in matrices)

Two neutralization modes:
  - "residualize"  : daily cross-sectional regression of signal on factor
                      loadings; signal -> residual.
  - "demean_groups": (legacy) demean within subindustry only.

The neutralizer takes a raw alpha matrix (date × ticker) and returns a
residualized matrix of the same shape, preserving NaNs where the alpha or
factors are missing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Cov-decomposition builders for the QP risk term ½λ (Σ_k ||L_k' w||² + s²·w²)
# ---------------------------------------------------------------------------

def build_diagonal(vol_vec: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Diagonal-only risk: Σ ≈ diag(σ²). Returns ([], σ²)."""
    return [], np.maximum(vol_vec.astype(float) ** 2, 1e-8)


def build_pca(R_window: np.ndarray, k: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """K-factor PCA decomposition: Σ ≈ B B' + diag(s²).

    R_window: (T, N) returns matrix.
    Returns ([B], s²) where B is (N, k).
    """
    T, N = R_window.shape
    R = R_window - R_window.mean(axis=0, keepdims=True)
    cov = (R.T @ R) / max(T - 1, 1)
    cov = (cov + cov.T) * 0.5 + 1e-10 * np.eye(N)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return [], np.maximum(np.diag(cov), 1e-8)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    k_use = min(k, int(np.sum(eigvals > 1e-12)))
    if k_use == 0:
        return [], np.maximum(np.diag(cov), 1e-8)
    B = eigvecs[:, :k_use] * np.sqrt(np.maximum(eigvals[:k_use], 0.0))[None, :]
    if k_use < k:
        B = np.hstack([B, np.zeros((N, k - k_use))])
    common = np.sum(B ** 2, axis=1)
    s2 = np.maximum(np.diag(cov) - common, 1e-8)
    return [B], s2


def _style_decompose(R_window: np.ndarray, B_style: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-sectional regression of returns on style loadings.

    For each historical day τ: r_τ = B_style @ f_τ + e_τ.
    Returns (L_style, E, s²) where:
      L_style = B_style @ chol(Σ_F)   shape (N, K)
      E       = residual returns       shape (T, N)
      s²      = idiosyncratic variance shape (N,)
    """
    T, N = R_window.shape
    K = B_style.shape[1]
    BtB = B_style.T @ B_style + 1e-10 * np.eye(K)
    BtB_inv = np.linalg.pinv(BtB)
    F = R_window @ B_style @ BtB_inv         # (T, K) factor returns
    E = R_window - F @ B_style.T              # (T, N) residual returns
    F_dm = F - F.mean(axis=0, keepdims=True)
    Sigma_F = (F_dm.T @ F_dm) / max(T - 1, 1)
    Sigma_F = (Sigma_F + Sigma_F.T) * 0.5 + 1e-10 * np.eye(K)
    try:
        L = np.linalg.cholesky(Sigma_F)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(Sigma_F)
        L = V @ np.diag(np.sqrt(np.maximum(ev, 0.0)))
    L_style = B_style @ L                     # (N, K)
    s2 = np.maximum(np.var(E, axis=0, ddof=1), 1e-8)
    return L_style, E, s2


def build_style(R_window: np.ndarray, B_style: np.ndarray
                ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Style-factor risk model. Returns ([L_style], s²)."""
    if B_style is None or B_style.shape[1] == 0:
        return [], np.maximum(np.var(R_window, axis=0, ddof=1), 1e-8)
    L_style, _E, s2 = _style_decompose(R_window, B_style)
    return [L_style], s2


def build_style_pca(R_window: np.ndarray, B_style: np.ndarray, k_pca: int
                    ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Style + PCA-on-residual risk model.

    Σ ≈ B_style Σ_F B_style' + B_pca B_pca' + diag(s²).
    Returns ([L_style, B_pca], s²).
    """
    if B_style is None or B_style.shape[1] == 0:
        return build_pca(R_window, k_pca)
    L_style, E, _s2_pre = _style_decompose(R_window, B_style)
    pca_Ls, s2 = build_pca(E, k_pca)
    if pca_Ls:
        return [L_style, pca_Ls[0]], s2
    return [L_style], s2


# ---------------------------------------------------------------------------
# Cross-sectional helpers (shared)
# ---------------------------------------------------------------------------

def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row, with NaN-safe std."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def _rank_cs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank in [0,1] per row."""
    return df.rank(axis=1, pct=True) - 0.5


def build_style_factors(matrices: dict) -> pd.DataFrame:
    """Build daily DataFrame of style-factor LOADINGS (date × ticker × factor).

    Returns a dict[factor_name, DataFrame].  Each DataFrame is z-scored
    cross-sectionally so that the regression coefficients are interpretable.
    """
    close = matrices["close"]
    dates, tickers = close.index, close.columns

    factors = {}

    # 1. market_beta — 252d rolling regression vs equal-weight universe return
    ret = close.pct_change(fill_method=None)
    mkt_ret = ret.mean(axis=1)
    rolling_cov = ret.rolling(252, min_periods=60).cov(mkt_ret)
    rolling_var = mkt_ret.rolling(252, min_periods=60).var()
    beta = rolling_cov.div(rolling_var, axis=0)
    factors["market_beta"] = _zscore_cs(beta.reindex(index=dates, columns=tickers))

    # 2. size
    if "market_cap" in matrices and matrices["market_cap"].notna().any().any():
        cap = matrices["market_cap"]
    elif "shares_out" in matrices:
        cap = close * matrices["shares_out"]
    else:
        cap = close  # fallback
    factors["size"] = _zscore_cs(np.log(cap.clip(lower=1)))

    # 3. value — composite of B/M, EY, FCFY (whichever exist)
    val_components = []
    for f in ("book_to_market", "earnings_yield", "free_cashflow_yield"):
        if f in matrices:
            val_components.append(_zscore_cs(matrices[f]))
    if val_components:
        val = sum(val_components) / len(val_components)
        factors["value"] = val

    # 4. momentum_12_1 — total return over [-252, -21]
    mom_12 = close / close.shift(252) - 1
    mom_1  = close / close.shift(21)  - 1
    mom    = mom_12 - mom_1
    factors["momentum"] = _zscore_cs(mom)

    # 5. profitability — GP/A or fallback operating_margin
    if "gross_profit" in matrices and "assets" in matrices:
        gpa = matrices["gross_profit"] / matrices["assets"].replace(0, np.nan)
        factors["profitability"] = _zscore_cs(gpa)
    elif "operating_margin" in matrices:
        factors["profitability"] = _zscore_cs(matrices["operating_margin"])

    # 6. low_vol — negated 60d realized vol
    if "historical_volatility_60" in matrices:
        factors["low_vol"] = _zscore_cs(-matrices["historical_volatility_60"])

    # 7. growth — 1y log-revenue change
    if "revenue" in matrices:
        rev = matrices["revenue"]
        growth = np.log(rev.clip(lower=1)) - np.log(rev.shift(252).clip(lower=1))
        factors["growth"] = _zscore_cs(growth)

    # 8. leverage — net_debt / equity (or debt/equity)
    if "net_debt" in matrices and "total_equity" in matrices:
        lev = matrices["net_debt"] / matrices["total_equity"].replace(0, np.nan)
        factors["leverage"] = _zscore_cs(lev)
    elif "debt_to_equity" in matrices:
        factors["leverage"] = _zscore_cs(matrices["debt_to_equity"])

    return factors


def neutralize(alpha_df: pd.DataFrame,
               factors: dict[str, pd.DataFrame],
               classifications: Optional[pd.Series] = None,
               include_industry: bool = True) -> pd.DataFrame:
    """Residualize alpha_df cross-sectionally each day against style factors
    (and optional industry dummies).

    Implementation: per-day OLS via numpy.linalg.lstsq on the
    (n_active_names, n_factors) design matrix; alpha_residual = alpha − X·β̂.

    NaN-safe: rows with insufficient cross-sectional coverage (< 50 names) are
    returned as the original signal (no neutralization).
    """
    out = alpha_df.copy().astype(float)
    factor_names = list(factors.keys())

    # Pre-build industry dummies once (constant over time except for membership churn)
    if include_industry and classifications is not None:
        # Drop NaN groups; one-hot
        groups_unique = sorted([g for g in classifications.dropna().unique()])
        # Drop one to avoid singularity
        if len(groups_unique) > 1:
            groups_unique = groups_unique[1:]
        industry_dummies = {g: (classifications == g).astype(float)
                            for g in groups_unique}
    else:
        industry_dummies = {}

    for dt in alpha_df.index:
        try:
            y = alpha_df.loc[dt].astype(float)
        except Exception:
            continue
        valid = y.notna()
        # Build design matrix X
        cols = []
        col_names = []
        for fn in factor_names:
            if dt in factors[fn].index:
                col = factors[fn].loc[dt]
                cols.append(col); col_names.append(fn)
        for g, d in industry_dummies.items():
            cols.append(d); col_names.append(f"ind_{g}")
        if not cols:
            continue
        X = pd.concat(cols, axis=1)
        X.columns = col_names
        # Restrict to names where alpha + all factors are available
        mask = valid & X.notna().all(axis=1)
        if mask.sum() < 50:
            continue
        Xv = X[mask].values
        yv = y[mask].values
        # Add intercept
        Xv = np.hstack([np.ones((Xv.shape[0], 1)), Xv])
        # Solve via lstsq
        try:
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = yv - Xv @ beta
        out.loc[dt, mask] = resid
    return out
