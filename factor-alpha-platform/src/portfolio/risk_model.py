"""
Multi-factor risk model — both cross-sectional alpha neutralization
AND covariance-decomposition builders for the QP optimizer.

Cov-decomposition builders (used by src/portfolio/qp.py):
  build_diagonal   : Σ ≈ diag(σ²)
  build_pca        : Σ ≈ B_pca B_pca' + diag(s²)
  build_style      : Σ ≈ B_style Σ_F B_style' + diag(s²)
  build_style_pca  : Σ ≈ B_style Σ_F B_style' + B_pca B_pca' + diag(s²)
  build_ipca       : Σ ≈ B_ipca Σ_F B_ipca' + diag(s²)  where
                     B_ipca,t = Z_t Γ — Bianchi-Babiak (2022) instrumented PCA
                     with time-varying loadings driven by characteristics.
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


def build_ipca(R_window: np.ndarray, Z_window: np.ndarray, Z_today: np.ndarray,
               k: int, *, n_iter: int = 50, tol: float = 1e-6
               ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Bianchi-Babiak (2022) instrumented PCA risk model.

    Returns model:  r_{i,t+1} = Z_{i,t}' Γ f_{t+1} + ε_{i,t+1}
    where Γ ∈ R^{L×K} is a constant coefficient matrix mapping the L
    characteristics in Z_{i,t} to K latent factor loadings.  Loadings are
    *time-varying* through Z_t even though Γ is fit once on the window.

    Estimation: alternating least squares.
        Step (a): given Γ, solve T cross-sectional regressions for f_t.
        Step (b): given {f_t}, solve one stacked LS for vec(Γ).
        Iterate until ‖ΔΓ‖ < tol or n_iter exhausted.
        Identification: Γ'Γ = I_K (QR-orthogonalize each iteration).

    Then for risk: B_today = Z_today @ Γ (N×K), Σ_F = sample cov of {f_t}.
    Σ ≈ B_today Σ_F B_today' + diag(σ²_e). Returns ([L_ipca], s²)
    where L_ipca = B_today @ chol(Σ_F).

    R_window: (T, N) returns matrix (point-in-time, t = end-of-bar)
    Z_window: (T, N, L) characteristics observable at end of each bar
    Z_today : (N, L) characteristics observed at the bar we're sizing
    k       : number of latent factors (Bianchi-Babiak default = 3)
    """
    T, N = R_window.shape
    L = Z_window.shape[2]
    if T < 5 or k < 1 or L < k:
        return build_diagonal(np.std(R_window, axis=0, ddof=1) if T > 1 else np.ones(N))

    R = R_window - R_window.mean(axis=0, keepdims=True)
    Z = np.where(np.isfinite(Z_window), Z_window, 0.0).astype(np.float64)
    Z_t0 = np.where(np.isfinite(Z_today), Z_today, 0.0).astype(np.float64)
    R = np.where(np.isfinite(R), R, 0.0).astype(np.float64)

    rng = np.random.default_rng(0)
    Gamma, _ = np.linalg.qr(rng.standard_normal((L, k)))
    F = np.zeros((T, k))

    for it in range(n_iter):
        # Step (a): estimate f_t given Γ
        for t in range(T):
            ZG = Z[t] @ Gamma                               # (N, K)
            A = ZG.T @ ZG + 1e-8 * np.eye(k)
            try:
                F[t] = np.linalg.solve(A, ZG.T @ R[t])
            except np.linalg.LinAlgError:
                F[t] = 0.0

        # Step (b): estimate Γ given {f_t} via stacked LS.
        # design[(t,i), (k,l)] = F[t,k] · Z[t,i,l] ;   target = R[t,i]
        # Solving with vec_Gamma layout (k_outer, l_inner) → reshape (K, L).T → (L, K)
        F_kron_Z = (F[:, None, :, None] * Z[:, :, None, :])  # (T, N, K, L)
        big_X = F_kron_Z.reshape(T * N, k * L)
        big_y = R.reshape(T * N)
        try:
            vec_G, *_ = np.linalg.lstsq(big_X, big_y, rcond=None)
        except np.linalg.LinAlgError:
            break
        Gamma_new = vec_G.reshape(k, L).T                   # (L, K)
        # Re-orthonormalize for identification
        Gamma_new, _ = np.linalg.qr(Gamma_new)
        diff = np.linalg.norm(Gamma_new - Gamma) / (np.linalg.norm(Gamma) + 1e-10)
        Gamma = Gamma_new
        if diff < tol:
            break

    # Final f_t with converged Γ
    for t in range(T):
        ZG = Z[t] @ Gamma
        A = ZG.T @ ZG + 1e-8 * np.eye(k)
        try:
            F[t] = np.linalg.solve(A, ZG.T @ R[t])
        except np.linalg.LinAlgError:
            F[t] = 0.0

    B_today = Z_t0 @ Gamma                                  # (N, K)
    F_dm = F - F.mean(axis=0, keepdims=True)
    Sigma_F = (F_dm.T @ F_dm) / max(T - 1, 1)
    Sigma_F = (Sigma_F + Sigma_F.T) * 0.5 + 1e-10 * np.eye(k)
    try:
        chol = np.linalg.cholesky(Sigma_F)
    except np.linalg.LinAlgError:
        ev, V = np.linalg.eigh(Sigma_F)
        chol = V @ np.diag(np.sqrt(np.maximum(ev, 0.0)))
    L_ipca = B_today @ chol                                 # (N, K)

    # Per-name residual variance over the window
    resid = R - np.einsum("til,lk,tk->ti", Z, Gamma, F)
    s2 = np.maximum(np.var(resid, axis=0, ddof=1), 1e-8)
    return [L_ipca], s2


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


def build_crypto_ipca_characteristics(matrices: dict) -> dict:
    """Crypto-perp characteristic stack for build_ipca, per Bianchi-Babiak (2022).

    Their five canonical return drivers — liquidity, size, market beta,
    reversal, downside risk — translated to KuCoin 4h fields. All factors are
    z-scored cross-sectionally per bar so Γ scaling is comparable across
    factors.

    Returns dict[name -> DataFrame(date×ticker)]. The runner stacks these into
    a (T, N, L) tensor and slices for IPCA fitting.
    """
    factors = {}

    # 1. liquidity — log dollar-traded over 60 bars (rebal cadence)
    if "adv60" in matrices:
        factors["liquidity"] = _zscore_cs(np.log1p(matrices["adv60"].clip(lower=0)))

    # 2. size — turnover-weighted size proxy: log(close × adv60)
    if "close" in matrices and "adv60" in matrices:
        size_raw = matrices["close"] * matrices["adv60"]
        factors["size"] = _zscore_cs(np.log1p(size_raw.clip(lower=0)))

    # 3. market_beta — direct field on KuCoin (rolling beta to BTC)
    if "beta_to_btc" in matrices:
        factors["market_beta"] = _zscore_cs(matrices["beta_to_btc"])

    # 4. momentum — slow trend (60d, like Bianchi-Babiak's MOM_60)
    if "momentum_60d" in matrices:
        factors["momentum"] = _zscore_cs(matrices["momentum_60d"])

    # 5. reversal — short-window negative momentum (1-3d at 4h cadence)
    if "momentum_5d" in matrices:
        factors["reversal"] = _zscore_cs(-matrices["momentum_5d"])

    # 6. downside_risk — Parkinson 60-bar vol (lower = quality)
    if "parkinson_volatility_60" in matrices:
        factors["downside_risk"] = _zscore_cs(-matrices["parkinson_volatility_60"])

    # 7. trade_intensity — volume_ratio z (relative-to-history flow)
    if "volume_ratio_20d" in matrices:
        factors["trade_intensity"] = _zscore_cs(np.log1p(
            matrices["volume_ratio_20d"].clip(lower=0)))

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
