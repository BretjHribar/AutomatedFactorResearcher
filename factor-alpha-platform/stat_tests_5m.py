"""
Statistical Factor Tests — Fama-MacBeth Cross-Sectional Regressions & GRS Test

Runs rigorous asset-pricing tests on all active alphas:
  1. Fama-MacBeth (1973): Cross-sectional regression of returns on lagged signals
     at each time t. Time-series of betas gives t-stats with Newey-West correction.
  2. GRS (Gibbons-Ross-Shanken 1989): Joint test that all alpha intercepts are zero.
     Tests whether the set of factors jointly prices the cross-section.
  3. Per-alpha: individual FM t-stat, mean lambda, R-squared, autocorrelation

Usage:
    python stat_tests_5m.py                # Full report on all active alphas
    python stat_tests_5m.py --alpha-id 16  # Test a single alpha
"""

import sys, os, argparse, sqlite3, time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config (match eval_alpha_5m.py) ──
TRAIN_START = "2026-02-15"
TRAIN_END   = "2026-03-05"
UNIVERSE    = "BINANCE_TOP100"
INTERVAL    = "5m"
BARS_PER_DAY = 288
COVERAGE_CUTOFF = 0.3
DB_PATH     = "data/alphas_5m.db"

# Statistical significance thresholds
FM_TSTAT_THRESHOLD = 1.96   # 95% confidence (two-sided)
GRS_PVALUE_THRESHOLD = 0.05


def load_matrices():
    """Load matrices and universe for train period."""
    mat_dir = Path("data/binance_cache/matrices/5m")
    uni_path = Path(f"data/binance_cache/universes/{UNIVERSE}_{INTERVAL}.parquet")
    
    universe_df = pd.read_parquet(uni_path)
    coverage = universe_df.sum(axis=0) / len(universe_df)
    valid_tickers = sorted(coverage[coverage > COVERAGE_CUTOFF].index.tolist())
    
    print(f"  Loading matrices ({len(valid_tickers)} tickers)...", flush=True)
    matrices = {}
    for fp in sorted(mat_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        cols = [c for c in valid_tickers if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols].loc[TRAIN_START:TRAIN_END]
    
    universe = universe_df[valid_tickers].loc[TRAIN_START:TRAIN_END]
    print(f"  Loaded {len(matrices)} fields", flush=True)
    return matrices, universe, valid_tickers


def evaluate_alpha(expression, matrices):
    """Evaluate an alpha expression."""
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=matrices)
    return engine.evaluate(expression)


def fama_macbeth_test(signal_df, returns_df, universe_df, nw_lags=6):
    """
    Fama-MacBeth (1973) cross-sectional regression.
    
    At each time t:
      r_i(t+1) = gamma_0(t) + gamma_1(t) * signal_i(t) + epsilon_i(t)
    
    Then test H0: E[gamma_1] = 0 using Newey-West adjusted standard errors.
    
    Returns dict with:
      - lambda_mean: average cross-sectional coefficient
      - lambda_tstat: NW-corrected t-statistic 
      - lambda_series: time series of lambdas
      - r2_mean: average cross-sectional R-squared
      - n_periods: number of valid cross-sections
    """
    # Align signal and returns
    sig = signal_df.copy()
    uni_mask = universe_df.reindex(index=sig.index, columns=sig.columns).fillna(False)
    sig = sig.where(uni_mask, np.nan)
    
    ret = returns_df.reindex(index=sig.index, columns=sig.columns)
    
    # Cross-sectional regression at each time t
    lambdas = []
    intercepts = []
    r2s = []
    dates_used = []
    
    common_dates = sig.index.intersection(ret.index)
    
    for i in range(len(common_dates) - 1):
        t = common_dates[i]
        t1 = common_dates[i + 1]
        
        x = sig.loc[t].values
        y = ret.loc[t1].values
        
        valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-15)
        n_valid = valid.sum()
        
        if n_valid < 15:
            continue
        
        x_v = x[valid]
        y_v = y[valid]
        
        # OLS: y = a + b*x
        X = np.column_stack([np.ones(n_valid), x_v])
        try:
            beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        
        intercepts.append(beta[0])
        lambdas.append(beta[1])
        
        # R-squared
        y_hat = X @ beta
        ss_res = np.sum((y_v - y_hat) ** 2)
        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2s.append(r2)
        dates_used.append(t)
    
    if len(lambdas) < 20:
        return None
    
    lambdas = np.array(lambdas)
    intercepts = np.array(intercepts)
    
    # Newey-West corrected standard error
    T = len(lambdas)
    mean_lambda = np.mean(lambdas)
    
    # Autocovariance function
    demeaned = lambdas - mean_lambda
    gamma_0 = np.mean(demeaned ** 2)
    
    nw_var = gamma_0
    for lag in range(1, nw_lags + 1):
        if lag >= T:
            break
        weight = 1 - lag / (nw_lags + 1)  # Bartlett kernel
        gamma_lag = np.mean(demeaned[lag:] * demeaned[:-lag])
        nw_var += 2 * weight * gamma_lag
    
    nw_se = np.sqrt(nw_var / T) if nw_var > 0 else 1e-10
    tstat = mean_lambda / nw_se
    
    # Lambda autocorrelation (useful diagnostic)
    if T > 2:
        lambda_ac1 = np.corrcoef(lambdas[:-1], lambdas[1:])[0, 1]
    else:
        lambda_ac1 = 0
    
    return {
        "lambda_mean": mean_lambda,
        "lambda_tstat": tstat,
        "lambda_pvalue": 2 * (1 - sp_stats.norm.cdf(abs(tstat))),
        "lambda_series": lambdas,
        "intercept_mean": np.mean(intercepts),
        "r2_mean": np.mean(r2s),
        "lambda_ac1": lambda_ac1,
        "n_periods": T,
        "nw_se": nw_se,
    }


def grs_test(alpha_returns, market_returns=None):
    """
    Gibbons-Ross-Shanken (1989) test.
    
    Tests H0: All alpha intercepts (from time-series regressions on the market)
    are jointly zero.
    
    If market_returns is provided, runs factor model: r_i = alpha_i + beta_i * r_m + eps_i
    If not, runs unconditional test on the alpha return means.
    
    Args:
        alpha_returns: DataFrame (T, K) of factor-mimicking portfolio returns
        market_returns: Series (T,) of market returns (optional)
    
    Returns dict with:
        - grs_stat: GRS F-statistic
        - grs_pvalue: p-value from F distribution
        - n_factors: K
        - n_periods: T
    """
    T, K = alpha_returns.shape
    
    if T < K + 5:
        return {"grs_stat": np.nan, "grs_pvalue": 1.0, "n_factors": K, "n_periods": T}
    
    if market_returns is not None:
        # Time-series regression for each alpha
        alphas_vec = np.zeros(K)
        resid = np.zeros((T, K))
        
        X = np.column_stack([np.ones(T), market_returns.values[:T]])
        
        for k in range(K):
            y = alpha_returns.iloc[:, k].values
            valid = np.isfinite(y) & np.isfinite(X[:, 1])
            if valid.sum() < 10:
                alphas_vec[k] = 0
                resid[:, k] = 0
                continue
            beta = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]
            alphas_vec[k] = beta[0]
            resid[valid, k] = y[valid] - X[valid] @ beta
        
        # Market Sharpe squared
        mkt_mean = np.nanmean(market_returns.values)
        mkt_var = np.nanvar(market_returns.values)
        theta_sq = (mkt_mean ** 2) / mkt_var if mkt_var > 0 else 0
        
        # Residual covariance
        Sigma = np.cov(resid.T)
        if Sigma.ndim < 2:
            Sigma = np.array([[Sigma]])
    else:
        # Unconditional: just test if means are jointly zero
        alphas_vec = alpha_returns.mean(axis=0).values
        Sigma = alpha_returns.cov().values
        theta_sq = 0
    
    # GRS statistic
    try:
        Sigma_inv = np.linalg.inv(Sigma)
        grs_core = alphas_vec @ Sigma_inv @ alphas_vec
        
        # GRS F-stat
        grs_stat = (T / K) * ((T - K - 1) / (T - 2)) * grs_core / (1 + theta_sq)
        
        # P-value from F(K, T-K-1)
        df1, df2 = K, T - K - 1
        if df2 > 0:
            grs_pvalue = 1 - sp_stats.f.cdf(grs_stat, df1, df2)
        else:
            grs_pvalue = 1.0
            
    except np.linalg.LinAlgError:
        grs_stat = np.nan
        grs_pvalue = 1.0
    
    return {
        "grs_stat": grs_stat,
        "grs_pvalue": grs_pvalue,
        "n_factors": K,
        "n_periods": T,
        "alpha_intercepts": alphas_vec,
    }


def build_factor_returns(signal_df, returns_df, universe_df, n_quantiles=5):
    """
    Build long-short factor-mimicking portfolio returns.
    
    At each bar: go long top quintile, short bottom quintile of signal.
    Weight equally within each leg. Return = mean(top) - mean(bottom).
    """
    sig = signal_df.copy()
    uni_mask = universe_df.reindex(index=sig.index, columns=sig.columns).fillna(False)
    sig = sig.where(uni_mask, np.nan)
    
    ret = returns_df.reindex(index=sig.index, columns=sig.columns)
    
    factor_rets = []
    dates_used = []
    
    for i in range(len(sig.index) - 1):
        t = sig.index[i]
        t1 = sig.index[i + 1] if i + 1 < len(ret.index) else None
        if t1 is None or t1 not in ret.index:
            continue
        
        s = sig.loc[t].dropna()
        r = ret.loc[t1].reindex(s.index).dropna()
        common = s.index.intersection(r.index)
        
        if len(common) < 20:
            factor_rets.append(np.nan)
            dates_used.append(t)
            continue
        
        s_c = s[common]
        r_c = r[common]
        
        # Quantile cutoffs
        q_low = s_c.quantile(1 / n_quantiles)
        q_high = s_c.quantile(1 - 1 / n_quantiles)
        
        long_mask = s_c >= q_high
        short_mask = s_c <= q_low
        
        if long_mask.sum() < 3 or short_mask.sum() < 3:
            factor_rets.append(np.nan)
        else:
            fr = r_c[long_mask].mean() - r_c[short_mask].mean()
            factor_rets.append(fr)
        
        dates_used.append(t)
    
    return pd.Series(factor_rets, index=dates_used)


def run_all_tests():
    """Run FM and GRS tests on all active alphas."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT a.id, a.expression, COALESCE(e.ic_mean, 0), COALESCE(e.sharpe_is, 0)
        FROM alphas a LEFT JOIN evaluations e ON e.alpha_id = a.id
        WHERE a.archived = 0
        ORDER BY COALESCE(e.ic_mean, 0) DESC
    """).fetchall()
    conn.close()
    
    if not rows:
        print("No active alphas found.")
        return
    
    print(f"\n{'='*80}")
    print(f"  STATISTICAL FACTOR TESTS — {len(rows)} Active Alphas")
    print(f"  Period: {TRAIN_START} to {TRAIN_END} ({INTERVAL})")
    print(f"{'='*80}")
    
    matrices, universe, valid_tickers = load_matrices()
    close = matrices.get("close")
    returns = close.pct_change()
    
    # Get market return (equal-weighted)
    market_ret = returns.mean(axis=1)
    
    results = []
    factor_return_df = {}
    
    for alpha_id, expression, ic_mean, is_sharpe in rows:
        try:
            alpha_raw = evaluate_alpha(expression, matrices)
            if alpha_raw is None:
                continue
        except Exception as e:
            print(f"  Alpha #{alpha_id}: evaluation failed: {e}")
            continue
        
        # 1. Fama-MacBeth test
        fm = fama_macbeth_test(alpha_raw, returns, universe)
        if fm is None:
            continue
        
        # 2. Build factor returns for GRS
        fret = build_factor_returns(alpha_raw, returns, universe)
        factor_return_df[alpha_id] = fret
        
        sig_star = ""
        if abs(fm["lambda_tstat"]) >= 2.58:
            sig_star = "***"
        elif abs(fm["lambda_tstat"]) >= 1.96:
            sig_star = "**"
        elif abs(fm["lambda_tstat"]) >= 1.65:
            sig_star = "*"
        
        results.append({
            "alpha_id": alpha_id,
            "ic_mean": ic_mean,
            "is_sharpe": is_sharpe,
            "fm_lambda": fm["lambda_mean"],
            "fm_tstat": fm["lambda_tstat"],
            "fm_pvalue": fm["lambda_pvalue"],
            "fm_r2": fm["r2_mean"],
            "fm_ac1": fm["lambda_ac1"],
            "fm_n": fm["n_periods"],
            "sig_star": sig_star,
        })
    
    # Print Fama-MacBeth results
    print(f"\n  {'='*78}")
    print(f"  FAMA-MACBETH CROSS-SECTIONAL REGRESSION RESULTS")
    print(f"  H0: Factor premium (lambda) = 0 | Newey-West corrected (6 lags)")
    print(f"  {'='*78}")
    print(f"  {'ID':>4s} {'IC':>8s} {'IS SR':>7s} {'Lambda':>10s} {'t-stat':>8s} {'p-val':>8s} {'R2':>7s} {'AC(1)':>7s} {'N':>5s} {'Sig':>4s}")
    print(f"  {'-'*78}")
    
    n_significant = 0
    for r in sorted(results, key=lambda x: abs(x["fm_tstat"]), reverse=True):
        if abs(r["fm_tstat"]) >= FM_TSTAT_THRESHOLD:
            n_significant += 1
        print(f"  #{r['alpha_id']:3d} {r['ic_mean']:+.5f} {r['is_sharpe']:+.2f} "
              f"{r['fm_lambda']:+.2e} {r['fm_tstat']:+7.3f} {r['fm_pvalue']:.4f} "
              f"{r['fm_r2']:.4f} {r['fm_ac1']:+.3f} {r['fm_n']:5d} {r['sig_star']:>4s}")
    
    print(f"\n  {n_significant}/{len(results)} alphas significant at 5% level (|t| >= 1.96)")
    
    # 3. GRS Joint Test
    if len(factor_return_df) >= 2:
        # Align all factor returns
        all_fret = pd.DataFrame(factor_return_df).dropna()
        
        if len(all_fret) > 50 and len(all_fret.columns) >= 2:
            grs = grs_test(all_fret, market_ret.reindex(all_fret.index))
            
            print(f"\n  {'='*78}")
            print(f"  GIBBONS-ROSS-SHANKEN (GRS) JOINT TEST")
            print(f"  H0: All factor alphas are jointly zero (after controlling for market)")
            print(f"  {'='*78}")
            print(f"  GRS F-stat:   {grs['grs_stat']:.4f}")
            print(f"  p-value:      {grs['grs_pvalue']:.6f}")
            print(f"  K (factors):  {grs['n_factors']}")
            print(f"  T (periods):  {grs['n_periods']}")
            
            if grs['grs_pvalue'] < GRS_PVALUE_THRESHOLD:
                print(f"  RESULT: REJECT H0 -- Factors have significant joint alpha (p < {GRS_PVALUE_THRESHOLD})")
            else:
                print(f"  RESULT: FAIL TO REJECT H0 -- No significant joint alpha")
            
            # Individual factor intercept magnitudes
            if 'alpha_intercepts' in grs and grs['alpha_intercepts'] is not None:
                intercepts = grs['alpha_intercepts']
                alpha_ids = list(factor_return_df.keys())
                print(f"\n  Per-factor intercepts (annualized bps):")
                for i, aid in enumerate(alpha_ids[:len(intercepts)]):
                    ann_bps = intercepts[i] * BARS_PER_DAY * 365 * 10000
                    print(f"    Alpha #{aid:3d}: {ann_bps:+.1f} bps/year")
    
    print(f"\n{'='*80}")
    return results


def test_single_alpha(alpha_id):
    """Run FM test on a single alpha for gate evaluation."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT expression FROM alphas WHERE id=? AND archived=0", 
                        (alpha_id,)).fetchone()
    conn.close()
    
    if not row:
        print(f"Alpha #{alpha_id} not found")
        return None
    
    matrices, universe, _ = load_matrices()
    close = matrices.get("close")
    returns = close.pct_change()
    
    alpha_raw = evaluate_alpha(row[0], matrices)
    if alpha_raw is None:
        return None
    
    fm = fama_macbeth_test(alpha_raw, returns, universe)
    if fm is None:
        return None
    
    print(f"\n  Alpha #{alpha_id} FM Test:")
    print(f"    Lambda:  {fm['lambda_mean']:+.2e}")
    print(f"    t-stat:  {fm['lambda_tstat']:+.3f}")
    print(f"    p-value: {fm['lambda_pvalue']:.6f}")
    print(f"    R2:      {fm['r2_mean']:.6f}")
    print(f"    AC(1):   {fm['lambda_ac1']:+.3f}")
    print(f"    Periods: {fm['n_periods']}")
    sig = abs(fm['lambda_tstat']) >= FM_TSTAT_THRESHOLD
    print(f"    Significant: {'YES' if sig else 'NO'} (threshold: |t| >= {FM_TSTAT_THRESHOLD})")
    return fm


# ============================================================================
# GATE FUNCTION (importable by eval_alpha_5m.py)
# ============================================================================

def fama_macbeth_gate(alpha_raw, returns_df, universe_df, threshold=1.65):
    """
    Quick Fama-MacBeth significance test for use as an alpha quality gate.
    
    Args:
        alpha_raw: raw alpha signal DataFrame
        returns_df: returns DataFrame
        universe_df: universe mask DataFrame
        threshold: minimum |t-stat| to pass (default 1.65 = 90% confidence)
    
    Returns:
        (passes, tstat, pvalue)
    """
    fm = fama_macbeth_test(alpha_raw, returns_df, universe_df, nw_lags=6)
    if fm is None:
        return False, 0.0, 1.0
    
    passes = abs(fm["lambda_tstat"]) >= threshold
    return passes, fm["lambda_tstat"], fm["lambda_pvalue"]


def main():
    parser = argparse.ArgumentParser(description="Statistical Factor Tests (FM + GRS)")
    parser.add_argument("--alpha-id", type=int, default=None, help="Test single alpha by ID")
    args = parser.parse_args()
    
    if args.alpha_id:
        test_single_alpha(args.alpha_id)
    else:
        run_all_tests()


if __name__ == "__main__":
    main()
