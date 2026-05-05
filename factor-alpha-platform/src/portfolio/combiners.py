"""
Alpha Signal Combiners — shared library for IB and Crypto portfolios.

All combiners have the same interface:
    combined_df = combiner_fn(alpha_signals, matrices, universe_df, returns_df, **kwargs)

Where:
    alpha_signals: dict of {alpha_id: raw_signal_DataFrame}
    matrices: dict of {field_name: DataFrame}  (must include 'close')
    universe_df: boolean DataFrame mask
    returns_df: DataFrame of returns
    Returns: DataFrame of combined signal weights (date × ticker)

Implements:
    1. Equal Weight
    2. Adaptive (rolling expected return)
    3. Risk Parity (inverse volatility)
    4. Billions (Kakushadze regression — "How to Combine a Billion Alphas")
"""

import numpy as np
import pandas as pd
from sklearn import linear_model


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def process_signal(alpha_df, universe_df=None, max_wt=0.01):
    """Normalize alpha signal: universe mask, demean, L1-normalize, clip."""
    signal = alpha_df.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(
            index=signal.index, columns=signal.columns
        ).fillna(False).astype(bool)
        signal = signal.where(uni_mask, np.nan)

    mean_val = signal.mean(axis=1)
    signal = signal.sub(mean_val, axis=0)
    abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
    signal = signal.div(abs_sum, axis=0)
    signal = signal.clip(lower=-max_wt, upper=max_wt)
    return signal.fillna(0.0)


# ============================================================================
# COMBINER: EQUAL WEIGHT
# ============================================================================

def _prepare_signals(alpha_signals, universe_df=None, max_wt=0.01,
                     signals_are_preprocessed=False):
    if signals_are_preprocessed:
        return {aid: raw.fillna(0.0) for aid, raw in alpha_signals.items()}
    return {
        aid: process_signal(raw, universe_df=universe_df, max_wt=max_wt)
        for aid, raw in alpha_signals.items()
    }


def combiner_equal(alpha_signals, matrices, universe_df, returns_df,
                   max_wt=0.01, signals_are_preprocessed=False):
    """Equal-weight: average all alpha signals after per-alpha normalization."""
    combined = None
    n = 0
    for aid, normed in _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    ).items():
        if combined is None:
            combined = normed.copy()
        else:
            combined = combined.add(normed, fill_value=0)
        n += 1
    if n > 0:
        combined = combined / n
    return combined


# ============================================================================
# COMBINER: ADAPTIVE (rolling expected return)
# ============================================================================

def combiner_adaptive(alpha_signals, matrices, universe_df, returns_df,
                      lookback=504, max_wt=0.01, signals_are_preprocessed=False):
    """Adaptive: weight by rolling expected factor return (positive ER only)."""
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data)
    rolling_er = fr_df.rolling(window=lookback, min_periods=60).mean()
    weights = rolling_er.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


# ============================================================================
# COMBINER: RISK PARITY (inverse volatility)
# ============================================================================

def combiner_risk_parity(alpha_signals, matrices, universe_df, returns_df,
                         lookback=504, max_wt=0.01, signals_are_preprocessed=False):
    """Risk Parity: weight inversely by rolling factor volatility."""
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data)
    rolling_vol = fr_df.rolling(window=lookback, min_periods=60).std()
    rolling_er  = fr_df.rolling(window=lookback, min_periods=60).mean()
    inv_vol = (1.0 / rolling_vol.replace(0, np.nan)).fillna(0)
    inv_vol = inv_vol.where(rolling_er > 0, 0)
    wsum = inv_vol.sum(axis=1).replace(0, np.nan)
    weights_norm = inv_vol.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


# ============================================================================
# COMBINER: BILLIONS (Kakushadze regression)
# ============================================================================

def combiner_billions(alpha_signals, matrices, universe_df, returns_df,
                      optim_lookback=60, max_wt=0.01, signals_are_preprocessed=False):
    """
    BILLIONS regression combiner — Kakushadze "How to Combine a Billion Alphas".

    Algorithm:
      1. Compute factor returns for each alpha (scalar PnL of normed signal, lagged 1 bar)
      2. alphasExpectedReturns = rolling SMA(optim_lookback), shifted 1 bar, clipped >= 0
      3. Walk-forward (one bar at a time), for each bar t:
           a. Window = factor returns [t : t + optim_lookback]   (shape: lookback x N)
           b. bilAlphasDFdemeaned = window - window.mean(axis=0)  (demean along time)
           c. sampleStd = bilAlphasDFdemeaned.std(axis=0)         (std per alpha)
           d. normalizedDemeanedReturns = bilAlphasDFdemeaned / sampleStd
           e. Y_is = A_is = normalizedDemeanedReturns.iloc[:, :optim_lookback]
           f. subAlphaExpRet = alphasExpectedReturns[t:t+lookback] / sampleStd
           g. reg.fit(A_is, subAlphaExpRet)  -- LinearRegression, no intercept
           h. residuals = reg.predict(A_is) - subAlphaExpRet
           i. optimizedWeights = residuals / sampleStd
           j. optimizedWeights = optimizedWeights / optimizedWeights.sum(axis=1)
           k. Store last row at alpha_weights_ts[t + optim_lookback + 1]
      4. combined = sum_i( alpha_weights_ts[:,i] * normed_signals[i] )
    """
    close_df = matrices["close"]
    dates    = close_df.index
    tickers  = close_df.columns.tolist()
    n_bars   = len(dates)
    aid_list = list(alpha_signals.keys())
    n_alphas = len(aid_list)

    # Step 0: normalize each alpha's raw signal
    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    # Step 1: compute factor returns (daily scalar PnL per alpha)
    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n  = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data, index=dates)   # shape (T, N_alphas)

    # Step 2: alphasExpectedReturns = sma(fr_df, lookback).shift(1), clip >= 0
    alphas_exp_ret = (
        fr_df.rolling(window=optim_lookback, min_periods=max(1, optim_lookback // 2))
             .mean()
             .shift(1)
    )
    alphas_exp_ret = alphas_exp_ret.clip(lower=0)

    # Step 3: initialize alpha weights (equal weight at every bar)
    alpha_weights_ts = pd.DataFrame(
        1.0 / n_alphas,
        index=dates, columns=aid_list
    )

    reg = linear_model.LinearRegression(fit_intercept=False)

    # Step 4: walk-forward loop
    optim_test_start = 0

    for test_start in range(optim_test_start + 1, n_bars - optim_lookback - 2):
        optim_end = test_start + optim_lookback

        if optim_end + 1 >= n_bars:
            break

        try:
            # a: window of factor returns
            bil_alphas_df = fr_df.iloc[test_start:optim_end].copy()

            # b: demean along time axis
            bil_demeaned = bil_alphas_df - bil_alphas_df.mean(axis=0)

            # c: sample std per alpha
            sample_std = bil_demeaned.std(axis=0)
            sample_std = sample_std.replace(0, np.nan)

            # d: normalize
            normalized = bil_demeaned.divide(sample_std)

            # e: Y_is = A_is = normalizedDemeanedReturns[:, :optim_lookback]
            Y_is = normalized.iloc[:, :optim_lookback]
            A_is = Y_is

            # f: expected returns for window, normalized by same std
            sub_exp_ret = alphas_exp_ret.iloc[test_start:optim_end].copy()
            sub_exp_ret = sub_exp_ret.divide(sample_std)
            sub_exp_ret = sub_exp_ret.fillna(0.0)

            # g: linear regression (multi-output, no intercept)
            X_train = A_is.fillna(0.0).values
            Y_train = sub_exp_ret.values
            reg.fit(X_train, Y_train)

            # h: residuals = predicted - actual
            residuals_vals = reg.predict(X_train) - Y_train
            residuals = pd.DataFrame(
                residuals_vals,
                index=sub_exp_ret.index, columns=sub_exp_ret.columns
            )

            # i: weights = residuals / std
            opt_weights = residuals.divide(sample_std)

            # j: normalize rows to sum = 1
            row_sums = opt_weights.sum(axis=1).replace(0, np.nan)
            opt_weights = opt_weights.div(row_sums, axis=0)

            # k: take last row, store at optimEnd + 1
            final_weights = opt_weights.iloc[-1]
            alpha_weights_ts.iloc[optim_end + 1] = final_weights.values

        except Exception:
            pass

    # Step 5: combine alpha signals using time-varying scalar weights
    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid in aid_list:
        w = alpha_weights_ts[aid]
        combined = combined.add(normed_signals[aid].mul(w, axis=0))

    return combined


# ============================================================================
# COMBINER: IC-WEIGHTED (rolling cross-sectional Pearson IC vs forward returns)
# ============================================================================

def combiner_ic_weighted(alpha_signals, matrices, universe_df, returns_df,
                         lookback=126, max_wt=0.01, signals_are_preprocessed=False):
    """
    IC-weighted: weight each alpha by its rolling mean cross-sectional IC.

    For each alpha and date t:
      ic_t = corr(signal_{t-1}, return_t) across the cross-section
      ic_smooth_t = rolling_mean(ic_t, lookback)
    Weights = max(ic_smooth, 0), L1-normalized across alphas.
    """
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    ret_df = returns_df.reindex(index=dates, columns=tickers)

    # Per-alpha daily IC: corr(signal at t-1, return at t) across cross-section
    ic_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ic = lagged.corrwith(ret_df, axis=1)
        ic_data[aid] = ic

    ic_df = pd.DataFrame(ic_data, index=dates)
    rolling_ic = ic_df.rolling(window=lookback, min_periods=20).mean()
    weights = rolling_ic.clip(lower=0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


# ============================================================================
# COMBINER: SHARPE-WEIGHTED (rolling standalone Sharpe of each alpha)
# ============================================================================

def combiner_sharpe_weighted(alpha_signals, matrices, universe_df, returns_df,
                             lookback=252, max_wt=0.01, signals_are_preprocessed=False):
    """
    Sharpe-weighted: weight each alpha by its rolling standalone Sharpe ratio
    of factor returns. Negative-Sharpe alphas get zero weight.
    """
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data, index=dates)
    rolling_mean = fr_df.rolling(window=lookback, min_periods=60).mean()
    rolling_std  = fr_df.rolling(window=lookback, min_periods=60).std()
    rolling_sr   = (rolling_mean / rolling_std.replace(0, np.nan))
    weights = rolling_sr.clip(lower=0).fillna(0)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined


# ============================================================================
# COMBINER: TOP-N BY PRECOMPUTED TRAIN SHARPE (research selection)
# ============================================================================

def combiner_topn_train(alpha_signals, matrices, universe_df, returns_df, *,
                        train_sharpes, top_n=30, max_wt=None,
                        signals_are_preprocessed=False):
    """Equal-weight the top-N alphas by precomputed TRAIN Sharpe.

    Convention matches the canonical eval framework (update_wq_alphas_db.py):
      1. select top-N alpha_ids by `train_sharpes`
      2. cross-sectionally z-score each selected signal per bar
      3. average the z-scored signals
      4. final demean + L1-normalize  (matches signal_to_portfolio)

    Unlike combiner_equal etc., this combiner does NOT call process_signal per
    alpha (which would apply universe mask + clip). The matrices/universe are
    expected to be pre-filtered before alpha evaluation — and no clip is
    applied here, leaving that to the caller's post-combiner stage.

    Args:
        alpha_signals: dict {alpha_id: raw_signal_DataFrame}
        matrices: dict (unused — kept for signature compat)
        universe_df: bool DataFrame (unused for this combiner)
        returns_df: DataFrame (unused for this combiner)
        train_sharpes: dict {alpha_id: train_sharpe}  — required, selects
                       top-N. Alphas not in this dict are skipped.
        top_n: select top-N by train_sharpes
        max_wt: optional final clip (None = no clip — the typical research
                convention)
    """
    eligible = {aid: s for aid, s in train_sharpes.items() if aid in alpha_signals}
    if not eligible:
        return pd.DataFrame()
    selected_ids = [aid for aid, _ in sorted(eligible.items(),
                                              key=lambda kv: -kv[1])[:top_n]]

    if signals_are_preprocessed:
        sig_sum = None
        for aid in selected_ids:
            s = alpha_signals[aid].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            sig_sum = s if sig_sum is None else sig_sum.add(s, fill_value=0)
        out = (sig_sum / len(selected_ids)).fillna(0.0)
    else:
        sig_sum = None
        for aid in selected_ids:
            s = alpha_signals[aid].replace([np.inf, -np.inf], np.nan)
            mu = s.mean(axis=1)
            sd = s.std(axis=1).replace(0, np.nan)
            s_zs = s.sub(mu, axis=0).div(sd, axis=0)
            sig_sum = s_zs if sig_sum is None else sig_sum.add(s_zs, fill_value=0)
        avg_sig = sig_sum / len(selected_ids)

        s = avg_sig.replace([np.inf, -np.inf], np.nan)
        demean = s.sub(s.mean(axis=1), axis=0)
        gross = demean.abs().sum(axis=1).replace(0, np.nan)
        out = demean.div(gross, axis=0).fillna(0)
    if max_wt is not None:
        out = out.clip(lower=-max_wt, upper=max_wt)
    return out


# ============================================================================
# COMBINER: TOP-N (select only top N alphas by rolling Sharpe each bar)
# ============================================================================

def combiner_topn_sharpe(alpha_signals, matrices, universe_df, returns_df,
                          lookback=252, top_n=10, max_wt=0.01,
                          signals_are_preprocessed=False):
    """
    Top-N: at each bar, equal-weight the N alphas with highest rolling Sharpe.
    Combats noise from low-quality alphas dragging down composite.
    """
    close_df = matrices["close"]
    dates = close_df.index
    tickers = close_df.columns.tolist()

    normed_signals = _prepare_signals(
        alpha_signals, universe_df=universe_df, max_wt=max_wt,
        signals_are_preprocessed=signals_are_preprocessed,
    )

    ret_df = returns_df.reindex(index=dates, columns=tickers)
    fr_data = {}
    for aid, norm in normed_signals.items():
        lagged = norm.shift(1)
        ab = lagged.abs().sum(axis=1).replace(0, np.nan)
        n = lagged.div(ab, axis=0)
        fr_data[aid] = (n * ret_df).sum(axis=1)

    fr_df = pd.DataFrame(fr_data, index=dates)
    rolling_mean = fr_df.rolling(window=lookback, min_periods=60).mean()
    rolling_std  = fr_df.rolling(window=lookback, min_periods=60).std()
    rolling_sr   = (rolling_mean / rolling_std.replace(0, np.nan)).fillna(-1e9)

    # For each row, top_n columns get 1/N weight, rest 0
    rank = rolling_sr.rank(axis=1, ascending=False, method="first")
    weights = (rank <= top_n).astype(float)
    wsum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(wsum, axis=0).fillna(0)

    combined = pd.DataFrame(0.0, index=dates, columns=tickers)
    for aid, norm in normed_signals.items():
        w = weights_norm[aid].values if aid in weights_norm.columns else np.zeros(len(dates))
        combined = combined.add(norm.mul(pd.Series(w, index=dates), axis=0))

    return combined
