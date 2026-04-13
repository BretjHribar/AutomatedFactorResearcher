"""
Round 11 FINAL: Need 3 more saves (have 7 new: #79-#85).

Key discoveries from Round 10:
  - ts_sum(returns, 12) is a POWER component (4/5 saves used it)
  - ts_corr(volume, trades_count, 12) was strong (SR=6.35, saved as #81)
  - cum_returns + different partners all passed
  - ArgMin(low) and close_pos_change were weak

Components in new saves:
  #79: ts_zscore(park_vol_60) + volume
  #80: ts_sum(returns,12) + neg park_vol_60
  #81: ts_corr(vol,trades) + taker_buy_vol
  #82: ts_sum(returns,12) + neg hist_vol_120
  #83: ts_sum(returns,12) + taker_buy_vol
  #84: ts_sum(returns,12) + neg hist_vol_10
  #85: ts_sum(returns,12) + adv20

Unused combos that should be different enough:
  - ts_sum(returns,12) + quote_volume
  - ts_sum(returns,12) + trades_count
  - ts_sum(returns,12) + neg park_vol_20
  - ts_corr(vol,trades) + neg park_vol_10 
  - ts_corr(vol,trades) + adv20
  - ts_zscore(park_vol_60) + taker_buy_vol
  - neg park_vol_60 zscore + trades_count
  - ts_sum(returns,12) + dollars_traded
  - ts_sum(returns,12) + ts_corr(vol,trades)
  - ts_corr(vol,trades) + neg hist_vol_60
"""
import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


CANDIDATES = [
    # ═══ 1: cum_returns_12 + quote_volume rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_sum(returns, 12), 36)),rank(sma(ts_rank(s_log_1p(quote_volume), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "cum_returns_12 + quote_volume rank + VWAP"
    ),

    # ═══ 2: cum_returns_12 + trades_count rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_sum(returns, 12), 36)),rank(sma(ts_rank(s_log_1p(trades_count), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "cum_returns_12 + trades_count rank + VWAP"
    ),

    # ═══ 3: cum_returns_12 + neg park_vol_20 rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_sum(returns, 12), 36)),rank(negative(sma(ts_rank(parkinson_volatility_20, 12), 36)))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "cum_returns_12 + neg park_vol_20 rank + VWAP"
    ),

    # ═══ 4: ts_corr(vol,trades) + neg park_vol_10 + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_corr(s_log_1p(volume), s_log_1p(trades_count), 12), 36)),rank(negative(sma(ts_rank(parkinson_volatility_10, 12), 18)))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "vol-trade corr + neg park_vol_10 rank + VWAP"
    ),

    # ═══ 5: ts_corr(vol,trades) + adv20 rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_corr(s_log_1p(volume), s_log_1p(trades_count), 12), 36)),rank(sma(ts_rank(s_log_1p(adv20), 36), 72))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "vol-trade corr + adv20 rank + VWAP"
    ),

    # ═══ 6: neg park_vol_60 zscore + trades_count rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(negative(sma(ts_zscore(parkinson_volatility_60, 72), 36))),rank(sma(ts_rank(s_log_1p(trades_count), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "neg park_vol_60 zscore + trades_count rank + VWAP"
    ),

    # ═══ 7: cum_returns_12 + dollars_traded rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_sum(returns, 12), 36)),rank(sma(ts_rank(s_log_1p(dollars_traded), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "cum_returns_12 + dollars_traded rank + VWAP"
    ),

    # ═══ 8: ts_corr(vol,trades) + neg hist_vol_60 rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_corr(s_log_1p(volume), s_log_1p(trades_count), 12), 36)),rank(negative(sma(ts_rank(historical_volatility_60, 12), 36)))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "vol-trade corr + neg hist_vol_60 rank + VWAP"
    ),

    # ═══ 9: cum_returns_12 + neg hist_vol_60 rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(sma(ts_sum(returns, 12), 36)),rank(negative(sma(ts_rank(historical_volatility_60, 12), 36)))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "cum_returns_12 + neg hist_vol_60 rank + VWAP"
    ),

    # ═══ 10: neg park_vol_60 zscore + taker_buy_vol rank + VWAP ═══
    (
        "df_min(df_max(add(add(rank(negative(sma(ts_zscore(parkinson_volatility_60, 72), 36))),rank(sma(ts_rank(s_log_1p(taker_buy_volume), 12), 36))),rank(negative(sma(vwap_deviation, 12)))), -1.5), 1.5)",
        "neg park_vol_60 zscore + taker_buy_vol rank + VWAP"
    ),
]


def fast_compute_ic(alpha_raw, returns_df, universe_df=None, sample_every=288):
    from scipy import stats as sp_stats
    signal = alpha_raw.copy()
    if universe_df is not None:
        uni_mask = universe_df.reindex(index=signal.index, columns=signal.columns).fillna(False)
        signal = signal.where(uni_mask, np.nan)
    signal_lagged = signal.shift(1)
    indices = signal_lagged.index[1::sample_every]
    ics = []
    for dt in indices:
        if dt not in returns_df.index: continue
        a = signal_lagged.loc[dt]; r = returns_df.loc[dt]
        valid = a.notna() & r.notna() & np.isfinite(a) & np.isfinite(r)
        a_v, r_v = a[valid], r[valid]
        if len(a_v) < 10 or a_v.std() < 1e-15: ics.append(np.nan); continue
        ic, _ = sp_stats.spearmanr(a_v, r_v)
        ics.append(ic)
    return pd.Series(ics, index=indices[:len(ics)])


def build_diversity_cache(conn, matrices, universe_df, max_wt):
    import eval_alpha_5m as ev
    rows = conn.execute("SELECT id, expression FROM alphas WHERE archived=0 AND universe=?",
                        (ev.UNIVERSE,)).fetchall()
    cache = {}
    for alpha_id, expr in rows:
        try:
            raw = ev.evaluate_expression(expr, matrices)
            if raw is None: continue
            processed = ev.process_signal(raw, universe_df=universe_df, max_wt=max_wt)
            cache[alpha_id] = (expr, processed)
        except Exception:
            continue
    return cache


def check_diversity_cached(new_alpha_raw, diversity_cache, corr_cutoff, universe_df, max_wt):
    import eval_alpha_5m as ev
    new_processed = ev.process_signal(new_alpha_raw, universe_df=universe_df, max_wt=max_wt)
    for alpha_id, (expr, existing_df) in diversity_cache.items():
        common_idx = new_processed.index.intersection(existing_df.index)
        common_cols = new_processed.columns.intersection(existing_df.columns)
        if len(common_idx) < 50 or len(common_cols) < 5: continue
        a = new_processed.loc[common_idx, common_cols].values.flatten()
        b = existing_df.loc[common_idx, common_cols].values.flatten()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 100: continue
        corr = np.corrcoef(a[mask], b[mask])[0, 1]
        if abs(corr) > corr_cutoff:
            print(f"  REJECTED: Signal corr={corr:.3f} with alpha #{alpha_id} (cutoff={corr_cutoff})")
            print(f"            Existing: {expr[:80]}")
            return False
    return True


def eval_full_fast(expression, conn, universe_df):
    import eval_alpha_5m as ev
    n_trials = ev.get_num_trials(conn) + 1
    is_m = ev.eval_single(expression, split="train", fees_bps=0)
    if not is_m["success"]: return {"success": False, "error": is_m["error"]}
    ic_series = fast_compute_ic(is_m["alpha_raw"], is_m["returns_pct"], universe_df=universe_df, sample_every=288)
    ic_clean = ic_series.dropna()
    ic_mean = ic_clean.mean() if len(ic_clean) > 0 else 0
    ic_std = ic_clean.std() if len(ic_clean) > 1 else 1
    icir = ic_mean / ic_std if ic_std > 0 else 0
    stability = {}
    for start, end, name in ev.SUBPERIODS:
        sub = ev.eval_single(expression, split=name.lower(), fees_bps=0)
        stability[name] = sub["sharpe"] if sub["success"] else 0
    dsr = ev.deflated_sharpe_ratio(is_m["sharpe"], n_trials, is_m["n_bars"])
    pnl = is_m["pnl_vec"]
    pnl_kurtosis = float(pd.Series(pnl).kurtosis()) if len(pnl) > 10 else 0
    pnl_skew = float(pd.Series(pnl).skew()) if len(pnl) > 10 else 0
    rolling_sr = pd.Series(pnl).rolling(2 * ev.BARS_PER_DAY).apply(lambda x: x.mean() / x.std() if x.std() > 0 else 0).dropna()
    rolling_sr_std = float(rolling_sr.std()) if len(rolling_sr) > 10 else 999
    from stat_tests_5m import fama_macbeth_gate
    fm_pass, fm_tstat, fm_pvalue = fama_macbeth_gate(is_m["alpha_raw"], is_m["returns_pct"], universe_df, threshold=ev.MIN_FM_TSTAT)
    return {
        "success": True, "is_sharpe": is_m["sharpe"], "is_fitness": is_m["fitness"],
        "turnover": is_m["turnover"], "max_drawdown": is_m["max_drawdown"],
        "returns_ann": is_m["returns_ann"], "n_bars": is_m["n_bars"],
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "stability_h1": stability.get("H1", 0), "stability_h2": stability.get("H2", 0),
        "deflated_sharpe": dsr, "n_trials": n_trials, "pnl_vec": is_m["pnl_vec"],
        "pnl_kurtosis": pnl_kurtosis, "pnl_skew": pnl_skew, "rolling_sr_std": rolling_sr_std,
        "fm_tstat": fm_tstat, "fm_pvalue": fm_pvalue, "_alpha_raw": is_m["alpha_raw"],
    }


def main():
    import eval_alpha_5m as ev
    os.makedirs(os.path.dirname(ev.DB_PATH) or ".", exist_ok=True)
    conn = ev.get_conn(); ev.ensure_tables(conn)
    matrices, universe_df = ev.load_data("train")

    print("Building diversity cache...")
    t0 = time.time()
    diversity_cache = build_diversity_cache(conn, matrices, universe_df, ev.MAX_WEIGHT)
    print(f"  Cached {len(diversity_cache)} alphas in {time.time()-t0:.1f}s")

    passed = 0; failed = 0; results = []

    for i, (expr, reasoning) in enumerate(CANDIDATES, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(CANDIDATES)}] {reasoning}")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            result = eval_full_fast(expr, conn, universe_df)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            ev.log_trial(conn, expr, 0, 0, saved=False); failed += 1
            results.append((i, reasoning, "ERROR", str(e)[:80])); continue
        if not result["success"]:
            print(f"  ERROR: {result['error']}")
            ev.log_trial(conn, expr, 0, 0, saved=False); failed += 1
            results.append((i, reasoning, "ERROR", result['error'][:80])); continue
        both_pos = result['stability_h1'] > 0 and result['stability_h2'] > 0
        min_sub = min(result['stability_h1'], result['stability_h2'])
        gates = [
            (result['ic_mean'] >= ev.MIN_IC_MEAN, f"IC>={ev.MIN_IC_MEAN}: {result['ic_mean']:+.5f}"),
            (result['is_sharpe'] >= ev.MIN_IS_SHARPE, f"SR>={ev.MIN_IS_SHARPE}: {result['is_sharpe']:+.3f}"),
            (both_pos, f"H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f}"),
            (min_sub >= ev.MIN_SUB_SHARPE, f"MinSub>={ev.MIN_SUB_SHARPE}: {min_sub:+.2f}"),
            (result['rolling_sr_std'] <= ev.MAX_ROLLING_SR_STD, f"RollSR<={ev.MAX_ROLLING_SR_STD}: {result['rolling_sr_std']:.4f}"),
            (result['pnl_skew'] >= ev.MIN_PNL_SKEW, f"Skew>={ev.MIN_PNL_SKEW}: {result['pnl_skew']:+.3f}"),
        ]
        all_pass = all(g[0] for g in gates)
        elapsed = time.time() - t0
        print(f"\n  IC={result['ic_mean']:+.5f} ICIR={result['icir']:.4f} SR={result['is_sharpe']:+.3f} DD={result['max_drawdown']:.3f}")
        print(f"  Skew={result['pnl_skew']:+.3f} Kurt={result['pnl_kurtosis']:.1f} RollSR={result['rolling_sr_std']:.4f}")
        print(f"  FM={result['fm_tstat']:+.3f} H1={result['stability_h1']:+.2f} H2={result['stability_h2']:+.2f} ({elapsed:.0f}s)")
        for p, d in gates: print(f"  [{'PASS' if p else 'FAIL'}] {d}")
        print(f"  ALL PASS: {'YES' if all_pass else 'NO'}")
        ev.log_trial(conn, expr, result['is_sharpe'], result['ic_mean'], saved=False)
        if all_pass:
            is_diverse = check_diversity_cached(
                result['_alpha_raw'], diversity_cache, ev.CORR_CUTOFF, universe_df, ev.MAX_WEIGHT)
            if is_diverse:
                existing = conn.execute("SELECT id FROM alphas WHERE expression=? AND archived=0", (expr,)).fetchone()
                if existing:
                    print(f"  >>> REJECTED (duplicate) <<<"); failed += 1
                    results.append((i, reasoning, "REJECTED", "duplicate")); continue
                c = conn.cursor()
                c.execute("""INSERT INTO alphas (expression, name, category, asset_class, interval, universe, source, notes)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                          (expr, expr[:80], '', 'crypto', ev.INTERVAL, ev.UNIVERSE, 'agent1_research', reasoning))
                alpha_id = c.lastrowid
                c.execute("""INSERT INTO evaluations (alpha_id, universe, sharpe_is, sharpe_train, return_ann,
                             max_drawdown, turnover, fitness, ic_mean, ic_ir, psr,
                             train_start, train_end, n_bars, evaluated_at)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                          (alpha_id, ev.UNIVERSE, result['is_sharpe'], result['is_sharpe'],
                           result.get('returns_ann', 0), result['max_drawdown'], result['turnover'],
                           result['is_fitness'], result['ic_mean'], result['icir'],
                           result['deflated_sharpe'], ev.TRAIN_START, ev.TRAIN_END, result.get('n_bars', 0)))
                conn.commit()
                new_processed = ev.process_signal(result['_alpha_raw'], universe_df=universe_df, max_wt=ev.MAX_WEIGHT)
                diversity_cache[alpha_id] = (expr, new_processed)
                passed += 1
                results.append((i, reasoning, "SAVED", f"#{alpha_id} SR={result['is_sharpe']:+.3f}"))
                print(f"  SAVED as alpha #{alpha_id}")
                print(f"  >>> SAVED <<<")
            else:
                results.append((i, reasoning, "REJECTED", "diversity")); failed += 1
                print(f"  >>> REJECTED (diversity) <<<")
        else:
            results.append((i, reasoning, "FAILED", "; ".join(d for p,d in gates if not p))); failed += 1

    conn.close()
    print(f"\n{'='*70}")
    print(f"  ROUND 11 FINAL: {passed} saved, {failed} failed out of {len(CANDIDATES)}")
    print(f"{'='*70}")
    for idx, r, s, d in results:
        print(f"    {idx:2d} [{s:8s}] {r[:60]}")
        if s not in ("SAVED",): print(f"              {d[:120]}")

if __name__ == "__main__":
    main()
