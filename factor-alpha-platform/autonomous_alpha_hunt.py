"""
Autonomous overnight alpha hunt — hypothesis-driven (per workflow rules):
  - SR >= 5, fit >= 6, corr < 0.7 (strict, no relaxation)
  - Each candidate is a distinct economic mechanism
  - PASS results auto-saved to DB
  - Combiner test re-run after each batch save

Designed to run unattended for ~8 hours.
"""
from __future__ import annotations
import sys, sqlite3, time, traceback
from pathlib import Path
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine

# === GATES (DO NOT CHANGE) ===
MIN_SR  = 5.0
MIN_FIT = 6.0
MAX_CORR = 0.70

UNIV_NAME   = "MCAP_100M_500M"
MAX_W       = 0.02
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-01-01"
BOOK        = 500_000

COMMISSION_PER_SHARE = 0.0045
PER_ORDER_MIN        = 0.35
SEC_FEE_PER_DOLLAR   = 27.80e-6
SELL_FRACTION        = 0.50
IMPACT_BPS           = 0.5
BORROW_BPS_ANNUAL    = 50

DATA_DIR = ROOT / "data/fmp_cache/matrices"
UNIV_DIR = ROOT / "data/fmp_cache/universes"
DB       = ROOT / "data/alpha_results.db"
TAG_PREFIX = "[SMALLCAP_D0_v3]"


def proc_signal(s, uni, cls):
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]; s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    ab = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(ab, axis=0).clip(-MAX_W, MAX_W).fillna(0)


def realistic_cost(w, close, book):
    pos = w * book; trd = pos.diff().abs()
    safe = close.where(close > 0); shares = trd / safe
    pn_comm = (shares * COMMISSION_PER_SHARE).clip(lower=0)
    has = trd > 1.0
    pn_comm = pn_comm.where(~has, np.maximum(pn_comm, PER_ORDER_MIN)).where(has, 0)
    cost = (pn_comm.sum(axis=1)
            + (trd * SEC_FEE_PER_DOLLAR * SELL_FRACTION).sum(axis=1)
            + (trd * IMPACT_BPS / 1e4).sum(axis=1)
            + (-pos.clip(upper=0)).sum(axis=1) * (BORROW_BPS_ANNUAL / 1e4) / 252.0
           ) / book
    return cost


def load_data():
    uni = pd.read_parquet(UNIV_DIR / f"{UNIV_NAME}.parquet").astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0)/len(uni); valid = sorted(cov[cov>0.5].index.tolist())
    uni = uni[valid]; dates = uni.index; tickers = uni.columns.tolist()
    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"): continue
        try: df = pd.read_parquet(fp)
        except: continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce"); df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc: mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(index=dates, columns=tickers)
    close = mats["close"]; ret = close.pct_change(fill_method=None)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    train_mask = (dates >= TRAIN_START) & (dates < TRAIN_END)
    return uni, dates, tickers, mats, close, ret, cls, train_mask


def load_existing_normed(engine, uni, cls, train_mask):
    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a
         WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
         ORDER BY a.id""").fetchall()
    existing = {}
    for aid, expr in rows:
        try:
            sig = proc_signal(engine.evaluate(expr), uni, cls)
            existing[aid] = sig.loc[train_mask]
        except Exception:
            pass
    return existing, len(rows)


def evaluate(name, expr, engine, uni, cls, mats, close, ret, train_mask, existing_train):
    try:
        raw = engine.evaluate(expr)
    except Exception as e:
        return None, f"parse fail: {e}"
    sig = proc_signal(raw, uni, cls)
    sig_tr = sig.loc[train_mask]
    nx_tr = ret.shift(-1).loc[train_mask]
    close_tr = close.loc[train_mask]
    g = (sig_tr * nx_tr).sum(axis=1).fillna(0)
    cost = realistic_cost(sig_tr, close_tr, BOOK)
    n = g - cost
    ann = np.sqrt(252)
    if g.std() == 0:
        return None, "zero std"
    sr = float(g.mean()/g.std()*ann)
    ret_a = float(g.mean()*252)
    to = float(sig_tr.diff().abs().sum(axis=1).mean() / 2)
    fit = sr * np.sqrt(abs(ret_a) / max(to, 0.125)) if sr != 0 else 0.0
    eq = (1 + n).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())
    lagged = sig_tr.shift(1)
    ic_d = lagged.corrwith(nx_tr, axis=1)
    ic = float(ic_d.mean()); ic_ir = float(ic_d.mean()/ic_d.std()*np.sqrt(252)) if ic_d.std() > 0 else 0.0
    n_bars = int(train_mask.sum())

    # corr to existing on TRAIN
    a_flat = sig_tr.values.flatten()
    a_msk = ~np.isnan(a_flat) & (a_flat != 0)
    max_corr = 0.0
    for aid, e_sig in existing_train.items():
        b_flat = e_sig.values.flatten()
        mm = a_msk & ~np.isnan(b_flat) & (b_flat != 0)
        if mm.sum() < 1000: continue
        c = abs(float(np.corrcoef(a_flat[mm], b_flat[mm])[0,1]))
        if c > max_corr: max_corr = c

    try:
        from scipy.stats import norm, skew, kurtosis
        sk = float(skew(g.dropna())); kt = float(kurtosis(g.dropna(), fisher=True))
        sr_d = sr / np.sqrt(252)
        std_sr = np.sqrt((1 - sk*sr_d + (kt)/4.0 * sr_d**2) / max(n_bars-1, 1))
        psr = float(norm.cdf(sr_d / std_sr)) if std_sr > 0 else 0.5
    except Exception:
        psr = 0.0

    is_pass = (sr >= MIN_SR and fit >= MIN_FIT and max_corr < MAX_CORR)
    return dict(name=name, expr=expr, sr=sr, fit=fit, ret_a=ret_a, mdd=mdd,
                ic=ic, ic_ir=ic_ir, n_bars=n_bars, psr=psr, to=to,
                max_corr=max_corr, is_pass=is_pass), None


def save_pass(r, sig_train_for_addition):
    """Save a strict-pass alpha to DB. Returns alpha_id or None if duplicate."""
    conn = sqlite3.connect(DB); c = conn.cursor()
    existing = c.execute("SELECT id FROM alphas WHERE expression=? AND archived=0", (r['expr'],)).fetchone()
    if existing:
        conn.close()
        return None
    notes = (f"{TAG_PREFIX} {r['name']}. AUTONOMOUS HUNT pass.  "
             f"SR={r['sr']:.2f} fit={r['fit']:.2f} ret_ann={r['ret_a']*100:+.1f}% "
             f"mdd={r['mdd']*100:+.1f}% IC={r['ic']:.4f} IC_IR={r['ic_ir']:+.2f} "
             f"TO={r['to']*100:.1f}%/d max|corr|={r['max_corr']:.3f}")
    c.execute("INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes) VALUES (?,?,?,?,?,?,?)",
              (r['expr'], r['expr'][:80], "composite", "equities", "daily", "autonomous", notes))
    aid = c.lastrowid
    c.execute("INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann, max_drawdown, "
              "turnover, fitness, ic_mean, ic_ir, psr, train_start, train_end, n_bars, delay, decay, "
              "universe, max_weight, neutralization) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (aid, r['sr'], r['sr'], r['ret_a'], r['mdd'], r['to'], r['fit'],
               r['ic'], r['ic_ir'], r['psr'], "2020-01-01", "2024-01-01", r['n_bars'],
               0, 0, "MCAP_100M_500M", 0.02, "subindustry"))
    conn.commit(); conn.close()
    return aid


def main():
    # === HYPOTHESIS QUEUE — each is a distinct economic mechanism ===
    # Recipe: rank(unused_FUNDAMENTAL) × DecayExp(0.10) of negative VWAP-deviation
    # Each row: (name, expression, mechanism_note)
    QUEUE = [
        # === Round 40: signed-power, autocorr-rev, vol-peak reversal ===
        ("H501_invMC_x_SqrtCIDR",     "rank(multiply(rank(negative(market_cap)), rank(signed_power(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.5))))", "small × sqrt-CIDR (amp tails)"),
        ("H502_invMC_x_AutoCorrRev",  "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_corr(returns, ts_delay(returns, 1), 5)), 5))))", "small × decay-autocorr-rev"),
        ("H503_invMC_x_VolPeakRev",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_max(stddev(returns, 5), 21)))))", "small × neg-recent-vol-peak"),
        ("H504_invMC_x_DecaySqrtRev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(signed_power(ts_delta(close, 5), 0.5)), 5))))", "small × decay-sqrt-Rev5"),
        ("H505_invMC_x_RangePos7",    "rank(multiply(rank(negative(market_cap)), rank(true_divide(subtract(ts_max(high, 7), close), df_max(subtract(ts_max(high, 7), ts_min(low, 7)), 0.01)))))", "small × close-near-7d-low"),
        ("H506_FCFY_x_RangePos7",     "rank(multiply(rank(free_cashflow_yield), rank(true_divide(subtract(ts_max(high, 7), close), df_max(subtract(ts_max(high, 7), ts_min(low, 7)), 0.01)))))", "FCFY × close-near-7d-low"),
        ("H507_invMC_x_DecayRangePos7", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(true_divide(subtract(ts_max(high, 7), close), df_max(subtract(ts_max(high, 7), ts_min(low, 7)), 0.01)), 3))))", "small × decay-RangePos7"),
        ("H508_invMC_x_DecayCIDR_signed", "rank(multiply(rank(negative(market_cap)), rank(signed_power(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5), 1.5))))", "small × decay-CIDR-amp"),
        ("H509_invMC_x_VWAPdev_signed","rank(multiply(rank(negative(market_cap)), rank(signed_power(decay_exp(negative(true_divide(close, vwap)), 0.10), 1.5))))", "small × VWAPdev-amp"),
        ("H510_invMC_x_DecayDayHighDev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(close, high)), 5))))", "small × decay close/today-high"),
        # === Round 39: decay-smoothed price-weighted return reversal ===
        ("H491_invMC_x_DecayPriceRetRev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(multiply(close, returns)), 5))))", "small × decay-price-weighted-rev"),
        ("H492_FCFY_x_DecayPriceRetRev",  "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(multiply(close, returns)), 5))))", "FCFY × decay-price-weighted-rev"),
        ("H493_BM_x_DecayPriceRetRev",    "rank(multiply(rank(book_to_market), rank(decay_linear(negative(multiply(close, returns)), 5))))", "value × decay-price-weighted-rev"),
        ("H494_invMC_x_DollarRetRev",     "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(multiply(returns, volume)), 5))))", "small × decay-dollar-rev"),
        ("H495_FCFY_x_DollarRetRev",      "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(multiply(returns, volume)), 5))))", "FCFY × decay-dollar-rev"),
        ("H496_invDE_x_DecayPriceRetRev", "rank(multiply(rank(negative(debt_to_equity)), rank(decay_linear(negative(multiply(close, returns)), 5))))", "low D/E × decay-price-rev"),
        ("H497_invADV_x_DecayPriceRetRev","rank(multiply(rank(negative(adv20)), rank(decay_linear(negative(multiply(close, returns)), 5))))", "low ADV × decay-price-rev"),
        ("H498_EY_x_DecayPriceRetRev",    "rank(multiply(rank(earnings_yield), rank(decay_linear(negative(multiply(close, returns)), 5))))", "EY × decay-price-rev"),
        ("H499_invIntangibles_x_DecayPriceRetRev", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_linear(negative(multiply(close, returns)), 5))))", "low intangibles × decay-price-rev"),
        ("H500_OM_x_DecayPriceRetRev",    "rank(multiply(rank(operating_margin), rank(decay_linear(negative(multiply(close, returns)), 5))))", "OM × decay-price-rev"),
        # === Round 38: truly novel directionals — VWAP slope, dev-from-recent-mean, abs-rev ===
        ("H481_invMC_x_VWAPmom",     "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(vwap, 5)))))", "small × neg-VWAP-momentum-5d"),
        ("H482_FCFY_x_VWAPmom",      "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_delta(vwap, 5)))))", "FCFY × neg-VWAP-momentum"),
        ("H483_invMC_x_DecayVWAPmom","rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_delta(vwap, 5)), 5))))", "small × decay-VWAP-mom"),
        ("H484_invMC_x_DevRecentRetMean", "rank(multiply(rank(negative(market_cap)), rank(negative(subtract(returns, sma(returns, 5))))))", "small × dev-from-recent-return-mean"),
        ("H485_invMC_x_AbsRetRev",   "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(abs(returns)), 5))))", "small × decay-abs-return-rev"),
        ("H486_FCFY_x_AbsRetRev",    "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(abs(returns)), 5))))", "FCFY × decay-abs-return-rev"),
        ("H487_invMC_x_RankPriceRet","rank(multiply(rank(negative(market_cap)), rank(negative(multiply(returns, close)))))", "small × neg-price-weighted-return"),
        ("H488_invMC_x_VolWeightedRet","rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(multiply(returns, true_divide(volume, sma(volume, 20)))), 5))))", "small × decay-vol-weighted-rev"),
        ("H489_FCFY_x_VolWeightedRet","rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(multiply(returns, true_divide(volume, sma(volume, 20)))), 5))))", "FCFY × decay-vol-weighted-rev"),
        ("H490_BM_x_VolWeightedRet", "rank(multiply(rank(book_to_market), rank(decay_linear(negative(multiply(returns, true_divide(volume, sma(volume, 20)))), 5))))", "value × decay-vol-weighted-rev"),
        # === Round 37: 4-ways with vol-of-vol regime (corr-break power) + variance directionals ===
        ("H471_4way_VWAP_CIDR_volofvol", "rank(multiply(multiply(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))), rank(negative(ts_zscore(stddev(returns, 5), 60)))))", "small × VWAP × CIDR × calm-vol regime"),
        ("H472_4way_FCFY_VWAP_CIDR_vov", "rank(multiply(multiply(multiply(rank(free_cashflow_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))), rank(negative(ts_zscore(stddev(returns, 5), 60)))))", "FCFY × VWAP × CIDR × calm-vol"),
        ("H473_invMC_x_DecayPctDev5",  "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(subtract(true_divide(close, sma(close, 5)), 1.0)), 5))))", "small × decay 5d-pct-dev"),
        ("H474_invMC_x_DecayRetSq",    "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(square(returns)), 5))))", "small × decay-variance reversal"),
        ("H475_invMC_x_RelVolScarcity","rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(volume, ts_max(volume, 60))))))", "small × relative-volume scarcity"),
        ("H476_invMC_x_CloseVsHigh",   "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(subtract(close, high), df_max(subtract(high, low), 0.01))))))", "small × distance-from-day's-high"),
        ("H477_FCFY_x_CloseVsHigh",    "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(subtract(close, high), df_max(subtract(high, low), 0.01))))))", "FCFY × distance-from-day's-high"),
        ("H478_invMC_x_DecayCloseVsHigh", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(subtract(close, high), df_max(subtract(high, low), 0.01))), 5))))", "small × decay-distance-from-day's-high"),
        ("H479_FCFY_x_DecayCloseVsHigh", "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(subtract(close, high), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × decay-distance-from-day's-high"),
        ("H480_BM_x_DecayCloseVsHigh", "rank(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(subtract(close, high), df_max(subtract(high, low), 0.01))), 5))))", "value × decay-distance-from-day's-high"),
        # === Round 36: VWAP-fast-slow + vol-adjusted directionals ===
        ("H461_invMC_x_VWAPfastSlow",  "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(sma(vwap, 5), sma(vwap, 21))))))", "small × VWAP 5d/21d ratio reversal"),
        ("H462_FCFY_x_VWAPfastSlow",   "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(sma(vwap, 5), sma(vwap, 21))))))", "FCFY × VWAP 5d/21d"),
        ("H463_invMC_x_DecayVWAP21",   "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(close, sma(vwap, 21))), 5))))", "small × decay close-vs-21d-vwap"),
        ("H464_FCFY_x_DecayVWAP21",    "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(close, sma(vwap, 21))), 5))))", "FCFY × decay close-vs-21d-vwap"),
        ("H465_invMC_x_VolAdjRet",     "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(returns, df_max(stddev(returns, 21), 0.005))))))", "small × vol-adjusted reversal"),
        ("H466_FCFY_x_VolAdjRet",      "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(returns, df_max(stddev(returns, 21), 0.005))))))", "FCFY × vol-adjusted reversal"),
        ("H467_invMC_x_DecayVolAdjRet","rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(returns, df_max(stddev(returns, 21), 0.005))), 5))))", "small × decay-vol-adj-rev"),
        ("H468_FCFY_x_DecayVolAdjRet", "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(returns, df_max(stddev(returns, 21), 0.005))), 5))))", "FCFY × decay-vol-adj-rev"),
        ("H469_invMC_x_OpenVWAPdev",   "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(open, vwap)), 5))))", "small × decay open-vs-vwap-dev"),
        ("H470_BM_x_DecayVWAP21",      "rank(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(close, sma(vwap, 21))), 5))))", "value × decay close-vs-21d-vwap"),
        # === Round 35: triples — fund × TWO different reversal directionals ===
        ("H451_invMC_CIDR_ZsVWAP",   "rank(multiply(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "small × decay-CIDR × decay-zsVWAP"),
        ("H452_FCFY_CIDR_ZsVWAP",    "rank(multiply(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "FCFY × decay-CIDR × decay-zsVWAP"),
        ("H453_BM_CIDR_ZsVWAP",      "rank(multiply(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "value × decay-CIDR × decay-zsVWAP"),
        ("H454_invMC_LogDiff_ZsVWAP","rank(multiply(multiply(rank(negative(market_cap)), rank(decay_linear(negative(log_diff(close)), 5))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "small × decay-LogDiff × decay-zsVWAP"),
        ("H455_FCFY_LogDiff_ZsVWAP", "rank(multiply(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(log_diff(close)), 5))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "FCFY × decay-LogDiff × decay-zsVWAP"),
        ("H456_invMC_VolConf_ZsVWAP","rank(multiply(multiply(rank(negative(market_cap)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "small × vol-conf × decay-zsVWAP"),
        ("H457_FCFY_VolConf_ZsVWAP", "rank(multiply(multiply(rank(free_cashflow_yield), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "FCFY × vol-conf × decay-zsVWAP"),
        ("H458_invMC_RP14_ZsVWAP",   "rank(multiply(multiply(rank(negative(market_cap)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "small × RP14 × decay-zsVWAP"),
        ("H459_FCFY_RP14_ZsVWAP",    "rank(multiply(multiply(rank(free_cashflow_yield), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "FCFY × RP14 × decay-zsVWAP"),
        ("H460_invMC_VWAP_CIDR",     "rank(multiply(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × DecayExpVWAP × decay-CIDR"),
        # === Round 34: funds × decay-smoothed zscore-VWAP-deviation (new directional) ===
        ("H441_FCFY_x_DecayZsVWAP",  "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "FCFY × decay-zscoreVWAPdev"),
        ("H442_invMC_x_DecayZsVWAP", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "small × decay-zscoreVWAPdev"),
        ("H443_BM_x_DecayZsVWAP",    "rank(multiply(rank(book_to_market), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "value × decay-zscoreVWAPdev"),
        ("H444_EY_x_DecayZsVWAP",    "rank(multiply(rank(earnings_yield), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "EY × decay-zscoreVWAPdev"),
        ("H445_invDE_x_DecayZsVWAP", "rank(multiply(rank(negative(debt_to_equity)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "low D/E × decay-zscoreVWAPdev"),
        ("H446_OM_x_DecayZsVWAP",    "rank(multiply(rank(operating_margin), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "OM × decay-zscoreVWAPdev"),
        ("H447_NIgrowth_x_DecayZsVWAP", "rank(multiply(rank(ts_delta(net_income, 252)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "NIgrowth × decay-zscoreVWAPdev"),
        ("H448_invIntangibles_x_DecayZsVWAP", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "low intangibles × decay-zscoreVWAPdev"),
        ("H449_invADV_x_DecayZsVWAP","rank(multiply(rank(negative(adv20)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "low ADV × decay-zscoreVWAPdev"),
        ("H450_invSGA_x_DecayZsVWAP","rank(multiply(rank(negative(sga_to_revenue)), rank(decay_linear(negative(ts_zscore(true_divide(close, vwap), 21)), 5))))", "low SGA × decay-zscoreVWAPdev"),
        # === Round 33: ts_rank directionals (different operator from ts_zscore) ===
        ("H431_invMC_x_TSrankClose21", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_rank(close, 21)))))", "small × ts-rank close 21d"),
        ("H432_FCFY_x_TSrankClose21",  "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_rank(close, 21)))))", "FCFY × ts-rank close 21d"),
        ("H433_invMC_x_TSrankRet21",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_rank(returns, 21)))))", "small × ts-rank returns 21d"),
        ("H434_BM_x_TSrankClose21",    "rank(multiply(rank(book_to_market), rank(negative(ts_rank(close, 21)))))", "value × ts-rank close 21d"),
        ("H435_invMC_x_DecayTSrank",   "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_rank(close, 21)), 5))))", "small × decay ts-rank close"),
        ("H436_invMC_x_TSrankCIDR21",  "rank(multiply(rank(negative(market_cap)), rank(ts_rank(true_divide(subtract(ts_max(high, 21), close), df_max(subtract(ts_max(high, 21), ts_min(low, 21)), 0.01)), 21))))", "small × ts-rank distance from 21d high"),
        ("H437_invMC_x_zscoreVWAPdev", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_zscore(true_divide(close, vwap), 21)))))", "small × z-score VWAP-dev 21d"),
        ("H438_FCFY_x_zscoreVWAPdev",  "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_zscore(true_divide(close, vwap), 21)))))", "FCFY × z-score VWAP-dev 21d"),
        ("H439_invMC_x_StdReversal",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(stddev(returns, 5), 5)))))", "small × stddev-reversal"),
        ("H440_BM_x_TSrankRet21",      "rank(multiply(rank(book_to_market), rank(negative(ts_rank(returns, 21)))))", "value × ts-rank returns 21d"),
        # === Round 32: truly new directionals (argmin recency, pasteurize, % deviation) ===
        ("H421_invMC_x_DecayMinRet5", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_min(returns, 5)), 5))))", "small × decay-min-return-5d"),
        ("H422_FCFY_x_DecayMinRet5",  "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(ts_min(returns, 5)), 5))))", "FCFY × decay-min-return-5d"),
        ("H423_invMC_x_PctDev7",      "rank(multiply(rank(negative(market_cap)), rank(negative(subtract(true_divide(close, sma(close, 7)), 1.0)))))", "small × % deviation from 7d sma"),
        ("H424_FCFY_x_PctDev7",       "rank(multiply(rank(free_cashflow_yield), rank(negative(subtract(true_divide(close, sma(close, 7)), 1.0)))))", "FCFY × % deviation from 7d sma"),
        ("H425_invMC_x_ArgMinClose14","rank(multiply(rank(negative(market_cap)), rank(negative(ts_argmin(close, 14)))))", "small × close-min-was-recent (14d)"),
        ("H426_invMC_x_DistMin21",    "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_min(close, 21))))))", "small × distance above 21d close-min"),
        ("H427_invMC_x_PasteurizeRev5","rank(multiply(rank(negative(market_cap)), rank(pasteurize(negative(ts_delta(close, 5))))))", "small × pasteurize-Rev5"),
        ("H428_FCFY_x_PasteurizeRev5","rank(multiply(rank(free_cashflow_yield), rank(pasteurize(negative(ts_delta(close, 5))))))", "FCFY × pasteurize-Rev5"),
        ("H429_invMC_x_VWAPmaxDev",   "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_max(vwap, 21))))))", "small × close-vs-21d-vwap-max"),
        ("H430_invMC_x_DecayPctDev7", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(subtract(true_divide(close, sma(close, 7)), 1.0)), 5))))", "small × decay-PctDev7"),
        # === Round 31: alternate vol-related legs in fund × vol × decay-CIDR ===
        ("H411_invMC_volZS_CIDR",     "rank(multiply(multiply(rank(negative(market_cap)), rank(ts_zscore(volume, 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × vol-zscore × decay-CIDR"),
        ("H412_invMC_dollarSurge_CIDR", "rank(multiply(multiply(rank(negative(market_cap)), rank(true_divide(dollars_traded, sma(dollars_traded, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × dollar-surge × decay-CIDR"),
        ("H413_invMC_dollarZS_CIDR",  "rank(multiply(multiply(rank(negative(market_cap)), rank(ts_zscore(dollars_traded, 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × dollarvol-zscore × decay-CIDR"),
        ("H414_FCFY_volZS_CIDR",      "rank(multiply(multiply(rank(free_cashflow_yield), rank(ts_zscore(volume, 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × vol-zscore × decay-CIDR"),
        ("H415_BM_volZS_CIDR",        "rank(multiply(multiply(rank(book_to_market), rank(ts_zscore(volume, 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "value × vol-zscore × decay-CIDR"),
        ("H416_invMC_VolAccel_CIDR",  "rank(multiply(multiply(rank(negative(market_cap)), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × vol-accel × decay-CIDR"),
        ("H417_FCFY_VolAccel_CIDR",   "rank(multiply(multiply(rank(free_cashflow_yield), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × vol-accel × decay-CIDR"),
        ("H418_invMC_volTurnover_CIDR", "rank(multiply(multiply(rank(negative(market_cap)), rank(ts_zscore(true_divide(volume, shares_out), 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × free-float-turnover-zs × decay-CIDR"),
        ("H419_invMC_dollarSurge60_CIDR", "rank(multiply(multiply(rank(negative(market_cap)), rank(true_divide(dollars_traded, sma(dollars_traded, 60)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × 60d dollar-surge × decay-CIDR"),
        ("H420_FCFY_dollarSurge_CIDR", "rank(multiply(multiply(rank(free_cashflow_yield), rank(true_divide(dollars_traded, sma(dollars_traded, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × dollar-surge × decay-CIDR"),
        # === Round 30: more fund × vol-conf × decay-CIDR triples ===
        ("H401_BM_VolConf_CIDR",      "rank(multiply(multiply(rank(book_to_market), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "value × vol-surge × decay-CIDR"),
        ("H402_invDE_VolConf_CIDR",   "rank(multiply(multiply(rank(negative(debt_to_equity)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low D/E × vol-surge × decay-CIDR"),
        ("H403_invIntangibles_VolConf_CIDR", "rank(multiply(multiply(rank(negative(intangibles_to_assets)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low intangibles × vol-surge × decay-CIDR"),
        ("H404_invADV_VolConf_CIDR",  "rank(multiply(multiply(rank(negative(adv20)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low ADV × vol-surge × decay-CIDR"),
        ("H405_NIgrowth_VolConf_CIDR","rank(multiply(multiply(rank(ts_delta(net_income, 252)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "NIgrowth × vol-surge × decay-CIDR"),
        ("H406_ROEgrowth_VolConf_CIDR", "rank(multiply(multiply(rank(ts_delta(roe, 252)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "ROEgrowth × vol-surge × decay-CIDR"),
        ("H407_OM_VolConf_CIDR",      "rank(multiply(multiply(rank(operating_margin), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "OM × vol-surge × decay-CIDR"),
        ("H408_invSGA_VolConf_CIDR",  "rank(multiply(multiply(rank(negative(sga_to_revenue)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low SGA × vol-surge × decay-CIDR"),
        ("H409_DivYield_VolConf_CIDR","rank(multiply(multiply(rank(dividend_yield), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "DivYield × vol-surge × decay-CIDR"),
        ("H410_invSharesGr_VolConf_CIDR", "rank(multiply(multiply(rank(negative(ts_delta(shares_out, 252))), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low dilution × vol-surge × decay-CIDR"),
        # === Round 29: triple composites mixing funds with vol-conf or regime + directionals ===
        ("H391_invMC_VolConf_CIDR",   "rank(multiply(multiply(rank(negative(market_cap)), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × vol-surge × decay-CIDR"),
        ("H392_FCFY_VolConf_CIDR",    "rank(multiply(multiply(rank(free_cashflow_yield), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × vol-surge × decay-CIDR"),
        ("H393_invMC_FCFY_invDE_CIDR","rank(multiply(multiply(multiply(rank(negative(market_cap)), rank(free_cashflow_yield)), rank(negative(debt_to_equity))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "4-way: small + FCFY + safe + CIDR"),
        ("H394_invMC_FCFY_invInt_CIDR","rank(multiply(multiply(multiply(rank(negative(market_cap)), rank(free_cashflow_yield)), rank(negative(intangibles_to_assets))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "4-way: small + FCFY + tangible + CIDR"),
        ("H395_invMC_VolofVol_CIDR",  "rank(multiply(multiply(rank(negative(market_cap)), rank(ts_zscore(stddev(returns, 5), 60))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × vol-of-vol × decay-CIDR"),
        ("H396_FCFY_zsRet21_CIDR",    "rank(multiply(multiply(rank(free_cashflow_yield), rank(negative(ts_zscore(returns, 21)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × zsRet21 × decay-CIDR"),
        ("H397_invMC_zsRet21_CIDR",   "rank(multiply(multiply(rank(negative(market_cap)), rank(negative(ts_zscore(returns, 21)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × zsRet21 × decay-CIDR"),
        ("H398_BM_zsRet21_CIDR",      "rank(multiply(multiply(rank(book_to_market), rank(negative(ts_zscore(returns, 21)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "value × zsRet21 × decay-CIDR"),
        ("H399_invMC_pvCorr_CIDR",    "rank(multiply(multiply(rank(negative(market_cap)), rank(negative(ts_corr(close, volume, 21)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × neg-pv-corr × decay-CIDR"),
        ("H400_EY_VolConf_CIDR",      "rank(multiply(multiply(rank(earnings_yield), rank(true_divide(volume, sma(volume, 20)))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "EY × vol-surge × decay-CIDR"),
        # === Round 28: less-explored reversal directionals ===
        ("H381_invMC_x_DecayBB7",   "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(subtract(close, sma(close, 7)), df_max(stddev(close, 7), 0.001))), 5))))", "small × decay-Bollinger7"),
        ("H382_FCFY_x_DecayBB7",    "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(subtract(close, sma(close, 7)), df_max(stddev(close, 7), 0.001))), 5))))", "FCFY × decay-Bollinger7"),
        ("H383_invMC_x_AboveLow7",  "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_min(low, 7))))))", "small × close-near-7d-low"),
        ("H384_FCFY_x_AboveLow7",   "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, ts_min(low, 7))))))", "FCFY × close-near-7d-low"),
        ("H385_invMC_x_DecayDeviation14", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(close, sma(close, 14))), 5))))", "small × decay 14d-mean-dev"),
        ("H386_BM_x_DecayDeviation14", "rank(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(close, sma(close, 14))), 5))))", "value × decay 14d-mean-dev"),
        ("H387_invMC_x_DecayHighRev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_delta(high, 5)), 5))))", "small × decay-high-reversal"),
        ("H388_invMC_x_DecayLowRev",  "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_delta(low, 5)), 5))))", "small × decay-low-reversal"),
        ("H389_invMC_x_DecayOpenRev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_delta(open, 5)), 5))))", "small × decay-open-reversal"),
        ("H390_FCFY_x_DecayOpenRev",  "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(ts_delta(open, 5)), 5))))", "FCFY × decay-open-reversal"),
        # === Round 27: more funds × decay-log-diff (after R26's 3 saves) ===
        ("H371_ROIC_x_DecayLogDiff",  "rank(multiply(rank(roic), rank(decay_linear(negative(log_diff(close)), 5))))", "ROIC × decay-log-diff"),
        ("H372_ROCE_x_DecayLogDiff",  "rank(multiply(rank(roce), rank(decay_linear(negative(log_diff(close)), 5))))", "ROCE × decay-log-diff"),
        ("H373_EBITDAm_x_DecayLogDiff", "rank(multiply(rank(ebitda_margin), rank(decay_linear(negative(log_diff(close)), 5))))", "EBITDA m × decay-log-diff"),
        ("H374_GM_x_DecayLogDiff",    "rank(multiply(rank(gross_margin), rank(decay_linear(negative(log_diff(close)), 5))))", "GM × decay-log-diff"),
        ("H375_NM_x_DecayLogDiff",    "rank(multiply(rank(net_margin), rank(decay_linear(negative(log_diff(close)), 5))))", "NM × decay-log-diff"),
        ("H376_invInterestExp_x_DecayLogDiff", "rank(multiply(rank(negative(interest_expense)), rank(decay_linear(negative(log_diff(close)), 5))))", "low intExp × decay-log-diff"),
        ("H377_invSharesGr_x_DecayLogDiff", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(decay_linear(negative(log_diff(close)), 5))))", "low dilution × decay-log-diff"),
        ("H378_invNetDebt_x_DecayLogDiff", "rank(multiply(rank(negative(net_debt)), rank(decay_linear(negative(log_diff(close)), 5))))", "low net debt × decay-log-diff"),
        ("H379_DivYield_x_DecayLogDiff", "rank(multiply(rank(dividend_yield), rank(decay_linear(negative(log_diff(close)), 5))))", "DivYield × decay-log-diff"),
        ("H380_invCommonStockIss_x_DecayLogDiff", "rank(multiply(rank(negative(common_stock_issuance)), rank(decay_linear(negative(log_diff(close)), 5))))", "buyback × decay-log-diff"),
        # === Round 26: funds × decay-smoothed log-diff (drops TO from 50% to ~25%) ===
        ("H361_invMC_x_DecayLogDiff",  "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(log_diff(close)), 5))))", "small × decay-log-diff"),
        ("H362_FCFY_x_DecayLogDiff",   "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(log_diff(close)), 5))))", "FCFY × decay-log-diff"),
        ("H363_BM_x_DecayLogDiff",     "rank(multiply(rank(book_to_market), rank(decay_linear(negative(log_diff(close)), 5))))", "value × decay-log-diff"),
        ("H364_EY_x_DecayLogDiff",     "rank(multiply(rank(earnings_yield), rank(decay_linear(negative(log_diff(close)), 5))))", "EY × decay-log-diff"),
        ("H365_invDE_x_DecayLogDiff",  "rank(multiply(rank(negative(debt_to_equity)), rank(decay_linear(negative(log_diff(close)), 5))))", "low D/E × decay-log-diff"),
        ("H366_invADV20_x_DecayLogDiff", "rank(multiply(rank(negative(adv20)), rank(decay_linear(negative(log_diff(close)), 5))))", "low ADV × decay-log-diff"),
        ("H367_OM_x_DecayLogDiff",     "rank(multiply(rank(operating_margin), rank(decay_linear(negative(log_diff(close)), 5))))", "OM × decay-log-diff"),
        ("H368_invIntangibles_x_DecayLogDiff", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_linear(negative(log_diff(close)), 5))))", "low intangibles × decay-log-diff"),
        ("H369_SGAToRev_x_DecayLogDiff", "rank(multiply(rank(negative(sga_to_revenue)), rank(decay_linear(negative(log_diff(close)), 5))))", "low SGA × decay-log-diff"),
        ("H370_NIgrowth_x_DecayLogDiff", "rank(multiply(rank(ts_delta(net_income, 252)), rank(decay_linear(negative(log_diff(close)), 5))))", "NIgrowth × decay-log-diff"),
        # === Round 25: delta-of-returns + log-scaled directionals ===
        ("H351_invMC_x_NegDeltaRet5", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(returns, 5)))))", "small × negative delta-returns 5d"),
        ("H352_FCFY_x_NegDeltaRet5",  "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_delta(returns, 5)))))", "FCFY × negative delta-returns"),
        ("H353_invMC_x_LogScaledRev", "rank(multiply(rank(negative(market_cap)), rank(s_log_1p(negative(ts_delta(close, 5))))))", "small × log-scaled Rev5"),
        ("H354_invMC_x_AvgDiff10Neg", "rank(multiply(rank(negative(market_cap)), rank(ts_av_diff(negative(returns), 10))))", "small × avg-diff of neg returns 10d"),
        ("H355_invMC_x_MaxNeg5",      "rank(multiply(rank(negative(market_cap)), rank(ts_max(negative(returns), 5))))", "small × biggest 5d loss (lottery)"),
        ("H356_invMC_x_LogDiff",      "rank(multiply(rank(negative(market_cap)), rank(negative(log_diff(close)))))", "small × neg log-diff close"),
        ("H357_FCFY_x_LogDiff",       "rank(multiply(rank(free_cashflow_yield), rank(negative(log_diff(close)))))", "FCFY × neg log-diff close"),
        ("H358_invMC_x_Dev7smaDecay", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(subtract(close, sma(close, 7))), 5))))", "small × decay-Dev7sma"),
        ("H359_FCFY_x_NegDeltaRet10", "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_delta(returns, 10)))))", "FCFY × neg delta-returns 10d"),
        ("H360_invMC_x_QuantC30",     "rank(multiply(rank(negative(market_cap)), rank(negative(ts_quantile(close, 30)))))", "small × low-percentile in 30d close"),
        # === Round 24: decay-smoothed DistVwap5 + different SMA windows ===
        ("H341_BM_x_DecayDistVwap5",   "rank(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(close, sma(vwap, 5))), 5))))", "value × decay-DistVwap5"),
        ("H342_invIntangibles_x_DecayDistVwap5", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_linear(negative(true_divide(close, sma(vwap, 5))), 5))))", "low intangibles × decay-DistVwap5"),
        ("H343_FCFY_x_DecayDistVwap5", "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(close, sma(vwap, 5))), 5))))", "FCFY × decay-DistVwap5"),
        ("H344_invMC_x_Dev30SMA",      "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, sma(close, 30))))))", "small × deviation from 30d sma"),
        ("H345_BM_x_Dev30SMA",         "rank(multiply(rank(book_to_market), rank(negative(true_divide(close, sma(close, 30))))))", "value × deviation from 30d sma"),
        ("H346_FCFY_x_Dev30SMA",       "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, sma(close, 30))))))", "FCFY × deviation from 30d sma"),
        ("H347_invDE_x_Dev30SMA",      "rank(multiply(rank(negative(debt_to_equity)), rank(negative(true_divide(close, sma(close, 30))))))", "low D/E × deviation from 30d sma"),
        ("H348_invIntangibles_x_Dev30SMA", "rank(multiply(rank(negative(intangibles_to_assets)), rank(negative(true_divide(close, sma(close, 30))))))", "low intangibles × deviation from 30d sma"),
        ("H349_invMC_x_Dev45SMA",      "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, sma(close, 45))))))", "small × deviation from 45d sma"),
        ("H350_FCFY_x_Dev45SMA",       "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, sma(close, 45))))))", "FCFY × deviation from 45d sma"),
        # === Round 23: fund × truly new directional structures ===
        ("H331_invMC_x_RegSlope",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_regression(close, ts_step(1), 21, 0, 0)))))", "small × negative regression slope (downtrend)"),
        ("H332_invMC_x_DistAbove21Low", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_min(low, 21))))))", "small × close-near-21d-low"),
        ("H333_invMC_x_Dev21SMA",  "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, sma(close, 21))))))", "small × deviation from 21d sma"),
        ("H334_FCFY_x_Dev21SMA",   "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, sma(close, 21))))))", "FCFY × deviation from 21d sma"),
        ("H335_invMC_x_NegMin5",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_min(returns, 5)))))", "small × biggest 5d losers (loser-buy)"),
        ("H336_invMC_x_DistVwap5", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, sma(vwap, 5))))))", "small × deviation from 5d vwap-mean"),
        ("H337_FCFY_x_DistVwap5",  "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, sma(vwap, 5))))))", "FCFY × deviation from 5d vwap-mean"),
        ("H338_invDE_x_DistVwap5", "rank(multiply(rank(negative(debt_to_equity)), rank(negative(true_divide(close, sma(vwap, 5))))))", "low D/E × deviation from 5d vwap-mean"),
        ("H339_BM_x_DistVwap5",    "rank(multiply(rank(book_to_market), rank(negative(true_divide(close, sma(vwap, 5))))))", "value × deviation from 5d vwap-mean"),
        ("H340_invIntangibles_x_DistVwap5", "rank(multiply(rank(negative(intangibles_to_assets)), rank(negative(true_divide(close, sma(vwap, 5))))))", "low intangibles × deviation from 5d vwap-mean"),
        # === Round 22: funds × decay-EXP-CIDR (different smoothing kernel from decay-linear) ===
        ("H321_invMC_x_DecayExpCIDR",  "rank(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "small × decay-exp-CIDR"),
        ("H322_FCFY_x_DecayExpCIDR",   "rank(multiply(rank(free_cashflow_yield), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "FCFY × decay-exp-CIDR"),
        ("H323_invDE_x_DecayExpCIDR",  "rank(multiply(rank(negative(debt_to_equity)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "low D/E × decay-exp-CIDR"),
        ("H324_BM_x_DecayExpCIDR",     "rank(multiply(rank(book_to_market), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "value × decay-exp-CIDR"),
        ("H325_invIntangibles_x_DecayExpCIDR", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "low intangibles × decay-exp-CIDR"),
        ("H326_NIgrowth_x_DecayExpCIDR", "rank(multiply(rank(ts_delta(net_income, 252)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "NIgrowth × decay-exp-CIDR"),
        ("H327_ROEgrowth_x_DecayExpCIDR", "rank(multiply(rank(ts_delta(roe, 252)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "ROEgrowth × decay-exp-CIDR"),
        ("H328_SGAToRev_x_DecayExpCIDR", "rank(multiply(rank(negative(sga_to_revenue)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "low SGA × decay-exp-CIDR"),
        ("H329_invInterestExp_x_DecayExpCIDR", "rank(multiply(rank(negative(interest_expense)), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "low intExp × decay-exp-CIDR"),
        ("H330_invSharesGr_x_DecayExpCIDR", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(decay_exp(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 0.10))))", "low dilution × decay-exp-CIDR"),
        # === Round 21: more funds × decay-CIDR ===
        ("H311_ROIC_x_DecayCIDR",  "rank(multiply(rank(roic), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "ROIC × decay-CIDR"),
        ("H312_NM_x_DecayCIDR",    "rank(multiply(rank(net_margin), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "NM × decay-CIDR"),
        ("H313_GM_x_DecayCIDR",    "rank(multiply(rank(gross_margin), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "GM × decay-CIDR"),
        ("H314_invInterestExp_x_DecayCIDR", "rank(multiply(rank(negative(interest_expense)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low intExp × decay-CIDR"),
        ("H315_RevGrowth_x_DecayCIDR", "rank(multiply(rank(ts_delta(revenue, 252)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "RevGrowth × decay-CIDR"),
        ("H316_NIgrowth_x_DecayCIDR",  "rank(multiply(rank(ts_delta(net_income, 252)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "NIgrowth × decay-CIDR"),
        ("H317_invCommonStockIss_x_DecayCIDR", "rank(multiply(rank(negative(common_stock_issuance)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "buyback × decay-CIDR"),
        ("H318_EBITDAm_x_DecayCIDR", "rank(multiply(rank(ebitda_margin), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "EBITDA m × decay-CIDR"),
        ("H319_invSBC_x_DecayCIDR",  "rank(multiply(rank(negative(sbc_to_revenue)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low SBC × decay-CIDR"),
        ("H320_DivYield_x_DecayCIDR", "rank(multiply(rank(dividend_yield), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "DivYield × decay-CIDR"),
        # === Round 20: funds × decay-smoothed CIDR (lower TO than raw CIDR) ===
        # R19 insight: raw CIDR has SR>5 corr<0.7 across many funds, but 50%+ TO kills fit.
        # decay-linear(CIDR, 5) drops TO without losing direction.
        ("H301_invMC_x_DecayCIDR",  "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "small × decay-CIDR"),
        ("H302_FCFY_x_DecayCIDR",   "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × decay-CIDR"),
        ("H303_invDE_x_DecayCIDR",  "rank(multiply(rank(negative(debt_to_equity)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low D/E × decay-CIDR"),
        ("H304_BM_x_DecayCIDR",     "rank(multiply(rank(book_to_market), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "value × decay-CIDR"),
        ("H305_invADV20_x_DecayCIDR", "rank(multiply(rank(negative(adv20)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low ADV × decay-CIDR"),
        ("H306_invSharesGr_x_DecayCIDR", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low dilution × decay-CIDR"),
        ("H307_invIntangibles_x_DecayCIDR", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low intangibles × decay-CIDR"),
        ("H308_EY_x_DecayCIDR",     "rank(multiply(rank(earnings_yield), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "EY × decay-CIDR"),
        ("H309_OM_x_DecayCIDR",     "rank(multiply(rank(operating_margin), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "OM × decay-CIDR"),
        ("H310_SGAToRev_x_DecayCIDR", "rank(multiply(rank(negative(sga_to_revenue)), rank(decay_linear(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))), 5))))", "low SGA × decay-CIDR"),
        # === Round 19: more funds × close-in-day-range directional ===
        ("H291_FCFY_x_CIDR",      "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "FCFY × close-in-day-range"),
        ("H292_invDE_x_CIDR",     "rank(multiply(rank(negative(debt_to_equity)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "low D/E × CIDR"),
        ("H293_BM_x_CIDR",        "rank(multiply(rank(book_to_market), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "value × CIDR"),
        ("H294_invADV20_x_CIDR",  "rank(multiply(rank(negative(adv20)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "low ADV × CIDR"),
        ("H295_invSharesGr_x_CIDR", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "low dilution × CIDR"),
        ("H296_invIntangibles_x_CIDR", "rank(multiply(rank(negative(intangibles_to_assets)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "low intangibles × CIDR"),
        ("H297_EY_x_CIDR",        "rank(multiply(rank(earnings_yield), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "EY × CIDR"),
        ("H298_OM_x_CIDR",        "rank(multiply(rank(operating_margin), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "OM × CIDR"),
        ("H299_SGAToRev_x_CIDR",  "rank(multiply(rank(negative(sga_to_revenue)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "low SGA × CIDR"),
        ("H300_ROEgrowth_x_CIDR", "rank(multiply(rank(ts_delta(roe, 252)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "ROEgrowth × CIDR"),
        # === Round 18: fund × truly different observable structures ===
        ("H281_invMC_x_TWvolxRP14", "rank(multiply(rank(negative(market_cap)), rank(trade_when(subtract(true_divide(volume, sma(volume, 20)), 1.5), true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)), 0.0))))", "small × vol-conditional RP14"),
        ("H282_invMC_x_QuantC60",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_quantile(close, 60)))))", "small × low percentile in 60d close distribution"),
        ("H283_invMC_x_RecentMax", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_argmax(close, 60)))))", "small × close-max-was-recent (fade)"),
        ("H284_invMC_x_LogRetRev", "rank(multiply(rank(negative(market_cap)), rank(s_log_1p(negative(returns)))))", "small × log-scale 1d reversal"),
        ("H285_invMC_x_SqrtRev5",  "rank(multiply(rank(negative(market_cap)), rank(signed_power(negative(ts_delta(close, 5)), 0.5))))", "small × sqrt-amplified Rev5"),
        ("H286_invMC_x_OpenRev5",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(open, 5)))))", "small × OPEN reversal (not close)"),
        ("H287_invMC_x_DistCloseMax14", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_max(close, 14))))))", "small × close-vs-14d-close-max"),
        ("H288_invMC_x_LowRev5",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(low, 5)))))", "small × LOW reversal"),
        ("H289_invMC_x_HighRev5",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(high, 5)))))", "small × HIGH reversal"),
        ("H290_invMC_x_CloseInDayRange", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(subtract(close, low), df_max(subtract(high, low), 0.01))))))", "small × close-in-day-range (high portion = fade)"),
        # === Round 17: fund × truly NEW directionals (ts_corr, hump-filter, lottery, kurtosis) ===
        ("H271_invMC_x_negPVcorr",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_corr(close, vwap, 21)))))", "small × negative close-vwap correlation"),
        ("H272_invMC_x_LotMAX5",    "rank(multiply(rank(negative(market_cap)), rank(negative(ts_max(returns, 5)))))", "small × lottery aversion (max-5d-return)"),
        ("H273_FCFY_x_LotMAX5",     "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_max(returns, 5)))))", "FCFY × lottery aversion"),
        ("H274_invMC_x_HumpRev5",   "rank(multiply(rank(negative(market_cap)), rank(hump(negative(ts_delta(close, 5)), 0.005))))", "small × hump-filtered Rev5 (selectivity)"),
        ("H275_invMC_x_negSkew",    "rank(multiply(rank(negative(market_cap)), rank(negative(ts_skewness(returns, 21)))))", "small × negative-skew preference"),
        ("H276_invMC_x_invKurt",    "rank(multiply(rank(negative(market_cap)), rank(negative(ts_kurtosis(returns, 21)))))", "small × low-kurtosis (calm tail)"),
        ("H277_FCFY_x_HumpRev5",    "rank(multiply(rank(free_cashflow_yield), rank(hump(negative(ts_delta(close, 5)), 0.005))))", "FCFY × hump-filtered Rev5"),
        ("H278_invMC_x_AvgDiff10",  "rank(multiply(rank(negative(market_cap)), rank(negative(ts_av_diff(close, 10)))))", "small × negative avg-diff(close, 10)"),
        ("H279_FCFY_x_negPVcorr",   "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_corr(close, vwap, 21)))))", "FCFY × negative close-vwap corr"),
        ("H280_invMC_x_WilliamsR21","rank(multiply(rank(negative(market_cap)), rank(true_divide(subtract(ts_max(high, 21), close), df_max(subtract(ts_max(high, 21), ts_min(low, 21)), 0.01)))))", "small × close-near-21d-low"),
        # === Round 16: triples WITHOUT FCFY (avoid saturation with #98) ===
        ("H261_ROEgr_invMC_RP14",  "rank(multiply(multiply(rank(ts_delta(roe, 252)), rank(negative(market_cap))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROEgrowth + small + RP14"),
        ("H262_NIgr_invMC_RP14",   "rank(multiply(multiply(rank(ts_delta(net_income, 252)), rank(negative(market_cap))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "NIgrowth + small + RP14"),
        ("H263_invMC_invADV_RP14", "rank(multiply(multiply(rank(negative(market_cap)), rank(negative(adv20))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "small + illiquid + RP14"),
        ("H264_invMC_BM_RP14",     "rank(multiply(multiply(rank(negative(market_cap)), rank(book_to_market)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "small + value + RP14"),
        ("H265_invMC_invDE_RP14",  "rank(multiply(multiply(rank(negative(market_cap)), rank(negative(debt_to_equity))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "small + safety + RP14"),
        ("H266_EY_invDE_RP14",     "rank(multiply(multiply(rank(earnings_yield), rank(negative(debt_to_equity))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "EY + safety + RP14"),
        ("H267_BM_invDE_RP14",     "rank(multiply(multiply(rank(book_to_market), rank(negative(debt_to_equity))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "value + safety + RP14"),
        ("H268_invADV_invDE_RP14", "rank(multiply(multiply(rank(negative(adv20)), rank(negative(debt_to_equity))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "illiquid + safety + RP14"),
        ("H269_invIntangibles_invMC_RP14", "rank(multiply(multiply(rank(negative(intangibles_to_assets)), rank(negative(market_cap))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "intangibles + small + RP14"),
        ("H270_invIntangibles_invDE_RP14", "rank(multiply(multiply(rank(negative(intangibles_to_assets)), rank(negative(debt_to_equity))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "intangibles + safety + RP14"),
        # === Round 15: more F1 + FCFY + RP14 triples (each F1 = different fundamental mechanism) ===
        ("H251_invMC_FCFY_RP14",   "rank(multiply(multiply(rank(negative(market_cap)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "small + FCFY + RP14"),
        ("H252_invDE_FCFY_RP14",   "rank(multiply(multiply(rank(negative(debt_to_equity)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low D/E + FCFY + RP14"),
        ("H253_invADV20_FCFY_RP14","rank(multiply(multiply(rank(negative(adv20)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low ADV + FCFY + RP14"),
        ("H254_BM_FCFY_RP14",      "rank(multiply(multiply(rank(book_to_market), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "value + FCFY + RP14"),
        ("H255_OM_FCFY_RP14",      "rank(multiply(multiply(rank(operating_margin), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "OM + FCFY + RP14"),
        ("H256_invSharesGr_FCFY_RP14", "rank(multiply(multiply(rank(negative(ts_delta(shares_out, 252))), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low dilution + FCFY + RP14"),
        ("H257_invInterestExp_FCFY_RP14", "rank(multiply(multiply(rank(negative(interest_expense)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low intExp + FCFY + RP14"),
        ("H258_ROAgrowth_FCFY_RP14", "rank(multiply(multiply(rank(ts_delta(roa, 252)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROAgrowth + FCFY + RP14"),
        ("H259_ROICgrowth_FCFY_RP14", "rank(multiply(multiply(rank(ts_delta(roic, 252)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROICgrowth + FCFY + RP14"),
        ("H260_invSGA_FCFY_RP14",  "rank(multiply(multiply(rank(negative(sga_to_revenue)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low SGA + FCFY + RP14"),
        # === Round 14: decay-smoothed novel directionals + more triples ===
        ("H241_invMC_x_DecayBodyRev", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 5))))", "small × smoothed sold-into-close"),
        ("H242_FCFY_x_DecayBodyRev",  "rank(multiply(rank(free_cashflow_yield), rank(decay_linear(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))), 5))))", "FCFY × smoothed sold-into-close"),
        ("H243_NIgr_x_FCFY_x_RP14",  "rank(multiply(multiply(rank(ts_delta(net_income, 252)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "NIgrowth + FCFY + RP14"),
        ("H244_ROEgr_x_FCFY_x_RP14", "rank(multiply(multiply(rank(ts_delta(roe, 252)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROEgrowth + FCFY + RP14"),
        ("H245_invMC_x_DecayZsRet5", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_zscore(returns, 5)), 5))))", "small × smoothed short z-score returns"),
        ("H246_FCFY_x_LowVolRegime", "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_zscore(stddev(returns, 5), 60)))))", "FCFY × calm-vol regime"),
        ("H247_invMC_x_LowVolRegime","rank(multiply(rank(negative(market_cap)), rank(negative(ts_zscore(stddev(returns, 5), 60)))))", "small × calm-vol regime"),
        ("H248_invMC_x_DecaySMA5dev","rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(subtract(close, sma(close, 5))), 3))))", "small × smoothed dev-from-5d-SMA"),
        # Cross-sectional zscore of fund (different normalization than rank)
        ("H249_zscoreCSinvMC_x_VWAP", "rank(multiply(zscore_cs(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "CS-zscore-small × VWAPdev"),
        ("H250_zscoreCSFCFY_x_VWAP", "rank(multiply(zscore_cs(free_cashflow_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "CS-zscore-FCFY × VWAPdev"),
        # === Round 13: novel directionals (intraday/range-exp) + triples of near-misses ===
        ("H231_invMC_x_BodyRev",  "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))))))", "small × neg body-to-range (sold-into-close)"),
        ("H232_FCFY_x_BodyRev",   "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(subtract(close, open), df_max(subtract(high, low), 0.01))))))", "FCFY × neg body-to-range"),
        ("H233_invMC_x_RangeExp", "rank(multiply(rank(negative(market_cap)), rank(true_divide(subtract(high, low), sma(subtract(high, low), 20)))))", "small × range expansion (vol regime)"),
        ("H234_invMC_x_zsRet5",   "rank(multiply(rank(negative(market_cap)), rank(negative(ts_zscore(returns, 5)))))", "small × short z-score returns"),
        ("H235_FCFY_x_zsRet5",    "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_zscore(returns, 5)))))", "FCFY × short z-score returns"),
        # Triples of two near-pass funds + RP14
        ("H236_RevGr_x_FCFY_x_RP14", "rank(multiply(multiply(rank(ts_delta(revenue, 252)), rank(free_cashflow_yield)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "RevGrowth + FCFY + RP14"),
        ("H237_NIgr_x_invMC_x_RP14", "rank(multiply(multiply(rank(ts_delta(net_income, 252)), rank(negative(market_cap))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "NIgrowth + small + RP14"),
        # Open-VWAP deviation as directional (different from close-VWAP)
        ("H238_invMC_x_OpenVWAP", "rank(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(open, vwap)), 0.10))))", "small × open-vs-vwap (intraday open-bid)"),
        # Acceleration of close (delta of delta)
        ("H239_invMC_x_AccelRev", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_delta(ts_delta(close, 5), 5)))))", "small × negative price acceleration"),
        # ts_corr of returns vs lagged returns (autocorrelation regime) × fund
        ("H240_invMC_x_AutoCorr", "rank(multiply(rank(negative(market_cap)), rank(negative(ts_corr(returns, ts_delay(returns, 1), 21)))))", "small × negative-autocorr regime"),
        # === Round 12: more unused fundamentals × the proven RP14 directional ===
        ("H221_ROIC_x_RP14",     "rank(multiply(rank(roic), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROIC × RP14"),
        ("H222_OM_x_RP14",       "rank(multiply(rank(operating_margin), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "OM × RP14"),
        ("H223_EBITDAm_x_RP14",  "rank(multiply(rank(ebitda_margin), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "EBITDA m × RP14"),
        ("H224_DivYield_x_RP14", "rank(multiply(rank(dividend_yield), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "DivYield × RP14"),
        ("H225_ROEgrowth_x_RP14","rank(multiply(rank(ts_delta(roe, 252)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "ROE growth × RP14"),
        ("H226_NIgrowth_x_RP14", "rank(multiply(rank(ts_delta(net_income, 252)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "NI growth × RP14"),
        ("H227_RevGrowth_x_RP14","rank(multiply(rank(ts_delta(revenue, 252)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "Rev growth × RP14"),
        ("H228_invIntangibles_x_RP14", "rank(multiply(rank(negative(intangibles_to_assets)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low intangibles × RP14"),
        ("H229_SGAtoRev_x_RP14", "rank(multiply(rank(negative(sga_to_revenue)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low SG&A × RP14"),
        ("H230_invSBC_x_RP14",   "rank(multiply(rank(negative(sbc_to_revenue)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low SBC × RP14"),
        # === Round 11: fundamentals × distance-from-anchor directionals (63d, 252d) ===
        # RangePos14 worked. Try other anchored-distance directionals.
        ("H211_invMC_x_DistFrom63", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_max(high, 63))))))", "small × distance from 63d high"),
        ("H212_invMC_x_DistFrom252", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_max(high, 252))))))", "small × distance from 252d high"),
        ("H213_FCFY_x_DistFrom63", "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(close, ts_max(high, 63))))))", "FCFY × distance from 63d high"),
        ("H214_invDE_x_DistFrom63", "rank(multiply(rank(negative(debt_to_equity)), rank(negative(true_divide(close, ts_max(high, 63))))))", "low D/E × distance from 63d high"),
        ("H215_BM_x_DistFrom63",   "rank(multiply(rank(book_to_market), rank(negative(true_divide(close, ts_max(high, 63))))))", "value × distance from 63d high"),
        # Distance from low (instead of high) — bullish anchor signal
        ("H216_invMC_x_AboveLow63", "rank(multiply(rank(negative(market_cap)), rank(true_divide(close, ts_min(low, 63)))))", "small × close above 63d low"),
        # mean-reversion to a slow MA (60d sma)
        ("H217_invMC_x_devSMA60", "rank(multiply(rank(negative(market_cap)), rank(negative(subtract(close, sma(close, 60))))))", "small × deviation from 60d sma"),
        ("H218_FCFY_x_devSMA60", "rank(multiply(rank(free_cashflow_yield), rank(negative(subtract(close, sma(close, 60))))))", "FCFY × deviation from 60d sma"),
        # range expansion-aware fund signal
        ("H219_invMC_x_invRangeExp", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(subtract(high, low), sma(subtract(high, low), 20))))))", "small × calm-day reversal"),
        # fund × pure decay smoothed reversal of close
        ("H220_invMC_x_decayRev10", "rank(multiply(rank(negative(market_cap)), rank(decay_linear(negative(ts_delta(close, 10)), 5))))", "small × smoothed 10d reversal"),
        # === Round 10: more fundamentals with the breakthrough RangePos14 directional ===
        # H193 (invMC × RangePos14) passed strict — close-near-14d-low is fresh directional.
        # Try other fundamentals with same directional.
        ("H201_FCFY_x_RP14",     "rank(multiply(rank(free_cashflow_yield), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "FCFY × close-near-14d-low"),
        ("H202_invDE_x_RP14",    "rank(multiply(rank(negative(debt_to_equity)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low D/E × close-near-14d-low"),
        ("H203_invADV20_x_RP14", "rank(multiply(rank(negative(adv20)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low ADV × close-near-14d-low"),
        ("H204_BM_x_RP14",       "rank(multiply(rank(book_to_market), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "value × close-near-14d-low"),
        ("H205_invSharesGr_x_RP14", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low dilution × close-near-14d-low"),
        ("H206_invNetIss_x_RP14", "rank(multiply(rank(negative(net_stock_issuance)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "buyback × close-near-14d-low"),
        ("H207_EY_x_RP14",       "rank(multiply(rank(earnings_yield), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "EY × close-near-14d-low"),
        ("H208_NM_x_RP14",       "rank(multiply(rank(net_margin), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "NM × close-near-14d-low"),
        ("H209_invSTD_x_RP14",   "rank(multiply(rank(negative(short_term_debt)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "low ST debt × close-near-14d-low"),
        ("H210_GM_x_RP14",       "rank(multiply(rank(gross_margin), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "GM × close-near-14d-low"),
        # === Round 9: fund × NEW directional (not VWAPdev) + triple of near-passers ===
        ("H191_invMC_x_BB14",     "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(subtract(close, sma(close, 14)), df_max(stddev(close, 14), 0.001))))))", "small × BB14 reversal"),
        ("H192_FCFY_x_BB14",      "rank(multiply(rank(free_cashflow_yield), rank(negative(true_divide(subtract(close, sma(close, 14)), df_max(stddev(close, 14), 0.001))))))", "FCFY × BB14"),
        ("H193_invMC_x_RangePos14", "rank(multiply(rank(negative(market_cap)), rank(true_divide(subtract(ts_max(high, 14), close), df_max(subtract(ts_max(high, 14), ts_min(low, 14)), 0.01)))))", "small × close-near-low(14)"),
        ("H194_invDE_x_BB14",     "rank(multiply(rank(negative(debt_to_equity)), rank(negative(true_divide(subtract(close, sma(close, 14)), df_max(stddev(close, 14), 0.001))))))", "low D/E × BB14"),
        # Triples of 2 near-pass fundamentals (each on its own had corr <0.7 but SR 4-5)
        ("H195_invSTD_x_invDE_x_VWAP", "rank(multiply(multiply(rank(negative(short_term_debt)), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "invSTD + invDE + VWAPdev (safety squared)"),
        ("H196_invSharesGr_x_invDE_x_VWAP", "rank(multiply(multiply(rank(negative(ts_delta(shares_out, 252))), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low dilution + low D/E + VWAPdev"),
        ("H197_invNetIss_x_invDE_x_VWAP", "rank(multiply(multiply(rank(negative(net_stock_issuance)), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "buyback + low D/E + VWAPdev"),
        ("H198_NetMargin_x_invDE_x_VWAP", "rank(multiply(multiply(rank(net_margin), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "NM + low D/E + VWAPdev"),
        # zsRet directional with FCFY (not yet tried — H85 used VWAPdev only)
        ("H199_FCFY_zsRet60",     "rank(multiply(rank(free_cashflow_yield), rank(negative(ts_zscore(returns, 60)))))", "FCFY × zsRet 60d"),
        # Conditional: trade only when fundamental is in TOP quartile
        ("H200_TWfund_x_VWAP",    "rank(trade_when(subtract(rank(free_cashflow_yield), 0.75), decay_exp(negative(true_divide(close, vwap)), 0.10), 0.0))", "trade VWAPdev only when FCFY top-quartile"),
        # === Round 8: ts_zscore of fundamentals (different fields) ===
        # R7 insight: ts_zscore captures "deviation from own history" cleanly. H173 ROE z-score
        # was SR 4.95 corr 0.64 — only 0.05 short. Try more fundamentals with this transform.
        ("H181_zscoreROA",      "rank(multiply(rank(ts_zscore(roa, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROA z-score × VWAPdev"),
        ("H182_zscoreOM",       "rank(multiply(rank(ts_zscore(operating_margin, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "OM z-score × VWAPdev"),
        ("H183_zscoreNM",       "rank(multiply(rank(ts_zscore(net_margin, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "NM z-score × VWAPdev"),
        ("H184_zscoreInvDE",    "rank(multiply(rank(ts_zscore(negative(debt_to_equity), 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low D/E z-score × VWAPdev"),
        ("H185_zscoreGM",       "rank(multiply(rank(ts_zscore(gross_margin, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "GM z-score × VWAPdev"),
        ("H186_zscoreAssetTurn","rank(multiply(rank(ts_zscore(asset_turnover, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "AT z-score × VWAPdev"),
        ("H187_zscoreROIC",     "rank(multiply(rank(ts_zscore(roic, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROIC z-score × VWAPdev"),
        ("H188_zscoreADV",      "rank(multiply(rank(ts_zscore(adv20, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ADV z-score × VWAPdev"),
        ("H189_zscoreFCFY",     "rank(multiply(rank(ts_zscore(free_cashflow_yield, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "FCFY z-score × VWAPdev"),
        ("H190_zscoreEY",       "rank(multiply(rank(ts_zscore(earnings_yield, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EY z-score × VWAPdev"),
        # === Round 7: ts_zscore of fundamentals + raw price + additive composites ===
        # All previous rounds used cross-sectional rank() of fundamentals.
        # ts_zscore captures DEVIATION from own history — different math, may break corr.
        ("H171_zscoreMC_x_VWAP",  "rank(multiply(rank(ts_zscore(market_cap, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "MC z-score vs own history × VWAPdev"),
        ("H172_zscoreBM_x_VWAP",  "rank(multiply(rank(ts_zscore(book_to_market, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "B/M z-score × VWAPdev"),
        ("H173_zscoreROE_x_VWAP", "rank(multiply(rank(ts_zscore(roe, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROE z-score × VWAPdev"),
        # Raw absolute price effects (penny-stock revert harder)
        ("H174_invClose_x_VWAP",  "rank(multiply(rank(negative(close)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low absolute price × VWAPdev"),
        ("H175_invSqrtPrice_x_VWAP", "rank(multiply(rank(negative(sqrt(close))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "1/sqrt price × VWAPdev"),
        # Additive composition (structurally different from multiplicative)
        ("H176_ADD_invMC_p_VWAP", "rank(add(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Additive: invMC + VWAPdev"),
        ("H177_ADD_FCFY_p_VWAP",  "rank(add(rank(free_cashflow_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Additive: FCFY + VWAPdev"),
        # Different VWAP reference (sma(vwap, 5) instead of vwap level)
        ("H178_invMC_x_VWAP5dev", "rank(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, sma(vwap, 5))), 0.10))))", "invMC × VWAP-5d-deviation"),
        # Distance from 60d high (anchoring) + fundamental
        ("H179_invMC_x_DistFrom60High", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(close, ts_max(high, 60))))))", "Small × distance from 60d high"),
        # Vol-of-vol regime × fundamental (no VWAPdev — direction comes from fund)
        ("H180_invMC_x_volofvol", "rank(multiply(rank(negative(market_cap)), rank(ts_zscore(stddev(returns, 5), 60))))", "Small × vol-of-vol regime"),
        # === Round 6 (informed by R5): YoY change of fundamentals (delta not level) + new fields ===
        # R5 insight: cashflow LEVELS correlate ~0.97 with FCFY (#98) — same direction.
        # CHANGES (delta) capture different info — slow growth signals.
        ("H161_NetMarginGrowth", "rank(multiply(rank(ts_delta(net_margin, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Net margin growth × VWAPdev"),
        ("H162_GMgrowth",        "rank(multiply(rank(ts_delta(gross_margin, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Gross margin growth × VWAPdev"),
        ("H163_AssetTurnGrowth", "rank(multiply(rank(ts_delta(asset_turnover, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Asset turnover growth × VWAPdev"),
        ("H164_ADVgrowth",       "rank(multiply(rank(ts_delta(adv20, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Liquidity (ADV) growth × VWAPdev"),
        ("H165_DaysPayables",    "rank(multiply(rank(days_payables_outstanding), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Supplier float × VWAPdev"),
        ("H166_invNetDebtGrowth","rank(multiply(rank(negative(ts_delta(net_debt, 252))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Debt-shrinking × VWAPdev"),
        ("H167_FCFperShare",     "rank(multiply(rank(fcf_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "FCF/share × VWAPdev"),
        ("H168_SalesPS",         "rank(multiply(rank(sales_ps), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Sales/share × VWAPdev"),
        ("H169_invInvTurnover",  "rank(multiply(rank(negative(inventory_turnover)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Slow inventory (opposite of H107) × VWAPdev"),
        ("H170_OperatingCycle",  "rank(multiply(rank(operating_cycle), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Operating cycle × VWAPdev"),
        # === Round 5 (informed by R4): debt structure + cashflow level + share actions ===
        # R4 insight: triples consistently fail corr. Stick to single fund × VWAPdev.
        # New mechanisms not yet tested.
        ("H151_invShortTermDebt", "rank(multiply(rank(negative(short_term_debt)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low ST debt × VWAPdev"),
        ("H152_invLongTermDebt",  "rank(multiply(rank(negative(long_term_debt)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low LT debt × VWAPdev"),
        ("H153_DebtRepayment",    "rank(multiply(rank(debt_repayment), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Active debt paydown × VWAPdev"),
        ("H154_CashflowOp",       "rank(multiply(rank(cashflow_op), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Operating CF level × VWAPdev"),
        ("H155_FreeCashflow",     "rank(multiply(rank(free_cashflow), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Free CF level × VWAPdev"),
        ("H156_EquityGrowth",     "rank(multiply(rank(ts_delta(equity, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Equity 252d growth × VWAPdev"),
        ("H157_WCgrowth",         "rank(multiply(rank(ts_delta(working_capital, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Working capital 252d growth × VWAPdev"),
        ("H158_OCFgrowth",        "rank(multiply(rank(ts_delta(cashflow_op, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Operating CF growth × VWAPdev"),
        ("H159_StockRepurchase",  "rank(multiply(rank(stock_repurchase), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Active buyback × VWAPdev"),
        ("H160_invCommonStockIss","rank(multiply(rank(negative(common_stock_issuance)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low common issuance × VWAPdev"),
        # === Round 4 (informed by R3): pair TWO orthogonal fundamentals with VWAPdev ===
        # Insight: single fund × VWAPdev hits corr ceiling. Two distinct fundamentals (each
        # with own corr-breaking power) should multiplicatively reduce composite corr.
        ("H141_invMC_x_FCFY_x_VWAP",
         "rank(multiply(multiply(rank(negative(market_cap)), rank(free_cashflow_yield)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple small + cash-cheap + VWAPdev"),
        ("H142_invMC_x_invDE_x_VWAP",
         "rank(multiply(multiply(rank(negative(market_cap)), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple small + low-leverage + VWAPdev"),
        ("H143_invMC_x_GM_x_VWAP",
         "rank(multiply(multiply(rank(negative(market_cap)), rank(gross_margin)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple small + quality + VWAPdev"),
        ("H144_BM_x_invDE_x_VWAP",
         "rank(multiply(multiply(rank(book_to_market), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple value + safety + VWAPdev"),
        ("H145_invADV_x_invDE_x_VWAP",
         "rank(multiply(multiply(rank(negative(adv20)), rank(negative(debt_to_equity))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple illiquid + safe + VWAPdev"),
        # Dupont decomposition components — never tried
        ("H146_TaxBurden",       "rank(multiply(rank(tax_burden), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Tax burden × VWAPdev"),
        ("H147_InterestBurden",  "rank(multiply(rank(interest_burden), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Interest burden × VWAPdev"),
        # Share dilution mechanism (NEW)
        ("H148_invSharesGrowth", "rank(multiply(rank(negative(ts_delta(shares_out, 252))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low share dilution × VWAPdev"),
        # Vol-of-vol (corr 0.13!) × strong directional that has natural low corr
        ("H149_volofvol_x_invMC_x_VWAP",
         "rank(multiply(multiply(rank(ts_zscore(stddev(returns, 5), 60)), rank(negative(market_cap))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Triple vol-of-vol + small + VWAPdev"),
        # Stock buybacks: net stock issuance reverse (companies buying back have anchor)
        ("H150_invNetIssuance",  "rank(multiply(rank(negative(net_stock_issuance)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low net issuance (buyback proxy) × VWAPdev"),
        # === Round 3 (informed by R1+R2): new fund fields (growth/efficiency/cycle) + triple refinements ===
        # Best R2 near-miss: H122 SGAToRev (SR 4.91 fit 6.67 corr 0.67) — try triple to lift SR.
        # New fields: revenue growth, earnings growth, dilution, capex efficiency, working capital cycle.
        ("H129_RevenueGrowth",   "rank(multiply(rank(ts_delta(revenue, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Revenue growth × VWAPdev"),
        ("H130_EarningsGrowth",  "rank(multiply(rank(ts_delta(net_income, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Earnings growth × VWAPdev"),
        ("H131_invSBCRev",       "rank(multiply(rank(negative(sbc_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low SBC dilution × VWAPdev"),
        ("H132_invRDRev",        "rank(multiply(rank(negative(rd_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low R&D intensity × VWAPdev"),
        ("H133_IncomeQuality",   "rank(multiply(rank(income_quality), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Income quality × VWAPdev"),
        ("H134_invCapexOCF",     "rank(multiply(rank(negative(capex_to_ocf)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low capex/OCF × VWAPdev"),
        ("H135_invDIO",          "rank(multiply(rank(negative(days_inventory_outstanding)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Fast inventory turn × VWAPdev"),
        ("H136_invDSO",          "rank(multiply(rank(negative(days_sales_outstanding)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Fast collections × VWAPdev"),
        ("H137_invCCC",          "rank(multiply(rank(negative(cash_conversion_cycle)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Short cash cycle × VWAPdev"),
        ("H138_FCFToFirm",       "rank(multiply(rank(fcf_to_firm), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "FCF/firm × VWAPdev"),
        # Triple refinements: SGAToRev (SR 4.91) × VWAPdev × second orthogonal axis
        ("H139_SGA_x_VWAP_x_invMC",
         "rank(multiply(multiply(rank(negative(sga_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))), rank(negative(market_cap))))", "Triple SGA + VWAPdev + small"),
        ("H140_invIntangibles_x_VWAP_x_invMC",
         "rank(multiply(multiply(rank(negative(intangibles_to_assets)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))), rank(negative(market_cap))))", "Triple invIntangibles + VWAPdev + small"),
        # === Round 2 (informed by R1): break corr by switching directional, OR try new fund fields ===
        # R1 observation: high-SR fundamentals (H91/H96/H97) all had corr 0.74-0.88 vs existing.
        # Hypothesis: pairing same fundamentals with ts_zscore(returns, 21) instead of VWAPdev
        # changes the temporal structure → may break correlation while keeping SR.
        ("H115_EarningsYield_zsRet21", "rank(multiply(rank(earnings_yield), rank(negative(ts_zscore(returns, 21)))))", "EarningsYield × zsRet21 — break corr"),
        ("H116_OpMargin_zsRet21",      "rank(multiply(rank(operating_margin), rank(negative(ts_zscore(returns, 21)))))", "OpMargin × zsRet21 — break corr"),
        ("H117_NetMargin_zsRet21",     "rank(multiply(rank(net_margin), rank(negative(ts_zscore(returns, 21)))))", "NetMargin × zsRet21 — break corr"),
        # NEW fundamental fields — not in VALUE/QUALITY/LEVERAGE families already saturated
        ("H118_FCFToEquity",   "rank(multiply(rank(fcf_to_equity), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "FCF/equity (capital efficiency) × VWAPdev"),
        ("H119_CashConvRatio", "rank(multiply(rank(cash_conversion_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Cash conversion ratio × VWAPdev"),
        ("H120_WorkingCap",    "rank(multiply(rank(working_capital), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Working capital × VWAPdev"),
        ("H121_CapexToRev",    "rank(multiply(rank(negative(capex_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low capex intensity × VWAPdev"),
        ("H122_SGAToRev",      "rank(multiply(rank(negative(sga_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low SG&A overhead × VWAPdev"),
        ("H123_invIntangibles", "rank(multiply(rank(negative(intangibles_to_assets)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Low intangibles (tangibility) × VWAPdev"),
        ("H124_TangibleBookPS", "rank(multiply(rank(tangible_book_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Tangible book/share × VWAPdev"),
        ("H125_NetCurrentAssetVal", "rank(multiply(rank(net_current_asset_value), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Graham NCAV × VWAPdev"),
        # Pure ACCRUAL signal (Sloan): NI - OCF normalized → buy low-accrual
        ("H126_invAccruals",   "rank(multiply(rank(negative(true_divide(subtract(net_income, cashflow_op), df_max(assets, 1.0)))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Sloan low-accrual × VWAPdev"),
        # Fund × OFI flow regime (different directional family)
        ("H127_invMC_OFInorm", "rank(multiply(rank(negative(market_cap)), rank(negative(true_divide(ts_sum(multiply(volume, sign(returns)), 5), df_max(ts_sum(volume, 5), 1.0))))))", "small × OFI flow regime"),
        ("H128_BM_OFInorm",    "rank(multiply(rank(book_to_market), rank(negative(true_divide(ts_sum(multiply(volume, sign(returns)), 5), df_max(ts_sum(volume, 5), 1.0))))))", "value × OFI flow regime"),
        # === Round 1 candidates kept for refinement (these still need testing under new corr-set) ===
        # VALUE family (Round 1 keeps for potential PASS)
        ("H85_FCFYield",         "rank(multiply(rank(free_cashflow_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "FCF yield × VWAPdev"),
        ("H86_invEVEBITDA",      "rank(multiply(rank(negative(ev_to_ebitda)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EV/EBITDA cheap × VWAPdev"),
        ("H87_invEVFCF",         "rank(multiply(rank(negative(ev_to_fcf)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EV/FCF cheap × VWAPdev"),
        ("H88_invEVOCF",         "rank(multiply(rank(negative(ev_to_ocf)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EV/OCF cheap × VWAPdev"),
        ("H89_invEVRev",         "rank(multiply(rank(negative(ev_to_revenue)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EV/Revenue cheap × VWAPdev"),
        ("H90_DivYield",         "rank(multiply(rank(dividend_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Dividend yield × VWAPdev"),
        ("H91_EarningsYield",    "rank(multiply(rank(earnings_yield), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Earnings yield × VWAPdev"),
        # QUALITY family
        ("H92_ROA",              "rank(multiply(rank(roa), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROA × VWAPdev"),
        ("H93_ROIC",             "rank(multiply(rank(roic), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROIC × VWAPdev"),
        ("H94_ROCE",             "rank(multiply(rank(roce), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROCE × VWAPdev"),
        ("H95_GrossMargin",      "rank(multiply(rank(gross_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Gross margin × VWAPdev"),
        ("H96_OperatingMargin",  "rank(multiply(rank(operating_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Op margin × VWAPdev"),
        ("H97_NetMargin",        "rank(multiply(rank(net_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "Net margin × VWAPdev"),
        ("H98_EBITDAMargin",     "rank(multiply(rank(ebitda_margin), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "EBITDA margin × VWAPdev"),
        # SAFETY/LEVERAGE family
        ("H99_invDE",            "rank(multiply(rank(negative(debt_to_equity)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low D/E × VWAPdev"),
        ("H100_invDebtAssets",   "rank(multiply(rank(negative(debt_to_assets)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low D/A × VWAPdev"),
        ("H101_CurrentRatio",    "rank(multiply(rank(current_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "current ratio × VWAPdev"),
        ("H102_QuickRatio",      "rank(multiply(rank(quick_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "quick ratio × VWAPdev"),
        ("H103_CashRatio",       "rank(multiply(rank(cash_ratio), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "cash ratio × VWAPdev"),
        ("H104_invNetDebt",      "rank(multiply(rank(negative(net_debt)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low net debt × VWAPdev"),
        ("H105_InterestCov",     "rank(multiply(rank(interest_coverage), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "interest coverage × VWAPdev"),
        # EFFICIENCY family
        ("H106_AssetTurnover",   "rank(multiply(rank(asset_turnover), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "asset turnover × VWAPdev"),
        ("H107_InventoryTurn",   "rank(multiply(rank(inventory_turnover), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "inventory turnover × VWAPdev"),
        # GROWTH/CHANGE family
        ("H108_ROEgrowth",       "rank(multiply(rank(ts_delta(roe, 252)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "ROE 252d change × VWAPdev"),
        ("H109_BMgrowth",        "rank(multiply(rank(ts_delta(book_to_market, 63)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "B/M 63d change × VWAPdev"),
        ("H110_invMCgrowth",     "rank(multiply(rank(negative(ts_delta(market_cap, 252))), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "shrinking MC × VWAPdev"),
        # MISC unused fields
        ("H111_BookPS",          "rank(multiply(rank(book_value_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "book/share × VWAPdev"),
        ("H112_RevenuePS",       "rank(multiply(rank(revenue_per_share), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "rev/share × VWAPdev"),
        ("H113_GrahamNum",       "rank(multiply(rank(graham_number), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "graham number × VWAPdev"),
        ("H114_invInterestExp",  "rank(multiply(rank(negative(interest_expense)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))", "low int expense × VWAPdev"),
    ]

    print(f"=== Loading data ===")
    uni, dates, tickers, mats, close, ret, cls, train_mask = load_data()
    print(f"  {len(tickers)} tickers, {train_mask.sum()} train bars")
    engine = FastExpressionEngine(data_fields=mats)

    print(f"=== Loading existing alphas ===")
    existing_train, n_existing = load_existing_normed(engine, uni, cls, train_mask)
    print(f"  {n_existing} existing alphas in DB, {len(existing_train)} loaded for corr ref")

    n_pass = 0
    n_skip_dup = 0
    print(f"\n=== Testing {len(QUEUE)} hypotheses ===")
    print(f"{'name':>30s}  {'SR':>5s}  {'fit':>5s}  {'corr':>5s}  {'TO':>5s}  flag")
    for name, expr, mech in QUEUE:
        try:
            r, err = evaluate(name, expr, engine, uni, cls, mats, close, ret, train_mask, existing_train)
            if r is None:
                print(f"  {name:>30s}  {'ERR':>5s}  -  -  -  {err}")
                continue
            flag = "PASS" if r['is_pass'] else "----"
            print(f"  {name:>30s}  {r['sr']:>+5.2f}  {r['fit']:>5.2f}  {r['max_corr']:>5.2f}  {r['to']*100:>4.1f}%  {flag}  ({mech})")
            if r['is_pass']:
                aid = save_pass(r, sig_train_for_addition=None)
                if aid is None:
                    n_skip_dup += 1
                    print(f"        -> already in DB, skipped")
                else:
                    n_pass += 1
                    print(f"        -> SAVED #{aid}")
                    # Add to existing_train for next iterations to compute corr against
                    sig = proc_signal(engine.evaluate(expr), uni, cls)
                    existing_train[aid] = sig.loc[train_mask]
        except Exception as e:
            print(f"  {name:>30s}  ERR  {e}")
            traceback.print_exc()

    print(f"\n=== DONE ===  saved {n_pass} new alphas (skipped {n_skip_dup} duplicates)")


if __name__ == "__main__":
    main()
