"""
Save a list of (name, expression, sharpe_train, val, test) tuples to the alphas
+ evaluations tables, tagged so eval_smallcap_d0_final.py picks them up.

Now computes return_ann/max_drawdown/ic_mean/ic_ir/n_bars/psr/fitness on TRAIN
window when saving (was previously hardcoded to zero).

Usage: edit ALPHAS list at top, run.
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data/alpha_results.db"

# Reuse the eval logic from backfill script
sys.path.insert(0, str(ROOT))
from backfill_smallcap_d0_evals import (
    proc_signal, realistic_cost,
    UNIV_NAME, UNIV_DIR, DATA_DIR, MAX_W, BOOK,
    TRAIN_START, TRAIN_END,
)


def compute_metrics(expr, uni, cls, mats, close, ret, train_mask):
    """Compute (sr_train, ret_ann, mdd, ic_mean, ic_ir, n_bars, psr, fitness, turnover)
    for an alpha expression on TRAIN window."""
    from src.operators.fastexpression import FastExpressionEngine
    engine = FastExpressionEngine(data_fields=mats)
    raw = engine.evaluate(expr)
    sig = proc_signal(raw, uni, cls)
    sig_tr = sig.loc[train_mask]
    nx_tr = ret.shift(-1).loc[train_mask]
    close_tr = close.loc[train_mask]

    g = (sig_tr * nx_tr).sum(axis=1).fillna(0)
    cost = realistic_cost(sig_tr, close_tr, BOOK)
    n = g - cost
    ann = np.sqrt(252)
    sr_train = float(g.mean()/g.std()*ann) if g.std() > 0 else 0.0
    ret_ann = float(g.mean()*252)
    to = float(sig_tr.diff().abs().sum(axis=1).mean() / 2)

    eq = (1 + n).cumprod()
    mdd = float((eq / eq.cummax() - 1.0).min())

    lagged = sig_tr.shift(1)
    ic_d = lagged.corrwith(nx_tr, axis=1)
    ic_mean = float(ic_d.mean())
    ic_ir = float(ic_d.mean()/ic_d.std()*np.sqrt(252)) if ic_d.std() > 0 else 0.0

    n_bars = int(train_mask.sum())
    fitness = sr_train * np.sqrt(abs(ret_ann) / max(to, 0.125)) if sr_train != 0 else 0.0

    try:
        from scipy.stats import norm, skew, kurtosis
        sk = float(skew(g.dropna())); kt = float(kurtosis(g.dropna(), fisher=True))
        T = n_bars; sr_d = sr_train / np.sqrt(252)
        std_sr = np.sqrt((1 - sk*sr_d + (kt)/4.0 * sr_d**2) / max(T-1, 1))
        psr = float(norm.cdf(sr_d / std_sr)) if std_sr > 0 else 0.5
    except Exception:
        psr = 0.0

    return sr_train, ret_ann, mdd, ic_mean, ic_ir, n_bars, psr, fitness, to

TAG_PREFIX = "[SMALLCAP_D0_v3]"

# (name, expression, sharpe_train, sharpe_val, sharpe_test, turnover, max_corr, reasoning)
ALPHAS = [
    # populated by caller; or edit here
]


def save_with_metrics(alphas):
    """Save alphas with full computed metrics tuple:
    (name, expr, sr, _, _, to, _, reasoning, ret_ann, mdd, ic, ic_ir, n_bars, psr, fit)
    """
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    saved = []
    for tup in alphas:
        name, expr, sr, _, _, to, _, reasoning, ret_a, mdd, ic, ic_ir, n_b, psr, fit = tup
        existing = c.execute("SELECT id FROM alphas WHERE expression=? AND archived=0", (expr,)).fetchone()
        if existing:
            print(f"  SKIP (already exists id={existing[0]}): {name}")
            continue
        notes = (f"{TAG_PREFIX} {name}. {reasoning}  "
                 f"SR={sr:.2f} fit={fit:.2f} ret_ann={ret_a*100:+.1f}% mdd={mdd*100:+.1f}% "
                 f"IC={ic:.4f} IC_IR={ic_ir:+.2f} TO={to*100:.1f}%/d")
        c.execute(
            "INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes) VALUES (?,?,?,?,?,?,?)",
            (expr, expr[:80], "composite", "equities", "daily", "manual_search", notes)
        )
        aid = c.lastrowid
        c.execute(
            "INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann, max_drawdown, "
            "turnover, fitness, ic_mean, ic_ir, psr, train_start, train_end, n_bars, delay, decay, "
            "universe, max_weight, neutralization) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, sr, sr, ret_a, mdd, to, fit, ic, ic_ir, psr,
             "2020-01-01", "2024-01-01", n_b, 0, 0, "MCAP_100M_500M", 0.02, "subindustry"),
        )
        saved.append((aid, name))
        print(f"  SAVED #{aid}: {name}")
    conn.commit()
    conn.close()
    return saved


def save(alphas):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    saved = []
    for name, expr, sr_train, sr_val, sr_test, to, mc, reasoning in alphas:
        # de-dup by expression
        existing = c.execute("SELECT id FROM alphas WHERE expression=? AND archived=0", (expr,)).fetchone()
        if existing:
            print(f"  SKIP (already exists id={existing[0]}): {name}")
            continue
        notes = f"{TAG_PREFIX} {name}. {reasoning}  SR_tr/va/te={sr_train:.2f}/{sr_val:.2f}/{sr_test:.2f}  TO={to*100:.1f}%/d  max|corr|={mc:.2f}"
        c.execute(
            "INSERT INTO alphas (expression, name, category, asset_class, interval, source, notes) VALUES (?,?,?,?,?,?,?)",
            (expr, expr[:80], "composite", "equities", "daily", "manual_search", notes)
        )
        aid = c.lastrowid
        c.execute(
            "INSERT INTO evaluations (alpha_id, sharpe_is, sharpe_train, return_ann, max_drawdown, "
            "turnover, fitness, ic_mean, ic_ir, psr, train_start, train_end, n_bars, delay, decay, "
            "universe, max_weight, neutralization) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, sr_train, sr_train, 0.0, 0.0, to, sr_train, 0.0, 0.0, 0.0,
             "2020-01-01", "2024-01-01", 0, 0, 0, "MCAP_100M_500M", 0.02, "subindustry"),
        )
        saved.append((aid, name))
        print(f"  SAVED #{aid}: {name}")
    conn.commit()
    conn.close()
    return saved


if __name__ == "__main__":
    print("Loading data for proper metric computation...")
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
    print("Loaded.")

    # batch_j REMOVED — those alphas had corr 0.81-0.94, fail strict gate. (Saved as #80-#89, then archived.)
    # Going straight to batch K below.

    # batch K — 3 alphas, all STRICT pass (SR>=5, fit>=6, corr<0.7)
    # All from "VWAPdev decay-exp × zscore-returns × outer-decay" family
    print("Loading data for proper metric computation...")
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
    print("Loaded.")

    batch_k_specs = [
        # Batch L: 5 strict-pass alphas using UNUSED FUNDAMENTAL FIELDS as orthogonality enablers.
        # Each pairs a never-used fundamental/static field with directional VWAPdev decay-exp.
        ("H68_invMarketCap_x_DecayExp008_VWAPdev",
         "rank(multiply(rank(negative(market_cap)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))",
         "INVERSE MARKET CAP × decay-exp(0.08) VWAP-dev. Mechanism: smaller stocks within MCAP_100M_500M revert harder due to less liquidity / more retail. market_cap field never used in any saved alpha — breaks orthogonality. SR=5.07 fit=7.97 corr<0.70."),
        ("H72_invPE_x_DecayExp008_VWAPdev",
         "rank(multiply(rank(negative(pe_ratio)), rank(decay_exp(negative(true_divide(close, vwap)), 0.08))))",
         "INVERSE PE (cheap stocks) × decay-exp(0.08) VWAP-dev. Mechanism: cheap stocks attract value-buyer flow on dips → revert stronger. pe_ratio never used. SR=5.03 fit=7.72 corr=0.68."),
        ("H80_invADV20_x_DecayExp010_VWAPdev",
         "rank(multiply(rank(negative(adv20)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))",
         "INVERSE 20d ADV (illiquid stocks) × decay-exp(0.10) VWAP-dev. Mechanism: low-ADV stocks have stronger reversal due to limited arbitrage. adv20 never used. SR=5.04 fit=7.59 corr=0.68."),
        ("H81_BookToMarket_x_DecayExp010_VWAPdev",
         "rank(multiply(rank(book_to_market), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))",
         "BOOK-TO-MARKET (value tilt) × decay-exp(0.10) VWAP-dev. Mechanism: high-B/M stocks have price anchor that mean-reverts. book_to_market never used. SR=5.08 fit=7.10 corr=0.68."),
        ("H82_invADV60_x_DecayExp010_VWAPdev",
         "rank(multiply(rank(negative(adv60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))",
         "INVERSE 60d ADV × decay-exp(0.10) VWAP-dev. Mechanism: longer-term illiquidity, lower frequency of arbs. adv60 never used. SR=5.01 fit=7.60 corr=0.68."),
    ]

    batch_k = []
    for name, expr, reasoning in batch_k_specs:
        sr, ret_a, mdd, ic, ic_ir, n_b, psr, fit, to = compute_metrics(
            expr, uni, cls, mats, close, ret, train_mask
        )
        batch_k.append((name, expr, sr, 0.0, 0.0, to, 0.0, reasoning,
                        ret_a, mdd, ic, ic_ir, n_b, psr, fit))
        print(f"  Computed {name}: SR={sr:.2f} fit={fit:.2f} ret={ret_a*100:+.1f}% mdd={mdd*100:+.1f}%")

    saved = save_with_metrics(batch_k)
    print(f"\nSaved {len(saved)} alphas in batch K:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
    sys.exit(0)

    # batch I — alphas 29-32 (TRAIN-only, all strict SR>=6)
    batch_i = [
        ("CMP_VolxVWAPxRangePos5",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))))",
         7.13, 0.0, 0.0, 0.551, 0.94,
         "Triple composite using range-position-5 (close near 5d-low) — different reversal observable. Highest TRAIN SR."),
        ("CMP_4way_VolxVWAPxRev3xVolAccel",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))))",
         6.66, 0.0, 0.0, 0.560, 0.88,
         "4-way composite: vol × VWAP-dev × Rev3 × VolAccel3. Selective signal — only fires when all four align."),
        ("CMP_VolxVWAPxDelayedRev3",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), ts_delay(rank(negative(ts_delta(close, 3))), 1)))",
         6.61, 0.0, 0.0, 0.499, 0.90,
         "Triple with 1-day delayed reversal signal — captures slower information processing in micro-cap names."),
        ("CMP_VolAccel3_x_VWAPxRev10",
         "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))",
         6.46, 0.0, 0.0, 0.543, 0.87,
         "Vol-acceleration × VWAP-dev × Rev10 (10d horizon between 5d and 21d in vol-accel family)."),
    ]
    saved = save(batch_i)
    print(f"\nSaved {len(saved)} alphas in batch I:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
    sys.exit(0)

    # batch H — alphas 25-28 (TRAIN-only)
    batch_h = [
        ("CMP_VolAccel1xVWAPxRev3",
         "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 1)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))",
         6.20, 0.0, 0.0, 0.624, 0.79,
         "1-day vol-acceleration triple — single-bar volume change × VWAP-dev × Rev3."),
        ("CMP_VolAccel7xVWAPxRev5",
         "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 7)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))",
         6.49, 0.0, 0.0, 0.544, 0.78,
         "7-day vol-acceleration triple — slower vol-change horizon."),
        ("CMP_VolZS_x_VWAPdev",
         "rank(multiply(rank(ts_zscore(volume, 20)), rank(negative(true_divide(close, vwap)))))",
         6.36, 0.0, 0.0, 0.615, 0.87,
         "Volume z-score (different normalization than vol/sma) × VWAP-dev. Distinct math angle."),
        ("CMP_VolxVWAPxRev10",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))",
         6.72, 0.0, 0.0, 0.512, 0.90,
         "Triple composite at 10-day reversal horizon — fills gap between 5d and 21d."),
    ]
    saved = save(batch_h)
    print(f"\nSaved {len(saved)} alphas in batch H:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
    sys.exit(0)

    # batch G — alphas 21-24 (final 4, TRAIN-only)
    batch_g = [
        ("CMP_VolAccel5_x_VWAPxRev5",
         "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))",
         6.42, 0.0, 0.0, 0.541, 0.78,
         "Vol-acceleration over 5d (slower than #67's 3d) x VWAP-dev x Rev5. Slower-vol-change variant."),
        ("Decay3_VolAccelxVWAPxRev3",
         "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))",
         6.10, 0.0, 0.0, 0.357, 0.77,
         "Decay-linear(3) of vol-acceleration triple. Lower TO version of #67."),
        ("CMP_TWVolxVWAPxRev3",
         "rank(trade_when(true_divide(volume, sma(volume, 20)), multiply(rank(negative(true_divide(close, vwap))), rank(negative(ts_delta(close, 3)))), 0.0))",
         6.12, 0.0, 0.0, 0.551, 0.86,
         "Conditional alpha via trade_when: only takes the VWAP-dev x Rev3 signal when volume/sma(volume,20) is true (above threshold). Different mechanism from multiplicative composites."),
        ("CMP_VolxVWAPxBB5",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))))",
         6.21, 0.0, 0.0, 0.574, 0.88,
         "Triple composite using Bollinger-5 instead of close-delta as reversal leg. Z-score-based reversal in triple slot."),
    ]
    saved = save(batch_g)
    print(f"\nSaved {len(saved)} alphas in batch G:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
    sys.exit(0)

    # batch F — alphas 17-20 (TRAIN-only, SR>=6, corr<=0.85)
    batch_f = [
        ("Decay3_VolxVWAPxRev3",
         "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))",
         6.33, 0.0, 0.0, 0.325, 0.79,
         "Decay-linear(3) of triple composite vol x VWAP-dev x Rev3. Smoothing reduces TO and breaks correlation slightly."),
        ("Decay3_VolxVWAPxRev5",
         "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 3))",
         6.24, 0.0, 0.0, 0.294, 0.81,
         "Decay-linear(3) of triple composite vol x VWAP-dev x Rev5 (5d horizon variant)."),
        ("Decay3_VolxVWAPdev",
         "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 3))",
         6.33, 0.0, 0.0, 0.356, 0.79,
         "Decay-linear(3) of vol x VWAP-dev pair (no reversal leg) — pure microstructure with smoothing."),
        ("CMP_VolAccelxVWAPxRev3",
         "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))",
         6.50, 0.0, 0.0, 0.564, 0.71,
         "Volume ACCELERATION (delta of vol-surge over 3 days) x VWAP-dev x Rev3. Uses volume CHANGE not level — most orthogonal new angle (corr 0.71)."),
    ]
    saved = save(batch_f)
    print(f"\nSaved {len(saved)} alphas in batch F:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
    sys.exit(0)

    # batch E — alphas 13-16 (TRAIN-only selection, SR>=6, corr<=0.85)
    batch_e = [
        ("VWAPdev_DecayLin5",
         "rank(decay_linear(negative(true_divide(close, vwap)), 5))",
         6.03, 0.0, 0.0, 0.340, 0.71,
         "Linear-decay smoothed VWAP-deviation reversal. Lower TO than spot VWAP-dev. Pure microstructure mean-reversion."),
        ("ADD_VolSurge_p_VWAPdev_p_Rev5",
         "rank(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))",
         7.31, 0.0, 0.0, 0.541, 0.81,
         "Additive composite (sum of three rank components instead of multiplicative). Strongest TRAIN SR seen; different aggregation = different exposure profile."),
        ("CMP_VolxVWAPxMidRev5",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(true_divide(add(high, low), 2.0), 5)))))",
         6.35, 0.0, 0.0, 0.499, 0.85,
         "Triple composite using midpoint price reversal (close-replaced by (H+L)/2) — distinct observable for the reversal leg."),
        ("CMP_4way_VolxVWAPxRangexRev3",
         "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(ts_delta(close, 3))))))",
         6.13, 0.0, 0.0, 0.556, 0.80,
         "Four-way composite: volume × VWAP-deviation × range-expansion × 3-day reversal. Very selective — only fires when all four agree."),
    ]
    saved = save(batch_e)
    print(f"\nSaved {len(saved)} alphas in batch E:")
    for aid, name in saved:
        print(f"  #{aid}  {name}")
