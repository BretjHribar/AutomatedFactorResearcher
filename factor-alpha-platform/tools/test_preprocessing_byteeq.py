"""Byte-equivalence test: src.portfolio.preprocessing.apply_preprocess matches
the legacy `proc_signal_subind` (equity) and `signal_to_portfolio` (crypto)
helpers exactly on real data.

If this passes, the unified pipeline can replace the ad-hoc helpers with no
behavior change.
"""
import sqlite3, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.operators.fastexpression import FastExpressionEngine
from src.portfolio.preprocessing import apply_preprocess


# ============================================================================
# Legacy helpers — exact copies from eval_smallcap_d0_final and
# update_wq_alphas_db (both pre-refactor canonical)
# ============================================================================

MAX_W_EQUITY = 0.02


def legacy_proc_signal_equity(s, uni, cls):
    """Copy of proc_signal_subind in eval_smallcap_d0_final.py."""
    s = s.astype(float).where(uni, np.nan)
    for g in cls.dropna().unique():
        m = (cls == g).values
        if m.any():
            sub = s.iloc[:, m]
            s.iloc[:, m] = sub.sub(sub.mean(axis=1), axis=0)
    ab = s.abs().sum(axis=1).replace(0, np.nan)
    return s.div(ab, axis=0).clip(-MAX_W_EQUITY, MAX_W_EQUITY).fillna(0)


def legacy_signal_to_portfolio_crypto(sig):
    """Copy of signal_to_portfolio in update_wq_alphas_db.py."""
    s = sig.replace([np.inf, -np.inf], np.nan)
    demean = s.sub(s.mean(axis=1), axis=0)
    gross = demean.abs().sum(axis=1).replace(0, np.nan)
    return demean.div(gross, axis=0).fillna(0)


# ============================================================================
# EQUITY test
# ============================================================================

def test_equity():
    print("=== EQUITY test (proc_signal_subind) ===")
    UNIV_PATH = ROOT / "data/fmp_cache/universes/MCAP_100M_500M.parquet"
    DATA_DIR  = ROOT / "data/fmp_cache/matrices"
    DB        = ROOT / "data/alpha_results.db"

    uni = pd.read_parquet(UNIV_PATH).astype(bool)
    if not isinstance(uni.index, pd.DatetimeIndex):
        uni.index = pd.to_datetime(uni.index)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > 0.5].index.tolist())
    uni = uni[valid]
    dates = uni.index
    tickers = uni.columns.tolist()

    mats = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.stem.startswith("_"):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
        cc = [c for c in df.columns if c in tickers]
        if cc:
            mats[fp.stem] = df.loc[df.index.isin(dates), cc].reindex(
                index=dates, columns=tickers)
    cls = pd.read_parquet(DATA_DIR / "subindustry.parquet").iloc[-1].reindex(tickers)
    engine = FastExpressionEngine(data_fields=mats)

    rows = sqlite3.connect(DB).execute("""
        SELECT a.id, a.expression FROM alphas a
         WHERE a.archived=0 AND (a.notes LIKE '%SMALLCAP_D0_v2%' OR a.notes LIKE '%SMALLCAP_D0_v3%')
         ORDER BY a.id LIMIT 5""").fetchall()

    fails = 0
    for aid, expr in rows:
        sig = engine.evaluate(expr).reindex(index=dates, columns=tickers)
        legacy = legacy_proc_signal_equity(sig.copy(), uni, cls)
        new = apply_preprocess(
            sig,
            universe_mask=True, universe=uni,
            demean_method="subindustry", classifications=cls,
            normalize="l1", clip_max_w=MAX_W_EQUITY,
        )
        diff = (legacy - new).abs()
        max_diff = float(diff.max().max())
        passed = max_diff < 1e-12
        print(f"  a{aid}: max_abs_diff={max_diff:.2e}  {'OK' if passed else 'FAIL'}")
        if not passed:
            fails += 1
    return fails


# ============================================================================
# CRYPTO test
# ============================================================================

def test_crypto():
    print()
    print("=== CRYPTO test (signal_to_portfolio) ===")
    UNIV_PATH = ROOT / "data/kucoin_cache/universes/KUCOIN_TOP100_4h.parquet"
    DATA_DIR  = ROOT / "data/kucoin_cache/matrices/4h"
    DB        = ROOT / "data/alphas.db"
    COVERAGE_CUTOFF = 0.3

    uni = pd.read_parquet(UNIV_PATH)
    cov = uni.sum(axis=0) / len(uni)
    valid = sorted(cov[cov > COVERAGE_CUTOFF].index.tolist())
    matrices = {}
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        if fp.parent.name == "prod":
            continue
        df = pd.read_parquet(fp)
        cols = [c for c in valid if c in df.columns]
        if cols:
            matrices[fp.stem] = df[cols]
    tickers = sorted(set(matrices["close"].columns))
    for k, v in matrices.items():
        matrices[k] = v[[t for t in tickers if t in v.columns]]
    engine = FastExpressionEngine(data_fields=matrices)

    rows = sqlite3.connect(DB).execute("""
        SELECT id, expression FROM alphas
         WHERE archived=0 AND asset_class='crypto' AND interval='4h'
         ORDER BY id LIMIT 5""").fetchall()

    fails = 0
    for aid, expr in rows:
        sig = engine.evaluate(expr)
        legacy = legacy_signal_to_portfolio_crypto(sig.copy())
        new = apply_preprocess(
            sig,
            universe_mask=False,
            demean_method="cross_section",
            normalize="l1", clip_max_w=None,
        )
        # legacy and new should align on the same index/columns
        common_cols = legacy.columns.intersection(new.columns)
        common_idx  = legacy.index.intersection(new.index)
        diff = (legacy.loc[common_idx, common_cols]
                - new.loc[common_idx, common_cols]).abs()
        max_diff = float(diff.max().max())
        passed = max_diff < 1e-12
        print(f"  a{aid}: max_abs_diff={max_diff:.2e}  shape_l={legacy.shape}  shape_n={new.shape}  {'OK' if passed else 'FAIL'}")
        if not passed:
            fails += 1
    return fails


if __name__ == "__main__":
    fails = 0
    fails += test_equity()
    fails += test_crypto()
    print()
    print("PASS" if fails == 0 else f"FAIL ({fails} mismatches)")
    sys.exit(0 if fails == 0 else 1)
