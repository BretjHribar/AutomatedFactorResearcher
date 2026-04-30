"""
Equal-weight portfolio evaluation for the SMALLCAP_D0 alpha cluster.

Combines all alphas tagged [SMALLCAP_D0] in the DB using the same
equal-weight combiner the kucoin trader uses (per-alpha process_signal,
then average), then walks the full TRAIN / VAL / TEST window producing:

  - Equity curve PNG (TRAIN/VAL/TEST shaded)
  - Per-split Sharpe / Fitness / max DD
  - Capacity estimate two ways:
      A. Simple: implied booksize at which avg-day position = X% of close-auction
         volume per name (auction volume modeled as MOC_FRAC of ADV20).
      B. Isichenko-style: max booksize before any position exceeds
         max_position_pct_adv constraint on the worst-case day.

Notes:
- Delay = 0 (signal at T traded into T's close-auction prints; PnL captured
  T-close to T+1-close).
- Universe = TOP1500TOP2500 (the universe these alphas were tuned on).
- Closing-auction execution differs from VWAP/TWAP: capacity is gated by MOC
  print volume specifically, not full-day ADV. We assume MOC_FRAC = 10% of
  ADV20 (typical for US large/mid-cap; for very small caps it can be 5-15%).
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Imports from project (avoid touching them) ────────────────────────────
from src.operators.fastexpression import FastExpressionEngine

# ── Config ────────────────────────────────────────────────────────────────
UNIVERSE        = "TOP1500TOP2500"
DELAY           = 0
MAX_WEIGHT      = 0.001            # 0.1% per name (matches what the alphas were saved at)
NEUTRALIZE      = "subindustry"
BOOKSIZE_REF    = 10_000_000.0     # only used to scale dollar metrics; SR is scale-invariant
MOC_FRAC        = 0.10             # closing-auction volume as fraction of ADV20
MAX_POS_PCT_ADV = 0.05             # Isichenko default — max 5% of name's ADV
TARGET_TRADE_PCT_MOC = 0.10        # we're willing to take 10% of the MOC print

DATA_DIR    = ROOT / "data/fmp_cache/matrices"
UNIV_DIR    = ROOT / "data/fmp_cache/universes"
DB          = ROOT / "data/alpha_results.db"
OUT_PNG     = ROOT / "data/smallcap_d0_equity_curve.png"
OUT_JSON    = ROOT / "data/smallcap_d0_portfolio_summary.json"

TRAIN_END = "2022-01-01"
VAL_END   = "2024-01-01"

# ──────────────────────────────────────────────────────────────────────────
def load_universe():
    df = pd.read_parquet(UNIV_DIR / f"{UNIVERSE}.parquet")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.values.dtype != bool:
        df = df.astype(bool)
    coverage = df.sum(axis=0) / len(df)
    valid = sorted(coverage[coverage > 0.5].index.tolist())
    return df[valid]

def load_matrices(tickers, dates):
    out = {}
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
        common_cols = [c for c in df.columns if c in tickers]
        if not common_cols:
            continue
        out[fp.stem] = df.loc[df.index.isin(dates), common_cols].reindex(index=dates, columns=tickers)
    return out

def load_classifications(tickers):
    cls = {}
    for k in ("subindustry", "industry", "sector"):
        fp = DATA_DIR / f"{k}.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp).reindex(columns=tickers)
        cls[k] = df.iloc[-1]
    return cls

def process_signal(sig, uni_mask, group_labels):
    """Same pipeline as eval_alpha_equity.process_signal but inline so we don't import."""
    sig = sig.astype(float)
    sig = sig.where(uni_mask, np.nan)
    # subindustry demean
    for grp in group_labels.dropna().unique():
        col_mask = (group_labels == grp).values
        if col_mask.any():
            sub = sig.iloc[:, col_mask]
            sig.iloc[:, col_mask] = sub.sub(sub.mean(axis=1), axis=0)
    abs_sum = sig.abs().sum(axis=1).replace(0, np.nan)
    sig = sig.div(abs_sum, axis=0)
    sig = sig.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT)
    return sig.fillna(0.0)

# ──────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {UNIVERSE} universe ...")
    uni_df = load_universe()
    print(f"  {uni_df.shape[1]} tickers x {uni_df.shape[0]} dates "
          f"({uni_df.index.min().date()} → {uni_df.index.max().date()})")

    dates = uni_df.index
    tickers = uni_df.columns.tolist()

    print("Loading matrices ...")
    matrices = load_matrices(tickers, dates)
    classifications = load_classifications(tickers)
    print(f"  {len(matrices)} matrices loaded")
    if "close" not in matrices:
        print("ERROR: close.parquet not found"); return 1

    # Pull SMALLCAP_D0 alphas from DB
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT a.id, a.expression
                   FROM alphas a JOIN evaluations e ON e.alpha_id=a.id
                   WHERE a.archived=0 AND a.notes LIKE '%SMALLCAP_D0%'
                   GROUP BY a.id ORDER BY a.id""")
    alphas = cur.fetchall()
    print(f"\nFound {len(alphas)} SMALLCAP_D0 alphas in DB:")
    for aid, expr in alphas:
        print(f"  #{aid}  {expr[:100]}")

    if not alphas:
        print("No alphas to combine"); return 1

    # Evaluate each
    engine = FastExpressionEngine(data_fields=matrices)
    group_labels = classifications.get(NEUTRALIZE, pd.Series(index=tickers))

    print("\nEvaluating alphas ...")
    normed = {}
    for aid, expr in alphas:
        try:
            raw = engine.evaluate(expr)
            n = process_signal(raw, uni_df, group_labels)
            normed[aid] = n
            print(f"  #{aid}  ok ({n.iloc[-1].notna().sum()} active in last bar)")
        except Exception as e:
            print(f"  #{aid}  ERROR: {e}")

    if not normed:
        print("All alphas failed"); return 1

    # Equal-weight combine: average then re-L1-normalize and re-clip
    print("\nCombining via equal-weight ...")
    combined = sum(normed.values()) / len(normed)
    combined = combined.div(combined.abs().sum(axis=1).replace(0, np.nan), axis=0)
    combined = combined.clip(lower=-MAX_WEIGHT, upper=MAX_WEIGHT).fillna(0.0)

    # Compute returns. delay=0: signal at T trades T-close → captures T to T+1 close-to-close return.
    close = matrices["close"]
    ret = close.pct_change().shift(-1)         # T's row holds (T+1 - T) / T
    pnl_per_bar = (combined * ret).sum(axis=1).dropna()

    eq = (1 + pnl_per_bar).cumprod()
    eq.name = "equity"

    # ── Splits ────────────────────────────────────────────────────────────
    train = pnl_per_bar.loc[:TRAIN_END]
    val   = pnl_per_bar.loc[TRAIN_END:VAL_END]
    test  = pnl_per_bar.loc[VAL_END:]

    def stats(r, name):
        if len(r) < 2:
            return {"name": name, "n": int(len(r))}
        ann = np.sqrt(252)
        sr = r.mean() / r.std() * ann if r.std() > 0 else float("nan")
        ret_ann = r.mean() * 252
        ret_cum = (1 + r).prod() - 1
        peak = (1 + r).cumprod().cummax()
        dd = ((1 + r).cumprod() / peak - 1).min()
        return {
            "name": name, "n": int(len(r)),
            "sharpe_ann": float(sr),
            "ret_ann": float(ret_ann),
            "ret_cum": float(ret_cum),
            "max_dd": float(dd),
            "vol_ann": float(r.std() * ann),
            "hit_rate": float((r > 0).mean()),
            "start": str(r.index.min().date()),
            "end":   str(r.index.max().date()),
        }

    s_train = stats(train, "TRAIN")
    s_val   = stats(val,   "VAL")
    s_test  = stats(test,  "TEST")
    s_full  = stats(pnl_per_bar, "FULL")

    print("\n========== PORTFOLIO PERFORMANCE ==========")
    for s in (s_train, s_val, s_test, s_full):
        print(f"  {s['name']:6s}  n={s['n']:>5}  SR={s['sharpe_ann']:+6.2f}  "
              f"ret={s['ret_ann']*100:+6.1f}%/yr  vol={s['vol_ann']*100:5.1f}%  "
              f"DD={s['max_dd']*100:+6.1f}%  hit={s['hit_rate']*100:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(eq.index, eq.values, color="black", lw=1.2)
    ax.axvline(pd.Timestamp(TRAIN_END), color="grey", ls="--", lw=0.8)
    ax.axvline(pd.Timestamp(VAL_END),   color="grey", ls="--", lw=0.8)
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx([ymin, ymax], eq.index[0], pd.Timestamp(TRAIN_END),
                     color="C0", alpha=0.07, label=f"TRAIN  SR={s_train['sharpe_ann']:.2f}")
    ax.fill_betweenx([ymin, ymax], pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END),
                     color="C1", alpha=0.10, label=f"VAL    SR={s_val['sharpe_ann']:.2f}")
    ax.fill_betweenx([ymin, ymax], pd.Timestamp(VAL_END), eq.index[-1],
                     color="C2", alpha=0.10, label=f"TEST   SR={s_test['sharpe_ann']:.2f}")
    ax.set_yscale("log")
    ax.set_title(f"SMALLCAP_D0 equal-weight portfolio  ({len(alphas)} alphas, "
                 f"{UNIVERSE}, delay={DELAY}, max_wt={MAX_WEIGHT})")
    ax.set_ylabel("Equity (log scale, start=1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved equity curve to: {OUT_PNG}")

    # ── Capacity (closing-auction aware) ─────────────────────────────────
    print("\n========== CAPACITY ANALYSIS ==========")
    print(f"  Closing-auction execution → capacity gated by MOC print volume,")
    print(f"  not full-day ADV. Modeling MOC volume = {MOC_FRAC*100:.0f}% of ADV20.")
    print(f"  Max participation: {TARGET_TRADE_PCT_MOC*100:.0f}% of MOC print = "
          f"{TARGET_TRADE_PCT_MOC*MOC_FRAC*100:.1f}% of ADV20.")

    if "adv20" not in matrices:
        print("  WARN: no adv20 matrix — capacity skipped")
        return 0

    adv20 = matrices["adv20"]
    # Cross-sectional weights → name-by-name fraction of book per day
    avg_abs_weights = combined.abs().mean(axis=0)              # avg weight per name
    avg_active = (combined.abs() > 0).sum(axis=1).mean()       # avg # names per day

    # Take the WORST CASE: at any given day a name receives up to MAX_WEIGHT
    # fraction of book. Capacity for that single name:
    #     max_book = (TARGET_TRADE_PCT_MOC * MOC_FRAC * ADV) / MAX_WEIGHT
    # Use the MEDIAN ADV among names that actually get traded by the strategy.
    # Filter to dates with positions and look at active-name ADV distribution.
    last_active_mask = combined.iloc[-252:].abs().mean(axis=0) > 1e-6  # active in last year
    active_tickers = last_active_mask[last_active_mask].index
    active_adv = adv20.iloc[-252:][active_tickers].mean(axis=0).dropna()
    adv_med = active_adv.median()
    adv_p10 = active_adv.quantile(0.10)
    adv_p25 = active_adv.quantile(0.25)

    # Simple model A: capacity = min over active names of (allowed $ per name) / weight
    # We use the typical (median) and conservative (10%-ile) ADV
    def cap_for_adv(adv):
        per_name_cap = TARGET_TRADE_PCT_MOC * MOC_FRAC * adv      # $ per day per name
        return per_name_cap / MAX_WEIGHT                          # implied book

    cap_med = cap_for_adv(adv_med)
    cap_p25 = cap_for_adv(adv_p25)
    cap_p10 = cap_for_adv(adv_p10)

    print(f"\n  Active-name ADV20 (last 252d, names with avg|w|>0):")
    print(f"    median:  ${adv_med/1e6:>6.2f}M    capacity (10% of MOC): ${cap_med/1e6:>6.2f}M book")
    print(f"    25%-ile: ${adv_p25/1e6:>6.2f}M    capacity (10% of MOC): ${cap_p25/1e6:>6.2f}M book")
    print(f"    10%-ile: ${adv_p10/1e6:>6.2f}M    capacity (10% of MOC): ${cap_p10/1e6:>6.2f}M book")
    print(f"    n_active names: {len(active_adv)}")

    # Isichenko-style: max book before ANY single position exceeds max_position_pct_adv.
    # weight_i * booksize <= max_position_pct_adv * ADV_i
    # => booksize <= max_position_pct_adv * ADV_i / weight_i for all i with weight_i > 0
    # Use the worst-case (min) over all (date, name) pairs in the last year.
    print(f"\n  Isichenko per-position cap ({MAX_POS_PCT_ADV*100:.0f}% of ADV per name):")
    recent_w = combined.iloc[-252:].abs()
    # broadcast adv to same shape
    recent_adv = adv20.iloc[-252:].reindex_like(recent_w)
    # implied max book: position_$ <= max_pct_adv * ADV  ==> book <= max_pct_adv * ADV / |w|
    safe = (recent_w > 1e-6) & (recent_adv > 0)
    implied_book = (MAX_POS_PCT_ADV * recent_adv / recent_w).where(safe)
    book_min = implied_book.min().min()
    book_p1  = np.nanpercentile(implied_book.values.flatten(), 1)
    book_p5  = np.nanpercentile(implied_book.values.flatten(), 5)
    print(f"    worst-day worst-name book cap:   ${book_min/1e6:>6.2f}M")
    print(f"    1st percentile of (date,name):   ${book_p1/1e6:>6.2f}M")
    print(f"    5th percentile of (date,name):   ${book_p5/1e6:>6.2f}M")

    # Daily turnover in dollars at REF book
    daily_turnover_frac = combined.diff().abs().sum(axis=1) / 2.0   # fractional one-way
    avg_to = daily_turnover_frac.mean()
    print(f"\n  Avg daily one-way turnover: {avg_to*100:.1f}% of GMV")
    print(f"  Avg active names per day:    {avg_active:.0f}")

    # Daily $ traded at user's target booksizes
    print(f"\n  Daily $ traded at various booksizes (one-way):")
    for bk in (100_000, 250_000, 500_000, 1_000_000, cap_p10, cap_med):
        print(f"    book ${bk:>10,.0f}  →  daily $ traded ${bk*avg_to:>10,.0f}  "
              f"(per-name avg ${bk*avg_to/avg_active:>8,.0f})")

    # Save summary JSON
    import json
    out = {
        "config": dict(universe=UNIVERSE, delay=DELAY, max_weight=MAX_WEIGHT,
                       neutralize=NEUTRALIZE, alphas=[a[0] for a in alphas]),
        "splits": {"train_end": TRAIN_END, "val_end": VAL_END},
        "performance": dict(train=s_train, val=s_val, test=s_test, full=s_full),
        "capacity": {
            "moc_frac_of_adv": MOC_FRAC,
            "target_trade_pct_moc": TARGET_TRADE_PCT_MOC,
            "active_adv_median_usd": float(adv_med),
            "active_adv_p10_usd": float(adv_p10),
            "active_adv_p25_usd": float(adv_p25),
            "n_active_names": int(len(active_adv)),
            "simple_capacity_median_book_usd": float(cap_med),
            "simple_capacity_p10_book_usd": float(cap_p10),
            "isichenko_max_book_worst_case_usd": float(book_min),
            "isichenko_max_book_p1_usd": float(book_p1),
            "isichenko_max_book_p5_usd": float(book_p5),
            "avg_daily_turnover_oneway": float(avg_to),
            "avg_active_names": float(avg_active),
        },
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary JSON: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
