"""
Populate the 11 fields in data/fmp_cache/matrices/ that are 100% NaN due to
schema-mismatched FMP column names in bulk_download.py:

  pe_ratio, pb_ratio, roe, roa, eps_diluted, sharesout,
  bookvalue_ps, revenue_per_share, fcf_per_share, debt_to_equity, ev_to_ebitda

Strategy:
- For 7 fields available in matrices_pit_v2/ (built by a newer pipeline that
  uses correct column names), reindex to matrices/ shape and copy.
- For 4 fields that don't exist in pit_v2 either (pe_ratio, pb_ratio,
  sharesout, revenue_per_share), derive from already-populated matrices.

Idempotent: only writes if the destination is currently 100% NaN.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent
MAT = ROOT / "data/fmp_cache/matrices"
PV2 = ROOT / "data/fmp_cache/matrices_pit_v2"


def load(dir_: Path, name: str) -> pd.DataFrame | None:
    fp = dir_ / f"{name}.parquet"
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def reindex_like(src: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    return src.reindex(index=ref.index, columns=ref.columns)


def coverage(df: pd.DataFrame) -> float:
    if df is None or df.size == 0:
        return 0.0
    arr = df.values
    if not np.issubdtype(arr.dtype, np.floating):
        return 0.0
    return float((~np.isnan(arr)).sum()) / arr.size


def main():
    close = load(MAT, "close")
    if close is None:
        print("ERROR: matrices/close.parquet missing; cannot proceed.")
        return 1

    print(f"Reference shape (matrices/close): {close.shape}")
    print(f"  dates {close.index.min().date()} to {close.index.max().date()}")
    print(f"  {len(close.columns)} tickers")
    print()

    # Sources from already-populated matrices/ (verified ~87% coverage in audit)
    revenue = load(MAT, "revenue")
    net_income = load(MAT, "net_income")
    equity = load(MAT, "equity")
    total_debt = load(MAT, "total_debt")
    free_cashflow = load(MAT, "free_cashflow")
    eps = load(MAT, "eps")

    # Step 1: shares_out + sharesout (alias) from pit_v2
    print("[1/3] Loading shares_out from matrices_pit_v2 ...")
    shares_pv2 = load(PV2, "shares_out")
    if shares_pv2 is None:
        print("  ERROR: pit_v2/shares_out.parquet missing")
        return 1
    shares_out = reindex_like(shares_pv2, close)
    print(f"  shares_out coverage after reindex: {coverage(shares_out)*100:.1f}%")

    # Step 2: copy 7 fields from pit_v2 (reindex to matrices/ shape)
    pv2_copy_fields = [
        "roe",
        "roa",
        "eps_diluted",
        "bookvalue_ps",
        "fcf_per_share",
        "debt_to_equity",
        "ev_to_ebitda",
    ]
    print(f"\n[2/3] Copying {len(pv2_copy_fields)} fields from matrices_pit_v2 ...")
    copied = {}
    for name in pv2_copy_fields:
        src = load(PV2, name)
        if src is None:
            print(f"  WARN: pit_v2/{name}.parquet missing; will skip")
            continue
        out = reindex_like(src, close)
        cov = coverage(out)
        copied[name] = out
        print(f"  {name:20s} coverage {cov*100:5.1f}%")

    # Step 3: derive pe_ratio, pb_ratio, sharesout, revenue_per_share
    print("\n[3/3] Deriving pe_ratio, pb_ratio, sharesout (alias), revenue_per_share ...")
    derived = {}

    # sharesout is just an alias for shares_out
    derived["sharesout"] = shares_out
    print(f"  sharesout            coverage {coverage(shares_out)*100:5.1f}% (alias of shares_out)")

    # pe_ratio = close / eps  (with eps>0 guard, otherwise NaN)
    if eps is not None:
        eps_safe = eps.where(eps > 0)  # only positive earnings
        pe = close / eps_safe
        # Clip extreme values: PE > 1000 is meaningless
        pe = pe.where(pe.abs() < 1000)
        derived["pe_ratio"] = pe
        print(f"  pe_ratio             coverage {coverage(pe)*100:5.1f}% (close/eps, eps>0, |pe|<1000)")
    else:
        print("  pe_ratio             SKIP: eps not available")

    # pb_ratio = close * shares_out / equity
    if equity is not None:
        eq_safe = equity.where(equity > 0)
        market_cap = close * shares_out
        pb = market_cap / eq_safe
        pb = pb.where(pb.abs() < 100)
        derived["pb_ratio"] = pb
        print(f"  pb_ratio             coverage {coverage(pb)*100:5.1f}% (close*shares/equity, equity>0, |pb|<100)")
    else:
        print("  pb_ratio             SKIP: equity not available")

    # revenue_per_share = revenue / shares_out
    if revenue is not None:
        so_safe = shares_out.where(shares_out > 0)
        rps = revenue / so_safe
        derived["revenue_per_share"] = rps
        print(f"  revenue_per_share    coverage {coverage(rps)*100:5.1f}% (revenue/shares_out)")
    else:
        print("  revenue_per_share    SKIP: revenue not available")

    # Step 3b: aliases that are also currently 100% NaN in matrices/ but are
    # just other names for fields we've already prepared.
    # bookvalue_ps ↔ book_value_per_share, roe ↔ return_equity, etc.
    aliases = {
        "book_value_per_share": copied.get("bookvalue_ps"),
        "return_equity": copied.get("roe"),
        "return_assets": copied.get("roa"),
        "sales_ps": derived.get("revenue_per_share"),
    }
    # tangible_book_per_share = (equity - intangibles - goodwill) / shares_out
    intangibles = load(MAT, "intangibles")
    goodwill = load(MAT, "goodwill")
    if equity is not None and intangibles is not None and goodwill is not None:
        tangible_eq = equity - intangibles.fillna(0) - goodwill.fillna(0)
        so_safe = shares_out.where(shares_out > 0)
        aliases["tangible_book_per_share"] = tangible_eq / so_safe

    # Step 3c: dividends_paid + debt_repayment from raw cashflow files.
    # Old bulk_download.py mapped these to columns that no longer exist
    # ("dividendsPaid", "debtRepayment"); current FMP cashflow uses
    # "commonDividendsPaid" and "longTermNetDebtIssuance".
    cashflow_dir = ROOT / "data/fmp_cache/cashflow"
    if cashflow_dir.exists():
        print("\n[3c] Extracting dividends_paid and debt_repayment from raw cashflow ...")
        div_mat = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        debt_rep_mat = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for sym in close.columns:
            fp = cashflow_dir / f"{sym}.parquet"
            if not fp.exists():
                continue
            try:
                cdf = pd.read_parquet(fp)
            except Exception:
                continue
            # PIT index: prefer acceptedDate, fall back to filingDate+1bd
            if "acceptedDate" in cdf.columns:
                try:
                    pit = pd.to_datetime(cdf["acceptedDate"]).dt.normalize()
                except Exception:
                    pit = cdf.index
            elif "filingDate" in cdf.columns:
                pit = pd.to_datetime(cdf["filingDate"]) + pd.Timedelta(days=1)
            else:
                pit = cdf.index + pd.Timedelta(days=90)
            cdf = cdf.copy()
            cdf.index = pit
            cdf = cdf.sort_index()
            if cdf.index.duplicated().any():
                cdf = cdf[~cdf.index.duplicated(keep="last")]

            if "commonDividendsPaid" in cdf.columns:
                div_mat[sym] = cdf["commonDividendsPaid"].reindex(close.index, method="ffill")
            if "longTermNetDebtIssuance" in cdf.columns:
                debt_rep_mat[sym] = cdf["longTermNetDebtIssuance"].reindex(close.index, method="ffill")

        derived["dividends_paid"] = div_mat
        derived["cashflow_dividends"] = div_mat  # alias
        derived["debt_repayment"] = debt_rep_mat
        print(f"  dividends_paid       coverage {coverage(div_mat)*100:5.1f}% (from commonDividendsPaid)")
        print(f"  debt_repayment       coverage {coverage(debt_rep_mat)*100:5.1f}% (from longTermNetDebtIssuance)")

        # dividend_yield = abs(dividends_paid) * 4 / market_cap (annualized)
        if "market_cap" in [p.stem for p in MAT.glob("*.parquet")]:
            mc = load(MAT, "market_cap")
            if mc is not None and div_mat is not None:
                annual_div = div_mat.abs() * 4
                mc_safe = mc.where(mc > 0)
                derived["dividend_yield"] = annual_div / mc_safe
                print(f"  dividend_yield       coverage {coverage(derived['dividend_yield'])*100:5.1f}% (4*|dividends_paid|/market_cap)")

    # Step 4: write back. For each, only overwrite if existing matrices file is
    # 100% NaN (or missing). This keeps the operation idempotent and safe.
    print("\nWriting fields to data/fmp_cache/matrices/ ...")
    all_fields = {**copied, **derived, "shares_out": shares_out, **{k: v for k, v in aliases.items() if v is not None}}
    written = []
    skipped = []
    for name, df in all_fields.items():
        target = MAT / f"{name}.parquet"
        cur = load(MAT, name)
        cur_cov = coverage(cur) if cur is not None else 0.0
        new_cov = coverage(df)
        if cur_cov > 0.05:
            skipped.append((name, cur_cov, new_cov))
            continue
        # Match dtype and reset to float
        df = df.astype(np.float64)
        df.to_parquet(target)
        written.append((name, new_cov))

    print(f"\nWrote {len(written)} fields:")
    for name, cov in written:
        print(f"  {name:20s} {cov*100:5.1f}% non-NaN")

    if skipped:
        print(f"\nSkipped {len(skipped)} fields (already had data, not overwriting):")
        for name, cur_cov, new_cov in skipped:
            print(f"  {name:20s} existing {cur_cov*100:5.1f}% / available {new_cov*100:5.1f}%")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
