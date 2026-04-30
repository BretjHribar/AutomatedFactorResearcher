"""
Capacity sweep on EQUITY: equal × {diag, style+pca} across book sizes
with the ADV-cap on the QP.

Thin driver over src.pipeline.runner — book + risk_model are config overrides
per cell. ADV cap = MOC_FRAC × MAX_PARTICIPATION × ADV20_i / book.

Reports VAL / TEST / avg(V,T) net SR + ret%/yr per book level so we can see
where capacity bites. (TRAIN dropped — alphas were selected on it.)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.runner import run, merge_overrides

CONFIG = ROOT / "prod" / "config" / "research_equity.json"

BOOKS = [100_000, 250_000, 500_000, 1_000_000, 1_500_000, 2_000_000]
RISK_MODELS = ["diagonal", "style+pca"]

# Capacity model
MOC_FRAC = 0.10
MAX_MOC_PART = 0.30
NET_SR_THRESHOLD = 4.0
CAP_HIT_THRESHOLD = 5.0  # %


def adv_metrics(w, adv, book):
    pos_dollar = (w * book).abs()
    moc_dollar = (adv * MOC_FRAC).reindex_like(w)
    moc_part = (pos_dollar / moc_dollar.where(moc_dollar > 0)).where(pos_dollar > 0)
    active = moc_part.stack().dropna()
    if len(active) == 0:
        return float("nan"), float("nan")
    p99 = float(active.quantile(0.99) * 100)
    cap_hit = float((active > MAX_MOC_PART).mean() * 100)
    return p99, cap_hit


def main():
    base = json.loads(CONFIG.read_text())
    print(f"=== capacity sweep (equity, equal × {{diag, style+pca}}) ===", flush=True)

    # ADV20 matrix is needed both for the QP cap and for the cap_hit metric.
    matrices_dir = ROOT / base["data"]["matrices_dir"]
    adv = pd.read_parquet(matrices_dir / "adv20.parquet")
    if not isinstance(adv.index, pd.DatetimeIndex):
        adv.index = pd.to_datetime(adv.index)

    all_results = {rname: [] for rname in RISK_MODELS}
    hdr = (f"{'Book':>10s}  {'cost%/yr':>9s}  "
           f"{'SR_val':>7s}  {'SR_test':>8s}  {'avgSR':>6s}  "
           f"{'ret_val%':>9s}  {'ret_test%':>10s}  {'avg_ret%':>9s}  "
           f"{'p99 MOC%':>8s}  {'cap_hit%':>8s}")

    for rname in RISK_MODELS:
        print(f"\n=== Risk model: {rname} ===", flush=True)
        print(hdr, flush=True)
        for book in BOOKS:
            cfg = merge_overrides(base, {
                "book": book,
                "combiner": {"name": "equal", "params": {"max_wt": 0.02}},
                "risk_model": {"name": rname,
                                "params": base["risk_model"]["params"]},
                "qp": merge_overrides(base["qp"], {
                    "adv_cap": {"adv_field": "adv20",
                                 "moc_frac": MOC_FRAC,
                                 "max_part": MAX_MOC_PART}
                }),
            })
            res = run(cfg, verbose=False)
            m = res.metrics
            cost_yr = float(res.cost.mean()) * base["annualization"]["bars_per_year"] * 100
            sr_v = m["VAL"]["SR_net"]
            sr_t = m["TEST"]["SR_net"]
            ret_v = m["VAL"]["ret_ann_net"] * 100
            ret_t = m["TEST"]["ret_ann_net"] * 100
            avg_sr = (sr_v + sr_t) / 2
            avg_ret = (ret_v + ret_t) / 2
            p99, cap_hit = adv_metrics(res.weights, adv, book)
            r = {"book": book, "cost_yr": cost_yr,
                 "sr_val": sr_v, "sr_test": sr_t,
                 "ret_val": ret_v, "ret_test": ret_t,
                 "p99": p99, "cap_hit": cap_hit}
            all_results[rname].append(r)
            print(f"  ${book/1000:>7.0f}K  {cost_yr:>7.2f}%  "
                  f"{sr_v:>+6.2f}  {sr_t:>+7.2f}  {avg_sr:>+5.2f}  "
                  f"{ret_v:>+8.2f}%  {ret_t:>+9.2f}%  {avg_ret:>+8.2f}%  "
                  f"{p99:>6.2f}%  {cap_hit:>6.2f}%", flush=True)

    # Side-by-side
    print("\n" + "=" * 110)
    print("SIDE-BY-SIDE: equal × {diag, style+pca}")
    print("=" * 110)
    print(f"{'Book':>11s} | "
          f"{'d avgSR':>8s} {'d avg_ret%':>11s} {'d cap%':>7s}  | "
          f"{'s+p avgSR':>9s} {'s+p avg_ret%':>13s} {'s+p cap%':>9s}")
    print("-" * 110)
    for i, book in enumerate(BOOKS):
        d = all_results["diagonal"][i]; s = all_results["style+pca"][i]
        d_avg_sr = (d["sr_val"] + d["sr_test"]) / 2
        s_avg_sr = (s["sr_val"] + s["sr_test"]) / 2
        d_avg_ret = (d["ret_val"] + d["ret_test"]) / 2
        s_avg_ret = (s["ret_val"] + s["ret_test"]) / 2
        print(f"  ${book/1e6:>7.2f}M | "
              f"{d_avg_sr:>+7.2f} {d_avg_ret:>+10.2f}% {d['cap_hit']:>6.2f}% | "
              f"{s_avg_sr:>+8.2f} {s_avg_ret:>+12.2f}% {s['cap_hit']:>8.2f}%")

    print(f"\n=== Capacity cutoffs (avg(VAL,TEST) net SR ≥ {NET_SR_THRESHOLD}, "
          f"cap_hit% ≤ {CAP_HIT_THRESHOLD}) ===")
    for rname in RISK_MODELS:
        df = pd.DataFrame(all_results[rname])
        df["avgSR"]  = (df["sr_val"]  + df["sr_test"])  / 2
        df["avg_ret"] = (df["ret_val"] + df["ret_test"]) / 2
        viable = df[(df["avgSR"] >= NET_SR_THRESHOLD) & (df["cap_hit"] <= CAP_HIT_THRESHOLD)]
        if len(viable):
            top = viable.iloc[-1]
            print(f"  {rname:10s}: max viable book = ${top['book']/1e6:.2f}M  "
                  f"(avgSR {top['avgSR']:+.2f}, avg_ret {top['avg_ret']:+.2f}%/yr, "
                  f"cost {top['cost_yr']:.2f}%/yr, cap_hit {top['cap_hit']:.1f}%)")
        else:
            print(f"  {rname:10s}: NO viable book at threshold")


if __name__ == "__main__":
    main()
