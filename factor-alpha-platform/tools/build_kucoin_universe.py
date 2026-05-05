"""
Build the KuCoin factor-research universe from prod/config/kucoin_universe.json.

Universe construction follows the empirically-validated recipe documented in
experiments/universe_construction_research.md:
  - top-N by ADV (default TOP30)
  - configurable rebalance frequency (default 60 days)
  - configurable minimum history filter (default 365 days)
  - configurable vol-based exclusion (default top 5% Parkinson vol over TRAIN)
  - equal-weight position formation

Usage:
    python tools/build_kucoin_universe.py [--config prod/config/kucoin_universe.json]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent


def build_from_config(cfg: dict) -> pd.DataFrame:
    """Build the universe parquet per JSON config. Returns the universe DataFrame
    (date × ticker, bool)."""
    bars_per_day = int(cfg["bars_per_day"])
    mat_dir = ROOT / cfg["data"]["matrices_dir"]

    adv = pd.read_parquet(mat_dir / f"{cfg['data']['adv_field']}.parquet")
    close = pd.read_parquet(mat_dir / f"{cfg['data']['close_field']}.parquet")
    print(f"[load] adv={adv.shape}  close={close.shape}", flush=True)
    print(f"[load] span: {adv.index.min()} → {adv.index.max()}", flush=True)

    u = cfg["universe"]
    top_n = int(u["top_n"])
    rebal_bars = int(u["rebalance_days"]) * bars_per_day
    min_history_days = int(u["min_history_days"])

    # Per-ticker first-active = first non-NaN close. If a ticker has data from
    # the dataset's first bar, treat as "pre-existing" (back-date by 400 days)
    # so it's eligible from the start of TRAIN. Otherwise it must wait
    # min_history_days from its first observed bar.
    BACKDATE_BUFFER = pd.Timedelta(days=400)
    data_start = close.index[0]
    first_active = {}
    for col in close.columns:
        nonna = close[col].dropna().index
        if not len(nonna):
            continue
        first_active[col] = (data_start - BACKDATE_BUFFER) if nonna[0] == data_start else nonna[0]
    fa = pd.Series(first_active)
    n_pre = int((fa < data_start).sum())
    print(f"[history] {len(fa)} tickers with data, {n_pre} treated as pre-existing", flush=True)

    # Build the exclusion set per the configured method
    excl_set = set()
    excl_cfg = u.get("exclusion", {})
    method = excl_cfg.get("method")
    if method == "vol_rank":
        field = excl_cfg.get("field", "parkinson_volatility_60")
        vol_df = pd.read_parquet(mat_dir / f"{field}.parquet")
        scope = excl_cfg.get("scope", "train_window")
        if scope == "train_window":
            train_end = pd.Timestamp(cfg["splits"]["train_end"])
            scope_vol = vol_df.loc[:train_end].mean()
        elif scope == "full":
            scope_vol = vol_df.mean()
        else:
            raise ValueError(f"unknown scope {scope!r}")
        pct = float(excl_cfg["percentile_threshold"])
        threshold = scope_vol.quantile(pct / 100.0)
        excl_set = set(scope_vol[scope_vol > threshold].index.tolist())
        print(f"[exclusion] vol_rank top-{100-pct:.0f}% over {scope}: "
              f"{len(excl_set)} tickers excluded (threshold={threshold:.5f})", flush=True)
    elif method == "name_pattern":
        patterns = excl_cfg.get("patterns", [])
        for col in adv.columns:
            for pat in patterns:
                if pat.upper() in col.upper():
                    excl_set.add(col)
                    break
        print(f"[exclusion] name_pattern: {len(excl_set)} tickers excluded "
              f"(patterns: {patterns})", flush=True)
    elif method is None or method == "none":
        print("[exclusion] none", flush=True)
    else:
        raise ValueError(f"unknown exclusion method {method!r}")

    # Drop excluded tickers from eligibility
    fa = fa[~fa.index.isin(excl_set)]

    # Build per-bar membership: at each rebalance date, rank by ADV among
    # eligible tickers (those that have ≥ min_history_days since first_active),
    # take the top-N, hold for rebal_bars bars.
    out = pd.DataFrame(False, index=adv.index, columns=adv.columns)
    rebal_idx = list(range(0, len(adv), rebal_bars))
    print(f"[rebalance] every {u['rebalance_days']}d ({rebal_bars} bars), "
          f"{len(rebal_idx)} rebalances total", flush=True)

    last_members = None
    for i, b in enumerate(rebal_idx):
        next_b = rebal_idx[i + 1] if i + 1 < len(rebal_idx) else len(adv)
        rebal_ts = adv.index[b]
        eligible = [c for c in adv.columns
                    if c in fa and (rebal_ts - fa[c]).days >= min_history_days]
        adv_row = adv.iloc[b].reindex(eligible).dropna()
        if len(adv_row) < top_n:
            if last_members is None:
                continue
            members = last_members
        else:
            members = adv_row.nlargest(top_n).index.tolist()
            last_members = members
        out.iloc[b:next_b, out.columns.get_indexer(members)] = True

    # Stats
    n_active = out.sum(axis=1)
    n_unique = int(out.any(axis=0).sum())
    diffs = out.astype(int).diff().abs().sum(axis=1)
    n_change = int((diffs > 0).sum())
    avg_swap = float(diffs[diffs > 0].mean()) if (diffs > 0).any() else 0.0

    print(f"[result] active per bar — min/median/max: "
          f"{int(n_active.min())} / {int(n_active.median())} / {int(n_active.max())}", flush=True)
    print(f"[result] unique tickers ever in universe: {n_unique}", flush=True)
    print(f"[result] rebalance change-bars: {n_change}, avg swap: {avg_swap:.1f}", flush=True)

    # Per-split coverage check
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    for label, sl in [("TRAIN", slice(None, train_end)),
                       ("VAL",   slice(train_end, val_end)),
                       ("TEST",  slice(val_end, None))]:
        a = out.loc[sl].sum(axis=1)
        if len(a):
            print(f"[result] {label:5s} active/bar: mean={a.mean():.1f} "
                  f"min={int(a.min())} max={int(a.max())}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="prod/config/kucoin_universe.json")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text())

    out = build_from_config(cfg)

    # Save
    name = cfg["universe"]["name"]
    out_dir = ROOT / cfg["output"]["universe_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["output"]["filename_template"].format(name=name)

    if out_path.exists():
        backup = out_path.with_suffix(cfg["output"].get("legacy_backup_suffix", ".legacy.parquet"))
        if not backup.exists():
            out_path.rename(backup)
            print(f"[backup] existing {out_path.name} → {backup.name}", flush=True)
        else:
            print(f"[backup] backup already exists; overwriting in place", flush=True)
            out_path.unlink()

    out.to_parquet(out_path)
    print(f"[saved] {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
