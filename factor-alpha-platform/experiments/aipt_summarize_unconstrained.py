"""Summarize unconstrained AIPT sweep results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fmt(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.3f}"


def summarize(path: Path, top: int) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print(f"No rows in {path}")
        return

    keys = ["scenario", "source_set", "P", "z", "activation", "demean_features"]
    for optional in ["projected_sources", "project_top_k"]:
        if optional in df.columns:
            keys.append(optional)
    g = (
        df.groupby(keys + ["split"], dropna=False)
        .agg(
            SR_mean=("SR", "mean"),
            SR_std=("SR", "std"),
            seeds=("seed", "nunique"),
            n_fields=("n_fields", "median"),
            n_names=("n_names", "median"),
            hjd_test=("hjd_test", "mean"),
        )
        .reset_index()
    )

    print(f"Summary file: {path}")
    print(f"Rows: {len(df):,}; completed cells: {len(df) // df['split'].nunique():,}")

    for split in ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]:
        part = g[g["split"] == split].sort_values(["scenario", "SR_mean"], ascending=[True, False])
        print(f"\nTop {split} by scenario")
        for scenario, s in part.groupby("scenario", sort=True):
            print(f"  {scenario}")
            cols = ["P", "z", "SR_mean", "SR_std", "seeds", "n_fields", "n_names", "hjd_test"]
            for row in s.head(top)[cols].itertuples(index=False):
                print(
                    f"    P={int(row.P):4d} z={row.z:g} "
                    f"SR={_fmt(row.SR_mean)} +/- {_fmt(row.SR_std)} "
                    f"seeds={int(row.seeds)} fields={int(row.n_fields)} names={int(row.n_names)} "
                    f"HJD={_fmt(row.hjd_test)}"
                )

    pivot = g.pivot_table(
        index=keys,
        columns="split",
        values="SR_mean",
        aggfunc="first",
    ).reset_index()
    if {"TRAIN", "VAL+TEST"}.issubset(pivot.columns):
        print("\nSelection overfit check: TRAIN-selected vs VAL+TEST-selected")
        for scenario, s in pivot.groupby("scenario", sort=True):
            train_best = s.sort_values("TRAIN", ascending=False).head(1).iloc[0]
            vt_best = s.sort_values("VAL+TEST", ascending=False).head(1).iloc[0]
            print(f"  {scenario}")
            for label, row in [("TRAIN-selected", train_best), ("VAL+TEST-selected", vt_best)]:
                gap = row["TRAIN"] - row["VAL+TEST"]
                print(
                    f"    {label}: TRAIN={_fmt(row['TRAIN'])} VAL+TEST={_fmt(row['VAL+TEST'])} "
                    f"TEST={_fmt(row.get('TEST', float('nan')))} gap={gap:+.3f} "
                    f"P={int(row.P)} z={row.z:g}"
                )

    vt = g[g["split"] == "VAL+TEST"].copy()
    if not vt.empty:
        print("\nComplexity table: best VAL+TEST per scenario/P")
        best = vt.sort_values("SR_mean", ascending=False).groupby(["scenario", "P"], as_index=False).head(1)
        best = best.sort_values(["scenario", "P"])
        for row in best.itertuples(index=False):
            print(
                f"  {row.scenario:20s} P={int(row.P):4d} z={row.z:g} "
                f"SR={_fmt(row.SR_mean)} +/- {_fmt(row.SR_std)}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("summary_csv", type=Path)
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()
    summarize(args.summary_csv, args.top)


if __name__ == "__main__":
    main()
