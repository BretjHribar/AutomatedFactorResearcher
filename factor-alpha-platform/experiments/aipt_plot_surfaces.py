"""Plot AIPT parameter surfaces without cherry picking.

The script scans result folders under experiments/results/aipt* and writes:

  - heatmaps of mean Sharpe over P x ridge z;
  - heatmaps of seed-to-seed Sharpe dispersion;
  - TRAIN minus VAL+TEST overfit-gap heatmaps;
  - all-point seed scatter plots;
  - stepwise cost/layer/tau bar and scatter plots.

It is safe to run while long sweeps are still appending rows; the plots simply
represent the CSV state at the time the script starts.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT_DEFAULT = ROOT / "experiments" / "plots" / "aipt_surfaces"
SPLITS = ["TRAIN", "VAL", "TEST", "VAL+TEST", "FULL"]


@dataclass
class Summary:
    run_dir: Path
    path: Path
    kind: str
    df: pd.DataFrame


def _safe_name(text: object) -> str:
    s = str(text)
    s = s.replace("+", "plus").replace("-", "neg")
    s = re.sub(r"[^A-Za-z0-9_.]+", "_", s)
    return s.strip("_")[:180] or "x"


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] {path}: {type(exc).__name__}: {exc}")
        return None


def discover_summaries(result_dirs: list[str] | None) -> list[Summary]:
    if result_dirs:
        dirs = [ROOT / d if not Path(d).is_absolute() else Path(d) for d in result_dirs]
    else:
        dirs = sorted(p for p in RESULTS.glob("aipt*") if p.is_dir())
    out: list[Summary] = []
    for run_dir in dirs:
        for filename, kind in [
            ("aipt_unconstrained_summary.csv", "unconstrained"),
            ("aipt_asset_signal_unconstrained_summary.csv", "unconstrained"),
            ("aipt_asset_signal_seed_ensemble_summary.csv", "unconstrained"),
            ("aipt_stepwise_summary.csv", "stepwise"),
            ("aipt_summary.csv", "legacy"),
        ]:
            path = run_dir / filename
            if not path.exists():
                continue
            df = _read_csv(path)
            if df is None or df.empty:
                continue
            df["run_dir"] = run_dir.name
            out.append(Summary(run_dir=run_dir, path=path, kind=kind, df=df))
            break
    return out


def _write_table(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{_safe_name(name)}.csv"
    df.to_csv(path, index=False)
    return path


def _heatmap(
    table: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    cmap: str = "RdYlGn",
) -> None:
    if table.empty:
        return
    pivot = table.pivot(index=row_col, columns=col_col, values=value_col).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: float(x)), axis=1)
    vals = pivot.to_numpy(dtype=float)
    fig_w = max(6.0, 1.0 + 0.9 * pivot.shape[1])
    fig_h = max(3.8, 1.0 + 0.7 * pivot.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    finite = vals[np.isfinite(vals)]
    if finite.size:
        vmax = float(np.nanpercentile(np.abs(finite), 95))
        vmin = -vmax if np.nanmin(finite) < 0 < np.nanmax(finite) else float(np.nanmin(finite))
        vmax = vmax if np.nanmin(finite) < 0 < np.nanmax(finite) else float(np.nanmax(finite))
    else:
        vmin, vmax = -1.0, 1.0
    image = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{float(x):g}" for x in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(x) for x in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            x = vals[i, j]
            label = "nan" if not np.isfinite(x) else f"{x:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_unconstrained(summary: Summary, out_dir: Path) -> list[Path]:
    df = summary.df.copy()
    if "seed" not in df.columns and "n_seeds" in df.columns:
        df["seed"] = 0
    needed = {"scenario", "P", "z", "seed", "split", "SR"}
    if not needed.issubset(df.columns):
        return []
    df["SR"] = pd.to_numeric(df["SR"], errors="coerce")
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df = df.dropna(subset=["P", "z"])
    df["P"] = df["P"].astype(int)
    paths: list[Path] = []
    base_keys = ["run_dir", "scenario"]
    for optional in ["source_set", "projected_sources", "activation", "demean_features"]:
        if optional in df.columns:
            base_keys.append(optional)

    grouped = (
        df.groupby(base_keys + ["split", "P", "z"], dropna=False)
        .agg(SR_mean=("SR", "mean"), SR_std=("SR", "std"), seeds=("seed", "nunique"))
        .reset_index()
    )
    paths.append(_write_table(grouped, out_dir, f"{summary.run_dir.name}_unconstrained_surface_table"))

    for group_vals, part in grouped.groupby(base_keys, dropna=False, sort=True):
        group = dict(zip(base_keys, group_vals if isinstance(group_vals, tuple) else (group_vals,)))
        prefix = "__".join(f"{_safe_name(k)}_{_safe_name(v)}" for k, v in group.items())
        for split in SPLITS:
            sp = part[part["split"] == split]
            if sp.empty:
                continue
            mean_path = out_dir / f"{prefix}__split_{_safe_name(split)}__mean_sr_heatmap.png"
            _heatmap(
                sp,
                row_col="P",
                col_col="z",
                value_col="SR_mean",
                title=f"{summary.run_dir.name} {group.get('scenario')} {split} mean SR",
                out_path=mean_path,
            )
            paths.append(mean_path)
            std_path = out_dir / f"{prefix}__split_{_safe_name(split)}__seed_std_heatmap.png"
            _heatmap(
                sp,
                row_col="P",
                col_col="z",
                value_col="SR_std",
                title=f"{summary.run_dir.name} {group.get('scenario')} {split} seed std",
                out_path=std_path,
                cmap="viridis",
            )
            paths.append(std_path)

        pivot = part.pivot_table(index=["P", "z"], columns="split", values="SR_mean", aggfunc="first").reset_index()
        if {"TRAIN", "VAL+TEST"}.issubset(pivot.columns):
            pivot["TRAIN_minus_VALTEST"] = pivot["TRAIN"] - pivot["VAL+TEST"]
            gap_path = out_dir / f"{prefix}__train_minus_valtest_heatmap.png"
            _heatmap(
                pivot,
                row_col="P",
                col_col="z",
                value_col="TRAIN_minus_VALTEST",
                title=f"{summary.run_dir.name} {group.get('scenario')} TRAIN minus VAL+TEST",
                out_path=gap_path,
                cmap="coolwarm",
            )
            paths.append(gap_path)

    point_splits = [s for s in ["TRAIN", "VAL+TEST", "TEST"] if s in set(df["split"].astype(str))]
    for scenario, part in df.groupby("scenario", sort=True):
        sub = part[part["split"].isin(point_splits)].copy()
        if sub.empty:
            continue
        fig, axes = plt.subplots(
            1,
            len(point_splits),
            figsize=(5.5 * len(point_splits), 4.2),
            sharey=False,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, split in zip(axes, point_splits):
            sp = sub[sub["split"] == split]
            for p_value, pp in sp.groupby("P", sort=True):
                ax.scatter(
                    np.log10(pp["z"].astype(float)),
                    pp["SR"],
                    s=26,
                    alpha=0.75,
                    label=f"P={int(p_value)}",
                )
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_title(split)
            ax.set_xlabel("log10(z)")
            ax.set_ylabel("SR")
            ax.grid(True, linewidth=0.3, alpha=0.4)
        axes[-1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{summary.run_dir.name} {scenario}: every seed point", fontsize=11)
        path = out_dir / f"{_safe_name(summary.run_dir.name)}__{_safe_name(scenario)}__all_seed_points.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _step_combo(row: pd.Series) -> str:
    layer = str(row.get("layer", "layer"))
    tau = row.get("cost_tau", 0.0)
    turn_cap = float(row.get("turnover_cap", 0.0) or 0.0)
    blend = float(row.get("blend", 1.0) or 1.0)
    qp_scale = float(row.get("qp_alpha_scale", 1.0) or 1.0)
    qp_risk = float(row.get("qp_risk_lambda", 0.0) or 0.0)
    if str(layer).startswith("kernel_"):
        label = f"{layer}\ntau={float(tau):g}"
    else:
        label = layer
    if "qp" in layer:
        label += f"\nqps={qp_scale:g} qpr={qp_risk:g}"
    if turn_cap > 0:
        label += f"\ncap={turn_cap:g}"
    if blend < 1.0:
        label += f"\nblend={blend:g}"
    return label


def _plot_stepwise(summary: Summary, out_dir: Path) -> list[Path]:
    df = summary.df.copy()
    needed = {"scenario", "P", "z", "seed", "split", "layer", "cost_tau", "SR_net"}
    if not needed.issubset(df.columns):
        return []
    for col in ["P", "z", "seed", "cost_tau", "SR_net", "SR_sdf", "SR_gross", "turnover_per_bar", "cost_ann"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["combo"] = df.apply(_step_combo, axis=1)
    df["P"] = df["P"].astype(int)
    paths: list[Path] = []

    agg = (
        df.groupby(
            [
                col
                for col in [
                    "run_dir",
                    "scenario",
                    "P",
                    "z",
                    "split",
                    "layer",
                    "cost_tau",
                    "turnover_cap",
                    "blend",
                    "qp_alpha_scale",
                    "qp_risk_lambda",
                    "combo",
                ]
                if col in df.columns
            ],
            dropna=False,
        )
        .agg(
            SR_net_mean=("SR_net", "mean"),
            SR_net_std=("SR_net", "std"),
            SR_sdf_mean=("SR_sdf", "mean") if "SR_sdf" in df.columns else ("SR_net", "mean"),
            turnover_mean=("turnover_per_bar", "mean") if "turnover_per_bar" in df.columns else ("SR_net", "count"),
            cost_ann_mean=("cost_ann", "mean") if "cost_ann" in df.columns else ("SR_net", "count"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    paths.append(_write_table(agg, out_dir, f"{summary.run_dir.name}_stepwise_surface_table"))

    for (scenario, p_value, z_value), part in df.groupby(["scenario", "P", "z"], sort=True):
        preferred = [s for s in ["TRAIN", "VAL+TEST", "TEST"] if s in set(part["split"].astype(str))]
        if not preferred:
            continue
        combos = list(dict.fromkeys(part.sort_values(["layer", "cost_tau"])["combo"]))
        fig, axes = plt.subplots(
            len(preferred),
            1,
            figsize=(max(8.0, 0.55 * len(combos)), 3.2 * len(preferred)),
            sharex=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for ax, split in zip(axes, preferred):
            sp = part[part["split"] == split].copy()
            means = sp.groupby("combo")["SR_net"].mean().reindex(combos)
            stds = sp.groupby("combo")["SR_net"].std().reindex(combos).fillna(0.0)
            x = np.arange(len(combos))
            ax.bar(x, means.values, yerr=stds.values, color="#6A9FB5", edgecolor="black", linewidth=0.5, alpha=0.75)
            for seed, ss in sp.groupby("seed"):
                y = ss.groupby("combo")["SR_net"].mean().reindex(combos)
                ax.scatter(x, y.values, s=18, alpha=0.75, label=f"seed {int(seed)}")
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_ylabel(f"{split} net SR")
            ax.grid(axis="y", linewidth=0.3, alpha=0.4)
        axes[-1].set_xticks(np.arange(len(combos)))
        axes[-1].set_xticklabels(combos, rotation=60, ha="right", fontsize=7)
        axes[0].legend(fontsize=7, ncol=3, loc="best")
        fig.suptitle(f"{summary.run_dir.name} {scenario} P={int(p_value)} z={float(z_value):g}: all layer/tau/seed points", fontsize=11)
        path = out_dir / (
            f"{_safe_name(summary.run_dir.name)}__{_safe_name(scenario)}"
            f"__P{int(p_value)}__z{_safe_name(float(z_value))}__stepwise_all_points.png"
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        sp_agg = agg[(agg["scenario"] == scenario) & (agg["P"] == p_value) & (agg["z"] == z_value)]
        if {"TRAIN", "VAL+TEST"}.issubset(set(sp_agg["split"].astype(str))):
            pivot = sp_agg.pivot_table(index=["layer", "cost_tau", "combo"], columns="split", values="SR_net_mean", aggfunc="first").reset_index()
            pivot["TRAIN_minus_VALTEST"] = pivot["TRAIN"] - pivot["VAL+TEST"]
            pivot["row"] = np.arange(len(pivot))
            gap_fig, gap_ax = plt.subplots(figsize=(max(8.0, 0.55 * len(pivot)), 3.8), constrained_layout=True)
            colors = ["#B2182B" if x > 0 else "#2166AC" for x in pivot["TRAIN_minus_VALTEST"].fillna(0.0)]
            gap_ax.bar(np.arange(len(pivot)), pivot["TRAIN_minus_VALTEST"], color=colors, alpha=0.8)
            gap_ax.axhline(0.0, color="black", linewidth=0.6)
            gap_ax.set_xticks(np.arange(len(pivot)))
            gap_ax.set_xticklabels(pivot["combo"], rotation=60, ha="right", fontsize=7)
            gap_ax.set_ylabel("TRAIN minus VAL+TEST net SR")
            gap_ax.set_title(f"{summary.run_dir.name} {scenario} P={int(p_value)} z={float(z_value):g} overfit gap")
            gap_path = out_dir / (
                f"{_safe_name(summary.run_dir.name)}__{_safe_name(scenario)}"
                f"__P{int(p_value)}__z{_safe_name(float(z_value))}__stepwise_overfit_gap.png"
            )
            gap_fig.savefig(gap_path, dpi=160)
            plt.close(gap_fig)
            paths.append(gap_path)
    return paths


def write_index(out_dir: Path, summaries: list[Summary], images: list[Path]) -> None:
    lines = [
        "# AIPT Parameter Surface Plots",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "These plots are generated from every row present in the scanned summary CSVs at generation time.",
        "",
        "## Inputs",
        "",
    ]
    for s in summaries:
        rel = s.path.relative_to(ROOT)
        lines.append(f"- `{rel}`: kind={s.kind}, rows={len(s.df):,}")
    lines.extend(["", "## Images", ""])
    for img in sorted(images):
        rel = img.relative_to(ROOT)
        lines.append(f"- [{img.name}]({rel.as_posix()})")
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dirs", nargs="*", default=None, help="Optional result dirs to plot.")
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = discover_summaries(args.result_dirs)
    images: list[Path] = []
    for summary in summaries:
        if summary.kind == "unconstrained":
            images.extend(_plot_unconstrained(summary, out_dir))
        elif summary.kind == "stepwise":
            images.extend(_plot_stepwise(summary, out_dir))
    write_index(out_dir, summaries, images)
    print(f"[done] summaries={len(summaries)} images={len(images)} out={out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
