from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparison_order import (
    COMPARISON_METRIC_COLUMNS,
    METRIC_LABEL,
    VAL_METRIC_HISTORY_KEYS,
    model_colors,
    model_display,
    reorder_comparison_df,
    sort_models,
    sort_variants,
    variant_colors,
    variant_display,
)
from src.config import ensure_dir
from src.plot_utils import save_show_close_figure

DEFAULT_METRICS: list[str] = list(COMPARISON_METRIC_COLUMNS)


def _candidate_history_paths(model_dir: Path) -> list[Path]:
    return [
        model_dir / "history.csv",
        model_dir / "history_full.csv",
        model_dir / "training_history.csv",
        model_dir / "history.json",
        model_dir / "training_history.json",
        model_dir / "history_head.csv",
        model_dir / "history_finetune.csv",
    ]


def _read_history_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "history" in data:
                data = data["history"]
            df = pd.DataFrame(data)
        else:
            return None
    except Exception as e:
        print(f"[WARN] Nem sikerult history-t olvasni: {path} | {e}")
        return None

    if len(df) == 0:
        return None

    if "epoch" not in df.columns:
        if "epoch_global" in df.columns:
            df.insert(0, "epoch", df["epoch_global"].astype(int))
        else:
            df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def find_training_history(model_dir: str | Path) -> pd.DataFrame | None:
    model_dir = Path(model_dir)
    for path in _candidate_history_paths(model_dir):
        df = _read_history_file(path)
        if df is not None:
            return df
    for path in sorted(model_dir.rglob("*history*.csv")) + sorted(model_dir.rglob("*history*.json")):
        df = _read_history_file(path)
        if df is not None:
            return df
    return None


def _infer_model_dir(row: pd.Series) -> Path | None:
    for col in ["out_dir", "model_dir"]:
        if col in row and pd.notna(row[col]):
            return Path(row[col])
    if "model_path" in row and pd.notna(row["model_path"]):
        return Path(row["model_path"]).parent
    return None


def _find_metric_pair(hist: pd.DataFrame, base_names: list[str]) -> tuple[str | None, str | None, str]:
    for name in base_names:
        train_col = name if name in hist.columns else None
        val_col = f"val_{name}" if f"val_{name}" in hist.columns else None
        if train_col is not None or val_col is not None:
            return train_col, val_col, name
    return None, None, base_names[0]


def _first_hist_col(hist: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in hist.columns:
            return name
    return None


def plot_metric_bars(
    comparison_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")
    df = reorder_comparison_df(comparison_df)
    labels = [
        f"{model_display(str(r['model']))}\n({variant_display(str(r['data_variant']))})"
        for _, r in df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df)), 5))
    ax.bar(labels, df[metric].tolist())
    ylab = METRIC_LABEL.get(metric, metric)
    ax.set_title(title or f"Model comparison — {ylab}")
    ax.set_ylabel(ylab)
    ax.set_xlabel("Model / variant")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_show_close_figure(fig, save_path=save_path, show=show)


def plot_metric_by_variant(
    comparison_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")
    df = comparison_df.dropna(subset=["model", "data_variant"]).copy()
    pivot_df = df.pivot(index="model", columns="data_variant", values=metric)
    ri = [m for m in sort_models(pivot_df.index) if m in pivot_df.index]
    ci = [v for v in sort_variants(pivot_df.columns) if v in pivot_df.columns]
    pivot_df = pivot_df.reindex(index=ri, columns=ci)
    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(ci)), 5))
    pivot_df.plot(kind="bar", ax=ax, color=[variant_colors().get(str(c), "gray") for c in pivot_df.columns])
    ylab = METRIC_LABEL.get(metric, metric)
    ax.set_title(title or f"{ylab} — modell × variáns")
    ax.set_ylabel(ylab)
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.3)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("variant")
        for t in leg.get_texts():
            t.set_text(variant_display(t.get_text()))
    ax.set_xticklabels([model_display(str(t.get_text())) for t in ax.get_xticklabels()])
    fig.tight_layout()
    save_show_close_figure(fig, save_path=save_path, show=show)


def plot_models_within_variants_metric_row(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False) -> None:
    """Egy variánson belül: a 4 metrika egy sorban; oszlopdiagramon a modellek külön színnel."""
    out_dir = ensure_dir(Path(out_dir) / "models_within_variants")
    df = reorder_comparison_df(comparison_df)
    variants = sort_variants(df["data_variant"].dropna().unique())
    models = sort_models(df["model"].dropna().unique())
    mc = model_colors()
    n_m = len(models)
    x = np.arange(n_m)
    w = 0.65
    for variant in variants:
        sub = df[df["data_variant"].astype(str) == variant]
        fig, axes = plt.subplots(1, len(COMPARISON_METRIC_COLUMNS), figsize=(4.2 * len(COMPARISON_METRIC_COLUMNS), 4.5), squeeze=False)
        row = axes[0]
        for ax, metric in zip(row, COMPARISON_METRIC_COLUMNS):
            if metric not in sub.columns:
                ax.set_visible(False)
                continue
            heights: list[float] = []
            colors: list = []
            for m in models:
                r = sub[sub["model"].astype(str) == m]
                heights.append(float(r[metric].iloc[0]) if len(r) else float("nan"))
                colors.append(mc.get(m, (0.5, 0.5, 0.5, 1.0)))
            ax.bar(x, heights, width=w, color=colors, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([model_display(m) for m in models], rotation=35, ha="right", fontsize=9)
            ylab = METRIC_LABEL.get(metric, metric)
            ax.set_title(ylab)
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            if metric != "loss":
                ax.set_ylim(0, 1)
        fig.suptitle(f"Modellek összehasonlítása variánson belül — {variant_display(variant)}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_show_close_figure(fig, save_path=out_dir / f"{variant}_metrics_row.png", show=show)


def plot_variants_within_models_metric_row(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False) -> None:
    """Egy modellen belül: a 4 metrika egy sorban; oszlopdiagramon a variánsok külön színnel."""
    out_dir = ensure_dir(Path(out_dir) / "variants_within_models")
    df = reorder_comparison_df(comparison_df)
    variants = sort_variants(df["data_variant"].dropna().unique())
    models = sort_models(df["model"].dropna().unique())
    vc = variant_colors()
    n_v = len(variants)
    x = np.arange(n_v)
    w = 0.65
    for model in models:
        sub = df[df["model"].astype(str) == model]
        fig, axes = plt.subplots(1, len(COMPARISON_METRIC_COLUMNS), figsize=(4.2 * len(COMPARISON_METRIC_COLUMNS), 4.5), squeeze=False)
        row = axes[0]
        for ax, metric in zip(row, COMPARISON_METRIC_COLUMNS):
            if metric not in sub.columns:
                ax.set_visible(False)
                continue
            heights: list[float] = []
            colors: list = []
            for v in variants:
                r = sub[sub["data_variant"].astype(str) == v]
                heights.append(float(r[metric].iloc[0]) if len(r) else float("nan"))
                colors.append(vc.get(v, (0.5, 0.5, 0.5, 1.0)))
            ax.bar(x, heights, width=w, color=colors, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([variant_display(v) for v in variants], rotation=0, fontsize=10)
            ylab = METRIC_LABEL.get(metric, metric)
            ax.set_title(ylab)
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            if metric != "loss":
                ax.set_ylim(0, 1)
        fig.suptitle(f"Variánsok összehasonlítása modellen belül — {model_display(model)}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_show_close_figure(fig, save_path=out_dir / f"{model}_metrics_row.png", show=show)


def plot_metric_heatmaps(comparison_df: pd.DataFrame, out_dir: str | Path, metrics: Iterable[str] | None = None, show: bool = False):
    out_dir = ensure_dir(Path(out_dir) / "heatmaps")
    if metrics is None:
        metrics = DEFAULT_METRICS
    df = comparison_df.dropna(subset=["model", "data_variant"]).copy()
    for metric in metrics:
        if metric not in df.columns:
            continue
        pivot_df = df.dropna(subset=[metric]).pivot_table(index="model", columns="data_variant", values=metric, aggfunc="mean")
        if pivot_df.empty:
            continue
        ri = [m for m in sort_models(pivot_df.index) if m in pivot_df.index]
        ci = [v for v in sort_variants(pivot_df.columns) if v in pivot_df.columns]
        pivot_df = pivot_df.reindex(index=ri, columns=ci)
        fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(pivot_df.columns)), max(5, 0.6 * len(pivot_df.index))))
        im = ax.imshow(pivot_df.values, aspect="auto")
        ylab = METRIC_LABEL.get(metric, metric)
        ax.set_title(f"Modell × variáns — {ylab}")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Model")
        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels([variant_display(str(c)) for c in pivot_df.columns], rotation=30, ha="right")
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels([model_display(str(i)) for i in pivot_df.index])
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                val = pivot_df.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=ylab)
        fig.tight_layout()
        save_show_close_figure(fig, save_path=out_dir / f"heatmap_{metric}.png", show=show)


def plot_all_main_metrics(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    out_dir = ensure_dir(out_dir)
    df = reorder_comparison_df(comparison_df)
    for metric in DEFAULT_METRICS:
        if metric not in df.columns:
            continue
        ylab = METRIC_LABEL.get(metric, metric)
        plot_metric_bars(
            df,
            metric,
            save_path=Path(out_dir) / f"bar_{metric}.png",
            title=f"Összehasonlítás — {ylab}",
            show=show,
        )
        plot_metric_by_variant(
            df,
            metric,
            save_path=Path(out_dir) / f"grouped_{metric}.png",
            title=f"{ylab} — modell / variáns",
            show=show,
        )
    plot_models_within_variants_metric_row(df, out_dir=out_dir, show=show)
    plot_variants_within_models_metric_row(df, out_dir=out_dir, show=show)
    plot_metric_heatmaps(df, out_dir, show=show)


def plot_training_history_for_row(row: pd.Series, out_dir: str | Path, show: bool = False) -> Path | None:
    model = str(row["model"])
    variant = str(row["data_variant"])
    model_dir = _infer_model_dir(row)
    if model_dir is None:
        return None
    hist = find_training_history(model_dir)
    if hist is None:
        return None
    out_dir = ensure_dir(Path(out_dir) / "training_curves")
    metric_specs: list[tuple[str | None, str | None, str]] = []
    for names, title in [
        (["loss"], "loss"),
        (["accuracy", "sparse_categorical_accuracy"], "Accuracy"),
        (["recall_macro", "macro_recall", "recall"], "Macro recall"),
        (["roc_auc_macro_ovr", "macro_auc", "auc"], "ROC-AUC"),
    ]:
        train_col, val_col, _ = _find_metric_pair(hist, names)
        if train_col or val_col:
            metric_specs.append((train_col, val_col, title))
    if not metric_specs:
        return None
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 4), squeeze=False)
    axes = axes[0]
    for ax, (train_metric, val_metric, title) in zip(axes, metric_specs):
        if train_metric and train_metric in hist.columns:
            ax.plot(hist["epoch"], hist[train_metric], marker="o", label=train_metric)
        if val_metric and val_metric in hist.columns:
            ax.plot(hist["epoch"], hist[val_metric], marker="o", label=val_metric)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training curves — {model_display(model)} / {variant_display(variant)}")
    fig.tight_layout()
    save_path = out_dir / f"{model}_{variant}_training_row.png"
    save_show_close_figure(fig, save_path=save_path, show=show)
    return save_path


def plot_all_training_histories(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    df = reorder_comparison_df(comparison_df)
    for _, row in df.iterrows():
        plot_training_history_for_row(row, out_dir=out_dir, show=show)


def plot_history_epoch_grid_by_variant(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False) -> None:
    """Variánsonként egy sor: 4 validációs metrika; modellek színes görbéi."""
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_variant")
    df = reorder_comparison_df(comparison_df)
    variants = sort_variants(df["data_variant"].dropna().unique())
    models = sort_models(df["model"].dropna().unique())
    mc = model_colors()
    for variant in variants:
        fig, axes = plt.subplots(1, len(VAL_METRIC_HISTORY_KEYS), figsize=(4.2 * len(VAL_METRIC_HISTORY_KEYS), 4.5), squeeze=False)
        row = axes[0]
        for ax, candidates, colname in zip(row, VAL_METRIC_HISTORY_KEYS, COMPARISON_METRIC_COLUMNS):
            ylab = METRIC_LABEL.get(colname, colname)
            plotted = False
            for m in models:
                rows = df[(df["data_variant"].astype(str) == variant) & (df["model"].astype(str) == m)]
                if rows.empty:
                    continue
                row0 = rows.iloc[0]
                model_dir = _infer_model_dir(row0)
                if model_dir is None:
                    continue
                hist = find_training_history(model_dir)
                if hist is None:
                    continue
                col = _first_hist_col(hist, candidates)
                if col is None:
                    continue
                x = hist["epoch"] if "epoch" in hist.columns else range(1, len(hist) + 1)
                ax.plot(x, hist[col], marker="o", markersize=2, label=model_display(m), color=mc.get(m))
                plotted = True
            if plotted:
                ax.legend(fontsize=7, loc="best")
            ax.set_title(ylab)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        row[0].set_ylabel("Érték")
        fig.suptitle(f"Epoch — modellek összehasonlítása ({variant_display(variant)})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_show_close_figure(fig, save_path=out_dir / f"{variant}_epoch_metrics_row.png", show=show)


def plot_history_epoch_grid_by_model(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False) -> None:
    """Modellenként egy sor: 4 validációs metrika; variánsok színes görbéi."""
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_model")
    df = reorder_comparison_df(comparison_df)
    variants = sort_variants(df["data_variant"].dropna().unique())
    models = sort_models(df["model"].dropna().unique())
    vc = variant_colors()
    for model in models:
        fig, axes = plt.subplots(1, len(VAL_METRIC_HISTORY_KEYS), figsize=(4.2 * len(VAL_METRIC_HISTORY_KEYS), 4.5), squeeze=False)
        row = axes[0]
        for ax, candidates, colname in zip(row, VAL_METRIC_HISTORY_KEYS, COMPARISON_METRIC_COLUMNS):
            ylab = METRIC_LABEL.get(colname, colname)
            plotted = False
            for v in variants:
                rows = df[(df["model"].astype(str) == model) & (df["data_variant"].astype(str) == v)]
                if rows.empty:
                    continue
                row0 = rows.iloc[0]
                model_dir = _infer_model_dir(row0)
                if model_dir is None:
                    continue
                hist = find_training_history(model_dir)
                if hist is None:
                    continue
                col = _first_hist_col(hist, candidates)
                if col is None:
                    continue
                x = hist["epoch"] if "epoch" in hist.columns else range(1, len(hist) + 1)
                ax.plot(x, hist[col], marker="o", markersize=2, label=variant_display(v), color=vc.get(v))
                plotted = True
            if plotted:
                ax.legend(fontsize=7, loc="best")
            ax.set_title(ylab)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        row[0].set_ylabel("Érték")
        fig.suptitle(f"Epoch — variánsok összehasonlítása ({model_display(model)})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        save_show_close_figure(fig, save_path=out_dir / f"{model}_epoch_metrics_row.png", show=show)


def plot_epoch_comparisons(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    plot_history_epoch_grid_by_variant(comparison_df, out_dir, show=show)
    plot_history_epoch_grid_by_model(comparison_df, out_dir, show=show)


def plot_report_best_models(
    best_by_variant: pd.DataFrame,
    best_by_model: pd.DataFrame,
    out_dir: str | Path,
    show: bool = False,
) -> tuple[Path | None, Path | None]:
    """A legjobb modell táblák grafikus kiegészítése (metrikák egy sorban)."""
    out_dir = ensure_dir(Path(out_dir) / "best_models_plots")
    p1 = p2 = None
    mc = model_colors()
    vc = variant_colors()

    if len(best_by_variant) > 0:
        b = best_by_variant.copy()
        variants = sort_variants(b["data_variant"].dropna().unique())
        b["_k"] = pd.Categorical(b["data_variant"].astype(str), categories=variants, ordered=True)
        b = b.sort_values("_k").drop(columns=["_k"])
        fig, axes = plt.subplots(1, len(COMPARISON_METRIC_COLUMNS), figsize=(4 * len(COMPARISON_METRIC_COLUMNS), 4.2), squeeze=False)
        row = axes[0]
        x = np.arange(len(b))
        for ax, metric in zip(row, COMPARISON_METRIC_COLUMNS):
            if metric not in b.columns:
                ax.set_visible(False)
                continue
            colors = [mc.get(str(row["model"]), "gray") for _, row in b.iterrows()]
            ax.bar(x, b[metric].tolist(), color=colors, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([variant_display(str(v)) for v in b["data_variant"]], rotation=0)
            ylab = METRIC_LABEL.get(metric, metric)
            ax.set_title(ylab)
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            if metric != "loss":
                ax.set_ylim(0, 1)
        fig.suptitle("Legjobb modell variánsonként (F1-makró szerint)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        p1 = out_dir / "best_by_variant_metrics_row.png"
        save_show_close_figure(fig, save_path=p1, show=show)

    if len(best_by_model) > 0:
        b = best_by_model.copy()
        models = sort_models(b["model"].dropna().unique())
        b["_k"] = pd.Categorical(b["model"].astype(str), categories=models, ordered=True)
        b = b.sort_values("_k").drop(columns=["_k"])
        fig, axes = plt.subplots(1, len(COMPARISON_METRIC_COLUMNS), figsize=(4 * len(COMPARISON_METRIC_COLUMNS), 4.2), squeeze=False)
        row = axes[0]
        x = np.arange(len(b))
        for ax, metric in zip(row, COMPARISON_METRIC_COLUMNS):
            if metric not in b.columns:
                ax.set_visible(False)
                continue
            colors = [vc.get(str(row["data_variant"]), "gray") for _, row in b.iterrows()]
            ax.bar(x, b[metric].tolist(), color=colors, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([model_display(str(m)) for m in b["model"]], rotation=35, ha="right")
            ylab = METRIC_LABEL.get(metric, metric)
            ax.set_title(ylab)
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
            if metric != "loss":
                ax.set_ylim(0, 1)
        fig.suptitle("Legjobb variáns modellenként (F1-makró szerint)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        p2 = out_dir / "best_by_model_metrics_row.png"
        save_show_close_figure(fig, save_path=p2, show=show)

    return p1, p2
