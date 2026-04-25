from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    MODELS_DIR,
    OUTPUT_DIR,
    PLOT_DPI,
    ensure_dir,
    save_json,
)
from src.evaluate import run_evaluation
from src.train import run_training


# =========================================================
# Helpers
# =========================================================

def _normalize_model_names(model_names: str | Iterable[str]) -> list[str]:
    if isinstance(model_names, str):
        return [model_names]
    return list(model_names)


def _normalize_variants(data_variants: str | Iterable[str]) -> list[str]:
    if isinstance(data_variants, str):
        return [data_variants]
    return list(data_variants)


def _safe_metric(metrics: dict[str, Any], key: str, default: float | None = None):
    return metrics.get(key, default)


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _model_variant_label(model: str, variant: str) -> str:
    return f"{model}_{variant}"


def _candidate_history_paths(model_dir: Path) -> list[Path]:
    return [
        model_dir / "history.csv",
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
        print(f"[WARN] Nem sikerült history-t olvasni: {path} | {e}")
        return None

    if len(df) == 0:
        return None

    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))

    return df


def find_training_history(model_dir: str | Path) -> pd.DataFrame | None:
    model_dir = Path(model_dir)

    for path in _candidate_history_paths(model_dir):
        df = _read_history_file(path)
        if df is not None:
            return df

    candidates = sorted(model_dir.rglob("*history*.csv")) + sorted(model_dir.rglob("*history*.json"))
    for path in candidates:
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


# =========================================================
# Plot helpers: summary metrics
# =========================================================

def plot_metric_bars(
    comparison_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")

    df = comparison_df.copy()
    df["label"] = df["model"].astype(str) + "\n(" + df["data_variant"].astype(str) + ")"
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["label"], df[metric])

    ax.set_title(title or f"Model comparison - {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model / variant")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_metric_by_variant(
    comparison_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")

    pivot_df = comparison_df.pivot(
        index="model",
        columns="data_variant",
        values=metric,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_df.plot(kind="bar", ax=ax)

    ax.set_title(title or f"{metric} by model and data variant")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="data_variant")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    plt.close(fig)


def plot_models_within_each_variant(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    metrics: Iterable[str] | None = None,
):
    out_dir = ensure_dir(Path(out_dir) / "models_within_variants")

    if metrics is None:
        metrics = ["accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr", "loss"]

    for variant, sub in comparison_df.groupby("data_variant"):
        for metric in metrics:
            if metric not in sub.columns:
                continue

            ascending = metric == "loss"
            df = sub.sort_values(metric, ascending=ascending)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(df["model"], df[metric])

            ax.set_title(f"Modellek összehasonlítása variánson belül\nvariant={variant} | metric={metric}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")

            if metric != "loss":
                ax.set_ylim(0, 1)

            fig.tight_layout()
            save_path = out_dir / f"{variant}_{metric}.png"
            fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)

            print(f"[INFO] Saved: {save_path}")


def plot_variants_within_each_model(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    metrics: Iterable[str] | None = None,
):
    out_dir = ensure_dir(Path(out_dir) / "variants_within_models")

    if metrics is None:
        metrics = ["accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr", "loss"]

    for model, sub in comparison_df.groupby("model"):
        for metric in metrics:
            if metric not in sub.columns:
                continue

            ascending = metric == "loss"
            df = sub.sort_values(metric, ascending=ascending)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(df["data_variant"], df[metric])

            ax.set_title(f"Variánsok összehasonlítása modellen belül\nmodel={model} | metric={metric}")
            ax.set_xlabel("Data variant")
            ax.set_ylabel(metric)
            ax.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")

            if metric != "loss":
                ax.set_ylim(0, 1)

            fig.tight_layout()
            save_path = out_dir / f"{model}_{metric}.png"
            fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close(fig)

            print(f"[INFO] Saved: {save_path}")


def plot_metric_heatmaps(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    metrics: Iterable[str] | None = None,
):
    out_dir = ensure_dir(Path(out_dir) / "heatmaps")

    if metrics is None:
        metrics = ["accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr", "loss"]

    for metric in metrics:
        if metric not in comparison_df.columns:
            continue

        pivot_df = comparison_df.pivot(
            index="model",
            columns="data_variant",
            values=metric,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot_df.values, aspect="auto")

        ax.set_title(f"Model × variáns heatmap - {metric}")
        ax.set_xlabel("Data variant")
        ax.set_ylabel("Model")

        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index)

        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                val = pivot_df.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()

        save_path = out_dir / f"heatmap_{metric}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved: {save_path}")


def plot_all_main_metrics(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
):
    out_dir = ensure_dir(out_dir)

    metrics_to_plot = [
        "accuracy",
        "recall_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
        "loss",
    ]

    for metric in metrics_to_plot:
        if metric not in comparison_df.columns:
            continue

        plot_metric_bars(
            comparison_df=comparison_df,
            metric=metric,
            save_path=Path(out_dir) / f"bar_{metric}.png",
            title=f"Comparison - {metric}",
        )

        plot_metric_by_variant(
            comparison_df=comparison_df,
            metric=metric,
            save_path=Path(out_dir) / f"grouped_{metric}.png",
            title=f"{metric} by model / variant",
        )

    plot_models_within_each_variant(comparison_df, out_dir)
    plot_variants_within_each_model(comparison_df, out_dir)
    plot_metric_heatmaps(comparison_df, out_dir)


# =========================================================
# Plot helpers: epoch history
# =========================================================

def plot_training_history_for_row(
    row: pd.Series,
    out_dir: str | Path,
):
    model = str(row["model"])
    variant = str(row["data_variant"])
    model_dir = _infer_model_dir(row)

    if model_dir is None:
        print(f"[WARN] Nem található model_dir: {model} / {variant}")
        return

    hist = find_training_history(model_dir)
    if hist is None:
        print(f"[WARN] Nincs training history: {model_dir}")
        return

    out_dir = ensure_dir(Path(out_dir) / "training_curves")

    metric_pairs = [
        ("loss", "val_loss"),
        ("accuracy", "val_accuracy"),
        ("recall", "val_recall"),
        ("auc", "val_auc"),
    ]

    for train_metric, val_metric in metric_pairs:
        if train_metric not in hist.columns and val_metric not in hist.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        if train_metric in hist.columns:
            ax.plot(hist["epoch"], hist[train_metric], marker="o", label=train_metric)

        if val_metric in hist.columns:
            ax.plot(hist["epoch"], hist[val_metric], marker="o", label=val_metric)

        ax.set_title(f"Epochonkénti fejlődés\n{model} / {variant} / {train_metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(train_metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        save_path = out_dir / f"{model}_{variant}_{train_metric}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved: {save_path}")


def plot_all_training_histories(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
):
    for _, row in comparison_df.iterrows():
        plot_training_history_for_row(row, out_dir=out_dir)


def plot_history_comparison_by_variant(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str = "val_accuracy",
):
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_variant")

    for variant, sub in comparison_df.groupby("data_variant"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for _, row in sub.iterrows():
            model = str(row["model"])
            model_dir = _infer_model_dir(row)
            if model_dir is None:
                continue

            hist = find_training_history(model_dir)
            if hist is None or metric not in hist.columns:
                continue

            ax.plot(hist["epoch"], hist[metric], marker="o", label=model)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(f"Modellek epochonkénti összehasonlítása\nvariant={variant} | metric={metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        save_path = out_dir / f"{variant}_{metric}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved: {save_path}")


def plot_history_comparison_by_model(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    metric: str = "val_accuracy",
):
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_model")

    for model, sub in comparison_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False

        for _, row in sub.iterrows():
            variant = str(row["data_variant"])
            model_dir = _infer_model_dir(row)
            if model_dir is None:
                continue

            hist = find_training_history(model_dir)
            if hist is None or metric not in hist.columns:
                continue

            ax.plot(hist["epoch"], hist[metric], marker="o", label=variant)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(f"Variánsok epochonkénti összehasonlítása\nmodel={model} | metric={metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        save_path = out_dir / f"{model}_{metric}.png"
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        print(f"[INFO] Saved: {save_path}")


def plot_epoch_comparisons(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
):
    for metric in ["val_accuracy", "val_loss", "accuracy", "loss"]:
        plot_history_comparison_by_variant(comparison_df, out_dir, metric=metric)
        plot_history_comparison_by_model(comparison_df, out_dir, metric=metric)


# =========================================================
# Core comparison logic
# =========================================================

def compare_existing_results(
    result_summaries: list[dict[str, Any]],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison",
    make_plots: bool = True,
) -> pd.DataFrame:
    out_dir = ensure_dir(Path(out_dir) / comparison_name)

    rows: list[dict[str, Any]] = []

    for item in result_summaries:
        train_summary = item.get("train_summary", {})
        eval_summary = item.get("eval_summary", {})
        metrics = eval_summary.get("metrics", {})

        model = eval_summary.get("model_name", train_summary.get("model_name"))
        data_variant = eval_summary.get("data_variant", train_summary.get("data_variant", "raw"))

        row = {
            "model": model,
            "data_variant": data_variant,
            "model_path": eval_summary.get("model_path", train_summary.get("best_model_path")),
            "data_root": eval_summary.get("data_root", train_summary.get("data_root")),
            "out_dir": eval_summary.get("out_dir", train_summary.get("out_dir")),
            "loss": _safe_float(_safe_metric(metrics, "loss")),
            "accuracy": _safe_float(_safe_metric(metrics, "accuracy")),
            "recall_macro": _safe_float(_safe_metric(metrics, "recall_macro")),
            "f1_macro": _safe_float(_safe_metric(metrics, "f1_macro")),
            "roc_auc_macro_ovr": _safe_float(_safe_metric(metrics, "roc_auc_macro_ovr")),
        }

        row["model_variant"] = _model_variant_label(str(model), str(data_variant))
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) == 0:
        raise ValueError("No results found to compare.")

    sort_cols = [c for c in ["f1_macro", "accuracy", "recall_macro"] if c in comparison_df.columns]
    comparison_df = comparison_df.sort_values(
        by=sort_cols,
        ascending=False,
    ).reset_index(drop=True)

    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json(
        {"rows": comparison_df.to_dict(orient="records")},
        out_dir / "comparison.json",
    )

    leaderboard_cols = [
        "model_variant",
        "model",
        "data_variant",
        "accuracy",
        "f1_macro",
        "recall_macro",
        "roc_auc_macro_ovr",
        "loss",
    ]
    leaderboard_cols = [c for c in leaderboard_cols if c in comparison_df.columns]
    leaderboard_df = comparison_df[leaderboard_cols].copy()
    leaderboard_df.to_csv(out_dir / "leaderboard.csv", index=False)

    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir)
        plot_all_training_histories(comparison_df, out_dir)
        plot_epoch_comparisons(comparison_df, out_dir)

    return comparison_df


def run_multiple_models(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_names: str | Iterable[str] = ("baseline_cnn", "resnet50", "vgg16", "efficientnetb0"),
    data_variants: str | Iterable[str] = ("raw",),
    pretrained: bool = True,
    do_fine_tuning: bool = False,
    epochs_head: int = 8,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
    comparison_name: str = "comparison",
    make_plots: bool = True,
) -> pd.DataFrame:
    model_names = _normalize_model_names(model_names)
    data_variants = _normalize_variants(data_variants)

    all_results: list[dict[str, Any]] = []

    print("=" * 72)
    print("RUN MULTIPLE MODELS")
    print("=" * 72)
    print("model_names   :", model_names)
    print("data_variants :", data_variants)
    print("comparison    :", comparison_name)
    print("=" * 72)

    for model_name in model_names:
        for data_variant in data_variants:
            print("\n" + "-" * 72)
            print(f"Running model={model_name} | data_variant={data_variant}")
            print("-" * 72)

            train_summary = run_training(
                split_dir=split_dir,
                out_dir=out_dir,
                model_name=model_name,
                pretrained=pretrained,
                do_fine_tuning=do_fine_tuning,
                epochs_head=epochs_head,
                epochs_finetune=epochs_finetune,
                learning_rate_head=learning_rate_head,
                learning_rate_finetune=learning_rate_finetune,
                data_variant=data_variant,
            )

            eval_summary = run_evaluation(
                model_path=train_summary["best_model_path"],
                split_dir=split_dir,
                out_dir=out_dir,
                model_name=model_name,
                data_variant=data_variant,
            )

            all_results.append(
                {
                    "train_summary": train_summary,
                    "eval_summary": eval_summary,
                }
            )

    comparison_df = compare_existing_results(
        result_summaries=all_results,
        out_dir=out_dir,
        comparison_name=comparison_name,
        make_plots=make_plots,
    )

    return comparison_df


# =========================================================
# Loading already finished experiments
# =========================================================

def load_metrics_from_model_dirs(
    model_dirs: Iterable[str | Path],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison_loaded",
    make_plots: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        metrics_path = model_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"[WARN] Missing metrics.json: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        model = data.get("model_name")
        data_variant = data.get("data_variant", "raw")

        row = {
            "model": model,
            "data_variant": data_variant,
            "model_path": data.get("model_path"),
            "data_root": data.get("data_root"),
            "out_dir": data.get("out_dir", str(model_dir)),
            "loss": _safe_float(_safe_metric(metrics, "loss")),
            "accuracy": _safe_float(_safe_metric(metrics, "accuracy")),
            "recall_macro": _safe_float(_safe_metric(metrics, "recall_macro")),
            "f1_macro": _safe_float(_safe_metric(metrics, "f1_macro")),
            "roc_auc_macro_ovr": _safe_float(_safe_metric(metrics, "roc_auc_macro_ovr")),
        }
        row["model_variant"] = _model_variant_label(str(model), str(data_variant))
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    if len(comparison_df) == 0:
        raise ValueError("No valid metrics.json files found.")

    sort_cols = [c for c in ["f1_macro", "accuracy", "recall_macro"] if c in comparison_df.columns]
    comparison_df = comparison_df.sort_values(
        by=sort_cols,
        ascending=False,
    ).reset_index(drop=True)

    out_dir = ensure_dir(Path(out_dir) / comparison_name)

    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json(
        {"rows": comparison_df.to_dict(orient="records")},
        out_dir / "comparison.json",
    )

    leaderboard_cols = [
        "model_variant",
        "model",
        "data_variant",
        "accuracy",
        "f1_macro",
        "recall_macro",
        "roc_auc_macro_ovr",
        "loss",
    ]
    leaderboard_cols = [c for c in leaderboard_cols if c in comparison_df.columns]
    comparison_df[leaderboard_cols].to_csv(out_dir / "leaderboard.csv", index=False)

    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir)
        plot_all_training_histories(comparison_df, out_dir)
        plot_epoch_comparisons(comparison_df, out_dir)

    return comparison_df


def load_metrics_from_comparison_csv(
    comparison_csv: str | Path,
    out_dir: str | Path = OUTPUT_DIR / "model_comparison_loaded",
    make_plots: bool = True,
) -> pd.DataFrame:
    comparison_csv = Path(comparison_csv)
    comparison_df = pd.read_csv(comparison_csv)

    if "model" not in comparison_df.columns and "model_name" in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"model_name": "model"})

    if "model_variant" not in comparison_df.columns:
        comparison_df["model_variant"] = (
            comparison_df["model"].astype(str) + "_" + comparison_df["data_variant"].astype(str)
        )

    out_dir = ensure_dir(out_dir)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)

    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir)
        plot_all_training_histories(comparison_df, out_dir)
        plot_epoch_comparisons(comparison_df, out_dir)

    return comparison_df


# =========================================================
# Convenience report
# =========================================================

def print_leaderboard(comparison_df: pd.DataFrame, top_k: int | None = None) -> None:
    df = comparison_df.copy()

    if top_k is not None:
        df = df.head(top_k)

    cols = [
        "model_variant",
        "accuracy",
        "f1_macro",
        "recall_macro",
        "roc_auc_macro_ovr",
        "loss",
    ]
    cols = [c for c in cols if c in df.columns]

    print("\nLeaderboard")
    print("-" * 100)
    print(df[cols].to_string(index=False))
    print("-" * 100)