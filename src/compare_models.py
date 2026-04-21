from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    MODELS_DIR,
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
    value = metrics.get(key, default)
    return value


# =========================================================
# Plot helpers
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
    df["label"] = df["model"] + "\n(" + df["data_variant"] + ")"
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["label"], df[metric])

    ax.set_title(title or f"Model comparison - {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model / Variant")
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


def plot_all_main_metrics(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

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
            save_path=out_dir / f"bar_{metric}.png",
            title=f"Comparison - {metric}",
        )

        plot_metric_by_variant(
            comparison_df=comparison_df,
            metric=metric,
            save_path=out_dir / f"grouped_{metric}.png",
            title=f"{metric} by model / variant",
        )


# =========================================================
# Core comparison logic
# =========================================================

def compare_existing_results(
    result_summaries: list[dict[str, Any]],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison",
) -> pd.DataFrame:
    out_dir = ensure_dir(Path(out_dir) / comparison_name)

    rows: list[dict[str, Any]] = []

    for item in result_summaries:
        train_summary = item.get("train_summary", {})
        eval_summary = item.get("eval_summary", {})
        metrics = eval_summary.get("metrics", {})

        row = {
            "model": eval_summary.get("model_name", train_summary.get("model_name")),
            "data_variant": eval_summary.get("data_variant", train_summary.get("data_variant")),
            "model_path": eval_summary.get("model_path", train_summary.get("best_model_path")),
            "data_root": eval_summary.get("data_root", train_summary.get("data_root")),
            "out_dir": eval_summary.get("out_dir", train_summary.get("out_dir")),
            "loss": _safe_metric(metrics, "loss"),
            "accuracy": _safe_metric(metrics, "accuracy"),
            "recall_macro": _safe_metric(metrics, "recall_macro"),
            "f1_macro": _safe_metric(metrics, "f1_macro"),
            "roc_auc_macro_ovr": _safe_metric(metrics, "roc_auc_macro_ovr"),
        }
        row["model_variant"] = f"{row['model']}_{row['data_variant']}"
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) == 0:
        raise ValueError("No results found to compare.")

    comparison_df = comparison_df.sort_values(
        by=["accuracy", "f1_macro", "recall_macro"],
        ascending=False,
    ).reset_index(drop=True)

    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json(
        {"rows": comparison_df.to_dict(orient="records")},
        out_dir / "comparison.json",
    )

    plot_all_main_metrics(comparison_df, out_dir)

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
    leaderboard_df = comparison_df[leaderboard_cols].copy()
    leaderboard_df.to_csv(out_dir / "leaderboard.csv", index=False)

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
    )

    return comparison_df


# =========================================================
# Loading already finished experiments
# =========================================================

def load_metrics_from_model_dirs(
    model_dirs: Iterable[str | Path],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison_loaded",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        model_dir = Path(model_dir)
        metrics_path = model_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"[WARN] Missing metrics.json: {metrics_path}")
            continue

        import json
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        row = {
            "model": data.get("model_name"),
            "data_variant": data.get("data_variant"),
            "model_path": data.get("model_path"),
            "data_root": data.get("data_root"),
            "out_dir": data.get("out_dir"),
            "loss": _safe_metric(metrics, "loss"),
            "accuracy": _safe_metric(metrics, "accuracy"),
            "recall_macro": _safe_metric(metrics, "recall_macro"),
            "f1_macro": _safe_metric(metrics, "f1_macro"),
            "roc_auc_macro_ovr": _safe_metric(metrics, "roc_auc_macro_ovr"),
        }
        row["model_variant"] = f"{row['model']}_{row['data_variant']}"
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    if len(comparison_df) == 0:
        raise ValueError("No valid metrics.json files found.")

    comparison_df = comparison_df.sort_values(
        by=["accuracy", "f1_macro", "recall_macro"],
        ascending=False,
    ).reset_index(drop=True)

    out_dir = ensure_dir(Path(out_dir) / comparison_name)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json(
        {"rows": comparison_df.to_dict(orient="records")},
        out_dir / "comparison.json",
    )

    plot_all_main_metrics(comparison_df, out_dir)

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
    comparison_df[leaderboard_cols].to_csv(out_dir / "leaderboard.csv", index=False)

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
