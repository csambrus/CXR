from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.compare_models import (
    plot_all_main_metrics,
    plot_all_training_histories,
    plot_epoch_comparisons,
    print_leaderboard,
    run_multiple_models,
)
from src.compare_models_plots import plot_report_best_models
from src.comparison_order import reorder_comparison_df
from src.config import MODELS_DIR, ensure_dir, save_json

INFO_SAVED = "[INFO] Saved:"


@dataclass(frozen=True)
class ModelRunConfig:
    label: str
    model_names: list[str]
    data_variants: list[str]
    pretrained: bool
    do_fine_tuning: bool
    epochs_head: int
    epochs_finetune: int
    learning_rate_head: float
    learning_rate_finetune: float
    comparison_name: str


def get_default_model_run_configs(data_variants: list[str]) -> list[ModelRunConfig]:
    return [
        ModelRunConfig(
            label="baseline_cnn_optimized",
            model_names=["baseline_cnn"],
            data_variants=data_variants,
            pretrained=False,
            do_fine_tuning=False,
            epochs_head=18,
            epochs_finetune=0,
            learning_rate_head=1e-3,
            learning_rate_finetune=1e-5,
            comparison_name="baseline_cnn_optimized",
        ),
        ModelRunConfig(
            label="efficientnetb0_fixed_optimized",
            model_names=["efficientnetb0"],
            data_variants=data_variants,
            pretrained=True,
            do_fine_tuning=True,
            epochs_head=12,
            epochs_finetune=12,
            learning_rate_head=3e-4,
            learning_rate_finetune=3e-6,
            comparison_name="efficientnetb0_fixed_optimized",
        ),
        ModelRunConfig(
            label="resnet50_optimized",
            model_names=["resnet50"],
            data_variants=data_variants,
            pretrained=True,
            do_fine_tuning=True,
            epochs_head=8,
            epochs_finetune=6,
            learning_rate_head=1e-3,
            learning_rate_finetune=1e-5,
            comparison_name="resnet50_optimized",
        ),
        ModelRunConfig(
            label="vgg16_optimized",
            model_names=["vgg16"],
            data_variants=data_variants,
            pretrained=True,
            do_fine_tuning=True,
            epochs_head=10,
            epochs_finetune=6,
            learning_rate_head=5e-4,
            learning_rate_finetune=1e-5,
            comparison_name="vgg16_optimized",
        ),
    ]


def run_training_pipeline(
    *,
    split_dir: str | Path,
    run_configs: list[ModelRunConfig],
    run_full_retrain: bool = False,
    models_dir: str | Path = MODELS_DIR,
) -> pd.DataFrame:
    result_frames: list[pd.DataFrame] = []

    for cfg in run_configs:
        print("\n" + "=" * 90)
        print("RUN CONFIG:", cfg.label)
        print("=" * 90)

        df = run_multiple_models(
            split_dir=split_dir,
            out_dir=models_dir,
            model_names=cfg.model_names,
            data_variants=cfg.data_variants,
            pretrained=cfg.pretrained,
            do_fine_tuning=cfg.do_fine_tuning,
            epochs_head=cfg.epochs_head,
            epochs_finetune=cfg.epochs_finetune,
            learning_rate_head=cfg.learning_rate_head,
            learning_rate_finetune=cfg.learning_rate_finetune,
            comparison_name=cfg.comparison_name,
            make_plots=False,
            show_plots=False,
            skip_if_complete=not run_full_retrain,
            trust_existing_without_fingerprint=True,
        )

        if "model" not in df.columns:
            df["model"] = cfg.model_names[0]

        df["run_config"] = cfg.label
        result_frames.append(df)

    if len(result_frames) == 0:
        raise RuntimeError("No model results were produced.")

    combined = pd.concat(result_frames, ignore_index=True)
    return combined


def build_final_comparison(
    *,
    model_results_df: pd.DataFrame,
    final_comparison_name: str,
    models_dir: str | Path = MODELS_DIR,
) -> tuple[pd.DataFrame, Path]:
    final_out_dir = ensure_dir(Path(models_dir) / final_comparison_name)

    comparison_df = (
        model_results_df
        .drop_duplicates(subset=["model", "data_variant"], keep="last")
    )
    comparison_df = reorder_comparison_df(comparison_df)

    comparison_csv = final_out_dir / "comparison.csv"
    leaderboard_csv = final_out_dir / "leaderboard.csv"
    comparison_json = final_out_dir / "comparison.json"

    comparison_df.to_csv(comparison_csv, index=False)
    comparison_df.to_csv(leaderboard_csv, index=False)
    save_json({"rows": comparison_df.to_dict(orient="records")}, comparison_json)

    print(INFO_SAVED, comparison_csv)
    print(INFO_SAVED, leaderboard_csv)
    print(INFO_SAVED, comparison_json)

    return comparison_df, final_out_dir


def save_pipeline_config(
    *,
    run_configs: list[ModelRunConfig],
    final_out_dir: str | Path,
    run_full_retrain: bool,
    data_variants: list[str],
) -> Path:
    final_out_dir = ensure_dir(final_out_dir)
    save_path = Path(final_out_dir) / "pipeline_config.json"

    payload: dict[str, Any] = {
        "run_full_retrain": bool(run_full_retrain),
        "data_variants": list(data_variants),
        "run_configs": [asdict(cfg) for cfg in run_configs],
    }
    save_json(payload, save_path)
    return save_path


def generate_final_plots(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = True) -> None:
    plot_all_main_metrics(comparison_df, out_dir, show=show)
    plot_all_training_histories(comparison_df, out_dir, show=show)
    plot_epoch_comparisons(comparison_df, out_dir, show=show)


def report_best_models(
    comparison_df: pd.DataFrame,
    out_dir: str | Path,
    *,
    show_plots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = ensure_dir(out_dir)

    best_by_variant = (
        comparison_df.sort_values("f1_macro", ascending=False)
        .groupby("data_variant", as_index=False)
        .first()
    )
    best_by_model = (
        comparison_df.sort_values("f1_macro", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )
    best_by_variant = reorder_comparison_df(best_by_variant)
    best_by_model = reorder_comparison_df(best_by_model)

    best_by_variant.to_csv(Path(out_dir) / "best_by_variant.csv", index=False)
    best_by_model.to_csv(Path(out_dir) / "best_by_model.csv", index=False)

    p1, p2 = plot_report_best_models(best_by_variant, best_by_model, out_dir, show=show_plots)
    if p1 is not None:
        print(INFO_SAVED, p1)
    if p2 is not None:
        print(INFO_SAVED, p2)

    return best_by_variant, best_by_model


def print_pipeline_leaderboard(comparison_df: pd.DataFrame) -> None:
    print_leaderboard(comparison_df)
