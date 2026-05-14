from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.compare_models_plots import (
    plot_all_main_metrics,
    plot_all_training_histories,
    plot_epoch_comparisons,
    plot_training_history_for_row,
)
from src.comparison_order import COMPARISON_METRIC_COLUMNS, leaderboard_display_columns, METRIC_LABEL, reorder_comparison_df
from src.config import MODELS_DIR, OUTPUT_DIR, ensure_dir, save_json
from src.evaluate import run_evaluation
from src.plot_utils import display_png_if_available
from src.train import run_training

DEFAULT_METRICS = list(COMPARISON_METRIC_COLUMNS)


def _normalize_to_list(values: str | Iterable[str]) -> list[str]:
    return [values] if isinstance(values, str) else list(values)


def _safe_metric(metrics: dict[str, Any], key: str, default: float | None = None):
    return metrics.get(key, default)


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def split_fingerprint(split_dir: str | Path) -> dict[str, Any]:
    split_dir = Path(split_dir)
    files = ["train.csv", "val.csv", "test.csv"]
    items: dict[str, Any] = {}
    h = hashlib.sha256()
    for name in files:
        path = split_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        file_hash = _hash_file(path)
        items[name] = {"path": str(path), "size": int(path.stat().st_size), "sha256": file_hash}
        h.update(name.encode("utf-8"))
        h.update(file_hash.encode("utf-8"))
    return {"split_dir": str(split_dir), "files": items, "sha256": h.hexdigest()}


def _candidate_model_dirs(out_dir: str | Path, model_name: str, data_variant: str) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        out_dir / f"{model_name}_{data_variant}",
        out_dir / data_variant / model_name,
        out_dir / model_name / data_variant,
        out_dir / model_name,
    ]


def _candidate_model_paths(model_dir: Path) -> list[Path]:
    return [model_dir / "best_model.keras", model_dir / "final_model.keras", model_dir / "last_model.keras", model_dir / "model.keras"]


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] JSON nem olvashato: {path} | {e}")
        return None


def _write_run_metadata(model_dir: str | Path, metadata: dict[str, Any]) -> None:
    save_json(metadata, ensure_dir(model_dir) / "run_metadata.json")


def _find_existing_run(*, model_name: str, data_variant: str, split_dir: str | Path, out_dir: str | Path, trust_existing_without_fingerprint: bool = True) -> dict[str, Any] | None:
    current_fp = split_fingerprint(split_dir)
    current_hash = current_fp["sha256"]
    for model_dir in _candidate_model_dirs(out_dir, model_name, data_variant):
        model_path = next((p for p in _candidate_model_paths(model_dir) if p.exists()), None)
        metrics_path = model_dir / "metrics.json"
        metadata_path = model_dir / "run_metadata.json"
        if model_path is None or not metrics_path.exists():
            continue
        metrics_json = _read_json_if_exists(metrics_path)
        if metrics_json is None:
            continue
        metadata = _read_json_if_exists(metadata_path) or {}
        stored_hash = metadata.get("split_fingerprint", {}).get("sha256") or metrics_json.get("split_fingerprint", {}).get("sha256")
        if stored_hash is not None and stored_hash != current_hash:
            continue
        if stored_hash is None and not trust_existing_without_fingerprint:
            continue
        metrics = metrics_json.get("metrics", metrics_json)
        eval_summary = {
            "model_name": metrics_json.get("model_name", model_name),
            "data_variant": metrics_json.get("data_variant", data_variant),
            "model_path": str(metrics_json.get("model_path", model_path)),
            "out_dir": str(metrics_json.get("out_dir", model_dir)),
            "data_root": metrics_json.get("data_root"),
            "metrics": metrics,
            "split_fingerprint": current_fp,
            "reused_existing": True,
        }
        train_summary = {
            "model_name": model_name,
            "data_variant": data_variant,
            "best_model_path": str(model_path),
            "out_dir": str(model_dir),
            "data_root": metrics_json.get("data_root"),
            "split_fingerprint": current_fp,
            "reused_existing": True,
        }
        if stored_hash is None:
            _write_run_metadata(
                model_dir,
                {
                    "model_name": model_name,
                    "data_variant": data_variant,
                    "split_fingerprint": current_fp,
                    "model_path": str(model_path),
                    "metrics_path": str(metrics_path),
                    "reused_existing_without_previous_fingerprint": True,
                },
            )
        return {"train_summary": train_summary, "eval_summary": eval_summary}
    return None


def _model_variant_label(model: str, variant: str) -> str:
    return f"{model}_{variant}"


def _evaluation_png_from_summary(result_item: dict[str, Any]) -> Path | None:
    eval_out_dir = result_item.get("eval_summary", {}).get("out_dir")
    if eval_out_dir is None:
        return None
    path = Path(eval_out_dir) / "evaluation_row.png"
    return path if path.exists() else None


# Metrikák, amik a futásonkénti history-sor plotoláshoz kellenek (ugyanaz, mint a leaderboard).
_RUN_PLOT_METRICS = ("accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr", "loss")


def _finalize_new_run_and_save_metadata(
    split_dir: str | Path,
    model_name: str,
    data_variant: str,
    train_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    *,
    pretrained: bool,
    do_fine_tuning: bool,
    epochs_head: int,
    epochs_finetune: int,
    learning_rate_head: float,
    learning_rate_finetune: float,
) -> dict[str, Any]:
    """
    Új train+eval után: split ujjlenyomat, run_metadata.json mentés,
    majd mindkét summary dict-be beírja a fingerprintet (régi viselkedés).
    """
    fp = split_fingerprint(split_dir)
    model_dir = Path(eval_summary.get("out_dir", Path(train_summary["best_model_path"]).parent))
    _write_run_metadata(
        model_dir,
        {
            "model_name": model_name,
            "data_variant": data_variant,
            "split_fingerprint": fp,
            "training": {
                "pretrained": pretrained,
                "do_fine_tuning": do_fine_tuning,
                "epochs_head": epochs_head,
                "epochs_finetune": epochs_finetune,
                "learning_rate_head": learning_rate_head,
                "learning_rate_finetune": learning_rate_finetune,
            },
            "model_path": train_summary.get("best_model_path"),
            "eval_out_dir": eval_summary.get("out_dir"),
        },
    )
    train_summary["split_fingerprint"] = fp
    eval_summary["split_fingerprint"] = fp
    return {"train_summary": train_summary, "eval_summary": eval_summary}


def _series_for_run_plots(model_name: str, data_variant: str, item: dict[str, Any]) -> pd.Series:
    """Egy sor a plot_training_history_for_row számára (model / out_dir / metrikák)."""
    te = item["train_summary"]
    ev = item["eval_summary"]
    m = ev.get("metrics", {})
    return pd.Series(
        {
            "model": model_name,
            "data_variant": data_variant,
            "model_path": ev.get("model_path", te.get("best_model_path")),
            "out_dir": ev.get("out_dir", te.get("out_dir")),
            **{k: _safe_float(m.get(k)) for k in _RUN_PLOT_METRICS},
        }
    )


def _plot_and_maybe_display_eval_png(
    model_name: str,
    data_variant: str,
    item: dict[str, Any],
    *,
    comparison_out: Path,
    show_plots: bool,
) -> None:
    """Futásonként: training görbe sor + opcionális eval PNG notebookban."""
    plot_training_history_for_row(
        _series_for_run_plots(model_name, data_variant, item),
        out_dir=comparison_out,
        show=show_plots,
    )
    if show_plots and (png := _evaluation_png_from_summary(item)):
        display_png_if_available(png)


def compare_existing_results(
    result_summaries: list[dict[str, Any]],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison",
    make_plots: bool = True,
    show_plots: bool = True,
) -> pd.DataFrame:
    out_dir = ensure_dir(Path(out_dir) / comparison_name)
    rows: list[dict[str, Any]] = []
    for item in result_summaries:
        train_summary = item.get("train_summary", {})
        eval_summary = item.get("eval_summary", {})
        metrics = eval_summary.get("metrics", {})
        model = eval_summary.get("model_name", train_summary.get("model_name"))
        data_variant = eval_summary.get("data_variant", train_summary.get("data_variant", "raw"))
        rows.append(
            {
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
                "model_variant": _model_variant_label(str(model), str(data_variant)),
            }
        )
    comparison_df = pd.DataFrame(rows)
    if len(comparison_df) == 0:
        raise ValueError("No results found to compare.")
    comparison_df = comparison_df.sort_values(by=[c for c in ["f1_macro", "accuracy", "recall_macro"] if c in comparison_df.columns], ascending=False).reset_index(drop=True)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json({"rows": comparison_df.to_dict(orient="records")}, out_dir / "comparison.json")
    leaderboard_cols = leaderboard_display_columns(comparison_df)
    comparison_df[leaderboard_cols].to_csv(out_dir / "leaderboard.csv", index=False)
    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir, show=show_plots)
        plot_all_training_histories(comparison_df, out_dir, show=show_plots)
        plot_epoch_comparisons(comparison_df, out_dir, show=show_plots)
    return comparison_df


def run_multiple_models(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_names: str | Iterable[str] = ("baseline_cnn", "efficientnetb0", "resnet50", "vgg16"),
    data_variants: str | Iterable[str] = ("raw",),
    pretrained: bool = True,
    do_fine_tuning: bool = False,
    epochs_head: int = 8,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
    comparison_name: str = "comparison",
    make_plots: bool = True,
    show_plots: bool = True,
    show_each_run: bool = True,
    skip_if_complete: bool = True,
    trust_existing_without_fingerprint: bool = True,
) -> pd.DataFrame:
    model_names = _normalize_to_list(model_names)
    data_variants = _normalize_to_list(data_variants)
    comparison_out = Path(out_dir) / comparison_name
    print(
        f"[BATCH] run_multiple_models | modellek={model_names} | variansok={data_variants} | "
        f"skip_if_complete={skip_if_complete} | comparison={comparison_name}"
    )
    all_results: list[dict[str, Any]] = []
    for model_name in model_names:
        for data_variant in data_variants:
            print(f"[RUN] {model_name} | data_variant={data_variant}")
            existing_result = (
                _find_existing_run(
                    model_name=model_name,
                    data_variant=data_variant,
                    split_dir=split_dir,
                    out_dir=out_dir,
                    trust_existing_without_fingerprint=trust_existing_without_fingerprint,
                )
                if skip_if_complete
                else None
            )
            if existing_result is not None:
                result_item = existing_result
                print("[SKIP] Meglévő eredmény újrafelhasználva.")
            else:
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
                result_item = _finalize_new_run_and_save_metadata(
                    split_dir,
                    model_name,
                    data_variant,
                    train_summary,
                    eval_summary,
                    pretrained=pretrained,
                    do_fine_tuning=do_fine_tuning,
                    epochs_head=epochs_head,
                    epochs_finetune=epochs_finetune,
                    learning_rate_head=learning_rate_head,
                    learning_rate_finetune=learning_rate_finetune,
                )
            all_results.append(result_item)
            if show_each_run:
                _plot_and_maybe_display_eval_png(
                    model_name,
                    data_variant,
                    result_item,
                    comparison_out=comparison_out,
                    show_plots=show_plots,
                )
    return compare_existing_results(
        result_summaries=all_results,
        out_dir=out_dir,
        comparison_name=comparison_name,
        make_plots=make_plots,
        show_plots=show_plots,
    )


def load_metrics_from_model_dirs(
    model_dirs: Iterable[str | Path],
    out_dir: str | Path = MODELS_DIR,
    comparison_name: str = "comparison_loaded",
    make_plots: bool = True,
    show_plots: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_dir in model_dirs:
        metrics_path = Path(model_dir) / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        model = data.get("model_name")
        data_variant = data.get("data_variant", "raw")
        rows.append(
            {
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
                "model_variant": _model_variant_label(str(model), str(data_variant)),
            }
        )
    comparison_df = pd.DataFrame(rows)
    if len(comparison_df) == 0:
        raise ValueError("No valid metrics.json files found.")
    comparison_df = comparison_df.sort_values(by=[c for c in ["f1_macro", "accuracy", "recall_macro"] if c in comparison_df.columns], ascending=False).reset_index(drop=True)
    out_dir = ensure_dir(Path(out_dir) / comparison_name)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    save_json({"rows": comparison_df.to_dict(orient="records")}, out_dir / "comparison.json")
    leaderboard_cols = leaderboard_display_columns(comparison_df)
    comparison_df[leaderboard_cols].to_csv(out_dir / "leaderboard.csv", index=False)
    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir, show=show_plots)
        plot_all_training_histories(comparison_df, out_dir, show=show_plots)
        plot_epoch_comparisons(comparison_df, out_dir, show=show_plots)
    return comparison_df


def load_metrics_from_comparison_csv(
    comparison_csv: str | Path,
    out_dir: str | Path = OUTPUT_DIR / "model_comparison_loaded",
    make_plots: bool = True,
    show_plots: bool = True,
) -> pd.DataFrame:
    comparison_df = pd.read_csv(Path(comparison_csv))
    if "model" not in comparison_df.columns and "model_name" in comparison_df.columns:
        comparison_df = comparison_df.rename(columns={"model_name": "model"})
    if "model_variant" not in comparison_df.columns:
        comparison_df["model_variant"] = comparison_df["model"].astype(str) + "_" + comparison_df["data_variant"].astype(str)
    comparison_df = reorder_comparison_df(comparison_df)
    out_dir = ensure_dir(out_dir)
    comparison_df.to_csv(out_dir / "comparison.csv", index=False)
    if make_plots:
        plot_all_main_metrics(comparison_df, out_dir, show=show_plots)
        plot_all_training_histories(comparison_df, out_dir, show=show_plots)
        plot_epoch_comparisons(comparison_df, out_dir, show=show_plots)
    return comparison_df


def print_leaderboard(comparison_df: pd.DataFrame, top_k: int | None = None) -> None:
    df = comparison_df.copy()
    if top_k is not None:
        df = df.head(top_k)
    df = reorder_comparison_df(df)
    cols = leaderboard_display_columns(df)
    rename = {**METRIC_LABEL, "f1_macro": "F1 macro"}
    disp = df[cols].rename(columns={k: v for k, v in rename.items() if k in cols})
    print("\nLeaderboard")
    print("-" * 100)
    print(disp.to_string(index=False))
    print("-" * 100)
