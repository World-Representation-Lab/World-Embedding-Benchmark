from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embedding_artifacts import encode_with_checkpoints, save_embedding_archive
from .models import EmbeddingModel
from .regression import (
    RegressionItem,
    _materialize_video,
    load_regression_data,
    regression_metrics,
)
from .regression_splits import load_and_validate_regression_split_manifest

DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
METRIC_NAMES = (
    "mae", "mse", "rmse", "r2", "pearson_r", "spearman_r", "normalized_mae", "normalized_rmse"
)


@dataclass
class ScalingRunResult:
    repetition: int
    seed: int
    train_size: int
    selected_alpha: float
    metrics: dict[str, float]


@dataclass
class RegressionScalingResult:
    dataset_dir: str
    subset: str
    split_manifest: str
    model: str
    regression_attribute: str
    num_examples: int
    train_pool_size: int
    test_size: int
    embedding_dimension: int
    protocol: dict[str, Any]
    runs: list[ScalingRunResult]
    summary: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def evaluate_video_regression_scaling(
    model: EmbeddingModel,
    *,
    dataset_dir: str | Path,
    subset: str,
    split_manifest: str | Path,
    model_name: str,
    split: str = "test",
    batch_size: int = 4,
    inner_folds: int = 4,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    embedding_output_dir: str | Path | None = None,
    checkpoint_size: int = 32,
    cache_signature: dict[str, Any] | None = None,
    predictions_output: str | Path | None = None,
) -> RegressionScalingResult:
    items = load_regression_data(dataset_dir, subset, split=split)
    manifest = load_and_validate_regression_split_manifest(
        split_manifest, items, subset=subset
    )
    alphas = tuple(float(alpha) for alpha in alphas)
    if not alphas or any(alpha <= 0 or not np.isfinite(alpha) for alpha in alphas):
        raise ValueError("alphas must contain positive finite values")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")

    embeddings = _encode_items(
        model, items, dataset_dir=dataset_dir, subset=subset, split=split,
        model_name=model_name, batch_size=batch_size,
        embedding_output_dir=embedding_output_dir, checkpoint_size=checkpoint_size,
        cache_signature=cache_signature,
    )
    targets = np.asarray([item.value for item in items], dtype=np.float64)
    id_to_index = {item.item_id: index for index, item in enumerate(items)}
    test_ids = [entry["id"] for entry in manifest["test_examples"]]
    test_idx = np.asarray([id_to_index[item_id] for item_id in test_ids])
    test_x, test_y = embeddings[test_idx], targets[test_idx]

    runs: list[ScalingRunResult] = []
    prediction_rows: list[dict[str, Any]] = []
    train_sizes = manifest["scaling"]["train_sizes"]
    for repetition, definition in enumerate(manifest["scaling"]["repetitions"]):
        ordered_indices = np.asarray(
            [id_to_index[item_id] for item_id in definition["train_id_order"]]
        )
        seed = int(definition["seed"])
        for train_size in train_sizes:
            train_idx = ordered_indices[: int(train_size)]
            train_x, train_y = embeddings[train_idx], targets[train_idx]
            alpha = select_ridge_alpha(
                train_x, train_y, alphas=alphas,
                folds=min(inner_folds, len(train_idx)), seed=seed,
            )
            predictions = ridge_predict(train_x, train_y, test_x, alpha)
            metrics = regression_metrics(test_y, predictions)
            runs.append(ScalingRunResult(
                repetition=repetition,
                seed=seed,
                train_size=int(train_size),
                selected_alpha=alpha,
                metrics=metrics,
            ))
            for item_id, target, prediction in zip(
                test_ids, test_y, predictions, strict=True
            ):
                prediction_rows.append({
                    "repetition": repetition,
                    "seed": seed,
                    "train_size": int(train_size),
                    "selected_alpha": alpha,
                    "id": item_id,
                    "target": float(target),
                    "prediction": float(prediction),
                    "absolute_error": float(abs(prediction - target)),
                })

    summary = []
    for train_size in train_sizes:
        selected = [run for run in runs if run.train_size == train_size]
        row: dict[str, Any] = {
            "train_size": int(train_size),
            "num_repetitions": len(selected),
        }
        for metric in METRIC_NAMES:
            values = np.asarray([run.metrics[metric] for run in selected])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)

    if predictions_output:
        _write_scaling_predictions(Path(predictions_output), prediction_rows)
    return RegressionScalingResult(
        dataset_dir=str(Path(dataset_dir)),
        subset=subset,
        split_manifest=str(Path(split_manifest)),
        model=model_name,
        regression_attribute=items[0].attribute,
        num_examples=len(items),
        train_pool_size=len(manifest["training_pool_ids"]),
        test_size=len(test_ids),
        embedding_dimension=int(embeddings.shape[1]),
        protocol={
            "name": "fixed_test_nested_validation_ridge_scaling",
            "test_selection": manifest["split"],
            "train_sizes": train_sizes,
            "num_repetitions": len(manifest["scaling"]["repetitions"]),
            "nested_training_prefixes": True,
            "inner_folds": inner_folds,
            "alphas": list(alphas),
            "standardize_embeddings": True,
            "standardize_target": True,
        },
        runs=runs,
        summary=summary,
    )


def select_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alphas: Sequence[float],
    folds: int,
    seed: int,
) -> float:
    splits = _fold_indices(len(y), folds, seed)
    errors = np.zeros((len(splits), len(alphas)), dtype=np.float64)
    all_indices = np.arange(len(y))
    for fold, validation_idx in enumerate(splits):
        train_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=True)
        predictions = ridge_predict_many(
            x[train_idx], y[train_idx], x[validation_idx], alphas
        )
        scale = max(float(np.ptp(y[train_idx])), np.finfo(np.float64).eps)
        errors[fold] = np.mean(
            np.abs(predictions - y[validation_idx, None]), axis=0
        ) / scale
    return float(alphas[int(np.argmin(errors.mean(axis=0)))])


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return ridge_predict_many(x_train, y_train, x_test, (alpha,))[:, 0]


def ridge_predict_many(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: Sequence[float],
) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    if target_scale < 1e-12:
        return np.full((len(x_test), len(alphas)), target_mean)
    standardized_y = (y_train - target_mean) / target_scale
    eigenvalues, eigenvectors = np.linalg.eigh(train @ train.T)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ standardized_y
    projected_test = test @ train.T @ eigenvectors
    predictions = np.column_stack([
        projected_test @ (projected_y / (eigenvalues + alpha))
        for alpha in alphas
    ])
    return predictions * target_scale + target_mean


def _fold_indices(num_examples: int, folds: int, seed: int) -> list[np.ndarray]:
    if folds < 2 or folds > num_examples:
        raise ValueError(f"folds must be in [2, {num_examples}]")
    indices = np.random.default_rng(seed).permutation(num_examples)
    return [part for part in np.array_split(indices, folds) if len(part)]


def _encode_items(
    model: EmbeddingModel,
    items: Sequence[RegressionItem],
    *,
    dataset_dir: str | Path,
    subset: str,
    split: str,
    model_name: str,
    batch_size: int,
    embedding_output_dir: str | Path | None,
    checkpoint_size: int,
    cache_signature: dict[str, Any] | None,
) -> np.ndarray:
    artifact_dir = Path(embedding_output_dir) if embedding_output_dir else None
    if artifact_dir:
        video_root = artifact_dir / "video_files"
        video_root.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="physics-regression-scaling-")
        video_root = Path(cleanup.name)
    try:
        paths = [
            _materialize_video(item.video, video_root / f"{index:08d}.mp4")
            for index, item in enumerate(items)
        ]
        embeddings = encode_with_checkpoints(
            modality="video",
            ids=[item.item_id for item in items],
            items=paths,
            encode=lambda videos: model.encode_videos(videos, batch_size=batch_size),
            output_dir=artifact_dir,
            checkpoint_size=checkpoint_size,
            signature={
                "task": "video_regression_scaling",
                "dataset_dir": str(Path(dataset_dir).resolve()),
                "subset": subset,
                "split": split,
                "model": model_name,
                "ids": [item.item_id for item in items],
                **(cache_signature or {}),
            },
        )
    finally:
        if cleanup is not None:
            cleanup.cleanup()
    if artifact_dir:
        save_embedding_archive(
            artifact_dir / "video_embeddings.npz",
            ids=[item.item_id for item in items],
            families=[item.family for item in items],
            embeddings=embeddings,
        )
    return embeddings


def _write_scaling_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = (
        "repetition", "seed", "train_size", "selected_alpha", "id",
        "target", "prediction", "absolute_error",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
