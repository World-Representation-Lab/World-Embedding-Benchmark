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

DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

REGRESSION_CONFIGS = (
    "pendulum",
    "rigidCollision",
    "sphereDrop3D",
    "dye2d",
    "flowpastcylinder2d",
    "totalInternalReflection",
    "doubleSlitInterference",
    "cylinderTorsion3D",
    "longCuboidBending3D",
    "hyperelasticNeoHookeanCuboid3D",
)


@dataclass(frozen=True)
class RegressionItem:
    item_id: str
    family: str
    video: dict[str, Any]
    attribute: str
    value: float


@dataclass
class RegressionResult:
    dataset_dir: str
    subset: str
    split: str
    model: str
    regression_attribute: str
    num_examples: int
    embedding_dimension: int
    protocol: dict[str, Any]
    selected_alphas: list[float]
    metrics: dict[str, float]
    target: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def load_regression_data(
    dataset_dir: str | Path,
    subset: str,
    *,
    split: str = "train",
    limit: int | None = None,
) -> list[RegressionItem]:
    try:
        from datasets import Video, load_dataset
    except ImportError as exc:
        raise RuntimeError("Regression datasets require the 'datasets' package") from exc
    if subset not in REGRESSION_CONFIGS:
        raise ValueError(
            f"Unknown subset {subset!r}. Available: {', '.join(REGRESSION_CONFIGS)}"
        )
    dataset = load_dataset(str(dataset_dir), subset, split=split)
    dataset = dataset.cast_column("video", Video(decode=False))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        dataset = dataset.select(range(min(limit, len(dataset))))
    items = [
        RegressionItem(
            item_id=str(row["id"]),
            family=str(row["family"]),
            video=row["video"],
            attribute=str(row["regression_attribute"]),
            value=float(row["regression_value"]),
        )
        for row in dataset
    ]
    if len(items) < 4:
        raise ValueError("Regression evaluation requires at least four examples")
    if len({item.item_id for item in items}) != len(items):
        raise ValueError("Regression example IDs must be unique")
    attributes = {item.attribute for item in items}
    if len(attributes) != 1:
        raise ValueError(f"Expected one regression attribute, found {sorted(attributes)}")
    values = np.asarray([item.value for item in items], dtype=np.float64)
    if not np.isfinite(values).all() or np.ptp(values) == 0:
        raise ValueError("Regression targets must be finite and non-constant")
    return items


def evaluate_video_regression(
    model: EmbeddingModel,
    *,
    dataset_dir: str | Path,
    subset: str,
    model_name: str,
    split: str = "train",
    batch_size: int = 4,
    folds: int = 5,
    inner_folds: int = 4,
    seed: int = 42,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    limit: int | None = None,
    embedding_output_dir: str | Path | None = None,
    checkpoint_size: int = 32,
    cache_signature: dict[str, Any] | None = None,
    predictions_output: str | Path | None = None,
) -> RegressionResult:
    items = load_regression_data(dataset_dir, subset, split=split, limit=limit)
    if folds < 2 or folds > len(items):
        raise ValueError(f"folds must be between 2 and {len(items)}")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    alphas = tuple(float(alpha) for alpha in alphas)
    if not alphas or any(alpha <= 0 or not np.isfinite(alpha) for alpha in alphas):
        raise ValueError("alphas must contain positive finite values")

    artifact_dir = Path(embedding_output_dir) if embedding_output_dir else None
    if artifact_dir:
        video_root = artifact_dir / "video_files"
        video_root.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="physics-regression-")
        video_root = Path(cleanup.name)
    try:
        paths = [
            _materialize_video(item.video, video_root / f"{index:08d}.mp4")
            for index, item in enumerate(items)
        ]
        signature = {
            "task": "video_regression",
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "subset": subset,
            "split": split,
            "model": model_name,
            "ids": [item.item_id for item in items],
            "targets": [item.value for item in items],
            **(cache_signature or {}),
        }
        embeddings = encode_with_checkpoints(
            modality="video",
            ids=[item.item_id for item in items],
            items=paths,
            encode=lambda videos: model.encode_videos(videos, batch_size=batch_size),
            output_dir=artifact_dir,
            checkpoint_size=checkpoint_size,
            signature=signature,
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

    targets = np.asarray([item.value for item in items], dtype=np.float64)
    predictions, selected_alphas, fold_ids = nested_ridge_predictions(
        embeddings, targets, folds=folds, inner_folds=inner_folds,
        alphas=alphas, seed=seed,
    )
    metrics = regression_metrics(targets, predictions)
    result = RegressionResult(
        dataset_dir=str(Path(dataset_dir)),
        subset=subset,
        split=split,
        model=model_name,
        regression_attribute=items[0].attribute,
        num_examples=len(items),
        embedding_dimension=int(embeddings.shape[1]),
        protocol={
            "name": "nested_cross_validated_ridge",
            "outer_folds": folds,
            "inner_folds": inner_folds,
            "shuffle": True,
            "seed": seed,
            "alphas": list(alphas),
            "standardize_embeddings": True,
            "standardize_target": True,
        },
        selected_alphas=selected_alphas,
        metrics=metrics,
        target={
            "min": float(targets.min()),
            "max": float(targets.max()),
            "mean": float(targets.mean()),
            "std": float(targets.std()),
        },
    )
    if predictions_output:
        _write_predictions(
            Path(predictions_output), items, targets, predictions, fold_ids
        )
    return result


def nested_ridge_predictions(
    embeddings: np.ndarray,
    targets: np.ndarray,
    *,
    folds: int,
    inner_folds: int,
    alphas: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Embeddings must be [examples, dimensions] and match targets")
    if len(y) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Regression inputs need at least four finite examples")
    if np.ptp(y) == 0:
        raise ValueError("Regression targets must be non-constant")
    if folds < 2 or folds > len(y):
        raise ValueError(f"folds must be between 2 and {len(y)}")
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    alphas = tuple(float(alpha) for alpha in alphas)
    if not alphas or any(alpha <= 0 or not np.isfinite(alpha) for alpha in alphas):
        raise ValueError("alphas must contain positive finite values")
    outer = _fold_indices(len(y), folds, seed)
    predictions = np.empty_like(y)
    fold_ids = np.empty(len(y), dtype=np.int64)
    selected: list[float] = []
    for fold, test_idx in enumerate(outer):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        alpha = _select_alpha(
            x[train_idx], y[train_idx], alphas, min(inner_folds, len(train_idx)),
            seed + 10_000 + fold,
        )
        predictions[test_idx] = _ridge_predict(
            x[train_idx], y[train_idx], x[test_idx], alpha
        )
        fold_ids[test_idx] = fold
        selected.append(float(alpha))
    return predictions, selected, fold_ids


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or len(y) < 2:
        raise ValueError("Targets and predictions must be matching one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Targets and predictions must be finite")
    if np.ptp(y) == 0:
        raise ValueError("Regression targets must be non-constant")
    residual = p - y
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    span = float(np.ptp(y))
    variance = float(np.sum((y - y.mean()) ** 2))
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    ranked_y, ranked_p = _average_ranks(y), _average_ranks(p)
    spearman = (
        float(np.corrcoef(ranked_y, ranked_p)[0, 1])
        if np.std(ranked_p) > 0 else 0.0
    )
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(residual ** 2) / variance),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "normalized_mae": mae / span,
        "normalized_rmse": float(np.sqrt(mse)) / span,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ranks, assigning the average rank to exact ties."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    folds: int,
    seed: int,
) -> float:
    splits = _fold_indices(len(y), folds, seed)
    scores = []
    all_idx = np.arange(len(y))
    for alpha in alphas:
        errors = []
        for validation_idx in splits:
            train_idx = np.setdiff1d(all_idx, validation_idx, assume_unique=True)
            prediction = _ridge_predict(
                x[train_idx], y[train_idx], x[validation_idx], alpha
            )
            scale = max(
                float(np.ptp(y[train_idx])), np.finfo(np.float64).eps
            )
            errors.append(
                np.mean(np.abs(prediction - y[validation_idx])) / scale
            )
        scores.append(float(np.mean(errors)))
    return float(alphas[int(np.argmin(scores))])


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    train = (x_train - x_mean) / x_scale
    test = (x_test - x_mean) / x_scale
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std())
    if y_scale < 1e-12:
        return np.full(len(x_test), y_mean)
    centered_y = (y_train - y_mean) / y_scale
    kernel = train @ train.T
    coefficients = np.linalg.solve(
        kernel + alpha * np.eye(len(train)), centered_y
    )
    return (test @ train.T @ coefficients) * y_scale + y_mean


def _fold_indices(num_examples: int, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_examples)
    return [part for part in np.array_split(indices, folds) if len(part)]


def _materialize_video(video: dict[str, Any], destination: Path) -> Path:
    payload = video.get("bytes")
    source = video.get("path")
    if payload is None:
        if not source:
            raise ValueError("Video has neither embedded bytes nor a path")
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == len(payload):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination


def _write_predictions(
    path: Path,
    items: Sequence[RegressionItem],
    targets: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "id", "family", "regression_attribute", "target", "prediction",
            "absolute_error", "fold",
        ])
        for item, target, prediction, fold in zip(
            items, targets, predictions, fold_ids, strict=True
        ):
            writer.writerow([
                item.item_id, item.family, item.attribute, float(target),
                float(prediction), float(abs(prediction - target)), int(fold),
            ])
    os.replace(temporary, path)
