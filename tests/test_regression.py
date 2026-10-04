from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from world_embedding_benchmark.regression import (
    DEFAULT_ALPHAS,
    RegressionItem,
    evaluate_video_regression,
    nested_ridge_predictions,
    regression_metrics,
)


class RegressionMetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        targets = np.asarray([1.0, 2.0, 2.0, 4.0])
        metrics = regression_metrics(targets, targets.copy())

        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertEqual(metrics["normalized_rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)
        self.assertAlmostEqual(metrics["pearson_r"], 1.0)
        self.assertAlmostEqual(metrics["spearman_r"], 1.0)

    def test_rejects_constant_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-constant"):
            regression_metrics(np.ones(4), np.ones(4))

    def test_normalized_errors_use_target_range(self) -> None:
        metrics = regression_metrics(
            np.asarray([0.0, 2.0]), np.asarray([1.0, 1.0])
        )

        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["normalized_mae"], 0.5)
        self.assertAlmostEqual(metrics["normalized_rmse"], 0.5)


class NestedRidgeTest(unittest.TestCase):
    def test_predictions_are_deterministic_and_out_of_fold(self) -> None:
        values = np.linspace(-2.0, 2.0, 24)
        embeddings = np.stack([values, values**2, np.sin(values)], axis=1)
        targets = 3.0 * values - 0.5

        first = nested_ridge_predictions(
            embeddings, targets, folds=4, inner_folds=3,
            alphas=DEFAULT_ALPHAS, seed=17,
        )
        second = nested_ridge_predictions(
            embeddings, targets, folds=4, inner_folds=3,
            alphas=DEFAULT_ALPHAS, seed=17,
        )

        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(len(first[1]), 4)
        self.assertEqual(set(first[2]), {0, 1, 2, 3})
        self.assertTrue(np.isfinite(first[0]).all())

    def test_rejects_invalid_alpha_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive finite"):
            nested_ridge_predictions(
                np.arange(12, dtype=float).reshape(6, 2),
                np.arange(6, dtype=float),
                folds=3, inner_folds=2, alphas=(), seed=1,
            )


class FakeVideoEmbeddingModel:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def encode_videos(self, videos, *, batch_size: int) -> np.ndarray:
        if self.fail:
            raise AssertionError("cached embeddings should have been reused")
        self.calls += 1
        values = np.asarray([Path(path).read_bytes()[0] for path in videos], dtype=np.float32)
        return np.stack([values, values**2, np.ones_like(values)], axis=1)


class RegressionEvaluationTest(unittest.TestCase):
    def test_model_agnostic_evaluation_writes_and_reuses_artifacts(self) -> None:
        items = [
            RegressionItem(
                item_id=f"item-{index}", family="synthetic",
                video={"bytes": bytes([index]), "path": f"{index}.mp4"},
                attribute="synthetic_parameter", value=float(index),
            )
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            predictions = root / "predictions.csv"
            model = FakeVideoEmbeddingModel()
            with patch(
                "world_embedding_benchmark.regression.load_regression_data",
                return_value=items,
            ):
                result = evaluate_video_regression(
                    model,
                    dataset_dir=root / "dataset",
                    subset="pendulum",
                    model_name="fake-video-model",
                    folds=3,
                    inner_folds=2,
                    batch_size=4,
                    checkpoint_size=5,
                    embedding_output_dir=artifacts,
                    predictions_output=predictions,
                )

            self.assertEqual(result.num_examples, 12)
            self.assertEqual(result.embedding_dimension, 3)
            self.assertEqual(result.protocol["name"], "nested_cross_validated_ridge")
            self.assertEqual(len(result.selected_alphas), 3)
            self.assertGreater(model.calls, 0)
            self.assertTrue((artifacts / "video_embeddings.npz").is_file())
            self.assertEqual(len(predictions.read_text().splitlines()), 13)

            with patch(
                "world_embedding_benchmark.regression.load_regression_data",
                return_value=items,
            ):
                cached = evaluate_video_regression(
                    FakeVideoEmbeddingModel(fail=True),
                    dataset_dir=root / "dataset",
                    subset="pendulum",
                    model_name="fake-video-model",
                    folds=3,
                    inner_folds=2,
                    batch_size=4,
                    checkpoint_size=5,
                    embedding_output_dir=artifacts,
                )
            self.assertEqual(cached.metrics, result.metrics)


if __name__ == "__main__":
    unittest.main()
