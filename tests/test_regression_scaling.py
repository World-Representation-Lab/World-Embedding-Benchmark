from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from world_embedding_benchmark.regression import RegressionItem
from world_embedding_benchmark.regression_scaling import (
    evaluate_video_regression_scaling,
    ridge_predict_many,
    select_ridge_alpha,
)
from world_embedding_benchmark.regression_splits import (
    build_regression_split_manifest,
    load_and_validate_regression_split_manifest,
)


def make_items(count: int = 20) -> list[RegressionItem]:
    return [
        RegressionItem(
            item_id=f"item-{index:03d}",
            family="pendulum",
            video={"bytes": str(index).encode(), "path": None},
            attribute="gravity",
            value=float(index),
        )
        for index in range(count)
    ]


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode_videos(self, paths, *, batch_size):
        self.calls += 1
        values = np.asarray([float(Path(path).read_bytes()) for path in paths])
        return np.column_stack((values, values**2, np.sin(values)))


class RegressionSplitTest(unittest.TestCase):
    def test_manifest_is_deterministic_stratified_and_nested(self):
        items = make_items()
        kwargs = dict(
            subset="pendulum", test_size=4, num_strata=4,
            train_sizes=(4, 8, 16), repetition_seeds=(10, 11),
        )
        first = build_regression_split_manifest(items, **kwargs)
        second = build_regression_split_manifest(items, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual({entry["stratum"] for entry in first["test_examples"]}, {0, 1, 2, 3})
        test_ids = {entry["id"] for entry in first["test_examples"]}
        self.assertFalse(test_ids & set(first["training_pool_ids"]))
        for repetition in first["scaling"]["repetitions"]:
            order = repetition["train_id_order"]
            self.assertEqual(set(order), set(first["training_pool_ids"]))
            self.assertEqual(order[:4], order[:8][:4])

    def test_manifest_validation_detects_dataset_drift(self):
        items = make_items()
        manifest = build_regression_split_manifest(
            items, subset="pendulum", test_size=4, num_strata=4,
            train_sizes=(4, 8), repetition_seeds=(10,),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            load_and_validate_regression_split_manifest(path, items, subset="pendulum")
            changed = list(items)
            changed[0] = RegressionItem(**{**changed[0].__dict__, "value": 999.0})
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_and_validate_regression_split_manifest(path, changed, subset="pendulum")


class RegressionScalingTest(unittest.TestCase):
    def test_ridge_helpers_are_deterministic_and_finite(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(12, 5))
        y = x[:, 0] - 0.5 * x[:, 1]
        alphas = (0.01, 0.1, 1.0)
        first = select_ridge_alpha(x, y, alphas=alphas, folds=3, seed=7)
        second = select_ridge_alpha(x, y, alphas=alphas, folds=3, seed=7)
        self.assertEqual(first, second)
        predictions = ridge_predict_many(x, y, x[:3], alphas)
        self.assertEqual(predictions.shape, (3, 3))
        self.assertTrue(np.isfinite(predictions).all())

    def test_end_to_end_encodes_once_and_writes_fixed_test_predictions(self):
        items = make_items()
        manifest = build_regression_split_manifest(
            items, subset="pendulum", test_size=4, num_strata=4,
            train_sizes=(4, 8), repetition_seeds=(10, 11),
        )
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "split.json"
            predictions_path = root / "predictions.csv"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "world_embedding_benchmark.regression_scaling.load_regression_data",
                return_value=items,
            ):
                result = evaluate_video_regression_scaling(
                    model,
                    dataset_dir=root,
                    subset="pendulum",
                    split_manifest=manifest_path,
                    model_name="fake",
                    batch_size=3,
                    inner_folds=2,
                    alphas=(0.01, 0.1),
                    embedding_output_dir=root / "embeddings",
                    checkpoint_size=7,
                    predictions_output=predictions_path,
                )
            self.assertEqual([row["train_size"] for row in result.summary], [4, 8])
            self.assertEqual(len(result.runs), 4)
            self.assertEqual(result.test_size, 4)
            self.assertEqual(model.calls, 3)
            with predictions_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)
            self.assertEqual(
                {row["id"] for row in rows},
                {entry["id"] for entry in manifest["test_examples"]},
            )


if __name__ == "__main__":
    unittest.main()
