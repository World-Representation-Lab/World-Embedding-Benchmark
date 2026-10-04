from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from world_embedding_benchmark.pair_classification import (
    _metrics,
    evaluate_pair_classification,
)
from world_embedding_benchmark.pair_classification_data import (
    build_pair_classification_manifest,
)
from world_embedding_benchmark.retrieval import Query, RetrievalData, VideoItem


def make_data() -> RetrievalData:
    videos = [
        VideoItem(f"{family}-v{index}", Path(f"/{family}-v{index}.mp4"), family, {})
        for family in ("a", "b") for index in range(5)
    ]
    queries = [
        Query(
            query_id=f"{family}-q{index}",
            text=f"query {family} {index}",
            relevant_ids=(f"{family}-v{index % 5}",),
            family=family,
        )
        for family in ("a", "b") for index in range(12)
    ]
    return RetrievalData(queries=queries, videos=videos)


class FakeModel:
    def encode_texts(self, texts, *, batch_size):
        return np.asarray([
            [len(text), sum(text.encode()) % 17, 1.0] for text in texts
        ], dtype=np.float32)

    def encode_videos(self, paths, *, batch_size):
        return np.asarray([
            [len(str(path)), sum(str(path).encode()) % 17, 1.0] for path in paths
        ], dtype=np.float32)


class PairManifestTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_respects_candidate_roles(self):
        data = make_data()
        kwargs = dict(
            branch="test", dataset_dir="datasets/test",
            queries_per_family=10, seed=42,
        )
        first = build_pair_classification_manifest(data, **kwargs)
        second = build_pair_classification_manifest(data, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["dataset_dir"], "datasets/test")
        self.assertEqual(first["num_examples"], 20)
        query_by_id = {query.query_id: query for query in data.queries}
        family_by_video = {video.item_id: video.family for video in data.videos}
        selected = {example["query_id"] for example in first["examples"]}
        head = {f"{family}-q{index}" for family in ("a", "b") for index in range(10)}
        self.assertNotEqual(selected, head)
        for example in first["examples"]:
            candidates = sorted(example["candidates"], key=lambda row: row["index"])
            positive, within, cross = (row["video_id"] for row in candidates)
            query = query_by_id[example["query_id"]]
            self.assertIn(positive, query.relevant_ids)
            self.assertEqual(family_by_video[positive], example["family"])
            self.assertEqual(family_by_video[within], example["family"])
            self.assertNotIn(within, query.relevant_ids)
            self.assertNotEqual(family_by_video[cross], example["family"])


class PairScoringTest(unittest.TestCase):
    def test_metrics_use_strict_comparisons_and_report_ties(self):
        rows = [
            {"positive_score": 3.0, "within_negative_score": 2.0, "cross_negative_score": 1.0},
            {"positive_score": 1.0, "within_negative_score": 1.0, "cross_negative_score": 2.0},
        ]
        metrics = _metrics(rows)
        self.assertEqual(metrics["within_family_accuracy"], 0.5)
        self.assertEqual(metrics["cross_family_accuracy"], 0.5)
        self.assertEqual(metrics["three_way_accuracy"], 0.5)
        self.assertEqual(metrics["within_family_tie_rate"], 0.5)

    def test_end_to_end_accepts_dataset_override_and_writes_predictions(self):
        data = make_data()
        manifest = build_pair_classification_manifest(
            data, branch="test", dataset_dir="datasets/missing",
            queries_per_family=2, seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            predictions = root / "predictions.csv"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "world_embedding_benchmark.pair_classification.load_retrieval_data",
                return_value=data,
            ) as loader:
                result = evaluate_pair_classification(
                    FakeModel(), manifest_path=manifest_path,
                    dataset_dir=root / "dataset", model_name="fake",
                    batch_size=2, embedding_output_dir=root / "embeddings",
                    checkpoint_size=3, predictions_output=predictions,
                )
            loader.assert_called_once_with(root / "dataset", text_column="parsed_text")
            self.assertEqual(result.num_examples, 4)
            self.assertEqual(set(result.per_family), {"a", "b"})
            self.assertTrue(predictions.is_file())
            self.assertIn("macro_family_within_family_accuracy", result.metrics)


if __name__ == "__main__":
    unittest.main()
