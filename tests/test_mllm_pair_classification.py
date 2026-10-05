from __future__ import annotations

import unittest

from world_embedding_benchmark.mllm_pair_classification import _score, build_questions


def make_manifest() -> dict:
    return {
        "examples": [
            {
                "example_id": f"e{index}",
                "family": "a",
                "query_id": f"q{index}",
                "candidates": [
                    {"role": "positive", "video_id": f"p{index}"},
                    {"role": "within_family_negative", "video_id": f"w{index}"},
                    {"role": "cross_family_negative", "video_id": f"c{index}"},
                ],
            }
            for index in range(2)
        ]
    }


class BuildQuestionsTest(unittest.TestCase):
    def test_expands_into_two_tasks_and_two_orderings(self) -> None:
        questions = build_questions(make_manifest())
        self.assertEqual(len(questions), 8)
        self.assertEqual({q["task"] for q in questions}, {"within_family", "cross_family"})

    def test_answer_follows_the_ordering(self) -> None:
        for question in build_questions(make_manifest()):
            expected = "A" if question["positive_first"] else "B"
            self.assertEqual(question["answer"], expected)
            positive = question["video_a"] if question["positive_first"] else question["video_b"]
            self.assertTrue(positive.startswith("p"))


class ScoreTest(unittest.TestCase):
    def records(self, predictions: list[str]) -> list[dict]:
        questions = [q for q in build_questions(make_manifest()) if q["task"] == "within_family"]
        return [{**q, "predicted": p} for q, p in zip(questions, predictions)]

    def test_always_picking_one_position_scores_at_chance(self) -> None:
        scores = _score(self.records(["A", "A", "A", "A"]))
        self.assertAlmostEqual(scores["accuracy"], 0.5)
        self.assertAlmostEqual(scores["both_orders_accuracy"], 0.0)
        self.assertAlmostEqual(scores["pick_a_rate"], 1.0)

    def test_perfect_answers(self) -> None:
        scores = _score(self.records(["A", "B", "A", "B"]))
        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["both_orders_accuracy"], 1.0)

    def test_requires_both_orderings(self) -> None:
        with self.assertRaises(ValueError):
            _score(self.records(["A", "B"])[:1])


if __name__ == "__main__":
    unittest.main()
