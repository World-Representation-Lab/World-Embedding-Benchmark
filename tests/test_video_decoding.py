from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from world_embedding_benchmark.models.lco_embedding import resolve_video_decoder
from world_embedding_benchmark.models.lco_vllm import LCOVLLMEmbedding


DECODER_LOADER = (
    "world_embedding_benchmark.models.lco_embedding."
    "_load_torchcodec_video_decoder"
)


class ResolveVideoDecoderTest(unittest.TestCase):
    def test_auto_prefers_torchcodec_when_available(self) -> None:
        with patch(DECODER_LOADER, return_value=object()):
            self.assertEqual(
                resolve_video_decoder("auto", video_sampling="processor"),
                "torchcodec",
            )

    def test_auto_warns_and_falls_back(self) -> None:
        with patch(DECODER_LOADER, side_effect=ImportError("not installed")):
            with self.assertWarnsRegex(RuntimeWarning, "falling back to FFmpeg"):
                actual = resolve_video_decoder("auto", video_sampling="processor")
        self.assertEqual(actual, "ffmpeg")

    def test_explicit_torchcodec_fails_when_unavailable(self) -> None:
        with patch(DECODER_LOADER, side_effect=OSError("missing shared library")):
            with self.assertRaisesRegex(RuntimeError, "TorchCodec was requested"):
                resolve_video_decoder("torchcodec", video_sampling="processor")

    def test_fixed_auto_uses_ffmpeg_without_importing_torchcodec(self) -> None:
        with patch(DECODER_LOADER) as loader:
            self.assertEqual(
                resolve_video_decoder("auto", video_sampling="fixed"), "ffmpeg"
            )
        loader.assert_not_called()

    def test_fixed_rejects_explicit_torchcodec(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed video sampling"):
            resolve_video_decoder("torchcodec", video_sampling="fixed")


class ParallelPrefetchTest(unittest.TestCase):
    def test_parallel_prompt_build_preserves_input_order(self) -> None:
        model = object.__new__(LCOVLLMEmbedding)
        model.video_prefetch_batches = 2
        model.video_decode_workers = 3
        model._build_video_prompt = lambda item: int(item)
        model._embed = lambda prompts, batch_size: np.asarray(prompts)[:, None]

        embeddings = model.encode_videos(["0", "1", "2", "3", "4"], batch_size=2)

        np.testing.assert_array_equal(embeddings[:, 0], np.arange(5))

    def test_empty_video_input_is_rejected(self) -> None:
        model = object.__new__(LCOVLLMEmbedding)
        with self.assertRaisesRegex(ValueError, "No videos"):
            model.encode_videos([], batch_size=2)


if __name__ == "__main__":
    unittest.main()
