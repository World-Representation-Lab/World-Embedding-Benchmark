"""Optional vLLM backend for LCO text/video embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lco_embedding import (
    TEXT_COMPRESSION_SUFFIX, VIDEO_COMPRESSION_SUFFIX,
    resolve_video_decoder, sample_qwen_processor_video_frames,
    sample_qwen_processor_video_frames_torchcodec, sample_video_frames,
)


@dataclass
class LCOVLLMEmbedding:
    """LCO inference through vLLM LAST pooling plus L2 normalization."""

    model_name: str
    dtype: str = "bfloat16"
    video_sampling: str = "processor"
    fps: float | None = 2.0
    max_frames: int | None = 64
    num_frames: int | None = None
    max_pixels: int | None = None
    use_audio_in_video: bool = False
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    video_prefetch_batches: int = 0
    video_decode_workers: int = 1
    video_decoder: str = "auto"
    engine_kwargs: dict[str, Any] | None = None
    device: str | None = None
    model_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.device not in (None, "cuda"):
            raise ValueError("The vLLM backend currently requires CUDA")
        if self.model_kwargs:
            raise ValueError("Use engine_kwargs for vLLM-specific options")
        if self.video_sampling not in {"processor", "fixed"}:
            raise ValueError("video_sampling must be 'processor' or 'fixed'")
        if self.video_sampling == "fixed" and self.num_frames is None and self.fps is None:
            raise ValueError("fixed sampling needs num_frames or fps")
        if self.video_prefetch_batches < 0:
            raise ValueError("video_prefetch_batches must be non-negative")
        if self.video_decode_workers < 1:
            raise ValueError("video_decode_workers must be positive")
        self.video_decoder = resolve_video_decoder(
            self.video_decoder, video_sampling=self.video_sampling
        )
        try:
            from transformers import Qwen2_5OmniProcessor
            from vllm import LLM
            from vllm.config import PoolerConfig
            from vllm.model_executor.models import ModelRegistry
        except ImportError as exc:
            raise RuntimeError(
                "Install the versions pinned in requirements.txt before using "
                "--backend vllm."
            ) from exc

        from .vllm_lco_model import (
            LCOQwen2_5OmniForConditionalGeneration,
            VLLM_LCO_ARCHITECTURE,
            wrap_lco_thinker_config,
        )

        ModelRegistry.register_model(VLLM_LCO_ARCHITECTURE, LCOQwen2_5OmniForConditionalGeneration)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        self.processor.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "tokenizer": self.model_name,
            "runner": "pooling",
            "convert": "embed",
            "pooler_config": PoolerConfig(task="embed", pooling_type="LAST"),
            "hf_overrides": wrap_lco_thinker_config,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "tensor_parallel_size": self.tensor_parallel_size,
            "limit_mm_per_prompt": {"video": 1},
        }
        kwargs.update(self.engine_kwargs or {})
        self.llm = LLM(**kwargs)

    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        prompts = self._chat_text([
            [{"role": "user", "content": [{"type": "text", "text": text + TEXT_COMPRESSION_SUFFIX}]}]
            for text in texts
        ])
        return self._embed(prompts, batch_size=batch_size)

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray:

        if not videos:
            raise ValueError("No videos to encode")
        embeddings: list[np.ndarray] = []
        window = max(batch_size, (1 + self.video_prefetch_batches) * batch_size)
        with ThreadPoolExecutor(
            max_workers=self.video_decode_workers,
            thread_name_prefix="lco-video-decode",
        ) as pool:
            futures: dict[int, Any] = {}
            next_video = 0
            while next_video < min(window, len(videos)):
                futures[next_video] = pool.submit(self._build_video_prompt, videos[next_video])
                next_video += 1
            for start in range(0, len(videos), batch_size):
                stop = min(start + batch_size, len(videos))
                prompts = [futures.pop(index).result() for index in range(start, stop)]
                while next_video < min(stop + window, len(videos)):
                    futures[next_video] = pool.submit(
                        self._build_video_prompt, videos[next_video]
                    )
                    next_video += 1
                embeddings.append(self._embed(prompts, batch_size=batch_size))
        return np.concatenate(embeddings, axis=0)

    def _build_video_prompts(
        self, videos: Sequence[str | Path]
    ) -> list[dict[str, Any]]:
        return [self._build_video_prompt(video) for video in videos]

    def _build_video_prompt(self, video: str | Path) -> dict[str, Any]:
        frames = self._load_video_frames(video)
        messages = [[{"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": VIDEO_COMPRESSION_SUFFIX},
        ]}]]
        return {
            "prompt": self._chat_text(messages)[0],
            "multi_modal_data": {"video": frames},
            "mm_processor_kwargs": {
                "do_sample_frames": False,
                "use_audio_in_video": False,
            },
        }

    def _load_video_frames(self, video: str | Path) -> Any:
        if self.video_sampling == "fixed":
            return sample_video_frames(
                video, fps=self.fps, max_frames=self.max_frames, num_frames=self.num_frames
            )
        sampler = (
            sample_qwen_processor_video_frames_torchcodec
            if self.video_decoder == "torchcodec"
            else sample_qwen_processor_video_frames
        )
        return sampler(
            video, fps=self.fps if self.fps is not None else 2.0,
            max_frames=self.max_frames,
        )

    def _embed(self, prompts: Sequence[Any], *, batch_size: int) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            outputs = self.llm.embed(list(prompts[start : start + batch_size]))
            embeddings.extend(np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs)
        return np.stack(embeddings)

    def _chat_text(self, messages: list[list[dict[str, Any]]]) -> list[str]:
        return [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
