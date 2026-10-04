"""Qwen3-VL-Embedding through the vLLM pooling runner."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import EmbeddingModel, register_model

# The model card reports a 1-5% gain from task-specific instructions and
# recommends writing them in English.
DEFAULT_TEXT_INSTRUCTION = "Retrieve a physics simulation video based on this text description"
DEFAULT_VIDEO_INSTRUCTION = "Retrieve a text caption describing this physics simulation video"

QWEN3VL_CHECKPOINTS = {
    "qwen3-vl-embedding-2b": "Qwen/Qwen3-VL-Embedding-2B",
    "qwen3-vl-embedding-8b": "Qwen/Qwen3-VL-Embedding-8B",
}


def sample_video_with_metadata(
    video: str | Path, *, fps: float, max_frames: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample frames like the LCO TorchCodec sampler and report the indices.

    Qwen3-VL derives per-frame timestamps from the metadata and raises
    "Video metadata is required but not found in mm input" without it, so the
    sampler has to return the indices alongside the frames.
    """
    import torch
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video), dimension_order="NCHW", num_ffmpeg_threads=1)
    total_frames = int(decoder.metadata.num_frames)
    source_fps = float(decoder.metadata.average_fps)
    maximum = min(max_frames if max_frames is not None else 768, total_frames)
    maximum -= maximum % 2
    target = min(max(total_frames / source_fps * fps, min(4, maximum)), maximum, total_frames)
    target = int(target) - int(target) % 2
    if target < 2:
        raise RuntimeError(f"No valid frame count for {video}")
    indices = torch.linspace(0, total_frames - 1, steps=target).round().long()
    metadata = {
        "fps": source_fps,
        "frames_indices": indices.tolist(),
        "total_num_frames": total_frames,
        "do_sample_frames": False,
        "video_backend": "torchcodec",
    }
    return np.asarray(decoder.get_frames_at(indices).data.cpu()), metadata


@dataclass
class Qwen3VLVLLMEmbedding:
    """Embed text and video in the shared Qwen3-VL-Embedding space.

    ``video_prefetch_batches`` is accepted for CLI compatibility and unused;
    frames are decoded one batch ahead of the forward pass by the thread pool.
    """

    model_name: str
    dtype: str = "bfloat16"
    text_instruction: str = DEFAULT_TEXT_INSTRUCTION
    video_instruction: str = DEFAULT_VIDEO_INSTRUCTION
    fps: float | None = 2.0
    max_frames: int | None = 128
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.7
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    video_decode_workers: int = 1
    video_prefetch_batches: int = 0
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from transformers import AutoProcessor
        from vllm import LLM

        if self.video_decode_workers < 1:
            raise ValueError("video_decode_workers must be positive")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            runner="pooling",
            trust_remote_code=True,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt={"video": 1},
            **self.engine_kwargs,
        )

    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        prompts = [{"prompt": self._chat_text(self.text_instruction, {"type": "text", "text": t})}
                   for t in texts]
        return self._embed(prompts, batch_size)

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray:
        if not videos:
            raise ValueError("No videos to encode")
        embeddings = []
        with ThreadPoolExecutor(max_workers=self.video_decode_workers) as pool:
            for start in range(0, len(videos), batch_size):
                chunk = videos[start : start + batch_size]
                prompts = list(pool.map(self._video_prompt, chunk))
                embeddings.append(self._embed(prompts, batch_size))
        return np.concatenate(embeddings, axis=0)

    def _chat_text(self, instruction: str, content: dict[str, Any]) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [content]},
        ]
        return self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    def _video_prompt(self, video: str | Path) -> dict[str, Any]:
        frames, metadata = sample_video_with_metadata(
            video, fps=self.fps if self.fps is not None else 2.0, max_frames=self.max_frames
        )
        return {
            "prompt": self._chat_text(self.video_instruction, {"type": "video", "video": frames}),
            "multi_modal_data": {"video": (frames, metadata)},
            # Frames are already sampled; do not let the processor resample.
            "mm_processor_kwargs": {"do_sample_frames": False},
        }

    def _embed(self, prompts: Sequence[Any], batch_size: int) -> np.ndarray:
        embeddings = []
        for start in range(0, len(prompts), batch_size):
            outputs = self.llm.embed(list(prompts[start : start + batch_size]))
            embeddings.extend(np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs)
        return np.stack(embeddings)


def register_qwen3vl_models() -> None:
    def make(default_model_name: str):
        def factory(**kwargs: Any) -> EmbeddingModel:
            if kwargs.pop("backend", "vllm") != "vllm":
                raise ValueError("Qwen3-VL-Embedding only implements the vLLM backend")
            for key in ("device", "video_sampling", "num_frames", "model_kwargs"):
                kwargs.pop(key, None)
            kwargs.setdefault("model_name", default_model_name)
            return Qwen3VLVLLMEmbedding(**kwargs)

        return factory

    for name, model_name in QWEN3VL_CHECKPOINTS.items():
        register_model(name, make(model_name))
