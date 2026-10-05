"""Omni-Embed-Nemotron-3B through Transformers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import EmbeddingModel, register_model
from .lco_embedding import sample_qwen_processor_video_frames_torchcodec

# The prefixes the checkpoint ships in config_sentence_transformers.json.
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_PASSAGE_PREFIX = "passage: "

NV_OMNI_CHECKPOINTS = {"nv-omni-embed-3b": "nvidia/omni-embed-nemotron-3b"}


@dataclass
class NVOmniEmbedTransformers:
    """Embed text and video with masked mean pooling over the last hidden state.

    vLLM is not an option: its 0.23 Qwen2.5-Omni processor lacks
    ``_get_num_multimodal_tokens`` so loading fails outright, and this
    checkpoint replaces the text tower with a bidirectional variant that the
    causal vLLM implementation would silently mis-encode.
    """

    model_name: str
    dtype: str = "bfloat16"
    query_prefix: str = DEFAULT_QUERY_PREFIX
    passage_prefix: str = DEFAULT_PASSAGE_PREFIX
    fps: float | None = 2.0
    max_frames: int | None = 128
    max_pixels: int = 64 * 28 * 28
    min_pixels: int = 32 * 14 * 14
    max_length: int = 32768
    device: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "right"
        self.model = (
            AutoModel.from_pretrained(
                self.model_name,
                dtype=getattr(torch, self.dtype),
                trust_remote_code=True,
                **self.model_kwargs,
            )
            .to(self._device)
            .eval()
        )

    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            content = [
                [{"role": "user", "content": [{"type": "text", "text": self.query_prefix + text}]}]
                for text in texts[start : start + batch_size]
            ]
            out.append(self._forward(self._processor_batch(content)))
        return np.concatenate(out, axis=0)

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray:
        if not videos:
            raise ValueError("No videos to encode")
        out = []
        for start in range(0, len(videos), batch_size):
            frames = [
                sample_qwen_processor_video_frames_torchcodec(
                    video,
                    fps=self.fps if self.fps is not None else 2.0,
                    max_frames=self.max_frames,
                )
                for video in videos[start : start + batch_size]
            ]
            content = [
                [{"role": "user", "content": [
                    {"type": "text", "text": self.passage_prefix},
                    {"type": "video", "video": clip},
                ]}]
                for clip in frames
            ]
            out.append(self._forward(self._processor_batch(content, frames)))
        return np.concatenate(out, axis=0)

    def _processor_batch(self, content: list[Any], frames: list[Any] | None = None) -> Any:
        prompts = [
            self.processor.apply_chat_template(item, add_generation_prompt=False, tokenize=False)
            for item in content
        ]
        kwargs: dict[str, Any] = {
            "text": prompts,
            "return_tensors": "pt",
            "text_kwargs": {"truncation": True, "padding": True, "max_length": self.max_length},
        }
        if frames is not None:
            # Frames are already sampled; keep the processor from resampling.
            kwargs["videos"] = frames
            kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "use_audio_in_video": False,
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
        return self.processor(**kwargs)

    def _forward(self, batch: Any) -> np.ndarray:
        torch = self._torch
        batch = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            hidden = self.model(**batch, output_hidden_states=True).hidden_states[-1]
        mask = batch["attention_mask"]
        masked = hidden.masked_fill(~mask[..., None].bool(), 0.0)
        pooled = masked.sum(dim=1) / mask.sum(dim=1)[..., None]
        return torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy()


def register_nv_omni_models() -> None:
    def make(default_model_name: str):
        def factory(**kwargs: Any) -> EmbeddingModel:
            if kwargs.pop("backend", "transformers") != "transformers":
                raise ValueError(
                    "omni-embed-nemotron-3b only implements the Transformers backend"
                )
            for key in ("video_sampling", "num_frames", "max_model_len",
                        "gpu_memory_utilization", "enforce_eager",
                        "video_prefetch_batches", "video_decode_workers", "video_decoder"):
                kwargs.pop(key, None)
            kwargs.setdefault("model_name", default_model_name)
            return NVOmniEmbedTransformers(**kwargs)

        return factory

    for name, model_name in NV_OMNI_CHECKPOINTS.items():
        register_model(name, make(model_name))
