from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import warnings

import numpy as np

from .base import EmbeddingModel, register_model


_FIXED_VIDEO_SENTINEL = "__FIXED_SAMPLED_VIDEO__"
TEXT_COMPRESSION_SUFFIX = "\nSummarize the above text in one word:"
VIDEO_COMPRESSION_SUFFIX = "\nSummarize the above video in one word:"


@dataclass
class LCOEmbedding:
    """Minimal LCO-Embedding wrapper for text-video retrieval.

    video_sampling:
        ``"processor"`` delegates video decoding/sampling to Qwen Omni's
        ``process_mm_info`` path. ``"fixed"`` decodes uniformly spaced frames
        before calling the processor, matching the MTEB-style sampling policy.
    """

    model_name: str = "LCO-Embedding/LCO-Embedding-Omni-3B"
    device: str | None = None
    dtype: str = "bfloat16"
    video_sampling: str = "processor"
    fps: float | None = 2.0
    max_frames: int | None = 64
    num_frames: int | None = None
    max_pixels: int | None = None
    use_audio_in_video: bool = False
    model_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.video_sampling not in {"processor", "fixed"}:
            raise ValueError("video_sampling must be 'processor' or 'fixed'")
        if self.video_sampling == "fixed" and self.num_frames is None and self.fps is None:
            raise ValueError("fixed sampling needs num_frames or fps")

        import torch
        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, self.dtype)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            **(self.model_kwargs or {}),
        ).to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return self._encode_batches(list(texts), batch_size=batch_size, modality="text")

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray:
        return self._encode_batches(
            [str(Path(v)) for v in videos], batch_size=batch_size, modality="video"
        )

    def _encode_batches(
        self, items: list[str], *, batch_size: int, modality: str
    ) -> np.ndarray:
        import torch
        from tqdm.auto import tqdm

        embeddings: list[Any] = []
        with torch.no_grad():
            for start in tqdm(range(0, len(items), batch_size), desc=f"Encoding {modality}"):
                batch = items[start : start + batch_size]
                inputs = self._build_inputs(batch, modality=modality)
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                ).hidden_states[-1][:, -1, :]
                outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
                embeddings.append(outputs.float().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _build_inputs(self, batch: list[str], *, modality: str) -> Any:
        if modality == "text":
            suffix = TEXT_COMPRESSION_SUFFIX
            messages = [
                [{"role": "user", "content": [{"type": "text", "text": text + suffix}]}]
                for text in batch
            ]
            text = self._chat_text(messages)
            return self.processor(text=text, padding=True, return_tensors="pt").to(
                self.device
            )

        suffix = VIDEO_COMPRESSION_SUFFIX
        if self.video_sampling == "processor":
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": video, **self._video_kwargs()},
                            {"type": "text", "text": suffix},
                        ],
                    }
                ]
                for video in batch
            ]
            return self._processor_video_inputs(messages)

        frames = [
            sample_video_frames(
                v, fps=self.fps, max_frames=self.max_frames, num_frames=self.num_frames
            )
            for v in batch
        ]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": _FIXED_VIDEO_SENTINEL},
                        {"type": "text", "text": suffix},
                    ],
                }
            ]
            for _ in batch
        ]
        text = self._chat_text(messages)
        self._assert_video_sentinel_not_tokenized(text)
        return self.processor(
            text=text,
            videos=frames,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_sample_frames": False, "use_audio_in_video": False},
        ).to(self.device)

    def _processor_video_inputs(self, messages: list[list[dict[str, Any]]]) -> Any:
        from qwen_omni_utils import process_mm_info

        text = self._chat_text(messages)
        audio_inputs, image_inputs, video_inputs = process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        return self.processor(
            text=text,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _video_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.fps is not None:
            kwargs["fps"] = self.fps
        if self.max_frames is not None:
            kwargs["max_frames"] = self.max_frames
        if self.max_pixels is not None:
            kwargs["max_pixels"] = self.max_pixels
        return kwargs

    def _chat_text(self, messages: list[list[dict[str, Any]]]) -> list[str]:
        return [
            self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]

    @staticmethod
    def _assert_video_sentinel_not_tokenized(texts: Sequence[str]) -> None:
        if any(_FIXED_VIDEO_SENTINEL in text for text in texts):
            raise RuntimeError(
                "Qwen chat template rendered the fixed-video sentinel as text. "
                "Do not use fixed frame sampling with this template until the "
                "prompt/video-token construction is updated."
            )


def sample_video_frames(
    video_path: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    num_frames: int | None = None,
) -> Any:
    """Return TCHW RGB frames sampled uniformly or by target FPS."""

    if num_frames is not None and fps is not None:
        raise ValueError("Cannot specify both num_frames and fps")

    # Probe first and ask FFmpeg to emit only the selected source frames.
    # This is byte-identical to full decode followed by indexing, but bounds
    # raw-frame memory by the requested sample count.
    info = _probe_video_with_ffmpeg(video_path)
    import torch

    n_source = int(info["num_frames"])
    if num_frames is not None:
        target = min(num_frames, n_source)
        indices = torch.linspace(0, n_source - 1, steps=target).round().long()
    elif fps is not None:
        source_fps = float(info["video_fps"])
        duration = n_source / source_fps
        target = max(1, int(duration * fps))
        if max_frames is not None:
            target = min(target, max_frames)
        target = min(target, n_source)
        indices = torch.linspace(0, n_source - 1, steps=target).round().long()
    else:
        indices = torch.arange(n_source)
    return _read_video_with_ffmpeg(video_path, indices=indices.tolist(), info=info)[0]


def sample_qwen_processor_video_frames(
    video_path: str | Path, *, fps: float = 2.0, max_frames: int | None = None
) -> Any:
    """Reproduce Qwen Omni FPS sampling while decoding only selected frames."""
    import torch

    info = _probe_video_with_ffmpeg(video_path)
    total_frames = int(info["num_frames"])
    source_fps = float(info["video_fps"])
    maximum = min(max_frames if max_frames is not None else 768, total_frames)
    maximum -= maximum % 2
    minimum = min(4, maximum)
    target = min(max(total_frames / source_fps * fps, minimum), maximum, total_frames)
    target = int(target) - int(target) % 2
    if target < 2:
        raise RuntimeError(f"No valid Qwen frame count for {video_path}")
    indices = torch.linspace(0, total_frames - 1, steps=target).round().long()
    return _read_video_with_ffmpeg(video_path, indices=indices.tolist(), info=info)[0]


def sample_qwen_processor_video_frames_torchcodec(
    video_path: str | Path, *, fps: float = 2.0, max_frames: int | None = None
) -> Any:
    """Reproduce Qwen Omni FPS sampling with TorchCodec indexed decoding."""
    import torch
    decoder = _load_torchcodec_video_decoder()(
        video_path, dimension_order="NCHW", num_ffmpeg_threads=1
    )
    total_frames = int(decoder.metadata.num_frames)
    source_fps = float(decoder.metadata.average_fps)
    maximum = min(max_frames if max_frames is not None else 768, total_frames)
    maximum -= maximum % 2
    minimum = min(4, maximum)
    target = min(max(total_frames / source_fps * fps, minimum), maximum, total_frames)
    target = int(target) - int(target) % 2
    if target < 2:
        raise RuntimeError(f"No valid Qwen frame count for {video_path}")
    indices = torch.linspace(0, total_frames - 1, steps=target).round().long()
    return decoder.get_frames_at(indices).data


def resolve_video_decoder(requested: str, *, video_sampling: str) -> str:
    """Resolve ``auto`` to TorchCodec when available, otherwise FFmpeg."""
    if requested not in {"auto", "torchcodec", "ffmpeg"}:
        raise ValueError("video_decoder must be auto, torchcodec, or ffmpeg")
    if video_sampling == "fixed":
        if requested == "torchcodec":
            raise ValueError("fixed video sampling currently requires ffmpeg")
        return "ffmpeg"
    if requested == "ffmpeg":
        return "ffmpeg"
    try:
        _load_torchcodec_video_decoder()
    except (ImportError, OSError, RuntimeError) as exc:
        if requested == "torchcodec":
            raise RuntimeError(
                "TorchCodec was requested but could not be loaded. Install a "
                "TorchCodec version compatible with PyTorch and system FFmpeg."
            ) from exc
        warnings.warn(
            f"TorchCodec is unavailable ({exc}); falling back to FFmpeg.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "ffmpeg"
    return "torchcodec"


def _load_torchcodec_video_decoder() -> Any:
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder


def _read_video_with_ffmpeg(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    info: dict[str, float] | None = None,
) -> tuple[Any, dict[str, float]]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Video sampling needs system ffmpeg and ffprobe")
    import torch

    path = str(video_path)
    info = info or _probe_video_with_ffmpeg(video_path)
    width = int(info["width"])
    height = int(info["height"])
    command = ["ffmpeg", "-v", "error", "-i", path]
    if indices is not None:
        if not indices:
            raise ValueError("indices must not be empty")
        expression = "+".join(f"eq(n\\,{index})" for index in indices)
        command.extend(["-vf", f"select={expression}", "-vsync", "0"])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    frame_size = width * height * 3
    n_frames = len(raw) // frame_size
    if n_frames == 0:
        raise RuntimeError(f"No decodable frames in {video_path}")
    if indices is not None and n_frames != len(indices):
        raise RuntimeError(
            f"FFmpeg returned {n_frames} frames for {len(indices)} requested "
            f"indices from {video_path}"
        )
    array = np.frombuffer(raw[: n_frames * frame_size], dtype=np.uint8)
    array = array.reshape(n_frames, height, width, 3).copy()
    return torch.from_numpy(array).permute(0, 3, 1, 2), info


def _probe_video_with_ffmpeg(video_path: str | Path) -> dict[str, float]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(video_path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    fps = _parse_rate(stream.get("avg_frame_rate", "0/1"))
    frame_count = stream.get("nb_frames")
    if frame_count in (None, "N/A"):
        frame_count = round(float(stream.get("duration") or 0.0) * fps)
    info = {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "video_fps": fps,
        "num_frames": int(frame_count),
    }
    if info["video_fps"] <= 0 or info["num_frames"] <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return info


def _parse_rate(rate: str) -> float:
    num, _, den = rate.partition("/")
    denominator = float(den or 1.0)
    return float(num) / denominator if denominator else 0.0



LCO_CHECKPOINTS: dict[str, str] = {
    "lco-omni-3b": "LCO-Embedding/LCO-Embedding-Omni-3B",
    "lco-omni-7b": "LCO-Embedding/LCO-Embedding-Omni-7B",
}


def make_lco_factory(default_model_name: str):
    def factory(**kwargs: Any) -> EmbeddingModel:
        backend = kwargs.pop("backend", "transformers")
        kwargs.setdefault("model_name", default_model_name)
        if backend == "transformers":
            return LCOEmbedding(**kwargs)
        if backend == "vllm":
            from .lco_vllm import LCOVLLMEmbedding

            return LCOVLLMEmbedding(**kwargs)
        raise ValueError("backend must be 'transformers' or 'vllm'")

    return factory


def register_lco_models() -> None:
    for name, model_name in LCO_CHECKPOINTS.items():
        register_model(name, make_lco_factory(model_name))
