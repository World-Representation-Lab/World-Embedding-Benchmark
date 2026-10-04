"""V-JEPA 2 video encoders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import EmbeddingModel, register_model

NUM_FRAMES = 64          # fpc64 in the checkpoint name; fixed by the architecture
FRAME_SIZE = 256         # 256 in the checkpoint name
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

VJEPA2_CHECKPOINTS = {
    "vjepa2-vitl": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2-vitg": "facebook/vjepa2-vitg-fpc64-256",
}


@dataclass
class VJEPA2Embedding:
    """Encode video only; V-JEPA 2 has no text tower.

    Three choices differ from the upstream evaluation recipe, each trading
    fidelity to the pretraining distribution for fidelity to the benchmark:

    - Frames are sampled uniformly across the whole clip rather than as a
      16 fps window, because the physical properties under test (a pendulum
      period, a coefficient of restitution) are not visible in a 4 second cut.
    - Frames are resized to a square rather than short-side resized and centre
      cropped, which keeps the full field of view for laterally moving scenes.
    - Patch tokens are mean pooled. Upstream uses a trained attentive probe,
      which does not fit a frozen-embedding protocol and would not be
      comparable to the other models here.
    """

    model_name: str
    num_frames: int = NUM_FRAMES
    frame_size: int = FRAME_SIZE
    dtype: str = "bfloat16"
    device: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import torch
        from transformers import VJEPA2Model

        self._torch = torch
        self._dtype = getattr(torch, self.dtype)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            VJEPA2Model.from_pretrained(self.model_name, dtype=self._dtype, **self.model_kwargs)
            .to(self._device)
            .eval()
        )

    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        raise NotImplementedError(
            "V-JEPA 2 is video only; use it with run_regression.py or "
            "run_regression_scaling.py, not retrieval or pair classification"
        )

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray:
        if not videos:
            raise ValueError("No videos to encode")
        torch = self._torch
        out = []
        for start in range(0, len(videos), batch_size):
            clips = np.stack([self._preprocess(v) for v in videos[start : start + batch_size]])
            with torch.no_grad():
                hidden = self.model(
                    pixel_values_videos=torch.from_numpy(clips).to(self._device, self._dtype),
                    skip_predictor=True,
                ).last_hidden_state
            out.append(hidden.float().mean(dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def _preprocess(self, video: str | Path) -> np.ndarray:
        import cv2

        capture = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            raise RuntimeError(f"Decoded no frames from {video}")

        indices = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        clip = np.empty((self.num_frames, 3, self.frame_size, self.frame_size), dtype=np.float32)
        for slot, index in enumerate(indices):
            frame = cv2.resize(frames[int(index)][:, :, ::-1], (self.frame_size, self.frame_size))
            frame = np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))
            clip[slot] = (frame - IMAGENET_MEAN) / IMAGENET_STD
        return clip


def register_vjepa2_models() -> None:
    def make(default_model_name: str):
        def factory(**kwargs: Any) -> EmbeddingModel:
            if kwargs.pop("backend", "transformers") != "transformers":
                raise ValueError("V-JEPA 2 only implements the Transformers backend")
            for key in ("video_sampling", "fps", "max_frames", "max_model_len",
                        "gpu_memory_utilization", "enforce_eager",
                        "video_prefetch_batches", "video_decode_workers", "video_decoder"):
                kwargs.pop(key, None)
            # The CLIs always pass --num-frames, defaulting to None, but 64 is
            # an architectural constant here.
            if kwargs.get("num_frames") is None:
                kwargs.pop("num_frames", None)
            kwargs.setdefault("model_name", default_model_name)
            return VJEPA2Embedding(**kwargs)

        return factory

    for name, model_name in VJEPA2_CHECKPOINTS.items():
        register_model(name, make(model_name))
