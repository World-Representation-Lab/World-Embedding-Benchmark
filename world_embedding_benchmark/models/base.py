from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    def encode_texts(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...

    def encode_videos(self, videos: Sequence[str | Path], *, batch_size: int) -> np.ndarray: ...


ModelFactory = Callable[..., EmbeddingModel]
_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    _REGISTRY[name] = factory


def get_model(name: str, **kwargs: Any) -> EmbeddingModel:
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc


