from __future__ import annotations

from .base import EmbeddingModel, ModelFactory, get_model, register_model
from .lco_embedding import LCO_CHECKPOINTS, LCOEmbedding, register_lco_models
from .lco_vllm import LCOVLLMEmbedding
from .nv_omni_embed import NVOmniEmbedTransformers, register_nv_omni_models
from .qwen3vl_embedding import Qwen3VLVLLMEmbedding, register_qwen3vl_models
from .vjepa2 import VJEPA2Embedding, register_vjepa2_models

register_lco_models()
register_nv_omni_models()
register_qwen3vl_models()
register_vjepa2_models()

__all__ = [
    "EmbeddingModel",
    "LCOEmbedding",
    "LCOVLLMEmbedding",
    "NVOmniEmbedTransformers",
    "Qwen3VLVLLMEmbedding",
    "VJEPA2Embedding",
    "LCO_CHECKPOINTS",
    "ModelFactory",
    "get_model",
    "register_model",
]
