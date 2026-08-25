"""vLLM adapter for standalone LCO Qwen2.5-Omni thinker checkpoints."""

from transformers import PretrainedConfig
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import Qwen2_5OmniThinkerForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper

VLLM_LCO_ARCHITECTURE = "LCOQwen2_5OmniForConditionalGeneration"


def wrap_lco_thinker_config(config: PretrainedConfig) -> PretrainedConfig:
    wrapped = Qwen2_5OmniConfig(thinker_config=config.to_dict())
    wrapped.architectures = [VLLM_LCO_ARCHITECTURE]
    return wrapped


class LCOQwen2_5OmniForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    """Native vLLM thinker with LCO's standalone weight prefixes."""

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
        "visual.": "visual.",
        "audio_tower.": "audio_tower.",
    })
