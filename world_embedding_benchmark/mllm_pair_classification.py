"""Pair classification by prompting a generative multimodal model.

The embedding evaluator compares two similarity scores. A generative model
has no such score, so each manifest triple becomes two binary questions, one
against the within-family negative and one against the cross-family negative,
and the model is asked which of the two videos matches the caption.

Generative models are sensitive to the order of the options, so every question
is asked twice with the candidates swapped. Scoring both answers independently
and averaging puts the metric on the same scale as the embedding evaluator: a
model that always names the same position lands at the 0.5 chance level.
``both_orders_accuracy`` is the stricter conjunction, where chance is 0.25.

Answers are constrained to "A" or "B" by vLLM structured output, so nothing
has to be parsed out of free text.

Only Qwen multimodal checkpoints have been tested. The prompt is assembled
with the processor's chat template and video frames are passed with the
metadata those models expect.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .pair_classification_data import MANIFEST_VERSION
from .retrieval import load_retrieval_data

NEGATIVE_ROLES = {"within_family": "within_family_negative",
                  "cross_family": "cross_family_negative"}
SYSTEM_PROMPT = "You are an expert at analysing physics simulation videos."
QUESTION = (
    'Text description: "{caption}"\n\n'
    "Video A and Video B are two physics simulation videos. "
    "Exactly one of them matches the text description above.\n"
    "Which video matches the description? Answer with a single letter, A or B."
)


def sample_uniform_frames(video: str | Path, num_frames: int) -> tuple[Any, dict[str, Any]]:
    """Take ``num_frames`` evenly spaced frames across the whole clip.

    The retrieval adapters sample at a fixed frame rate, which would cut a
    long clip short. Here the question is about the whole physical process, so
    the sample is spread over the full duration instead.
    """
    import numpy as np
    import torch
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video), dimension_order="NCHW", num_ffmpeg_threads=1)
    total_frames = int(decoder.metadata.num_frames)
    target = min(num_frames, total_frames)
    target -= target % 2
    if target < 2:
        raise RuntimeError(f"No valid frame count for {video}")
    indices = torch.linspace(0, total_frames - 1, steps=target).round().long()
    metadata = {
        "fps": float(decoder.metadata.average_fps),
        "frames_indices": indices.tolist(),
        "total_num_frames": total_frames,
        "do_sample_frames": False,
        "video_backend": "torchcodec",
    }
    return np.asarray(decoder.get_frames_at(indices).data.cpu()), metadata


@dataclass
class MLLMPairClassificationResult:
    dataset_dir: str
    manifest: str
    branch: str
    model: str
    num_questions: int
    metrics: dict[str, dict[str, float]]
    per_family: dict[str, dict[str, dict[str, float]]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def build_questions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand each triple into two tasks times two candidate orderings."""
    questions = []
    for example in manifest["examples"]:
        by_role = {c["role"]: c["video_id"] for c in example["candidates"]}
        for task, negative_role in NEGATIVE_ROLES.items():
            for positive_first in (True, False):
                positive, negative = by_role["positive"], by_role[negative_role]
                questions.append({
                    "example_id": example["example_id"],
                    "family": example["family"],
                    "query_id": example["query_id"],
                    "task": task,
                    "positive_first": positive_first,
                    "video_a": positive if positive_first else negative,
                    "video_b": negative if positive_first else positive,
                    "answer": "A" if positive_first else "B",
                })
    return questions


def _score(records: list[dict[str, Any]]) -> dict[str, float]:
    by_example: dict[str, dict[bool, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_example[record["example_id"]][record["positive_first"]] = record
    orderings = [pair for pair in by_example.values() if len(pair) == 2]
    if not orderings:
        raise ValueError("Every example needs both candidate orderings")
    answers = [record for pair in orderings for record in pair.values()]
    return {
        "accuracy": sum(r["predicted"] == r["answer"] for r in answers) / len(answers),
        "both_orders_accuracy": sum(
            all(r["predicted"] == r["answer"] for r in pair.values()) for pair in orderings
        ) / len(orderings),
        "pick_a_rate": sum(r["predicted"] == "A" for r in answers) / len(answers),
        "num_examples": len(orderings),
    }


def evaluate_mllm_pair_classification(
    *,
    manifest_path: str | Path,
    model_name: str,
    dataset_dir: str | Path | None = None,
    num_frames: int = 16,
    batch_size: int = 4,
    max_model_len: int = 49152,
    gpu_memory_utilization: float = 0.88,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 32,
    predictions_output: str | Path | None = None,
) -> MLLMPairClassificationResult:
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    dataset_dir = Path(dataset_dir or manifest["dataset_dir"])
    data = load_retrieval_data(dataset_dir, text_column=manifest["text_column"])
    caption_of = {query.query_id: query.text for query in data.queries}
    path_of = {video.item_id: video.video_path for video in data.videos}

    questions = build_questions(manifest)
    clips = {
        video_id: sample_uniform_frames(path_of[video_id], num_frames)
        for video_id in sorted({q[key] for q in questions for key in ("video_a", "video_b")})
    }

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        # Hybrid-attention checkpoints hold one cache block per decode
        # sequence, and the default of 1024 does not fit on one GPU.
        max_num_seqs=max_num_seqs,
        limit_mm_per_prompt={"video": 2},
    )
    sampling = SamplingParams(
        max_tokens=4,
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(choice=["A", "B"]),
    )

    def prompt_for(question: dict[str, Any]) -> dict[str, Any]:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": "Video A:"}, {"type": "video"},
                {"type": "text", "text": "Video B:"}, {"type": "video"},
                {"type": "text", "text": QUESTION.format(caption=caption_of[question["query_id"]])},
            ]},
        ]
        try:
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:          # template without a thinking switch
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        return {
            "prompt": text,
            "multi_modal_data": {"video": [clips[question["video_a"]], clips[question["video_b"]]]},
            "mm_processor_kwargs": {"do_sample_frames": False},
        }

    records = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        outputs = llm.generate([prompt_for(q) for q in chunk], sampling)
        for question, output in zip(chunk, outputs):
            records.append({**question, "predicted": output.outputs[0].text.strip().upper()})

    metrics = {task: _score([r for r in records if r["task"] == task]) for task in NEGATIVE_ROLES}
    families = sorted({r["family"] for r in records})
    per_family = {
        family: {
            task: _score([r for r in records if r["task"] == task and r["family"] == family])
            for task in NEGATIVE_ROLES
        }
        for family in families
    }

    if predictions_output:
        import csv

        path = Path(predictions_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    return MLLMPairClassificationResult(
        dataset_dir=str(dataset_dir),
        manifest=str(manifest_path.resolve()),
        branch=manifest["branch"],
        model=model_name,
        num_questions=len(records),
        metrics=metrics,
        per_family=per_family,
    )
