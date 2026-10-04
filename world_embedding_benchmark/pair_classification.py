from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embedding_artifacts import encode_with_checkpoints, save_embedding_archive
from .models import EmbeddingModel
from .pair_classification_data import MANIFEST_VERSION, dataset_fingerprint
from .retrieval import load_retrieval_data


@dataclass
class PairClassificationResult:
    dataset_dir: str
    manifest: str
    branch: str
    model: str
    num_examples: int
    num_unique_queries: int
    num_unique_videos: int
    metrics: dict[str, float]
    per_family: dict[str, dict[str, float]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def evaluate_pair_classification(
    model: EmbeddingModel,
    *,
    manifest_path: str | Path,
    dataset_dir: str | Path | None = None,
    model_name: str,
    batch_size: int = 4,
    embedding_output_dir: str | Path | None = None,
    checkpoint_size: int = 32,
    predictions_output: str | Path | None = None,
    cache_signature: dict[str, Any] | None = None,
) -> PairClassificationResult:
    """Evaluate fixed text/video triples using only the generic model protocol."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    dataset_dir = Path(dataset_dir or manifest["dataset_dir"])
    data = load_retrieval_data(dataset_dir, text_column=manifest["text_column"])
    if dataset_fingerprint(data) != manifest["dataset_fingerprint"]:
        raise ValueError("Source retrieval data does not match the manifest fingerprint")

    query_by_id = {query.query_id: query for query in data.queries}
    video_by_id = {video.item_id: video for video in data.videos}
    query_ids = list(dict.fromkeys(example["query_id"] for example in manifest["examples"]))
    video_ids = list(dict.fromkeys(
        candidate["video_id"]
        for example in manifest["examples"] for candidate in example["candidates"]
    ))
    missing_queries = sorted(set(query_ids) - set(query_by_id))
    missing_videos = sorted(set(video_ids) - set(video_by_id))
    if missing_queries or missing_videos:
        raise ValueError(
            f"Manifest references missing IDs: {len(missing_queries)} queries, "
            f"{len(missing_videos)} videos"
        )

    signature = {
        "task": "pair_classification",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "model": model_name,
        "query_ids": query_ids,
        "video_ids": video_ids,
        **(cache_signature or {}),
    }
    artifact_dir = Path(embedding_output_dir) if embedding_output_dir else None
    query_embeddings = encode_with_checkpoints(
        modality="text", ids=query_ids, items=[query_by_id[i].text for i in query_ids],
        encode=lambda items: model.encode_texts(items, batch_size=batch_size),
        output_dir=artifact_dir, checkpoint_size=checkpoint_size, signature=signature,
    )
    video_embeddings = encode_with_checkpoints(
        modality="video", ids=video_ids, items=[video_by_id[i].video_path for i in video_ids],
        encode=lambda items: model.encode_videos(items, batch_size=batch_size),
        output_dir=artifact_dir, checkpoint_size=checkpoint_size, signature=signature,
    )
    if query_embeddings.shape[1] != video_embeddings.shape[1]:
        raise ValueError("Text and video embedding dimensions differ")
    if artifact_dir:
        save_embedding_archive(
            artifact_dir / "text_embeddings.npz", ids=query_ids,
            families=[query_by_id[i].family for i in query_ids], embeddings=query_embeddings)
        save_embedding_archive(
            artifact_dir / "video_embeddings.npz", ids=video_ids,
            families=[video_by_id[i].family for i in video_ids], embeddings=video_embeddings)

    q_index, v_index = ({v: i for i, v in enumerate(query_ids)},
                        {v: i for i, v in enumerate(video_ids)})
    rows = []
    for example in manifest["examples"]:
        candidates = sorted(example["candidates"], key=lambda value: value["index"])
        if [candidate["role"] for candidate in candidates] != [
            "positive", "within_family_negative", "cross_family_negative"
        ]:
            raise ValueError(f"Invalid candidate roles in {example['example_id']}")
        query_embedding = query_embeddings[q_index[example["query_id"]]]
        scores = [float(query_embedding @ video_embeddings[v_index[c["video_id"]]])
                  for c in candidates]
        rows.append({
            "example_id": example["example_id"], "family": example["family"],
            "query_id": example["query_id"],
            "positive_video_id": candidates[0]["video_id"],
            "within_negative_video_id": candidates[1]["video_id"],
            "cross_negative_video_id": candidates[2]["video_id"],
            "positive_score": scores[0], "within_negative_score": scores[1],
            "cross_negative_score": scores[2],
        })
    metrics = _metrics(rows)
    families = sorted({row["family"] for row in rows})
    per_family = {family: _metrics([row for row in rows if row["family"] == family])
                  for family in families}
    for key in list(metrics):
        metrics[f"macro_family_{key}"] = float(np.mean([value[key] for value in per_family.values()]))
    if predictions_output:
        _write_predictions(Path(predictions_output), rows)
    return PairClassificationResult(
        dataset_dir=str(dataset_dir), manifest=str(manifest_path), branch=manifest["branch"],
        model=model_name, num_examples=len(rows), num_unique_queries=len(query_ids),
        num_unique_videos=len(video_ids), metrics=metrics, per_family=per_family)


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot score an empty set")
    positive = np.asarray([row["positive_score"] for row in rows])
    within = np.asarray([row["within_negative_score"] for row in rows])
    cross = np.asarray([row["cross_negative_score"] for row in rows])
    return {
        "within_family_accuracy": float(np.mean(positive > within)),
        "cross_family_accuracy": float(np.mean(positive > cross)),
        "three_way_accuracy": float(np.mean(positive > np.maximum(within, cross))),
        "within_family_mean_margin": float(np.mean(positive - within)),
        "cross_family_mean_margin": float(np.mean(positive - cross)),
        "within_family_tie_rate": float(np.mean(positive == within)),
        "cross_family_tie_rate": float(np.mean(positive == cross)),
    }


def _write_predictions(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
