from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = 1


def encode_with_checkpoints(
    *,
    modality: str,
    ids: Sequence[str],
    items: Sequence[Any],
    encode: Callable[[Sequence[Any]], np.ndarray],
    output_dir: str | Path | None,
    checkpoint_size: int,
    signature: dict[str, Any],
) -> np.ndarray:
    """Encode ordered items, atomically checkpointing reusable shards."""
    if len(ids) != len(items):
        raise ValueError(f"{modality} IDs/items differ: {len(ids)} != {len(items)}")
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    if output_dir is None:
        return np.asarray(encode(items), dtype=np.float32)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected_manifest = {
        "version": CACHE_VERSION,
        "signature": signature,
        "checkpoint_size": checkpoint_size,
    }
    if manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected_manifest:
            raise ValueError(
                f"Embedding cache configuration changed: {manifest_path}. "
                "Use a new output directory or restore the original arguments."
            )
    else:
        _atomic_write_text(
            manifest_path, json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
        )

    shards: list[np.ndarray] = []
    for start in range(0, len(items), checkpoint_size):
        stop = min(start + checkpoint_size, len(items))
        shard_ids = list(ids[start:stop])
        path = root / f"{modality}-{start:08d}-{stop:08d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                saved_ids = saved["ids"].astype(str).tolist()
                embeddings = np.asarray(saved["embeddings"], dtype=np.float32)
            if saved_ids != shard_ids:
                raise ValueError(f"ID mismatch in cached shard {path}")
            if embeddings.shape[0] != len(shard_ids):
                raise ValueError(f"Row-count mismatch in cached shard {path}")
            print(f"Reusing {path} ({len(shard_ids)} {modality} items)")
        else:
            embeddings = np.asarray(encode(items[start:stop]), dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != len(shard_ids):
                raise ValueError(
                    f"Unexpected {modality} embedding shape {embeddings.shape}; "
                    f"expected ({len(shard_ids)}, dimension)"
                )
            _atomic_savez(path, ids=np.asarray(shard_ids), embeddings=embeddings)
            print(f"Saved {path} ({len(shard_ids)} {modality} items)")
        shards.append(embeddings)

    if not shards:
        raise ValueError(f"No {modality} items to encode")
    dimensions = {shard.shape[1] for shard in shards}
    if len(dimensions) != 1:
        raise ValueError(f"Cached {modality} shards have inconsistent dimensions")
    return np.concatenate(shards, axis=0)


def save_embedding_archive(
    path: str | Path,
    *,
    ids: Sequence[str],
    families: Sequence[str | None],
    embeddings: np.ndarray,
) -> None:
    _atomic_savez(
        Path(path),
        ids=np.asarray(ids),
        families=np.asarray([family or "" for family in families]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def save_similarity_matrices(
    output_dir: str | Path,
    *,
    text_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_save(root / "text_video_similarity.npy", text_embeddings @ video_embeddings.T)
    _atomic_save(root / "text_text_similarity.npy", text_embeddings @ text_embeddings.T)
    _atomic_save(root / "video_video_similarity.npy", video_embeddings @ video_embeddings.T)


def save_family_confusions(
    output_dir: str | Path,
    *,
    query_families: Sequence[str | None],
    video_families: Sequence[str | None],
    text_video_scores: np.ndarray,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    labels = sorted(
        {family for family in [*query_families, *video_families] if family is not None}
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    text_to_video = _family_confusion(
        source_families=query_families,
        predicted_families=[video_families[i] for i in text_video_scores.argmax(axis=1)],
        label_to_index=label_to_index,
    )
    video_to_text = _family_confusion(
        source_families=video_families,
        predicted_families=[query_families[i] for i in text_video_scores.argmax(axis=0)],
        label_to_index=label_to_index,
    )
    for name, matrix in (
        ("text_to_video_family_confusion", text_to_video),
        ("video_to_text_family_confusion", video_to_text),
    ):
        _write_confusion_csv(root / f"{name}.csv", labels, matrix)
        _atomic_write_text(
            root / f"{name}.json",
            json.dumps({"labels": labels, "counts": matrix.tolist()}, indent=2) + "\n",
        )


def _family_confusion(
    *,
    source_families: Sequence[str | None],
    predicted_families: Sequence[str | None],
    label_to_index: dict[str, int],
) -> np.ndarray:
    matrix = np.zeros((len(label_to_index), len(label_to_index)), dtype=np.int64)
    for source, predicted in zip(source_families, predicted_families, strict=True):
        if source in label_to_index and predicted in label_to_index:
            matrix[label_to_index[source], label_to_index[predicted]] += 1
    return matrix


def _write_confusion_csv(path: Path, labels: Sequence[str], matrix: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_family", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row.tolist()])
    os.replace(temporary, path)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32))
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
