from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .regression import RegressionItem, load_regression_data

SPLIT_MANIFEST_VERSION = 1
DEFAULT_TRAIN_SIZES = (25, 50, 100, 200, 300, 400)
DEFAULT_REPETITIONS = 5
DEFAULT_SPLIT_SEED = 42
DEFAULT_REPETITION_SEEDS = (1000, 1001, 1002, 1003, 1004)


def build_regression_split_manifest(
    items: Sequence[RegressionItem],
    *,
    subset: str,
    split: str = "test",
    test_size: int = 100,
    num_strata: int = 10,
    split_seed: int = DEFAULT_SPLIT_SEED,
    train_sizes: Sequence[int] = DEFAULT_TRAIN_SIZES,
    repetition_seeds: Sequence[int] = DEFAULT_REPETITION_SEEDS,
) -> dict[str, Any]:
    if not 0 < test_size < len(items):
        raise ValueError("test_size must be between 1 and num_examples - 1")
    if not 1 <= num_strata <= test_size:
        raise ValueError("num_strata must be between 1 and test_size")
    if test_size % num_strata:
        raise ValueError("test_size must be divisible by num_strata")
    train_pool_size = len(items) - test_size
    train_sizes = tuple(int(size) for size in train_sizes)
    if not train_sizes or sorted(set(train_sizes)) != list(train_sizes):
        raise ValueError("train_sizes must be unique and strictly increasing")
    if train_sizes[-1] > train_pool_size or train_sizes[0] < 2:
        raise ValueError(f"train_sizes must be within [2, {train_pool_size}]")
    if not repetition_seeds:
        raise ValueError("At least one repetition seed is required")

    values = np.asarray([item.value for item in items], dtype=np.float64)
    order = np.asarray(sorted(range(len(items)), key=lambda i: (values[i], items[i].item_id)))
    strata = np.array_split(order, num_strata)
    per_stratum = test_size // num_strata
    effective_seed = _derived_seed(split_seed, subset)
    rng = np.random.default_rng(effective_seed)
    test_indices: list[int] = []
    stratum_by_index: dict[int, int] = {}
    for stratum_id, candidates in enumerate(strata):
        if len(candidates) < per_stratum:
            raise ValueError(f"Stratum {stratum_id} has too few examples")
        chosen = rng.choice(candidates, size=per_stratum, replace=False)
        test_indices.extend(int(index) for index in chosen)
        stratum_by_index.update({int(index): stratum_id for index in chosen})
    test_indices.sort()
    test_set = set(test_indices)
    train_indices = [index for index in range(len(items)) if index not in test_set]

    repetitions = []
    train_ids = [items[index].item_id for index in train_indices]
    for seed in repetition_seeds:
        repetition_rng = np.random.default_rng(_derived_seed(int(seed), subset))
        order_ids = [train_ids[index] for index in repetition_rng.permutation(len(train_ids))]
        repetitions.append({"seed": int(seed), "train_id_order": order_ids})

    return {
        "version": SPLIT_MANIFEST_VERSION,
        "dataset": "World-Embedding-Regression",
        "subset": subset,
        "source_split": split,
        "num_examples": len(items),
        "dataset_fingerprint": regression_dataset_fingerprint(items),
        "split": {
            "strategy": "equal_count_target_rank_strata",
            "base_seed": int(split_seed),
            "effective_seed": effective_seed,
            "num_strata": num_strata,
            "test_size": test_size,
            "examples_per_stratum": per_stratum,
        },
        "test_examples": [
            {
                "id": items[index].item_id,
                "original_index": index,
                "regression_value": items[index].value,
                "stratum": stratum_by_index[index],
            }
            for index in test_indices
        ],
        "training_pool_ids": train_ids,
        "scaling": {
            "train_sizes": list(train_sizes),
            "nested_prefixes": True,
            "repetitions": repetitions,
        },
    }


def create_regression_split_manifest(
    dataset_dir: str | Path,
    subset: str,
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    split = str(kwargs.get("split", "train"))
    items = load_regression_data(dataset_dir, subset, split=split)
    manifest = build_regression_split_manifest(items, subset=subset, **kwargs)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Refusing to overwrite a different split manifest: {path}. "
                "Choose another path or explicitly remove the old manifest."
            )
        return manifest
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return manifest


def load_and_validate_regression_split_manifest(
    path: str | Path,
    items: Sequence[RegressionItem],
    *,
    subset: str,
) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != SPLIT_MANIFEST_VERSION:
        raise ValueError(f"Unsupported split manifest version: {manifest.get('version')}")
    if manifest.get("subset") != subset:
        raise ValueError(f"Manifest is for {manifest.get('subset')!r}, not {subset!r}")
    fingerprint = regression_dataset_fingerprint(items)
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Dataset metadata does not match the split manifest fingerprint")
    all_ids = [item.item_id for item in items]
    known = set(all_ids)
    test_ids = [entry["id"] for entry in manifest["test_examples"]]
    train_ids = list(manifest["training_pool_ids"])
    if len(test_ids) != manifest["split"]["test_size"] or len(test_ids) != len(set(test_ids)):
        raise ValueError("Manifest test IDs are duplicated or have the wrong count")
    if len(train_ids) != len(set(train_ids)) or set(test_ids) & set(train_ids):
        raise ValueError("Manifest training/test IDs overlap or contain duplicates")
    if set(test_ids) | set(train_ids) != known:
        raise ValueError("Manifest IDs do not exactly partition the dataset")
    train_set = set(train_ids)
    for repetition in manifest["scaling"]["repetitions"]:
        order = repetition["train_id_order"]
        if len(order) != len(train_ids) or set(order) != train_set:
            raise ValueError("A scaling repetition is not a permutation of the training pool")
    return manifest


def regression_dataset_fingerprint(items: Sequence[RegressionItem]) -> str:
    records = [
        {
            "family": item.family,
            "id": item.item_id,
            "regression_attribute": item.attribute,
            "regression_value": item.value,
        }
        for item in items
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derived_seed(base_seed: int, subset: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{subset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")
