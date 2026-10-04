from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .retrieval import Query, RetrievalData, load_retrieval_data

MANIFEST_VERSION = 1


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(":".join((str(seed), *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _choice(values: Sequence[str], seed: int, *parts: str) -> str:
    rng = np.random.default_rng(_seed(seed, *parts))
    return values[int(rng.integers(len(values)))]


def dataset_fingerprint(data: RetrievalData) -> str:
    records = {
        "queries": sorted((q.query_id, q.text, sorted(q.relevant_ids), q.family) for q in data.queries),
        "videos": sorted((v.item_id, v.family) for v in data.videos),
    }
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build_pair_classification_manifest(
    data: RetrievalData, *, branch: str, dataset_dir: str | Path,
    text_column: str = "parsed_text", queries_per_family: int = 10, seed: int = 42,
) -> dict[str, Any]:
    """Sample query/positive/within-negative/cross-negative triples reproducibly."""
    if queries_per_family < 1:
        raise ValueError("queries_per_family must be positive")
    video_family = {v.item_id: v.family for v in data.videos}
    if len(video_family) != len(data.videos):
        raise ValueError("Video IDs must be unique")
    videos_by_family: dict[str, list[str]] = defaultdict(list)
    for video in data.videos:
        if video.family is not None:
            videos_by_family[str(video.family)].append(video.item_id)
    videos_by_family = {k: sorted(set(v)) for k, v in videos_by_family.items()}

    eligible: dict[str, list[Query]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    for query in data.queries:
        family = str(query.family) if query.family is not None else None
        relevant = {i for i in query.relevant_ids if i in video_family}
        positives = {i for i in relevant if str(video_family[i]) == family}
        within = set(videos_by_family.get(family or "", ())) - relevant
        has_cross = any(ids for other, ids in videos_by_family.items() if other != family)
        if family is None:
            excluded["missing_family"] += 1
        elif not positives:
            excluded["missing_same_family_positive"] += 1
        elif not within:
            excluded["missing_within_family_negative"] += 1
        elif not has_cross:
            excluded["missing_cross_family_negative"] += 1
        else:
            eligible[family].append(query)
    if not eligible:
        raise ValueError("No eligible queries")

    examples, summaries = [], {}
    for family in sorted(eligible):
        pool = sorted(eligible[family], key=lambda q: q.query_id)
        if len(pool) < queries_per_family:
            raise ValueError(f"Family {family!r}: only {len(pool)} eligible queries")
        rng = np.random.default_rng(_seed(seed, branch, family, "queries"))
        # This samples across the complete family pool, never by taking its head.
        selected = [pool[int(i)] for i in rng.choice(len(pool), queries_per_family, replace=False)]
        for rank, query in enumerate(selected):
            relevant = set(query.relevant_ids)
            positives = sorted(i for i in relevant if i in video_family and str(video_family[i]) == family)
            within = sorted(set(videos_by_family[family]) - relevant)
            cross = sorted(i for other, ids in videos_by_family.items() if other != family for i in ids)
            ids = (
                _choice(positives, seed, branch, family, query.query_id, "positive"),
                _choice(within, seed, branch, family, query.query_id, "within"),
                _choice(cross, seed, branch, family, query.query_id, "cross"),
            )
            roles = ("positive", "within_family_negative", "cross_family_negative")
            examples.append({
                "example_id": f"{branch}:{family}:{rank:02d}", "family": family,
                "query_id": query.query_id,
                "candidates": [{"index": i, "role": roles[i], "video_id": ids[i]} for i in range(3)],
            })
        summaries[family] = {"available_queries": len(pool), "sampled_queries": queries_per_family,
                             "available_videos": len(videos_by_family[family])}
    return {
        "version": MANIFEST_VERSION, "task": "text_video_pair_classification",
        "branch": branch, "dataset_dir": str(Path(dataset_dir)),
        "text_column": text_column, "dataset_fingerprint": dataset_fingerprint(data),
        "sampling": {"strategy": "uniform_without_replacement_per_family", "base_seed": seed,
                     "queries_per_family": queries_per_family, "negative_trials_per_query": 1,
                     "candidate_order": ["positive", "within_family_negative", "cross_family_negative"]},
        "num_families": len(summaries), "num_examples": len(examples),
        "families": summaries, "excluded_queries": dict(sorted(excluded.items())), "examples": examples,
    }


def create_pair_classification_manifest(
    dataset_dir: str | Path, output: str | Path, *, branch: str | None = None,
    text_column: str = "parsed_text", queries_per_family: int = 10, seed: int = 42,
) -> dict[str, Any]:
    dataset_dir, output = Path(dataset_dir), Path(output)
    branch = branch or dataset_dir.name
    manifest = build_pair_classification_manifest(
        load_retrieval_data(dataset_dir, text_column=text_column), branch=branch,
        dataset_dir=dataset_dir, text_column=text_column,
        queries_per_family=queries_per_family, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError(f"Refusing to overwrite different manifest: {output}")
        return manifest
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return manifest
