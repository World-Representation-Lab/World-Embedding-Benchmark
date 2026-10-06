from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedding_artifacts import (
    encode_with_checkpoints,
    save_embedding_archive,
    save_family_confusions,
    save_similarity_matrices,
)

from .models import EmbeddingModel


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_ids: tuple[str, ...]
    family: str | None = None
    query_type: str | None = None


@dataclass(frozen=True)
class VideoItem:
    item_id: str
    video_path: Path
    family: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalData:
    queries: list[Query]
    videos: list[VideoItem]


@dataclass
class RetrievalResult:
    name: str
    direction: str
    dataset_dir: str
    model: str
    num_queries: int
    num_corpus: int
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


@dataclass
class DirectionalRetrievalResult:
    overall: RetrievalResult
    per_family: dict[str, RetrievalResult]


@dataclass
class RetrievalSuiteResult:
    dataset_dir: str
    model: str
    text_column: str
    text_to_video: DirectionalRetrievalResult
    video_to_text: DirectionalRetrievalResult

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def evaluate_text_video_retrieval(
    model: EmbeddingModel,
    *,
    dataset_dir: str | Path,
    model_name: str,
    text_column: str = "parsed_text",
    query_types: set[str] | None = None,
    limit_queries: int | None = None,
    limit_videos: int | None = None,
    limit_videos_per_family: int | None = None,
    batch_size: int = 4,
    k_values: Sequence[int] = (1, 5, 10),
) -> RetrievalResult:
    return evaluate_text_video_retrieval_suite(
        model,
        dataset_dir=dataset_dir,
        model_name=model_name,
        text_column=text_column,
        query_types=query_types,
        limit_queries=limit_queries,
        limit_videos=limit_videos,
        limit_videos_per_family=limit_videos_per_family,
        batch_size=batch_size,
        k_values=k_values,
        per_family=False,
    ).text_to_video.overall


def evaluate_text_video_retrieval_suite(
    model: EmbeddingModel,
    *,
    dataset_dir: str | Path,
    model_name: str,
    text_column: str = "parsed_text",
    query_types: set[str] | None = None,
    limit_queries: int | None = None,
    limit_videos: int | None = None,
    limit_videos_per_family: int | None = None,
    batch_size: int = 4,
    k_values: Sequence[int] = (1, 5, 10),
    per_family: bool = True,
    embedding_output_dir: str | Path | None = None,
    checkpoint_size: int = 32,
    cache_signature: dict[str, Any] | None = None,
    save_similarities: bool = False,
    save_family_confusion: bool = False,
) -> RetrievalSuiteResult:
    dataset_dir = Path(dataset_dir)
    data = load_retrieval_data(dataset_dir, text_column=text_column, query_types=query_types)
    if limit_videos is not None and limit_videos_per_family is not None:
        raise ValueError("Use either limit_videos or limit_videos_per_family, not both")

    videos = data.videos
    if limit_videos_per_family is not None:
        videos = limit_items_per_family(videos, limit_videos_per_family)
    elif limit_videos is not None:
        videos = videos[:limit_videos]
    video_ids = {video.item_id for video in videos}
    queries = [q for q in data.queries if any(doc_id in video_ids for doc_id in q.relevant_ids)]
    if limit_queries is not None:
        queries = queries[:limit_queries]
    if limit_queries is not None:
        selected_video_ids = {item_id for q in queries for item_id in q.relevant_ids}
        videos = [video for video in videos if video.item_id in selected_video_ids]
    if not queries:
        raise ValueError("No queries have positives in the selected video set")

    artifact_dir = Path(embedding_output_dir) if embedding_output_dir is not None else None
    signature = {
        "dataset_dir": str(dataset_dir.resolve()),
        "model": model_name,
        "text_column": text_column,
        "query_ids": [query.query_id for query in queries],
        "video_ids": [video.item_id for video in videos],
        **(cache_signature or {}),
    }
    query_embeddings = encode_with_checkpoints(
        modality="text",
        ids=[query.query_id for query in queries],
        items=[query.text for query in queries],
        encode=lambda items: model.encode_texts(items, batch_size=batch_size),
        output_dir=artifact_dir,
        checkpoint_size=checkpoint_size,
        signature=signature,
    )
    video_embeddings = encode_with_checkpoints(
        modality="video",
        ids=[video.item_id for video in videos],
        items=[video.video_path for video in videos],
        encode=lambda items: model.encode_videos(items, batch_size=batch_size),
        output_dir=artifact_dir,
        checkpoint_size=checkpoint_size,
        signature=signature,
    )
    if artifact_dir is not None:
        save_embedding_archive(
            artifact_dir / "text_embeddings.npz",
            ids=[query.query_id for query in queries],
            families=[query.family for query in queries],
            embeddings=query_embeddings,
        )
        save_embedding_archive(
            artifact_dir / "video_embeddings.npz",
            ids=[video.item_id for video in videos],
            families=[video.family for video in videos],
            embeddings=video_embeddings,
        )
        text_video_scores = query_embeddings @ video_embeddings.T
        if save_similarities:
            save_similarity_matrices(
                artifact_dir,
                text_embeddings=query_embeddings,
                video_embeddings=video_embeddings,
            )
        if save_family_confusion:
            save_family_confusions(
                artifact_dir,
                query_families=[query.family for query in queries],
                video_families=[video.family for video in videos],
                text_video_scores=text_video_scores,
            )

    text_to_video_overall = _evaluate_text_to_video_slice(
        name="overall",
        dataset_dir=dataset_dir,
        model_name=model_name,
        queries=queries,
        videos=videos,
        query_embeddings=query_embeddings,
        video_embeddings=video_embeddings,
        query_indices=list(range(len(queries))),
        video_indices=list(range(len(videos))),
        k_values=k_values,
    )
    video_to_text_overall = _evaluate_video_to_text_slice(
        name="overall",
        dataset_dir=dataset_dir,
        model_name=model_name,
        queries=queries,
        videos=videos,
        query_embeddings=query_embeddings,
        video_embeddings=video_embeddings,
        query_indices=list(range(len(queries))),
        video_indices=list(range(len(videos))),
        k_values=k_values,
    )

    text_to_video_families: dict[str, RetrievalResult] = {}
    video_to_text_families: dict[str, RetrievalResult] = {}
    if per_family:
        video_by_id = {v.item_id: i for i, v in enumerate(videos)}
        families = sorted({v.family for v in videos if v.family is not None})
        for family in families:
            family_video_ids = [v.item_id for v in videos if v.family == family]
            video_indices = [video_by_id[item_id] for item_id in family_video_ids]
            family_video_id_set = set(family_video_ids)
            query_indices = [
                i
                for i, query in enumerate(queries)
                if query.family == family
                and any(item_id in family_video_id_set for item_id in query.relevant_ids)
            ]
            if not query_indices or not video_indices:
                continue
            text_to_video_families[family] = _evaluate_text_to_video_slice(
                name=f"family/{family}",
                dataset_dir=dataset_dir,
                model_name=model_name,
                queries=queries,
                videos=videos,
                query_embeddings=query_embeddings,
                video_embeddings=video_embeddings,
                query_indices=query_indices,
                video_indices=video_indices,
                k_values=k_values,
            )
            video_to_text_families[family] = _evaluate_video_to_text_slice(
                name=f"family/{family}",
                dataset_dir=dataset_dir,
                model_name=model_name,
                queries=queries,
                videos=videos,
                query_embeddings=query_embeddings,
                video_embeddings=video_embeddings,
                query_indices=query_indices,
                video_indices=video_indices,
                k_values=k_values,
            )

    return RetrievalSuiteResult(
        dataset_dir=str(dataset_dir),
        model=model_name,
        text_column=text_column,
        text_to_video=DirectionalRetrievalResult(
            overall=text_to_video_overall, per_family=text_to_video_families
        ),
        video_to_text=DirectionalRetrievalResult(
            overall=video_to_text_overall, per_family=video_to_text_families
        ),
    )

def limit_items_per_family(videos: Sequence[VideoItem], limit: int) -> list[VideoItem]:
    if limit <= 0:
        raise ValueError("limit_videos_per_family must be positive")
    counts: dict[str | None, int] = {}
    limited: list[VideoItem] = []
    for video in videos:
        count = counts.get(video.family, 0)
        if count < limit:
            limited.append(video)
            counts[video.family] = count + 1
    return limited



def _evaluate_text_to_video_slice(
    *,
    name: str,
    dataset_dir: Path,
    model_name: str,
    queries: Sequence[Query],
    videos: Sequence[VideoItem],
    query_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
    query_indices: Sequence[int],
    video_indices: Sequence[int],
    k_values: Sequence[int],
) -> RetrievalResult:
    q_emb = query_embeddings[list(query_indices)]
    v_emb = video_embeddings[list(video_indices)]
    sliced_queries = [queries[i] for i in query_indices]
    sliced_videos = [videos[i] for i in video_indices]
    scores = q_emb @ v_emb.T
    return _build_retrieval_result(
        name=name,
        direction="text_to_video",
        dataset_dir=dataset_dir,
        model_name=model_name,
        scores=scores,
        relevant_ids=[q.relevant_ids for q in sliced_queries],
        corpus_ids=[v.item_id for v in sliced_videos],
        num_queries=len(sliced_queries),
        num_corpus=len(sliced_videos),
        k_values=k_values,
    )


def _evaluate_video_to_text_slice(
    *,
    name: str,
    dataset_dir: Path,
    model_name: str,
    queries: Sequence[Query],
    videos: Sequence[VideoItem],
    query_embeddings: np.ndarray,
    video_embeddings: np.ndarray,
    query_indices: Sequence[int],
    video_indices: Sequence[int],
    k_values: Sequence[int],
) -> RetrievalResult:
    q_emb = query_embeddings[list(query_indices)]
    v_emb = video_embeddings[list(video_indices)]
    sliced_queries = [queries[i] for i in query_indices]
    sliced_videos = [videos[i] for i in video_indices]
    scores = v_emb @ q_emb.T
    relevant_query_ids_by_video: dict[str, list[str]] = {
        video.item_id: [] for video in sliced_videos
    }
    for query in sliced_queries:
        for video_id in query.relevant_ids:
            if video_id in relevant_query_ids_by_video:
                relevant_query_ids_by_video[video_id].append(query.query_id)
    return _build_retrieval_result(
        name=name,
        direction="video_to_text",
        dataset_dir=dataset_dir,
        model_name=model_name,
        scores=scores,
        relevant_ids=[tuple(relevant_query_ids_by_video[v.item_id]) for v in sliced_videos],
        corpus_ids=[q.query_id for q in sliced_queries],
        num_queries=len(sliced_videos),
        num_corpus=len(sliced_queries),
        k_values=k_values,
    )


def _build_retrieval_result(
    *,
    name: str,
    direction: str,
    dataset_dir: Path,
    model_name: str,
    scores: np.ndarray,
    relevant_ids: Sequence[Sequence[str]],
    corpus_ids: Sequence[str],
    num_queries: int,
    num_corpus: int,
    k_values: Sequence[int],
) -> RetrievalResult:
    metrics = retrieval_metrics(
        scores,
        relevant_ids,
        corpus_ids,
        k_values=k_values,
    )
    diagnostics = retrieval_diagnostics(
        scores,
        relevant_ids,
        corpus_ids,
    )
    return RetrievalResult(
        name=name,
        direction=direction,
        dataset_dir=str(dataset_dir),
        model=model_name,
        num_queries=num_queries,
        num_corpus=num_corpus,
        metrics=metrics,
        diagnostics=diagnostics,
    )

def load_retrieval_data(
    dataset_dir: str | Path,
    *,
    text_column: str = "parsed_text",
    query_types: set[str] | None = None,
) -> RetrievalData:
    dataset_dir = Path(dataset_dir)
    table_data = _load_hf_table_data(dataset_dir, text_column=text_column)
    if table_data is not None:
        return table_data

    queries = load_queries(dataset_dir / "queries.jsonl", query_types=query_types)
    videos = load_video_items(dataset_dir)
    video_family = {v.item_id: v.family for v in videos}
    queries = [
        Query(q.query_id, q.text, q.relevant_ids, video_family.get(q.relevant_ids[0]), q.query_type)
        for q in queries
    ]
    return RetrievalData(queries=queries, videos=videos)


def _load_hf_table_data(dataset_dir: Path, *, text_column: str) -> RetrievalData | None:
    family_parquets = sorted(dataset_dir.glob("*/test-*.parquet"))
    if not family_parquets:
        family_parquets = sorted(dataset_dir.glob("*/train-*.parquet"))
    if family_parquets:
        return _load_family_parquet_data(
            dataset_dir, family_parquets=family_parquets, text_column=text_column
        )

    split_name = next(
        (name for name in ("test", "train") if (dataset_dir / name / "metadata.parquet").exists()),
        None,
    )
    if split_name is None:
        return None

    from datasets import Video, load_dataset

    dataset = load_dataset(str(dataset_dir))[split_name].cast_column("video", Video(decode=False))
    if text_column not in dataset.column_names:
        raise ValueError(
            f"Query column {text_column!r} not found. Available columns: {dataset.column_names}"
        )

    queries: list[Query] = []
    videos: list[VideoItem] = []
    for row in dataset:
        item_id = str(row["case_id"])
        family = str(row["family"]) if row.get("family") is not None else None
        video_path = Path(row["video"]["path"])
        if not video_path.exists():
            video_path = dataset_dir / split_name / "videos" / f"{item_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video for {item_id!r} not found: {video_path}")
        videos.append(VideoItem(item_id=item_id, video_path=video_path, family=family, metadata=dict(row)))
        queries.append(
            Query(
                query_id=item_id,
                text=str(row[text_column]),
                relevant_ids=(item_id,),
                family=family,
                query_type=text_column,
            )
        )
    return RetrievalData(queries=queries, videos=videos)


def _load_family_parquet_data(
    dataset_dir: Path,
    *,
    family_parquets: Sequence[Path],
    text_column: str,
) -> RetrievalData:
    """Load one embedded-video Parquet file per evaluation family."""
    from datasets import Video, load_dataset

    cache_dir = dataset_dir / ".video_cache"
    queries: list[Query] = []
    videos: list[VideoItem] = []
    seen_query_ids: set[str] = set()
    seen_video_ids: set[str] = set()
    for parquet_path in family_parquets:
        family = parquet_path.parent.name
        dataset = load_dataset(
            "parquet", data_files={"test": str(parquet_path)}, split="test"
        ).cast_column("video", Video(decode=False))
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Text column {text_column!r} not found in {parquet_path}. "
                f"Available columns: {dataset.column_names}"
            )
        required = {"query_id", "case_id", "video"}
        missing = required.difference(dataset.column_names)
        if missing:
            raise ValueError(f"Missing columns in {parquet_path}: {sorted(missing)}")
        for row in dataset:
            query_id = str(row["query_id"])
            case_id = str(row["case_id"])
            if query_id in seen_query_ids:
                raise ValueError(f"Duplicate query_id {query_id!r}")
            seen_query_ids.add(query_id)
            if case_id not in seen_video_ids:
                video_path = _materialize_video(
                    row["video"], cache_dir=cache_dir, case_id=case_id
                )
                metadata = {k: v for k, v in row.items() if k != "video"}
                videos.append(VideoItem(case_id, video_path, family, metadata))
                seen_video_ids.add(case_id)
            queries.append(
                Query(
                    query_id=query_id,
                    text=str(row[text_column]),
                    relevant_ids=(case_id,),
                    family=family,
                    query_type=text_column,
                )
            )
    if not videos:
        raise FileNotFoundError(f"No evaluation rows found under {dataset_dir}")
    return RetrievalData(queries=queries, videos=videos)


def _materialize_video(video: Any, *, cache_dir: Path, case_id: str) -> Path:
    if not isinstance(video, dict):
        raise TypeError(f"Unexpected video value for {case_id!r}: {type(video).__name__}")
    video_bytes = video.get("bytes")
    source_path = video.get("path")
    if video_bytes is None:
        if source_path and Path(source_path).exists():
            return Path(source_path)
        raise FileNotFoundError(f"Video bytes are missing for {case_id!r}")
    suffix = Path(source_path or "video.mp4").suffix or ".mp4"
    output_path = cache_dir / f"{case_id}{suffix}"
    if output_path.exists() and output_path.stat().st_size == len(video_bytes):
        return output_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(video_bytes)
    os.replace(temporary_path, output_path)
    return output_path



def load_queries(path: str | Path, *, query_types: set[str] | None = None) -> list[Query]:
    queries: list[Query] = []
    for row in read_jsonl(path):
        query_type = row.get("type")
        if query_types is not None and query_type not in query_types:
            continue
        queries.append(
            Query(
                query_id=str(row["query_id"]),
                text=str(row["text"]),
                relevant_ids=tuple(str(x) for x in row["gt_item_ids"]),
                query_type=query_type,
            )
        )
    return queries


def load_video_items(dataset_dir: str | Path) -> list[VideoItem]:
    dataset_dir = Path(dataset_dir)
    preview_path = dataset_dir / "metadata_preview.jsonl"
    if preview_path.exists():
        items = []
        for row in read_jsonl(preview_path):
            video_path = dataset_dir / "train" / row["file_name"]
            if video_path.exists():
                items.append(
                    VideoItem(str(row["case_id"]), video_path, row.get("family"), row)
                )
        if items:
            return items

    cases_path = dataset_dir / "release_cases.jsonl"
    items = []
    for row in read_jsonl(cases_path):
        video_path = Path(row["assets"]["video_path"])
        if not video_path.exists():
            fallback = dataset_dir / "train" / "videos" / f"{row['case_id']}.mp4"
            video_path = fallback
        if video_path.exists():
            items.append(
                VideoItem(str(row["case_id"]), video_path, row.get("family"), row)
            )
    if not items:
        raise FileNotFoundError(f"No local videos found under {dataset_dir}")
    return items


def retrieval_diagnostics(
    scores: np.ndarray,
    relevant_ids: Sequence[Sequence[str]],
    corpus_ids: Sequence[str],
    *,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    order = np.argsort(-scores, axis=1)
    corpus_ids = list(corpus_ids)
    positive_ranks: list[int] = []
    score_stds: list[float] = []
    top_tie_counts: list[int] = []
    positives_tied_at_top = 0
    top1_ids: list[str] = []

    for i, ranked_indices in enumerate(order):
        positives = set(relevant_ids[i])
        ranked_ids = [corpus_ids[j] for j in ranked_indices]
        top1_ids.append(ranked_ids[0])
        rank = next(
            (idx + 1 for idx, doc_id in enumerate(ranked_ids) if doc_id in positives),
            len(corpus_ids) + 1,
        )
        positive_ranks.append(rank)

        row = scores[i]
        top_score = float(np.max(row))
        top_ties = int(np.sum(np.abs(row - top_score) <= tie_tolerance))
        top_tie_counts.append(top_ties)
        score_stds.append(float(np.std(row)))
        if any(
            doc_id in positives
            for doc_id, score in zip(corpus_ids, row, strict=True)
            if abs(float(score) - top_score) <= tie_tolerance
        ):
            positives_tied_at_top += 1

    rank_hist: dict[str, int] = {}
    for rank in positive_ranks:
        rank_hist[str(rank)] = rank_hist.get(str(rank), 0) + 1

    top1_hist: dict[str, int] = {}
    for item_id in top1_ids:
        top1_hist[item_id] = top1_hist.get(item_id, 0) + 1

    return {
        "positive_rank_histogram": rank_hist,
        "top1_id_histogram": top1_hist,
        "top1_unique_count": len(top1_hist),
        "mean_positive_rank": float(np.mean(positive_ranks)) if positive_ranks else 0.0,
        "mean_score_std": float(np.mean(score_stds)) if score_stds else 0.0,
        "all_scores_tied_fraction": float(
            np.mean([std <= tie_tolerance for std in score_stds])
        )
        if score_stds
        else 0.0,
        "mean_top_tie_count": float(np.mean(top_tie_counts)) if top_tie_counts else 0.0,
        "positive_tied_at_top_fraction": positives_tied_at_top / max(1, len(positive_ranks)),
    }


def retrieval_metrics(
    scores: np.ndarray,
    relevant_ids: Sequence[Sequence[str]],
    corpus_ids: Sequence[str],
    *,
    k_values: Sequence[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    order = np.argsort(-scores, axis=1)
    corpus_ids = list(corpus_ids)

    for k in k_values:
        recalls = []
        mrrs = []
        ndcgs = []
        for i, ranked_indices in enumerate(order):
            positives = set(relevant_ids[i])
            top = [corpus_ids[j] for j in ranked_indices[:k]]
            hits = [1 if doc_id in positives else 0 for doc_id in top]
            recalls.append(sum(hits) / max(1, len(positives)))
            mrrs.append(_reciprocal_rank(hits))
            ndcgs.append(_ndcg(hits, ideal_hits=min(len(positives), k)))
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"mrr@{k}"] = float(np.mean(mrrs))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return metrics


def _reciprocal_rank(hits: Sequence[int]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def _ndcg(hits: Sequence[int], *, ideal_hits: int) -> float:
    if ideal_hits == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float(np.sum(np.asarray(hits) * discounts))
    ideal = float(np.sum(discounts[:ideal_hits]))
    return dcg / ideal if ideal else 0.0


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
