from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from world_embedding_benchmark.retrieval import load_retrieval_data
from tqdm.auto import tqdm
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info


@dataclass(frozen=True)
class Row:
    case_id: str
    family: str
    query: str
    video_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal single-file LCO text-video retrieval debugger."
    )
    parser.add_argument("--dataset-dir", default="datasets/physics-bench-solid-eval")
    parser.add_argument("--text-column", default="parsed_text")
    parser.add_argument("--model-name", default="/workspace/mteb-main/checkpoints/LCO-Embedding-Omni-3B")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-videos-per-family", type=int)
    parser.add_argument("--family", action="append", help="Restrict to one or more families.")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--input-device", default="cuda")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--save", help="Write JSON debug results.")
    parser.add_argument("--save-scores", help="Write overall score matrix as .npy.")
    parser.add_argument("--save-matrices-dir", help="Write labeled overall and per-family similarity matrices as CSV files.")
    parser.add_argument("--print-topk", type=int, default=5)
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> list[Row]:
    data = load_retrieval_data(args.dataset_dir, text_column=args.text_column)
    families = set(args.family or [])
    query_by_case_id = {query.relevant_ids[0]: query for query in data.queries}
    rows = [
        Row(
            case_id=video.item_id,
            family=str(video.family),
            query=query_by_case_id[video.item_id].text,
            video_path=str(video.video_path),
        )
        for video in data.videos
        if video.item_id in query_by_case_id
        and (not families or video.family in families)
    ]
    if args.limit_videos and args.limit_videos_per_family:
        raise ValueError("Use either --limit-videos or --limit-videos-per-family, not both")
    if args.limit_videos_per_family:
        rows = limit_rows_per_family(rows, args.limit_videos_per_family)
    elif args.limit_videos:
        rows = rows[: args.limit_videos]
    return rows



def limit_rows_per_family(rows: list[Row], limit: int) -> list[Row]:
    counts: Counter[str] = Counter()
    out: list[Row] = []
    for row in rows:
        if counts[row.family] < limit:
            out.append(row)
            counts[row.family] += 1
    return out


def load_lco(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = getattr(torch, args.dtype)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        # device_map=args.device_map,
    ).to("cuda")
    model.eval()
    return processor, model


def maybe_normalize(x: torch.Tensor, normalize: bool) -> torch.Tensor:
    if normalize:
        return torch.nn.functional.normalize(x.float(), p=2, dim=-1)
    return x.float()


@torch.no_grad()
def encode_texts(
    texts: list[str], processor: Any, model: Any, args: argparse.Namespace
) -> torch.Tensor:
    prompt = "{}\nSummarize the above text in one word:"
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="encode text"):
        batch_texts = [prompt.format(text) for text in texts[start : start + args.batch_size]]
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": text}]}]
            for text in batch_texts
        ]
        text_inputs = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(args.input_device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        emb = outputs.hidden_states[-1][:, -1, :]
        chunks.append(maybe_normalize(emb, not args.no_normalize).cpu())
        del inputs, outputs, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_videos(
    videos: list[str], processor: Any, model: Any, args: argparse.Namespace
) -> torch.Tensor:
    video_prompt = "\nSummarize the above video in one word:"
    chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(videos), args.batch_size), desc="encode video"):
        batch_videos = videos[start : start + args.batch_size]
        messages = []
        for video in batch_videos:
            video_item: dict[str, Any] = {"type": "video", "video": video}
            if args.fps is not None:
                video_item["fps"] = args.fps
            if args.max_frames is not None:
                video_item["max_frames"] = args.max_frames
            if args.max_pixels is not None:
                video_item["max_pixels"] = args.max_pixels
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            video_item,
                            {"type": "text", "text": video_prompt},
                        ],
                    }
                ]
            )

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio_inputs, image_inputs, video_inputs = process_mm_info(
            messages, use_audio_in_video=args.use_audio_in_video
        )
        inputs = processor(
            text=text,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(args.input_device)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        emb = outputs.hidden_states[-1][:, -1, :]
        chunks.append(maybe_normalize(emb, not args.no_normalize).cpu())
        del inputs, outputs, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def save_similarity_matrices(
    scores: np.ndarray,
    rows: list[Row],
    output_dir: str | Path,
    *,
    prefix: str = "text_video",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_similarity_csv(output_dir / f"{prefix}_overall.csv", scores, rows, rows)

    families = sorted({row.family for row in rows})
    for family in families:
        idxs = [i for i, row in enumerate(rows) if row.family == family]
        family_scores = scores[np.ix_(idxs, idxs)]
        family_rows = [rows[i] for i in idxs]
        write_similarity_csv(
            output_dir / f"{prefix}_family_{safe_filename(family)}.csv",
            family_scores,
            family_rows,
            family_rows,
        )


def write_similarity_csv(
    path: Path, scores: np.ndarray, query_rows: list[Row], video_rows: list[Row]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_family", *[row.case_id for row in video_rows]])
        for i, row in enumerate(query_rows):
            writer.writerow([row.case_id, row.family, *[float(x) for x in scores[i]]])


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def embedding_diagnostics(sim: np.ndarray, rows: list[Row], *, name: str) -> dict[str, Any]:
    off_diag = sim[~np.eye(sim.shape[0], dtype=bool)] if sim.shape[0] > 1 else np.array([])
    nearest = []
    nearest_ids = []
    ids = [row.case_id for row in rows]
    if sim.shape[0] > 1:
        masked = sim.copy()
        np.fill_diagonal(masked, -np.inf)
        nearest_idx = np.argmax(masked, axis=1)
        nearest = [float(masked[i, j]) for i, j in enumerate(nearest_idx)]
        nearest_ids = [ids[int(j)] for j in nearest_idx]

    top_neighbor_hist = dict(Counter(nearest_ids))
    return {
        "name": name,
        "num_items": len(rows),
        "mean_offdiag_similarity": float(np.mean(off_diag)) if off_diag.size else 0.0,
        "std_offdiag_similarity": float(np.std(off_diag)) if off_diag.size else 0.0,
        "min_offdiag_similarity": float(np.min(off_diag)) if off_diag.size else 0.0,
        "max_offdiag_similarity": float(np.max(off_diag)) if off_diag.size else 0.0,
        "mean_nearest_neighbor_similarity": float(np.mean(nearest)) if nearest else 0.0,
        "top_neighbor_unique_count": len(top_neighbor_hist),
        "top_neighbor_id_histogram": top_neighbor_hist,
    }


def rank_debug(
    scores: np.ndarray, rows: list[Row], topk: int, *, include_per_family: bool = True
) -> dict[str, Any]:
    ids = [r.case_id for r in rows]
    families = [r.family for r in rows]
    order = np.argsort(-scores, axis=1)
    ranks: list[int] = []
    top1: list[str] = []
    examples = []
    for i, ranked in enumerate(order):
        rank = int(np.where(ranked == i)[0][0] + 1)
        ranks.append(rank)
        top1.append(ids[int(ranked[0])])
        examples.append(
            {
                "query_id": ids[i],
                "family": families[i],
                "positive_rank": rank,
                "positive_score": float(scores[i, i]),
                "top": [
                    {
                        "rank": j + 1,
                        "case_id": ids[int(idx)],
                        "family": families[int(idx)],
                        "score": float(scores[i, int(idx)]),
                        "is_positive": int(idx) == i,
                    }
                    for j, idx in enumerate(ranked[:topk])
                ],
            }
        )

    out: dict[str, Any] = {
        "num_queries": len(rows),
        "num_videos": len(rows),
        "mean_positive_rank": float(np.mean(ranks)),
        "recall@1": float(np.mean([r <= 1 for r in ranks])),
        "recall@5": float(np.mean([r <= min(5, len(rows)) for r in ranks])),
        "recall@10": float(np.mean([r <= min(10, len(rows)) for r in ranks])),
        "positive_rank_histogram": dict(Counter(str(r) for r in ranks)),
        "top1_id_histogram": dict(Counter(top1)),
        "top1_unique_count": len(set(top1)),
        "mean_score_std": float(np.mean(np.std(scores, axis=1))),
        "examples": examples,
    }

    if include_per_family:
        per_family: dict[str, Any] = {}
        for family in sorted(set(families)):
            idxs = [i for i, row in enumerate(rows) if row.family == family]
            family_scores = scores[np.ix_(idxs, idxs)]
            family_rows = [rows[i] for i in idxs]
            per_family[family] = rank_debug(
                family_scores,
                family_rows,
                topk=min(topk, len(idxs)),
                include_per_family=False,
            )
        out["per_family"] = per_family
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args)
    print(f"Loaded {len(rows)} rows from {args.dataset_dir}")
    print("Families:", dict(Counter(row.family for row in rows)))
    processor, model = load_lco(args)

    text_emb = encode_texts([row.query for row in rows], processor, model, args)
    video_emb = encode_videos([row.video_path for row in rows], processor, model, args)
    scores = (text_emb @ video_emb.T).numpy()
    text_text_scores = (text_emb @ text_emb.T).numpy()
    video_video_scores = (video_emb @ video_emb.T).numpy()
    result = rank_debug(scores, rows, args.print_topk)
    result["embedding_diagnostics"] = {
        "text_text": embedding_diagnostics(text_text_scores, rows, name="text_text"),
        "video_video": embedding_diagnostics(video_video_scores, rows, name="video_video"),
    }
    result.update(
        {
            "dataset_dir": args.dataset_dir,
            "model_name": args.model_name,
            "query_column": args.text_column,
            "normalize": not args.no_normalize,
            "fps": args.fps,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
        }
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if args.save_scores:
        path = Path(args.save_scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)
        np.save(path.with_name(path.stem + "_text_text.npy"), text_text_scores)
        np.save(path.with_name(path.stem + "_video_video.npy"), video_video_scores)
    if args.save_matrices_dir:
        save_similarity_matrices(scores, rows, args.save_matrices_dir, prefix="text_video")
        save_similarity_matrices(text_text_scores, rows, args.save_matrices_dir, prefix="text_text")
        save_similarity_matrices(video_video_scores, rows, args.save_matrices_dir, prefix="video_video")


if __name__ == "__main__":
    main()
