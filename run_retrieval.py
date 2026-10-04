from __future__ import annotations

import argparse
from pathlib import Path

from world_embedding_benchmark.models import get_model
from world_embedding_benchmark.retrieval import evaluate_text_video_retrieval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bidirectional text-video retrieval.")
    parser.add_argument("--dataset-dir", default="datasets/physics-bench-solid-eval")
    parser.add_argument("--model", default="lco-omni-3b")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--video-prefetch-batches", type=int, default=0)
    parser.add_argument("--video-decode-workers", type=int, default=1)
    parser.add_argument(
        "--video-decoder", choices=["auto", "ffmpeg", "torchcodec"], default="auto",
        help="vLLM video decoder; auto prefers TorchCodec and falls back to FFmpeg.",
    )
    parser.add_argument(
        "--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--limit-videos-per-family", type=int)
    parser.add_argument("--query-type", action="append")
    parser.add_argument("--text-column", default="parsed_text")
    parser.add_argument("--no-per-family", action="store_true")
    parser.add_argument("--video-sampling", choices=["processor", "fixed"], default="processor")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--device")
    parser.add_argument("--model-name", help="Override the checkpoint path for the selected model key.")
    parser.add_argument("--embedding-output-dir")
    parser.add_argument("--checkpoint-size", type=int, default=32)
    parser.add_argument("--save-similarity-matrices", action="store_true")
    parser.add_argument("--save-family-confusion", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.save_similarity_matrices or args.save_family_confusion) and not args.embedding_output_dir:
        raise ValueError("Analysis outputs require --embedding-output-dir")
    model_kwargs = {
        "backend": args.backend,
        "device": args.device,
        "video_sampling": args.video_sampling,
        "fps": None if args.num_frames is not None else args.fps,
        "max_frames": args.max_frames,
        "num_frames": args.num_frames,
    }
    if args.backend == "vllm":
        model_kwargs["max_model_len"] = args.vllm_max_model_len
        model_kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        model_kwargs["video_prefetch_batches"] = args.video_prefetch_batches
        model_kwargs["video_decode_workers"] = args.video_decode_workers
        model_kwargs["video_decoder"] = args.video_decoder
        model_kwargs["enforce_eager"] = args.vllm_enforce_eager
    if args.model_name:
        model_kwargs["model_name"] = args.model_name
    model = get_model(args.model, **model_kwargs)
    result = evaluate_text_video_retrieval_suite(
        model,
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        text_column=args.text_column,
        batch_size=args.batch_size,
        limit_queries=args.limit_queries,
        limit_videos=args.limit_videos,
        limit_videos_per_family=args.limit_videos_per_family,
        query_types=set(args.query_type) if args.query_type else None,
        per_family=not args.no_per_family,
        embedding_output_dir=args.embedding_output_dir,
        checkpoint_size=args.checkpoint_size,
        cache_signature={
            "backend": args.backend,
            "checkpoint": args.model_name or args.model,
            "video_sampling": args.video_sampling,
            "fps": None if args.num_frames is not None else args.fps,
            "max_frames": args.max_frames,
            "num_frames": args.num_frames,
            "video_prefetch_batches": args.video_prefetch_batches,
            "video_decode_workers": args.video_decode_workers,
            "video_decoder": getattr(model, "video_decoder", args.video_decoder),
            "vllm_enforce_eager": args.vllm_enforce_eager,
        },
        save_similarities=args.save_similarity_matrices,
        save_family_confusion=args.save_family_confusion,
    )
    text = result.to_json()
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
