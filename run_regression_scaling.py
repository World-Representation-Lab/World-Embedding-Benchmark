from __future__ import annotations

import argparse
from pathlib import Path

from world_embedding_benchmark.models import get_model
from world_embedding_benchmark.regression import REGRESSION_CONFIGS
from world_embedding_benchmark.regression_scaling import (
    DEFAULT_ALPHAS,
    evaluate_video_regression_scaling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-test video-regression scaling evaluation."
    )
    parser.add_argument("--dataset-dir", default="datasets/World-Embedding-Regression")
    parser.add_argument("--subset", required=True, choices=REGRESSION_CONFIGS)
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="lco-omni-3b")
    parser.add_argument("--model-name")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--alpha", type=float, action="append")
    parser.add_argument("--embedding-output-dir")
    parser.add_argument("--checkpoint-size", type=int, default=32)
    parser.add_argument("--predictions-output")
    parser.add_argument("--output")
    parser.add_argument("--video-sampling", choices=["processor", "fixed"], default="processor")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--video-prefetch-batches", type=int, default=0)
    parser.add_argument("--video-decode-workers", type=int, default=1)
    parser.add_argument(
        "--video-decoder", choices=["auto", "ffmpeg", "torchcodec"], default="auto"
    )
    parser.add_argument("--vllm-max-model-len", type=int, default=32768)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--vllm-enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.split_manifest or (
        f"regression_splits/physics-bench-regression-500/{args.subset}.json"
    )
    model_kwargs = {
        "backend": args.backend,
        "device": args.device,
        "video_sampling": args.video_sampling,
        "fps": None if args.num_frames is not None else args.fps,
        "max_frames": args.max_frames,
        "num_frames": args.num_frames,
    }
    if args.backend == "vllm":
        model_kwargs.update({
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "video_prefetch_batches": args.video_prefetch_batches,
            "video_decode_workers": args.video_decode_workers,
            "video_decoder": args.video_decoder,
            "enforce_eager": args.vllm_enforce_eager,
        })
    if args.model_name:
        model_kwargs["model_name"] = args.model_name
    model = get_model(args.model, **model_kwargs)

    predictions_output = args.predictions_output
    if predictions_output is None and args.output:
        predictions_output = str(Path(args.output).with_name("predictions.csv"))
    result = evaluate_video_regression_scaling(
        model,
        dataset_dir=args.dataset_dir,
        subset=args.subset,
        split_manifest=manifest,
        model_name=args.model_name or args.model,
        split=args.split,
        batch_size=args.batch_size,
        inner_folds=args.inner_folds,
        alphas=args.alpha or DEFAULT_ALPHAS,
        embedding_output_dir=args.embedding_output_dir,
        checkpoint_size=args.checkpoint_size,
        predictions_output=predictions_output,
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
    )
    text = result.to_json()
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
