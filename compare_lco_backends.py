"""Compare Transformers and vLLM LCO embeddings in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="datasets/physics-bench-solid-eval")
    parser.add_argument("--model", default="lco-omni-3b")
    parser.add_argument("--model-name")
    parser.add_argument("--video-sampling", choices=["processor", "fixed"], default="processor")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-items", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--transformers-python", default="/workspace/envs/mieb/bin/python")
    parser.add_argument("--vllm-python", default="/workspace/envs/vllm/bin/python")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-absolute-error", type=float, default=0.02)
    parser.add_argument("--output")
    parser.add_argument("--worker-backend", choices=["transformers", "vllm"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def worker(args: argparse.Namespace) -> None:
    from world_embedding_benchmark.models import get_model
    from world_embedding_benchmark.retrieval import load_retrieval_data

    data = load_retrieval_data(args.dataset_dir, text_column="parsed_text")
    queries = data.queries[: args.num_items]
    video_by_id = {video.item_id: video for video in data.videos}
    videos = [video_by_id[query.relevant_ids[0]] for query in queries]
    kwargs = {
        "backend": args.worker_backend,
        "video_sampling": args.video_sampling,
        "fps": None if args.num_frames is not None else args.fps,
        "max_frames": args.max_frames,
        "num_frames": args.num_frames,
    }
    if args.worker_backend == "vllm":
        kwargs["max_model_len"] = args.vllm_max_model_len
        kwargs["gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        kwargs["engine_kwargs"] = {"disable_log_stats": True}
    if args.model_name:
        kwargs["model_name"] = args.model_name
    model = get_model(args.model, **kwargs)
    text = model.encode_texts([query.text for query in queries], batch_size=args.batch_size)
    video = model.encode_videos([item.video_path for item in videos], batch_size=args.batch_size)
    np.savez(args.worker_output, text=text, video=video)


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Embedding shapes differ: {reference.shape} != {candidate.shape}")
    cosine = np.sum(reference * candidate, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    )
    difference = np.abs(reference - candidate)
    return {
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def run_worker(python: str, backend: str, output: Path, argv: list[str]) -> None:
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--output", "--worker-backend", "--worker-output"}:
            skip_next = True
            continue
        filtered.append(arg)
    subprocess.run(
        [python, str(Path(__file__).resolve()), *filtered,
         "--worker-backend", backend, "--worker-output", str(output)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    if args.worker_backend:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="lco-parity-") as directory:
        root = Path(directory)
        transformer_path = root / "transformers.npz"
        vllm_path = root / "vllm.npz"
        run_worker(args.transformers_python, "transformers", transformer_path, sys.argv[1:])
        run_worker(args.vllm_python, "vllm", vllm_path, sys.argv[1:])
        reference = np.load(transformer_path)
        candidate = np.load(vllm_path)
        result = {
            "video_sampling": args.video_sampling,
            "num_items": args.num_items,
            "text": compare(reference["text"], candidate["text"]),
            "video": compare(reference["video"], candidate["video"]),
        }
    result["passed"] = all(
        metrics["min_cosine"] >= args.min_cosine
        and metrics["max_absolute_error"] <= args.max_absolute_error
        for metrics in (result["text"], result["video"])
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
