from __future__ import annotations

import argparse
from pathlib import Path

from world_embedding_benchmark.mllm_pair_classification import evaluate_mllm_pair_classification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate text-video pair classification by prompting a generative MLLM."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-dir", help="Override the dataset path stored in the manifest.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vllm-max-model-len", type=int, default=49152)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=32)
    parser.add_argument("--predictions-output")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_mllm_pair_classification(
        manifest_path=args.manifest,
        dataset_dir=args.dataset_dir,
        model_name=args.model_name,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        max_model_len=args.vllm_max_model_len,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_num_seqs=args.vllm_max_num_seqs,
        predictions_output=args.predictions_output,
    )
    text = result.to_json()
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
