from __future__ import annotations

import argparse
from pathlib import Path

from world_embedding_benchmark.regression import REGRESSION_CONFIGS
from world_embedding_benchmark.regression_splits import (
    DEFAULT_REPETITION_SEEDS,
    DEFAULT_TRAIN_SIZES,
    create_regression_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic fixed-test manifests for video regression."
    )
    parser.add_argument("--dataset-dir", default="datasets/World-Embedding-Regression")
    parser.add_argument("--subset", action="append", choices=REGRESSION_CONFIGS)
    parser.add_argument("--output-dir", default="regression_splits/physics-bench-regression-500")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--num-strata", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, action="append")
    parser.add_argument("--repetition-seed", type=int, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = args.subset or list(REGRESSION_CONFIGS)
    output_dir = Path(args.output_dir)
    for subset in subsets:
        output = output_dir / f"{subset}.json"
        manifest = create_regression_split_manifest(
            args.dataset_dir,
            subset,
            output,
            test_size=args.test_size,
            num_strata=args.num_strata,
            split_seed=args.split_seed,
            train_sizes=args.train_size or DEFAULT_TRAIN_SIZES,
            repetition_seeds=args.repetition_seed or DEFAULT_REPETITION_SEEDS,
        )
        print(
            f"{subset}: {len(manifest['training_pool_ids'])} train, "
            f"{len(manifest['test_examples'])} test -> {output}"
        )


if __name__ == "__main__":
    main()
