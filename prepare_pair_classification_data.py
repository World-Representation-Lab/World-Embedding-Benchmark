import argparse
import json
from pathlib import Path

from world_embedding_benchmark.pair_classification_data import create_pair_classification_manifest

DATASET_DIRS = {
    "physics-bench-dynamics-eval": "World-Embedding-Dynamics-Retrieval",
    "physics-bench-fluid-eval": "World-Embedding-Fluid-Retrieval",
    "physics-bench-optics-eval": "World-Embedding-Optics-Retrieval",
    "physics-bench-solid-eval": "World-Embedding-Solid-Retrieval",
}
BRANCHES = tuple(DATASET_DIRS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed pair-classification triples.")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--branch", action="append", choices=BRANCHES)
    parser.add_argument("--output-dir", default="pair_classification_data/physics-bench")
    parser.add_argument("--text-column", default="parsed_text")
    parser.add_argument("--queries-per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summaries = []
    for branch in args.branch or BRANCHES:
        output = Path(args.output_dir) / f"{branch}.json"
        result = create_pair_classification_manifest(
            Path(args.datasets_root) / DATASET_DIRS[branch], output, branch=branch,
            text_column=args.text_column, queries_per_family=args.queries_per_family, seed=args.seed)
        summaries.append({"branch": branch, "families": result["num_families"],
                          "examples": result["num_examples"], "output": str(output)})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
