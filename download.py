from __future__ import annotations

import argparse
from pathlib import Path

DATASETS = {
    "dynamics": "World-Representation-Lab/World-Embedding-Dynamics-Retrieval",
    "fluid": "World-Representation-Lab/World-Embedding-Fluid-Retrieval",
    "optics": "World-Representation-Lab/World-Embedding-Optics-Retrieval",
    "solid": "World-Representation-Lab/World-Embedding-Solid-Retrieval",
    "regression": "World-Representation-Lab/World-Embedding-Regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download World Embedding datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASETS,
        help="Download only this dataset; repeat to select multiple datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before downloading datasets") from exc
    selected = args.dataset or list(DATASETS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        repo_id = DATASETS[key]
        destination = args.output_dir / repo_id.rsplit("/", 1)[1]
        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=destination,
        )


if __name__ == "__main__":
    main()
