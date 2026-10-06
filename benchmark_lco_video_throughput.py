from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from world_embedding_benchmark.models import get_model


def memory_gib(field: str, path: str) -> float:
    for line in Path(path).read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError(f"Missing {field} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bounded LCO video batching.")
    parser.add_argument("--num-videos", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(Path("datasets/World-Embedding-Solid-Retrieval/.video_cache").glob("*.mp4"))
    configurations = ((1, 0), (1, 1), (2, 1), (4, 1), (4, 2))
    sample_count = args.num_videos * len(configurations) + 1
    step = max(1, len(paths) // sample_count)
    sampled = [str(paths[min(i * step, len(paths) - 1)]) for i in range(sample_count)]
    model = get_model(
        "lco-omni-3b",
        backend="vllm",
        video_sampling="processor",
        fps=2,
        max_frames=128,
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        video_prefetch_batches=0,
    )
    print("Warming up with", Path(sampled[-1]).stem)
    model.encode_videos([sampled[-1]], batch_size=1)
    for run_index, (batch_size, prefetch) in enumerate(configurations):
        start = run_index * args.num_videos
        videos = sampled[start : start + args.num_videos]
        print("videos", [Path(video).stem for video in videos])
        model.video_prefetch_batches = prefetch
        stop = threading.Event()
        samples: list[tuple[float, float]] = []

        def monitor() -> None:
            while not stop.is_set():
                samples.append(
                    (
                        memory_gib("MemAvailable", "/proc/meminfo"),
                        memory_gib("VmRSS", "/proc/self/status"),
                    )
                )
                stop.wait(0.05)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        started = time.perf_counter()
        embeddings = model.encode_videos(videos, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join()
        print(
            {
                "batch_size": batch_size,
                "prefetch_batches": prefetch,
                "seconds": round(elapsed, 3),
                "videos_per_second": round(len(videos) / elapsed, 3),
                "min_available_gib": round(min(sample[0] for sample in samples), 2),
                "max_rss_gib": round(max(sample[1] for sample in samples), 2),
                "embedding_norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
                "embedding_norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
