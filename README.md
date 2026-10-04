# World Embedding Benchmark

Text-video retrieval and video-representation regression for PhysicsBench,
with faithful LCO-Embedding inference through Transformers and vLLM.

The benchmark uses `parsed_text` as the default text prompt and the embedded
Parquet `video` field as video input. It reports text-to-video and video-to-text
Recall, MRR, and nDCG globally and per family, and can save embeddings,
similarity matrices, checkpoints, and family-confusion tables.

## Quick start (standard x86_64 CUDA server)

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
huggingface-cli login
```

Use official prebuilt PyTorch/vLLM wheels on x86_64. Do not repeat the DGX
Spark ARM64 source-build procedure described in `ENVIRONMENT.md`.

Place datasets under:

```text
datasets/physics-bench-solid-eval
datasets/physics-bench-optics-eval
datasets/physics-bench-fluid-eval
datasets/physics-bench-dynamics-eval
```

Each family is a directory containing Parquet shards with `query_id`, `case_id`,
`raw_text`, `parsed_text`, and `video` columns.

## Smoke test

```bash
python run_retrieval.py \
  --dataset-dir datasets/physics-bench-solid-eval \
  --model lco-omni-3b --model-name ./models/LCO-Embedding-Omni-3B \
  --backend vllm \
  --vllm-max-model-len 32768 --vllm-gpu-memory-utilization 0.7 \
  --video-sampling processor --fps 2 --max-frames 128 \
  --batch-size 4 --video-prefetch-batches 2 \
  --video-decode-workers 8 --video-decoder auto --no-vllm-enforce-eager \
  --text-column parsed_text --limit-videos-per-family 1 \
  --output results/solid_lco_3b_smoke.json
```

Remove the limit for a full run. The shown batch, prefetch, and worker values were
fast on a dual RTX 6000D server; benchmark them on the target hardware. Decoder
`auto` prefers TorchCodec and emits a warning before falling back to selective
FFmpeg decoding. Registered model keys are `lco-omni-3b` and
`lco-omni-7b`; `--model-name` overrides their Hugging Face checkpoints.

## Video regression

`run_regression.py` evaluates frozen video embeddings with a nested
cross-validated ridge probe. Every reported prediction is out of fold, and the
ridge strength is selected using only an inner split of the corresponding outer
training fold. Feature and target standardization are fit on training data only.

```bash
python run_regression.py \
  --dataset-dir datasets/physics-bench-regression-500 \
  --subset pendulum \
  --model lco-omni-3b --model-name ./models/LCO-Embedding-Omni-3B \
  --backend vllm --video-decoder auto \
  --batch-size 4 --video-prefetch-batches 2 --video-decode-workers 8 \
  --embedding-output-dir results/regression/pendulum/embeddings \
  --output results/regression/pendulum/result.json
```

The evaluator reports MAE, MSE, RMSE, R², Pearson correlation, tie-aware
Spearman correlation, and MAE/RMSE normalized by the target range. Encoding is
resumable when `--embedding-output-dir` is set; per-example out-of-fold
predictions are written beside `--output` unless a separate path is supplied.

### Fixed test set and scaling protocol

Canonical manifests for all ten subsets are included under
`regression_splits/physics-bench-regression-500/`. Reproduce or validate them with:

```bash
python prepare_regression_splits.py
```

The manifests use stable example IDs and target-rank stratified sampling. The
defaults reserve 100 test examples and define nested training sizes of 25, 50,
100, 200, 300, and 400 across five deterministic repetitions. Existing
manifests are validated and never silently overwritten.

Run all training sizes after encoding the 500 videos once:

```bash
python run_regression_scaling.py \
  --subset pendulum \
  --model lco-omni-3b --model-name ./models/LCO-Embedding-Omni-3B \
  --backend vllm --video-decoder auto \
  --batch-size 4 --video-prefetch-batches 2 --video-decode-workers 8 \
  --embedding-output-dir results/regression-scaling/pendulum/embeddings \
  --output results/regression-scaling/pendulum/result.json
```

Every ridge strength is selected using only the corresponding training subset;
the fixed test set is used only for final metrics. Training sets are prefixes of
one deterministic order per repetition, so larger sizes contain smaller sizes.

## Fidelity and sampling

- Processor mode reproduces Qwen Omni FPS sampling while decoding only selected
  frames. `--video-decoder auto` prefers TorchCodec indexed decoding and falls
  back to selective FFmpeg decoding; explicit `torchcodec` fails fast if its
  runtime is unavailable. Fixed mode currently uses FFmpeg and uniformly selects
  exactly `--num-frames N` frames.
- Both backends preserve LCO's compression prompts, LAST-token pooling, and L2
  normalization.
- `compare_lco_backends.py` validates Transformers/vLLM embedding parity.

## Main files

- `run_retrieval.py`: bidirectional retrieval CLI.
- `run_regression.py`: nested-CV video regression CLI.
- `run_regression_scaling.py`: fixed-test regression scaling CLI.
- `prepare_regression_splits.py`: deterministic split-manifest generator.
- `run_retrieval.sh`: readable full-run examples.
- `world_embedding_benchmark/retrieval.py`: retrieval loading, scoring, and metrics.
- `world_embedding_benchmark/regression.py`: regression loading, probing, and metrics.
- `world_embedding_benchmark/regression_splits.py`: reproducible test and scaling splits.
- `world_embedding_benchmark/regression_scaling.py`: fixed-test scaling evaluation.
- `world_embedding_benchmark/models/`: Transformers/vLLM LCO adapters.
- `world_embedding_benchmark/embedding_artifacts.py`: resumable artifacts.
- `compare_lco_backends.py`: backend parity.
- `benchmark_lco_video_throughput.py`: throughput tuning.
- `debug_lco_retrieval.py`: retrieval diagnostics.
- `visualize_similarity_matrices.py`: embedding visualization.
- `merge_fluid_eval_captions.py`: legacy fluid reconstruction.
- `HANDOFF.md`: project state and next steps.
- `ENVIRONMENT.md`: x86_64 setup and historical ARM64 notes.

Datasets, weights, results, caches, environments, and local build trees are
intentionally excluded from Git.
