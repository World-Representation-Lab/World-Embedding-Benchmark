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

Use official prebuilt PyTorch and vLLM wheels on x86_64.

## Data

The datasets are available in the [World Embedding Benchmark collection](https://huggingface.co/collections/World-Representation-Lab/world-embedding-benchmark):

| Dataset | Hugging Face repository |
| --- | --- |
| Dynamics retrieval | [`World-Embedding-Dynamics-Retrieval`](https://huggingface.co/datasets/World-Representation-Lab/World-Embedding-Dynamics-Retrieval) |
| Fluid retrieval | [`World-Embedding-Fluid-Retrieval`](https://huggingface.co/datasets/World-Representation-Lab/World-Embedding-Fluid-Retrieval) |
| Optics retrieval | [`World-Embedding-Optics-Retrieval`](https://huggingface.co/datasets/World-Representation-Lab/World-Embedding-Optics-Retrieval) |
| Solid retrieval | [`World-Embedding-Solid-Retrieval`](https://huggingface.co/datasets/World-Representation-Lab/World-Embedding-Solid-Retrieval) |
| Regression | [`World-Embedding-Regression`](https://huggingface.co/datasets/World-Representation-Lab/World-Embedding-Regression) |

Download all five repositories into `datasets/`:

```bash
python download.py
```

Retrieval datasets have one `test` split. Regression uses one config per physics
family and also has only a `test` split:

```python
from datasets import load_dataset

solid = load_dataset(
    "World-Representation-Lab/World-Embedding-Solid-Retrieval",
    split="test",
)
pendulum = load_dataset(
    "World-Representation-Lab/World-Embedding-Regression",
    "pendulum",
    split="test",
)
```

## Smoke test

```bash
python run_retrieval.py \
  --dataset-dir datasets/World-Embedding-Solid-Retrieval \
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
FFmpeg decoding. `--model-name` overrides a key's Hugging Face checkpoint.

## Registered models

| Key | Backend | Notes |
| --- | --- | --- |
| `lco-omni-3b`, `lco-omni-7b` | `transformers`, `vllm` | |
| `qwen3-vl-embedding-2b`, `qwen3-vl-embedding-8b` | `vllm` | Instruction-conditioned; pass `--batch-size 4` |
| `nv-omni-embed-3b` | `transformers` | Bidirectional text tower, so vLLM would mis-encode it |
| `vjepa2-vitl`, `vjepa2-vitg` | `transformers` | Video only; regression CLIs only |

Qwen3-VL-Embedding prepends a task instruction to each side, defaulting to
retrieving a physics video from a caption and the reverse. Omni-Embed-Nemotron
uses the `query: ` and `passage: ` prefixes shipped with the checkpoint. Both
reuse the LCO TorchCodec sampler, so `--fps` and `--max-frames` mean the same
thing across models.

## Video regression

`run_regression.py` evaluates frozen video embeddings with a nested
cross-validated ridge probe. Every reported prediction is out of fold, and the
ridge strength is selected using only an inner split of the corresponding outer
training fold. Feature and target standardization are fit on training data only.

```bash
python run_regression.py \
  --dataset-dir datasets/World-Embedding-Regression \
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

## Pair classification

The fixed manifests contain one positive, one within-family negative, and one
cross-family negative for 10 sampled queries per family. Run one branch with:

```bash
python run_pair_classification.py \
  --manifest pair_classification_data/physics-bench/physics-bench-solid-eval.json \
  --model lco-omni-3b --model-name ./models/LCO-Embedding-Omni-3B \
  --backend vllm --video-decoder auto \
  --embedding-output-dir results/pair-classification/solid/embeddings \
  --output results/pair-classification/solid/result.json
```

The output reports within-family, cross-family, and three-way accuracy, plus
margins and tie rates. Use `--dataset-dir` if the dataset is not under
`datasets/`.

### Prompting a generative model

A generative model produces no similarity score, so `run_mllm_pair_classification.py`
turns each triple into two binary questions and asks which of two videos matches
the caption:

```bash
python run_mllm_pair_classification.py \
  --manifest pair_classification_data/physics-bench/physics-bench-solid-eval.json \
  --model-name Qwen/Qwen3-VL-8B-Instruct \
  --num-frames 16 --batch-size 4 \
  --output results/mllm-pair-classification/solid/result.json
```

Every question is asked twice with the candidates swapped, because generative
models are sensitive to option order. `accuracy` scores the two answers
independently and averages them, so a model that always names the same position
lands at the 0.5 chance level, matching the embedding metric;
`both_orders_accuracy` is the stricter conjunction, where chance is 0.25.
`pick_a_rate` reports how lopsided the choices were. Answers are constrained to
a single letter by vLLM structured output, so no parsing is involved.

Only Qwen multimodal checkpoints have been tested: the prompt is assembled with
the processor's chat template and frames are passed with the metadata those
models expect. Unlike the retrieval adapters, frames are spread evenly over the
whole clip rather than sampled at a fixed rate.

## Fidelity and sampling

- Processor mode reproduces Qwen Omni FPS sampling while decoding only selected
  frames. `--video-decoder auto` prefers TorchCodec indexed decoding and falls
  back to selective FFmpeg decoding; explicit `torchcodec` fails fast if its
  runtime is unavailable. Fixed mode currently uses FFmpeg and uniformly selects
  exactly `--num-frames N` frames.
- Both backends preserve LCO's compression prompts, LAST-token pooling, and L2
  normalization.

## Repository structure

```text
world_embedding_benchmark/   Dataset loaders, model adapters, and evaluators
run_retrieval.py             Retrieval CLI
run_regression.py            Nested-CV regression CLI
run_regression_scaling.py    Fixed-test regression scaling CLI
run_pair_classification.py   Pair-classification CLI
run_mllm_pair_classification.py  Pair classification by prompting an MLLM
prepare_*.py                 Deterministic manifest generators
regression_splits/           Canonical regression splits
pair_classification_data/    Canonical pair-classification manifests
tests/                       Unit tests
```

Datasets, weights, results, caches, environments, and local build trees are
intentionally excluded from Git.


## Citation
If you find our work useful, please cite our [paper](https://arxiv.org/abs/2610.03632):
```bibtex
@misc{liu2026worldembeddingbenchmark,
title         = {World Embedding Benchmark},
author        = {Yiqi Liu and Ruifeng Yuan and Yang Wang and Long Li and Fengyu Cai and Hou Pong Chan and Jialin Yu and Hao Zhang and Chenghua Lin and Chenghao Xiao},
year          = {2026},
eprint        = {2610.03632},
archivePrefix = {arXiv},
primaryClass  = {cs.CV},
url           = {https://arxiv.org/abs/2610.03632}
}
