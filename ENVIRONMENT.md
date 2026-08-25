# Environment setup

## Recommended x86_64 NVIDIA server

The next server is expected to be x86_64. Use normal binary packages:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify with:

```bash
python - <<'PY'
import platform, torch, transformers, vllm
print(platform.machine(), torch.__version__, torch.cuda.is_available())
print(transformers.__version__, vllm.__version__)
PY
ffmpeg -version | head -1
ffprobe -version | head -1
```

If exact pins are unavailable for the target CUDA/Python combination, choose a
compatible official PyTorch/vLLM wheel pair. The adapter imports vLLM internals,
so rerun backend parity after any vLLM upgrade.

TorchCodec is not required. The optimized path uses system FFmpeg/FFprobe and
passes sampled TCHW tensors directly to the model, avoiding full-video decode.

Authenticate using `huggingface-cli login` or `HF_TOKEN`; never store tokens in
source. Default checkpoints are:

- `LCO-Embedding/LCO-Embedding-Omni-3B`
- `LCO-Embedding/LCO-Embedding-Omni-7B`

Dataset repository names may change while fluid is revised. Download the final
four repositories to paths in `README.md`; do not blindly use the legacy
selection inside `download.py`.

## First-run validation

1. `python -m compileall world_embedding_benchmark *.py`
2. `python run_retrieval.py --help`
3. One-example-per-family Transformers smoke test.
4. One-example-per-family vLLM smoke test.
5. Parity test for processor and fixed-frame modes.
6. Only then launch full evaluation.

When both backends share one environment:

```bash
python compare_lco_backends.py --transformers-python .venv/bin/python \
  --vllm-python .venv/bin/python --video-sampling processor \
  --fps 2 --max-frames 8 --num-items 1
python compare_lco_backends.py --transformers-python .venv/bin/python \
  --vllm-python .venv/bin/python --video-sampling fixed \
  --num-frames 8 --num-items 1
```

## Historical DGX Spark / ARM64 setup

The original machine was an NVIDIA DGX Spark (`aarch64`) with unified CPU/GPU
memory. Desktop was mounted into an NVIDIA PyTorch container as `/workspace`.
Two Python 3.12 environments were kept separate:

- `/workspace/envs/mieb`: reference Transformers implementation.
- `/workspace/envs/vllm`: isolated vLLM implementation.

Typical session:

```bash
docker exec -it mieb-new bash
source /workspace/envs/vllm/bin/activate
cd /workspace/world-embedding-benchmark
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

The final environment had Python 3.12.3, PyTorch 2.11.0+cu130, torchvision
0.26.0+cu130, Transformers 5.12.1, vLLM 0.23.0, qwen-omni-utils 0.0.9,
PyAV 18.0.0, PyArrow 24.0.0, NumPy 2.3.5, and system FFmpeg 6.1.1.

ARM64 wheels and video dependencies were unreliable. vLLM was tested separately
and a source checkout/build tree was temporarily retained under `.build/`.
TorchCodec was deliberately avoided. Those artifacts are not portable and are
not copied to the GitHub repository.

Unified memory made model workspace and decoded frames compete. Conservative
stable settings were GPU utilization 0.35, batch 1, prefetch 1, model length
4096, FPS 2, and max 128 frames. Selective FFmpeg decoding was the main memory
and latency fix. Do not reproduce this ARM64 source-build path on x86_64 unless
official wheels are truly unavailable.
