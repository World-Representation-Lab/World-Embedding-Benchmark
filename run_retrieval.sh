# python run_retrieval.py \
#     --dataset-dir datasets/World-Embedding-Fluid-Retrieval \
#     --model lco-omni-3b \
#     --video-sampling processor \
#     --fps 2 \
#     --max-frames 128 \
#     --batch-size 1 \
#     --text-column parsed_text \
#     --output results/fluid_lco_3b_per_family8.json

python run_retrieval.py \
    --dataset-dir datasets/World-Embedding-Fluid-Retrieval \
    --model lco-omni-3b \
    --backend vllm \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.35 \
    --video-sampling processor \
    --fps 2 \
    --max-frames 128 \
    --batch-size 1 \
    --video-prefetch-batches 1 \
    --text-column parsed_text \
    --embedding-output-dir results/embeddings/fluid_lco_3b_vllm \
    --checkpoint-size 32 \
    --save-similarity-matrices \
    --save-family-confusion \
    --output results/fluid_lco_3b_vllm.json

python run_retrieval.py \
    --dataset-dir datasets/World-Embedding-Dynamics-Retrieval \
    --model lco-omni-3b \
    --backend vllm \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.35 \
    --video-sampling processor \
    --fps 2 \
    --max-frames 128 \
    --batch-size 1 \
    --video-prefetch-batches 1 \
    --text-column parsed_text \
    --embedding-output-dir results/embeddings/dynamics_lco_3b_vllm \
    --checkpoint-size 32 \
    --save-similarity-matrices \
    --save-family-confusion \
    --output results/dynamics_lco_3b_vllm.json

python run_retrieval.py \
    --dataset-dir datasets/World-Embedding-Fluid-Retrieval \
    --model lco-omni-7b \
    --backend vllm \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.35 \
    --video-sampling processor \
    --fps 2 \
    --max-frames 128 \
    --batch-size 1 \
    --video-prefetch-batches 1 \
    --text-column parsed_text \
    --embedding-output-dir results/embeddings/fluid_lco_7b_vllm \
    --checkpoint-size 32 \
    --save-similarity-matrices \
    --save-family-confusion \
    --output results/fluid_lco_7b_vllm.json

python run_retrieval.py \
    --dataset-dir datasets/World-Embedding-Dynamics-Retrieval \
    --model lco-omni-7b \
    --backend vllm \
    --vllm-max-model-len 4096 \
    --vllm-gpu-memory-utilization 0.35 \
    --video-sampling processor \
    --fps 2 \
    --max-frames 128 \
    --batch-size 1 \
    --video-prefetch-batches 1 \
    --text-column parsed_text \
    --embedding-output-dir results/embeddings/dynamics_lco_7b_vllm \
    --checkpoint-size 32 \
    --save-similarity-matrices \
    --save-family-confusion \
    --output results/dynamics_lco_7b_vllm.json