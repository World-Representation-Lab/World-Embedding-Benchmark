#   python debug_lco_retrieval.py \
#     --limit-videos-per-family 8 \
#     --batch-size 1 \
#     --fps 2 \
#     --max-frames 128 \
#     --save results/debug_lco_8pf.json \
#     --save-scores results/debug_lco_8pf_scores.npy
#   python debug_lco_retrieval.py \
#     --limit-videos-per-family 8 \
#     --batch-size 1 \
#     --save results/debug_lco_8pf.json \
#     --save-scores results/debug_lco_8pf_scores.npy \
#     --save-matrices-dir results/debug_lco_8pf_matrices

#   python debug_lco_retrieval.py \
#     --limit-videos-per-family 8 \
#     --batch-size 1 \
#     --save results/debug_lco_8pf.json \
#     --save-scores results/debug_lco_8pf_scores.npy \
#     --save-matrices-dir results/debug_lco_8pf_matrices

  python debug_lco_retrieval.py \
    --model-name /workspace/mteb-main/checkpoints/LCO-Embedding-Omni-7B \
    --limit-videos-per-family 8 \
    --batch-size 1 \
    --save results/debug_lco_7b_8pf.json \
    --save-scores results/debug_lco_7b_8pf_scores.npy \
    --save-matrices-dir results/debug_lco_7b_8pf_matrices