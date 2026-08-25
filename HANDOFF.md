# Development handoff

## Purpose and benchmark contract

This project evaluates multimodal world embeddings through bidirectional
text-video retrieval on PhysicsBench, currently using LCO-Embedding Omni 3B/7B.

- Text: `parsed_text` by default; `raw_text` is selectable.
- Video: embedded Parquet `video` bytes.
- Family: Parquet parent directory, never inferred from every 100 rows.
- Metrics: Recall/MRR/nDCG at 1, 5, 10 in both directions, globally/per family.
- Exact prompt suffixes: `\nSummarize the above text in one word:` and
  `\nSummarize the above video in one word:`.
- Embedding: LAST-token final hidden state, then L2 normalization.

## Implemented components

`retrieval.py` discovers family shards, atomically extracts videos to
`<dataset>/.video_cache`, reuses size-matched files, creates relevance, and
handles filtering/metrics. `lco_embedding.py` is the Transformers reference.
`lco_vllm.py` uses vLLM LAST pooling with identical prompts/normalization.
`vllm_lco_model.py` maps standalone LCO prefixes onto vLLM's Qwen2.5-Omni
thinker implementation.

Both processor/FPS and fixed-total-frame sampling work. FFmpeg emits only the
selected source frames, preventing the old full-decode memory growth.

`embedding_artifacts.py` saves resumable shards, final text/video embeddings
with IDs/families, three similarity matrices, and family-confusion CSV/JSON.
Parity, throughput, rank diagnostics, and visualization have dedicated scripts.

## Dataset and taxonomy state

Solid and optics evaluations were completed on the old server. Fluid and
dynamics examples are in `run_retrieval.sh`; results are not included here.

Legacy fluid arrived as correct-order/wrong-caption and wrong-order/correct-
caption versions. `merge_fluid_eval_captions.py` preserves order, IDs, videos,
and directories from the former and maps captions by `case_id`, validating
`query_id`. Its 700-row output was byte/order/caption validated.

Fluid alone uses seven coarse directory families containing 15 explicit fine
case types. Other branches are closer to fine granularity. The intended revision
is 100 examples per fine fluid family while retaining explicit `coarse_family`
and fine `family` metadata.

Key fluid details:

- buoyancy = `buoyancy2d` 56 + `rayleighBenard2d` 44.
- cavity = `cavity2d_turb` 58 + `lidDrivenCavity3d` 42.
- external wake = cylinder 56 + square 44.
- internal step = backward-facing step 26, Newtonian sudden expansion 21,
  sudden contraction 14, Herschel-Bulkley expansion 18, power-law expansion 21.
  Keep all five fine cases: geometry and rheology are independent axes.
- jet axisymmetric = `jetaxi` 53 + `jetaxi_turb` 47.

The jet split needs clarification. `jetaxi` contains 32 captioned laminar and
21 turbulent cases; its turbulent Re is 2,825-19,800 (median 7,800).
`jetaxi_turb` has 47 turbulent cases, Re 5,500-40,000 (median 21,000). Captions
describe gradual spreading versus strong mixing/core breakup, but name no
RANS/LES/DNS model, inlet turbulence, or solver difference. Ask the data author
whether the simulation method differs. If only Re differs, use one family plus
`flow_regime`, or define non-overlapping regimes.

All families in all inspected branches contained 100 examples, with no missing
captions or ID-prefix mismatches. A 1-2-example-per-family caption audit found
coherent physics. Re-run these checks after downloading revised data.

## Performance lessons

DGX Spark unified memory made vLLM and decoded videos compete, producing slower
encoding and instability. Selective decode was the important fix. Batch 1 and
prefetch 1 were safest there. On the new server, benchmark `(batch,prefetch)` =
`(1,0)`, `(1,1)`, `(2,1)`, `(4,1)`, `(4,2)` while watching GPU and host memory.
More vLLM memory utilization helps only if engine cache/workspace is limiting;
it does not accelerate video decoding.

## Next-server checklist

1. Install the x86_64 environment from `ENVIRONMENT.md`.
2. Download final datasets and LCO checkpoints.
3. Verify revised fluid fine/coarse labels.
4. Run syntax/CLI checks and one-example-per-family smoke tests.
5. Run Transformers/vLLM parity in both frame modes.
6. Benchmark batch/prefetch and choose stable settings.
7. Run full fluid/dynamics evaluation with embedding/confusion artifacts.

Never commit datasets, weights, results, video caches, environments, tokens, or
the historical ARM64 build tree.
