# Full-data question-query/message-KV communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T19:08:42Z` |
| Finished | `2026-10-16T19:14:32Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | primary implementation/configuration commit `c7e51f00729c6bda3fb0550571778bfa17022e8a`; correctness-reviewed closure `f56d215185495e60f9bd6a69f19cb8d60e4642fb`; cleanup-reviewed closure `47623a895371247f647dd3b98f2b9227bfc26247`; immutable launch source `c00dd35e56504a506b9acadb3ea9bc69f391bfc4` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag; output_dir=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag; launcher_log=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml 2>&1 \| tee "$launcher_log"'` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices; terminal state `READY` / `HEALTHY` |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 2000
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: false
receiver_question_input: true
receiver_question_query_only: true
communication_rounds: 1
bridge_supervision_weight: 0.0
message_length: 4
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 2
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Receiver question/message integration | Concatenate question and messages before receiver self-attention | Encode separately; question supplies queries and messages supply all keys/values | Remove question embeddings from decoder content while preserving question-directed retrieval from fixed, question-blind sender messages |

Data, seed, batch size, optimization, hidden width, output width, untied encoders, message length, depth, communication rounds, role gates, sender question blindness, answer-token objective, and causal evaluation remain fixed. No bridge or other intermediate supervision is used.

## Hypothesis and Decision Rule

The matched baseline reaches `10/256` exact answers whether messages are intact, zeroed, rotated to another world, swapped, or reduced to either sender alone. Its final sender-path gradient norms are `0.0045245` and `0.0054019`, versus `1.1364310` for the receiver. This shows that answer cross-entropy remains high while almost no useful credit reaches the question-blind senders.

The treatment keeps the senders question-blind. A shared receiver encoder separately contextualizes the question and the eight message vectors. The question representation is used only as the cross-attention query; the messages provide every key and value. The attention result has no question residual and is the only memory exposed to the answer decoder. This tests whether structurally removing question content from the decoder memory makes ordinary answer cross-entropy open the message gradient path without any additional label or auxiliary loss.

Treat the architecture as communicating only if intact messages beat both zero and wrong-world messages in answer exact match and mean joint log probability. Raw intact accuracy alone is insufficient. Compare sender-path gradients and steady-state throughput to the exact baseline, but interpret gradient magnitude only alongside the causal interventions. A same-world fixed-slot query code or global answer prior can still exist, so only intact-versus-wrong-world improvement identifies useful passage content.

## Results

| Evaluation arm | Exact match | Correct | Mean joint log probability |
|---|---:|---:|---:|
| Intact messages | 0.01171875 | 3/256 | -9.404946327209473 |
| Zero messages | 0.0 | 0/256 | -7.162421226501465 |
| Wrong-world messages | 0.01171875 | 3/256 | -9.404233932495117 |
| Swapped senders | 0.01171875 | 3/256 | -9.405538558959961 |
| Agent 1 only | 0.01171875 | 3/256 | -8.69467544555664 |
| Agent 2 only | 0.01171875 | 3/256 | -8.985088348388672 |

The treatment fails the preregistered communication rule. Intact and
wrong-world messages tie at `3/256`, and wrong-world messages are marginally
better by `0.0007124` nats. Zeroing the messages removes those three exact
matches, but improves mean joint log probability by `2.2425251` nats; this
out-of-distribution control establishes nonzero-message dependence, not useful
passage transport. Either single-sender arm also keeps all three exact matches
and improves likelihood relative to intact. The final logged training batch
greedily emitted `pelican` for all 256 examples, so output concentration remains
severe; validation generations were not persisted per example.

Over the last 50 logged training steps, median gradient norms are `0.0049345`
and `0.0049162` for the two sender paths and `1.6263727` for the receiver, a
`329.59x` / `330.82x` receiver-to-sender imbalance. Relative to the matched
baseline medians, absolute sender gradients rise `1.60x` / `1.79x`, but the
receiver also rises `1.28x`; the added gradient does not produce a beneficial
wrong-world gap. Median steady-state throughput is `5686.64` examples/s versus
`5869.27` for the baseline, a `3.11%` reduction. Median final-50 training loss
is `4.21608` versus `3.99575`, and final validation exact match falls from
`10/256` to `3/256`.

For this seed and validation prefix, making the question only the attention
query is therefore not a sufficient unsupervised credit-assignment fix. It
removes the direct question-value shortcut and modestly increases sender-path
gradients, but the model still does not use aligned private passage content
beneficially and generalizes worse than the original receiver.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml` | local and remote SHA-256 `dcdc5b43c04b5665e905042fa8ba8bbdb4ba33633ba7d28f538a36d7077b25f0` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | immutable launch source pushed before staging |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag-20261016T185253.120559935Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` | local and remote base source matched before and after launch: 60 files, 454098 bytes, relative SHA-256-manifest digest `e218b4847b5bc6fd643a4edec1acd5dc278fe84933a606db0e67d0c504d9ef98`; no cache files; W&B's separate run directory excluded |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`; six gzip shards passed integrity checking; 125000 verified questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/checkpoint.msgpack` | 28799159 bytes; SHA-256 `99f0d4b40124ac66fc4357d74d43f1c4d639115e8c85e5aa1e6ebb1ac0e9763f`; CPU restore reports step 2000 |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/metrics.jsonl` | 325735 bytes; SHA-256 `9baca85683db4fd7d9d35102f2f315b781699a8995c5f0fd9ff0d434b8f607fc`; byte-identical to remote |
| Outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/outputs.jsonl` | 2848492 bytes; SHA-256 `071b056fcfaed88c50199872087dc3b94826714a5f690070398d4542aef8a7c6`; byte-identical to remote |
| Launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag/launcher.log.gz` | 3877 bytes; SHA-256 `bc63622c2851dfbc5c2f3ddd6472830d940ce2ad3f7540365668bc8ffccbee65`; decompressed SHA-256 `73ec1285d337153b6d8c810a0eeed4a22e22bacae9a2d50d3535fabe23274f3c` matches the 18722-byte remote log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/vuwudmeg` | authenticated API reports `finished`, last history step 2000, and matching summary metrics |
| Preflight incident | `experiment_log/2026-10-16/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag-preflight-runtime-probe-attempt1.md` | recovered before launch; no trainer, W&B run, output, or checkpoint was created by the failed inventory probe |

## Review and Verification

- Focused query-routing closure passed: 4 cases, including configuration boundaries, question/value separation, nonzero message gradient, default-path preservation, and receiver-question masking.
- `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py tests/test_linked_facts_generator.py`: 67 passed with two existing Pydantic warnings in 208.27 seconds.
- Post-cleanup `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py`: 58 passed with the same two existing warnings in 202.31 seconds.
- Correctness passed at `f56d215185495e60f9bd6a69f19cb8d60e4642fb`; cleanup passed at `47623a895371247f647dd3b98f2b9227bfc26247`.
- The first mandatory availability check on `2026-10-16` returned no final `AVAILABLE` row: the intended prior Kristine v4-8 was absent, one Kristine v6e-16 was deleting, and one Kristine v5e-4 was actively occupied and left untouched. No TPU was selected or modified.
- The filtered availability check at `2026-10-16T18:36Z` returned the same state and no final `AVAILABLE` row. The occupied v5e-4 was again left untouched.
- At `2026-10-16T18:46Z`, the intended v4-8 appeared in the final `AVAILABLE` table. The next parallel checker ended before its required final table; direct API evidence showed the TPU remained `READY` / `HEALTHY`, so the incomplete audit was not treated as absence or launch authority.
- A later complete audit selected the exact v4-8 from its final `AVAILABLE` table. Exact occupancy, source, config, JAX 0.6.2 runtime, W&B authentication, regional dataset, 85 GiB free-space, output-freshness, and lock/VFIO gates passed. The official inventory's intermittent runtime-probe reconnect is documented separately; its exact isolated readiness probe passed before the final occupancy gate.
- Terminal verification found `TRAINING_COMPLETE` at step 2000, no fatal pattern, no trainer, no lockfile or VFIO holder, TPU `READY` / `HEALTHY`, 201 metrics records, 200 output records, a restorable step-2000 checkpoint, and W&B `finished` at step 2000. Completed-result correctness passed at `2e25d578b1b3b05071ec975586cab7439042cd7b`; cleanup passed after the status corrections at `1eaed07fa082a3fd09b68f8b7189a98e277335d6`.

## Failure Details

- Last known step: `2000`; run completed successfully.
- Error summary: no training failure. A pre-launch inventory runtime probe intermittently returned `UNKNOWN` even though the exact probe passed; the zero-step incident is preserved in its separate report.
- Detailed output: canonical launcher log and the separate preflight report linked above.

## Updates

- `2026-10-16`: Selected the question-query/message-key-value receiver as a single architectural intervention after confirming that the original receiver could remain insensitive to messages while answer cross-entropy stayed unsaturated.
- `2026-10-16`: The initial mandatory filtered audit found no available Kristine TPU. The occupied Kristine v5e-4 was left untouched; implementation and review continue while capacity is unavailable.
- `2026-10-16T18:52:53Z`: Created the immutable source snapshot, copied it once to node-local storage, and verified file count, byte count, manifest, selected-config hash, and cache exclusion.
- `2026-10-16T19:08:42Z`: Launched on the exact Kristine v4-8 after direct state, runtime, data, output, lock, and occupancy gates. W&B run ID is `vuwudmeg`.
- `2026-10-16T19:14:32Z`: Completed all 2000 steps, final causal evaluation, checkpoint save, and W&B synchronization. A first read-only checkpoint probe had a quoting-only syntax error; the corrected CPU probe restored step 2000 without modifying the checkpoint.
