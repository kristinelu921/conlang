# Full-data bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Status | `planned` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Git revision | immutable launch snapshot HEAD `ce27b32f14ccbeb3c37e830d3902c050e4dc5550`; corrected reviewed implementation/configuration closure `c83e7a92994b118f9fd1d7fe4e2d55f894470a32` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
receiver_question_input: false
communication_rounds: 1
bridge_supervision_weight: 1.0
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
| Bridge supervision weight | `0.0` | `1.0` | Give the person-side sender direct credit for selecting the question-specific middle entity while leaving all model inputs unchanged |

All data, seed, optimization, schedule, widths, depth, message length, one-round topology, sharing, gates, sender question conditioning, receiver question masking, hardware, and causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

The receiver-question bypass and matched wrong-question controls show that the senders can relay question identity, but target-question/wrong-passages controls show no beneficial use of private facts. Doubling the message length and adding one synchronous exchange were both insufficient. The leading remaining hypothesis is that the end-to-end answer objective gives too little direct credit for selecting the question-specific middle entity from the person-side passage.

For each training example only, the generator's existing reasoning path supplies the middle entity as an auxiliary label, and its sender assignment identifies which sender holds the person-to-middle fact. A shared vocabulary head reads the mean of that sender's final four message vectors. The label and sender assignment are supervision only; neither is passed into the sender or receiver. The total objective is answer loss plus `1.0 * bridge loss`. A zero weight omits the auxiliary head, preserving existing checkpoint trees.

The bridge hypothesis is supported only if intact bridge exact match learns above its controls, especially the matched wrong-question re-encoding. The communication fix is useful only if intact answer messages then beat target-question/wrong-passages messages in exact match and preferably mean joint log probability. If bridge accuracy learns but the answer remains passage-flat, the bottleneck moves downstream to the answer-side lookup or receiver join. If bridge accuracy itself does not learn, revise the auxiliary readout/objective before stacking width, depth, tying, gates, or a second exchange.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` | parses and validates; launch pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | corrected closure passed correctness and cleanup review; push pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag-20261016T113307.495856067Z`; remote path pending | local snapshot verified: 57 files, 441219 bytes, relative SHA-256-manifest digest `4970c19a2ddfa884af78653bfdf3b0e2727bd9071f4b104d024038e4beca4b36`; selected-config SHA-256 `007774172e2348652f10d49e295b69ade13d0226df103400c535f36827756a2a`; no bytecode or test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | expected manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`; remote revalidation pending |
| Tracking run | pending | W&B enabled; exact URL pending authenticated launch |
| Checkpoint | pending | final step-2000 restore and checksum pending |
| Metrics | pending | final JSONL copy, parse, and checksum pending |
| Generated outputs | pending | final JSONL copy, parse, and checksum pending |
| Canonical launcher log | pending | process-zero identification, completion sentinel, preservation, and checksum pending |

## Review and Verification

- The initial focused CPU closure passed 15 tests covering dataset labels, static tokenization, configuration validation, person-side sender selection, auxiliary-head updates, zero-weight parameter-tree compatibility, and bridge metrics under all causal interventions.
- After correcting swapped-channel bridge attribution, the full CPU regression closure passed 65 tests with two existing Pydantic warnings in 180.32 seconds. The discriminating selector/evaluation pair also passed independently in 30.06 seconds.
- The cleanup-only tokenizer-fixture consolidation then passed all 56 JAX trainer tests with the same two existing warnings in 179.78 seconds; the nine unchanged generator tests had passed in the corrected full closure.
- Correctness and cleanup reviews passed for exact range `5a5ea7110b73791077f984afd0311b8d8a566ef5..f3c402a1606063cd3d5652689dfb55f511d7ca53` after the diagnostic correction and test-fixture consolidation.
- Remote source, runtime, dataset, disk, output-path, process, lock, VFIO, and TPU health checks remain pending.

## Failure Details

- Last known step: not started.
- No failure. Launch is intentionally blocked on the required reviews, immutable source snapshot, and remote preflight.

## Updates

- `2026-10-16`: Selected explicit bridge-entity supervision after the message-8 and two-round sender arms preserved question relay but showed no beneficial private-passage gap.
- `2026-10-16`: Implemented a training-only middle-entity target on the person-side sender message, preserved zero-weight checkpoint compatibility, and passed the focused and full CPU test closures.
- `2026-10-16`: Correctness review found and verified a diagnostic-only fix so swapped bridge accuracy follows the semantic sender rather than its original channel index; the corrected full suite passed.
- `2026-10-16`: Cleanup consolidated the compact-example tokenizer into one test helper and its complete 56-test JAX file passed; final cleanup re-review passed.
- `2026-10-16T11:33:07Z`: After the exact Kristine v4-8 appeared in the mandatory final `AVAILABLE` table, created immutable launch snapshot `ce27b32f14ccbeb3c37e830d3902c050e4dc5550` and verified its 57-file local manifest, selected-config hash, and cache exclusions.
