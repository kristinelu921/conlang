# Full-data message-4 depth-2 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T06:52:28Z` |
| Finished | `2026-10-16T06:57:44Z` |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100` |
| Git revision | snapshot HEAD `b542fee798782c922b4f75254aee71f30c72cf14`; reviewed code/config closure `758129f76a41f6bbe24b641fca10ad6e8f7b1feb` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Evaluation interval | 100 steps | 2000 steps | selected config |
| Causal message interventions | disabled | enabled | selected config |

Training data, batch size, optimization, seed, hidden width, output width, untied encoders, disabled role gates, four message vectors per sender, two-layer stacks, precision, and regional data are unchanged. The less frequent evaluation avoids repeated intervention overhead while preserving an exact final comparison.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.0390625` (10/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0588636398` | 2000 | max | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-4.0565662384` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message mean joint log probability | `-4.0588860512` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0589790344` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0590910912` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0581893921` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9090199471` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `5869.8167 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.0045244885` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.0054018954` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.1364309788` | 2000 | diagnostic | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-commdiag.yaml` | reviewed at `758129f76a41f6bbe24b641fca10ad6e8f7b1feb` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-commdiag-20261016T064945.610203803Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-commdiag` | yes; local and remote contain 50 files and 376372 bytes; relative SHA-256-manifest digest `9a8952b46b9957e0decd59ead4e96571c5eeebce726be6c1e843b0542eeed3aa` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; six gzip files passed integrity, manifest reports 15625 worlds and 125000 questions, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-commdiag/checkpoint.msgpack` | yes; 26418137 bytes; SHA-256 `2596e857c843538e76f18b7325dcecdd281d7a0761818cd5de0372a2e9dbd9c4`; a temporary local copy restored at step 2000 and was not retained in Git |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/7v02cpjc` | synced and finalized by the canonical launcher |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-commdiag/metrics.jsonl` | yes; SHA-256 `e6beab96cb957af0b5b4c436652dd0b8c868b5cd42289d05b2c9f4cd3b8af5c4` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-commdiag/outputs.jsonl` | yes; SHA-256 `997f0d8b6d0bc347edfa1d62ad92f9f0ecae2faca53ee1a18d1f074fdc4143ee` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-commdiag/launcher.log.gz` | yes; compressed SHA-256 `877e98291daa7506bd95422fadf93958967b251c69eb7e87abdf2ddf975bcb5b`; uncompressed remote SHA-256 `8495eaf04d42be679f4fb15b6b89dff17af735818a63925ab6094e8603d8e30a`; contains `TRAINING_COMPLETE` |

## Failure Details

- Last known step: `2000`
- Error summary: training completed successfully. The first artifact SCP stream suffered intermittent SSH MAC/packet corruption and left a partial local file; a clean per-file retry produced a complete canonical archive whose hashes match the TPU copies.
- Detailed output: `experiment_log/2026-10-16/20261016-conlang-v4-8-full-m4-d2-commdiag-artifact-transfer-retry.md`

## Observations

- The treatment is evaluation-only: the optimization path is byte-for-byte identical to the established full-data message-4/depth-2 baseline until final evaluation.
- `wrong_world_message` rotates messages by one verified eight-question world within each device-local validation shard. Each 64-example shard contains eight aligned worlds, so every example receives another world's messages rather than another question's identical same-world messages.
- W&B inventory at `2026-10-16T06:20:15Z` reached the entity and listed 21 projects, but every project run query was inaccessible; no W&B state was inferred from the empty result. Local immutable artifacts remain the primary evidence.
- Every intervention produced the same 10 correct examples out of 256. Zero messages improved mean joint log probability by `0.0022974` relative to intact messages; the other changes were all within `0.0006743` of intact. There is no measurable example-specific message contribution.
- At step 2000, the receiver gradient norm was approximately 251 times agent 1's and 210 times agent 2's. This independently supports sender-path gradient starvation.

## Review and Verification

- Correctness review found that the first wrong-world control reversed rows and could therefore preserve same-world messages. Commit `c6f4c6261fbc155a9977f51e771cf38455a360d4` replaces it with a verified whole-world rotation and rejects invalid device-local layouts.
- Cleanup review found duplicated intervention registration and fallthrough dispatch. Commit `758129f76a41f6bbe24b641fca10ad6e8f7b1feb` makes every supported intervention explicit and rejects unknown values.
- `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py tests/test_linked_facts_generator.py`: 47 passed with two pre-existing Pydantic warnings.
- `git diff --check`: passed.

## Interpretation

- The baseline has not learned example-specific communication: intact messages neither change exact predictions nor improve likelihood over zero or wrong-world messages.
- Sender order is not the blocker: swapping the two senders leaves both exact match and likelihood unchanged.
- Neither sender contributes useful information alone, and keeping both does not help. The most likely immediate bottleneck is optimization/credit assignment rather than width, output width, weight tying, or sender identity.

## Updates

- `2026-10-16T06:35:00Z`: Planned as the causal baseline after existing full-data, width-512, gated, tied/untied, and small-prefix runs all remained near a question-only answer-prior baseline.
- `2026-10-16T06:47:36Z`: Correctness and cleanup review findings were fixed; the complete targeted test suite passed at immutable revision `758129f76a41f6bbe24b641fca10ad6e8f7b1feb`.
- `2026-10-16T06:51:30Z`: The mandatory availability audit selected the requested Kristine v4-8 from the final `AVAILABLE` table. Runtime JAX 0.6.2 saw all four TPU devices, node-local source and regional data passed integrity verification, and remote preflight reported 86 GiB free with VFIO idle.
- `2026-10-16T06:52:28Z`: Launched on the same Kristine v4-8 after a second immediate occupancy gate returned `AVAILABLE`; W&B run `7v02cpjc` authenticated and began syncing.
- `2026-10-16T06:57:44Z`: Completed step 2000 and all six final intervention arms. The canonical process-zero log emitted `TRAINING_COMPLETE`, W&B synced and finalized, the checkpoint restored at step 2000, no run process remained, VFIO was idle, and the TPU remained `READY` / `HEALTHY`.
