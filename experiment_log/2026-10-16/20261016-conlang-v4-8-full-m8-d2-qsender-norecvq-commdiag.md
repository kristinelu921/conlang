# Full-data sender-only question-path message-8 diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T09:46:38Z` |
| Finished | `2026-10-16T09:52:39Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Git revision | immutable launch snapshot HEAD `88a296a353902e74bfe44c5d4874c8c164e7b20e`; reviewed implementation/configuration closure `45550aa07ed5cf5b01ef10c7c3cdc30fa8c90514` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m8-d2-qsender-norecvq-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m8-d2-qsender-norecvq-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m8-d2-qsender-norecvq-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: true
receiver_question_input: false
message_length: 8
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 2
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters and Diagnostics

| Parameter | Baseline | Current | Rationale |
|---|---|---|---|
| Message vectors per sender | 4 | 8 | Test whether channel capacity helps only after the direct receiver-question shortcut is blocked and learned messages become necessary |
| Wrong-question evaluation arm | absent | within-world sender-question rotation | Isolate question identity while keeping both private passages and the learned-message distribution intact |

All data, seed, optimization, schedule, widths, depth, sharing, gates, sender conditioning, receiver masking, hardware, and other causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

The message-4 bypass made learned messages causally necessary and increased sender gradients roughly tenfold, but target-question/wrong-passages remained tied with intact messages. Message length may therefore matter only after the cheap direct-question path is blocked. Eight vectors give each sender one slot per private fact instead of forcing eight mixed facts through four query slots.

The new matched wrong-question control rotates questions by one position inside every contiguous eight-question world block before sender encoding. Each example keeps its original two private passages, target answer, and receiver inputs; the receiver question path remains masked. An intact-versus-wrong-question gap tests whether learned messages preserve target-question identity. The existing target-question/wrong-passages arm tests whether they preserve useful private evidence. The all-zero arm establishes only dependence on learned messages, not what those messages encode.

Keep the wider channel only if intact messages gain materially over target-question/wrong-passages messages, ideally in exact match as well as likelihood. If intact also beats wrong-question messages, question identity is being transmitted; if not, the strong zero gap reflects generic learned-message dependence. If private-passage sensitivity remains negligible, the next intervention is explicit intermediate supervision or sequential cross-agent exchange rather than more generic width, depth, or sharing.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.03515625` (9/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0538139343` | 2000 | max | `metrics.jsonl` |
| Validation | Wrong-question exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question mean joint log probability | `-10.3296432495` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0534553528` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-6.2542524338` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0536913872` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0534548759` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0553836823` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9214277267` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `4412.9498 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.7966639996` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.3486467302` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.2140392065` | 2000 | diagnostic | `metrics.jsonl` |

## Interpretation

- The matched control establishes beneficial question-identity transport through learned messages. Re-encoding the same passages with another question from the same world drops exact match from 9/256 to 2/256 and mean joint log probability by `6.2758293` nats. This conclusion does not rely on the out-of-distribution all-zero arm.
- Private-passage transport remains absent under the task decision rule. Target-question/wrong-passages preserves all 9 exact answers and improves mean joint log probability by `0.0003586`; doubling message length therefore does not produce a beneficial passage-specific causal gap.
- The all-zero arm also falls to 2/256, but its likelihood gap is only `2.2004385` nats. Wrong-question messages are substantially worse than zeros in likelihood, consistent with the receiver expecting question-selective message content rather than merely nonzero activations.
- Exact accuracy remains at the 9/256 question-domain-prior baseline. Message length 8 successfully exposes what the active channel carries—question identity—but does not make the two agents bind their complementary private hops.
- The final-step sender gradients spike to `0.7966640` and `0.3486467`, but the last-50 logged medians are only `0.0576564` and `0.0471884` versus receiver `1.2445597`. This is more sender signal than message length 4, but it still does not translate into private-fact use.
- Throughput is `0.84%` below the message-length-4 bypass (`4412.95` versus `4450.50 examples/s`), so the wider channel has negligible compute cost but no fact-communication benefit. Per the preregistered rule, do not retain it as a solved communication setting; the next intervention should directly supervise intermediate content or add sequential cross-agent exchange.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m8-d2-qsender-norecvq-commdiag.yaml` | parses; correctness and cleanup reviews passed |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed launch closure pushed; completed-result correctness and cleanup reviews passed |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag-20261016T093615.911531509Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag` | local and remote matched before launch: 55 files, 418494 bytes, relative SHA-256-manifest digest `81784292bc60f6e727f4c6c928b3fa31ffc30c2a54262fd2c6c0e51fcfd02eab`; selected-config SHA-256 `9a747238adf2db764b51874f101f3263847c5fe2b43faf7eb1c9bf0fe8f4a2c6`; all 55 snapshot files remained byte-identical after the run, excluding W&B's separate run directory |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | revalidated on the TPU: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/0cs19yd6` | synced and finalized by the canonical launcher |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag/checkpoint.msgpack` | 26,424,281 bytes; SHA-256 `9e134e2ece80a67bfe3c4b3b6180acc6b74f235cfbe5749b8f5c122f6709fabf`; CPU-only restore verified step 2000 |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag/metrics.jsonl` | 348,736 bytes; SHA-256 `e1b1db2df4d94b808e559e6c6cb4e21b0c484684ce3af8b0f6e616161f908587` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag/outputs.jsonl` | 2,809,638 bytes; SHA-256 `1dd61639830fba7c1cc74de109af77d9d57d0cd8e4903fa1e6c8c211c6d59cf5` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m8-d2-qsender-norecvq-commdiag/launcher.log.gz` | 3,953 bytes; compressed SHA-256 `780b360f4390b08bbcf5367bc3efe1b8e949dd2c9b66b206b15f49bbe46024e9`; uncompressed remote SHA-256 `4b4b835ccc8074d2d9a45450fdb20d730b1949d088e434b2ba3f79749b3cd563` |

## Review and Verification

- The full CPU regression closure passed 54 tests with two existing Pydantic warnings.
- Focused tests verify that the wrong-question arm keeps passage rows fixed, rotates questions only within aligned world blocks, preserves the target receiver question in teacher-forced scoring, works in greedy decoding, and appears only for conditioned senders.
- Correctness and cleanup reviews passed for the exact implementation/configuration closure.
- Correctness and cleanup reviews passed for exact completed-result range `3a6dffb075e5293589f8adf47d5f62562bea85f4..7597b5c5a962eaa4a7275a2d771516c2a6e5c508`.
- Terminal verification found the exact trainer absent, VFIO idle, a canonical `TRAINING_COMPLETE ... step=2000` sentinel, a CPU-restorable step-2000 checkpoint, and the TPU still `READY` / `HEALTHY`.

## Failure Details

- Last known step: `2000`.
- Training and evaluation completed successfully. The only pre-launch issue was the documented bytecode-cache contamination created by a read-only import check; the exact caches were removed and the source manifest was restored before launch.
- W&B finalized, the exact training process exited, VFIO became idle, all 55 snapshot files remained byte-identical, and the TPU remained `READY` / `HEALTHY`.

## Updates

- `2026-10-16`: Selected message length 8 as the next controlled interaction after the receiver-question bypass established learned-message necessity but not useful private-content transport.
- `2026-10-16`: Added a matched within-world wrong-question control so the next evaluation can distinguish question identity from generic learned-message dependence.
- `2026-10-16T09:36:15Z`: Created immutable launch snapshot `88a296a353902e74bfe44c5d4874c8c164e7b20e` and verified its 55-file local manifest.
- `2026-10-16T09:40:31Z`: Verified an identical node-local source manifest, JAX `0.6.2` runtime imports, the complete regional dataset, 85 GiB free space, and idle TPU devices; the exact run and output paths were absent.
- `2026-10-16T09:43:30Z`: Preflight imports created three current-user `linked_facts/__pycache__/*.pyc` files in the remote source. With no trainer active, removed only those regenerable caches and their empty directory, then restored and reverified the exact 55-file, 418494-byte snapshot and digest. Further import checks use `PYTHONDONTWRITEBYTECODE=1`.
- `2026-10-16T09:46:38Z`: Launched after the exact same Kristine v4-8 passed the final mandatory `AVAILABLE` audit and launch-boundary path, lock, and VFIO checks; W&B run `0cs19yd6` authenticated and synced.
- `2026-10-16T09:52:39Z`: Completed step 2000, all seven causal arms, checkpoint save, and W&B finalization.
- `2026-10-16T09:57:20Z`: Verified the canonical completion sentinel, exact process absence, VFIO idleness, checkpoint restore, artifact hashes, unchanged snapshot files, and TPU `READY` / `HEALTHY` state.
- `2026-10-16`: Correctness and cleanup reviews passed for the exact completed-result update; all reported metrics, causal deltas, artifact hashes, provenance, and qualified interpretations were independently verified.
