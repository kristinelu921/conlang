# Full-data message-4 depth-4 communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d4-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T07:28:31Z` |
| Finished | `2026-10-16T07:37:04Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | snapshot HEAD `a25c215d944ab862765b5492f4ef5342100d4ef6`; reviewed configuration commit `52c47a8bd817f63c586d0deea87aea25784fe0df` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d4-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d4-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d4-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
role_gate_channels: null
message_length: 4
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 4
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Transformer layers per stack | 2 | 4 | Test whether character parsing, fact selection, and the two-hop join need more sequential computation |

Every other data, optimization, schedule, seed, width, message-length, sharing, gating, precision, hardware, and causal-evaluation setting remains fixed.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message exact match | `0.0390625` (10/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0590238571` | 2000 | max | `metrics.jsonl` |
| Validation | Zero-message exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-4.0584769249` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message exact match | `0.04296875` (11/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-world-message mean joint log probability | `-4.0588827133` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender mean joint log probability | `-4.0589389801` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only mean joint log probability | `-4.0586633682` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only exact match | `0.03515625` (9/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only mean joint log probability | `-4.0586700439` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Loss | `3.9194679260` | 2000 | min | `metrics.jsonl` |
| Train | Throughput | `3093.1063 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 gradient norm | `0.0053278436` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Agent-2 gradient norm | `0.0058730883` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Receiver gradient norm | `1.1559120417` | 2000 | diagnostic | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d4-commdiag.yaml` | correctness and cleanup reviews passed at `52c47a8bd817f63c586d0deea87aea25784fe0df` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | open; update pending |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d4-commdiag-20261016T072649.664969527Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d4-commdiag` | yes; local and remote contain 52 files and 378346 bytes; relative SHA-256-manifest digest `cbfd1da927bac9fe5da4a5baa5480a380dedb0a56abf5bd37620e27bbff95822` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; same verified Central2 data, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d4-commdiag/checkpoint.msgpack` | yes; SHA-256 `d70ac4c2123fb51e6f5688e0830bf60cafc74cb655cc5df5324a2843f950ec03`; CPU-only restore verified step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/v6cc1t3o` | synced and finalized by the canonical launcher |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d4-commdiag/metrics.jsonl` | yes; SHA-256 `317834825a9e9162854431db3837723cd619ce449f35645375a3109dda237fa8` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d4-commdiag/outputs.jsonl` | yes; SHA-256 `8232770b1d7a11d9c8606d59f6212294269d318acb6918ea9da8390e85f7e271` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d4-commdiag/launcher.log.gz` | yes; compressed SHA-256 `909e6bae7af3174b6a94751ac1743ef4b79eec08c73cbafd4e590fc8a6987390`; uncompressed remote SHA-256 `074fe99095815235620280aa2b507ee5a4261589fe02582154d850695d502c5a` |

## Failure Details

- Last known step: `2000`
- Error summary: training completed successfully. One shell-quoting typo affected only the first read-only checkpoint validation command; the corrected command restored step 2000. Artifact SCP retried transient SSH corruption internally and all local hashes match remote.
- Detailed output: canonical launcher log and metrics above

## Hypothesis and Decision Rule

Message length 8 did not create any causal message dependence, ruling out four slots as the primary bottleneck. The remaining capacity hypothesis is that two layers per sender/receiver stack are insufficient for parsing, selection, and composition. This arm returns to message length 4 and doubles only depth.

Keep depth 4 for later stacking or width/sharing comparisons only if intact messages outperform zero or wrong-world messages in exact match or answer likelihood. If it stays causally flat, the next intervention should target optimization/credit assignment rather than stack capacity.

## Interpretation

- Depth 4 does not create useful communication. Intact, zero, swapped, and agent-1-only messages all produce 10/256; wrong-world messages produce one more correct answer than intact.
- Zero messages improve joint log probability by `0.0005469`, and wrong-world messages improve it by `0.0001411`. Both signs oppose the communication hypothesis.
- Sender gradients remain roughly 197--217 times smaller than receiver gradients.
- Doubling depth cuts throughput from about 5,870 to 3,093 examples/s without a causal benefit. It should not be stacked into subsequent arms.
- With both channel length and depth ruled out, the next test should shorten the credit path architecturally by conditioning each sender on the question, before considering an explicit contrastive or auxiliary communication loss.

## Review and Verification

- Correctness review: pass with no findings. `num_layers` reaches all sender-encoder, receiver-encoder, and decoder loops; a CPU forward pass confirmed four layers, sender messages `(2, 4, 128)`, and logits `(2, 4, 81)`.
- Cleanup review: pass. The immutable closure is one YAML and changes only run identity plus `num_layers: 2 -> 4`.
- `TrainConfig.validate()`, matched-config assertion, three focused tests, and `git diff --check` passed.
- The model has 3,523,599 parameters; reviewer estimates and the prior larger hidden-512 run show no likely v4-8 HBM blocker, with TPU compilation retained as the authoritative gate.

## Updates

- `2026-10-16T07:23:00Z`: Planned after the matched message-8 arm remained causally flat and reduced sender-path gradient norms.
- `2026-10-16T07:28:00Z`: Both reviews passed, PR 1 was updated, the same Kristine v4-8 appeared in the mandatory final `AVAILABLE` table, the immutable source matched locally and remotely, and preflight reported 86 GiB free with VFIO idle.
- `2026-10-16T07:28:31Z`: Launched after an immediate final occupancy gate returned `AVAILABLE`; W&B run `v6cc1t3o` authenticated and began syncing.
- `2026-10-16T07:37:04Z`: Completed step 2000 and all causal arms. The log emitted `TRAINING_COMPLETE`, W&B finalized, checkpoint restore passed, no run process remained, VFIO was idle, and the TPU remained `READY` / `HEALTHY`.
