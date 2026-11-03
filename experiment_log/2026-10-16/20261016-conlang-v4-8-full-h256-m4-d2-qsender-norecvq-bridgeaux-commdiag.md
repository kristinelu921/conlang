# Full-data hidden-256 bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T12:14:53Z` |
| Finished | `2026-10-16T12:24:21Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Git revision | primary configuration commit `570334c62e6ed33d0fe1f6e3b2b30061ace79b7c`; final reviewed configuration/report closure `d45924a8816df48be91e2f8588fd285d063599f4`; immutable launch snapshot HEAD `68a819dad88d1d2bd705af3b696081dabdd9ee00` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU only: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag
wandb_enabled: true
hidden_size: 256
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
| Transformer, embedding, and message-vector width | `128` | `256` | Test coupled model and continuous-channel capacity only after explicit bridge credit has produced beneficial private-passage use |

Configured output hidden width remains `256`, so the final answer vocabulary projection stays fixed; the preceding answer output-hidden input kernel widens from `(128, 256)` to `(256, 256)` with `hidden_size`. The bridge vocabulary projection also widens from `(128, vocab)` to `(256, vocab)` because it consumes the message representation directly. Message length remains four slots per sender, but each continuous message vector widens from 128 to 256 features together with the transformer and token embeddings. Data, seed, optimization, schedule, depth, untied encoders, gates, sender conditioning, receiver-question masking, communication rounds, bridge objective, hardware, and evaluation interventions remain fixed.

## Hypothesis and Decision Rule

The hidden-128 bridge-supervised arm is the first experiment in this campaign where intact messages beat target-question/wrong-passages messages: `17/256` versus `12/256`, with a `+0.0430698`-nat likelihood gap and `32/256` intact bridge accuracy. Its residual error may now reflect insufficient coupled model or continuous-channel capacity rather than the earlier missing sender credit. Because `hidden_size` controls both the transformer representation and each message vector's feature dimension, this experiment cannot isolate which of those two capacity changes causes any gain.

Doubling hidden width should improve the learned bridge representation and the receiver's two-hop join while preserving the private-passage causal gap. Treat width as useful only if intact-over-wrong-passages answer exact match remains positive and preferably exceeds `5/256`, intact bridge accuracy improves beyond `32/256`, and any raw exact-match gain is not explained by question transport alone. Reject width as the next stacked setting if the matched passage gap collapses, even if intact exact match rises. After this untied arm, compare a matched hidden-256 tied-encoder arm rather than confounding width and sharing.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message answer exact match | `0.0546875` (14/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0377454758` | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message bridge exact match | `0.1328125` (34/256) | 2000 | max | `metrics.jsonl` |
| Validation | Wrong-question answer exact match | `0.00390625` (1/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question mean joint log probability | `-10.2477397919` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question bridge exact match | `0.0234375` (6/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages answer exact match | `0.04296875` (11/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0786933899` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages bridge exact match | `0.01953125` (5/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message answer exact match | `0.0` (0/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-7.5067110062` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message bridge exact match | `0.01171875` (3/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender answer / bridge exact match | `0.0546875` / `0.1328125` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only answer / bridge exact match | `0.04296875` / `0.08203125` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only answer / bridge exact match | `0.04296875` / `0.0625` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Total / answer / bridge loss | `7.7067995071` / `3.9201622009` / `3.7866373062` | 2000 | min | `metrics.jsonl` |
| Train | Bridge minibatch accuracy | `0.109375` (28/256) | 2000 | max | `metrics.jsonl` |
| Train | Throughput | `3248.8548 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 / agent-2 / receiver gradient norm | `0.6736315` / `0.7218927` / `1.4012870` | 2000 | diagnostic | `metrics.jsonl` |

## Interpretation

- Beneficial private-passage use replicates at hidden width 256: intact messages beat target-question/wrong-passages messages by 3 exact answers (`14/256` versus `11/256`) and `0.0409479` mean joint-log-probability nats. The matched control retains the target question and changes both private passages before sender encoding, so this gap supports passage-specific use under this evaluation.
- The coupled width increase does not pass the preregistered stacking rule. Relative to hidden width 128, intact answer exact match falls from `17/256` to `14/256`, the exact passage gap narrows from `5/256` to `3/256`, and the likelihood gap narrows from `0.0430698` to `0.0409479` nats. Do not keep width 256 as the default setting from this single-seed comparison.
- Bridge prediction improves slightly from `32/256` to `34/256`. Wrong-question and wrong-passage bridge controls fall to `6/256` and `5/256`, at or below the `6/256` empirical constant-bridge baseline, so the learned bridge signal remains question- and passage-specific. The two-answer bridge gain is too small to offset the weaker downstream answer result.
- Question transport remains strong: matched wrong-question re-encoding drops answer exact match from `14/256` to `1/256` and loses `6.2099943` nats with both passages and the receiver target fixed. Zero messages reach `0/256` and lose `3.4689655` nats; the matched wrong-question arm remains the identifying control.
- Each single-message arm reaches `11/256`, while both intact messages reach `14/256`; this is consistent with useful joint contribution, although aggregate counts do not establish whether the same examples are correct. Swapping sender channels preserves aggregate answer and bridge accuracy.
- The last-50 logged median agent gradients are `0.6741556` and `0.7001513`, versus `1.4652187` for the receiver, or `2.17x` and `2.09x` smaller. Direct bridge credit therefore remains active, but the balance does not improve over the hidden-128 arm's `1.88x` / `1.76x` ratios.
- Final and last-50 median throughput are both `26.81%` below the matched hidden-128 arm. Complete the preregistered matched tied-encoder comparison at hidden width 256 to isolate weight sharing, but do not treat doubled width itself as a kept improvement.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` | committed in `570334c62e6ed33d0fe1f6e3b2b30061ace79b7c`, parses and validates locally, and passed correctness and cleanup review after report-only corrections |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed configuration closure pushed to `codex/jax-tpu-training` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag-20261016T120728.329172883Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` | local and remote matched before launch: 58 files, 442420 bytes, relative SHA-256-manifest digest `f950b1800036e09fbaab6da482b73cecf9d41ddc5e0bea761df84a7e268a8d56`; selected-config SHA-256 `454571ea9d0625088324f58cb5efb55e305217c2f2b84346d6c0eb11297530b9`; no bytecode or test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | complete Central2 data revalidated: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/zhd3x4ld` | authenticated API reports `finished`, step `2000`, answer exact `0.0546875`, and bridge exact `0.1328125`; canonical launcher synced five files |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack` | 84,264,599 bytes; SHA-256 `f57f83e7842d80a2e07bed6a855313cee1335c48631d282d1e9c89611ca809be`; CPU-only Flax msgpack restore verified step 2000 and bridge kernel shape `(256, 2191)` |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/metrics.jsonl` | 364,253 bytes; SHA-256 `dd4594f0e31499a68c0b0f08144d599e005d40f1a4964ad8d1a1b92a2fcef66c`; 201 JSON records parsed |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/outputs.jsonl` | 2,812,186 bytes; SHA-256 `a878cc5487aea7959ea2b9dba2017e08ca699e4c3ede39824cd9fe1f979db01d`; 200 strictly increasing logged steps and 256 rows at step 2000 |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.launcher.log.gz` | 3,966 bytes; compressed SHA-256 `9f9ad1439ba00bb591466b82e81c9446a701d46e1333cb84243730fd2c96b07c`; uncompressed remote SHA-256 `ed63ba4afa2515c1064e13d1a3ac1837eb437b09bcb49f95fbdc859631d82a41`; gzip and completion sentinel verified |

## Review and Verification

- The configuration parses and validates locally. Its exact YAML diff from the completed hidden-128 bridge-supervised baseline is limited to run identity/output fields plus `hidden_size: 128` to `256`.
- Correctness review passed for the committed configuration/report closure after correcting the coupled-width and output-projection descriptions.
- Cleanup review identified two report-only provenance/projection clarifications; cleanup re-review passed at `d45924a8816df48be91e2f8588fd285d063599f4`.
- Remote node-local snapshot, region-local dataset, JAX `0.6.2` runtime, authenticated W&B file presence, writable `/tmp/tpu_logs`, 85 GiB free space, fresh output path, process absence, and TPU-device idleness passed preflight.
- The exact Kristine v4-8 remained in the final mandatory `AVAILABLE` table immediately before launch; the launch-boundary output, lock, and VFIO checks passed.
- The mandatory checker listed the exact Kristine v4-8 in its final `AVAILABLE` table, and `tpu_scripts/run_staging.sh` created the verified immutable 58-file snapshot at HEAD `68a819dad88d1d2bd705af3b696081dabdd9ee00`.
- Completed evidence verification parsed all 201 metrics and 200 output records, recomputed the causal gaps and last-50 medians, matched every preserved artifact hash to its remote source, restored the final checkpoint on CPU, and confirmed W&B `finished` at step 2000.
- The completed-result correctness and cleanup reviews passed at `11d35aee4d66fb7ddd1c2eb276cd42aa21796d66`; cleanup only clarified that both completed bridge-supervised runs had separate authenticated W&B verification.
- Terminal verification found the exact trainer absent, no stale TPU lock, idle VFIO devices, a canonical `TRAINING_COMPLETE ... step=2000` sentinel, and the TPU still `READY` / `HEALTHY`.

## Failure Details

- Last known step: `2000`.
- The first legacy-SCP transfer form used a `/.` source suffix that SCP rejected before copying any files. The fresh remote directory was proved empty and removed, then the complete snapshot directory was copied once and matched the local 58-file manifest exactly. No trainer, run output, or checkpoint existed during recovery.
- Training and all answer/bridge causal evaluations completed successfully. W&B finalized, the checkpoint restored, the exact process exited, the lock disappeared, VFIO became idle, and the TPU remains `READY` / `HEALTHY`.

## Updates

- `2026-10-16`: Selected hidden width 256 as the next single-variable test after explicit bridge supervision established beneficial private-passage use at hidden width 128.
- `2026-10-16T12:07:28Z`: The exact same Kristine v4-8 appeared in the mandatory final `AVAILABLE` table; created and verified the immutable local launch snapshot, selected-config hash, and cache exclusions.
- `2026-10-16`: Recovered the zero-file SCP attempt without changing the run identity, copied the full snapshot directory including dotfiles, and verified the exact remote file count, byte count, manifest digest, and selected-config hash.
- `2026-10-16`: Revalidated JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, W&B `0.28.1`, the complete regional dataset, 85 GiB free space, fresh output path, W&B authentication-file presence, writable `/tmp/tpu_logs`, and idle TPU devices.
- `2026-10-16T12:14:53Z`: Launched on the exact same Kristine v4-8 after it remained in the mandatory final `AVAILABLE` table and all launch-boundary freshness/lock/VFIO checks passed; W&B run `zhd3x4ld` authenticated and began syncing.
- `2026-10-16`: Passed the launch gate with finite losses, bridge accuracy above the empirical constant baseline, active sender gradients, and approximately 3,249 examples/s.
- `2026-10-16T12:24:21Z`: Completed step 2000, every answer and bridge causal arm, checkpoint save, and W&B finalization.
- `2026-10-16`: Verified exact process absence, lock and VFIO idleness, 201 metrics records, 200 output records, artifact hashes, CPU checkpoint restore, W&B `finished`, and TPU `READY` / `HEALTHY` state.
- `2026-10-16`: Completed independent correctness and cleanup review of the preserved result; both passes now report `PASS`.
