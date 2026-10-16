# Full-data hidden-256 bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Status | `planned` |
| Started | not started |
| Finished | not applicable |
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

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` | committed in `570334c62e6ed33d0fe1f6e3b2b30061ace79b7c`, parses and validates locally, and passed correctness and cleanup review after report-only corrections |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed configuration closure pushed to `codex/jax-tpu-training` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag-20261016T120728.329172883Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` | local and remote matched before launch: 58 files, 442420 bytes, relative SHA-256-manifest digest `f950b1800036e09fbaab6da482b73cecf9d41ddc5e0bea761df84a7e268a8d56`; selected-config SHA-256 `454571ea9d0625088324f58cb5efb55e305217c2f2b84346d6c0eb11297530b9`; no bytecode or test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | complete Central2 data revalidated: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | pending | W&B enabled; final run identity must match the selected configuration |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack` | pending |

## Review and Verification

- The configuration parses and validates locally. Its exact YAML diff from the completed hidden-128 bridge-supervised baseline is limited to run identity/output fields plus `hidden_size: 128` to `256`.
- Correctness review passed for the committed configuration/report closure after correcting the coupled-width and output-projection descriptions.
- Cleanup review identified two report-only provenance/projection clarifications; cleanup re-review passed at `d45924a8816df48be91e2f8588fd285d063599f4`.
- Remote node-local snapshot, region-local dataset, JAX `0.6.2` runtime, authenticated W&B file presence, writable `/tmp/tpu_logs`, 85 GiB free space, fresh output path, process absence, and TPU-device idleness passed preflight.
- Pending final mandatory TPU availability and launch-boundary VFIO/lock/output checks immediately before the foreground launch.
- The mandatory checker listed the exact Kristine v4-8 in its final `AVAILABLE` table, and `tpu_scripts/run_staging.sh` created the verified immutable 58-file snapshot at HEAD `68a819dad88d1d2bd705af3b696081dabdd9ee00`.

## Failure Details

- Last known step: not started.
- The first legacy-SCP transfer form used a `/.` source suffix that SCP rejected before copying any files. The fresh remote directory was proved empty and removed, then the complete snapshot directory was copied once and matched the local 58-file manifest exactly. No trainer, run output, or checkpoint existed during recovery.

## Updates

- `2026-10-16`: Selected hidden width 256 as the next single-variable test after explicit bridge supervision established beneficial private-passage use at hidden width 128.
- `2026-10-16T12:07:28Z`: The exact same Kristine v4-8 appeared in the mandatory final `AVAILABLE` table; created and verified the immutable local launch snapshot, selected-config hash, and cache exclusions.
- `2026-10-16`: Recovered the zero-file SCP attempt without changing the run identity, copied the full snapshot directory including dotfiles, and verified the exact remote file count, byte count, manifest digest, and selected-config hash.
- `2026-10-16`: Revalidated JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, W&B `0.28.1`, the complete regional dataset, 85 GiB free space, fresh output path, W&B authentication-file presence, writable `/tmp/tpu_logs`, and idle TPU devices.
