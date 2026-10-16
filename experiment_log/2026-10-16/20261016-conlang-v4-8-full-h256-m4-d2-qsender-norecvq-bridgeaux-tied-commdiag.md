# Full-data hidden-256 tied bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T12:53:52Z` |
| Finished | `2026-10-16T13:03:06Z` |
| Baseline | `20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Git revision | primary configuration commit `a8358bcb425953a4ca668fab675ed6e9b8f25a81`; final reviewed configuration/report closure `9adb7ee7096c91fe4b50b50e63d0f2a62761f91a`; immutable launch snapshot HEAD `ea94b35523da4d1e5758ec8bb706afb210e56c98` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU only: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag
wandb_enabled: true
hidden_size: 256
output_hidden_size: 256
tie_sender_receiver_encoder: true
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
| Sender/receiver encoder sharing | untied | tied | Test whether a common fact/question representation improves the already active bridge-supervised communication path |

Data, seed, optimization, schedule, transformer/message width, output hidden width, message length, depth, gates, sender conditioning, receiver-question masking, communication rounds, bridge objective, hardware, and causal evaluation interventions remain fixed.

## Hypothesis and Decision Rule

The hidden-256 untied arm preserves beneficial private-passage use but does not improve over hidden width 128: intact answer accuracy is `14/256`, target-question/wrong-passages accuracy is `11/256`, the likelihood gap is `+0.0409479` nats, bridge accuracy is `34/256`, and throughput is approximately 3,249 examples/s. The remaining sharing hypothesis is that one encoder representation can align sender fact selection with receiver message interpretation and reduce redundant parameters without changing the information path.

Treat tying as useful at hidden width 256 only if the intact-over-wrong-passages answer gap remains positive and preferably exceeds the untied `3/256`, intact bridge accuracy remains above its controls and preferably exceeds `34/256`, and any raw exact-match improvement is passage-specific. Compare the tied result both to the matched hidden-256 untied arm and to the stronger hidden-128 untied bridge arm (`17/256`, `5/256` passage gap). Reject tying if it erases the passage gap or improves only question transport. Do not attribute a tied-versus-untied difference to width because width is held at 256 in both arms.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message answer exact match | `0.05859375` (15/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0369391441` | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message bridge exact match | `0.1484375` (38/256) | 2000 | max | `metrics.jsonl` |
| Validation | Wrong-question answer exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question mean joint log probability | `-10.3030071259` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question bridge exact match | `0.015625` (4/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages answer exact match | `0.05078125` (13/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0623950958` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages bridge exact match | `0.0390625` (10/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message answer exact match | `0.015625` (4/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-6.8123536110` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message bridge exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender answer / bridge exact match | `0.0546875` / `0.1484375` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only answer / bridge exact match | `0.05078125` / `0.0859375` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only answer / bridge exact match | `0.04296875` / `0.0703125` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Total / answer / bridge loss | `7.6111402512` / `3.9295759201` / `3.6815640926` | 2000 | min | `metrics.jsonl` |
| Train | Bridge minibatch accuracy | `0.15234375` (39/256) | 2000 | max | `metrics.jsonl` |
| Train | Throughput | `3333.3300 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 / agent-2 / receiver gradient norm | `3.2667866` / `5.1430597` / `1.4357967` | 2000 | diagnostic | `metrics.jsonl` |

## Interpretation

- Beneficial private-passage use persists with tied encoders: intact messages beat target-question/wrong-passages messages by 2 exact answers (`15/256` versus `13/256`) and `0.0254560` mean joint-log-probability nats. The matched control keeps the target question fixed and changes both sender passages, so the positive gap supports passage-specific use under this evaluation.
- Tying does not pass the preregistered stacking rule. Relative to matched hidden-256 untied, raw exact match improves by one answer (`15/256` versus `14/256`) and bridge accuracy rises by four (`38/256` versus `34/256`), but the answer passage gap narrows from `3/256` to `2/256` and from `0.0409479` to `0.0254560` nats. The bridge passage-control gap also does not improve: `38-10=28` answers versus `34-5=29` untied.
- The stronger hidden-128 untied bridge arm remains best: `17/256` intact answers, a `5/256` passage gap, and a `0.0430698`-nat likelihood gap. Do not keep tying or hidden width 256 as the default from these single-seed comparisons.
- Question transport remains strong. Matched wrong-question re-encoding drops answer exact match from `15/256` to `2/256` and loses `6.2660680` nats with both passages and the receiver target fixed. Zero messages reach `4/256` and lose `2.7754145` nats; the matched wrong-question arm remains the identifying question control.
- Agent-1-only and agent-2-only messages reach `13/256` and `11/256`, while both intact messages reach `15/256`; this is consistent with useful joint contribution, though aggregate counts do not identify example overlap. Swapping sender channels preserves bridge accuracy but loses one answer (`14/256` versus `15/256`), showing slight aggregate order sensitivity.
- The last-50 logged median agent gradients are `1.0572457` and `1.0121885`, versus `1.4390182` for the receiver, or `1.36x` and `1.42x` smaller. Bridge supervision remains an active sender-side learning signal under tying.
- Final throughput is `2.60%` above matched hidden-256 untied, and the last-50 median gain is `2.42%`; the tied checkpoint is also smaller. Those efficiency benefits do not compensate for the weaker private-passage causal gap, so sharing is not selected as the scientific best arm.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.yaml` | committed in `a8358bcb425953a4ca668fab675ed6e9b8f25a81`, parses and validates locally, and passed correctness and cleanup review |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed configuration closure pushed to `codex/jax-tpu-training` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag-20261016T124907.241580898Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag` | local and remote matched before launch: 59 files, 443635 bytes, relative SHA-256-manifest digest `028745fe8092d4599664871869e6360d926cc2d030529c40c4066dde05397451`; selected-config SHA-256 `b493f9b5a3af7a407ff53930bcef2f2424675baad906850f5e612b980d72bac6`; no bytecode or test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | complete Central2 data revalidated: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/vg8dl4h2` | authenticated API reports `finished`, step `2000`, answer exact `0.05859375`, and bridge exact `0.1484375`; canonical launcher synced five files |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/checkpoint.msgpack` | 65,303,687 bytes; SHA-256 `8fc8521de7dfd75ef9c915e7b56a505700b504b24d210a272ba004368dc6fcbe`; CPU-only Flax msgpack restore verified step 2000, one `shared_sender_receiver_encoder`, and bridge kernel shape `(256, 2191)` |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/metrics.jsonl` | 377,754 bytes; SHA-256 `ac1de89c2e1e4a2ee38ab90700355fd6764de41cda5772b8ecc8ecc745712763`; 201 JSON records parsed |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/outputs.jsonl` | 2,814,217 bytes; SHA-256 `3c715202348a71c9475ee971293ac78c7534dbafa5d45ff7ae574fae1156b176`; 200 strictly increasing logged steps and 256 rows at step 2000 |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag/20261016-conlang-v4-8-full-h256-m4-d2-qsender-norecvq-bridgeaux-tied-commdiag.launcher.log.gz` | 4,044 bytes; compressed SHA-256 `e64086d74416546f4110e207a92b2d7df05874b02bd6ea944eb1030712c07b00`; uncompressed remote SHA-256 `e5149f7d35da1113b9e335a30ed1d63fbe014bbc123c201b9186115029ed33aa`; gzip and completion sentinel verified |

## Review and Verification

- The configuration parses and validates locally. Its exact YAML diff from the completed hidden-256 untied bridge-supervised baseline is limited to run identity/output fields plus `tie_sender_receiver_encoder: false` to `true`.
- Correctness review passed for the committed configuration/report closure, including five focused CPU cases in 63.84 seconds.
- Cleanup review identified this report-only provenance update; cleanup re-review passed at `9adb7ee7096c91fe4b50b50e63d0f2a62761f91a`.
- The mandatory checker listed the exact Kristine v4-8 in its final `AVAILABLE` table, and `tpu_scripts/run_staging.sh` created the verified immutable 59-file snapshot at HEAD `ea94b35523da4d1e5758ec8bb706afb210e56c98`.
- Remote node-local snapshot, region-local dataset, JAX `0.6.2` runtime, authenticated W&B file presence, writable `/tmp/tpu_logs`, 85 GiB free space, fresh output path, process absence, and TPU-device idleness passed preflight.
- The exact Kristine v4-8 remained in the final mandatory `AVAILABLE` table immediately before launch; the launch-boundary output, lock, and VFIO checks passed.
- Completed evidence verification parsed all 201 metrics and 200 output records, recomputed the causal gaps and last-50 medians, matched every preserved artifact hash to its remote source, restored the final checkpoint on CPU, and confirmed W&B `finished` at step 2000.
- Terminal verification found the exact trainer absent, no stale TPU lock, idle VFIO devices, a canonical `TRAINING_COMPLETE ... step=2000` sentinel, and the TPU still `READY` / `HEALTHY`.

## Failure Details

- Last known step: `2000`.
- Training and all answer/bridge causal evaluations completed successfully. W&B finalized, the checkpoint restored, the exact process exited, the lock disappeared, VFIO became idle, and the TPU remains `READY` / `HEALTHY`.

## Updates

- `2026-10-16`: Selected encoder tying as the next single-variable test after hidden width 256 preserved private-passage use but failed the capacity stacking rule.
- `2026-10-16T12:49:07Z`: The exact same Kristine v4-8 appeared in the mandatory final `AVAILABLE` table; created and verified the immutable local launch snapshot, selected-config hash, and cache exclusions.
- `2026-10-16`: Copied the complete snapshot directory to node-local storage and verified the exact remote file count, byte count, manifest digest, selected-config hash, and cache exclusions.
- `2026-10-16`: Revalidated JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, W&B `0.28.1`, the complete regional dataset, 85 GiB free space, fresh output path, W&B authentication-file presence, writable `/tmp/tpu_logs`, and idle TPU devices.
- `2026-10-16T12:53:52Z`: Launched on the exact same Kristine v4-8 after it remained in the mandatory final `AVAILABLE` table and all launch-boundary freshness/lock/VFIO checks passed; W&B run `vg8dl4h2` authenticated and began syncing.
- `2026-10-16`: Passed the launch gate with finite losses, bridge accuracy above the empirical constant baseline, active sender gradients, and approximately 3,330 examples/s.
- `2026-10-16T13:03:06Z`: Completed step 2000, every answer and bridge causal arm, checkpoint save, and W&B finalization.
- `2026-10-16`: Verified exact process absence, lock and VFIO idleness, 201 metrics records, 200 output records, artifact hashes, CPU checkpoint restore, W&B `finished`, and TPU `READY` / `HEALTHY` state.
