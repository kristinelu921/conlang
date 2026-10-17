# T5-tiny warm start with 2x base learning rate and 3x scale

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `running` |
| Started | `2026-10-17T23:10:08Z` |
| Finished | `not recorded` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Git revision | `e38af6e2f3e6761943ed786b32c0737ebf1462aa` (configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` |
| Command | exact command below |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; READY, HEALTHY, and idle at final preflight |

## Command

```bash
/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; trap "" HUP; source_path=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T230015.160947320Z; config_path="$source_path/configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml"; output_path=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200; launcher_log=/home/kristinelu/elf-runs/logs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.launcher.log; model_path=/home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8; test -d "$source_path"; test -f "$config_path"; test ! -e "$output_path"; test ! -e "$launcher_log"; test "$(sha256sum "$model_path/flax_model.msgpack" | cut -d" " -f1)" = 16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26; test "$(sha256sum "$model_path/tokenizer.json" | cut -d" " -f1)" = 88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34; mkdir -p /home/kristinelu/elf-runs/logs /home/kristinelu/elf-runs/wandb; cd "$source_path"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache WANDB_DIR=/home/kristinelu/elf-runs/wandb PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config "$config_path" 2>&1 | tee "$launcher_log"' --quiet
```

## Configuration

Source: `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1-train3m
dataset_prefix: kristine/datasets/conlangs-train3m
batch_size: 256
learning_rate: 0.0006
sender_learning_rate_multiplier: 4.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 30000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack
pretrained_initialization: google_t5_efficient_tiny
pretrained_model_dir: /home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8
pretrained_checkpoint_sha256: 16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26
pretrained_tokenizer_sha256: 88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
wandb_enabled: true
hidden_size: 256
output_hidden_size: 256
sender_hidden_size: 256
sender_num_layers: 4
sender_num_heads: 4
sender_feedforward_size: 1024
sender_average_residual: true
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
num_layers: 4
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Completed baseline | Current | Source |
|---|---:|---:|---|
| Base learning rate | `0.0003` | `0.0006` | selected configurations |
| Training questions | `1,000,000` | `3,000,000` | deterministic dataset manifests |
| Training worlds | `125,000` | `375,000` | 8 questions per world |
| Training steps | `10,000` | `30,000` | selected configurations |
| Training presentations | `2,560,000` | `7,680,000` | steps x global batch 256 |
| Nominal passes over training questions | `2.56` | `2.56` | presentations / training questions |
| Scheduled causal evaluations | `50` | `150` | unchanged 200-step cadence |

Architecture, T5 checkpoint initialization, global batch, sender learning-rate
multiplier, optimizer, gradient clipping, seed, question/message routing, and
evaluation prefix remain matched. The treatment changes learning rate, data
size, and duration together as requested, so it cannot isolate any one change.

## Hypothesis and Decision Rule

The run asks whether the warm-started model's strong sender credit becomes
useful when given three times as much distinct training data, three times as
many optimization steps, and twice the base learning rate while preserving the
same 2.56 nominal data passes. Success still requires a repeatable positive
intact-over-wrong-world validation gap, preferably with improving exact match;
lower teacher-forced loss alone is insufficient. The 150 evaluations will also
show whether any communication transition is persistent rather than a single
loss discontinuity.

## Results

Running. The first compiled interval completed with finite metrics, and the run
reached step 480 at the first installed-monitor check. Steps 20--200 sustained
approximately 2,144--2,153 examples/s, matching the completed baseline. Final
scientific results remain pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` | parses and validates; local and remote SHA-256 `7cb028221a5ee781697ac9c03e325333fdc37edd52d405aead9145ff0bd8ff80` |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T230015.160947320Z`; matching node-local `/home/kristinelu/elf-runs/sources/` path in the command | 71 files / 520,961 bytes / relative-manifest SHA-256 `7354189dd705d2dc07fe5bf1de3f01f65f72e8c1290e29c3833988d2abb0ea88`; no bytecode caches |
| Source model | `/home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8` | checkpoint and tokenizer hashes match the configured fail-closed values |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train3m` | generated node-locally in Central2; 378,125 worlds / 3,025,000 questions / 280,390,862 bytes; full streaming verifier passed; manifest SHA-256 `0e37bb12e22e49e74868bb01d225a29e01bd2d8508df8efbcaa1d2d8aca1c1c4` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack` | no; pending launch |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/rsjglvk6` | W&B ID `rsjglvk6`, state `running`; first monitor observed summary step 440 |
| Recurring monitor | `conlang-monitor-rsjglvk6.service`; run-scoped script `/tmp/conlang_monitor_rsjglvk6.sh` | five-minute cadence; clean-environment cloud/W&B check passed; first service check saw READY/HEALTHY, live trainer, step 480 |

## Failure Details

- Last known step: not applicable; not launched.
- Error summary: not applicable.
- Detailed output: not recorded.

## Observations

- The requested scaling preserves data exposure in epochs: both arms present
  each training example 2.56 times on average.
- The completed baseline sustained a last-50 median `2,143.996` examples/s. At
  that rate, 30,000 training steps contain about 59.7 minutes of steady-state
  work, plus compilation and 150 intervention evaluations.
- The required filtered inventory selected only `kristine` resources; the
  Central2 v4-8 candidate is preferred because it is the proven topology for
  this exact model and retains region-local predecessor data/model artifacts.
- The Central2 GCS prefix contained no 3M dataset objects, so the deterministic
  seed-1729 dataset was generated directly on the selected Central2 host. No
  dataset crossed regions. All gzip streams and stored content hashes verify.
- The final CPU-only preflight reports JAX/JAXlib `0.6.2`, Flax `0.10.5`,
  Optax `0.2.8`, NumPy `1.26.4`, valid W&B authentication, 84 GiB free, fresh
  run output/log paths, writable `/tmp/tpu_logs`, no libtpu lock, and idle VFIO.
- Startup verified all four TPU devices, 3,000,000 training examples, 256 fixed
  validation examples, and the exact 12,580,239-parameter warm-start tree. Step
  10 was finite; step 200 completed at `2147.43` examples/s.
- The generic monitor named by repository policy is absent from this source
  tree. The installed run-scoped fallback pins the direct gcloud binary and a
  W&B-capable Python, is TERM-safe, and passed both a cron-like clean-environment
  check and its first systemd cadence check.

## Interpretation

- The doubled base learning rate is a meaningful stability intervention. The
  global norm clip remains one, but pre-clipping spikes can still change AdamW
  dynamics; finite early steps and the causal curves must be checked before
  interpreting the longer run.

## Updates

- `2026-10-17T22:51:50Z`: Created the planned report and immutable selected
  configuration at `e38af6e2f3e6761943ed786b32c0737ebf1462aa`.
- `2026-10-17T22:56:12Z`: Recorded the configuration/report closure SHA and
  clarified that the throughput estimate uses the predecessor's last-50
  median.
- `2026-10-17T23:09:01Z`: Recorded the exact immutable snapshot, launch
  command, selected-device preflight, source-model hashes, and fully verified
  node-local Central2 dataset.
- `2026-10-17T23:29:04Z`: Marked the run running after finite step 200, recorded
  W&B run `rsjglvk6`, and installed the verified five-minute recurring monitor.
