# T5-Efficient-Tiny maximal warm-start pilot

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `running` |
| Started | `2026-10-17T22:08:09Z` |
| Finished | pending |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1` |
| Git revision | `b986703a62168220314820334b5cbea1ad0a59d3` (implementation/configuration closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` |
| Command | exact command below; pending final occupancy gate |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; pending final occupancy gate |

## Command

```bash
/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=0 --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; trap "" HUP; source_path=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T220459.865879959Z; config_path="$source_path/configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml"; output_path=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200; launcher_log=/home/kristinelu/elf-runs/logs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.launcher.log; model_path=/home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8; test -d "$source_path"; test -f "$config_path"; test ! -e "$output_path"; test ! -e "$launcher_log"; test "$(sha256sum "$model_path/flax_model.msgpack" | cut -d" " -f1)" = 16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26; test "$(sha256sum "$model_path/tokenizer.json" | cut -d" " -f1)" = 88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34; mkdir -p /home/kristinelu/elf-runs/logs /home/kristinelu/elf-runs/wandb; cd "$source_path"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache WANDB_DIR=/home/kristinelu/elf-runs/wandb PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config "$config_path" 2>&1 | tee "$launcher_log"' --quiet
```

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1-train1m
dataset_prefix: kristine/datasets/conlangs-train1m
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 4.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 10000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack
pretrained_initialization: google_t5_efficient_tiny
pretrained_model_dir: /home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8
pretrained_checkpoint_sha256: 16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26
pretrained_tokenizer_sha256: 88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200
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
| Initialization | random Flax initializers | maximal `google/t5-efficient-tiny` warm start | selected config and `pretrained_initialization.py` |
| Total parameters | `14,845,711` | `12,580,239` | initialized production trees |
| Sender hidden width | `384` | `256` | selected configs |
| Sender depth | `4` | `4` | selected configs |
| Sender heads / head width | `12 / 32` | `4 / 64` | selected configs |
| Sender FFN width | `1536` | `1024` | selected configs |
| Receiver hidden width | `256` | `256` | selected configs |
| Receiver depth | `2` | `4` | selected configs |
| Receiver heads / head width | `8 / 32` | `4 / 64` | selected configs |
| Receiver FFN width | `1024` | `1024` | selected configs |
| Output-hidden width | `1024` | `256` | selected configs |

Training data, batch size, 10,000-step budget, seed, optimizer, sender learning-rate
multiplier, message length, sender sharing, question-only receiver query path,
question-blind senders, and 200-step causal evaluation cadence remain unchanged.
This is necessarily a coupled architecture-and-initialization pilot, not an
isolated estimate of pretraining benefit. A random-initialized run with this
exact 256/4/1024 geometry would be required for that isolation.

## Initialization Contract

The source is the native Flax checkpoint for
`google/t5-efficient-tiny` at immutable Hugging Face revision
`3441d7e8bf3f89841f366d39452b95200416e4a9`. The 62,286,648-byte
`flax_model.msgpack` has SHA-256
`16e959d18596c0cdd9da07fd4fb913f5f9aa7835decfd6a4b9f9fd96e960da26`;
the 1,387,470-byte `tokenizer.json` has SHA-256
`88fe1105ba0554a7bf747a0da453b6d5d23c31805f915731ddb4f419bdc45e34`.
Both hashes were verified on the control host and are launch-time fail-closed
inputs.

Every target parameter leaf receives an explicit value. Of 12,580,239 target
parameters across 272 leaves, 10,167,296 parameters in 119 leaves are copied or
reshaped directly from matching T5 encoder/decoder tensors; 2,368,000 parameters
in 22 leaves are checkpoint-derived task embeddings, sentinel query slots,
attention-scaled query kernels, T5 final-normalization diagonals, and the tied-style
task readout; and 44,943
parameters in 131 leaves use explicit zero fallbacks for biases and decoder
cross-position biases that do not exist in T5. Thus 99.64% of scalar parameters
are checkpoint-direct or checkpoint-derived, while no random initializer value
survives.

The task keeps its existing character inputs and one-token answer vocabulary.
Each task token is initialized by averaging its T5 SentencePiece rows; the whole
source embedding table is RMS-normalized first so its unusually large trained
scale does not create pathological post-norm logits. The four sender query slots
use `<extra_id_0>` through `<extra_id_3>`. The sender message and receiver output-
hidden adapters use the T5 encoder and decoder final-normalization weights as
diagonal projections. The answer readout reuses the derived task embeddings with
the T5 `1/sqrt(d_model)` scale. Imported query kernels are multiplied by
`sqrt(head_dim) = 8` because native Flax T5 cancels Flax attention's built-in
query scaling; a differential logit oracle verifies content logits and relative
position bias retain their source balance.

This transfers parameter content, not the exact pretrained T5 function. T5 uses
pre-norm RMSNorm and bias-free layers, while this communication model retains
post-norm LayerNorm and explicit biases. The character-level input representation
also differs from T5 pretraining. These are interpretation limitations rather
than hidden configuration changes.

## Hypothesis and Decision Rule

The pilot asks whether language-pretrained directions and a live nonzero sender
path can help answer-token cross-entropy discover passage-specific communication.
The primary success criterion remains a repeatable intact-over-wrong-world
validation advantage across the 50 scheduled evaluations, preferably in exact
match and with a stable positive likelihood gap. Intact-versus-zero remains only
an out-of-distribution learned-carrier diagnostic. Earlier loss reduction is
interesting only if it is accompanied by the matched passage intervention gap;
otherwise it may again be question-domain prior learning.

## Results

Running. The compiled path reached step 10 with training loss `10.3851`, then
step 200 with loss `6.6595` and `2,159.90` examples/s. The independent recurring
gate observed one exact trainer, READY/HEALTHY TPU state, W&B `running`, and
step 640 at `2026-10-17T22:21:41Z`. Final causal evaluations and checkpoint are
pending.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` | yes, parses and validates |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T220459.865879959Z` | yes, 70 files / 519,174 bytes / relative-manifest SHA-256 `b0b01aacfd2e284c4fe7778be47d132e0fa3999fb0a876f85204683965dcdb85` |
| Source model | `https://huggingface.co/google/t5-efficient-tiny/tree/3441d7e8bf3f89841f366d39452b95200416e4a9` | yes, control-host hashes above |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | yes, manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | node-local path from the configuration | pending final step |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/scrhodwe` | yes, authenticated and running |
| Metrics and outputs | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/` | yes, advancing |
| Recurring monitor | `conlang-monitor-t5tiny-scrhodwe.service` | yes, five-minute cadence; first check completed and entered one sleep child |

## Failure Details

- Last known step: not applicable
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- A CPU oracle with the exact production parameter tree initialized all 272
  leaves and produced a finite one-example loss of `16.6716576`, sender gradient
  norm `16.9545975`, and receiver gradient norm `149.6481628`.
- All 71 sender leaves had nonzero first-backward gradients; this removes the
  prior zero-message-projection one-update gate.
- The oracle is a numerical startup check, not an experiment result.
- The complete CPU repository suite passed at the reviewed closure: `147 passed,
  7 subtests passed, 2 existing Pydantic warnings` in `211.29s`.

## Interpretation

- Any result is jointly attributable to exact T5 geometry, checkpoint transfer,
  checkpoint-derived boundary initialization, and the retained communication
  adapters. It cannot by itself establish that pretraining rather than geometry
  caused a change.
- This is not a muP experiment; optimizer scaling is unchanged from the completed
  sender-4x baseline.

## Updates

- `2026-10-17T21:35:00Z`: planned report created after artifact-integrity,
  complete-tree mapping, finite-loss, and first-backward gradient checks.
- `2026-10-17T21:48:00Z`: recorded immutable implementation/configuration
  closure and the complete CPU verification result.
- `2026-10-17T21:50:23Z`: corrected imported query-kernel scaling to preserve
  native Flax T5 attention logits and recorded the differential and production-tree
  oracle results.
- `2026-10-17T21:59:19Z`: resolved the cleanup review by centralizing attention
  transfer policy and pretrained artifact binding; the final full CPU suite passed.
- `2026-10-17T22:06:00Z`: selected the available READY/HEALTHY Central2 v4-8,
  staged and byte-verified the immutable source and pinned model artifacts, verified
  the existing regional dataset and W&B login, and passed the standard disk/device/
  output-freshness preflight through the documented explicit-worker SSH recovery.
- `2026-10-17T22:08:09Z`: launched the exact recorded command; W&B run
  `scrhodwe` initialized before JAX opened the TPU.
- `2026-10-17T22:21:41Z`: first-step and steady-state execution verified. The
  five-minute recurring monitor observed one trainer and step 640 while the TPU
  remained READY/HEALTHY and W&B remained running.
