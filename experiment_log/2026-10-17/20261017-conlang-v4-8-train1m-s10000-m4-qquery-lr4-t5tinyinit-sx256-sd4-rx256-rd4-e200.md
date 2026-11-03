# T5-Efficient-Tiny maximal warm-start pilot

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `completed` |
| Started | `2026-10-17T22:08:09Z` |
| Finished | `2026-10-17T22:40:27Z` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1` |
| Git revision | `b986703a62168220314820334b5cbea1ad0a59d3` (implementation/configuration closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` |
| Command | exact command below |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; terminal API state READY/HEALTHY |

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

The run completed all 10,000 steps but fails the preregistered communication
criterion.

| Intervention at step 10,000 | Exact match | Exact count | Mean joint log probability |
|---|---:|---:|---:|
| Intact | `0.00000000` | 0/256 | `-6.8025866` |
| Wrong-world messages | `0.00000000` | 0/256 | `-6.8023481` |
| Zero messages | `0.00000000` | 0/256 | `-6.8010817` |
| Swapped senders | `0.00000000` | 0/256 | `-6.8023524` |
| Agent 1 only | `0.00000000` | 0/256 | `-6.8025599` |
| Agent 2 only | `0.00000000` | 0/256 | `-6.8019981` |

Intact and wrong-world exact counts tie at zero in all 50 scheduled evaluations.
The intact-minus-wrong-world likelihood gap is positive in only 24/50 checks,
with median `-0.0000172` nats, range `-0.0007157` to `+0.0007038`, and final
gap `-0.0002384`. The sign is unstable and the effect is negligible. Zeroing
both messages is slightly better at the final check by `0.0015049` nats; its
gap is likewise sign-unstable across the run. There is no evidence of useful
passage-specific communication or even a reliably beneficial learned carrier.

The warm start does materially change credit flow. Over the final 50 logged
steps, median path-gradient norms are `42.6011` and `42.8675` for the sender
paths versus `30.4333` for the receiver. The receiver-to-sender ratios are
`0.714x` and `0.710x`, reversing the completed baseline's `5.65x` and `6.39x`
receiver dominance. Median relative update is `0.000423988` for the sender and
`0.000143369` for the receiver, a `2.96x` sender advantage. Strong sender
gradient is therefore not sufficient by itself to learn the desired content.

Norm growth is also not the failure mode in this arm: from step 10 to 10,000,
sender, receiver, and embedding weight norms fall `9.06%`, `2.66%`, and `2.95%`,
respectively. Message RMS still grows about `6.0x` to `8.237/8.289`. Four logged
pre-clipping gradients exceed one million and the maximum is `89,957,224`, but
the configured global norm clip of one keeps updates finite. The largest adjacent
five-record loss decline after step 100 begins at step 5,280 (`6.24545` to
`5.73271`); it is not a communication transition because validation accuracy
does not move and the wrong-world control remains matched.

The final-50 median training loss is `5.4734824`, `1.4816464` nats worse than
the completed larger random-initialized baseline's `3.9918360`. Median throughput
is `2,143.996` examples/s, `67.33%` faster than that baseline's `1,281.261`,
consistent with this run's smaller 12.58M-parameter geometry. This comparison
is not a clean estimate of pretraining because the architectures differ.

Greedy training-batch generations are collapsed from the first logged step
through the last: all 1,000 output records contain one unique decoded string,
the empty string. At step 10,000 every row emits token IDs `(2, 0, 0, 0)`, i.e.
EOS followed by padding. Teacher-forced loss improves, but free generation never
becomes useful.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` | yes, parses and validates |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T220459.865879959Z` | yes, 70 files / 519,174 bytes / relative-manifest SHA-256 `b0b01aacfd2e284c4fe7778be47d132e0fa3999fb0a876f85204683965dcdb85` |
| Source model | `https://huggingface.co/google/t5-efficient-tiny/tree/3441d7e8bf3f89841f366d39452b95200416e4a9` | yes, control-host hashes above |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | yes, manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack` | 150,992,703 bytes; SHA-256 `6ccd6333bcce9dc311b37d722ef2e689ed956fe2651301d26c36f8efa981c1e3`; CPU restore verified step 10,000 |
| Metrics | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/metrics.jsonl` | local-only, not pushed; 1,952,994 bytes; SHA-256 `e8bb86ac9d20849650c5619cb25a68e669d1140b8e5f6987cdb1c0ec5259693f`; 1,050 records: 1,000 training and 50 evaluation |
| Outputs | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/outputs.jsonl` | local-only, not pushed because it contains dataset-derived text; 11,702,369 bytes; SHA-256 `41e6c2180b27d473ea32b69da18adfe161b793ccabe465179488f8f40812d6b7`; 1,000 records through step 10,000 |
| Launcher log | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/launcher.log.gz` | local-only, not pushed; 17,593 bytes; SHA-256 `5ddeeb38fac6b4a1e10b17979826049c6ae19ad3f8361e3f85a94e990d3e242e`; decompressed SHA-256 `6d9df6c5bcfc6c4e64792e45635b096136f7867d3ec573b724eff2d00e2a5f65` matches the 87,907-byte remote log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/scrhodwe` | API state `finished`, summary step 10,000, exact match 0, five files synced |
| Recurring monitor | `conlang-monitor-t5tiny-scrhodwe.service` | five-minute cadence stayed healthy through step 8,540; retired after independent terminal verification |

## Failure Details

- Last known step: 10,000, completed.
- Error summary: not applicable.
- Detailed output: canonical launcher log recorded under Artifacts. The recovered
  prelaunch SSH transport incident is preserved separately in
  `20261017-conlang-t5tinyinit-preflight-ssh-banner-attempt1.md`.

## Observations

- A CPU oracle with the exact production parameter tree initialized all 272
  leaves and produced a finite one-example loss of `16.6716576`, sender gradient
  norm `16.9545975`, and receiver gradient norm `149.6481628`.
- All 71 sender leaves had nonzero first-backward gradients; this removes the
  prior zero-message-projection one-update gate.
- The oracle is a numerical startup check, not an experiment result.
- The complete CPU repository suite passed at the reviewed closure: `147 passed,
  7 subtests passed, 2 existing Pydantic warnings` in `211.29s`.
- Terminal verification found no trainer, libtpu lock, or VFIO holder; the TPU
  remained READY/HEALTHY. The checkpoint restored on CPU at step 10,000 and W&B
  independently reported `finished` at summary step 10,000.
- Artifact SCP encountered transient packet/MAC errors and retried automatically.
  Every preserved local artifact's final SHA-256 matches its remote source.
- The raw evidence directory remains local and untracked. It was intentionally
  excluded from the GitHub push because `outputs.jsonl` contains dataset-derived
  target text; the aggregate report contains no raw examples.

## Interpretation

- Maximal T5-tiny transfer is not sufficient to make this objective discover
  passage-specific communication. It successfully strengthens sender-side credit
  relative to the receiver, but the model still learns no beneficial message
  content and collapses greedy generation to immediate EOS.
- The result is jointly attributable to exact T5 geometry, checkpoint transfer,
  checkpoint-derived boundary initialization, and the retained communication
  adapters. Because it lacks a random-initialized control at identical geometry,
  it cannot isolate the effect of pretraining itself.
- This is not a muP experiment; optimizer scaling is unchanged from the completed
  sender-4x baseline.

## Review and Verification

The correctness pass found that native Flax T5 cancels Flax attention's built-in
query scaling. The importer now multiplies all imported query kernels by
`sqrt(head_dim)` and a differential content-plus-position-bias oracle verifies
source/target logit equivalence. The cleanup pass then centralized that attention
policy and the pretrained artifact boundary. Focused checks and the complete
147-test CPU suite passed before immutable staging. The pushed completed-evidence
delta contains only this aggregate report; the three checksum-matched raw evidence
artifacts remain local and untracked.

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
- `2026-10-17T22:40:27Z`: completed step 10,000, final causal evaluation,
  checkpoint save, completion sentinel, and five-file W&B sync. Independent
  terminal gates restored the checkpoint at step 10,000, matched all artifact
  hashes, found no trainer or libtpu lock, and retired the recurring monitor.
