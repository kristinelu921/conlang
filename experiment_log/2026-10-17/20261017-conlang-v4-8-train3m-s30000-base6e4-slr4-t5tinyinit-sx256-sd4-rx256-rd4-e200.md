# T5-tiny warm start with 2x base learning rate and 3x scale

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Status | `completed` |
| Started | `2026-10-17T23:10:08Z` |
| Finished | `2026-10-18T00:28:51Z` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-t5tinyinit-sx256-sd4-rx256-rd4-e200` |
| Git revision | `e38af6e2f3e6761943ed786b32c0737ebf1462aa` (configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` |
| Command | exact command below |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; terminal API state READY/HEALTHY |

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

The run completed all 30,000 steps but fails the preregistered communication
criterion. No intervention produced a correct example in any of the 150 fixed
validation-prefix evaluations.

| Intervention at step 30,000 | Exact match | Exact count | Mean joint log probability |
|---|---:|---:|---:|
| Intact | `0.00000000` | 0/256 | `-12.3712816` |
| Wrong-world messages | `0.00000000` | 0/256 | `-12.3695698` |
| Zero messages | `0.00000000` | 0/256 | `-12.3032227` |
| Swapped senders | `0.00000000` | 0/256 | `-12.3710213` |
| Agent 1 only | `0.00000000` | 0/256 | `-12.3836479` |
| Agent 2 only | `0.00000000` | 0/256 | `-12.3906155` |

Intact-minus-wrong-world likelihood is positive in only 70/150 evaluations,
with median `-0.0000575` nats, range `-0.0035400` to `+0.0042601`, and final
gap `-0.0017118`. The sign is unstable and the magnitude remains negligible.
Intact-minus-zero is positive in 73/150 checks, with median `-0.0000763` and
final gap `-0.0680590`; zero messages are better at the final evaluation. The
longer treatment therefore provides neither passage-specific communication nor
a reliably beneficial learned carrier.

At the matched step-10,000 evaluation in the trajectory, the treatment has
zero exact matches and likelihood `-7.7996254`, `0.9970388` nats worse than the
completed baseline's `-6.8025866`. At step 30,000 it is `5.5686951` nats worse
than that baseline endpoint. Because data scale, duration, and learning rate
change together, this is a treatment comparison rather than an isolated
learning-rate effect.

The longer run strengthens sender-side optimization without making it useful.
Over the final 50 logged steps, median path-gradient norms are `33.3000` and
`27.9988` for the sender paths versus `3.18493` for the receiver, giving
receiver-to-sender ratios `0.0956x` and `0.1138x`. Median relative updates are
`0.000942118` for the sender and `0.000136782` for the receiver, a `6.89x`
sender advantage. Sender, receiver, and embedding weight norms fall `34.24%`,
`14.71%`, and `16.44%` from step 10 to 30,000, so rising weight norms are not
the failure mode. In contrast, message activation RMS grows `28.28x` and
`28.49x` to `43.8333` and `43.5214`, indicating a large activation-scale drift.

Only one logged pre-clipping gradient exceeds one million; its maximum is
`5,796,095`, below the baseline's `89,957,224`. Global clipping keeps every
update finite. The largest adjacent five-record loss decline after step 100
begins at step 15,650 (`5.96569` to `5.42605`), but it is not a communication
transition: exact match remains zero and wrong-world stays matched.

The final-50 median training loss is `5.3405187`, only `0.1329637` lower than
the 10,000-step baseline. Median throughput is `2,121.250` examples/s, `1.06%`
below the matched baseline's `2,143.996`. Greedy training-batch generation is
collapsed in all 3,000 output records: every generated string is empty, and at
step 30,000 every row emits token IDs `(2, 0, 0, 0)` (EOS then padding).

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200.yaml` | parses and validates; local and remote SHA-256 `7cb028221a5ee781697ac9c03e325333fdc37edd52d405aead9145ff0bd8ff80` |
| Immutable source | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200-20261017T230015.160947320Z`; matching node-local `/home/kristinelu/elf-runs/sources/` path in the command | 71 files / 520,961 bytes / relative-manifest SHA-256 `7354189dd705d2dc07fe5bf1de3f01f65f72e8c1290e29c3833988d2abb0ea88`; no bytecode caches |
| Source model | `/home/kristinelu/elf-runs/pretrained/google-t5-efficient-tiny-3441d7e8` | checkpoint and tokenizer hashes match the configured fail-closed values |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train3m` | generated node-locally in Central2; 378,125 worlds / 3,025,000 questions / 280,390,862 bytes; full streaming verifier passed; manifest SHA-256 `0e37bb12e22e49e74868bb01d225a29e01bd2d8508df8efbcaa1d2d8aca1c1c4` |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/checkpoint.msgpack` | 150,992,703 bytes; SHA-256 `ce936c4edb80cd77e943d81659cd2f79805a8f3c39366ce2ca10a2cd3a52f2a4`; CPU restore verified step 30,000 |
| Metrics | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/metrics.jsonl` | local-only, not pushed; 5,759,458 bytes; SHA-256 `e3fc6ee20652935e127b5cb1da924eaac4f8b7ede83ee2c04915a62698856629`; 3,150 records: 3,000 training and 150 evaluation |
| Outputs | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/outputs.jsonl` | local-only, not pushed because it contains dataset-derived text; 35,102,397 bytes; SHA-256 `ab463f15a197c7163cdc22577c2f63e82a863fc38b5cb9e467bbb4ece0ce62b4`; 3,000 records through step 30,000 |
| Launcher log | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train3m-s30000-base6e4-slr4-t5tinyinit-sx256-sd4-rx256-rd4-e200/launcher.log.gz` | local-only, not pushed; 44,222 bytes; SHA-256 `58c41294fd23a79b7e53a1a82229d6d7d94510be6196ded7f4d2dccee12cb700`; decompressed SHA-256 `ca1b559cda265ba89bb14bcc5de3e3cb34111f8618246598752e36d3be481724` matches the 251,907-byte remote log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/rsjglvk6` | API state `finished`, summary step 30,000, exact match 0, five files synced |
| Recurring monitor | `conlang-monitor-rsjglvk6.service`; run-scoped script `/tmp/conlang_monitor_rsjglvk6.sh` | five-minute cadence remained healthy; retired after independent terminal verification |

## Failure Details

- Last known step: 30,000, completed.
- Error summary: not applicable.
- Detailed output: canonical launcher log recorded under Artifacts. A local-only
  credential-path preflight failure that made no cloud contact is preserved in
  `20261017-conlang-train3m-runtime-preflight-attempt1.md`.

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
- Terminal verification found no trainer, libtpu lock, or VFIO holder; the TPU
  remained READY/HEALTHY. The checkpoint restored on CPU at step 30,000 and W&B
  independently reported `finished` at summary step 30,000.
- All preserved local artifacts match their remote byte counts and SHA-256
  values. The raw evidence directory remains local and untracked; it is excluded
  from the GitHub push because `outputs.jsonl` contains dataset-derived text.

## Interpretation

- Doubling the base learning rate while tripling data and duration does not
  rescue this objective. The model receives stronger sender-relative updates,
  yet learns neither correct free generation nor a beneficial passage-specific
  message gap.
- Weight norms decline substantially, ruling out simple weight-norm growth as
  the explanation for the failure. Message activations instead expand about
  28x, so activation-scale drift is a concrete target for a narrower follow-up
  such as message normalization or explicit message-scale regularization.
- The result does not isolate learning rate, data size, or training duration.
  It only rejects their requested joint treatment as sufficient for this seed,
  fixed validation prefix, and architecture.

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
- `2026-10-18T00:28:51Z`: Completed step 30,000, saved and restored the final
  checkpoint, verified W&B `finished`, preserved byte-matched evidence, and
  retired the recurring monitor after the TPU returned to READY/HEALTHY idle.
