# Question-query/message-KV sender-update balancing diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T19:57:55Z` |
| Finished | `2026-10-16T20:03:52Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` |
| Git revision | primary implementation/configuration commit `f60a841729595dee407d66fbaf6b6b7f1ab07ad5`; correctness-fix and cleanup-reviewed closure `75dd86d259ec97bc17ddc53d3368175769ed74af`; immutable launch source `e4f4faf3abd2b7660f48ef399ed121f289a36a47` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag; output_dir=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag; launcher_log=/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml 2>&1 \| tee "$launcher_log"'` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host and four TPU devices; terminal state `READY` / `HEALTHY` |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 4.0
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
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
num_layers: 2
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Sender-private post-Adam update multiplier | 1.0 | 4.0 | Increase actual sender parameter movement without changing the answer-token objective, receiver optimizer, shared token embedding, model, data, or seed |

The implementation also logs sender, receiver, embedding, and total update
norms plus sender/receiver update-to-weight ratios. This telemetry does not
change optimization, but the historical baseline did not record these fields
and did not pay their logged-step computation cost.

## Hypothesis and Decision Rule

The matched query/message-KV run starts from sender and receiver weight norms of
`60.0652142` and `107.1733465`, with per-parameter RMS values `0.0952922` and
`0.0816988`. The receiver has `4.33x` as many private parameters, but that size
difference is far too small to explain the final approximately `330x` raw path-gradient
ratio. By step 2000, sender weight norm changes only `+0.04%`, while receiver
weight norm rises approximately `+8.0%`. This argues against a simple initial
weight-scale mismatch and instead suggests persistent end-to-end credit
attenuation along the longer sender path.

Adam normalizes each coordinate by its gradient moments, so multiplying raw
sender gradients by a constant would mostly cancel inside Adam. This treatment
therefore multiplies the sender-private update after AdamW by `4.0`, including
its proportionate weight-decay update, while leaving receiver and shared-token
updates unchanged. It adds no labels or auxiliary objective.

Treat balancing as a communication improvement only if intact messages beat
both zero and wrong-world messages in exact match and mean joint log
probability. Interpret sender/receiver update-to-weight ratios within this run;
the historical baseline cannot supply a matched update-ratio comparison.
Compare weight-norm drift, path-gradient norms, loss, and causal evaluation to
the exact query/message-KV baseline. Report throughput descriptively, but do
not attribute its difference solely to the multiplier because the new logged-
step update telemetry adds instrumentation work. A larger sender update without
an intact-versus-wrong-world advantage is optimizer activity, not evidence of
useful content transport.

## Results

| Evaluation arm | Exact match | Correct | Mean joint log probability |
|---|---:|---:|---:|
| Intact messages | 0.015625 | 4/256 | -8.108983993530273 |
| Zero messages | 0.0078125 | 2/256 | -6.84065055847168 |
| Wrong-world messages | 0.015625 | 4/256 | -8.108094215393066 |
| Swapped senders | 0.015625 | 4/256 | -8.111234664916992 |
| Agent 1 only | 0.015625 | 4/256 | -6.694980621337891 |
| Agent 2 only | 0.015625 | 4/256 | -6.669425010681152 |

The treatment fails the preregistered communication rule. Intact and
wrong-world messages tie at `4/256`, and wrong-world messages are marginally
better by `0.0008898` nats. Zero messages lose two exact answers but improve
mean joint log probability by `1.2683334` nats. Both single-sender arms keep
all four exact answers and improve likelihood even more. The final logged
training batch greedily emits `panther` for all 256 examples. The larger sender
updates therefore improve the model's global/type-level answer fit, but do not
make aligned private passage content beneficial.

The sharp middle-of-training loss transition is reproducible structure rather
than a logging artifact. With the `4x` sender update, the largest non-initial
50-step loss-level change after step 100 begins near step 830
(`5.2886` before versus `4.9078` after). The matched `1x` baseline shows the
analogous transition around step 980 (`5.2908` versus `4.8860`), approximately
150 steps later. In the treatment,
sender-path gradients and sender update-to-weight ratios jump at step 820 before
the loss breaks, while message RMS begins rising. However, greedy outputs remain
one constant token across every example in each logged batch throughout this
window, and the terminal wrong-world control remains tied. The transition is
therefore consistent with an earlier coupled receiver/message-carrier phase
change or better answer-prior fitting, not acquisition of passage-specific
communication.

Over the last 50 logged training steps, median raw path-gradient norms are
`0.0014334` and `0.0014490` for the senders and `1.4967552` for the receiver,
giving receiver-to-sender ratios of `1044.22x` and `1032.96x`. Those raw ratios
are larger than the baseline's `329.59x` / `330.82x`, but they do not describe
the applied Adam updates. Within the treatment, median sender-private and
receiver-private update norms are `0.0685696` and `0.0564448`; their median
update-to-weight ratios are `0.00114784` and `0.000485916`. Sender parameters
therefore move `2.36x` as far relative to their weights as receiver parameters.
The final sender weight norm is `60.8536835`, `+1.31%` from the shared
same-seed initialization, versus only `+0.04%` in the baseline. The intervention
successfully reallocates optimizer motion and rules out insufficient aggregate
sender movement as the sole cause of the communication failure. Sender motion
also accelerates again over steps 1910--2000: the final sender update-to-weight
ratio reaches `0.0112227`, `15.42x` the receiver's `0.000727815`, without
opening a beneficial wrong-world gap.

Median final-50 training loss improves from `4.21608` to `4.01814`, while
intact validation exact match moves only from `3/256` to `4/256` and the causal
gap remains absent. Median measured throughput is `5555.26` examples/s versus
`5686.64` for the historical baseline, a `2.31%` reduction; this difference is
descriptive because the treatment also computes new update telemetry on logged
steps.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` | local and remote SHA-256 `d8e092d2f9f57bd10cd7cc916eb38fab81a484d7e57dd80b7c0b00e5e3466d72` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | immutable launch source pushed before staging |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag-20261016T195506.721502047Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag` | local and remote base source match: 61 files, 459639 bytes, relative SHA-256-manifest digest `615741272057ee3b1ea73a282b1256b858fb925f673eaba24461f9ed65eb96e7`; no cache files; W&B's separate run directory excluded |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`; trainer verified 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack` | 28799175 bytes; SHA-256 `029372d70381524c4d9e15dc21504f4e06e1d9e178172b2f08f156c02477a634`; CPU-only Flax msgpack restore verified step 2000 |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/metrics.jsonl` | 381162 bytes; SHA-256 `3812cd92f6c0bf38ef8dfddacae91654354dfc33d0dfed6076d1bd1da105bb39`; byte-identical to remote |
| Outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/outputs.jsonl` | 2816457 bytes; SHA-256 `2ffb99b4416056bd49469d8236c959a5c9fb3574b32ea97963c57339b3f9f09c`; byte-identical to remote |
| Launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag/launcher.log.gz` | 3987 bytes; SHA-256 `9bd8df882c45a96aec82d8784e9025ccbee6799c37dfd540f23987f40d1a727c`; decompressed SHA-256 `1b789075c21808592ecf449832b1b7b9b12023ce5def69bd3a5ee215f7e45579` matches the 18668-byte remote log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/2tc50jk9` | authenticated API reports `finished`, step 2000, and matching intact/wrong-world exact match |

## Review and Verification

- Focused optimizer, default-compatibility, telemetry, and validation closure: 5 passed with two existing Pydantic warnings in 28.14 seconds.
- `USE_TORCH_XLA=0 JAX_PLATFORMS=cpu python -m pytest -q tests/test_jax_train.py tests/test_linked_facts_generator.py`: 70 passed with the same two existing warnings in 203.53 seconds.
- Correctness reviewed `32c72ce5e70b40b93c2c37048f4c84562a8027cb`; its matched-telemetry comparison finding was fixed in `75dd86d259ec97bc17ddc53d3368175769ed74af` by limiting update-ratio interpretation to the treatment and marking throughput instrumentation-confounded.
- Cleanup passed on `75dd86d259ec97bc17ddc53d3368175769ed74af` with no confirmed findings.
- Completed-result correctness passed at `12114a61848a00aebf01d437175e9d73eabab074`; completed-result cleanup re-review is pending this review-ledger correction.
- The mandatory filtered TPU audit completed with three Kristine v4-8 candidates and one Kristine v5e-8 candidate in its final `AVAILABLE` table. The selected prior Central2 v4-8 then passed isolated JAX 0.6.2 runtime, W&B authentication, region-local data, 85 GiB free-space, source/config hash, output freshness, exact process, lock, and VFIO gates.
- Terminal verification found `TRAINING_COMPLETE` at step 2000, no exact trainer, no lockfile or VFIO holder, TPU `READY` / `HEALTHY`, 201 metrics records, 200 output records, a restorable step-2000 checkpoint, and W&B `finished` at step 2000.

## Failure Details

- Last known step: `2000`; run completed successfully.
- Error summary: no training failure. One read-only progress SSH connection retried a transient packet-authentication error and then succeeded without affecting the attached training process.
- Detailed output: canonical launcher log linked above.

## Updates

- `2026-10-16`: Existing query/message-KV dynamics show comparable initialization scales but persistent sender credit attenuation and essentially flat sender weight norm. Selected a `4x` sender-private post-Adam update treatment with direct update-norm telemetry.
- `2026-10-16T19:55:06Z`: Created the immutable source snapshot, copied it once to node-local storage, and verified its file count, byte count, manifest digest, selected-config hash, and cache exclusion.
- `2026-10-16T19:57:55Z`: Launched on the selected Kristine v4-8 after state, runtime, data, disk, output, lock, and VFIO gates. W&B run ID is `2tc50jk9`.
- `2026-10-16T20:03:52Z`: Completed all 2000 steps, final causal evaluation, checkpoint save, and W&B synchronization. The checkpoint restored at step 2000; the process and lock were absent, devices were idle, and the TPU remained `READY` / `HEALTHY`.
