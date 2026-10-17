# Approximately 4x sender and receiver capacity retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1` |
| Status | `completed` |
| Started | `2026-10-17T20:11:37Z` |
| Finished | `2026-10-17T20:55:16Z` |
| Baseline | `20261017-conlang-v4-8-train1m-s10000-m4-qquery-senderlr4-sx192-sd4-sff768-avgres-eval200-retry1` |
| Supersedes | zero-step launch-control failure `20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200` |
| Git revision | `3d7597c29275bbc074cb03d26795e8bf9e9e8b13` (retry configuration/report closure) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml` |
| Command | recorded under Exact Launch Command below |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b`, `us-central2-b`, Spot `v4-8`; terminal API state READY/HEALTHY |

## Exact Launch Command

```bash
/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-17-12-29-15-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; trap "" HUP; source_dir=/home/kristinelu/elf-runs/sources/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1-20261017T200706.184345335Z; output_dir=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1; launcher_log=/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; mkdir -p /home/kristinelu/elf-runs/wandb; test -w /home/kristinelu/elf-runs/wandb; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache WANDB_DIR=/home/kristinelu/elf-runs/wandb PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml 2>&1 | tee "$launcher_log"' --quiet
```

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1.yaml`

```yaml
run_id: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1
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
output_path: /home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1
wandb_enabled: true
hidden_size: 256
output_hidden_size: 1024
sender_hidden_size: 384
sender_num_layers: 4
sender_num_heads: 12
sender_feedforward_size: 1536
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
num_layers: 2
num_heads: 8
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changed Hyperparameters

Scientific configuration is byte-for-byte identical to the failed zero-step
attempt. Only `run_id`, `output_path`, and `wandb_run_name` receive the required
fresh `retry1` identity. Relative to the completed baseline, receiver hidden,
feed-forward, output-hidden, and message widths scale `128/512/256/128` to
`256/1024/1024/256`; sender hidden/feed-forward widths scale `192/768` to
`384/1536`; attention heads scale receiver `4` to `8` and sender `6` to `12`.
Depths remain receiver 2 and sender 4. Initialized production counts remain
`7,298,176` sender, `6,986,639` receiver, and `14,845,711` total parameters.

## Hypothesis and Decision Rule

This retry preserves the original coupled-capacity hypothesis: approximately
four times as many sender and receiver parameters may let answer-token cross-
entropy discover passage-specific two-hop communication. Success requires a
repeatable intact-over-wrong-world validation advantage across the scheduled
200-step evaluations, preferably in exact match. Intact-versus-zero remains
only an out-of-distribution learned-carrier diagnostic. Because sender,
receiver, message, and output widths scale together, any gain belongs to the
coupled treatment rather than one isolated component.

## Results

The run completed all 10,000 steps, but it fails the preregistered communication
rule.

| Intervention at step 10,000 | Exact match | Exact count | Mean joint log probability |
|---|---:|---:|---:|
| Intact | `0.00000000` | 0/256 | `-7.5358849` |
| Wrong-world messages | `0.00000000` | 0/256 | `-7.5358505` |
| Zero messages | `0.00000000` | 0/256 | `-11.2020397` |
| Swapped senders | `0.00000000` | 0/256 | `-7.5371208` |
| Agent 1 only | `0.00000000` | 0/256 | `-7.4040928` |
| Agent 2 only | `0.00000000` | 0/256 | `-7.3717365` |

Intact never beats wrong-world in exact count across the 50 scheduled
evaluations: 49 are tied, and wrong-world leads by one answer at step 4,400.
The intact-minus-wrong-world likelihood gap is positive in 31/50 evaluations,
but its median is only `+0.0001070` nats, with range `-0.0005860` to
`+0.0013790`; the final gap is `-0.0000343` nats. This is small, sign-unstable,
and not accompanied by any exact-answer advantage. The final `+3.6661549`-nat
intact-versus-zero gap establishes dependence on a learned nonzero carrier,
not useful passage-specific content. The final logged training batch emits
`jaguar` for all 256 examples.

The largest adjacent five-record-versus-five-record loss decline after step
100 begins at step 1,020, from `5.3262455` before to `4.7486009` after. That is
40 steps later and larger than the matched baseline's step-980 decline from
`5.2955249` to `4.9101499`. It is not a communication transition: the matched
wrong-world control stays tied through the region and the final passage gap is
absent.

Over the last 50 logged steps, median path-gradient norms are `0.1445738` and
`0.1278845` for the two sender paths versus `0.8170325` for the receiver,
receiver ratios of `5.65x` and `6.39x`. The matched baseline ratios are `5.49x`
and `4.96x`, so scaling does not materially rebalance credit. Median sender
relative update is `0.00302706`, `12.75x` the receiver's `0.000237340`; message
RMS is `1.91x` the baseline. Sender weight norm grows `93.59%` from step 10 to
10,000, versus `42.88%` for the receiver. The maximum logged pre-clipping
gradient is `75.8312` at step 9,460, and 14 logged steps exceed 10; clipping
keeps training finite.

Median final-50 loss is `3.9918360`, effectively unchanged from baseline
`3.9889567`. Median throughput is `1,281.26` examples/second versus baseline
`2,380.07`, a `46.17%` reduction.

## Initialization and Step-1,020 Follow-up

The production tree is not muP-parameterized. Token embeddings, sender query
slots, and relative-position tables use an unscaled normal initializer with
standard deviation 1. Dense and attention kernels otherwise use Flax's
fan-in-scaled LeCun initializer, layer-normalization scales start at one, and
biases start at zero. The wider sender's `256 -> 384` input projection is a
rectangular identity, while its `384 -> 256` learned message projection starts
at exactly zero. Consequently, the initial message is exactly the four masked
regional averages of the token embeddings.

A focused gradient oracle confirmed the consequence of that zero projection:
within the sender subtree, only the message-projection kernel and bias receive
nonzero gradient on the first backward pass. After the first optimizer update
makes that projection nonzero, gradients reach the input projection, query
slots, attention, feed-forward, normalization, and relative-bias parameters.
This is a real one-update gate, but it cannot explain a transition at step
1,020.

The loss levels identify the later transition much more directly. Before the
drop, training loss sits near `5.3031` nats, the entropy of the dataset's
unconditional answer prior: four disjoint answer supports have probabilities
`0.4/0.2/0.2/0.2` and sizes `320/16/16/16`. Once the receiver recognizes the
answer domain from the question, the best message-independent loss is
`0.4 ln(320) + 0.6 ln(16) = 3.9709` nats. The run moves from the first level
toward the second and finishes at `3.9918`, while wrong-world messages remain
matched. The step-1,020 dip therefore marks acquisition of a question-to-answer-
domain prior circuit, not two-hop passage communication.

The norm audit is also localized rather than a uniform explosion. The shared
token-embedding norm falls from `748.9035` at initialization to `726.7633`
(`-2.96%`). Receiver norm growth is almost entirely the vocabulary readout:
the output-projection norm rises from `46.8129` to `154.0862` (`3.29x`), while
the remainder of the receiver moves only from `138.5418` to `141.2280`
(`+1.94%`). Sender growth is broader: the zero message projection reaches
norm `26.3267`, the identity input projection grows from `16.0000` to
`38.5524`, and several attention/feed-forward kernels reach roughly two to
four times their initialized RMS. This is consistent with the sender learning
a high-amplitude carrier while the receiver readout sharpens the domain prior;
it is not evidence that the carrier contains the correct passage-specific fact.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | primary configuration above | parses and validates; SHA-256 `fe801cd1caf23def3d97e4da3a0bea20a92e7265e11cb9bf26cccd0b2cab3298`; committed at `3d7597c29275bbc074cb03d26795e8bf9e9e8b13` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | retry closure and reviewed launch metadata pushed; completed evidence is included in this report/artifact update |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1-20261017T200706.184345335Z` locally; corresponding `/home/kristinelu/elf-runs/sources/` path in the command remotely | exact local/remote match: 67 files, 480,583 bytes, relative-manifest SHA-256 `2790ee0c5861bd0045f304bbace256bec8b1f8af85113bf9521940a7218f9b81`; no cache files |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | 95,042,272 bytes; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20`; 128,125 worlds / 1,025,000 questions / 8 questions per world; all gzip and content checks passed |
| Checkpoint | `/home/kristinelu/elf-runs/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/checkpoint.msgpack` | 178,168,842 bytes; SHA-256 `c32fc19fd21c20747de2c27ccfb8337edea2884d044d89139b1a0810def04a73`; CPU restore verified step 10,000 |
| Metrics | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/metrics.jsonl` | 1,995,482 bytes; SHA-256 `9953fd0d7c4132c8aafec45bbfb74329df877422cee01366239002068baef8a2`; 1,050 JSON records: 1,000 training and 50 evaluation |
| Outputs | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/outputs.jsonl` | 14,037,402 bytes; SHA-256 `39ead016ffb09fbf1d9b6ca4f798451404d726de6143b7f7838c1097b601da3d`; 1,000 JSON records through step 10,000 |
| Launcher log | `experiment_log/2026-10-17/artifacts/20261017-conlang-v4-8-train1m-s10000-m4-qquery-lr4-sx384-sd4-sff1536-rx256-rd2-rff1024-oh1024-avgres-e200-retry1/launcher.log.gz` | 15,033 bytes; SHA-256 `47ad750255f38f0fa594a4ba72cb90df20d7dd10962035dea3edfe947899d19b`; decompressed SHA-256 `e26cb08d589c39873b44f4c18f9cea4c8bef53096f02d60b4b95fb79ee6d9174` matches the 85,540-byte remote canonical log |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/iv9wan21` | API state `finished`, summary step 10,000, exact match 0, five files synced |

## Failure Details

- Last known step: 10,000, completed.
- Error summary: not applicable.
- Detailed output: canonical launcher log recorded in Artifacts.

## Observations

- The predecessor failed before W&B and JAX initialization because its wrapper
  created the exact scientific output directory before the trainer's freshness
  guard. No model or dataset behavior was exercised.
- This retry directed W&B's base directory to the separate sibling
  `/home/kristinelu/elf-runs/wandb` path. The exact wrapper creates and checks
  only that sibling after output freshness checks; it never creates the retry
  output directory before `train.py`.
- The first snapshot-copy invocation was rejected locally before transfer
  because TPU `scp` accepts `--scp-flag`, not `--ssh-flag`. No remote path was
  created. The corrected invocation used `--scp-flag=-F`,
  `--scp-flag=/dev/null`, and `--scp-flag=-O`; the exact-copy verification above
  then passed. This was transport-command validation, not a run attempt.
- The first terminal checkpoint-inspection one-liner had an f-string quoting
  error after the completion sentinel, file sizes, and hashes had already
  verified. The corrected read-only inspection restored the checkpoint at step
  10,000 and parsed every metric/output record; no run state changed.
- Parallel artifact transfer encountered transient SCP packet/MAC errors and
  automatically retried. Every local artifact's final size and SHA-256 matches
  its remote source, and the compressed log's decompressed hash matches too.
- Terminal gates found no trainer, libtpu lock, or VFIO holder. The TPU remains
  READY/HEALTHY and the immutable remote source still matches 67 files,
  480,583 bytes, and manifest digest `2790ee0c...9b81` with no cache files.

## Interpretation

- Approximately fourfold sender/receiver parameter scaling is not sufficient
  to make this objective learn useful passage-specific communication. It makes
  sender weights and messages much larger and preserves substantial sender
  path gradients, but intact messages never outperform wrong-world messages in
  exact count and the likelihood gap stays negligible.
- The step-1,020 loss transition is a real optimization phase change, not
  grokking of the desired circuit: validation correctness does not jump, the
  wrong-world control remains matched, and the model ends in a one-answer
  `jaguar` collapse. The evidence points to a stronger learned carrier/answer-
  prior circuit rather than successful two-hop information transport.
- This is one deterministic run and scales sender, receiver, message vector,
  and output widths together. It rejects this coupled scaling treatment as a
  sufficient fix; it does not prove that all larger or differently regularized
  architectures must fail.

## Review and Verification

- Correctness review of
  `bc1d16258fbbcb7d6171c57c18c19db19ce338d0..3d7597c29275bbc074cb03d26795e8bf9e9e8b13`
  verified the identity-only YAML delta, config validation, predecessor startup
  ordering and remote zero-step evidence, and found one launch-control issue:
  the proposed sibling W&B base did not yet exist and no exact command was
  recorded. The exact command above resolves that finding by creating only the
  sibling base after freshness checks. Cleanup review then found only a
  noncanonical predecessor status/stale interpretation; both were corrected
  before launch. The completed-evidence delta contains only report/artifact
  files, so no additional code-bearing review pass applies.

## Updates

- `2026-10-17T20:00:48Z`: Created the fresh retry plan after the predecessor's
  zero-step launch-control failure; scientific settings are unchanged.
- `2026-10-17T20:07:06Z`: Committed and pushed the retry closure, created the
  immutable local snapshot, and corrected the wrapper to create only the
  sibling W&B base after retry-output freshness checks.
- `2026-10-17T20:09:28Z`: Copied the reviewed snapshot node-locally after one
  rejected local `scp` flag invocation, then verified exact file count, byte
  count, manifest digest, config digest, and absence of cache files remotely.
- `2026-10-17T20:11:37Z`: Launched on the selected v4-8. W&B initialized as
  `iv9wan21`; the trainer verified 1,025,000 questions, 1,000,000 training
  examples, 256 validation examples, and all four TPU devices.
- `2026-10-17T20:21:30Z`: First-step gate passed. Training advanced through
  step 140 at approximately 1,275 examples/second with finite loss and
  gradients. Foreground launch and recurring monitoring remain active.
- `2026-10-17T20:55:16Z`: Completed step 10,000, final causal evaluation,
  checkpoint save, completion sentinel, and five-file W&B sync. Independent
  terminal checks verified all artifacts, checkpoint step, W&B `finished`
  state, idle process/lock/VFIO state, READY/HEALTHY TPU state, and immutable
  source digest.
