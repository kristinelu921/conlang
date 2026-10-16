# Ten-thousand-step, one-million-example sender-balancing diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Status | `running` |
| Started | `2026-10-16T20:47:46Z` |
| Finished | `not started` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-senderlr4-commdiag` |
| Git revision | primary configuration/report commit `914c0570f242ff75f592d3e030ae05dcbbcdb886`; reviewed immutable launch source `773f1d90789dfde666b8b5abff224a7d9121148e` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; source_dir=/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag; output_dir=/home/kristinelu/elf-runs/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag; launcher_log=/home/kristinelu/elf-runs/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag.launcher.log; test -d "$source_dir"; test ! -e "$output_dir"; test ! -e "$launcher_log"; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; cd "$source_dir"; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml 2>&1 | tee "$launcher_log"' --quiet` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host and four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1-train1m
dataset_prefix: kristine/datasets/conlangs-train1m
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 4.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 10000
eval_every: 10000
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag
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

## Changed Hyperparameters and Data

| Parameter | Baseline | Current | Rationale |
|---|---:|---:|---|
| Training steps | 2,000 | 10,000 | Observe whether the middle loss transition develops into useful communication over a longer horizon |
| Training questions | 100,000 | 1,000,000 | Increase training diversity tenfold so 2.56 million presentations span about 2.56 nominal passes rather than 25.6 |
| Final evaluation step | 2,000 | 10,000 | Preserve one matched final 256-example causal evaluation |
| Node-local dataset | `conlangs-v1` | `conlangs-v1-train1m` | Keep the expanded deterministic dataset distinct and region-local |

All other model, optimizer, seed, batch, precision, logging, intervention,
validation, and test settings remain matched. The new deterministic dataset
uses seed `1729`, eight entities/questions per world, `125,000` training
worlds, `625` validation worlds, and `625` worlds in each test split. Because
split/world seeds are independently derived, the existing 5,000 validation
questions and all four 5,000-question test splits remain unchanged; only the
training split grows from 100,000 to 1,000,000 questions.

## Hypothesis and Decision Rule

The 2,000-step treatment showed a sharp non-initial loss transition near step
830, approximately 150 steps earlier than its matched 1x-sender-update
baseline. It also increased sender-relative parameter motion, but at step 2,000
intact and wrong-world messages still tied at `4/256`, and wrong-world messages
were slightly better in likelihood. This run tests whether that transition is
only an early answer-prior phase or becomes passage-specific communication with
more time and substantially more unique worlds.

Treat the longer run as a communication improvement only if intact messages
beat both zero and wrong-world messages in exact match and mean joint log
probability at step 10,000. Use the logged loss, gradient, update, message-RMS,
and weight-norm trajectories to locate any later phase changes. A lower loss,
larger sender motion, or an intact-versus-zero gap without an
intact-versus-wrong-world advantage is not evidence of useful private-content
transport. Compare the final result to the exact 2,000-step treatment while
noting that steps and training-set size change together, so this arm does not
separately identify which change caused any difference.

## Results

Pending launch and verified completion.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag.yaml` | local snapshot and remote SHA-256 `e579b0178ce06277d4da14792e5c4831b626fd3bdddf96118d01940a8954d4f4` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | reviewed closure pushed before staging |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag-20261016T204214.567755894Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag` | local and remote match: 62 files, 460936 bytes, relative SHA-256-manifest digest `b966f6ca228ae44fdf3f719b86d8b387a9b8f9c6d0f69285329e17743eac76f2`; no cache files before launch |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1-train1m` | full verification passed: 128,125 worlds / 1,025,000 questions / 8 questions per world; 95,042,272 file bytes; manifest SHA-256 `1a860a5a0ace2d249c0b7e7273925d9545f61aaa261e00f5b9fe75900da91b20` |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-train1m-s10000-m4-d2-qquery-msgkv-senderlr4-commdiag/checkpoint.msgpack` | pending |
| Metrics | pending | pending |
| Outputs | pending | pending |
| Launcher log | pending | pending |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/0rka2x7a` | authenticated, running, and reporting training progress |

## Review and Verification

- The mandatory TPU inventory completed before selection and reported three
  Kristine v4-8 resources in `us-central2-b` in its final `AVAILABLE` table.
- The selected prior-run v4-8 currently reports 400 GiB host RAM and 85 GiB
  free local disk. The current compressed training shard is 9,267,279 bytes;
  a tenfold deterministic expansion and approximately 4.6 GiB of fixed int32
  token arrays fit with substantial host-memory and disk headroom.
- The matching `us-central2` bucket path
  `gs://kmh-gcp-us-central2/kristine/datasets/conlangs-train1m` currently has no
  objects. The dataset will therefore be generated and fully verified on the
  selected Central2 TPU without any cross-region transfer.
- Focused configuration validation and dataset tests passed 12 cases.
  Correctness review passed the closure at
  `f76311fd9654de4588ff06fe74741dab6929800e`. Cleanup review found one stale
  pending-metadata issue; it was corrected in
  `a8e4115ded7db481915bc721368cc3cd0df53d5b` and the coordinator verified the
  exact report delta.
- The immutable node-local source matched the NFS snapshot, the complete new
  dataset passed its streaming integrity verifier, and the unchanged
  validation/test split hashes match the established dataset. Remote preflight
  found 85 GiB free, writable TPU logs, authenticated W&B, fresh output and
  launcher-log paths, and no trainer, libtpu lock, or VFIO holder.
- The isolated runtime reported JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, W&B
  `0.28.1`, and four local TPU devices. The launched trainer confirmed the
  expected node-local dataset, 1,000,000 train examples, 256 validation
  examples, and four TPU devices. Step 190 completed at approximately 5,700
  examples per second before this status update.

## Failure Details

- Last known step: `190`; training remains active.
- Error summary: none.
- Detailed output: pending.

## Updates

- `2026-10-16`: Planned a matched 10,000-step continuation from a fresh
  initialization with 1,000,000 unique training questions. Validation and test
  splits remain exactly matched; the treatment changes training horizon and
  training-set size together.
- `2026-10-16T20:42:14Z`: Created the immutable source snapshot, copied it once
  to the selected TPU's node-local storage, and verified the complete source
  manifest and selected-config hash.
- `2026-10-16T20:46Z`: Generated and fully verified the deterministic
  one-million-question train split on the selected Central2 host. No dataset
  crossed regions; validation and test hashes match the original dataset.
- `2026-10-16T20:47:46Z`: Started W&B run `0rka2x7a`. The TPU trainer completed
  its first logged step group and reached step 190 at normal matched throughput.
