# Full-data bridge-supervised sender diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag` |
| Status | `completed` |
| Started | `2026-10-16T11:38:50Z` |
| Finished | `2026-10-16T11:45:05Z` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Git revision | immutable launch snapshot HEAD `ce27b32f14ccbeb3c37e830d3902c050e4dc5550`; corrected reviewed implementation/configuration closure `c83e7a92994b118f9fd1d7fe4e2d55f894470a32` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command='set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null \|\| true)"; cd /home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES= ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.launcher.log'` |
| Seed | `1730` |
| Hardware | exact same authorized TPU: `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml`

```yaml
run_id: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag
wandb_enabled: true
hidden_size: 128
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
| Bridge supervision weight | `0.0` | `1.0` | Give the person-side sender direct credit for selecting the question-specific middle entity while leaving all model inputs unchanged |

All data, seed, optimization, schedule, widths, depth, message length, one-round topology, sharing, gates, sender question conditioning, receiver question masking, hardware, and causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

The receiver-question bypass and matched wrong-question controls show that the senders can relay question identity, but target-question/wrong-passages controls show no beneficial use of private facts. Doubling the message length and adding one synchronous exchange were both insufficient. The leading remaining hypothesis is that the end-to-end answer objective gives too little direct credit for selecting the question-specific middle entity from the person-side passage.

For each training example only, the generator's existing reasoning path supplies the middle entity as an auxiliary label, and its sender assignment identifies which sender holds the person-to-middle fact. A shared vocabulary head reads the mean of that sender's final four message vectors. The label and sender assignment are supervision only; neither is passed into the sender or receiver. The total objective is answer loss plus `1.0 * bridge loss`. A zero weight omits the auxiliary head, preserving existing checkpoint trees.

The bridge hypothesis is supported only if intact bridge exact match learns above its controls, especially the matched wrong-question re-encoding. The communication fix is useful only if intact answer messages then beat target-question/wrong-passages messages in exact match and preferably mean joint log probability. If bridge accuracy learns but the answer remains passage-flat, the bottleneck moves downstream to the answer-side lookup or receiver join. If bridge accuracy itself does not learn, revise the auxiliary readout/objective before stacking width, depth, tying, gates, or a second exchange.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Validation | Intact-message answer exact match | `0.06640625` (17/256) | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message mean joint log probability | `-4.0323076248` | 2000 | max | `metrics.jsonl` |
| Validation | Intact-message bridge exact match | `0.125` (32/256) | 2000 | max | `metrics.jsonl` |
| Validation | Wrong-question answer exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question mean joint log probability | `-10.1678810120` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Wrong-question bridge exact match | `0.0234375` (6/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages answer exact match | `0.046875` (12/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages mean joint log probability | `-4.0753774643` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Target-question/wrong-passages bridge exact match | `0.0234375` (6/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message answer exact match | `0.0078125` (2/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message mean joint log probability | `-9.0433568954` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Zero-message bridge exact match | `0.015625` (4/256) | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Swapped-sender answer / bridge exact match | `0.06640625` / `0.125` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-1-only answer / bridge exact match | `0.046875` / `0.06640625` | 2000 | diagnostic | `metrics.jsonl` |
| Validation | Agent-2-only answer / bridge exact match | `0.046875` / `0.07421875` | 2000 | diagnostic | `metrics.jsonl` |
| Train | Total / answer / bridge loss | `7.6994633675` / `3.8913917542` / `3.8080716133` | 2000 | min | `metrics.jsonl` |
| Train | Bridge minibatch accuracy | `0.1015625` (26/256) | 2000 | max | `metrics.jsonl` |
| Train | Throughput | `4439.2029 examples/s` | 2000 | max | `metrics.jsonl` |
| Train | Agent-1 / agent-2 / receiver gradient norm | `0.6994043` / `0.6238572` / `1.2041546` | 2000 | diagnostic | `metrics.jsonl` |

## Interpretation

- The preregistered communication criterion passes for the first time. Intact messages beat target-question/wrong-passages messages by 5 exact answers (`17/256` versus `12/256`) and `0.0430698` mean joint-log-probability nats. Because this matched control retains the target question and changes only both private passages before sender encoding, the gap is evidence of beneficial private-passage use under this evaluation.
- The auxiliary target learned useful question-specific selection. Intact bridge accuracy is `32/256`, while both wrong-question and wrong-passage controls reach only `6/256`, equal to the empirical constant-bridge baseline on this validation prefix. The wrong-question bridge collapse also shows that the head is not merely recovering a passage-global bridge prior.
- Question transport remains strong: wrong-question re-encoding drops answer exact match from `17/256` to `2/256` and loses `6.1355734` nats with both passages and the receiver target fixed. Zero messages also reach `2/256` and lose `5.0110493` nats, but the matched wrong-question control—not the out-of-distribution zero tensor—isolates question identity.
- Exact answer accuracy nearly doubles relative to the matched one-round bypass (`17/256` versus `9/256`). This is a single-seed result on one 256-example prefix, so it supports bridge-selection credit as a material blocker in this setup rather than establishing it as the unique cause.
- Sender credit assignment changes qualitatively. The last-50 logged median agent gradients are `0.6813031` and `0.7267302`, versus `1.2801915` for the receiver: only `1.88x` and `1.76x` smaller, compared with roughly `51x` and `62x` in the matched bypass. The auxiliary loss therefore supplies the missing sender-side gradient signal it was designed to test.
- Each single-message arm reaches `12/256`, while both intact messages reach `17/256`; this is consistent with useful joint contribution, though aggregate counts do not identify whether the same examples are correct. Swapping sender channels preserves both answer and bridge accuracy, as expected for the shared bridge head and receiver that can attend to both channels.
- Final throughput is only `0.254%` below the matched bypass (`4439.20` versus `4450.50 examples/s`), so the shared bridge projection has negligible steady-state cost. Keep bridge supervision as the first causally communicating arm and test hidden width 256 with a matched tied-versus-untied comparison next.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.yaml` | parses and validates; correctness and cleanup reviews passed |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | corrected reviewed closure and snapshot provenance pushed to `codex/jax-tpu-training` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag-20261016T113307.495856067Z`; remote `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag` | local and remote matched before launch: 57 files, 441219 bytes, relative SHA-256-manifest digest `4970c19a2ddfa884af78653bfdf3b0e2727bd9071f4b104d024038e4beca4b36`; selected-config SHA-256 `007774172e2348652f10d49e295b69ade13d0226df103400c535f36827756a2a`; no bytecode or test caches |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | revalidated on the TPU: 15,625 worlds / 125,000 questions / 8 questions per world; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/uidxid93` | authenticated API reports `finished`, step `2000`, answer exact `0.06640625`, and bridge exact `0.125`; canonical launcher synced five files |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/checkpoint.msgpack` | 29,810,057 bytes; SHA-256 `e8ade42ed61a45695a17c198c62d73f2b8daf6bcab26cf56155eb81f4e53831c`; CPU-only Flax msgpack restore verified step 2000 and bridge kernel shape `(128, 2191)` |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/metrics.jsonl` | 365,850 bytes; SHA-256 `2a169f55b062858506c639bd404a099bb2b9a6df9fdddd61388114400f5b9fc5`; 201 JSON records parsed |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/outputs.jsonl` | 2,809,574 bytes; SHA-256 `ac6d6e73117d500ba0ce5f732c0e8c936a69efcd45c40e70d7690153026adbcc`; 200 strictly increasing logged steps and 256 rows at step 2000 |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag/20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-bridgeaux-commdiag.launcher.log.gz` | 4,056 bytes; compressed SHA-256 `9f7383d315af6584a24c3577ffb85211333eb629e0ce9c45d8b7cacd0918f43b`; uncompressed remote SHA-256 `fd1717837fd6d360b1953a49878c3c0af73bb73e58dc2009ba7915cb2ef632ad`; gzip and completion sentinel verified |

## Review and Verification

- The initial focused CPU closure passed 15 tests covering dataset labels, static tokenization, configuration validation, person-side sender selection, auxiliary-head updates, zero-weight parameter-tree compatibility, and bridge metrics under all causal interventions.
- After correcting swapped-channel bridge attribution, the full CPU regression closure passed 65 tests with two existing Pydantic warnings in 180.32 seconds. The discriminating selector/evaluation pair also passed independently in 30.06 seconds.
- The cleanup-only tokenizer-fixture consolidation then passed all 56 JAX trainer tests with the same two existing warnings in 179.78 seconds; the nine unchanged generator tests had passed in the corrected full closure.
- Correctness and cleanup reviews passed for exact range `5a5ea7110b73791077f984afd0311b8d8a566ef5..f3c402a1606063cd3d5652689dfb55f511d7ca53` after the diagnostic correction and test-fixture consolidation.
- Remote source, JAX `0.6.2` runtime, authenticated W&B file presence, complete regional dataset, 85 GiB free space, fresh output path, process absence, lock absence, VFIO idleness, and TPU health checks passed.
- Completed evidence verification parsed all 201 metrics and 200 output records, recomputed the causal gaps and last-50 medians, matched every local artifact hash to its remote source, restored the final checkpoint on CPU, and confirmed W&B `finished` at step 2000.
- Terminal verification found the exact trainer absent, no stale TPU lock, idle VFIO devices, a canonical `TRAINING_COMPLETE ... step=2000` sentinel, and the TPU still `READY` / `HEALTHY`.

## Failure Details

- Last known step: `2000`.
- Training and all answer/bridge causal evaluations completed successfully. W&B finalized, the checkpoint restored, the exact process exited, the lock disappeared, VFIO became idle, and the TPU remains `READY` / `HEALTHY`.

## Updates

- `2026-10-16`: Selected explicit bridge-entity supervision after the message-8 and two-round sender arms preserved question relay but showed no beneficial private-passage gap.
- `2026-10-16`: Implemented a training-only middle-entity target on the person-side sender message, preserved zero-weight checkpoint compatibility, and passed the focused and full CPU test closures.
- `2026-10-16`: Correctness review found and verified a diagnostic-only fix so swapped bridge accuracy follows the semantic sender rather than its original channel index; the corrected full suite passed.
- `2026-10-16`: Cleanup consolidated the compact-example tokenizer into one test helper and its complete 56-test JAX file passed; final cleanup re-review passed.
- `2026-10-16T11:33:07Z`: After the exact Kristine v4-8 appeared in the mandatory final `AVAILABLE` table, created immutable launch snapshot `ce27b32f14ccbeb3c37e830d3902c050e4dc5550` and verified its 57-file local manifest, selected-config hash, and cache exclusions.
- `2026-10-16T11:36:47Z`: Repaired an incomplete first source transfer after verification found that the shell glob omitted the snapshot's two top-level dotfiles; no training had started. The complete 57-file remote tree then matched the local byte count and manifest digest exactly.
- `2026-10-16T11:36:47Z`: Verified the node-local config hash, JAX `0.6.2`, the complete region-local dataset, 85 GiB free space, a fresh output path, W&B authentication-file presence, and idle TPU devices. The selected config parses with bridge weight `1.0` and one communication round.
- `2026-10-16T11:38:50Z`: Launched after the exact same Kristine v4-8 appeared in the final mandatory `AVAILABLE` table and the launch-boundary output-path, lock, and VFIO checks passed; W&B run `uidxid93` authenticated and synced.
- `2026-10-16`: Passed the launch gate with finite losses, bridge accuracy above its empirical constant baseline, and approximately 4,400 examples/s. At step 810, bridge accuracy was `12/256` and sender gradients had risen to `0.4565` / `0.5004`.
- `2026-10-16T11:45:05Z`: Completed step 2000, every answer and bridge causal arm, checkpoint save, and W&B finalization.
- `2026-10-16`: Verified exact process absence, lock and VFIO idleness, 201 metrics records, 200 output records, artifact hashes, CPU checkpoint restore, W&B `finished`, and TPU `READY` / `HEALTHY` state.
