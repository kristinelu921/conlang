# Conlang JAX v5e-8 batch-256 2,000-step W&B retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1` |
| Status | `failed` |
| Started | not recorded; zero-step foreground attempt immediately preceded the terminal audit |
| Finished | `2026-10-15T02:39:50Z` |
| Baseline | `20261015-conlang-jax-v5e8-b256-steps2000`; telemetry retry of failed W&B run `20261015-conlang-jax-v5e8-b256-full2` |
| Git revision | reviewed config commit `62bb682fa63ad90dadfb750a4f7996d3b11fe5fc`; immutable launch tree `d3526108544329f228612679b62fa1e843ef0b37` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a --project=he-vision-group --zone=us-west4-a --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1-20261015T023534.813287384Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-steps2000-wandb-retry1.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, topology `2x4`, expected 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry1.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 2000
eval_every: 100
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1
wandb_enabled: true
hidden_size: 128
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

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| `wandb_enabled` | `false` | `true` | baseline and selected run YAMLs |

Run ID, output path, and W&B display name are independently fresh. Model, optimizer, batch, precision, data, evaluation, logging cadence, training horizon, and RNG seed are unchanged. The telemetry implementation now collects diagnostics at `log_every: 10`, omits vocabulary-sized token histograms, and resets throughput timing after evaluation and W&B logging.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | baseline median steady-state throughput | 5581.683917 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `20261015-conlang-jax-v5e8-b256-steps2000` |
| training | sustained-throughput floor | 6463.955546 examples/s | post-warmup intervals | higher is better | retained 80% floor from pre-telemetry full run |
| training | loss | not recorded | not recorded | lower is better | pending |
| validation | exact match | not recorded | not recorded | higher is better | pending |
| validation | mean joint log probability | not recorded | not recorded | higher is better | pending |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-steps2000-wandb-retry1.yaml` | yes; parsed and validated, SHA-256 `a67dd4d3e428219343955715295b6ad615a35600927e6b5b87cfd29f20935a7a`, committed alone as `62bb682` |
| Effective configuration | none | no; the interpreter never started |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; config commit pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1-20261015T023534.813287384Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 33 files, 315774 bytes, relative SHA-256-manifest digest `6e6a78e9727d68b15c0110630227a8aae342e2e895b047170b2f58a77ab531ad` matched locally and remotely |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, sourced from `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; bucket location `US-WEST4`; 7 files, 11643131 bytes, 15625 worlds and 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1/checkpoint.msgpack` | no; the output directory is absent |
| Tracking run | none | no; W&B initialization was never reached |
| Metrics | none | no; the output directory is absent |
| Evaluation outputs | none | no; the output directory is absent |
| Canonical log | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000-wandb-retry1.launcher.log` | no; the command stopped before `tee` started |

## Failure Details

- Last known step: `0`; step 1 did not start
- Error summary: the foreground wrapper exited `1` before Python because its inline `pgrep -f '[p]ython.*train.py'` matched the containing launch shell, whose own command line included the later trainer argv
- Detailed output: foreground control-host output contained only the successful `AVAILABLE` final occupancy result and SSH connection messages; no remote launcher log was created

## Observations

- The mandatory filtered inventory selected the runtime-ready us-west4 Kristine v5e-8 from its final `AVAILABLE` table. Exact describe returned `READY` and `HEALTHY`; occupancy returned `AVAILABLE`.
- Preflight found 4 GiB free on the data and output filesystems, an absent fresh output path, and idle VFIO.
- The node-local runtime verified JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, Muon availability, W&B `0.28.1`, and authenticated W&B access without exposing credentials.
- Baseline and post-change CPU tests passed: 17 tests before the fix; focused regression coverage passed after each fix; the final suite passed 93 tests and 7 subtests in 74.19 seconds.
- Correctness review at `cdd4470` identified the unconditional diagnostics overhead and W&B timer contamination; both were fixed and focused tests passed. Cleanup review at `7ce593e` identified the dead token-histogram branch and stale cadence documentation; both were fixed and focused tests passed. The separate user-owned edit to the historical gate YAML remains outside this retry; the selected run pins `message_length: 4` explicitly.
- The terminal audit found the run output and launcher log absent, no Python process, no libtpu lock, no VFIO holder, and writable `/tmp/tpu_logs`. A local reproduction proved that the same full-command `pgrep` pattern matches a containing shell when its later argv text includes `python ... train.py`. All other final predicates remained satisfied.

## Interpretation

- This attempt produced no scientific or telemetry result. The failure was in the redundant inline process probe, after the canonical occupancy helper had already returned `AVAILABLE` and before the interpreter or W&B started.
- Recovery requires a fresh run/output/W&B identity. The retry will retain the same immutable scientific source and replace the self-matching inline probe with the canonical occupancy helper plus direct lock/output checks.

## Updates

- 2026-10-15T02:35:34Z: Planned, reviewed, tested, and staged the fresh W&B retry. The exact config commit was pushed to PR 1; immutable local and node-local source hashes, the regional dataset, runtime versions, and W&B authentication all passed prelaunch verification.
- 2026-10-15T02:39:50Z: The final canonical occupancy helper returned `AVAILABLE`, but the foreground wrapper exited `1` before Python because its inline full-command `pgrep` matched its own containing launch shell. The terminal all-worker audit verified a zero-step attempt with no output, log, W&B run, checkpoint, process, VFIO holder, or libtpu lock.
