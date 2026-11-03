# Conlang JAX v5e-8 batch-256 2,000-step run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-steps2000` |
| Status | `completed` |
| Started | `2026-10-15T01:43:45Z` |
| Finished | `2026-10-15T01:49:17Z` |
| Baseline | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| Git revision | reviewed config commit `2f299c5298f9cc7652b28a060446ac0b14128837`; local experiment merge `23f8e5d95a87e0344848cf02a29050f0c1cdb8f1` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a --project=he-vision-group --zone=us-west4-a --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-steps2000.yaml 2>&1 \| tee /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`, topology `2x4`, 8 JAX devices |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-steps2000.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-steps2000
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-steps2000
wandb_enabled: false
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
| `steps` | 1000 | 2000 | baseline and selected run YAMLs |

Run ID, output path, and W&B display name are independently fresh. All model, optimizer, batch, precision, data, evaluation, logging, telemetry, and RNG settings remain matched to the successful seed-1730 retry.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | immediate-baseline median steady-state throughput | 5468.273897 examples/s | 99 post-warmup intervals through step 1000 | higher is better | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| training | like-for-like sustained-throughput floor | 4374.619118 examples/s | post-warmup intervals | higher is better | 20% regression tolerance from the immediate baseline |
| training | prior full-run acceptance floor | 6463.955546 examples/s | post-warmup intervals | higher is better | retained context from `20261014-conlang-jax-v5e8-b256-full1` |
| training | median steady-state throughput | 5581.683917 examples/s | 199 post-warmup intervals through step 2000 | higher is better | archived `metrics.jsonl` |
| training | mean steady-state throughput | 5553.340547 examples/s | 199 post-warmup intervals through step 2000 | higher is better | archived `metrics.jsonl` |
| training | throughput range | 4308.311189–5802.448559 examples/s | 199 post-warmup intervals through step 2000 | higher is better | archived `metrics.jsonl` |
| training | first-half median throughput | 5616.261958 examples/s | 99 intervals through step 1000 | higher is better | archived `metrics.jsonl` |
| training | second-half median throughput | 5537.137114 examples/s | 100 intervals after step 1000 | higher is better | archived `metrics.jsonl` |
| training | intervals meeting like-for-like floor | 196 / 199 | steps 20–2000 | higher is better | archived `metrics.jsonl` |
| training | median relative to immediate baseline | +2.074% | steps 20–2000 | higher is better | derived from archived metrics and baseline |
| training | median relative to prior acceptance floor | -13.649% | steps 20–2000 | higher is better | derived from archived metrics and prior floor |
| training | loss | 3.954409122 | step 2000 | lower is better | archived `metrics.jsonl` |
| training | gradient norm | 0.840890586 | step 2000 | lower is not intrinsically better | archived `metrics.jsonl` |
| validation | exact match | 0.042968750 | step 2000 | higher is better | archived `metrics.jsonl` |
| validation | mean joint log probability | -4.054060936 | step 2000 | higher is better | archived `metrics.jsonl` |
| validation | best exact match | 0.054687500 | step 600 | higher is better | archived `metrics.jsonl` |
| validation | best mean joint log probability | -4.052084923 | step 1800 | higher is better | archived `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-steps2000.yaml` | yes; parsed and validated, SHA-256 `c3d1f6563c27c6e649be739441bb608a11e4709b77f98f9beb252a740c166409`, committed alone as `2f299c5` |
| Effective configuration | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-steps2000/config.yml` | yes; 815 bytes, SHA-256 `05c0a3b502abfbbfb210f6061bb9f889982318259ee8857323d0b609db2dbd2f` matched the remote artifact |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed config commit `2f299c5` pushed to the PR branch |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z`; remote `/home/kristinelu/elf-runs/source/20261015-conlang-jax-v5e8-b256-steps2000-20261015T013651.963374738Z` | yes; 32 files, 311663 bytes, sorted SHA-256-manifest digest `bca20aa8e3383acf7cbaeb3fc7ea6ab1326512450fee1a5cdd6e18e864e80f3a` matched locally and remotely |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, sourced from `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; bucket location `US-WEST4`; 7 files, 11643131 bytes, 15625 worlds and 125000 questions passed complete manifest verification |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-steps2000/checkpoint.msgpack` | yes; 22656218 bytes, SHA-256 `b5d79ec9fb33ec91a9ac34e630647dded791d12604a52c118fcca11f0947cbd4`, restored payload step `2000` |
| Tracking run | disabled because the prior W&B histogram path deterministically rejected vocabulary-sized telemetry | not applicable |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-steps2000/metrics.jsonl` | yes; 5560470 bytes, 220 rows, SHA-256 `c0f99d912ed459f2e3de9302debf9502be0aa110bf505bb99a0520f4399c3b53` matched the remote artifact |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-steps2000/outputs.jsonl` | yes; 28042836 bytes, 2000 rows, SHA-256 `19aa0726e0376c6d423a49dc5b5a8535c431e5509848dee0501f2f6e7db89f6d` matched the remote artifact |
| Canonical log | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v5e8-b256-steps2000/20261015-conlang-jax-v5e8-b256-steps2000.launcher.log.gz` | yes; 3340 bytes, deterministic gzip SHA-256 `41375a8854abb01db3e2aa3dc7901c946d366e1731072fa98bd9e3aa7ce0a42c`, decompressed SHA-256 `7f4b055492f4ae4199323937b067033016ad5ed05d091e7a66fbd5a4a90a4d58` matched the remote log |

## Failure Details

- Last known step: `2000`; `TRAINING_COMPLETE` emitted
- Error summary: not applicable; the training process exited successfully
- Detailed output: canonical log and node-local metrics are archived above

## Observations

- The clean reviewed config worktree passed 88 tests and 7 subtests; the exact staged source worktree passed 91 tests and 7 subtests in 68.43 seconds.
- The correctness and cleanup reviews reported no confirmed findings for base `e94f185` through reviewed head `2f299c5`.
- The selected TPU was `READY`, `HEALTHY`, Spot, runtime-ready, and `AVAILABLE`; preflight found 4 GiB free on both relevant filesystems, idle VFIO, no active trainer, a fresh output identity, and writable `/tmp/tpu_logs`.
- The immutable snapshot contains the same `train.py` bytes as the successful 1,000-step retry: SHA-256 `c6c5cd7351a017f60c9a584bc4d989b8c165a6367a7f6c499bca0b07919f6dac`.
- The first 1000 evaluation-output rows are byte-identical to the successful baseline output artifact: SHA-256 `7263b6dc580bd7a366888169e7c96a69834ad90e9b08479c5f869ce7fcd5a7be`.
- The run completed all 2000 training steps and 20 scheduled validation evaluations. Terminal all-worker inspection found no trainer, no VFIO or libtpu-lock holder, and the TPU remained `READY` and `HEALTHY`.
- Median throughput passed the like-for-like sustained gate. Three isolated intervals fell below that floor, but the full-run median was 2.074% above the immediate baseline. All 199 intervals remained below the older pre-telemetry acceptance floor.

## Interpretation

- This run tests the effect of doubling only the training horizon under the successful retry's exact scientific and telemetry settings.
- Doubling the horizon reduced final training loss from 4.103382111 to 3.954409122 and improved mean joint log probability from -4.065857887 to -4.054060936, while final exact match remained 0.042968750. This controlled comparison does not show an exact-match benefit at step 2000.
- The immediate-baseline throughput gate passed, but the older full-run floor remains unmet. The completed run therefore supports the longer-horizon scientific comparison without resolving the already documented telemetry-era performance regression.

## Updates

- 2026-10-15T01:40:00Z: Planned and validated the 2,000-step run. The fresh mandatory inventory selected the runtime-ready us-west4 Kristine v5e-8 from the final `AVAILABLE` table. The reviewed one-file config commit was pushed to PR 1, the immutable source was copied once and matched locally/remotely, and regional dataset integrity passed before launch.
- 2026-10-15T01:43:45Z: Launched the immutable snapshot in the foreground on the eight-device JAX TPU backend after the final READY/HEALTHY, output-freshness, VFIO, and lock checks passed.
- 2026-10-15T01:49:17Z: The run emitted `TRAINING_COMPLETE` at step 2000 and wrote the final checkpoint. Terminal all-worker inspection verified TPU health, trainer exit, idle VFIO and lock state, checkpoint payload step, and remote artifact hashes. Localized artifacts matched those hashes; the canonical log was archived as deterministic gzip.
