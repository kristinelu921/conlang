# Conlang JAX v5e-8 b128 gate 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b128-gate1` |
| Status | `completed` |
| Started | `2026-10-14T21:19:20Z` |
| Finished | `2026-10-14T21:21:46Z` |
| Baseline | not specified |
| Git revision | `cbefd6190fab0f9969c55c751e36e678c5bf39ee` on `codex/jax-tpu-training` |
| Command | `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 /home/kristinelu/elf-runs/runtime/venv/bin/python train.py --config configs/conlang-jax-v5e8-gate1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8; single host; READY/HEALTHY and AVAILABLE at 2026-10-14 preflight |

## Configuration

Source: `configs/conlang-jax-v5e8-gate1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b128-gate1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 128
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 200
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b128-gate1
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

No confirmed changes; no explicit experimental baseline was specified. The b128 configuration is a bounded throughput gate rather than a scientific comparison.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| validation | exact match | 0.03515625 | 200 | higher is better | `metrics.jsonl`, W&B `2iyz7bkx` |
| validation | mean joint log probability | -4.2622604370 | 200 | higher is better | `metrics.jsonl`, W&B `2iyz7bkx` |
| training | final loss | 4.4551658630 | 200 | lower is better | `metrics.jsonl`, W&B `2iyz7bkx` |
| training | median steady-state throughput | 5,193.3586 examples/s | 19 intervals after warmup | higher is better | `metrics.jsonl` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-gate1.yaml` | yes |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261014-conlang-jax-v5e8-b128-gate1-20261014T211431.116766090Z`; aggregate SHA-256 `45a0e2fa460c726db92fc77ee10031b9e48cfa6428f50ea1b7a685f91c21deb6` | yes |
| Regional dataset | `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` localized to `/home/kristinelu/elf-data/conlangs-v1`; 125,000 questions | yes |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/checkpoint.msgpack`; 22,656,217 bytes; SHA-256 `66d247cc5dee2390f76c8af7879d4625c2b362e2547798634a702c2d536ba271` | yes; deserialized at step 200 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/2iyz7bkx` | yes; `finished`, step 200 |
| Metrics | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1/metrics.jsonl`; 2,590 bytes | yes |
| Canonical log | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b128-gate1.launcher.log`; SHA-256 `63c43449f32fb153e56e872cb639d86cdc3b0a0461bd483ca677d7f226362624` | yes; completion sentinel at step 200 |

## Failure Details

- Last known step: 200, completed
- Error summary: one zero-step data-staging verification attempt used the nonexistent stale interpreter path `/home/kristinelu/elf-venv/bin/python`
- Detailed output: the regional copy completed into a temporary directory; verification was retried with `/home/kristinelu/elf-runs/runtime/venv/bin/python`, passed the complete manifest and checksum audit, and the directory was promoted atomically. No trainer, W&B run, output directory, or TPU client started.

## Observations

- The selected TPU was READY, HEALTHY, runtime-ready, and idle in the repository-required filtered inventory.
- `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` is in `US-WEST4` and contained the manifest plus all six compressed splits, totaling 11,643,131 bytes during preflight.
- CPU tests passed for static token shapes, pmapped parameter updates, bfloat16 compute with a float32 loss, relative-position attention, YAML field validation, and TPU path validation.

## Interpretation

- Static shapes, one-time host tokenization, and eight-way local-device data parallelism sustained a median 5,193.36 examples/s after compilation and warmup. This is the comparison baseline for one batch-256 optimization arm with the same model, precision, sequence lengths, seed, and hardware.

## Updates

- 2026-10-14T20:51:22Z: Created the planned report after selecting the same-region v5e-8 and preparing the native JAX throughput gate; launch remains blocked before staging.
- 2026-10-14T21:10:29Z: Pushed reviewed code/config commit `cbefd6190fab0f9969c55c751e36e678c5bf39ee` and opened PR 1; the run remains unlaunched pending fresh preflight and immutable staging.
- 2026-10-14T21:16:55Z: Preflight passed with 4 GiB free and a 1 GiB minimum. Copied the 27-file immutable source snapshot to node-local storage with matching size and aggregate SHA-256. Localized the complete US-WEST4 dataset and verified all 125,000 questions after correcting the zero-step verification interpreter path. The trainer remains unlaunched.
- 2026-10-14T21:21:46Z: Completed all 200 steps on eight TPU devices. W&B run `2iyz7bkx` finished at step 200; the final checkpoint deserialized successfully, the canonical log contained the exact completion sentinel, and the TPU remained READY/HEALTHY with no trainer, VFIO holder, or libtpu lock.
