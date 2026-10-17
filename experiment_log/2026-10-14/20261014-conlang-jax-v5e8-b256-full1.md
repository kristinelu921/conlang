# Conlang JAX v5e-8 batch-256 full run 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b256-full1` |
| Status | `completed` |
| Started | `2026-10-14T21:59:43Z` |
| Finished | `2026-10-14T22:02:20Z` |
| Baseline | `20261014-conlang-jax-v5e8-b256-opt1` |
| Git revision | `bde42367a09836ee775ced42e1a37d6fe085649f` on `codex/jax-tpu-training`; run config introduced by `b337b68be31f36a314e0dae5720a72813c598ddd` |
| Command | `cd /home/kristinelu/elf-runs/source/20261014-conlang-jax-v5e8-b256-full1-20261014T215704.497568387Z && JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=$PWD HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1 /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v5e8-b256-full1.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1.launcher.log` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8, topology 2x4; single host; Spot; `READY`/`HEALTHY`; initial occupancy `AVAILABLE` |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-full1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b256-full1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 1000
eval_every: 100
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b256-full1
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
| `steps` | 200 | 1000 | baseline and full-run YAMLs |
| `eval_every` | 200 | 100 | baseline and full-run YAMLs |

Run identity, output path, and W&B display name are fresh. Model shape, optimizer, batch size, precision, dataset, seed, and topology remain identical to the accepted optimization gate.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | accepted promotion-gate median throughput | 7900.318449 examples/s | 19 post-warmup intervals through step 200 | higher is better | `20261014-conlang-jax-v5e8-b256-opt1` |
| training | sustained-throughput floor | 6320.254759 examples/s | post-warmup full-run intervals | higher is better | 20% regression tolerance from the accepted gate |
| training | median steady-state throughput | 8079.944432 examples/s | 99 post-warmup intervals through step 1000 | higher is better | preserved `metrics.jsonl` |
| training | throughput change versus accepted gate | +179.625983 examples/s (+2.273655%) | matched model/batch/topology | higher is better | both run metrics |
| validation | exact match | 0.03125 | step 1000 on 256 examples | higher is better | preserved `metrics.jsonl`, W&B `exf0hkel` |
| validation | mean joint log probability | -4.050580978393555 | step 1000 on 256 examples | higher is better | preserved `metrics.jsonl`, W&B `exf0hkel` |
| training | final loss | 3.9922327995300293 | step 1000 | lower is better | preserved `metrics.jsonl`, W&B `exf0hkel` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-full1.yaml` | yes; parsed and validated locally |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed head `d0da78079cbcd1301f9867b1b739c5760e12c4c6` pushed |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261014-conlang-jax-v5e8-b256-full1-20261014T215704.497568387Z`; node-local copy under `/home/kristinelu/elf-runs/source/`; aggregate SHA-256 `bf07f75ad01b3b11c417cf0b001714b7e1c52567e16a88d4ec0a16791b11da23` | yes; 29 files, 293,002 file bytes, exact local/remote match |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`; source `gs://kmh-gcp-us-west4/kristine/datasets/conlangs` | yes; US-WEST4, 7 canonical objects, 11,643,131 file bytes, 125,000 questions |
| Effective configuration | `experiment_log/2026-10-14/artifacts/20261014-conlang-jax-v5e8-b256-full1/effective-config.txt`; SHA-256 `3a231d43aadba1f0a1a62bada352b36029e3736d5efb9e1821bb489d2f355e92` | yes; 31 keys and semantically equal to the selected source YAML |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-full1/checkpoint.msgpack`; 22,656,218 bytes; SHA-256 `a68eef517b06f401fdb8fed49baeb013087d82f7495a5eb53ffbf090f747b7ba` | yes; deserialized payload step 1000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/exf0hkel` | yes; API state `finished`, step 1000 |
| Metrics | `experiment_log/2026-10-14/artifacts/20261014-conlang-jax-v5e8-b256-full1/metrics.jsonl`; 110 rows, 13,659 bytes; SHA-256 `851eda557b3149746b258143dbc1020f0a8df8cd9e772e880ff83afd6a87aeaa` | yes; exact remote/local checksum match |
| Canonical log | `experiment_log/2026-10-14/artifacts/20261014-conlang-jax-v5e8-b256-full1/launcher.log.gz`; 2,556 compressed bytes; gzip SHA-256 `60c8038a323fe0bc83f47ab2ce50de1a4fcbc41566690a27c7bdd925a188aa53` | yes; decompressed 144 lines and 10,927 bytes with SHA-256 `cbdf16eef5dc68cac4e69f3b502f26a5b0f0d3566c6cf5d03b6f74a077ef96f6`, exactly matching the remote log and completion sentinel |

## Failure Details

- Last known step: 1000, completed
- Error summary: none
- Detailed output: preserved canonical log listed above

## Observations

- The mandatory filtered inventory selected the available, runtime-ready US-WEST4 v5e-8 that completed the accepted batch-256 gate.
- The accepted gate used the same explicit `message_length: 4`, model dimensions, precision, batch size, seed, regional data, and TPU topology planned here.
- The staged snapshot captured the current worktree, including pre-existing uncommitted default/documentation edits for a separate one-message experiment. The selected immutable YAML explicitly pins `message_length: 4`; its SHA-256 is `16bfabe83aded6d17e171ec697a2439675e096bf2bdb3076925f13ef4510f122`.
- All 99 post-warmup throughput intervals remained above the 6320.254759 examples/s floor. The median was 8079.944432, the mean was 8056.134385, and the range was 6925.859394 to 8121.308926 examples/s.
- The terminal audit found the TPU `READY`/`HEALTHY`, W&B `finished`, no live trainer, no VFIO holder, no `/tmp/libtpu_lockfile`, the exact completion sentinel, and a step-1000 checkpoint that deserialized successfully.
- The repository does not contain the prescribed `tpu_scripts/watch_elf_run.sh`. The foreground session and a minimal-environment direct health/progress gate continuously covered this 156-second run; it completed while the recurring-monitor installation path was being validated, so no recurring entry was left behind.

## Interpretation

- This is the promoted full run from the accepted batch-256 throughput gate. A sustained post-warmup throughput below 80% of the matched 7900.318449 examples/s median is a material regression requiring diagnosis rather than silent continuation.
- The full run passed that throughput gate and finished with finite loss and a verified checkpoint. The 256-example validation slice is too small for a strong quality conclusion from its exact-match fluctuation.

## Updates

- 2026-10-14T21:41:23Z: User authorized launch. The mandatory filtered inventory returned the selected v5e-8 as `AVAILABLE` with ELF runtime `READY`. The complete YAML parsed successfully and was committed alone as `b337b68be31f36a314e0dae5720a72813c598ddd`; immutable staging, final preflight, and launch remain pending.
- 2026-10-14T21:50:24Z: The required correctness review found that evaluation time would contaminate the following throughput interval. Commit `bde42367a09836ee775ced42e1a37d6fe085649f` resets the interval after evaluation and adds a deterministic delay regression test. The focused test and full CPU suite passed (`89` tests and `7` subtests).
- 2026-10-14T21:57:04Z: The final cleanup review found no confirmed issues; reviewed head `d0da78079cbcd1301f9867b1b739c5760e12c4c6` was pushed to PR 1. The selected TPU is currently `READY`/`HEALTHY`, Spot, and initially `AVAILABLE`; the runtime reports JAX 0.6.2, Flax 0.10.5, and Optax 0.2.8. The regional bucket is US-WEST4 and the verified node-local dataset matches its seven canonical objects.
- 2026-10-14T21:57:04Z: Created and copied the immutable snapshot exactly once. The local and node-local copies contain 29 files and 293,002 file bytes with matching aggregate SHA-256 `bf07f75ad01b3b11c417cf0b001714b7e1c52567e16a88d4ec0a16791b11da23`; the node-local effective YAML parsed as batch 256, 1,000 steps, message length 4, and TPU execution.
- 2026-10-14T21:59:43Z: W&B assigned run `exf0hkel`. JAX formed the TPU backend on eight devices and reported the verified node-local dataset, 100,000 training examples, 256 validation examples, batch 256, and bfloat16 before the first compiled step.
- 2026-10-14T22:02:20Z: Completed all 1,000 steps and ten scheduled validations. W&B finished, the checkpoint payload restored at step 1000, the canonical log and effective config were preserved with exact remote/local hashes, and the terminal all-worker audit found the selected TPU healthy and idle.
