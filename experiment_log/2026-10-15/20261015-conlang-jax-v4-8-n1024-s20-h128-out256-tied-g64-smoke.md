# Conlang JAX v4-8 1,024-example hidden-128 tied-encoder smoke test

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke` |
| Status | `completed` |
| Started | `2026-10-16T03:55:35Z` |
| Finished | `2026-10-16T03:58:56Z` (W&B runtime 201.29 seconds) |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-h512-output-h256-tied-encoders-gates64` |
| Git revision | reviewed source and configuration `ef22ee5a3c5d7cf45615315d5538dd7570eb510a` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke-20261016T035317.800843722Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, expected four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
learning_rate: 0.0003
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 20
eval_every: 10
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/checkpoint.msgpack
limit_examples: 1024
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: true
role_gate_channels: 64
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
| Training examples | 100000 | 1024 | selected config and dataset loader |
| Training steps | 2000 | 20 | selected config |
| Evaluation interval | 100 steps | 10 steps | selected config |
| Transformer/message hidden width | 512 | 128 | selected config |

The batch size 256, output hidden dimension 256, tied sender/receiver encoder, separate 64-channel role gates, optimization, seed, validation size, regional dataset, and compact vocabulary histograms remain unchanged. The 1,024-example limit is exactly four full batches per epoch.

## Results

| Dataset / split | Metric | Value | Step | Direction | Source |
|---|---|---:|---:|---|---|
| Training prefix | Train loss | 7.737129211425781 | 20 | min | `metrics.jsonl` |
| Validation | Exact match | 0.0 | 20 | max | `metrics.jsonl` |
| Validation | Mean joint log probability | -7.7169189453125 | 20 | max | `metrics.jsonl` |
| Training prefix | Examples per second | 5853.003865536729 | 20 | max | `metrics.jsonl`; single post-warmup interval |

- Train loss fell from 9.361905097961426 at step 10 to 7.737129211425781 at step 20. Mean joint log probability improved from -9.012317657470703 to -7.7169189453125; exact match remained zero, as expected for a setup smoke test.
- Sender gate mean changed from 0.1285565048456192 to 0.12801194190979004, and receiver gate mean changed from 0.13244710862636566 to 0.12438870221376419. All gate statistics were finite and bounded, with no near-zero or near-one saturation in this short run.
- Both logged training steps contain compact two-bin generated-vocabulary histograms, below the configured 100-token cap. W&B contains no media or table artifact.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke.yaml` | yes; SHA-256 `e63b3b8086612cf0997ff20fdf6132600fa54ee2c21b62e06793ff045250976c` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed configuration pushed at `ef22ee5` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke-20261016T035317.800843722Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 42 files and 356842 bytes with relative SHA-256-manifest digest `58c1b291c2cc2a8cffe9c66465265d76a0de286666afd79f8f0b881091cc304c` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; deterministic Central2 dataset reverified with 15625 worlds and 125000 questions; manifest SHA-256 from the established record `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Runtime | `/home/kristinelu/elf-runs/runtime/venv` | yes; pinned JAX 0.6.2 profile passed a no-initialize readiness check |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/checkpoint.msgpack` | yes; 22250295 bytes, SHA-256 `ff66a8be20e9bfcfceb2a969864791ab5f2fc304dce35cc347a4d22f1c79c8ff`, CPU restore verified step 20 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/1lr5w1l8` | yes; state `finished`, summary/config identity verified, five log/metadata files and no media/table files |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/metrics.jsonl` | yes; 4 lines, SHA-256 `dcf30c5d3dd2c941b56a3b9dfcc6143fd14527bc75c5e2920e4f522890cc5479` |
| Generated outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/outputs.jsonl` | yes; 2 lines, SHA-256 `75b670cf01b478e15c3d851d9635982ffdf40d3fc1a597b659c5aa81654f4168` |
| Effective config | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/config.yml` | yes; SHA-256 `99163d473fa4e77944c0010f5958d7bda574eda9ab786153f35ff8b38e6be785` |
| Canonical launcher log | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke/launcher.log` and deterministic gzip | yes; 48 lines, raw SHA-256 `b2ab25b38b204b69d288b87e4519672f3946a5eb990205da17f0652f15151e5c`, gzip SHA-256 `01dbe7e614b2942c2c996e05fc995056c42ccd20876e267441a9eee637170859` |

## Failure Details

- Last known step: `20`
- Error summary: not applicable
- Detailed output: canonical launcher log listed above; no fatal marker found

## Observations

- The mandatory Kristine-only inventory selected this v4-8 from the final `AVAILABLE` table. Direct checks reported `READY`, `HEALTHY`, and `AVAILABLE`.
- Remote preflight reported 86 GiB free for data and outputs, an absent run output path, and an idle TPU device.
- Correctness review found no issue in the exact one-file configuration commit. Cleanup's only suggestion was rejected because `20261015` is the current date in the required America/New_York client timezone despite the UTC rollover; no material review finding remains.
- JAX reported the TPU backend, four devices, 1,024 training examples, and 256 validation examples. The canonical log contains exactly one completion sentinel at step 20 and no fatal marker.
- After completion, the trainer and libtpu lock were absent, VFIO was idle, the TPU remained `READY` and `HEALTHY`, W&B was `finished`, and the checkpoint restored at step 20.
- The repository has no `tpu_scripts/watch_elf_run.sh`; the preserved foreground launch session and direct terminal checks continuously covered this 201-second run.

## Interpretation

- This smoke test is intended to validate the complete tied-encoder width-128 pipeline on a small deterministic prefix, not to estimate model quality or compare against the 2,000-step baseline.
- The paired untied-encoder arm must use the same data limit, seed, hardware, and all other hyperparameters before any shared-vs-untied interpretation.
- The single post-warmup throughput value proves the optimized step executes but is not a stable throughput benchmark.

## Updates

- `2026-10-16T03:54:22Z`: Reviewed config commit was pushed to PR 1, the existing runtime and regional dataset were reverified, the immutable source snapshot matched locally and remotely, and remote preflight passed. Launch remains pending the final occupancy gate.
- `2026-10-16T03:55:35Z`: Final state and occupancy gates reported `READY`, `HEALTHY`, and `AVAILABLE`; launched the tied-encoder arm and registered W&B run `1lr5w1l8`.
- `2026-10-16T03:58:56Z`: Training completed at step 20. W&B, checkpoint restore, artifact hashes, terminal log, process absence, VFIO idleness, and TPU health were independently verified.
