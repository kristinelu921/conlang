# Conlang JAX v4-8 batch-256 2,000-step output-head-256 W&B top-100 vocabulary histogram

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100` |
| Status | `completed` |
| Started | `2026-10-15T05:35:26Z` |
| Finished | `2026-10-15T05:40:12Z` |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100` |
| Git revision | reviewed source and configuration `9fb9ce12a4e10bff4517a512b81b94143f40a36d` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100-20261015T053225.239402639Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
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

| Parameter | Baseline | This run |
|---|---:|---:|
| Transformer/message `hidden_size` | 128 | 128 (unchanged) |
| Output head | direct 128-to-vocabulary projection | 128-to-256 ReLU hidden projection, then vocabulary projection |

All other scientific hyperparameters, the seed, W&B settings, and top-100 generated-vocabulary histogram telemetry match the baseline.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | final loss | 3.909019947 | 2000 | lower is better | `metrics.jsonl` and W&B |
| training | median steady-state throughput | 5842.997587 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| training | mean steady-state throughput | 5829.561196 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| validation | exact match | 0.0390625 | 2000 | higher is better | `metrics.jsonl` and W&B |
| validation | mean joint log probability | -4.058863640 | 2000 | higher is better | `metrics.jsonl` and W&B |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.yaml` | yes; SHA-256 `2ad1f7044133bdf0bcdd56d87b2bb1da20ee9c199daa47aa7139f1441a4a2503`, committed in the reviewed run closure `9fb9ce1` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `9fb9ce1` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100-20261015T053225.239402639Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 38 files, 332243 bytes, relative SHA-256-manifest digest `52a11a0e125da5550f1ffc5385d75ec8eaa2b2db27cb79544c9bb61935ca99e9` matched locally and remotely; `train.py` SHA-256 `776d3963632165ae7dc13cd95d4f57dd21cabcbd1c6ce2440ec1893022f55b94` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, generated deterministically on the central2 TPU | yes; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`, 15625 worlds, 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/checkpoint.msgpack` | yes; 26418137 bytes, SHA-256 `2596e857c843538e76f18b7325dcecdd281d7a0761818cd5de0372a2e9dbd9c4`, serialized step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/1yme3x2f` | yes; ID `1yme3x2f`, authenticated state `finished`, summary step 2000, histogram present in summary, no token-ID list or table file |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/metrics.jsonl` | yes; 220 records, 347471 bytes, SHA-256 `62719f55f70a3e748b3b8ea2a1cd32e981216d844e02c87cea85bca251efa530` |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/outputs.jsonl` | yes; 200 diagnostic-cadence records from steps 10 through 2000, 2806305 bytes, SHA-256 `997f0d8b6d0bc347edfa1d62ad92f9f0ecae2faca53ee1a18d1f074fdc4143ee` |
| Effective configuration | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/config.yml` | yes; 913 bytes, SHA-256 `8888be7e132913757fe2a289e051f7d69a5535de27d42278c4cf44b3d681770d` |
| Canonical log | remote raw log `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.launcher.log`; deterministic local gzip `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100/20261015-conlang-jax-v4-8-b256-steps2000-output-h256-wandb-top100.launcher.log.gz` | yes; raw 20332 bytes and SHA-256 `ab2ab4aebd6e1ea0659d8e10974ab86b5f709a309642835f5188e46dca7a7d79`; gzip 4171 bytes and SHA-256 `08d13db1630038f8454027e2445ae0a8f842133baaa4e13f421b983558f22a83` |

## Observations

- JAX exposed all 4 TPU devices, used the complete node-local central2 dataset, reached every expected evaluation boundary, emitted `TRAINING_COMPLETE` at step 2000, and W&B finalized as `finished` at summary step 2000.
- The architecture remained 128-wide through the transformer and message path. Only the receiver output head changed to a 256-wide ReLU hidden projection before vocabulary logits.
- All 200 logged training records contain one generated-vocabulary histogram with no more than 100 bins; the observed maximum was 18 bins. Every histogram has exactly one more edge than count. Exact top token IDs remain only in local metrics.
- The authenticated W&B summary contains `generated_vocabulary_histogram`, while `generated_vocabulary_top_token_ids` is absent and the run has no table-like files.
- The 199 post-warmup throughput intervals had minimum 5218.204713, median 5842.997587, mean 5829.561196, and maximum 5915.509043 examples/s. Median throughput is 0.320% below the direct-output baseline on the same TPU.
- The output-head checkpoint is 3761919 bytes (16.604%) larger than the baseline checkpoint, consistent with the additional projection and optimizer state.
- One post-run artifact-preservation SCP connection reported a transient packet-authentication error and gcloud retried automatically. Every preserved artifact matched the independently recorded remote SHA-256.
- The terminal single-worker audit found the TPU `READY` and `HEALTHY`, the exact trainer absent, every VFIO resource idle, and `/tmp/libtpu_lockfile` absent. The checkpoint independently deserialized with `step=2000`.

## Interpretation

- The requested 256-wide output hidden layer completed successfully without changing the transformer or message hidden width.
- Against the direct-output baseline, final loss improved by 0.025607, exact match was unchanged, mean joint log probability decreased by 0.005648, and median throughput decreased by 0.320%. One seed is not enough to attribute these small metric changes to the output head.
- W&B contains the requested top-100 vocabulary histogram without a large vocabulary table or token-ID list.

## Updates

- `2026-10-15T05:32:25Z`: The two-pass review returned no findings, commit `9fb9ce1` was pushed to PR 1, and the immutable source snapshot was created.
- `2026-10-15T05:35:26Z`: The canonical occupancy helper returned `AVAILABLE`; the foreground JAX trainer started on the selected Kristine v4-8 and W&B created run `1yme3x2f`.
- `2026-10-15T05:40:12Z`: Training completed step 2000, W&B finalized, checkpoint and artifact verification passed, and the independent terminal audit found the TPU healthy and idle.
