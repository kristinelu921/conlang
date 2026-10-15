# Conlang JAX v4-8 batch-256 2,000-step W&B top-100 vocabulary histogram

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100` |
| Status | `completed successfully` |
| Started | `2026-10-15T05:03:54Z` |
| Finished | `2026-10-15T05:08:42Z` |
| Baseline | `20261015-conlang-jax-v4-8-b256-steps2000-wandb-retry5` |
| Git revision | reviewed source and configuration `3ec7d7777d769756043c8ef16c29739fe352b039` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/usr/bin/ssh -T -i /home/kristinelu/.ssh/google_compute_engine -o CheckHostIP=no -o HashKnownHosts=no -o HostKeyAlias=tpu.7819895973943982184-0-0GC0kY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/home/kristinelu/.ssh/google_compute_known_hosts -F /dev/null kristinelu@10.130.0.103 -- 'set -euo pipefail; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100; test ! -e /tmp/libtpu_lockfile; test -z "$(sudo fuser /dev/vfio/* 2>/dev/null || true)"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100-20261015T045722.853880285Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES="" ELF_TPU_WORKER_INDEX=0 PYTHONPATH="$PWD" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-b256-steps2000-wandb-top100.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100.launcher.log'` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-15-01-33-19-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, 4 JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-b256-steps2000-wandb-top100.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100
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

No scientific hyperparameter changed from retry 5. Telemetry adds one generated-token-ID histogram containing at most the 100 most frequent tokens at each `log_every: 10` boundary. Exact selected token IDs remain in local metrics only; W&B receives no vocabulary table or token-ID list. Existing compact 32-bin activation histograms remain enabled.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | final loss | 3.934626579 | 2000 | lower is better | `metrics.jsonl` and W&B |
| training | median steady-state throughput | 5861.744102 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| training | mean steady-state throughput | 5839.345120 examples/s | 199 post-warmup intervals through step 2000 | higher is better | `metrics.jsonl` |
| validation | exact match | 0.0390625 | 2000 | higher is better | `metrics.jsonl` and W&B |
| validation | mean joint log probability | -4.053215981 | 2000 | higher is better | `metrics.jsonl` and W&B |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-b256-steps2000-wandb-top100.yaml` | yes; SHA-256 `b6cd34ccd221b5b66cdd40f74765ad9c2a94acc450fa2c8da2736d488ee3009b`, committed alone as `bd0c641` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed launch closure pushed at `3ec7d77` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100-20261015T045722.853880285Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; 37 files, 328422 bytes, relative SHA-256-manifest digest `7ff62e5eb596da06e0047d7baf7ba660f8a1dbf083b0d2b309968d6745db7622` matched locally and remotely; `train.py` SHA-256 `f6c408674957872a8afa727ef9914840a416f8042a6bb1493099a16c994b1ed5` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, generated deterministically on the central2 TPU | yes; manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab3c29588e8`, 15625 worlds, 125000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/checkpoint.msgpack` | yes; 22656218 bytes, SHA-256 `45a11967333646825f4ac50c47e9e5c9f3b316908aa54af427003c909d65a538`, serialized step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/532rxucx` | yes; ID `532rxucx`, authenticated state `finished`, last history step 2000, histogram present in summary, no token-ID list or table file |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/metrics.jsonl` | yes; 220 records, 351926 bytes, SHA-256 `bf04d04135a4900ac9b991ec3800c4f3d8c6f93d7cc03027324ddee81e37b561` |
| Evaluation outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/outputs.jsonl` | yes; 200 diagnostic-cadence records from steps 10 through 2000, 2807042 bytes, SHA-256 `d41dc4c31f7cb876ba4bbc8b437ed883ab3fca9cf2b0d2da3385324978ae360b` |
| Effective configuration | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/config.yml` | yes; 853 bytes, SHA-256 `29a5edd9d46a61df27f990786f62357621499a0b415543938a42fd93444ab055` |
| Canonical log | remote raw log `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100.launcher.log`; deterministic local gzip `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100/20261015-conlang-jax-v4-8-b256-steps2000-wandb-top100.launcher.log.gz` | yes; raw 20201 bytes and SHA-256 `042da48507e84e09d99e7acd370682c0a526ae5fc14a0df10d01fd75a4534a2a`; gzip 4200 bytes and SHA-256 `b8ee4f709f905fecbd5e6923eb21405a89d1d7c494dcf7da214148ce56862d14` |

## Prelaunch Recovery

- At `2026-10-15T04:57Z`, the first post-copy gcloud verification reconnect exited `255` before authentication with `kex_exchange_identification: banner line contains invalid characters`. The TPU API still reported `READY`, `HEALTHY`, Spot `v4-8`; no trainer had started, step 1 had not completed, the fresh output remained absent, and there was no checkpoint.
- The known control-transport failure was isolated from authentication and TPU health by printing gcloud's exact worker-0 SSH command and using that bounded direct command with `-F /dev/null`. The direct public-key probe exited `0`; source hashes, the CPU-only runtime (`jax 0.6.2`, `flax 0.10.5`, `optax 0.2.8`, `wandb 0.28.1`), W&B credential presence, and the complete node-local regional dataset then verified successfully.
- Two operator-authored CPU verification snippets exited before touching the TPU backend: one imported `verify_dataset` from the wrong module and one passed a string instead of the producer-supported `Path`. The corrected typed check returned the complete manifest counts above. No application repair or retry identity was required.
- The first terminal worker audit also hit a transient packet-authentication transport error while the direct TPU API remained `READY` and `HEALTHY`. A bounded direct-SSH retry succeeded and supplied all terminal evidence below.

## Observations

- JAX exposed the intended 4 TPU devices. The trainer reached every expected evaluation boundary, emitted `TRAINING_COMPLETE` at step 2000, and W&B finalized as `finished` at history step 2000.
- All 200 logged training records contain one generated-vocabulary histogram with no more than 100 bins; the observed maximum was 20 bins because no logged batch generated more than 20 unique token IDs. Every record has exactly one more edge than count. Exact top token IDs remain only in the local metric records.
- The authenticated W&B summary contains `generated_vocabulary_histogram`, while `generated_vocabulary_top_token_ids` is absent and the run has no table-like files.
- The 199 post-warmup throughput intervals had minimum 4238.216810, median 5861.744102, mean 5839.345120, and maximum 5913.089662 examples/s. Median throughput is 0.682% above retry 5 on the same TPU and configuration.
- The terminal single-worker audit found the TPU `READY` and `HEALTHY`, the exact trainer absent, all VFIO resources idle, and `/tmp/libtpu_lockfile` absent. The checkpoint independently deserialized with `step=2000`.
- Final loss and validation metrics exactly match retry 5, as expected from the unchanged scientific configuration and seed.

## Interpretation

- The requested 2,000-step JAX rerun completed successfully with authenticated W&B tracking.
- The top-100 vocabulary histogram is available in W&B without a vocabulary table or token-ID list, and it stayed safely below W&B's histogram serialization limit.
- The telemetry-only change preserved deterministic scientific output and did not materially reduce throughput.

## Updates

- `2026-10-15T05:02:18Z`: The reviewed configuration closure was pushed, the mandatory inventory selected this exact Kristine TPU from the final `AVAILABLE` table, preflight reported 85 GiB free with the accelerator idle, and the immutable source and regional data passed verification. Launch remains pending the final occupancy gate.
- `2026-10-15T05:03:54Z`: The canonical occupancy helper returned `AVAILABLE`; the foreground TPU trainer started and W&B created run `532rxucx`.
- `2026-10-15T05:08:42Z`: Training completed step 2000, checkpoint and artifact verification passed, W&B finalized, and the independent terminal audit found the TPU healthy and idle.
