# Conlang JAX v5e-8 b256 optimization 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261014-conlang-jax-v5e8-b256-opt1` |
| Status | `completed` |
| Started | `2026-10-14T21:29:35Z` |
| Finished | `2026-10-14T21:31:52Z` |
| Baseline | `20261014-conlang-jax-v5e8-b128-gate1` |
| Git revision | `3cd53e9c0d656dc9bc5634511dff405017960515` on `codex/jax-tpu-training` |
| Command | `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 /home/kristinelu/elf-runs/runtime/venv/bin/python train.py --config configs/conlang-jax-v5e8-b256-opt1.yaml` |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`; v5litepod-8; single host |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-opt1.yaml`

```yaml
run_id: 20261014-conlang-jax-v5e8-b256-opt1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 256
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
output_path: /home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261014-conlang-jax-v5e8-b256-opt1
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

- `batch_size`: 128 -> 256. Every other model, optimizer, precision, data, seed, sequence-length, evaluation, and hardware setting is unchanged.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| validation | exact match | 0.0234375 | 200 | higher is better | `metrics.jsonl`, W&B `okba6v37` |
| validation | mean joint log probability | -4.1818027496 | 200 | higher is better | `metrics.jsonl`, W&B `okba6v37` |
| training | final loss | 4.1102037430 | 200 | lower is better | `metrics.jsonl`, W&B `okba6v37` |
| training | median steady-state throughput | 7,900.3184 examples/s | 19 intervals after warmup | higher is better | `metrics.jsonl` |
| training | throughput gain versus batch 128 | 52.1235% | matched 19-interval summaries | higher is better | both run metrics |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-opt1.yaml` | yes |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261014-conlang-jax-v5e8-b256-opt1-20261014T212713.304017826Z`; aggregate SHA-256 `2333c371e16c195d24f239210fcfa514d6e03aaaf07abff3e29b8f149399c320` | yes; 28 files, 310,723 bytes, exact remote match |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1`, reused from the verified US-WEST4 copy | yes; 125,000 questions |
| Checkpoint | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/checkpoint.msgpack`; 22,656,217 bytes; SHA-256 `3208ec1e65d7797e5acc72e8015d4f319ccf187b6a2ade541a7fe256efc565fb` | yes; deserialized at step 200 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/okba6v37` | yes; `finished`, step 200 |
| Metrics | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1/metrics.jsonl`; 2,596 bytes | yes |
| Canonical log | `/home/kristinelu/elf-runs/20261014-conlang-jax-v5e8-b256-opt1.launcher.log`; SHA-256 `f96f46565fef33b550a6e51f971cf1129ad73a6d28040e27ad9e1de451ada62c` | yes; completion sentinel at step 200 |

## Failure Details

- Last known step: 200, completed
- Error summary: none
- Detailed output: all 200 steps, final evaluation, checkpoint serialization, and W&B synchronization completed successfully

## Observations

- The batch-128 baseline completed at a median 5,193.3586 examples/s across 19 post-warmup intervals, ranging from 5,160.9233 to 5,208.0937 examples/s.
- The same node-local, fully verified US-WEST4 dataset was reused without another regional transfer.
- Batch 256 completed at a median 7,900.3184 examples/s across 19 post-warmup intervals, ranging from 7,847.8531 to 8,019.0929 examples/s.
- Validation exact match was 0.0234375 versus 0.03515625 for batch 128 on the same 256 examples. This tiny single-seed gate is a throughput comparison, not enough evidence to claim a quality difference.

## Interpretation

- Doubling the per-device batch from 16 to 32 improved median throughput by 52.1235% with finite loss, no memory failure, and a successfully deserialized final checkpoint. Batch 256 is the selected optimized v5e-8 setting for this model and sequence shape. The validation sample is too small for a quality conclusion, so longer scientific training should retain evaluation monitoring.

## Updates

- 2026-10-14T21:23:45Z: Selected the directly comparable batch-256 optimization arm after the batch-128 run completed and passed its terminal audit.
- 2026-10-14T21:25:36Z: Pushed the selected run config and report in commit `3cd53e9c0d656dc9bc5634511dff405017960515` to PR 1; the optimization arm remains unlaunched pending fresh inventory, preflight, and immutable staging.
- 2026-10-14T21:28:22Z: Mandatory inventory and preflight selected the same available v5e-8. Copied the immutable snapshot to node-local storage with an exact aggregate hash match and confirmed the effective config remains batch 256, message length 4, and 200 steps. Reused the fully verified regional dataset without another transfer.
- 2026-10-14T21:31:52Z: Completed all 200 steps on eight TPU devices. W&B run `okba6v37` finished at step 200; the final checkpoint deserialized successfully, the canonical log contained the exact completion sentinel, and the TPU remained READY/HEALTHY with no trainer, VFIO holder, or libtpu lock. Median throughput improved 52.1235% over the matched batch-128 gate.
