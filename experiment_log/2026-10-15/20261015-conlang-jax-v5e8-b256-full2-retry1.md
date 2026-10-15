# Conlang JAX v5e-8 batch-256 full run 2 retry 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v5e8-b256-full2-retry1` |
| Status | `planned` |
| Started | not recorded |
| Finished | not recorded |
| Baseline | `20261014-conlang-jax-v5e8-b256-full1`; retry of `20261015-conlang-jax-v5e8-b256-full2` |
| Git revision | `e94f185` on `codex/jax-tpu-training` |
| Command | not recorded; immutable snapshot identity and TPU selection pending |
| Seed | `1730` |
| Hardware | target v5e-8 currently unavailable: the mandatory filtered inventory found a healthy unrelated InstaManip smoke holding VFIO; waiting without touching it |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-full2-retry1.yaml`

```yaml
run_id: 20261015-conlang-jax-v5e8-b256-full2-retry1
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
seed: 1730
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2-retry1/checkpoint.msgpack
limit_examples: null
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v5e8-b256-full2-retry1
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

| Parameter | Failed attempt | Current | Source |
|---|---:|---:|---|
| `wandb_enabled` | true | false | failed-attempt and retry YAMLs |

Run identity, output path, and W&B display name are fresh. All scientific settings are identical to the failed seed-1730 attempt; W&B is disabled solely to bypass its deterministic oversized-histogram failure while retaining node-local metrics and outputs.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| training | baseline median steady-state throughput | 8079.944432 examples/s | 99 post-warmup intervals through step 1000 | higher is better | `20261014-conlang-jax-v5e8-b256-full1` |
| training | sustained-throughput floor | 6463.955546 examples/s | post-warmup retry intervals | higher is better | 20% regression tolerance from the baseline full run |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v5e8-b256-full2-retry1.yaml` | yes; parsed, matched scientifically to the failed attempt, and committed alone as `e94f185` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | existing PR; retry update pending |
| Source snapshot | not recorded | no |
| Regional dataset | not recorded | no; fresh selected-TPU preflight pending |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v5e8-b256-full2-retry1/checkpoint.msgpack` | no; run not started |
| Tracking run | disabled for this telemetry-recovery retry | not applicable |
| Metrics | not recorded | no |
| Canonical log | not recorded | no |

## Failure Details

- Last known step: not applicable; retry not started
- Error summary: not applicable
- Detailed output: not recorded

## Observations

- The failed attempt completed step 10, then W&B rejected a vocabulary-sized histogram before step 20. The model, optimizer, regional data, TPU backend, and compile all passed their gates.
- This retry changes only fresh identities and disables W&B. Full metrics, per-step model outputs, the foreground log, and terminal checkpoint remain node-local artifacts for direct monitoring and archival.

## Interpretation

- This is an infrastructure-only retry of the seed-1730 replica. Disabling W&B avoids the confirmed telemetry failure without altering optimization, model shape, data, precision, RNG seed, or evaluation schedule.

## Updates

- 2026-10-15T00:30:09Z: Planned a fresh retry identity after preserving the failed step-10 attempt. The retry disables W&B and retains all scientific settings; config validation, mandatory capacity selection, commit, PR update, immutable staging, and launch remain pending.
- 2026-10-15T00:33:31Z: The complete retry YAML parsed and differed from the failed attempt only in fresh identities and `wandb_enabled: false`; it was committed alone as `e94f185`. The fresh mandatory inventory found the prior v5e-8 occupied by an unrelated, healthy UID-2001 InstaManip smoke. Its owner, PID `2160637`, elapsed time, full command, and run identity were recorded read-only and the workload was left untouched. The remaining available Kristine v4-8s lack the pinned runtime and Central2 regional dataset, so the orchestrator is waiting for the proven v5e path rather than silently changing hardware and data provenance.
