# Tiny 32-dimensional single-link one-epoch 2.56M run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1` |
| Status | `planned` |
| Predecessor | `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1` completed 2,000 steps on the 100,000-example dataset |
| Git revision | `74c6486328e2f3e6a0514943da126ff2e9f63989` (exact configuration commit) |
| Pull request | `https://github.com/kristinelu921/conlang/pull/2` |
| Primary configuration | `configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml` |
| Seed | `1729` |
| Intended hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, Spot `v5litepod-8` |

## Configuration

```yaml
run_id: 20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-single-link-n2560000-v1
dataset_prefix: kristine/datasets/conlangs-single-link-n2560000-v1
batch_size: 256
learning_rate: 0.0003
sender_learning_rate_multiplier: 1.0
weight_decay: 0.01
max_gradient_norm: 1.0
steps: 10000
eval_every: 200
eval_examples: 256
log_every: 10
throughput_warmup_steps: 10
seed: 1729
device: tpu
precision: bfloat16
output_path: /home/kristinelu/elf-runs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1/checkpoint.msgpack
pretrained_initialization: null
pretrained_model_dir: null
pretrained_checkpoint_sha256: null
pretrained_tokenizer_sha256: null
limit_examples: null
repeat_training_examples_to_batch_size: false
evaluation_message_interventions: true
wandb_project: conlangs
wandb_run_name: 20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1
wandb_enabled: true
hidden_size: 32
output_hidden_size: null
sender_hidden_size: 32
sender_num_layers: 2
sender_num_heads: 4
sender_feedforward_size: 128
sender_average_residual: false
tie_sender_receiver_encoder: false
role_gate_channels: null
sender_question_conditioning: false
receiver_question_input: true
receiver_question_query_only: false
communication_rounds: 1
bridge_supervision_weight: 0.0
message_length: 1
max_answer_length: 4
max_text_length: 512
max_question_length: 128
num_layers: 2
num_heads: 4
dropout: 0.1
relative_attention_num_buckets: 32
relative_attention_max_distance: 128
```

## Changes From Baseline

Relative to `20261018-conlang-jax-v5e8-b256-tiny32-single-link-full1`:

- training data increases from 100,000 to exactly 2,560,000 examples;
- steps increase from 2,000 to 10,000;
- batch size remains 256, so `10,000 * 256 = 2,560,000` and the run is exactly one epoch;
- the model remains one message token with a 32-dimensional continuous embedding, sender depth 2, and receiver depth 2;
- all optimizer hyperparameters and seed remain unchanged.

## Dataset Generation

Exact command:

```bash
python -m linked_facts.generate_dataset \
  --output-dir /tmp/conlangs-single-link-n2560000-v1 \
  --seed 1729 \
  --entities-per-world 8 \
  --train-worlds 320000 \
  --validation-worlds 625 \
  --test-worlds-per-split 625 \
  --compression gzip
```

Full verification command:

```bash
python -m linked_facts.generate_dataset \
  --output-dir /tmp/conlangs-single-link-n2560000-v1 \
  --verify-only
```

Verification result: 323,125 worlds, 2,585,000 total questions, and 8 questions per world. The training split contains 320,000 worlds and exactly 2,560,000 questions. Each sender input contains one necessary fact; the two sender facts together form the two-hop chain.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| `manifest.json` | `d8576253b65786a99f5143582c8b561b063a3ecb239ff6b54044286f1854835f` | 3,087 |
| `train.jsonl.gz` | `f267ed081a4ead1886a353967d19b2204800bbb60a7b8b05d392e5c812e3a247` | 244,578,956 |
| `validation.jsonl.gz` | `79c51d6784694af833f4227121e424829d2aa0e233e318a354b0ec9e5619e049` | 480,142 |
| `test_familiar.jsonl.gz` | `10a6df68c5a649efb9ee29f11494daa426c403e77d61d71dd2fedcc72897cfe3` | 479,873 |
| `test_new_names.jsonl.gz` | `a7ce0a2b2ba53747c94a00e79ed765b05affa151cdcecf2185bdc65b52bdcb70` | 485,327 |
| `test_new_wording.jsonl.gz` | `f40080851ab8e5baaae23148164eaf1f2c594ac9674754f59c889a0452b7284e` | 491,162 |
| `test_new_names_and_wording.jsonl.gz` | `e4b428586fb3ff04bcac10ab3276abdd5b2d73769397c7f957b62096cb319adf` | 499,394 |

Regional source: `gs://kmh-gcp-us-west4/kristine/datasets/conlangs-single-link-n2560000-v1` in `us-west4`. The upload inventory contains exactly seven objects and 247,017,941 bytes, matching the local file sizes.

## Validation

- Configuration parse and arithmetic assertions passed: 10,000 steps, batch size 256, and 2,560,000 training presentations.
- Architecture assertions passed: message length 1, continuous width 32, sender depth 2, and receiver depth 2.
- Full dataset streaming validation passed, including world structure, counts, byte sizes, and uncompressed content checksums.
- Focused configuration/batching tests: 17 passed.
- Broader generator and JAX training suite: `77 passed` with `JAX_PLATFORMS=cpu` in 216.74 seconds.
- TPU-local dataset verification passed with the same 323,125 worlds, 2,585,000 total questions, 8 questions per world, and all seven local SHA-256 hashes matching the generation host.

## Review

The required correctness pass reviewed the complete PR diff from base
`051db79bc7092a3c5422f3ace9e7aaa42c32d53e` through head
`c300d249ea614affd6478f595626e2582183a7d8` and returned
`NO_CONFIRMED_FINDINGS`. Its focused generator, model-shape, and staging checks
passed 12 tests. The cleanup pass is pending on the updated frozen head.

## Launch and Artifacts

The exact immutable source path, launch command, W&B URL, checkpoints, metrics, outputs, and terminal status will be appended after staging and launch. No TPU process has been started for this run yet.
