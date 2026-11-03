# Conlang JAX v4-8 one-example 2,000-step hidden-128 untied-encoder run

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64` |
| Status | `completed` |
| Started | `2026-10-16T06:07:52Z` |
| Finished | `2026-10-16T06:10:39Z` (W&B runtime 166.398865089 seconds) |
| Baseline | `20261016-conlang-jax-v4-8-n1-s2000-h128-out256-tied-g64` |
| Git revision | reviewed configuration `c322228865eecb4c8568110a91f308beadd87047`; one-example runtime source `b8adc3fe683781839b978d71fdc1cd1701f5a314` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64.yaml`

```yaml
run_id: 20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64
dataset_dir: null
dataset_cache_dir: /home/kristinelu/elf-data/conlangs-v1
dataset_prefix: kristine/datasets/conlangs
batch_size: 4
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
output_path: /home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/checkpoint.msgpack
limit_examples: 1
repeat_training_examples_to_batch_size: true
wandb_project: conlangs
wandb_run_name: 20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64
wandb_enabled: true
hidden_size: 128
output_hidden_size: 256
tie_sender_receiver_encoder: false
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
| Tie sender and receiver encoder weights | true | false | selected config |

Run identity, output path, and W&B name are also fresh. The one selected datum, explicit four-way batch replication, seed, 2,000-step schedule, hidden size 128, output hidden dimension 256, separate 64-channel role gates, optimization, regional dataset, validation set, and compact top-100 vocabulary histogram remain matched to the tied baseline.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| Training prefix | Train loss | 1.5199091194517678e-06 | 2000 | min | `metrics.jsonl` |
| Validation | Exact match | 0.0 | 2000 | max | `metrics.jsonl` |
| Validation | Mean joint log probability | -35.6571341753006 | 2000 | max | `metrics.jsonl` |
| Training prefix | Median throughput | 220.68392653160566 examples/s | 20--2000 | max | `metrics.jsonl`; 199 post-warmup intervals |

- The metrics file contains 200 training records, 20 evaluation records, and 199 post-warmup throughput measurements. Throughput averaged 220.50826127745947 examples/s and ranged from 210.29754195859886 to 225.52665633625122 examples/s.
- Train loss fell from 4.41436243057251 at step 10 to 1.5199091194517678e-06 at step 2,000. Validation exact match remained 0.0 throughout, while mean joint log probability declined from -25.40328887104988 at step 100 to -35.6571341753006.
- Sender gate mean changed from 0.11881542205810547 to 0.10746645927429199; its final near-zero and near-one fractions were both 0.0. Receiver gate mean changed from 0.12403798848390579 to 0.10396237671375275; its final near-zero and near-one fractions were both 0.0.
- All 200 training records contain compact generated-vocabulary histograms with exactly three populated top-token bins. W&B contains no media or table artifact.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64.yaml` | yes; SHA-256 `eda79639972f1ee8e74cc344d5fda6bd2ef3504264e4394867ef1886025e8b79` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed configuration pushed at `c322228` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64-20261016T060515.513675859Z`; remote path `/home/kristinelu/elf-runs/source/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64` | yes; local and remote contain 386599 bytes; selected file hashes match; relative SHA-256-manifest digest `749f44ba486bb09e314b5b28da28dde3aee0d740a2949078eee358beafe62350` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; six gzip files passed integrity, manifest reports 15625 worlds and 125000 questions, manifest SHA-256 `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Runtime | `/home/kristinelu/elf-runs/runtime/venv` | yes; READY marker, JAX 0.6.2, Flax, Optax, and W&B imports verified |
| Checkpoint | `/home/kristinelu/elf-runs/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/checkpoint.msgpack` | yes; 27015425 bytes, SHA-256 `66f732a97d19553a2a60be2591e6fd8bd9c1956d56314ec05e89c51972c03cdf`, CPU restore verified step 2000 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/ucq58yxl` | yes; state `finished`, exact config identity verified, 200 history rows, five metadata/log files, and no media/table files |
| Metrics | `experiment_log/2026-10-16/artifacts/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/metrics.jsonl` | yes; 220 lines, SHA-256 `5ed6a1a577e4c5992f0ffb8eeb7d93dfed59f85660116bfd8ff3241b94dc9248` |
| Generated outputs | `experiment_log/2026-10-16/artifacts/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/outputs.jsonl` | yes; 200 lines, SHA-256 `1cfed29c9a2a6dfbcbef39f605c55dc92cf92257e418526b3c4eed017b6454da` |
| Effective config | `experiment_log/2026-10-16/artifacts/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/config.yml` | yes; SHA-256 `38b362b48193d6a91e140ded06b9ca62ec6bac0ef355f8744ffa82ce97d68f94` |
| Canonical launcher log | `experiment_log/2026-10-16/artifacts/20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64/launcher.log.gz` | yes; deterministic lossless gzip SHA-256 `9b5677a6006c74a5af3d4f27a09cf1fc75c669b58366022c60117c7972897bcb`; decompressed 262-line log SHA-256 `569d53b0af88453256cb1022439633eac4547ee7b4080e4e0e3614b021035c59` |

## Failure Details

- Last known step: `2000`
- Error summary: not applicable
- Detailed output: canonical launcher log listed above; no fatal marker found

## Observations

- The mandatory fresh Kristine-only inventory selected this exact v4-8 from the final `AVAILABLE` table after the tied arm completed. Direct checks reported `READY`, `HEALTHY`, Spot `true`, and `AVAILABLE`.
- Correctness and cleanup reviews both returned `NO_CONFIRMED_FINDINGS` for immutable range `fd708627d784a9c33928934456c153a251997a45..c322228865eecb4c8568110a91f308beadd87047`.
- The actual YAML parsed and validated, and structured comparison confirmed only matched-arm identity fields plus encoder tying changed. The unchanged source passed all 35 CPU-only JAX training tests before the tied arm launched.
- The source snapshot is immutable on the shared mount. The selected config and `train.py` have SHA-256 values `eda79639972f1ee8e74cc344d5fda6bd2ef3504264e4394867ef1886025e8b79` and `5bdf6852dbacf524ec86d0ce17a3583fbf28536ab14e42eebbf15c81f9ea33bb`.
- JAX reported TPU backend, four devices, exactly one unique training example, 256 validation examples, global batch 4, and bfloat16 precision. The canonical log contains one completion sentinel at step 2,000 and no fatal marker.
- After completion, the exact trainer and libtpu lock were absent, VFIO was idle, the TPU remained `READY` and `HEALTHY`, W&B was `finished`, and the checkpoint restored at step 2,000.
- The repository has no `tpu_scripts/watch_elf_run.sh`; the preserved foreground session and direct terminal checks continuously covered this 166-second run.
- Artifact-copy SCP encountered several `message authentication code incorrect` transport errors, then its built-in retries succeeded. Artifact hashes match the already-recorded remote values, so this was a control-plane transport event rather than a TPU or training failure.
- The report and copied evidence are no-code items, so their finalization is `SKIPPED_NO_CODE_ITEMS`; the runtime YAML received the required correctness and cleanup passes before launch.

## Interpretation

- This is a controlled tied-vs-untied comparison on one unique training example; neither arm estimates generalization.
- Both arms memorized the datum and scored 0/256 exact matches. Untying improved the final mean joint log probability by 1.9911144375801086 but reduced median throughput by 52.60661229999681 examples/s (19.24933535017552%) relative to the tied arm. This reports an association in this single paired run, not causal certainty.

## Matched Comparison

| Run | Encoder weights | Final train loss (min) | Validation exact match (max) | Mean joint log probability (max) | Median throughput (max) |
|---|---|---:|---:|---:|---:|
| `20261016-conlang-jax-v4-8-n1-s2000-h128-out256-tied-g64` | tied | **1.1922384146600962e-07** | **0.0** | -37.64824861288071 | **273.2905388316025 examples/s** |
| `20261016-conlang-jax-v4-8-n1-s2000-h128-out256-untied-g64` | untied | 1.5199091194517678e-06 | **0.0** | **-35.6571341753006** | 220.68392653160566 examples/s |

The two runs used the same one-example training prefix, four-way replicated global batch, validation split and method, seed, schedule, precision, topology, output width, and gate configuration.

## Updates

- `2026-10-16T06:05:27Z`: Reviewed config commit was pushed to PR 1, the immutable source snapshot was created, and the selected v4-8 remained READY, HEALTHY, Spot, and idle. Launch remains pending node-local source copy, complete preflight, and final occupancy gate.
- `2026-10-16T06:07:52Z`: Node-local source hashes and application-level dataset validation passed; final state and occupancy gates reported `READY`, `HEALTHY`, and `AVAILABLE`; launched the untied arm, registered W&B run `ucq58yxl`, and confirmed useful progress.
- `2026-10-16T06:10:39Z`: Training completed at step 2,000. W&B, checkpoint restore, artifact hashes, terminal log, process absence, VFIO idleness, and TPU health were independently verified.
