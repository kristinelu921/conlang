# Conlang JAX v4-8 1,024-example hidden-128 untied-encoder smoke test

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke` |
| Status | `completed` |
| Started | `2026-10-16T04:06:33Z` |
| Finished | `2026-10-16T04:09:53Z` (W&B runtime 200.01 seconds) |
| Baseline | `20261015-conlang-jax-v4-8-n1024-s20-h128-out256-tied-g64-smoke` |
| Git revision | reviewed source and configuration `0f6cbcc14ad32f5db59134986ca529bd9f38c642` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/snap/google-cloud-cli/current/bin/gcloud compute tpus tpu-vm ssh kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b --project=he-vision-group --zone=us-central2-b --worker=all --internal-ip --ssh-flag=-F --ssh-flag=/dev/null --command="set -euo pipefail; mkdir -p /home/kristinelu/elf-runs; test -O /home/kristinelu/elf-runs; test -w /home/kristinelu/elf-runs; test ! -e /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke; test ! -e /tmp/libtpu_lockfile; test -z \"\$(sudo fuser /dev/vfio/* 2>/dev/null || true)\"; if test -e /tmp/tpu_logs; then test -w /tmp/tpu_logs; fi; cd /home/kristinelu/elf-runs/source/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke-20261016T040421.513908189Z; export JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' ELF_TPU_WORKER_INDEX=0 PYTHONPATH=\"\$PWD\" HF_HOME=/home/kristinelu/elf-runs/hf-cache PYTHONUNBUFFERED=1; ulimit -c 0; /home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke.yaml 2>&1 | tee /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke.launcher.log"` |
| Seed | `1730` |
| Hardware | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host, expected four JAX TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke.yaml`

```yaml
run_id: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke
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
output_path: /home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/checkpoint.msgpack
limit_examples: 1024
wandb_project: conlangs
wandb_run_name: 20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke
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
| Sender/receiver encoder sharing | enabled | disabled | selected config |

Run identity/output fields also differ by necessity. The data subset, seed, hardware, batch size, schedule, model dimensions, separate 64-channel role gates, optimizer, validation set, and logging settings match the completed tied arm exactly.

## Results

| Metric | Untied result | Tied baseline | Delta |
|---|---:|---:|---:|
| Final train loss | 7.687716484069824 | 7.737129211425781 | -0.04941272735595703 |
| Final validation exact match | 0.0 | 0.0 | 0.0 |
| Final mean joint log probability | -7.686850070953369 | -7.7169189453125 | +0.03006887435913086 |
| Step-20 examples per second | 5772.179528289805 | 5853.003865536729 | -80.824337246924 (-1.38%) |

- Untied train loss fell from 9.139741897583008 at step 10 to 7.687716484069824 at step 20. Mean joint log probability improved from -8.87811279296875 to -7.686850070953369; exact match remained zero.
- Sender gate mean changed from 0.12553581595420837 to 0.1183924525976181, and receiver gate mean changed from 0.1341700404882431 to 0.12590786814689636. All gate statistics were finite and bounded, with no near-zero or near-one saturation.
- Both logged training steps contain compact two-bin generated-vocabulary histograms, below the configured 100-token cap. W&B contains no media or table artifact.
- The tied and untied `outputs.jsonl` files are byte-identical (SHA-256 `75b670cf01b478e15c3d851d9635982ffdf40d3fc1a597b659c5aa81654f4168`).

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `configs/conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke.yaml` | yes; SHA-256 `84061eec386321617754b269d68651779b3963605433ef8882d9c231903f7c20` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` | yes; reviewed configuration pushed at `0f6cbcc` |
| Source snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke-20261016T040421.513908189Z`; remote path with the same basename under `/home/kristinelu/elf-runs/source/` | yes; local and remote both contain 43 files and 357807 bytes with relative SHA-256-manifest digest `c0c3c0d735a5074f130c1114979b759b3ea6c8b5ccefee3826ff3452fa9a881a` |
| Regional dataset | `/home/kristinelu/elf-data/conlangs-v1` | yes; deterministic Central2 dataset reverified with 15625 worlds and 125000 questions; manifest SHA-256 from the established record `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8` |
| Runtime | `/home/kristinelu/elf-runs/runtime/venv` | yes; pinned JAX 0.6.2 profile was reused from the verified tied arm |
| Checkpoint | `/home/kristinelu/elf-runs/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/checkpoint.msgpack` | yes; 27015423 bytes, SHA-256 `c403c7b004bf6d9b9768cc675ced782582fb46c875c20227ff6ee37368330a6f`, CPU restore verified step 20 |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/mx7bf6qz` | yes; state `finished`, summary/config identity verified, five log/metadata files and no media/table files |
| Metrics | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/metrics.jsonl` | yes; 4 lines, SHA-256 `ea006466efc49212d7646f6625c3cbe4fd1cff412c4cfcef2ea541eec790131d` |
| Generated outputs | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/outputs.jsonl` | yes; 2 lines, SHA-256 `75b670cf01b478e15c3d851d9635982ffdf40d3fc1a597b659c5aa81654f4168` |
| Effective config | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/config.yml` | yes; SHA-256 `939f86ba149749289e55a14adb8382aa63eccdebaed1bef21ee0e93406b53850` |
| Canonical launcher log | `experiment_log/2026-10-15/artifacts/20261015-conlang-jax-v4-8-n1024-s20-h128-out256-untied-g64-smoke/launcher.log` and deterministic gzip | yes; 48 lines, raw SHA-256 `f477768fecb25548e127f60b4b824333e98d3180b97b280865a02cf61efad5db`, gzip SHA-256 `b200ed2e2426259119fba445f7092731ab3264a248b8bbba6d7225ddb74b94b6` |

## Failure Details

- Last known step: `20`
- Error summary: not applicable
- Detailed output: canonical launcher log listed above; no fatal marker found

## Observations

- The mandatory fresh Kristine-only inventory selected this exact v4-8 again from its final `AVAILABLE` table after the tied arm completed.
- Remote preflight reported 86 GiB free for data and outputs, an absent run output path, and an idle TPU device.
- Correctness and cleanup reviews both returned no confirmed findings for the exact one-file untied configuration commit.
- JAX reported the TPU backend, four devices, 1,024 training examples, and 256 validation examples. The canonical log contains exactly one completion sentinel at step 20 and no fatal marker.
- After completion, the trainer and libtpu lock were absent, VFIO was idle, the TPU remained `READY` and `HEALTHY`, W&B was `finished`, and the checkpoint restored at step 20.
- The untied checkpoint is 4765128 bytes (21.42%) larger than the tied checkpoint, consistent with the additional encoder parameter and optimizer state tree.
- The repository has no `tpu_scripts/watch_elf_run.sh`; the preserved foreground session and direct terminal checks continuously covered this 200-second run.

## Interpretation

- This arm validates the untied-encoder width-128 pipeline and forms a controlled setup comparison with the completed tied arm. Twenty steps on a deterministic 1,024-example prefix are insufficient for a model-quality conclusion.
- The untied arm's small numerical advantage and 1.38% lower single-interval throughput are descriptive smoke results only. The byte-identical generated outputs reinforce that neither arm learned useful answer behavior within 20 steps.

## Updates

- `2026-10-16T04:05:12Z`: Both reviews passed, the config commit was pushed to PR 1, the existing runtime and regional dataset were reverified, the immutable snapshot matched locally and remotely, and remote preflight passed. Launch remains pending the final occupancy gate.
- `2026-10-16T04:06:33Z`: Final state and occupancy gates reported `READY`, `HEALTHY`, and `AVAILABLE`; launched the untied-encoder arm and registered W&B run `mx7bf6qz`.
- `2026-10-16T04:09:53Z`: Training completed at step 20. W&B, checkpoint restore, artifact hashes, terminal log, process absence, VFIO idleness, and TPU health were independently verified.
