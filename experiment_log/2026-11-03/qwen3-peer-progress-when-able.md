# Qwen3 peer progress-when-able rerun

## Run Summary

| Field | Value |
|---|---|
| Run ID | progress-when-able-20261103-attempt-01 |
| Status | running |
| Started / Finished | not recorded |
| Baseline | forward-step-20261104-attempt-01, source 42f532d |
| Git revision / PR | not recorded; assigned before staging |
| Seed | 20261103 |
| Hardware | existing Spot v5litepod-8 kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a, us-west4-a |

User explicitly authorized this existing TPU for two hours including tests,
training, evaluation, and W&B reporting. One TPU maximum. No development-machine
experiment jobs. Read-only availability audit returned AVAILABLE / runtime READY.
Fresh identity/occupancy gates remain required before staging and launch.

## Configuration

Primary configuration: `experiments/peer_tree_rl/progress_when_able.json`.
Complete selected configuration (runtime effective copy pending):

```json
{
  "run_name": "qwen3-1p7b-peer-progress-when-able-20261103",
  "status": "prepared_not_launched",
  "model": "Qwen/Qwen3-1.7B",
  "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
  "enable_thinking": false,
  "compute_dtype": "float32",
  "agents": 2,
  "independent_lora_adapters": true,
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_targets": [
    "q_proj",
    "v_proj"
  ],
  "shared_scratchpad": false,
  "maximum_messages": 8,
  "maximum_prompt_tokens": 1024,
  "maximum_new_tokens": 256,
  "candidate_actions": 4,
  "continuations_per_candidate": 2,
  "branch_turns": [
    0,
    1
  ],
  "training_dataset": "EleutherAI/hendrycks_math",
  "training_dataset_revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
  "training_subset": "all_subjects_integer_final_answers",
  "training_problem_limit": 64,
  "validation_problem_limit": 16,
  "evaluation_dataset": "HuggingFaceH4/aime_2024",
  "evaluation_dataset_revision": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
  "evaluation_problem_limit": 30,
  "seed": 20261103,
  "temperature": 0.7,
  "top_p": 1.0,
  "learning_rate": 1e-05,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "gradient_clip_norm": 1.0,
  "ratio_clip": 0.2,
  "reward_baseline": 0.5,
  "reward_scale": 2.0,
  "reference_kl_coefficient": 0.01,
  "update_epochs_per_tree": 1,
  "maximum_training_trees": 32,
  "maximum_compute_seconds": 7200,
  "finalization_reserve_seconds": 1800,
  "maximum_active_tpus": 1,
  "requested_tpu_type": "v5litepod-8",
  "spot_required": true,
  "wandb_project": "conlangs",
  "wandb_entity": "kaynelu921-massachusetts-institute-of-technology",
  "wandb_group": "qwen3-1p7b-peer-progress-when-able-20261103",
  "frozen_data": {
    "directory": "/home/kristinelu/peer-tree/progress-when-able-20261103/baseline-data",
    "region": "us-west4",
    "source_checkpoint": "gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235/attempts/pilot-e225874-attempt-02/checkpoints/completed",
    "files": {
      "test.json": {
        "bytes": 13232,
        "sha256": "41ceb759c1db13f2c6b21f31e8662ceab0ba33c371682caef316676074e233ee"
      },
      "train.json": {
        "bytes": 19438,
        "sha256": "3ce686e6ed24125f9b6613d449b9496b66af1c8b0a36756df814b0f79afb28d7"
      },
      "validation.json": {
        "bytes": 4203,
        "sha256": "c9c30681221f06024752fbec021eb23be1c45fd16e679f38f2aed68dc1a62e0c"
      }
    }
  },
  "answer_matching": "integer_search"
}
```

## Changed Hyperparameters

| Parameter | Previous run | Current run |
|---|---|---|
| Prompt | progress plus mandatory follow-up question | continue solving when capable; question optional |
| answer_matching | strict (default) | integer_search |
| Run/group/data-local directory | forward-step identity | progress-when-able identity |

Temperature stays 0.7. Same pinned base model, fresh adapters, frozen splits,
seed, optimizer, 32 trees, branching and turn/token limits. Historical viewer
artifacts retain the actual previous prompt. The reward change was requested
previously and has already passed 21 remote CPU tests; this complete source
will receive fresh remote tests before training. Answer-search may reward an
incidental mention; strict metrics remain separately reported. This is not a
prompt-only ablation. Test answers never enter prompts or training selection.

## Command

From the immutable node-local experiment source directory:

```sh
TPU_LOG_DIR=/home/kristinelu/peer-tree/progress-when-able-20261103/tpu-logs JAX_PLATFORMS=tpu,cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/progress-when-able-20261103/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -u supervisor.py --config /home/kristinelu/peer-tree/progress-when-able-20261103/source/experiments/peer_tree_rl/progress_when_able.json --receipt /home/kristinelu/peer-tree/progress-when-able-20261103/execution-receipt.json --output /home/kristinelu/peer-tree/progress-when-able-20261103/progress-when-able-20261103-attempt-01
```

## Results

Not recorded; not launched. Remote protocol/backend/lifecycle tests and model
parity, replay, finite-update, HBM, throughput and durable-checkpoint gates pending.

## Artifacts

Configuration verified locally. Source snapshot, checkpoint and W&B URL not
recorded yet. Frozen data source and byte/hash manifest are pinned in config.

## Failure Details

Not applicable; no workload started.

## Observations and Interpretation

The previous viewer repeats shared-prefix nodes across paths; these are not
independent identical samples. Repetition also occurs among independently
sampled outputs. Mandatory questions and limited retained context are plausible
contributors, not isolated causal findings. This rerun changes the prompt and
requested scoring while retaining the no-history state.

## Updates

- Prepared and explicitly authorized on 2026-11-03 (America/New_York).

## Review ledger

| Pass | Reviewed base / head | Result | Verification |
|---|---|---|---|
| Correctness | 9cd6c6e / 0c7c33b | NO_CONFIRMED_FINDINGS | Read-only producer/caller/config/test trace; remote tests pending |
| Cleanup | 9cd6c6e / 0c7c33b | NO_CONFIRMED_FINDINGS | Diff/scope check; optional README grammar and strict-scoring qualification corrected |

Preflight found `/tmp/tpu_logs` owned by sqa:sqa mode 0775 and not writable
by kristinelu. No launch attempted and no collaborator files modified. Checking
for a supported run-owned log destination before proceeding. Runtime freeze
differs from the earlier pilot; exact core package compatibility remains a gate.

- Source selected: `ce8b4b305a4fcfeec555d01ea2a7082c3516f584`, pushed to PR
  https://github.com/kristinelu921/conlang/pull/4. Snapshot:
  `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-progress-when-able-ce8b4b3-20261104T041746.914878463Z`.
- The installed libtpu binary exposes `TPU_LOG_DIR`. Set it explicitly to the
  new run-owned mode-0700 directory; never use or modify the nonwritable
  collaborator-owned default. This is log-path plumbing only. Remote CPU tests
  and subsequent first-use TPU logs must confirm the selected path works.
- Budget begins with staging on the existing allocation at Unix
  `1793765897.0003126`, deadline `1793773097.0003126`. The legacy receipt field
  `allocation_created_unix_seconds` represents authorized-use start for this
  reused VM; its actual Cloud creation time is separately recorded.
- Core runtime: Python 3.10.18, JAX/JAXLIB 0.6.2, libtpu 0.0.17, Flax 0.10.5,
  Optax 0.2.8, Transformers 4.57.6, Torch 2.2.2+cpu, NumPy 1.26.4, W&B 0.28.1.

- Further producer trace confirms `Study.run` already sets `TPU_LOG_DIR` to
  `<output>/tpu-driver-logs` before importing the TPU backend. That run-owned
  directory is the effective training log destination (overriding the wrapper's
  preparatory directory). The default `/tmp/tpu_logs` is never selected by this
  harness; no source change or permission change is required.
- Source staging verified all 11 files and the three pinned same-region split
  hashes. Exact storage upload/readback passed.

- Fresh immutable-source remote suite passed: 30 tests, 11.534 seconds, zero
  failures. Command: `JAX_PLATFORMS=cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8
  OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/progress-when-able-20261103/cache
  TPU_LOG_DIR=/home/kristinelu/peer-tree/progress-when-able-20261103/tpu-logs
  /home/kristinelu/elf-runs/runtime/venv/bin/python -m unittest -v test_protocol
  test_backend test_lifecycle`, from the staged experiment source.
  Log: `/tmp/qwen-peer-progress-remote-tests.log`. No model test ran locally.

- Started remote supervisor and one 60-second read-only monitor at approximately
  2026-11-03 23:20 America/New_York. Exact trainer PID 38603, supervisor 38602,
  host t1v-n-4b587646-w-0. Fresh occupancy AVAILABLE and API READY/HEALTHY/Spot.
  W&B: https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/1f6jmcj2 .
  Model loading/numerical gates pending; process startup is not training success.
  Launch log `/tmp/qwen-peer-progress-launch.log`; recurring monitor
  `/tmp/qwen-peer-progress-monitor.py`, history `qwen-peer-progress-monitor-history.jsonl`.
