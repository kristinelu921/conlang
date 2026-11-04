# Qwen3 peer forward-step prompt rerun

## Run Summary

| Field | Value |
|---|---|
| Run ID | forward-step-20261104-attempt-01 (selected, not launched) |
| Status | planned |
| Started | not recorded |
| Finished | not recorded |
| Baseline | pilot-e225874-attempt-02, source e225874c4f019b38947c91737d942834abaf45ae |
| Git revision | not recorded; preparation commit pending |
| Pull request | not recorded; draft preparation pending |
| Seed | 20261103 |
| Hardware | proposed one new kristine Spot v5litepod-8 in us-west4-a; user approval pending |

The user requested the forward-step prompt edit and then requested a rerun.
They separately required explicit authorization before TPU use and prohibited
all development-machine experiment jobs. The previous TPU was deleted.
No TPU has been acquired or provisioned for this rerun. Read-only inventory
found one idle kristine v4-8 in us-central2-b (runtime missing), which does not
match the validated backend guard or the frozen data's region. No other owner's
items were inspected or modified.

## Configuration

Primary config: `experiments/peer_tree_rl/forward_step.json`.

```json
{
  "run_name": "qwen3-1p7b-peer-forward-step-20261104",
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
  "wandb_group": "qwen3-1p7b-peer-forward-step-20261104",
  "frozen_data": {
    "directory": "/home/kristinelu/peer-tree/forward-step-20261104/baseline-data",
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
  }
}
```

## Changed Hyperparameters

| Parameter | Baseline | Rerun |
|---|---|---|
| System prompt | collaboration instruction | add a concrete forward step on every turn |
| Split selection | seeded filtering and shuffle | reuse exact baseline train/validation/test bytes and order |
| Compute cap | 86400 seconds | proposed 7200 seconds |
| Finalization reserve | 3600 seconds | 1800 seconds |
| Run name / W&B group | qwen3-1p7b-peer-tree-pilot-20261103 | qwen3-1p7b-peer-forward-step-20261104 |

Only the prompt is a scientific intervention. Frozen split reuse prevents the
longer prompt from changing which training or validation questions are selected.
The new run starts from the same pinned pretrained weights with fresh adapters,
not the trained pilot checkpoint. Parser, reward, 256-token outputs, eight-turn
limit, branching, sampling, optimizer, and seed remain unchanged. Consequently,
strict-format failures may persist; no parser or reward fix is included.

## Planned Command

Run only on an authorized remote VM, from its immutable source directory:

```sh
JAX_PLATFORMS=tpu,cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/forward-step-20261104/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -u supervisor.py --config /home/kristinelu/peer-tree/forward-step-20261104/source/experiments/peer_tree_rl/forward_step.json --receipt /home/kristinelu/peer-tree/forward-step-20261104/execution-receipt.json --output /home/kristinelu/peer-tree/forward-step-20261104/forward-step-20261104-attempt-01
```

The immutable receipt will pin the approved exact resource, remote/control
hostnames, source hashes, creation time, and deadline. All three baseline split
files must be localized directly from the same-region checkpoint evidence and
verified against the hashes in the selected configuration. No dataset will be
copied from the control host or across regions.

## Results and verification

No new experiment results. Local source syntax inspection passed. Remote protocol,
backend, and lifecycle tests are pending compute authorization; do not run them
on the development machine. Full-model logit parity, sampling/replay parity,
finite updates, memory, throughput, and durable checkpoint gates must pass before
continuing training. Before/after validation and AIME metrics, raw conversation
outputs, and terminal evidence will be preserved and sent to W&B.

## Artifacts

- Baseline W&B: https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/voqv99rt
- New W&B run/checkpoint: not recorded; not launched.
- Primary configuration: `experiments/peer_tree_rl/forward_step.json`.
- Baseline split hashes: pinned in the primary configuration.

## Review ledger

| Pass | Reviewed head | Findings and resolution | Verification |
|---|---|---|---|
| Correctness | pending | pending | pending |
| Cleanup | pending | pending | pending |

## Failure Details

Not applicable: no run has started.

## Observations and limitations

The previous v5e-8 allocation lasted approximately 62 minutes including setup,
tests, the failed numerical attempt, successful training, evaluation, and cleanup.
The rerun retains one seed, 32 trees, 16 MATH validation items, and 30 AIME items.
A small prompt ablation cannot establish general collaborative improvement.

## Updates

- 2026-11-04: prepared exact rerun; awaiting review and explicit compute authorization.
