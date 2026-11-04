# Qwen3 peer one-step prompt rerun

## Run Summary

| Field | Value |
|---|---|
| Run ID | one-step-20261103-attempt-01 |
| Status | completed |
| Started / finished | approximately 2026-11-03 23:47 / 2026-11-04 00:03 America/New_York; exact event timestamps retained |
| Baseline | progress-when-able-20261103-attempt-01, source ce8b4b3 |
| Git revision / PR | 29f2609518ae4027cb7fd882b2222ab4c6b625a8 / https://github.com/kristinelu921/conlang/pull/4 |
| Seed | 20261103 |
| Hardware | existing authorized kristine Spot v5litepod-8, us-west4-a |

The user requested another run after changing the prompt to exactly one concrete
forward step followed by a handoff. Reuse the existing authorized resource
`kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`
within the original two-hour budget: start 1793765897.0003126, hard deadline
1793773097.0003126. No budget reset, provisioning, extra TPU, or development-host
experiment jobs. Fresh availability, ownership, source and runtime gates apply.

## Configuration

Selected primary configuration: `experiments/peer_tree_rl/one_step.json`.

```json
{
  "run_name": "qwen3-1p7b-peer-one-step-20261103",
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
  "wandb_group": "qwen3-1p7b-peer-one-step-20261103",
  "frozen_data": {
    "directory": "/home/kristinelu/peer-tree/one-step-20261103/baseline-data",
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
| Prompt | continue while capable | exactly ONE concrete step, then hand off |
| Run/group/node-local paths | progress-when-able identity | one-step identity |

Temperature 0.7, integer-search scoring, strict diagnostic scoring, pinned base
model with fresh adapters, frozen splits, seed, 32 trees, branch factor, turn and
token limits, and optimizer are unchanged. This is a prompt-only scientific
change relative to the immediate previous run. Historical outputs are preserved.

## Command

From the immutable node-local source directory:

```sh
JAX_PLATFORMS=tpu,cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/one-step-20261103/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -u supervisor.py --config /home/kristinelu/peer-tree/one-step-20261103/source/experiments/peer_tree_rl/one_step.json --receipt /home/kristinelu/peer-tree/one-step-20261103/execution-receipt.json --output /home/kristinelu/peer-tree/one-step-20261103/one-step-20261103-attempt-01
```

## Results

Not recorded. Fresh remote tests and runtime gates pending.

## Artifacts

Selected configuration exists; immutable snapshot, W&B URL and checkpoint pending.

## Failure Details

Not applicable; not launched.

## Observations and Interpretation

Previous run produced only one valid handoff and no held-out improvement.
This rerun tests whether an explicit one-step instruction improves handoffs.
Answer-search may reward incidental mentions; strict metrics remain separate.

## Updates

- Prepared 2026-11-03 America/New_York under the unchanged authorized deadline.

## Review ledger

| Pass | Base / head | Findings | Verification |
|---|---|---|---|
| Correctness | 2cc2bb8 / 29f2609 | NO_CONFIRMED_FINDINGS | Prompt/state/generation/replay and persisted-budget trace; remote tests pending |
| Cleanup | 2cc2bb8 / 29f2609 | NO_CONFIRMED_FINDINGS | Full diff, config closure and documentation inspection |

- Selected source `29f2609518ae4027cb7fd882b2222ab4c6b625a8` pushed to PR #4.
  Immutable snapshot:
  `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-one-step-29f2609-20261104T044513.114998020Z`.
- Fresh inventory reports the exact authorized v5e-8 AVAILABLE with runtime READY.
  Cloud creation timestamp matches the previous receipt; READY/HEALTHY/Spot.
- Deadline remains 1793773097.0003126. Separate source, output, checkpoint and
  W&B identities preserve the completed prior run.

- All 30 immutable-source remote CPU tests passed in 11.534 seconds. Full command
  is the planned environment with `JAX_PLATFORMS=cpu` and
  `python -m unittest -v test_protocol test_backend test_lifecycle`; evidence
  `/tmp/qwen-peer-one-step-stage-tests.log`. Source and same-region dataset
  hashes and storage upload/readback verified by staging.
- Final occupancy AVAILABLE, API READY/HEALTHY/Spot with same Cloud creation
  timestamp. Started the exact planned supervisor command and one 60-second
  monitor (`/tmp/qwen-peer-one-step-monitor.py`). Numerical/first-update gates
  pending; no new allocation or deadline extension.

## Final verified results

All 32 trees completed with stop reason `tree_limit`; supervisor exit 0 and
`deadline_expired=false`. W&B is finished, trainer exited, and final VFIO
occupancy is AVAILABLE. No new TPU was provisioned; the existing authorized
allocation remains intact and idle. The original deadline was not reset.

| Metric (higher is better) | Before RL | After RL |
|---|---:|---:|
| MATH validation answer-search | 7/16 (43.75%) | 8/16 (50%) |
| MATH validation strict pass@1 | 1/16 (6.25%) | 1/16 (6.25%) |
| AIME 2024 answer-search | 0/30 | 0/30 |
| AIME 2024 strict pass@1 | 0/30 | 0/30 |

Single-agent control: 2/30 under both rules, 50,131 generated tokens.

| Training diagnostic | Progress-when-able | Exactly one step |
|---|---:|---:|
| Trees | 32 | 32 |
| Distinct sampled messages | 82 | 196 |
| Agent A / B messages | 80 / 2 | 122 / 74 |
| Valid handoffs (`reason=message`) | 1 | 84 |
| Terminal leaves | 81 | 112 |
| Answer-search rewarded leaves | 27 | 48 |
| Positive-credit message actions | 0 | 42 |
| Optimizer updates A / B | 32 / 1 | 37 / 13 |

The one-step prompt produces substantially more protocol-valid handoffs in this
fixed-seed pilot. That is evidence of increased participation, not proof of
greater combined intelligence. RL adds one answer-search hit on 16 validation
questions; strict validation is unchanged and AIME remains zero. Search can
reward incidental mentions. Do not treat shared tree leaves as independent
benchmark samples or count every positive message action as a useful question.

Training transitions: 84 messages, 26 submitted, 41 token-limit, 29 invalid-JSON,
eight invalid-schema, and eight turn-limit endings. Format and length failures
therefore remain material. After RL, validation ends with nine token-limit,
five invalid-JSON and two submitted responses; AIME has 27 token-limit and three
invalid-JSON responses.

## Verification and artifacts

- All 30 remote CPU tests passed in 11.534 seconds. No experiment tests ran on
  the development host.
- Full-model logit parity passed (max difference 2.288818359375e-05, tolerance
  0.02; greedy next token matches). Sampling/replay max difference
  1.9073486328125e-05 across 921 tokens. Finite-update, memory and checkpoint
  gates passed; peak device-memory fraction 0.821671020806612.
- Training phase elapsed 603.103109906001 seconds. Final supervisor time was
  1793768605.9431427, earlier than the unchanged deadline 1793773097.0003126.
- All 56 checkpoint files independently redownloaded and size/SHA-256 verified
  on the remote VM. All 53 local evidence files match the durable manifest.
  One transient SCP transport error retried automatically; exit was zero and
  all hashes passed. This was not a model or training failure.
- W&B (verified finished): https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/rqv0b3tt
- Durable checkpoint (verified): `gs://kmh-gcp-us-west4/kristine/qwen3-peer-one-step-20261104-0445/attempts/one-step-20261103-attempt-01/checkpoints/completed`
- Complete metrics/gates: `qwen3-peer-one-step-29f2609-results.json`.
- Raw local evidence: `/tmp/qwen-peer-one-step-artifacts/evidence`.
- New exact-input/output viewer: http://localhost:60805/one-step-trees/ .
- Previous viewers and artifacts preserved.
- Remote console, summary, test log and verification helper archived under
  the checkpoint's sibling `analysis/` prefix.
- The single read-only monitor exited on the supervisor terminal sentinel.

## Remaining limitations

One seed, 32 training questions, 16 validation questions and 30 AIME problems;
no pretrained contamination audit, no reliable semantic verifier, and a fixed
256-token message limit. More conversation does not establish useful
collaboration. No further run was launched.
