# Qwen3 symmetric peer tree-RL pilot

## Run Summary

| Field | Value |
|---|---|
| Run ID | pilot-d282304-attempt-01 |
| Status | planned |
| Started | not recorded |
| Finished | not recorded |
| Baseline | Pretrained Qwen3-1.7B, identical two-peer conversation; separate single-agent baseline |
| Git revision | d28230441a5ded6ae26de58f6dbe12078daf3dd1 |
| Pull request | https://github.com/kristinelu921/conlang/pull/3 |
| Seed | 20261103 |
| Hardware | one Spot v5litepod-8, eight chips, one VM, us-west4-a |

Exact resource: `kmh-tpuvm-v5e-8-spot-kristine-03-22-35-29-776996979-0-us-west4-a`.
Fixed deadline: 2026-11-04T22:35:33.659878Z, 24 hours after allocation creation.

Exact remote command (from the source directory):
```sh
JAX_PLATFORMS=tpu,cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/pilot-20261103/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -u supervisor.py --config /home/kristinelu/peer-tree/pilot-20261103/source-d282304/experiments/peer_tree_rl/pilot.json --receipt /home/kristinelu/qwen-peer-receipt-d282304.json --output /home/kristinelu/peer-tree/pilot-20261103/pilot-d282304-attempt-01
```

## Configuration

Source: `experiments/peer_tree_rl/pilot.json`. No runtime hyperparameter overrides.

```json
{
  "run_name": "qwen3-1p7b-peer-tree-pilot-20261103",
  "status": "prepared_not_launched",
  "model": "Qwen/Qwen3-1.7B",
  "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
  "enable_thinking": false,
  "agents": 2,
  "independent_lora_adapters": true,
  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_targets": ["q_proj", "v_proj"],
  "shared_scratchpad": false,
  "maximum_messages": 8,
  "maximum_prompt_tokens": 1024,
  "maximum_new_tokens": 256,
  "candidate_actions": 4,
  "continuations_per_candidate": 2,
  "branch_turns": [0, 1],
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
  "learning_rate": 0.00001,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "gradient_clip_norm": 1.0,
  "ratio_clip": 0.2,
  "reward_baseline": 0.5,
  "reward_scale": 2.0,
  "reference_kl_coefficient": 0.01,
  "update_epochs_per_tree": 1,
  "maximum_training_trees": 32,
  "maximum_compute_seconds": 86400,
  "finalization_reserve_seconds": 3600,
  "maximum_active_tpus": 1,
  "requested_tpu_type": "v5litepod-8",
  "spot_required": true,
  "wandb_project": "conlangs",
  "wandb_entity": "kaynelu921-massachusetts-institute-of-technology",
  "wandb_group": "qwen3-1p7b-peer-tree-pilot-20261103"
}
```

## Changed Hyperparameters

The within-run baseline uses pretrained zero-initialized LoRA residuals. Treatment
updates separate rank-16 q/v LoRA adapters with signed downstream outcome credit.
All before/after evaluation prompts, seeds, sampling temperature, and limits are fixed.
The single-agent comparison has the same maximum generated-token budget; actual compute differs.

## Results

20 remote tests passed in 10.676 seconds on the exact final source revision.
Full-model logits, replay, backward, memory, throughput and scientific metrics remain pending.
No inference, evaluation or training executed on the development machine.

## Artifacts

- Immutable staging snapshot: `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-tree-pilot-20261103-d282304-20261103T230119.788270040Z`.
- Remote exact source: `/home/kristinelu/peer-tree/pilot-20261103/source-d282304/experiments/peer_tree_rl`.
- Remote output: `/home/kristinelu/peer-tree/pilot-20261103/pilot-d282304-attempt-01` (must be absent before launch).
- Regional durable prefix: `gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235`.
- W&B: not recorded; initialization pending.
- Checkpoint: none yet.

## Review ledger

| Pass | Reviewed head | Findings and resolution | Verification |
|---|---|---|---|
| Correctness | 1bb4161050b534ae632c2aa0716ec9adf15f7e21 | C1 resume, C2 durable evidence, C3 ancestry: fixed | Restored optimizer next update exactly matches; corrupted evidence and changed deadline rejected; all parent IDs resolve |
| Cleanup | 77926030ce5c9e259c32560cf98674d4715d5802 | CL-01 unsupported settings rejected; CL-02 unused mask/helper/imports removed | Contradictory configs fail; real replay masks and overflow tested; 20 remote tests pass |

Both required critic passes completed. Final coordinator diff/scope check passed.
Full-size runtime gates remain pending; draft PR retained until observed execution.

## Failure Details

No scientific run failure yet. Provisioning maintenance-health failure and recovery
are recorded separately. Initial small tests passed.

## Observations and limitations

One small feasibility pilot, 32 trees, integer-answer MATH only, 30 AIME 2024 items.
Exact normalized question overlap is excluded; pretrained contamination is not audited.
Training-tree success is not held-out pass@1 or a causal estimate of message utility.

## Updates

- 2026-11-03T23:02:58.479862+00:00: final source tested, committed, pushed, and staged; full run pending fresh occupancy gate.

- 2026-11-03T23:05:16.681845+00:00: process started: trainer PID 21765, supervisor PID 21764; status running/data_preparation. W&B https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/bco186i7. Exact cloud health READY/HEALTHY. One control monitor checks cloud state, exact process, metrics, and checkpoint every 60 seconds; monitor session 67234.
- Data audit: 7,500 MATH source training rows; 2,787 non-integer answers; 1 overlap/duplicate; 41 prompts too long; 4,671 eligible questions. Seeded fixed split 64 train / 16 validation; AIME 30 test. Reference reasoning not exposed.
- Diagnostic logging warning: pre-existing /tmp/tpu_logs is sqa:sqa mode 0775. Left untouched. Google documents failed log-directory creation as nonfatal (https://docs.cloud.google.com/tpu/docs/troubleshooting/trouble-tf); console is preserved under this exact attempt. Future attempts must set TPU_LOG_DIR to a fresh task-owned path before JAX initialization. The initial fresh preflight missed this directory check; recorded for correction.

- 2026-11-03T23:11:53.207351+00:00: FAILED before optimizer step 1. Worker-0 terminal sentinel reports cached/replay log-probability mismatch: maximum 1.2847671508789062, mean 0.013021769228486496 (limits 0.1 / 0.02). Supervisor exit 1; exact trainer PID no longer exists; TPU remains READY/HEALTHY, fresh inventory AVAILABLE/READY, no /tmp/libtpu_lockfile. One host is canonical process zero. No weights updated.
- Full-model independent logit gate passed with maximum difference 2.288818359375e-05 and matching greedy next token. Before validation 1/16; before AIME 0/30; separate single-agent AIME 3/30, 48,696 generated tokens. These are bfloat16 failed-attempt baselines, not training improvements.
- First tree: 17 distinct messages, 1,341 generated tokens, rollout 3.082521614000143 seconds, zero terminal successes. Replay failed before the first update. Latest verified checkpoint is initial, zero completed training trees.
- Failure evidence and console successfully copied to `gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235/attempts/pilot-d282304-attempt-01/failure-evidence/`. Console SHA-256 `9b29477c2ba95c16b828f8d52c31042c0af4247f7e2e7a99df55a1dc43ca16da`. Earlier initial checkpoint/evidence mirror also verified.
- Root cause hypothesis: bfloat16 numerical divergence between cached token decoding and full-message scoring. Full float32 retry selected to test this hypothesis without loosening the threshold; all baselines will be regenerated. No preemption, process collision, dataset error, or model-logit parity failure was observed.
- Repair also gives the retry its own TPU_LOG_DIR before runtime initialization and archives failure evidence automatically without labeling partial optimizer state complete. No foreign directory or job was modified.
