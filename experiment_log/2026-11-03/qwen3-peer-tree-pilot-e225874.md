# Qwen3 symmetric peer tree-RL pilot

## Run Summary

| Field | Value |
|---|---|
| Run ID | pilot-e225874-attempt-02 |
| Status | planned |
| Started | not recorded |
| Finished | not recorded |
| Baseline | Pretrained Qwen3-1.7B, identical two-peer conversation; separate single-agent baseline |
| Git revision | e225874c4f019b38947c91737d942834abaf45ae |
| Pull request | https://github.com/kristinelu921/conlang/pull/3 |
| Seed | 20261103 |
| Hardware | one Spot v5litepod-8, eight chips, one VM, us-west4-a |

Exact resource: `kmh-tpuvm-v5e-8-spot-kristine-03-22-35-29-776996979-0-us-west4-a`.
Fixed deadline: 2026-11-04T22:35:33.659878Z, 24 hours after allocation creation.

Exact remote command (from the source directory):
```sh
JAX_PLATFORMS=tpu,cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/pilot-20261103/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -u supervisor.py --config /home/kristinelu/peer-tree/pilot-20261103/source-e225874/experiments/peer_tree_rl/pilot.json --receipt /home/kristinelu/qwen-peer-receipt-e225874.json --output /home/kristinelu/peer-tree/pilot-20261103/pilot-e225874-attempt-02
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
  "compute_dtype": "float32",
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

22 remote tests passed in 11.588 seconds on the exact final source revision.
Full-model logits, replay, backward, memory, throughput and scientific metrics remain pending.
No inference, evaluation or training executed on the development machine.

## Artifacts

- Immutable staging snapshot: `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-tree-pilot-20261103-e225874-20261103T231258.179245430Z`.
- Remote exact source: `/home/kristinelu/peer-tree/pilot-20261103/source-e225874/experiments/peer_tree_rl`.
- Remote output: `/home/kristinelu/peer-tree/pilot-20261103/pilot-e225874-attempt-02` (must be absent before launch).
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

## Retry repair and verification

Retry 2 uses float32 model weights, activations and KV caches to investigate the
first attempt's bfloat16 cached/replay mismatch. This is a numerical backend
repair, with unchanged model, data, seed, reward, LoRA and optimizer settings.
All pretrained baselines rerun under the selected precision. No trained checkpoint
exists to resume: the prior attempt failed before any update. Its initial
zero-adapter checkpoint and complete failure evidence remain separately preserved.
The exact original 24-hour deadline is retained.

The retry also sets TPU_LOG_DIR to its fresh output's tpu-driver-logs directory
before JAX initialization, and automatically archives failure evidence. No foreign
logging directory or job is changed. The coordinator directly verified these
runtime repairs after the two required critic passes; no third critic was spawned.
Added tests compare sampled probabilities with full-message replay and preserve
failed-attempt artifacts without accepting partial optimizer states.

## Updates

- 2026-11-03T23:14:05.076353+00:00: source committed, pushed, and staged; 22 remote tests passed; source bytes verified identical to tested code. No optimizer update yet.

- 2026-11-03T23:15:08.164281+00:00: retry running on same TPU; trainer PID 29342, supervisor PID 29341. W&B https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/voqv99rt. Data audit exactly matches attempt 1. One monitor session 76789 checks this successor every minute; predecessor monitor terminated after its supervisor failure sentinel. Original deadline retained.

- 2026-11-03T23:20:01.787331+00:00: full-size cached/replay gate PASSED at tree 0 on 1,081 sampled tokens: maximum difference 4.57763671875e-05; mean 6.175504361557586e-07. Thresholds remain 0.1 and 0.02. First backward compilation is in progress. This supports numerical precision as the source of the earlier mismatch; no threshold relaxation was used.

- 2026-11-03T23:21:33.139954+00:00: first training tree accepted. Both peers completed finite optimizer updates. Peak device memory fraction 0.821671020806612; first-tree end-to-end time 91.679213979 s (includes first compile, replay gate and checkpoint). Conservative remaining-time projection 2842.0555815789976 s fits the original deadline and finalization reserve.
- Steady update microbatches at tree 2: 8 valid messages each, no gradient accumulation; agent 0 elapsed 0.5732005310001114 s, agent 1 0.5731529700001374 s. Approximately 13.96 actual messages/s per update interval; excludes rollout, compilation and checkpoint time. No prior accepted same-model backend throughput benchmark exists, so this is the initial pilot reference.
- tree-0000 and tree-0001 regional checkpoints verified. First three tree terminal-success rates are zero; training metrics are not test performance.

## Completed results — 2026-11-03T23:34:06.329174+00:00

Status: **completed**. Trainer and supervisor exited cleanly (supervisor exit 0).
W&B independently reports `finished`, group `qwen3-1p7b-peer-tree-pilot-20261103`,
last scalar step 134, and the verified final checkpoint URL.

| Evaluation | Before RL | After 32 trees | Direction |
|---|---:|---:|---|
| MATH validation conversation pass@1 | 1/16 (6.25%) | 0/16 (0%) | higher is better |
| AIME 2024 conversation pass@1 | 0/30 (0%) | 0/30 (0%) | higher is better |
| Pretrained single-agent AIME pass@1 | 2/30 (6.666666666666667%) | not trained | higher is better |

The single agent has the same 2,048-token maximum allowance, but actual generated
tokens differ: 50,131 single-agent tokens versus 6,577 peer tokens before RL and
6,917 after RL across the 30 AIME problems. This is not a measured FLOP-matched
comparison. All results are one fixed seed; the sample is too small for a broad
model-capability claim. No test-based checkpoint selection occurred.

### Training behavior

- Completed all 32 planned trees in 589.0615630800003 seconds including the first
  backward compilation and per-tree durable checkpointing.
- 117 distinct sampled messages; 17,641 generated tokens; 93 terminal leaves.
- 4/93 terminal leaves correct. Five messages received positive credit; 112
  received negative credit; none received zero credit. These are training-tree
  outcomes, not held-out pass@1.
- Agent A produced 93 messages and received 33 optimizer steps; B produced 24
  messages and received 4 optimizer steps. The action/prompt roles are symmetric,
  but A always opens each conversation, creating an exposure imbalance.
- Transition counts: 24 valid message transfers, 43 token-limit failures,
  18 submitted answers, 28 invalid JSON actions, and 4 invalid-schema actions.
- Only one nonterminal message received positive credit. It asked “What is the
  age of John?” and repeated the original age problem in its context. This is
  evidence that outcome credit can reinforce a preceding request; it does not
  establish that the request caused the success or decomposed the problem.
- Final AIME endings: 23 token-limit failures, 4 invalid JSON actions, and
  3 invalid-schema actions. No final AIME trajectory reached a valid submission.

### Interpretation

The implementation passed numerical, replay, gradient, memory, checkpoint and
recovery tests, but this pilot did not demonstrate a collaborative gain. Sparse
positive reward, a fragile 256-token JSON action interface, and unequal exposure
between the peers are concrete limitations of this setup. Outcome success alone
also rewards requests that merely restate an easy problem. A follow-up should
separate action validity from mathematical performance and counterbalance the
opening peer before increasing the training budget. A causal usefulness claim
would need a comparison with the message removed or replaced. These are
proposed follow-ups, not experiments run or improvements established here.

### Verified artifacts

- Summary: `experiment_log/2026-11-03/qwen3-peer-tree-pilot-e225874-results.json`.
- W&B: https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/voqv99rt
- Final checkpoint: `gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235/attempts/pilot-e225874-attempt-02/checkpoints/completed`.
- Downloaded this final checkpoint back onto the same-region VM and verified all
  57 listed files against their SHA-256 and byte counts. Both serialized optimizer
  steps matched the recorded metrics.
- Completion manifest SHA-256: `3bd08f94711b5dad850c169cd1a3ff0d83082b219ec1f857e96616820f1a6a00`.
- Canonical console SHA-256: `7a846b38368635e421bcb7ee24200d02b30c9ceed2f0217c329bf3f3d39efdd9`.
- Analysis script, summary, console, process identity and supervisor terminal
  sentinel preserved under the same attempt’s `analysis/` prefix.
- Raw generations, sampled token IDs and sampled-token log probabilities are
  preserved; full vocabulary logit vectors were not stored.

### Final checks

22 remote tests passed. Full-size independent logit maximum error was
2.288818359375e-05; cached/replay maximum error was 4.57763671875e-05; first-tree
peak memory fraction was 0.821671020806612. The predecessor’s bfloat16 failure
remains separately documented. No development-host model job was run.

## Cleanup — 2026-11-03T23:38:26.003458+00:00

After verifying the complete regional checkpoint/evidence copy, the trainer and
supervisor were absent and every VFIO resource was idle. Direct cloud describe
confirmed the exact same creation timestamp (2026-11-03T22:35:33.659878064Z).
The single TPU created for this pilot was then deleted successfully; the delete
command exited 0 and confirmed its exact resource name. No other TPU was deleted.

```sh
gcloud compute tpus tpu-vm delete kmh-tpuvm-v5e-8-spot-kristine-03-22-35-29-776996979-0-us-west4-a --project=he-vision-group --zone=us-west4-a --quiet
```

Both attempt monitors terminated after their respective supervisor sentinels.
Regional checkpoints, raw conversation evidence, logs, analysis and immutable
source records are retained. Total allocation lifetime was about 62 minutes,
within the user-authorized 24-hour maximum.
