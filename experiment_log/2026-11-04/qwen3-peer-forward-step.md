# Qwen3 peer forward-step prompt rerun

## Run Summary

| Field | Value |
|---|---|
| Run ID | forward-step-20261104-attempt-01 (selected, not launched) |
| Status | planned |
| Started | not recorded |
| Finished | not recorded |
| Baseline | pilot-e225874-attempt-02, source e225874c4f019b38947c91737d942834abaf45ae |
| Git revision | f6cf33edde5f4f2768eddab19ce85654fd96e19f (selected code/config revision) |
| Pull request | https://github.com/kristinelu921/conlang/pull/4 |
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
| Correctness | d14d4c0f49d52d0546a37d9bd2e85d087655fb0b | C1: configured compute cap was unused; both entry points now enforce it | Static producer/caller trace passed; remote 7200/7201/86400-second boundary tests pending authorization |
| Cleanup | f6cf33edde5f4f2768eddab19ce85654fd96e19f | NO_CONFIRMED_FINDINGS | Static diff/scope and producer/caller checks passed; remote tests pending authorization |

## Failure Details

Not applicable: no run has started.

## Observations and limitations

The previous v5e-8 allocation lasted approximately 62 minutes including setup,
tests, the failed numerical attempt, successful training, evaluation, and cleanup.
The rerun retains one seed, 32 trees, 16 MATH validation items, and 30 AIME items.
A small prompt ablation cannot establish general collaborative improvement.

## Updates

- 2026-11-04: prepared exact rerun; awaiting review and explicit compute authorization.

- 2026-11-04: source bucket location verified US-WEST4. Correctness review completed; added explicit configured-cap enforcement to supervisor and trainer before workload startup. Draft PR #4 opened.

- 2026-11-04: both required read-only reviews complete; C1 implemented, no cleanup findings. Coordinator syntax, scope, diff, and all three baseline split SHA-256/byte checks passed. Runtime readiness remains pending authorized remote tests and full-model gates. No new TPU or workload has been started.

- 2026-11-04: immutable snapshot prepared at `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-forward-step-20261104-f6cf33e-20261104T012506.552053195Z`; every peer_tree_rl source/config file verified byte-identical to selected revision f6cf33edde5f4f2768eddab19ce85654fd96e19f. Preparation receipt: `/tmp/qwen-peer-forward-step-preparation.json`. Compute authorization remains pending.

- 2026-11-04: user refined the intervention: each non-final message must share a concrete forward step and then ask the partner one focused question. Final submission remains allowed. Updated the prompt and its JSON example together; parser/reward/configuration are unchanged. Coordinator syntax and source-copy consistency checks passed. The two prior critic passes are retained; no third critic was run. A new source snapshot will supersede the previously prepared snapshot. Remote validation and compute authorization remain pending.

- 2026-11-04: selected source revision is now `42f532d1b011758027887bf5dd7fb47fe5bc2ccc`; new immutable snapshot: `/kmh-nfs-ssd-us-mount/staging/kristine/elf/qwen3-peer-forward-step-20261104-42f532d-20261104T012810.779100335Z`. Every source/config file matches the selected commit byte-for-byte. This supersedes the f6cf33e snapshot for launch without changing any scientific setting beyond the requested prompt refinement. No TPU use is authorized yet.

- 2026-11-04T01:30:18.228290+00:00: user authorized provisioning one new kristine v5-8 in response to the two-hour Spot v5e-8 approval request. Selected type v5litepod-8, zone us-west4-a, one active TPU maximum, budget 7200 seconds from allocation creation including tests, training and evaluation. Old pilot remains completed and its TPU deleted. Starting a single provisioning orchestrator with explicit COUNT=1; no development-machine experiment work.

- 2026-11-04T01:30:37.098297893Z: cloud accepted exact Spot resource `kmh-tpuvm-v5e-8-spot-kristine-04-01-30-32-976463421-0-us-west4-a`, v5litepod-8 topology 2x4, one VM, us-west4-a. Creation timestamp pins the immutable deadline `2026-11-04T03:30:37.098297893Z`. Direct API state initially CREATING. Provisioning log: `/tmp/qwen-peer-forward-step-provision.log`; one orchestrator only.

- 2026-11-04T01:34:00.199318+00:00: TPU creation succeeded, but the initializer failed closed at READY/UNHEALTHY_MAINTENANCE before runtime setup. Provisioning orchestrator exited 1 and has stopped. No additional TPU created. See `qwen3-peer-forward-step-initialization-01.md`; checking this same resource for natural health recovery.

- 2026-11-04T01:35:10.309335+00:00: exact cloud API reports the same creation timestamp and READY/HEALTHY after natural maintenance recovery. No reboot, replacement, process action, or second create. Starting fresh read-only final-table availability audit before retrying runtime initialization.

- 2026-11-04T01:39:32.156072+00:00: runtime retry completed successfully on t1v-n-4ade8c65-w-0. Freeze SHA-256 `0c68420bed5e26153bcb167c3e672c83c522c85f504477c7301a92602f6b4aa7` exactly matches the baseline runtime. All ten source/config hashes and three same-region frozen split files verified; Cloud SDK 463.0.0 with bundled Python 3.11.7 passed the exact recursive checkpoint-upload/readback probe. Source, data, and outputs are node-local. Runtime log `/tmp/qwen-peer-forward-step-initialize-02.log`; staging log `/tmp/qwen-peer-forward-step-staging.log`. Remote CPU tests started.

- Control-host orchestration note: Python 3.10 rejected the cloud nanosecond timestamp during receipt construction. Corrected parsing to microsecond precision, retaining the raw cloud timestamp in allocation metadata. The first SCP retried while the receipt was absent and then succeeded; no model process started and all remote bytes were subsequently independently verified. Deadline remains exactly 7200 seconds after the parsed allocation creation time.

- 2026-11-04T01:40:23.213065+00:00: all 23 remote CPU tests passed in 11.938 seconds on selected source 42f532d. Command: `JAX_PLATFORMS=cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 HF_HOME=/home/kristinelu/peer-tree/forward-step-20261104/cache /home/kristinelu/elf-runs/runtime/venv/bin/python -m unittest -v test_protocol test_backend test_lifecycle`, from the node-local experiment source directory. Evidence `/tmp/qwen-peer-forward-step-remote-tests.log` and remote `remote-tests-42f532d.log`. This resolves the review C1 runtime boundary check and frozen-split tests. Full-model and first-tree gates remain pending.

- 2026-11-04T01:42:00.745269+00:00: final dedicated occupancy probe returned AVAILABLE on t1v-n-4ade8c65-w-0, with no VFIO/lockfile holder; direct API was READY/HEALTHY, Spot, same creation timestamp. Started the exact planned node-local supervisor command with receipt deadline 1793763037.098297. Launch session 67302 retained. Started one run-specific 60-second read-only monitor (session 6873; `/tmp/qwen-peer-forward-step-monitor.py`), observing the peer supervisor PID, status, process command, cloud health and newest checkpoint. This custom peer supervisor uses run.py rather than the standard ELF train.py entry point. No duplicate trainer or recovery orchestrator. Full-model and first-tree gates are pending.

- 2026-11-04T01:43:09.826609+00:00: W&B initialized as https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/192ahtxm . Exact trainer PID 18209 and supervisor PID 18208 observed on the authorized remote host. Full-model numerical preflight passed at 2026-11-04T01:42:18.596350Z: maximum logit difference 2.288818359375e-05, tolerance 0.02, greedy next token agrees. Before-training evaluation is starting; no training result yet.

## Verified runtime gates and pre-training results — 2026-11-04T01:50:58.007359+00:00

```json
[
  {
    "event": "data_audit",
    "timestamp": "2026-11-04T01:41:20.984787+00:00",
    "reference_reasoning_exposed": false,
    "pretrained_contamination_audited": false,
    "source_checkpoint": "gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235/attempts/pilot-e225874-attempt-02/checkpoints/completed",
    "reused_frozen_splits": true,
    "maximum_initial_prompt_tokens": 664,
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
    },
    "train": 64,
    "validation": 16,
    "test": 30
  },
  {
    "event": "numerical_preflight",
    "timestamp": "2026-11-04T01:42:18.596350+00:00",
    "maximum_logit_difference": 2.288818359375e-05,
    "tolerance": 0.02,
    "greedy_next_token_matches": true,
    "reference_device": "remote_cpu"
  },
  {
    "event": "before_validation",
    "timestamp": "2026-11-04T01:43:26.498863+00:00",
    "correct": 2,
    "total": 16,
    "pass_at_1": 0.125,
    "generated_tokens": 4458,
    "terminal_reasons": {
      "invalid_schema": 5,
      "invalid_json": 3,
      "submitted": 3,
      "turn_limit": 2,
      "token_limit": 3
    }
  },
  {
    "event": "before_aime2024",
    "timestamp": "2026-11-04T01:44:29.361313+00:00",
    "correct": 0,
    "total": 30,
    "pass_at_1": 0.0,
    "generated_tokens": 8966,
    "terminal_reasons": {
      "token_limit": 7,
      "invalid_json": 11,
      "submitted": 3,
      "invalid_schema": 6,
      "turn_limit": 3
    }
  },
  {
    "event": "single_agent_baseline",
    "timestamp": "2026-11-04T01:46:43.152496+00:00",
    "correct": 2,
    "total": 30,
    "pass_at_1": 0.06666666666666667,
    "generated_tokens": 50131,
    "maximum_generated_tokens_per_problem": 2048,
    "comparison": "equal maximum generated-token budget, actual compute may differ"
  },
  {
    "event": "replay_gate",
    "timestamp": "2026-11-04T01:47:22.776330+00:00",
    "maximum_log_probability_difference": 6.866455078125e-05,
    "mean_log_probability_difference": 2.745388384126467e-07,
    "compared_tokens": 3835
  },
  {
    "event": "first_tree_gate",
    "timestamp": "2026-11-04T01:48:34.321133+00:00",
    "seconds": 98.92556673800004,
    "messages": 60,
    "optimizer_updates_finite": true,
    "checkpoint_verified": true,
    "maximum_peak_memory_fraction": 0.821671020806612,
    "remaining_seconds_upper_projection": 3066.692524517002
  }
]
```

The runtime freeze exactly matches the original pilot. The single-agent control reproduced 2/30 and 50,131 generated tokens; token-by-token comparison remains pending. Updated prompt pre-RL validation is 2/16 (original 1/16), while AIME remains 0/30. These are small single-seed samples. First-tree peak memory was 0.821671020806612, with finite optimizer updates and a verified durable checkpoint. The 98.9256-second first-tree interval includes compilation and checkpointing and is used only as a conservative budget bound, not steady-state throughput. A later 7-message update took 0.57275 seconds; steady timing remains consistent with the baseline update implementation. No final training effect is claimed yet.

## Completed run — 2026-11-04T02:04:54.862893+00:00
This timestamped result supersedes the earlier planned/running status. **Status: completed.** The scientific result is negative: the prompt yielded more training dialogue, but the trained agents lost the required conversation format and did not improve held-out accuracy.
- Exact run: `forward-step-20261104-attempt-01`; executed source `42f532d1b011758027887bf5dd7fb47fe5bc2ccc`; primary config `experiments/peer_tree_rl/forward_step.json`; PR https://github.com/kristinelu921/conlang/pull/4.
- Effective configuration matches the full configuration recorded above and the selected committed file exactly; no launch overrides. Seed 20261103. Exact supervisor invocation is the Planned Command above, executed from the node-local source directory.
- Allocation created 2026-11-04T01:30:37.098297893Z. W&B initialization began at 01:41:15 UTC; training ended at 01:58:51.047427 UTC; study terminal at 01:59:19.564970 UTC; supervisor exit 0 at 01:59:45.037939 UTC. Deadline was not reached. Training duration including checkpointing: 715.6518697019999 seconds; stopped at the configured 32-tree limit.
- Hardware: one Spot v5litepod-8, 2x4 topology, 8 chips, one VM, exact resource `kmh-tpuvm-v5e-8-spot-kristine-04-01-30-32-976463421-0-us-west4-a`; remote hostname `t1v-n-4ade8c65-w-0`. No development-machine experiment jobs.
- Remote runtime: JAX/JAXLIB 0.6.2, libtpu 0.0.17, Flax 0.10.5, Optax 0.2.8, Transformers 4.57.6, CPU Torch 2.2.2, datasets 5.0.1, W&B 0.28.1. Freeze SHA-256 `0c68420bed5e26153bcb167c3e672c83c522c85f504477c7301a92602f6b4aa7` exactly matches the original pilot.
### Final evaluation and baseline comparison
All scores below are correct final answers from one sampled rollout per fixed problem; higher is better. The paired runs use the identical frozen question bytes/order, pinned model and runtime, seed, and sampling settings. They differ in the collaboration prompt and in administrative budget/split-loading changes recorded above. No held-out outcome entered training or checkpoint selection.
| Prompt | Stage | MATH validation (16) | AIME 2024 (30) |
|---|---|---|---|
| Original | before RL | 1/16 (6.25%) | 0/30 (0%) |
| Original | after RL | 0/16 (0%) | 0/30 (0%) |
| Forward step + question | before RL | 2/16 (12.5%) | 0/30 (0%) |
| Forward step + question | after RL | 0/16 (0%) | 0/30 (0%) |

The updated prompt starts at 2/16 validation versus the original 1/16, a difference of one item (+6.25 percentage points); both finish at 0/16. The revised run drops 12.5 percentage points after RL. AIME remains 0/30 throughout both peer pilots. The pretrained single-agent control is 2/30 (0.06666666666666667) in both pilots, with exactly 50,131 generated tokens and **all 30 prompt-token arrays, output-token arrays, and correctness flags identical**. Control maximum output is 2048 tokens, matching the conversation maximum; actual compute differs.
| Training diagnostic | Original prompt | Forward step + question |
|---|---:|---:|
| Training trees | 32 | 32 |
| Distinct sampled messages | 117 | 466 |
| Generated training tokens | 17,641 | 40,448 |
| Successful leaves / leaves | 4 / 93 | 11 / 145 |
| Positive-credit messages | 5 | 17 |
| Positive-credit nonterminal messages | 1 | 6 |
| Negative-credit messages | 112 | 448 |
| Zero-credit messages | 0 | 1 |
| Optimizer updates A / B | 33 / 4 | 52 / 32 |

The numbers of leaves differ because invalid actions or submissions terminate branches; leaves share prefixes and are not independent benchmark samples. Six rewarded nonterminal messages are not six learned questions: several provide a solution or announce a plan without asking a follow-up question. This pilot does not establish causal usefulness or generalization of collaboration.
### Observed failure and a useful training example
All **46** post-RL evaluation conversations terminate on their first response: 41 hit the 256-token output limit, four fail JSON parsing, and one fails the schema. In particular, all 30 AIME outputs reach the token limit. Saved examples write free-form mathematical derivations instead of the required JSON action. This is a conversation-protocol failure as well as a scoring failure; the partner receives no turn.
The unchanged strict parser and outcome-only objective assign negative weight to 448 of 466 training messages. A plausible mechanism is that repeated negative sequence credit suppresses shared protocol tokens; the present experiment does not isolate that mechanism. The positive and negative counts describe the actual recorded weights, not a newly introduced reward or parser adjustment.
In `tree-0017.json`, candidate 0 / continuation 1 contains an actual successful exchange: A proposes pairing the alternating sum, B writes `(-1 + 2) + (-3 + 4) + ... + (-9999 + 10000)`, A derives 5000 pairs of sum 1 and asks “Is this correct?”, and B submits `5000`. Those continuation messages receive weight +1 from their successful terminal outcome. This demonstrates that the harness can assign credit to useful intermediate dialogue; it does not show that the final policy retained it.
### Verification and durable artifacts
- All 23 focused protocol/backend/lifecycle tests passed on the dedicated remote VM (11.938 seconds). Full-model logit parity, sampling/replay parity, finite updates, first-tree HBM/throughput gates, and checkpoint verification passed. No experiment tests ran on the development host.
- Canonical log is the single worker supervisor console. The trainer and supervisor exited; study terminal says completed, supervisor exit is 0, deadline-expired is false. Terminal status was checked independently of W&B and the foreground SSH result.
- The final cloud checkpoint was downloaded again on the remote VM and all **56** file sizes/SHA-256 hashes verified. Serialized optimizer step counts match the logged updates. Manifest SHA-256 `c88b8fc9b626977eee9c4fc2f0379809504a717fda07f04fb688f223ff92cd3a`.
- Canonical console: 464509 bytes; SHA-256 `56f7dbb986698f2aec99a80f8f6e99cdbf0558da09dfcb23c6d87f23fbbdc099`. Preserved under the cloud analysis path below.
- The 53 evidence files copied for local inspection were independently checked against the final manifest. No model evaluation was performed during local artifact inspection.
- Final checkpoint: `gs://kmh-gcp-us-west4/kristine/qwen3-peer-forward-step-20261104-0130/attempts/forward-step-20261104-attempt-01/checkpoints/completed` (verified). Both adapters and optimizer states, all split files, sampled output tokens/log probabilities, tree credit records, exact configuration, receipt, and metrics are saved. Full-vocabulary per-token logits were not retained.
- W&B: https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/192ahtxm (authenticated API state `finished`; group and final checkpoint link verified).
- Analysis archive: `gs://kmh-gcp-us-west4/kristine/qwen3-peer-forward-step-20261104-0130/attempts/forward-step-20261104-attempt-01/analysis/`; additional monitor/control evidence under sibling `control/`.
- Machine-readable results: `qwen3-peer-forward-step-42f532d-results.json` in this directory, copied without modification from the independently verified remote analysis.
- Comparison viewer: http://localhost:58615/comparison/ . It serves the exact two system prompts, paired score table, and all 308 saved training/evaluation records across the two pilots. HTTP page and JSON record count verified; server kept running. Rebuild script is archived under the cloud `control/` path.
- Remaining limitations: one seed, 32 training trees, 16 MATH validation items, 30 AIME items, no pretrained-contamination audit, strict protocol, and sparse successful outcomes. This result does not support claiming improved collective intelligence.

### Cleanup completed — 2026-11-04T02:04:38.064731+00:00

The final availability audit and a second exact-resource check confirmed the trainer and supervisor exited and every VFIO/lockfile resource was idle. Cloud creation timestamp matched the allocation receipt. Deleted only `kmh-tpuvm-v5e-8-spot-kristine-04-01-30-32-976463421-0-us-west4-a` in `us-west4-a`; deletion returned 0 and a fresh describe returned `NOT_FOUND`. Allocation lifetime was 2040.966434 seconds (34.02 minutes), including provisioning, maintenance recovery, tests, training, evaluation, verification, and cleanup; below the approved 7200-second cap. No other TPU or user's files/processes were modified. The run monitor exited after the terminal sentinel; the local comparison viewer remains running. Detailed cleanup evidence is archived under the cloud `control/qwen-peer-forward-step-cleanup.json` path.
