# Qwen3 two-peer tree RL pilot preparation

## Run Summary

| Field | Value |
|---|---|
| Run ID | qwen3-1p7b-peer-tree-pilot-20261103 |
| Status | planned |
| Started | not recorded; no experiment launched |
| Finished | not recorded |
| Baseline | pretrained Qwen/Qwen3-1.7B under the same conversation protocol |
| Git revision | not recorded; preparation in progress |
| Command | not recorded; runtime implementation pending |
| Seed | 20261103 |
| Hardware | none acquired; intended single kristine Spot v5e-8 |

## Configuration

Complete proposed configuration: ../../experiments/peer_tree_rl/pilot.json
(repository path: experiments/peer_tree_rl/pilot.json). Unresolved Hugging Face
revision fields must be pinned before download or execution. Protocol:
experiments/peer_tree_rl/README.md.

## Changed Hyperparameters

This is a new experiment. No changes to the completed SmolLM2 study or its
artifacts. Proposed treatment adds two separate rank-16 LoRA adapters, symmetric
communication, and downstream-success policy-gradient updates.

## Results

No model results, tests, or accelerator jobs have run.

## Artifacts

Prepared protocol and configuration exist locally. No checkpoint, W&B run,
remote source snapshot, or dataset download exists for this pilot.

## Failure Details

- Read-only inventory first failed locally with snap-confine capability error;
  rerunning with authorized network/process access succeeded.
- Exact inventory command: `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`.
- Final AVAILABLE table: none. Only discovered kristine TPU was
  `kmh-tpuvm-v6e-16-spot-kristine-05-15-33-35-N-us-central1-b`, state DELETING.
- Exact describe of the previous authorized v5e-8
  `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`
  returned NOT_FOUND. No provisioning, claims, deletion, signals, or foreign
  resource inspection were performed.

## Observations

Current launch authorization covers an existing available kristine resource;
none is currently available. The prior SmolLM2 resource no longer exists.
Source implementation and remote validation remain pending.

## Authorization update

The user explicitly approved provisioning one new kristine Spot v5e-8 and
replaced the proposed one-hour cap with up to 24 hours: "if u are able to
provision, you may use it for 24 hors". At most one active TPU remains the
capacity limit. Provisioning will use a task-specific copy of the repository
query script with its notification URL changed from the shared KMH topic to
the user-authorized kluping topic; the original script remains untouched.

## 2026-11-03 23:01 UTC update

The authorized Spot v5e-8 was created at 2026-11-03T22:35:33.659878064Z.
The fixed allocation deadline is 2026-11-04T22:35:33.659878064Z.
Maintenance health recovered without mutation; the exact-resource AVAILABLE
audit passed. Runtime initialization then succeeded. No second TPU was created.

- Exact resource: `kmh-tpuvm-v5e-8-spot-kristine-03-22-35-29-776996979-0-us-west4-a`.
- Remote hostname: `t1v-n-a09199da-w-0`; control hostname: `t1v-n-21fe7514-w-0`.
- Runtime: `/home/kristinelu/elf-runs/runtime/venv/bin/python`, Python 3.10.18.
- Frozen dependencies SHA-256: `0c68420bed5e26153bcb167c3e672c83c522c85f504477c7301a92602f6b4aa7`.
- Regional bucket `gs://kmh-gcp-us-west4` verified US-WEST4; the pilot dataset prefix was absent.
- Same-region write/read probe passed under `gs://kmh-gcp-us-west4/kristine/qwen3-peer-tree-pilot-20261103-2235/setup/storage-probe.txt`.
- Initial remote tests: 16 passed in 10.639 s. Recovery/masking/evidence tests: 20 passed in 10.728 s.
- Latest exact test command on remote VM: `cd /home/kristinelu/peer-tree/pilot-20261103/source-v2 && JAX_PLATFORMS=cpu USE_TORCH_XLA=0 OMP_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m unittest -v test_protocol test_backend test_lifecycle`.
- Draft PR: https://github.com/kristinelu921/conlang/pull/3.
- Configuration/source head: `77926030ce5c9e259c32560cf98674d4715d5802`.
- Primary configuration: `experiments/peer_tree_rl/pilot.json`.
- Full-size inference, training, and benchmark metrics remain pending.

### Frozen effective pilot configuration

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
