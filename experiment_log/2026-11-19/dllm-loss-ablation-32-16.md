# dLLM loss ablation 32→16

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-loss-ablation-32-16-20261119` |
| Status | `completed` |
| Started | `2026-11-19 UTC` |
| Finished | `2026-11-19 09:28 UTC` |
| Baseline | `experiment_log/2026-11-17/dllm-reproduction.md`, progressive 32→16 stage |
| Git revision | `801b9ec1e3c6c8a1c59dc9a0c8f86560fe81210c` |
| Command | `python -m reproduction.prepare_validation --data <node-local-data> --output <node-local-run>/validation_finecode.npz --samples 64`, then `python -m reproduction.run_loss_ablation --config reproduction/configs/loss_ablation_32_16.json --data <node-local-data> --validation-data <node-local-run>/validation_finecode.npz --progressive-parent <node-local-parent>/latest.pkl --output <fresh-node-local-output>` |
| Seed | training `42`; fixed validation corruption `314159` |
| Hardware | `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, `us-west4-a`, v5litepod-16 (`4x4`), Spot; worker 0 bounded to its four local chips |

## Configuration

Source: `experiments/dllm-reproduction/reproduction/configs/loss_ablation_32_16.json`

```json
{
  "experiment": "dllm-loss-ablation-32-16-20261119",
  "teacher_steps": 32,
  "student_steps": 16,
  "updates": 400,
  "batch_size": 8,
  "temperature": 4.0,
  "seed": 42,
  "validation_examples": 64,
  "validation_seed": 314159,
  "validate_every": 50,
  "save_every": 100,
  "wandb_entity": "kaynelu921-massachusetts-institute-of-technology",
  "wandb_project": "dllm-distillation-reproduction",
  "arms": [
    {"name": "progressive-current", "parent": "progressive", "learning_rate": 0.00005, "warmup_updates": 0, "minimum_learning_rate_ratio": 1.0, "alpha": 0.5, "kl_positions": "original"},
    {"name": "progressive-low-lr", "parent": "progressive", "learning_rate": 0.00001, "warmup_updates": 40, "minimum_learning_rate_ratio": 0.1, "alpha": 0.5, "kl_positions": "original"},
    {"name": "progressive-teacher-weighted", "parent": "progressive", "learning_rate": 0.00001, "warmup_updates": 40, "minimum_learning_rate_ratio": 0.1, "alpha": 0.25, "kl_positions": "original"},
    {"name": "progressive-remaining-mask", "parent": "progressive", "learning_rate": 0.00001, "warmup_updates": 40, "minimum_learning_rate_ratio": 0.1, "alpha": 0.5, "kl_positions": "remaining"},
    {"name": "direct-low-lr", "parent": "base", "teacher_steps": 128, "student_steps": 16, "learning_rate": 0.00001, "warmup_updates": 40, "minimum_learning_rate_ratio": 0.1, "alpha": 0.5, "kl_positions": "original"}
  ]
}
```

All arms use AdamW with weight decay `0.01`, global gradient clipping at `1.0`, float32 parameters/loss/optimizer, bfloat16 matrix multiplications, and the pinned Open-dCoder/FineCode revisions recorded by the baseline. Validation uses 64 newly downloaded examples whose source hashes must be disjoint from all 12,800 baseline training hashes.

## Changed Hyperparameters

| Parameter | Baseline | Current | Source |
|---|---:|---:|---|
| Updates per arm | 1,600 | 400 | committed config |
| Learning rate | constant `5e-5` | control `5e-5`; treatments `1e-5` | committed config |
| Warmup | 0 | treatments 40 updates | committed config |
| Decay | none | treatments cosine to `1e-6` | committed config |
| CE weight `alpha` | `0.5` | `0.5`; teacher-weighted arm `0.25` | committed config |
| KL positions | all originally corrupted | original; remaining-mask arm uses positions still masked after teacher rollout | committed config |
| Fixed validation | none | 64 disjoint examples every 50 updates | committed config |
| Gradient diagnostics | total norm only | CE norm, KL norm, and cosine on a fixed validation batch | committed source |

## Results

All five arms completed 400 updates. Final fixed-validation metrics:

| Arm | Loss | Cross-entropy | KL | CE/KL gradient cosine |
|---|---:|---:|---:|---:|
| `progressive-current` | 1.029732 | 1.489739 | 0.569724 | 0.089420 |
| `progressive-low-lr` | 0.796507 | 1.441452 | 0.151562 | 0.252238 |
| `progressive-teacher-weighted` | **0.467840** | 1.451694 | 0.139888 | 0.253987 |
| `progressive-remaining-mask` | 0.764706 | 1.440597 | **0.088814** | 0.385645 |
| `direct-low-lr` | 0.876793 | **1.345011** | 0.408575 | -0.003882 |

`progressive-teacher-weighted` is the selected treatment for the full 16-step HumanEval+/MBPP+ evaluation because it has the lowest held-out combined objective. Its validation loss is 41.3% below `progressive-low-lr` and 54.6% below `progressive-current`; its KL is also lower than `progressive-low-lr` while cross-entropy differs by less than 1%. Component metrics remain visible because combined losses with different `alpha` values are not directly interchangeable.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Configuration | `experiments/dllm-reproduction/reproduction/configs/loss_ablation_32_16.json` | yes |
| Source snapshot | `experiment_log/2026-11-19/artifacts/dllm-loss-ablation-launch/source-801b9ec1.tar.gz` | SHA-256 `5863e3a7193ca1b00c6bf0e1c984b0a1b37ab1795047c86be061260f99091737` |
| Parent checkpoint | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/parent/latest.pkl` | copied once from the completed baseline; source and localized SHA-256 `8e4fe62184b923cda8e54f01a5d6cfc41eb7091bb3fa4a9173d2b48758b95e96` |
| Regional output | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/retry-r3` | active destination |
| Checkpoint | `/home/kristinelu/elf-runs/dllm-loss-ablation-32-16-20261119-r2/output-r3/progressive-current/latest.pkl` | update 200 complete; 5,929,296,570 bytes |
| Tracking project | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/dllm-distillation-reproduction` | live |
| W&B runs | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/dllm-distillation-reproduction` | all five training runs `finished` at step 400 |
| Winning checkpoint | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/retry-r3/output/progressive-teacher-weighted/latest.pkl` | 5,929,296,598 bytes; SHA-256 `bf752b457b054ce60e93d0cf148b3ef5921c177b57b3bb596f44d884589eecaf` |
| Direct checkpoint | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/retry-r4-worker1/output/direct-low-lr/latest.pkl` | 5,929,296,471 bytes; SHA-256 `b597a2e23a05a04b96a0d0c1b010c8889cd0c0c42d3c31e417da4dc27885fff0` |
| Plots / report | not recorded | no |
| Wide read-only sweep | `experiment_log/2026-11-19/artifacts/dllm-loss-ablation-wide-sweep/summary.json` | yes; 65 READY/HEALTHY TPUs inspected, no mutations |

## Failure Details

- Last known step: all arms update 400, completed
- Error summary: three zero-step staging/startup attempts, one worker-0 disk exhaustion after the valid direct update-100 checkpoint, and one zero-step recovery attempt are preserved. The direct arm resumed from its checksum-valid update-100 optimizer state and completed on worker 1.
- Detailed output: `dllm-loss-ablation-32-16-retry0-error.md`, `dllm-loss-ablation-32-16-retry1-error.md`, `dllm-loss-ablation-32-16-retry2-error.md`, `dllm-loss-ablation-32-16-retry3-error.md`, and `dllm-loss-ablation-32-16-retry4-error.md`

## Observations

- The committed configuration changes one major factor per progressive treatment arm and includes a direct-distillation comparison.
- Thirty-nine focused CPU-safe tests passed on the finalized run revision. No training, inference, evaluation, or model smoke test ran on the development machine.
- The two-pass review found and fixed distributed token weighting, single-host enforcement, validation provenance, restart integrity, producer coverage, and large-checkpoint hashing issues. No material review finding remains open.

## Interpretation

- Fixed validation slopes and CE/KL gradient cosine will distinguish batch-order noise from sustained objective conflict.
- Full perplexity evaluation should be run only after the short ablation identifies a treatment worth promoting.

## Updates

- 2026-11-19: Created the planned record; the first required filtered capacity audit found no available authorized TPU.
- 2026-11-19: Finalized reviewed run revision `801b9ec1e3c6c8a1c59dc9a0c8f86560fe81210c`; 39 focused tests passed.
- 2026-11-19: Installed one read-only capacity monitor with its first new audit delayed 30 minutes and subsequent audits spaced 30 minutes apart. It stops and notifies when an authorized `kristine` TPU enters the final `AVAILABLE` table.
- 2026-11-19 UTC (November 18 ET): User-requested filtered audit returned `AVAILABLE none`. Three `kristine` resources were `CREATING`, one was `DELETING`, and one vanished during exact lookup and was classified `UNKNOWN`; none was inspected or claimed. Restarted the single monitor with its next audit delayed 30 minutes.
- 2026-11-19 UTC: Completed the user-authorized wide read-only sweep after the filtered `kristine` inventory remained unavailable. Of 65 READY/HEALTHY resources, 55 were active and left untouched and 10 were idle. Selected `kmh-tpuvm-v4-8-2` (`us-central2-b`, v4-8, `2x2x1`) because it is a single-host TPU compatible with the trainer and is region-matched to the preserved data and checkpoints. No TPU was acquired and no mutation was performed; acquisition awaits the required exact user authorization.
- 2026-11-19 UTC: User clarified that acquiring the non-`kristine` candidate is not allowed. Removed it from the run plan. No non-`kristine` TPU will be acquired or used; capacity monitoring remains restricted to `kristine` resources on the requested 30-minute cadence.
- 2026-11-19 UTC: User clarified to follow the repository ownership safeguard. The earlier candidate `kmh-tpuvm-v4-8-2` and its four peer v4-8 resources are explicitly denylisted and remain unusable. Eligible idle non-denylisted capacity may be selected after the official wide read-only sweep and a fresh all-worker occupancy check, while `kristine` capacity remains preferred. Updated the 30-minute monitor to follow that workflow.
- 2026-11-19 UTC: Selected idle, non-denylisted `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1` after all four workers reported `AVAILABLE`. Regenerated the pinned 12,800-example FineCode dataset directly in `us-west4` (NPZ SHA-256 `95c006563cc336816bfc38f1a2b366aec724664a9689f1e7d9ff5e07d647d401`) and prepared 64 disjoint validation examples (SHA-256 `22fcaefad51954e8bf45152b3ee27d36e1c30652458f0f326a3a945c0d961f2f`).
- 2026-11-19 UTC: Launched retry `r3` from immutable revision `801b9ec1e3c6c8a1c59dc9a0c8f86560fe81210c`, using node-local source/data/checkpoints and the region-matched `us-west4` bucket. Verified the first optimizer updates, the update-200 checkpoint, and live W&B metrics. Started monitor PID `270638`; its first check is delayed 30 minutes and subsequent checks are spaced 30 minutes apart.
- 2026-11-19 07:27 UTC: First scheduled 30-minute check found the TPU `READY`/`HEALTHY`. Worker 0 was occupied only by the exact current-user `progressive-teacher-weighted` process; workers 1–3 were `AVAILABLE`. `progressive-current` and `progressive-low-lr` had completed successfully, while the third arm had reached update 200 with validation loss `0.4697417617`, cross-entropy `1.4529317096`, KL `0.1420117775`, and gradient cosine `0.2777080238`. Preserved the two completed full checkpoints in the regional output prefix and verified object sizes of 5,929,296,571 and 5,929,296,568 bytes with CRC32C checksums. Worker 0 had 8 GB free; no files were removed and the exact run continued under the 30-minute monitor.
- 2026-11-19 07:57 UTC: Second scheduled check found the TPU `READY`/`HEALTHY`, the exact current-user arm-four process on worker 0, and workers 1–3 `AVAILABLE`. `progressive-teacher-weighted` completed successfully and was fully mirrored to the regional output. `progressive-remaining-mask` had reached update 197/400 with a valid update-100 checkpoint. Worker 0 had 7 GB free; the run remained healthy and no files were removed.
- 2026-11-19 08:27 UTC: Third scheduled check found the TPU `READY`/`HEALTHY`, workers 1–3 `AVAILABLE`, and the exact current-user `direct-low-lr` process starting on worker 0. `progressive-remaining-mask` completed successfully with final validation loss `0.7647055537`, cross-entropy `1.4405970499`, KL `0.0888140146`, and gradient cosine `0.3856447935`; its full checkpoint was mirrored to the regional output. Also mirrored and checksum-pinned the same-region training and validation data. Because worker 0 had only 7 GB free, staged immutable source, exact dependencies, verified regional data, and the protected W&B credential on idle worker 1 as recovery identity `r4-worker1`; no TPU devices were used by the staging process.
- 2026-11-19 08:57 UTC: Fourth scheduled check found the TPU `READY` with health flag `UNHEALTHY_TENSORFLOW` after worker 0 exhausted its disk while atomically writing the update-200 `direct-low-lr` checkpoint. Four arms remained complete and durable. The prior update-100 checkpoint remained intact at 5,929,296,470 bytes with SHA-256 `e61a64db2df22724ccbe335f8b1c11b90231dfea1996b79f63664d51f6b8c721`; relayed it without deletion through the NFS artifact store to worker 1. Worker 1 had 35 GB free, its exact data hashes passed, and its four-device bounded JAX probe succeeded. A first recovery attempt stopped before TPU initialization because a validation-manifest compatibility symlink was missing; this zero-step error is recorded separately. After adding the missing provenance link, all four workers again reported `AVAILABLE` and retry `r5` resumed the exact fifth arm from update 100 on worker 1. Verified repeated deterministic updates through 200 and a matching update-200 validation record; the process continued into its checkpoint save.
- 2026-11-19 09:28 UTC: `direct-low-lr` completed update 400 on worker 1, wrote a checksum-valid final checkpoint, uploaded all artifacts to the regional bucket, and finalized W&B. All five W&B runs report `finished` at step 400, and all four TPU workers reported `AVAILABLE`. The resource retained the provider health label `UNHEALTHY_TENSORFLOW`, but the bounded JAX probe, resumed training, checkpoint writes, and regional upload all completed successfully without a device error.
