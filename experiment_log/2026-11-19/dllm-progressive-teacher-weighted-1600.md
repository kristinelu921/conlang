# dLLM progressive teacher-weighted 32→16, 1,600 updates

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-progressive-teacher-weighted-1600-20261119` |
| Status | `completed` |
| Started | `2026-11-19 16:25 UTC` |
| Finished | `2026-11-19 18:19 UTC` |
| Baseline | `dllm-loss-ablation-32-16-20261119`, arm `progressive-teacher-weighted` |
| Git revision | `35a1613e32afe21316ff2f00cc1a2d1593501de9` |
| Command | `python -m reproduction.run_loss_ablation --config <node-local-source>/reproduction/configs/progressive_teacher_weighted_1600.json --data <node-local-data> --validation-data <node-local-validation>/validation_finecode.npz --progressive-parent <node-local-parent>/latest.pkl --output <fresh-node-local-output>` |
| Seeds | training `42`; fixed validation corruption `314159` |
| Hardware | `kmh-tpuvm-v5e-16-kaiminghe-4612fc`, `us-west4-a`, v5litepod-16; worker 0 bounded to its four local chips |

## Configuration

Source: `experiments/dllm-reproduction/reproduction/configs/progressive_teacher_weighted_1600.json`

- Teacher/student diffusion steps: `32 → 16`
- Updates: `1600`
- Batch size: `8` (`12,800` examples exposed; one full pass through the fixed permutation)
- Learning rate: warm up for `40` updates to `1e-5`, then cosine decay to `1e-6`
- Objective: `0.25 × cross-entropy + 0.75 × KL`, temperature `4.0`
- KL positions: all originally corrupted positions
- Validation: `64` fixed disjoint examples every `50` updates
- Checkpoint cadence: every `100` updates
- Optimizer: AdamW, weight decay `0.01`, global gradient clipping at `1.0`

## Changed Hyperparameters

| Parameter | Baseline | Treatment |
|---|---:|---:|
| Updates | `400` | `1600` |
| Training examples exposed | `3,200` | `12,800` |
| Cosine-decay horizon | `400` updates | `1600` updates |
| W&B run identity | `dllm-loss-ablation-32-16-20261119-progressive-teacher-weighted` | `dllm-progressive-teacher-weighted-1600-20261119-progressive-teacher-weighted-1600` |

All other scientific hyperparameters, data, parent checkpoint, validation examples, and seeds are unchanged.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Parent checkpoint | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/parent/latest.pkl` | localized at 5,929,291,352 bytes; SHA-256 `8e4fe62184b923cda8e54f01a5d6cfc41eb7091bb3fa4a9173d2b48758b95e96` |
| Training data | `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/retry-r3/data` | NPZ SHA-256 `95c006563cc336816bfc38f1a2b366aec724664a9689f1e7d9ff5e07d647d401` |
| Validation data | same regional data prefix | NPZ SHA-256 `22fcaefad51954e8bf45152b3ee27d36e1c30652458f0f326a3a945c0d961f2f` |
| Final checkpoint | `gs://kmh-gcp-us-west4/kristine/dllm-progressive-teacher-weighted-1600-20261119/retry1/output/progressive-teacher-weighted-1600/latest.pkl` | update 1,600; 5,929,296,710 bytes; SHA-256 `54553fbf494bbed7198392dd99f96e921ddafe4402cbcd996f4f362eb42092bf` |
| Regional output | `gs://kmh-gcp-us-west4/kristine/dllm-progressive-teacher-weighted-1600-20261119/retry1/output` | completion marker, configuration, 1,600 training rows, 33 validation rows, and final checkpoint verified |
| W&B project | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/dllm-distillation-reproduction` | run finished at update 1,600 |
| Source snapshot | `experiment_log/2026-11-19/artifacts/dllm-progressive-teacher-weighted-1600/source-35a1613.tar.gz` | SHA-256 `dac6e70edfd54b85161b40ee773f5ad4cb8454a30d32912c84ba731b201a49fa` |
| W&B run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/dllm-distillation-reproduction/runs/dllm-progressive-teacher-weighted-1600-20261119-progressive-teacher-weighted-1600` | verified `running` after step 1 |

## Results

| Fixed validation metric | 400 updates | 1,600 updates | Relative change |
|---|---:|---:|---:|
| Total loss | `0.4678398110` | `0.4590677582` | `-1.875%` |
| Cross-entropy | `1.4516941011` | `1.4199886918` | `-2.184%` |
| KL | `0.1398883732` | `0.1387607912` | `-0.806%` |

The longer run improved every fixed-validation objective. Final CE/KL gradient norms were
`10.1914958954`/`4.4746837616`, with cosine similarity `0.2016116679`. The separate
HumanEval+/MBPP+ evaluation uses this exact final checkpoint and is logged independently.

## Updates

- 2026-11-19 UTC: User requested 1,600 optimizer updates after reviewing the 400-update training and validation curves.
- 2026-11-19 UTC: Required filtered capacity audit selected the region-matched `kristine` v5e-16 candidate above. The all-worker ownership check reported all four workers `AVAILABLE`; no process or lock was touched.
- 2026-11-19 UTC: Retry 0 stopped at the final preflight with zero updates because `/tmp/tpu_logs` was collaborator-owned and non-writable on every worker. Source, data, validation, and parent localization had succeeded, but no output, checkpoint, or W&B run was created. Preserved the exact evidence in `dllm-progressive-teacher-weighted-1600-retry0-error.md` and left all collaborator state untouched.
- 2026-11-19 16:25 UTC: The official wide read-only sweep selected idle, non-denylisted, region-matched `kmh-tpuvm-v5e-16-kaiminghe-4612fc`. Every worker passed the exact ownership check; worker 0 had writable TPU logs and about 74 GB free before staging. Retry 1 verified the immutable source, pinned runtime, regional data, validation set, parent checkpoint, W&B authentication, and four-chip topology. Training reached update 1 and W&B reported `running`. Initial fixed-validation loss was `0.4860155378`, cross-entropy `1.4662208930`, and KL `0.1592804091`.
- 2026-11-19 16:59 UTC: First scheduled 30-minute check found the TPU `READY`/`HEALTHY`, the exact expected trainer on worker 0 at update 800, and no trainer on workers 1–3. Fixed-validation loss was `0.4671776555`, cross-entropy `1.4331262186`, and KL `0.1451948015`. W&B was `running` at step 799. The newest completed checkpoint was update 700, 5,929,296,710 bytes, SHA-256 `c1a1ef9cc2a18f70ef0219057a99cd0087f2c0ba737ff6744ad761784a760b4a`; worker 0 had 48 GB free. The first monitor wrapper had a local quoting error after its health check; the corrected inspection completed the same scheduled check, the trainer was unaffected, and the infrastructure error is preserved separately.
- 2026-11-19 17:33 UTC: Second scheduled check found the TPU `READY`/`HEALTHY`, the exact expected trainer on worker 0 at update 1,100, workers 1–3 `AVAILABLE`, and no foreign holder. Fixed-validation loss improved to `0.4630476329`, cross-entropy to `1.4255541861`, and KL was `0.1422121227`. W&B was `running` at step 1,099. The newest completed checkpoint was update 1,000, 5,929,296,710 bytes, SHA-256 `75bb300f27ff06823a657d308ebec0e471ace23eddc2232e39fe4c220d5bfd1a`; worker 0 retained 48 GB free. No recovery action was needed.
- 2026-11-19 18:04 UTC: Third scheduled check found the TPU `READY`/`HEALTHY`, the exact expected trainer on worker 0 at update 1,500, workers 1–3 `AVAILABLE`, and no foreign holder. Fixed-validation loss improved to `0.4589898083`, cross-entropy to `1.4205546081`, and KL to `0.1384682087`. W&B was `running` at step 1,499. The newest completed checkpoint was update 1,400, 5,929,296,710 bytes, SHA-256 `843135483f870fc96c4968b6266fdc8bcbd21181b750e2bf2e0071d9437a85d8`; worker 0 retained 48 GB free. No recovery action was needed.
- 2026-11-19 18:19 UTC: Training completed all 1,600 updates and uploaded the final checkpoint and completion marker to the region-matched bucket. Final fixed-validation loss was `0.4590677582`, cross-entropy `1.4199886918`, and KL `0.1387607912`. W&B finalized successfully. The Spot TPU disappeared after the completed upload; no training recovery was required because the final object size, SHA-256, completion marker, metrics, and W&B terminal state were already verified.
