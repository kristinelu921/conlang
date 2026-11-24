# SmolLM2-135M GSM8K supervised fine-tuning

## Run Summary

| Field | Value |
|---|---|
| Run ID | `smollm2-gsm8k-sft-20261102-attempt1` |
| Status | completed |
| Started | training supervisor timestamp 2026-11-02T21:17:57Z |
| Finished | training 2026-11-02T21:23:19Z; final evaluation 2026-11-02T21:29:50Z |
| Baseline | [base-model benchmark](smollm2-135m-gsm8k-tpu.md): strict 20/1319, flexible 25/1319 |
| Git revision | `3cf8cb45f56080c5361dc87929deff4376069061`; exact standalone training source archived with SHA-256 |
| Command | `cd /home/kristinelu/smollm2-gsm8k/sft-20261102-attempt1 && bash run_training.sh` |
| Evaluation command | `cd /home/kristinelu/smollm2-gsm8k/sft-20261102-eval-fp32 && bash evaluate.sh` |
| Seed | training 1234; evaluation Python 0, NumPy/Torch/few-shot 1234 |
| Hardware | dedicated Spot v5e-8, one worker / eight chips, `us-west4-a` |
| Exact TPU | `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a` |

## Configuration

Complete effective configuration: [train_config.json](smollm2-135m-gsm8k-sft-artifacts/train_config.json).
Source bundle: `smollm2-gsm8k-sft-attempt1.tgz`, SHA-256
`d25313d1d2617129c9c3121d96a5164a6f11e4668f6ab1b63586c12490b84154`.

- Base checkpoint: `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Dataset: `openai/gsm8k`, `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`, training split only.
- Full-parameter supervised fine-tuning, three fixed epochs over all 7,473
  training examples. No hyperparameter selection using test results.
- Prompt: `Question: {question}\nAnswer:`; target: leading space, original
  worked solution and `####` answer, then EOS. Answer-only next-token loss.
  Input and output embeddings share one trainable parameter; their gradients add.
- Sequence length 640, right padding, no truncation or packing. Observed maximum
  is 543 tokens, median 178; 1,423,168 input/target tokens across one epoch.
- Global batch 32: four examples per chip, eight chips, accumulation one.
  Final batch padding has zero loss weight, so all examples occur exactly once
  per epoch. Deterministic shuffle uses seed `1234 + epoch_index`.
- 234 steps/epoch, 702 total updates; 22 warmup steps. AdamW learning rate
  rises from zero to 5e-5, then cosine-decays to 5e-6; beta1 0.9, beta2 0.999,
  epsilon 1e-8, weight decay 0.01 on all trainable leaves, gradient norm clip 1.0.
- Float32 weights and computation, highest matmul precision. Gradient
  checkpointing disabled. Global answer-token-weighted gradients across replicas.
- Full model and optimizer state saved and mirrored after every epoch.
  Final export includes Hugging Face PyTorch safetensors and Flax weights.
- Final test: all 1,319 examples, the baseline's five-shot prompt/sampler, greedy
  decoding, maximum 256 tokens, lm-eval 0.4.13 strict and flexible exact match.
- Installed runtime is inherited from the verified remote benchmark:
  JAX 0.6.2, Flax 0.10.5, Optax 0.2.5, Transformers 4.57.6, Torch 2.7.0+cpu,
  NumPy 1.26.4. [Complete lock](smollm2-135m-gsm8k-tpu-artifacts/completed/requirements.txt).

The completion-only objective follows the standard prompt-completion SFT
formulation documented by [Hugging Face](https://huggingface.co/docs/trl/sft_trainer#train-on-completion-only).
The verified TPU generation adapter is reused for loading and final evaluation.

## Changed Hyperparameters

The baseline had no fine-tuning. The treatment applies the training configuration
above. The test dataset, few-shot examples and seed, generation budget, precision,
and scoring remain identical for the final comparison.

## Results

Training completed all three epochs and 702 optimizer updates: 22,419 example
presentations. Training plus checkpoint/export/mirroring took 290.9322935739999
seconds after model and data preparation. Answer-token-weighted mean losses
(nats, lower is better) were 0.7379579547941112, 0.5765657470686538, and
0.5246872219575326 for epochs 1, 2, and 3, respectively.

All 1,319 GSM8K test questions were evaluated at the final step-702 checkpoint.
Accuracy is higher-is-better; standard errors below are in proportion units.

| Run | Strict exact match | Flexible exact match |
|---|---|---|
| Base model | 20/1319 = 0.015163002274450341 | 25/1319 = 0.018953752843062926 |
| Three-epoch fine-tune | **51/1319 = 0.03866565579984837** | **52/1319 = 0.039423805913570885** |
| Absolute gain, percentage points | 2.350265352539803 | 2.047005307050796 |
| Fine-tuned standard error | 0.005310583162098028 | 0.00536028003034247 |

Source: [validated summary](smollm2-135m-gsm8k-sft-artifacts/completed/summary.json)
and its archived per-question records. Both filters have exactly 1,319 unique
question IDs; aggregate scores equal sample-level means. Every document,
prompt and target hash matches the baseline across all 2,638 filter records:
[comparison verification](smollm2-135m-gsm8k-sft-artifacts/baseline-comparison-verification.json).
Generation took 27.844235187999857 seconds, including 8.992153947000133 seconds
for the first batch; lm-eval reported 61.46425813700034 seconds total.

The [comparison chart and training-loss curve](http://localhost:8768/sft/)
are served by the remote VM through the existing localhost SSH tunnel.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Trainer and tests | [source directory](smollm2-135m-gsm8k-sft-artifacts/) | remote source checksums match; four tests passed in 6.37 seconds |
| Node-local training output | `/home/kristinelu/smollm2-gsm8k/sft-20261102-attempt1/output` | training completed, exit code 0 |
| Regional durable outputs | `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-sft-20261102/attempt1/` | all three resumable checkpoints and final export mirrored |
| Canonical float32 model | `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-sft-20261102/attempt1/final-model-fp32/` | exact weight comparison and numerical/generation parity passed |
| W&B | [xvex9eas](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/xvex9eas) | authenticated read-back: finished, all 702 training rows, both final scores |
| Completed results | [archive](smollm2-135m-gsm8k-sft-artifacts/completed.tgz) and [extracted files](smollm2-135m-gsm8k-sft-artifacts/completed/) | archive and every result SHA-256 verified |
| Durable result archive | `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-sft-20261102/attempt1/smollm2-sft-completed.tgz` | upload completed |

Completed result archive SHA-256:
`f278d42712ae877318f4b5524fe8645d0ba91388c941b7267166cf040da02534`.
The [W&B verification record](smollm2-135m-gsm8k-sft-artifacts/completed/training-evidence/wandb-final-verification.json)
preserves the authenticated final status and metrics. Model weights and raw
examples were not uploaded to W&B.

## Failure Details

No training attempt failed. Automatic approval review rejected the initial W&B
upload scope, which included raw evaluation examples, logs, source and model
files. The approved replacement publishes only scalar metrics and an explicit
allowlist of hyperparameters. Model files remain in regional GCS storage;
raw results remain in the experiment artifact directory.

The first export verification failed before any test generation. The original
directory mixed Flax and PyTorch checkpoint formats, and its PyTorch construction
inherited the base config's bfloat16 dtype. The full trained float32 Flax state
was preserved. A separate explicit-float32 export created `final-model-fp32`;
the fresh evaluator compares every weight tensor against the trained Flax state
before numerical parity and test scoring. See the
[immutable failure record](smollm2-135m-gsm8k-sft-eval-attempt1.md).

## Observations

- Final filtered inventory selected the previously provisioned dedicated Spot
  v5e-8. No new resource was provisioned for this fine-tune.
- Training data and base model are reused from the verified node-local cache;
  no dataset is transferred across regions.
- Remote Cloud SDK 463.0.0 with system Python 3.10.12 successfully wrote and read
  a run-scoped object in the US-WEST4 bucket. Free node-local disk: 75 GiB.
- All tests, training, inference and evaluation run on the dedicated remote VM.
  The development machine is used for editing, orchestration and result inspection.
- Four focused remote tests passed. Exported PyTorch weights exactly match the
  trained Flax checkpoint; maximum absolute logit difference was
  0.00003528594970703125, greedy continuations matched, and two stop-string
  checks passed. No additional training was needed for the export correction.
- Final inspection of the single physical worker confirmed both exit codes are
  zero, the exact training/evaluation PIDs are absent, and every TPU device is
  idle. Cloud state remains READY/HEALTHY. The Spot allocation remains present;
  no deletion or shutdown was requested.

## Interpretation

This is one prescribed fine-tuning run, with the final checkpoint chosen by the
fixed three-epoch schedule. It is not a hyperparameter search or a claim of
optimal attainable GSM8K performance.

## Updates

- 2026-11-02T21:12Z: Completed filtered inventory and selected the existing
  dedicated TPU from the final AVAILABLE table.
- 2026-11-02T21:14Z: Audited training lengths on the remote VM and verified
  regional artifact storage before staging the source bundle.
- 2026-11-02T21:17:59Z: Source checksums and four focused tests passed remotely.
  Fresh all-device occupancy returned AVAILABLE; API confirmed READY, HEALTHY,
  Spot=true. Launched the detached supervisor with an atomic duplicate-launch
  claim. One recurring read-only monitor checks cloud health, process identity,
  recent metrics and checkpoint manifests every 30 seconds. First useful
  training progress remains pending.
- 2026-11-02T21:20Z: Training reached the first checkpoint. Steady-state updates
  process 32 examples in approximately 0.1224 seconds (about 261 examples/s),
  excluding initial compilation and checkpoint/mirroring intervals.
- 2026-11-02T21:23Z: Training completed all 702 updates and mirrored the final
  export. The exact supervisor wrote exit code 0; the trainer no longer exists.
  Epoch-1 answer-token loss is 0.7379579547941112; epoch 2 is 0.5765657470686538.
- 2026-11-02T21:24Z: User requested W&B publishing. A separate remote environment
  runs W&B 0.22.2 with hardware/system monitoring disabled, backfills numeric
  metrics, and awaits final test scores. No active training runtime was modified.
  The W&B destination is the repository's established `conlangs` project.
- 2026-11-02T21:24Z: A fresh device occupancy check returned AVAILABLE. Started
  exact-export verification and the full five-shot test evaluation under a
  detached supervisor, with one recurring 30-second evaluation monitor.
- 2026-11-02T21:27Z: Corrected export completed without repeating training.
  Canonical PyTorch checkpoint:
  `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-sft-20261102/attempt1/final-model-fp32/`.
  `model.safetensors` is 538,090,408 bytes, SHA-256
  `409c660ab1294fd3e4aebcc04f71ffc04fe669812a4ef8a836b2fc4398014da8`.
  Original trained Flax SHA-256:
  `f98dce186fd597303d59f43f6cf8ca10f4b23ba60f1d5d989b752b246ea9d677`.
  The corrected evaluator snapshot SHA-256 is
  `2f05e7df65f32518297462b8bd787a3cab92944adf2e9835d87b3beb812a2c28`.
- 2026-11-02T21:27Z: Authenticated W&B read-back confirmed `last_training_step=702`,
  `train/loss=0.5137691497802734` for the final batch, and `epochs=3`.
- 2026-11-02T21:29:50Z: Corrected evaluation completed with exit code 0 after
  passing exact-weight and backend parity checks. Strict accuracy is 51/1319;
  flexible accuracy is 52/1319. All generated questions and scores reconciled.
- 2026-11-02T21:31Z: W&B finalized successfully. Authenticated history contains
  precisely steps 1 through 702 and both final test scores. The remote report
  is available at `http://localhost:8768/sft/`.
- 2026-11-02T21:32Z: Retrieved and checksummed the canonical worker's result
  archive, preserved raw outputs, and verified identical baseline/test prompts.
  SCP encountered transient message-authentication errors before succeeding;
  the archive was also mirrored to the run's existing US-WEST4 GCS directory.
  Final device inspection confirmed idle accelerators; recurring experiment
  monitoring ended after verified completion.
