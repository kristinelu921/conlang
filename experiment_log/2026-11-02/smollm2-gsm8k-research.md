# SmolLM2-135M GSM8K automatic fine-tuning study

## Run Summary

| Field | Value |
|---|---|
| Study ID | `smollm2-gsm8k-research-20261102` |
| Status | completed; publication verification recovered after an API pagination-check failure |
| Started | 2026-11-02T21:40:46Z |
| Budget | User authorized up to four hours on the existing single Spot v5e-8; sequential trials only |
| Baseline | [Three-epoch fine-tune](smollm2-135m-gsm8k-sft.md), strict 51/1319 and flexible 52/1319 on the five-shot test protocol |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`, project `he-vision-group`, zone `us-west4-a` |
| Command | `cd /home/kristinelu/smollm2-gsm8k/research-20261102 && bash run_study.sh` |
| Source | [Immutable study sources](smollm2-gsm8k-research-artifacts/source/) |
| Effective study configuration | [study_config.json](smollm2-gsm8k-research-artifacts/source/study_config.json) |
| Source archive SHA-256 | `0744705a96d0816b4cbe0fe4056d6bad70106a07154f8d84e0fc0b1f5ffe7682` |
| Output | `/home/kristinelu/smollm2-gsm8k/research-20261102/output` |
| Checkpoint mirror | `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-research-20261102/` |

## Completed Results

Completed 18 main-search candidates, one label-smoothing probe, and two fixed
full-data refits on the same dedicated Spot v5e-8. All model computation ended
at 2026-11-03T01:29:00Z, about 3 hours 48 minutes after the study budget clock
started, within the authorized four hours. No experiment or smoke job used a
development machine. The TPU is READY/HEALTHY and idle; it remains allocated.

Loss is unweighted answer-token NLL in nats, including rationale and EOS, measured
on all 7,473 training and 1,319 test examples. Accuracy is strict exact match on
all 1,319 test questions, greedy decoding, maximum 256 generated tokens.

| Model | Full train NLL | Full test NLL | Zero-shot test | Five-shot test |
|---|---:|---:|---:|---:|
| Base SmolLM2-135M | 1.588614 | 1.592871 | 0/1319 = 0.00% | 20/1319 = 1.52% |
| Previous three-epoch fine-tune | 0.507470 | 0.610524 | 74/1319 = 5.61% | 51/1319 = 3.87% |
| Accuracy-selected research refit | 0.000310 | 1.437200 | 153/1319 = **11.60%** | 13/1319 = 0.99% |
| Loss-selected research refit | 0.357530 | **0.601065** | 96/1319 = 7.28% | 64/1319 = **4.85%** |

The accuracy-selected model improves zero-shot strict accuracy by 5.99 percentage
points over the previous fine-tune, but substantially worsens test NLL and
five-shot performance. The loss-selected model gives the best historical
five-shot result and lowest fine-tuned test NLL: improvements of 0.99 percentage
points and 0.009459 nats respectively. These are the best results among the tested
recipes, not a claim of a global optimum or statistical significance.

The five-shot prompt/document/target hashes exactly match the original benchmark
for both new models. The poor five-shot result of the accuracy-selected model
is retained and reported. Its sampled outputs mix unrelated entities and
arithmetic despite passing exact checkpoint/export checks. No model was changed
or checkpoint selected in response to these test results. Matched zero-shot
baseline measurements were added afterward to interpret the prompt-format shift.

Flexible extraction scores are: base zero-shot 23/1319 (1.74%), prior fine-tune
zero-shot 75/1319 (5.69%), accuracy refit zero/five-shot 153/1319 (11.60%) and
13/1319 (0.99%), loss refit zero/five-shot 97/1319 (7.35%) and 65/1319 (4.93%).
The base zero-shot strict score of zero reflects the strict required answer
format; flexible extraction finds some correct numeric answers.

Use the accuracy-selected checkpoint with zero-shot prompts when optimizing that
metric. Use the loss-selected checkpoint for lower held-out NLL and the original
five-shot benchmark protocol. Both selections were fixed using validation data.

## Final Artifacts and Verification

- [Live localhost report](http://localhost:8769/),
  [PNG figure](smollm2-gsm8k-research-artifacts/report/research.png),
  [SVG figure](smollm2-gsm8k-research-artifacts/report/research.svg),
  [complete scalar summary](smollm2-gsm8k-research-artifacts/report/summary.json).
- [Accuracy-model W&B run](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/wril0ygi)
  and [loss-model W&B run](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/d8ml8ua1).
  All 21 study/refit W&B runs are finished. Verified 80,186 optimizer updates and
  241 full-split loss records against immutable node-local events. Final summaries
  include train/test NLL, zero/five-shot results, and matched baseline metrics.
- Accuracy model:
  `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-research-20261102/final-model-fp32/`.
  `model.safetensors`: 538,090,408 bytes; SHA-256
  `cb7a39bf874e8496b43243cd2f6323a7f979f906f7771fd710569e25faa8eb58`.
- Loss model:
  `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-research-20261102/loss-final-model-fp32/`.
  `model.safetensors`: 538,090,408 bytes; SHA-256
  `c0b75beba2a7c1d6b8ed358be78a44350c64757189fbf966809c51a59f998569`.
- Both exports preserve float32 and exactly match trained Flax weights. Maximum
  PyTorch/TPU logit differences were 0.0000324249267578125 (accuracy) and
  0.00002956390380859375 (loss), below 0.02; long-position logits, greedy
  sequences, and stop handling passed. Earlier masked-NLL parity agreed within
  0.00002019902878025448 nats. All 24 focused tests passed remotely.
- Full raw evidence remains node-local at
  `/home/kristinelu/smollm2-gsm8k/research-20261102/completed-results.tgz`:
  31,153,543 bytes, 512 checksummed files; archive SHA-256
  `a9daea616ac169fde4f139a919dc6cad85bd22cfec04302d15eb966577e60824`.
  Automatic approval review rejected the proposed broad archive/source upload
  to GCS because its payload/destination scope was not explicitly approved.
  That upload was not performed. The safer completion keeps the raw archive
  on the VM, publishes the requested scalar W&B results, and saves only derived
  charts/scalar reports in the shared workspace. Existing checkpoint mirrors
  remain available at the paths above.
- [Final evidence/manifests](smollm2-gsm8k-research-artifacts/final-evidence.json),
  [publication recovery](smollm2-gsm8k-research-artifacts/publication-recovery.log),
  [final health and canonical log checksums](smollm2-gsm8k-research-artifacts/final-health.log),
  [pagination postmortem](../2026-11-03/smollm2-wandb-pagination.md).
  All four downloaded report files match their remote SHA-256 values; the PNG
  was visually inspected. The prior SSH tunnel intermittently closed larger
  transfers, so the verified derived files are served directly by a local
  static-file server for result inspection. All four HTTP responses returned
  200 and matched their saved byte counts and SHA-256 values. The server remains
  running on port 8769; it performs no experiment computation.

This is a standalone Hugging Face model study using the verified TPU adapter,
not an ELF model/trainer launch. The exact standalone source and effective JSON
configuration are checksummed and copied to node-local storage. Existing ELF
configuration files and unrelated worktree changes are not part of this study.

## Configuration

- Model: base `HuggingFaceTB/SmolLM2-135M`, revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; all trials initialize from it.
- Dataset: `openai/gsm8k`, `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`. Training data and model are
  reused from the verified node-local cache in US-WEST4.
- Deterministic split of the original 7,473 training examples, seed 20261102:
  6,726 optimization examples and 747 held-out validation examples. Save all
  source indices and verify disjointness, completeness and reproducibility.
- Training prompt `Question: {question}\nAnswer:`; supervise original rationale,
  `####` numeric answer and EOS. Train all model weights with a tied embedding.
- Float32, highest matmul precision, sequence length 640, no truncation,
  accumulation one, no packing, no gradient checkpointing. Right padding and
  zero-weight final-batch slots preserve exact example coverage.
- AdamW beta1 0.9, beta2 0.999, epsilon 1e-8; global gradient norm clip 1.0.
  Default weight decay 0.01, warmup 3%, cosine decay to 10% of peak learning rate.
- Initial six candidates: learning rates 1e-5, 3e-5, 5e-5, 1e-4, 3e-4 and 1e-3
  with global batch 32. Then adapt around the best validation candidate, changing
  batch size (8, 16, 64), learning rate (half/double, capped at 0.002), weight decay
  (0, 0.1), final-answer loss weight (2, 4), warmup (10%), terminal learning-rate
  ratio (1%), schedule length (20 epochs), and shuffle seeds (2345, 3456).
- Default 12-epoch schedule with early stopping after three epochs without
  improvement in either validation accuracy or validation loss; minimum three
  epochs. Training shuffle seed is 1234 plus epoch index unless the candidate
  explicitly changes the seed. Every selected candidate's complete config is
  written before its first update.
- At most 20 sequential main-search candidates, plus the optional two loss probes
  described below. Reserve 40 minutes of the four-hour budget
  for final fitting/export/evaluation; stop proposing candidates when less than
  eight minutes remain in the search window or node-local free disk drops below
  12 GiB. Keep finalized checkpoints; do not delete previous experiment files.
- For each epoch, measure unweighted answer-token NLL on the complete optimization
  and validation partitions in evaluation mode. Training update loss is a
  separate metric, especially when final-answer weighting differs from one.
- Validation accuracy uses zero-shot greedy decoding with 256 new tokens and
  the pinned benchmark's strict/flexible numeric extraction. All 747 validation
  examples are scored each epoch. Validation has no in-context demonstrations.
- Select checkpoint and recipe by maximum validation strict exact match, with
  lower unweighted validation NLL as the tie-breaker. Test metrics are not inputs
  to selection. Write an immutable selection record before opening the test set.
- Refit the selected recipe from base on all 7,473 original training examples,
  using its selected epoch count and original schedule horizon, with seed 1234.
  During this fixed final fit, plot full training and full test answer-token NLL
  after each epoch. The test curves are diagnostic; they cannot change the already
  locked epoch count or hyperparameters. Preserve the selected validation model.
- Final generation evaluation will include the previous five-shot test protocol
  for comparability. Zero-shot validation accuracy is not directly comparable to
  the historical five-shot test score.
- A separately declared secondary objective preserves the recipe with minimum
  validation NLL. If its configuration/selected epoch differs from the accuracy
  winner and the remaining budget permits, refit it once on the complete training
  set and report both models. This secondary criterion was declared before any
  research test loss was observed; test results do not choose either recipe.
- Based on the initial validation curves, a separate bounded loss refinement was
  declared before any research test metric was observed: at most two probes with
  label smoothing 0.05 and 0.1 around the main search's lowest-validation-NLL
  recipe. Preserve its optimizer, batch, seed and schedule horizon; allow up to
  six epochs, minimum three, patience two. Each probe starts only if at least
  30 minutes remain in the original budget and reserves 20 minutes for final
  refitting/export/evaluation. These probes do not extend the four-hour budget.
  Only the training objective mixes targets with a uniform distribution;
  reported training/validation/test NLL always uses the original hard targets.

## Research Rationale

[SmolTulu](https://arxiv.org/html/2412.08347v1) reports sensitivity of SmolLM2-135M
reasoning performance to learning-rate/batch-size ratio. This motivates a broad
ratio search; its reported optimum is not assumed to transfer to this dataset.
[Hugging Face's SFT documentation](https://huggingface.co/docs/trl/sft_trainer)
defines the prompt-completion objective and masked, shifted next-token loss.
The current study keeps the established objective and separately tests final-answer
emphasis while always reporting unweighted held-out NLL for comparability.

[When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) motivates the
two targeted smoothing probes as an overconfidence/calibration hypothesis. Its
findings are not assumed to transfer to GSM8K or this model. The smoothing
implementation is checked against PyTorch for both values and gradients, and
zero smoothing must exactly preserve the established training objective.

## Changed Hyperparameters

The previous run used all training examples, three fixed epochs, learning rate
5e-5 and global batch 32. The study introduces a fixed validation partition,
checkpoint selection, a longer search horizon, and the explicitly listed
candidate variations. The original base model and pinned GSM8K revision remain
the same. No external training dataset or larger model is introduced.

## Results and Artifacts

Pending. Each candidate will have scalar training/validation metrics and complete
public hyperparameters in W&B project `conlangs`, group
`smollm2-gsm8k-research-20261102`. Raw examples, code, credentials and model weights
are excluded from W&B uploads. Checkpoints are mirrored to regional GCS with
completion manifests and SHA-256 records.

## Limitations

The result will be the strongest validation candidate found within this bounded
search, not a proof of globally optimal fine-tuning. Searching many checkpoints
can overfit the validation set; the locked final test evaluation provides an
independent generalization measurement. Final refitting on all training examples
can change performance relative to the validation-selected checkpoint.

## Updates

- 2026-11-02T21:33Z: User authorized a four-hour budget. Filtered inventory
  returned the existing dedicated Spot v5e-8 as the sole available TPU. The ELF
  runtime is absent, while the separately verified SmolLM runtime remains local.
- 2026-11-02T21:38Z: Fresh occupancy inspection returned AVAILABLE. Staged the
  finalized source snapshot and began focused tests on the dedicated remote VM.
  No development machine is used for training, inference, evaluation or smoke runs.
- 2026-11-02T21:40:46Z: All 13 focused tests passed in 6.65 seconds. Exact
  source checksums passed; free local disk was 69 GiB. Fresh occupancy was
  AVAILABLE and API health READY/HEALTHY, Spot=true. Launched the single study
  supervisor and scalar-only W&B bridge. A recurring read-only monitor checks
  process identity, cloud health, recent progress and durable checkpoint markers
  every 30 seconds. Trainer PID is 41051; supervisor PID is 41045.
- 2026-11-02T21:42Z: Independent three-example distributed-loss parity passed:
  TPU 1.8818566133659385 versus PyTorch 1.8818364143371582 nats/token,
  absolute difference 0.00002019902878025448, matching 369 answer tokens.
  Base full-partition NLL is 1.5878968272529566 for training and
  1.5950332557048283 for validation. First W&B run:
  [trial-01](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/4yw9lbfr).
- 2026-11-02T21:43Z: Before any research test metric was observed, declared the
  secondary validation-NLL champion and its optional distinct full-data refit.
- 2026-11-02T21:53Z: Trial 01 (learning rate 1e-5, batch 32) completed 12 epochs.
  Selected epoch 6: validation strict 39/747 = 0.05220883534136546,
  validation NLL 0.6383289165378986, training NLL 0.5593657170265216.
  Lowest validation NLL was 0.6320163502222119 at epoch 12. Its complete
  1,614,217,920-byte optimizer checkpoint was checksummed and mirrored to
  `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-research-20261102/trial-01-checkpoint/`.
  [Completion evidence](smollm2-gsm8k-research-artifacts/trial-01-completion.log).
  Trial 02 (learning rate 3e-5, batch 32) started from identical base losses:
  [W&B](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/z7a99opu).
- 2026-11-02T22:01Z: Trial 02 stopped after eight epochs under the preset
  validation rule. Selected epoch 5: strict 47/747 = 0.06291834002677377,
  training NLL 0.43616088103773176 and validation NLL 0.6166174625333476.
  Minimum validation NLL was 0.6153458259391221 at epoch 4. Checkpoint and
  completion marker are mirrored under `trial-02-checkpoint/`.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-02-completion.log).
  [Trial 03](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/dh08qwsw)
  started at learning rate 5e-5.
- 2026-11-02T22:09Z: Trial 03 (5e-5, batch 32) stopped after eight epochs.
  Selected epoch 5: strict 51/747 = 0.06827309236947791, training NLL
  0.3400717714390477, validation NLL 0.6313953449466887. Minimum validation
  NLL was 0.606304569195386 at epoch 3. The optimizer checkpoint and marker
  were mirrored under `trial-03-checkpoint/`.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-03-completion.log).
  Trial 04 began at learning rate 1e-4 with the same split and batch size.
- 2026-11-02T22:17Z: Trial 04 (1e-4, batch 32) stopped after seven epochs.
  Selected epoch 4: strict 66/747 = 0.08835341365461848, training NLL
  0.2711135410460458, validation NLL 0.6421886719537868. Minimum validation
  NLL was 0.6045382790792161 at epoch 3. Full checkpoint and completion
  marker are mirrored under `trial-04-checkpoint/`.
  [W&B](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/um1kg6kp),
  [evidence](smollm2-gsm8k-research-artifacts/trial-04-completion.log).
  Corrected the local read-only monitor to sort checkpoint/run files by time;
  revalidated and replaced only monitor PID 2367347, leaving the remote study
  process untouched. One recurring monitor remains active.
- 2026-11-02T22:27Z: Trial 05 (3e-4, batch 32) completed 12 epochs. Selected
  epoch 10: strict 80/747 = 0.107095046854083, training NLL
  0.0018910771099737195 and validation NLL 1.2406770557925058. Its lowest
  validation NLL was 0.6135300958043454 at epoch 2. This exposes a substantial
  accuracy/NLL tradeoff; the secondary loss selection remains independent.
  Checkpoint and marker are mirrored under `trial-05-checkpoint/`.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-05-completion.log).
  A normalized-question audit found 6,726 unique training and 747 unique
  validation questions, with zero cross-split duplicate texts.
  [Trial 06](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/6mbyxov4)
  began at learning rate 1e-3.
- 2026-11-02T22:32Z: Declared the optional two-probe label-smoothing refinement
  using only the observed training/validation tradeoff. Its new source is staged
  separately; the active main search and its immutable source remain unchanged.
  The final loss recipe will be selected exclusively from validation evidence,
  with its own selection lock before its test-scored full-data refit.
- 2026-11-02T22:35Z: The separately staged smoothing source passed all 19 tests
  in 7.72 seconds on the dedicated VM, including independent PyTorch value and
  gradient checks for smoothing 0, 0.05, 0.1 and 0.4. Source archive SHA-256:
  `f4f2367bdc97d34e3acd7de460443e028b528b818bb778ea865dee81e34dd31f`.
  Updated finalization source SHA-256:
  `8b46bb5a7d553f6bf8e66ebbcaf8f773238762e240bac7b2b89e68092b4c41c7`.
  [Test evidence](smollm2-gsm8k-research-artifacts/loss-refinement-staging-tests.log).
  No smoothing experiment has been launched yet; these are reserved for the
  secondary validation-only phase and remain subject to the original budget.
- 2026-11-02T22:38Z: The six-rate grid completed. Trial 06 (1e-3, batch 32)
  selected epoch 11: strict 86/747 = 0.11512717536813923 and validation NLL
  1.3444893260874902. Its minimum validation NLL was 0.6858111542072653
  at epoch 2. The overall loss leader remains Trial 04, epoch 3, at
  0.6045382790792161. [Complete initial grid](smollm2-gsm8k-research-artifacts/trial-06-and-initial-grid.log).
  Checkpoint 06 is mirrored. [Trial 07](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/92y5r4fx)
  started with the accuracy leader's learning rate 1e-3 and global batch 8.
- 2026-11-02T22:42Z: Faster small-batch updates could push epoch summaries out
  of the console tail between monitor checks. Added a read-only inspector of
  complete `events.jsonl` records, reporting latest evaluation, current update,
  best validation accuracy and minimum validation NLL. Replaced only the local
  monitor; remote trainer PID 41051 continued uninterrupted.
- 2026-11-02T22:52Z: Trial 07 (1e-3, batch 8) completed 12 epochs. Selected
  epoch 11: strict 82/747 = 0.10977242302543508, training NLL
  0.001417669102330048 and validation NLL 1.5968346720387108. Minimum
  validation NLL was 0.8082080030438247 at epoch 2. Both objectives remain
  behind the corresponding batch-32 leaders. Checkpoint 07 is mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-07-completion.log).
  [Trial 08](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/q0rwlgss)
  started with batch 16 at the same learning rate.
- 2026-11-02T22:58Z: Staged finalization source v3 in a new immutable remote
  directory, without changing the active trainer. Archive SHA-256
  `be38baa3c117f4b3e6fdd8f943832a45ddb739f63816a024bb39a5812a401552`.
  Added a sequential finalization controller with per-phase commands, logs and
  completion markers; made the loss W&B bridge recognize a budget-limited
  secondary refit; final publication removes validation-selection summary keys
  that do not apply to full-data refits. Archive/file checksums and remote
  Python/shell syntax checks passed.
  [Evidence](smollm2-gsm8k-research-artifacts/finalization-v3-staging.log).
- 2026-11-02T23:03Z: Trial 08 (1e-3, batch 16) completed 12 epochs and became
  the validation-accuracy leader. Selected epoch 11: strict 100/747 =
  0.13386880856760375, train NLL 0.0003800072524835822 and validation NLL
  1.4551265762959849. Minimum validation NLL was 0.7426382344112419 at
  epoch 2, so trial 04 remains the loss leader. The selected full optimizer
  checkpoint is complete and mirrored; SHA-256
  `91967f14a1500201d082abe5c6c2075170bfde590f42937f0d51370a45ea2876`.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-08-completion.log).
  [Trial 09](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/vghugnuh)
  started at learning rate 1e-3, batch 64. Independent W&B API readback
  confirmed trials 06/07 finished and trial 08 metrics uploaded while running;
  [evidence](smollm2-gsm8k-research-artifacts/wandb-trials-06-to-08-verification.log).
- 2026-11-02T23:14Z: Trial 09 (1e-3, batch 64) completed 12 epochs.
  Selected epoch 10: strict 87/747 = 0.11646586345381527, train NLL
  0.0009209335214592311 and validation NLL 1.264095531628923. Minimum
  validation NLL was 0.6686096988345192 at epoch 2. Checkpoint 09 is complete
  and mirrored. [Evidence](smollm2-gsm8k-research-artifacts/trial-09-completion.log).
  Batch 16 remains the accuracy leader across the tested 8/16/32/64 batch
  sizes. [Trial 10](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/fygcsg1w)
  started with learning rate 5e-4, batch 16; other settings match trial 08.
- 2026-11-02T23:21Z: Before any research test evaluation (main trial 10 is
  still running), revised only the optional loss-probe time gate: start with
  at least 30 minutes remaining and reserve 20 minutes for finalization,
  previously 40/30 minutes. Measured prior full five-shot benchmarks took
  61–119 seconds and the low-rate validation trials establish sufficient
  throughput for this reserve. The two probe coefficients, objectives,
  validation-only selection, maximum epochs and original four-hour deadline
  are unchanged. No active trainer source was changed.
  New immutable loss-refinement v2 archive SHA-256:
  `de4ebfa5f53200bcda55a63fadd6f4314c4f27842d92bfd02ce030d877c7dee6`;
  matching finalization v4 archive SHA-256:
  `c51e4d432ac7a9f31a8c8f5d1d7f65e93505cbdef013a61b512031fc61bd11fe`.
  Both are staged in versioned node-local directories and mirrored in the
  same-region study bucket. Checksums and remote syntax checks passed;
  [evidence](smollm2-gsm8k-research-artifacts/finalization-v4-staging.log).
- 2026-11-02T23:26Z: Trial 10 (5e-4, batch 16) completed 12 epochs.
  Selected epoch 11: strict 95/747 = 0.12717536813922356, train NLL
  0.00048350529045466337 and validation NLL 1.300386004568308. Minimum
  validation NLL was 0.6651405471669258 at epoch 2. Checkpoint 10 is complete
  and mirrored. [Evidence](smollm2-gsm8k-research-artifacts/trial-10-completion.log).
  [Trial 11](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/zciiw1w6)
  started at learning rate 2e-3, batch 16; all other settings match trial 08.
- 2026-11-02T23:37Z: Trial 11 (2e-3, batch 16) stopped after 11 epochs.
  Selected epoch 8: strict 76/747 = 0.10174029451137885, train NLL
  0.013031595159945758 and validation NLL 1.4435148597312824. Minimum
  validation NLL was 0.8652711909874359 at epoch 3. Checkpoint 11 is complete
  and mirrored. [Evidence](smollm2-gsm8k-research-artifacts/trial-11-completion.log).
  [Trial 12](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/81s96g6x)
  started at learning rate 1e-3, batch 16, with weight decay disabled (0).
- 2026-11-02T23:49Z: Trial 12 (1e-3, batch 16, weight decay 0) completed
  12 epochs. Selected epoch 12: strict 99/747 = 0.13253012048192772, train
  NLL 0.00022170160811642555 and validation NLL 1.4943920948715925.
  Minimum validation NLL was 0.7447056598930192 at epoch 2. This is only one
  correct answer below trial 08, not evidence of a robust weight-decay effect.
  Checkpoint 12 is complete and mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-12-completion.log).
  [Trial 13](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/gtr8wy6d)
  started at learning rate 1e-3, batch 16, weight decay 0.1.
- 2026-11-03T00:01Z: Trial 13 (1e-3, batch 16, weight decay 0.1) completed
  12 epochs. Selected epoch 10: strict 93/747 = 0.12449799196787148, train
  NLL 0.0011541637892672062 and validation NLL 1.1625307081955532.
  Minimum validation NLL was 0.7474484878883112 at epoch 2. Stronger decay
  reduced later-checkpoint NLL but did not improve the maximum strict accuracy.
  Checkpoint 13 is complete and mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-13-completion.log).
  [Trial 14](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/xk5tfu5m)
  started from the accuracy-leading recipe (1e-3, batch 16, weight decay 0.01),
  changing final-answer training weight from 1 to 2. Reported full-split NLL
  remains unweighted.
- 2026-11-03T00:12Z: Trial 14 (1e-3, batch 16, final-answer weight 2)
  completed 12 epochs. Selected epoch 10: strict 79/747 =
  0.10575635876840696, train NLL 0.0012095336933283535 and validation NLL
  1.420709709400245. Minimum validation NLL was 0.7423473840934076 at
  epoch 2. Checkpoint 14 is complete and mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-14-completion.log).
  [Trial 15](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/e24txzcp)
  started with final-answer weight 4, preserving trial 08's other settings.
- 2026-11-03T00:21Z: Trial 15 (1e-3, batch 16, final-answer weight 4)
  completed 12 epochs. Selected epoch 9: strict 89/747 =
  0.11914323962516733, train NLL 0.0023054226159434647 and validation NLL
  1.3629977885371993. Minimum validation NLL was 0.734385522185702 at
  epoch 2. Neither tested answer-weight increase improved the accuracy leader.
  Checkpoint 15 is complete and mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-15-completion.log).
  [Trial 16](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/pjale36r)
  started from trial 08's recipe, increasing warmup fraction from 0.03 to 0.1.
- 2026-11-03T00:34Z: Trial 16 (1e-3, batch 16, warmup fraction 0.1)
  completed 12 epochs. Selected epoch 12: strict 79/747 =
  0.10575635876840696, train NLL 0.00021616009615237804 and validation NLL
  1.4404478810329133. Minimum validation NLL was 0.7374785977187819 at
  epoch 2. Checkpoint 16 is complete and mirrored. An initial read preceded
  publication of the summary; the subsequent read verified completion.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-16-completion-verified.log).
  [Trial 17](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/vc7p45ix)
  started from trial 08's recipe, reducing the terminal/peak learning-rate
  ratio from 0.1 to 0.01.
- 2026-11-03T00:45Z: Trial 17 (1e-3, batch 16, terminal learning-rate
  ratio 0.01) completed 12 epochs. Selected epoch 10: strict 86/747 =
  0.11512717536813923, train NLL 0.0003964405434129997 and validation NLL
  1.4174414381793912. Minimum validation NLL was 0.7396795655463128 at
  epoch 2. Checkpoint 17 is complete and mirrored.
  [Evidence](smollm2-gsm8k-research-artifacts/trial-17-completion.log).
  [Trial 18](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/wr2teuhx)
  started from trial 08's recipe with a 20-epoch schedule/horizon; the original
  search deadline and early-stopping rule still apply.
- 2026-11-03T00:55Z: Trial 18 stopped after 10 epochs without improving
  either objective for three epochs. Selected epoch 7: strict 71/747 =
  0.09504685408299866; minimum validation NLL 0.7404463763085245 at epoch 2.
  Checkpoint 18 is complete and mirrored. The main search then locked its
  selection after 18 completed candidates, before opening the test split:
  trial 08, epoch 11 (strict 100/747, validation NLL 1.4551265762959849).
  Planned seed variants 19/20 were not started because the search's start
  guard preserves the finalization reserve. The full-data refit uses all
  7,473 original training examples, learning rate 1e-3, batch 16, weight decay
  0.01, 3% warmup, terminal ratio 0.1, 12-epoch schedule horizon, 11 actual
  epochs, ordinary answer weighting, and seed 1234.
  [Immutable selection/configuration evidence](smollm2-gsm8k-research-artifacts/main-search-selection.log).
  [Final accuracy refit on W&B](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/wril0ygi).
  Full train/test NLL curves during this fixed refit are diagnostic; they do
  not change the locked recipe or epoch count.
- 2026-11-03T01:05:44Z: The fixed accuracy refit completed successfully
  (exit 0), 11 epochs and 5,148 optimizer updates. Final full-dataset NLL:
  train 0.0003099812218933697 over 937,677 answer tokens / 7,473 examples;
  test 1.437200022346236 over 168,823 answer tokens / 1,319 examples.
  The checkpoint is complete and mirrored. The W&B bridge finalized all
  18 search trials plus the accuracy refit. Main source checksums passed.
- 2026-11-03T01:06Z: Fresh filtered TPU inventory again listed the exact
  authorized Spot v5e-8 in its final AVAILABLE table. VFIO inspection found
  no holders. The old monitor and inventory process both exited successfully.
  Launched the previously staged `finalization-v4/run_finalization.sh`,
  controller PID 101959, with scalar-only loss-study W&B logging and a new
  30-second read-only monitor. It will run the predeclared loss probe(s) only
  within their remaining-budget gate, refit the validation-loss winner, then
  export and benchmark both selected models sequentially.
  [Inventory](smollm2-gsm8k-research-artifacts/finalization-tpu-inventory.log),
  [preflight](smollm2-gsm8k-research-artifacts/finalization-preflight.log),
  [launch](smollm2-gsm8k-research-artifacts/finalization-launch.log).
- 2026-11-03T01:14Z: Loss probe 01 (trial 04's recipe plus label smoothing
  0.05) completed six epochs. Lowest unweighted validation NLL was
  0.6466419589850799 at epoch 3, above the unsmoothed trial 04 minimum
  0.6045382790792161. Its best accuracy was 76/747 at epoch 6, also below
  the accuracy leader. The 0.1 smoothing probe was not started because only
  1,615 seconds remained, below the predeclared 1,800-second start gate.
  The secondary selection locked trial 04 at its minimum-NLL epoch 3,
  independently of the measured test curves. The full-data loss refit is
  learning rate 1e-4, batch 32, three actual epochs, 12-epoch schedule horizon,
  weight decay 0.01, 3% warmup, terminal ratio 0.1, seed 1234, and no label
  smoothing or extra answer weight.
  [Selection evidence](smollm2-gsm8k-research-artifacts/loss-selection.log),
  [loss refit W&B](https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/d8ml8ua1).
- 2026-11-03T01:19Z: The fixed loss refit completed its three epochs and
  702 updates. Final full-data NLL is train 0.35753009805707014 and test
  0.6010654320418056, with the same 937,677 training / 168,823 test answer
  tokens as the accuracy refit. Its epoch-2 test loss was slightly lower
  (0.5985240209484147); the selected three-epoch checkpoint is retained
  because stopping was determined by validation, not test loss.
  Started a separate CPU-only, read-only W&B reconciliation of every
  completed optimizer-step history and full-split loss curve while final
  model export/evaluation continues sequentially on the TPU.
- 2026-11-03T01:25Z: Both exported models passed exact weight comparison,
  long-position PyTorch/TPU logit checks, greedy generation and stop checks.
  Matched five-shot strict test scores: accuracy-selected 13/1,319 =
  0.009855951478392721; loss-selected 64/1,319 = 0.04852160727824109.
  All 2,638 per-filter records matched the archived baseline's document,
  prompt and target hashes. Fixed zero-shot test scores were respectively
  153/1,319 = 0.11599696739954511 and 96/1,319 = 0.07278241091736164.
  The accuracy-selected model's validation gain transfers to the zero-shot
  test protocol but not to the historical five-shot prompt format. Neither
  model recipe was changed in response to these test results.
- 2026-11-03T01:25Z: Corrected a post-training W&B API pagination verification
  failure. Repeated page-boundary rows had identical internal IDs and values;
  no optimizer steps were missing. Five focused remote tests passed, then all
  21 run histories reconciled: 80,186 updates and 241 full-split loss records.
  The original controller's publication-check exit 1 is preserved and will be
  paired with a distinct successful publication-recovery record.
  [Immutable postmortem](../2026-11-03/smollm2-wandb-pagination.md).
- 2026-11-03T01:27Z: To interpret the observed prompt-format difference,
  added a matched zero-shot measurement of the unchanged base and original
  three-epoch model, using the same existing evaluator, all 1,319 test
  examples, greedy decoding and 256-token limit. This is additional baseline
  measurement, not a selection or tuning step. Confirmed all VFIO devices idle
  before launch; process PID 121454 runs on the same authorized Spot TPU.
  Selected final-report source archive SHA-256:
  `0e6e359c0e5b2c3e7d21a583c2f92d792264047bdc7cc54f570064f1933cf2d3`.
  [Source/preflight evidence](smollm2-gsm8k-research-artifacts/final-report-preflight.log).
- 2026-11-03T01:29:00Z: All model computation finished successfully within
  the four-hour budget. Matched zero-shot baselines: base strict 0/1319,
  flexible 23/1319; prior fine-tune strict 74/1319, flexible 75/1319.
- 2026-11-03T01:30:43Z: Corrected scalar publication completed and verified
  both final W&B summaries. Original failed publication evidence remains intact.
  Subsequent checks confirmed READY/HEALTHY, no tracked experiment processes,
  and every VFIO device idle. Canonical logs were checksummed.
- 2026-11-03T01:37Z: Delivered the checksum-verified scalar report and PNG/SVG
  charts at `http://localhost:8769/`, with a static server kept running for
  result inspection. The broad raw-archive GCS upload was rejected by automatic
  approval review and not attempted again; the raw evidence remains node-local.
  Requested W&B publication, model training/evaluation, reports and focused
  checks are complete. The Spot TPU remains allocated and idle.
