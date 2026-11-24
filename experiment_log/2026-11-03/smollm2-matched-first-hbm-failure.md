# SmolLM2 matched validation: first long-context backward HBM failure

## Run Summary

- Status: failed.
- Attempt: search-first; candidate matched-02, first epoch, zero optimizer updates completed.
- UTC failure: 2026-11-03T03:51:46.202192Z; supervisor exit 1.
- TPU: `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`, he-vision-group/us-west4-a, Spot v5litepod-8, one VM/eight chips. TPU remains READY/HEALTHY; subsequent filtered AVAILABLE audit and idle VFIO/lockfile check passed.
- Command: `/home/kristinelu/smollm2-gsm8k/runtime/venv/bin/python /home/kristinelu/smollm2-gsm8k/matched-validation-20261103/training-source-v2/matched_train.py --attempt-id first`.
- Stage: first training-step backward compilation for five-shot mixture.
- Source: immutable training-source-v2 manifest `ea33cf63bbb4126e9fc1c8136f11cc493c4c41387710fc1870f23aa116ad75c2`.

## Configuration

Frozen authorized protocol `smollm2-matched-validation-authorized.json`; matched-02 LR 1e-4, batch 32, 50% five-shot prompts, 1920-token untruncated sequence width, float32, seed 1234. Per-chip batch 4. All scientific settings unchanged.

## Failure Details

`RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 21.76G of 15.75G hbm. Exceeded hbm capacity by 6.01G.` Multiple decoder attention intermediates each occupy 506.25 MiB at shape f32[4,9,1920,1920]. This is backward activation HBM pressure; forward-only parity/generation preflight had passed. The full long-context backward gate did not pass, and no matched-02 trained checkpoint exists.

Newest valid predecessor checkpoint: matched-01 epoch 7, optimizer step 1477; fixed-reference epoch 3 and all control epochs are mirrored with verified completion markers and object sizes in us-west4. Source/data/output were node-local. Do not rerun completed reference/control work.

## Recovery

Investigating parameter-free rematerialization or exact token-weighted gradient accumulation, preserving effective batch, query order, seed, LR schedule, prompt construction and losses. Require remote loss/gradient parity and a new immutable retry source/output identity before continuation. Original four-hour deadline remains 07:16:34.858111 UTC. No additional hardware or provisioning.

## Evidence

Log: `smollm2-matched-validation-artifacts/search-first-failed-console.log`, SHA-256 `4ab59552b108c0d116acc3f880c59e4823935cfe71724c6b03bd6a5ba6ad6370`. W&B read-back of all 19 audits and completed reference/control histories passed (633 + 1477 exact optimizer updates); failed candidate status must be recorded separately from completed work.

- 2026-11-03T04:02:41.057650+00:00: Selected repair is exact global-token-weighted gradient accumulation, effective batch 32 unchanged, global microbatch 8, four accumulation steps. Gradients are summed before global reduction, clipping and a single AdamW update; query coverage/order and optimizer schedule are unchanged. Pinned attention dropout must equal zero. Recovery source training-source-v3 SHA-256 `ad401734ecda702bde838ce8e9d8625b2c07461862b650cf11fa8eec6ba0b8da` passed 17 remote CPU tests, including loss/gradient/optimizer-update equivalence for 1/2/4 accumulation factors, unequal query masks and an all-padding microbatch. Accelerator-specific confirmation pending.

- 2026-11-03T04:10:32.353939+00:00: Repair verified on the actual 1920-token mixed-prompt model workload: retry matched-02-r1 reached update 144 with finite loss/gradient values. Runtime peak reserved bytes per chip 7,747,846,144 (7.215743064880371 GiB) versus physical limit 16,909,332,480 (15.748043060302734 GiB). Busiest chip observed peak bytes in use 3,359,948,800. Old/new epoch-one coverage files have identical SHA-256 `814039e2f4c5f3c0de6af391c089da0f457af40d9136bbf87f69bfdaa13331ef`. Fresh retry W&B https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/9lprie66; failed original W&B https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/conlangs/runs/fhku02zx has native failed state. Training continues under original deadline.

- Unit-conversion correction: the exact byte values above correspond to 7.2157440185546875 GiB reserved and 15.748043060302734 GiB physical. Use these computed conversions; the byte measurements are unchanged.
