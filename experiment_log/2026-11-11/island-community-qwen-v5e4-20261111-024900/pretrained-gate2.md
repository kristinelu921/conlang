# Pretrained frozen TPU gate 2 — running

Fresh exact-resource preflight at 2026-11-11 03:27:08 UTC confirmed the authorized `kmh-tpuvm-v5e-4-spot-myang-ar` in `us-west4-a` remains READY/HEALTHY/Spot, with its sole worker IDLE. Evidence: `pretrained-gate2-preflight.json`. The preceding test lock was removed only after the user-authorized ownership, exact timestamp, and all-device idle checks (`cleanup-check4-lock.txt`).

Source commit `f23846d9ef73c681848b5779a34585d9b1b8ac54` was pushed to the private repository before staging: https://github.com/kristinelu921/island-community/commit/f23846d9ef73c681848b5779a34585d9b1b8ac54. Node-local source is `source-f23846d`; source archive SHA-256 `e55e5bc085dbbd64fd73b55381d77ee217a17e069addc5df1fc90171023185ca` is checked before extraction.

Exact launcher: `frozen-pretrained-gate2.sh`. It runs `python -u -m island_community run --config island_community/scenarios/qwen_10_frozen.json --turns 1 --device tpu --checkpoint-every 1 --output /tmp/island-community-qwen-v5e4-20261111-024900/frozen-pretrained-gate2-recording`.

Scientific configuration matches failed gate 1: pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, float32, seed 42, ten founders, frozen weights, 96 reasoning tokens per stage, temperature 0.6, top-p 0.95, top-k 20, 16,384-token context, four CPU threads. All world and reward settings are the unchanged committed scenario. The only backend changes are eager attention, 512-token prompt chunking with full context retained, and disabled optional generation auto-compilation. Five focused TPU tests, including full attention-head geometry CPU parity and complete recorder/resume checks, passed before launch.

All source/model/output paths are node-local; the model was downloaded directly to this VM. No dataset or checkpoint crossed regions. Gate 1 and failed check 3 are preserved separately. This is a pretrained correctness/runtime gate, not the full three-condition experiment comparison. Completion, action audit, final frozen-weight hash, timings, and any failure will be appended after inspection.

## Completed and audited

The run completed at approximately 03:31 UTC with exit 0 in **207.84 seconds**, including model load and checkpointing. Recorded generation time was **188.21 seconds** for 1,123 generated tokens across 11 stages. All 11 reasoning stages reached the configured 96-token budget, so the recorded reasoning should not be interpreted as unconstrained complete deliberation.

All **10 of 10 requested actions** exactly matched submitted and executed world actions; no formatting failures or world rejections occurred. Actions: 2 wait, 4 manufacture, 3 use item, 1 trade proposal. All ten islands survived with mean health 300.0. One trade was proposed and none accepted in this first simultaneous turn; that is not evidence of later trading effectiveness. Total reward was 5.07. Learning updates: 0.

The base-weight SHA-256 was identical before and after: `f78b762477299bdb4b1d10c627e8976f662fca37e296f23eb74e45f98353c3ab`. The runtime reports 751,632,384 parameter entries after native-device conversion; this count includes the device runtime's representation and should not be assumed to be the model card's nominal parameter count without checking tied-weight identity.

Full recording: `frozen-pretrained-gate2-recording/`; audit: `frozen-pretrained-gate2-analysis.json`; timing: `frozen-pretrained-gate2-timing.txt`. Rsync/stream transfers disconnected. Automatic approval review then rejected an unbounded recursive copy due to filesystem capacity. A read-only size check established the completed recording was 2,135,434 bytes versus 5,654,970,368 bytes free at the destination; the bounded copy was approved and completed, and the audit passed against the retrieved files. No extra user permission was needed after establishing those bounds.

The separate CPU baseline remains running and monitored. The matched scientific comparison should use the same TPU backend for all arms; CPU pilot trajectories are not interchangeable with TPU trajectories because numerical differences can change sampled continuations.
