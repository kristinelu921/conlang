# Pretrained reward-adapter gate 1 — running

Fresh preflight 2026-11-11 03:41:14 UTC: exact authorized `kmh-tpuvm-v5e-4-spot-myang-ar`, `us-west4-a`, v5litepod-4, 2x2, Spot, READY/HEALTHY, one worker IDLE. Evidence: `reward-gate1-preflight.json`. The preceding test's empty current-user lock was cleaned under the explicit authorization and fresh device-idle checks (`cleanup-check5-lock.txt`).

Published source commit: `7ac1c85084269fd54471f6cd2448c1b64c19a6ba`, https://github.com/kristinelu921/island-community/commit/7ac1c85084269fd54471f6cd2448c1b64c19a6ba. Archive SHA-256: `3d848934f26729fe1db2ace87e8228241cafb7447a8cad8348f4eeb202854012`; node-local source `source-7ac1c85`. Source was committed and pushed before staging. Exact launcher: `reward-pretrained-gate1.sh`.

Command: `python -u -m island_community run --config island_community/scenarios/qwen_10_reward_adapter.json --turns 1 --device tpu --checkpoint-every 1 --output /tmp/island-community-qwen-v5e4-20261111-024900/reward-pretrained-gate1-recording`.

The scenario is structurally identical to the frozen scenario except `language_model.mode=reward_adapter`: seed 42, ten founders, health 300, the same gifts/world/rewards, pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, float32, 96 reasoning tokens/stage, temperature 0.6, top-p 0.95, top-k 20, context 16,384 tokens, and four CPU threads. Native backend: eager attention, 512-token prompt chunks, static cache, generation auto-compile disabled.

Each island gets rank-eight LoRA factors on query/value projections in the last two transformer layers, its own Adam optimizer at 1e-5, gradient clipping at 1, and immediate reward scale 0.1. Shared pretrained weights stay frozen. This gate checks real-model backward memory, nonzero independent adapter updates, action mapping, and persisted optimizer state. The tiny-model gradient/isolation tests and actual disk-checkpoint restoration passed before launch. Full-model learning is not claimed until this run's recorded updates and checkpoint are audited.

All source/model/output paths are node-local, and the pinned model was downloaded directly to this VM. No cross-region dataset/checkpoint copy. The CPU frozen pilot remains running; the full matched comparison will use the same TPU backend across arms. The third scientific condition remains awaiting the user's clarification between an added value baseline and a no-history frozen control.
