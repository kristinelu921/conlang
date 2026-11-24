# Pretrained reward-adapter gate 2 — running

Fresh preflight 2026-11-11 03:59:04 UTC confirmed the exact authorized `kmh-tpuvm-v5e-4-spot-myang-ar` in `us-west4-a`: READY/HEALTHY/Spot, topology 2x2, one worker IDLE. Evidence: `reward-gate2-preflight.json`. Prior test cleanup was limited to the current-user empty lock with the recorded timestamp and no device holders (`cleanup-check6-lock.txt`).

Published source commit: `fed6e39bdd8ea3b8d86c7094e4adec7ff62a7fae`, https://github.com/kristinelu921/island-community/commit/fed6e39bdd8ea3b8d86c7094e4adec7ff62a7fae. Archive SHA-256 `e45d38f88292464ea3da39b62f4e22173741a2ced213814326b19f89d9ed96fc`; source committed/pushed before staging and verified before extraction to `source-fed6e39`.

Exact launcher/environment: `reward-pretrained-gate2.sh`. Command: `python -u -m island_community run --config island_community/scenarios/qwen_10_reward_adapter.json --turns 1 --device tpu --checkpoint-every 1 --output /tmp/island-community-qwen-v5e4-20261111-024900/reward-pretrained-gate2-recording`.

All scientific settings match failed adapter gate 1: seed 42, ten founders, the same world and reward rules, frozen pinned Qwen3-0.6B, independent rank-eight query/value LoRA adapters on the last two layers, Adam 1e-5, reward scale 0.1, gradient clip 1, float32, 96 reasoning tokens/stage, temperature 0.6, top-p 0.95, top-k 20, context 16,384, four CPU threads. Native attention is eager with 512-token prefill chunks and optional generation compilation disabled.

The learning repair saves actual post-processor token supports during generation and differentiates logits on those exact sets, avoiding a different replay-time top-k/top-p cutoff. Six focused TPU tests passed, including the rank-change regression, real gradients/isolation, actual disk checkpoint restoration, and the complete recorder workflow. The previous failed attempt is preserved separately.

All source/model/output paths remain node-local; no training dataset or checkpoint crossed regions. Authenticated AES-GCM SSH is used after default-transport message-authentication failures; integrity checks remain enabled. `monitor_qwen.py reward-pretrained-gate2` runs every 30 seconds, checks exact cloud ID/state, exact run processes, latest metrics/checkpoint, bounded recording size/free space, and copied-checkpoint SHA-256. Its PID/log/status files are named `reward-pretrained-gate2-monitor.*`.

This is a first-update gate. Successful pretrained learning, adapter-change norms, action audits, timings, and checkpoint status will be appended only after verification. The third scientific condition remains pending user clarification.

## Completed and audited — 2026-11-11 04:10 UTC

Exit 0; wall time 266.68 seconds, including 203.75 seconds of generation. All ten requested actions matched submission, world execution, and recorded experience; all ten executed. Actions were identical to the frozen TPU first turn: 2 wait, 4 manufacture, 3 use item, and 1 trade proposal. All ten islands survived with mean health 300.0; total reward 5.07.

Ten independent adapter updates completed. Every island has a positive adapter-change L2 norm (0.00221657–0.00221679). The shared base SHA-256 is unchanged: `f78b762477299bdb4b1d10c627e8976f662fca37e296f23eb74e45f98353c3ab`. This verifies the update path; one turn does not establish learning effectiveness or improved trade.

The complete 12,052,741-byte recording was mirrored and audited locally. Checkpoint: 9,970,559 bytes; matching remote/local SHA-256 `a1732f1701ea92dcf87c8d57cde5e86ab6554d156b350477549d08221938f4c2`. Evidence: `reward-pretrained-gate2-analysis.json`, `reward-pretrained-gate2-monitor-latest.json`, and the full recording. All eleven reasoning stages hit their 96-token budget.
