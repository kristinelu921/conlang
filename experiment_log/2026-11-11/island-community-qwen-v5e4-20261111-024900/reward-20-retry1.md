# Reward adapter continuation retry 1 — prepared

Published source `e9f1a64bdff3e93a4f5010f915529bd286526b56`, https://github.com/kristinelu921/island-community/commit/e9f1a64bdff3e93a4f5010f915529bd286526b56; archive SHA-256 `f78be030b779ec12e5bbf5c3e2910a7ba0346ae8c39426768f3da15419ec77a0`. Exact command/environment `reward-20-retry1.sh`. Same authorized `kmh-tpuvm-v5e-4-spot-myang-ar`, us-west4-a, v5litepod-4 / 2x2 / one worker / Spot; requires fresh READY/HEALTHY identity and idle probes.

Resume the verified original turn-one checkpoint from `reward-pretrained-gate2-recording/checkpoint.pt`, SHA-256 a1732f1701ea92dcf87c8d57cde5e86ab6554d156b350477549d08221938f4c2, for 19 additional turns to turn 20. This discards only the failed attempt's uncommitted in-memory turn-two updates; every prior artifact is preserved. Ten founders, seed 42, private eight-turn history, frozen pinned Qwen3-0.6B float32, rank-eight last-two-layer query/value adapters, independent Adam 1e-5, clip 1, reward scale 0.1, temperature 0.6, top-p 0.95, top-k 20, context 16384, reasoning 96 tokens, four CPU threads.

Only backend memory handling changes: 512-query attention blocks retain all keys/values and the original causal mask; gradient-bearing blocks recompute intermediates. All seven remote tests passed, including same-backend output/gradient parity, CPU-logit/cached-decoding comparison, isolation and disk continuation. Check7's cross-backend discrepancy and check8's transient initialization failure are preserved separately. No dtype, reward, model, context, sampling, or learning hyperparameter changed.

All source/model/checkpoints/output remain node-local; no cross-region data copy. Monitor every 30 seconds, checkpoint every complete turn, checksum-verified bounded local mirrors. A successful resumed update, latest metrics and any failure will be appended when verified.

## Resumed turn two validated — 2026-11-11 04:37 UTC

The full pretrained run passed the prior memory failure: all ten islands completed their second independent adapter updates with finite losses/gradient norms and positive parameter-change norms. All ten requested actions matched submission, world execution, and saved experience; all executed. Actions: 1 wait, 6 manufacture, 3 trade proposals. Each prompt correctly restored one private prior turn. Exact audit: `reward-20-retry1-turn2-analysis.json`. The new checkpoint is 9,991,679 bytes; remote and mirrored SHA-256 `7308514d51267b38bfc89a475f388e67823517944e18da6844328dd187355e2a`. Turn three is in progress; final base-weight rehash and the full 20-turn comparison remain pending.

The comparison dashboard and replay endpoints are verified at http://127.0.0.1:35403/. Intermittent transport failures persist but retries now preserve the last good replay and eventually verified the turn-two copy. A separate AES-CTR + HMAC diagnostic copied the same checkpoint with matching SHA-256; this does not establish the cause of intermittent failures.

2026-11-11T06:24:01.684097+00:00: Completed turn 20; action/base-weight audit and checkpoint checksum passed. All processes exited; authorized cleanup removed only our empty unheld lock. Full results: reward-20-retry1-analysis.json
