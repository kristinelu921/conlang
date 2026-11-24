# Native TPU disk-checkpoint check — passed

The three parameterized frozen/reward-adapter/value-adapter tests passed with actual `torch.save` to disk and `torch.load(map_location="cpu")` before restoration to TPU: **3 passed, 2 deselected, 19.25 seconds**, exit 0. Exact launcher: `native-check5.sh`; log: `native-check5.log`; source archive SHA-256: `781937116dc9000a739ff5d344f8eee92aee89ac418ccd61f99b26fa2fc50426`.

This extends the prior in-memory restore test to cover serialized independent adapter weights, Adam moments, optional critic state, private history, and deterministic continuation. It runs on the same authorized v5e-4 in `us-west4-a` with the repaired eager/chunked native backend and unchanged pinned packages.

The new `qwen_10_reward_adapter.json` scenario was validated and compared structurally with `qwen_10_frozen.json`: the only difference is `language_model.mode=reward_adapter`. It retains seed 42, ten founders, the same world/gifts/rewards, sampling settings, rank-eight per-island adapters on the last two query/value projections, Adam learning rate 1e-5, gradient clipping at 1, and reward scale 0.1. A pretrained update remains to be checked before the longer adapter run.

Empty current-user postflight lock timestamp: `2026-11-11 03:37:32.336040747 +0000`. Future cleanup remains subject to the explicit user authorization and fresh all-device idle verification.
