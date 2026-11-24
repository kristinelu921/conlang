# frozen-20 — prepared

Selected source: `bf6f8d12c63a13442f8227c353e219bcf186ba25` (published to https://github.com/kristinelu921/island-community/commit/bf6f8d12c63a13442f8227c353e219bcf186ba25). Archive SHA-256 `07a2b097d18eecb0d23076d3e041c876ae5a69d06f8fd544848b6eb7cb0df44c`. Exact launcher and environment: `frozen-20.sh`. Selected existing authorized TPU: `kmh-tpuvm-v5e-4-spot-myang-ar`, `us-west4-a`, v5litepod-4 / 2x2 / one worker, Spot. Launch requires fresh cloud identity and all-device idle checks.

Mode: `frozen`. Total target: 20 turns, seed 42, 10 founders, pinned Qwen3-0.6B, float32, private eight-turn history, temperature 0.6, top-p 0.95, top-k 20, 96 reasoning tokens/stage, 16,384-token context, four CPU threads, eager attention, 512-token prompt chunks. World/reward configuration is identical in all arms. All data, model, checkpoint and source stay node-local in this region.

Continuation: frozen-pretrained-gate2; checkpoint SHA-256: 190c891e89d8cc3d6096491738c905103046d1e3e181dfd47f3ba38a075d254c. Requested additional turns: 19. Resumed arms preserve exact configuration, histories, optimizer moments and world state. Only the value arm adds a per-island width-64 value MLP with Adam 1e-3 and discount 0.99. Both learning arms retain rank-eight adapters on query/value projections in the last two layers, Adam 1e-5, reward scale 0.1, clip 1. Frozen has no updates. Third-arm choice follows the stated default after the optional clarification was unanswered; no reward labels are learned or replaced.

Runtime outputs: `/tmp/island-community-qwen-v5e4-20261111-024900/frozen-20-recording`; exact command is recorded in its manifest. Every-turn checkpointing and bounded checksum-verified local mirroring are required. Pretrained frozen and reward first-turn gates passed; six focused TPU tests cover all modes, gradients/isolation, actual disk restore and full recorder continuation. Results and material transitions will be appended after verification.

Selected source updated before any launch to published `e9f1a64bdff3e93a4f5010f915529bd286526b56`, archive SHA-256 `f78be030b779ec12e5bbf5c3e2910a7ba0346ae8c39426768f3da15419ec77a0`, to include the tested attention memory repair. The old prepared launcher is retained as `frozen-20-prepared-bf6f8d1.sh`. Scientific configuration is unchanged; seven remote TPU tests passed.

2026-11-11T06:24:34.561157+00:00: Launched the committed exact command on the freshly verified authorized resource; every-30-second monitor PID 2020823.
