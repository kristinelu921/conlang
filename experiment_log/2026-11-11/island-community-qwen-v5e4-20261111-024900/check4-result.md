# Native TPU check 4 — passed

All **5 tests passed in 47.79 seconds**, exit 0, on the authorized v5e-4 in `us-west4-a`. Complete output: `native-check4.log`; exact launcher: `native-check4.sh`; source archive SHA-256: `34cc741b2afbaa675210ae53d9d4fd3791885a125f0934c19bb9f59cdab5719d`.

The repaired native backend uses eager attention, a static cache, 512-token prompt chunks, and `disable_compile=True`. No weights, prompts, sampling parameters, histories, world rules, or rewards changed. Runtime metadata records the selected attention implementation, chunk size, and compile setting.

The new regression uses Qwen's actual attention head geometry: 16 query heads, 8 KV heads, dimension 128, with a smaller one-layer model for focused testing. TPU forward logits agree with the CPU reference within rtol/atol 0.005; three-token greedy decoding with a 600-token prompt and 512-token chunks matches exactly. Existing tests verify immutable shared weights, separate adapter updates and optimizer state, inheritance, deterministic checkpoint continuation, and exact requested/submitted/executed action records in a ten-island turn plus a resumed second turn.

Five warnings were emitted: one experimental launch API warning and non-leaf `.grad` access warnings in the third-party attention implementation. Adapter change and isolation assertions passed. No VMEM halt, CPU-scalar compile failure, or shared log-directory permission error occurred.

This verifies the backend repair with focused models. The full pretrained 0.6B recorded-turn retry is the next gate; the earlier failed pretrained attempt remains separate and preserved. The postflight lock is current-user owned and empty, timestamp `2026-11-11 03:23:42.888776490 +0000`; cleanup requires the already-authorized fresh idle check.
