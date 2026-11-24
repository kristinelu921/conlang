# Attention memory regression check — prepared

Source archive SHA-256 `c65821e1617739de6c1f84873815c02db67094ce9ab67e49215ff18b6e5e0a7b`; baseline published commit bf6f8d1 plus the local query-chunking/activation-checkpointing patch. Exact environment and command: `native-check7.sh`. Authorized v5litepod-4 / topology 2x2 / single worker, `kmh-tpuvm-v5e-4-spot-myang-ar`, us-west4-a. No world or learning hyperparameters changed.

Run seven focused neural tests on remote TPU only. New regression compares 577-token uneven query chunks, all 16 query heads / 8 KV heads / 128 head dimension, outputs and query/key/value gradients against ordinary eager CPU attention with the identical causal mask and random seed 128. Existing full-head model-forward and cached-generation parity test now exercises the registered chunked backend. Adapter isolation, all modes, real disk restore, recorded sampling supports and recorder continuation remain covered. Outcomes and timings are pending.

## Result — 2026-11-11 04:20 UTC

Six passed, one failed in 52.32 seconds, exit 1. All existing adapter, disk-checkpoint, full-model CPU-logit/cached-decoding and recorder tests passed. The new raw attention CPU-versus-TPU output comparison failed for 18 of 1,181,696 elements at atol/rtol 0.005 (maximum absolute difference 0.0140759); gradient comparison was not reached. This does not yet isolate a chunking error from baseline CPU/TPU numerical differences. Follow-up will compare chunked and ordinary eager attention on the same backend, retain the independent end-to-end CPU parity test, and measure cross-backend output and gradient errors. Do not declare equivalence or loosen thresholds without that evidence.
