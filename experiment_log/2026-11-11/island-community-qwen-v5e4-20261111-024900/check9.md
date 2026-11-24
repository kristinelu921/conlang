# Attention diagnostic after transient initialization failure

Identical immutable source `source-memory-check2`, SHA-256 59e44c1228115754351e41d6457118a12d7f2762018ac879f5b3cf290e460ee4, exact command/environment `native-check9.sh`. Check8 performed no inference due to an initialization-in-use report. Multiple subsequent probes found no device holder or lock; fresh cloud identity and idle checks are required immediately before this retry. No reboot, reset, process signaling, or scientific setting change. Outcome pending.

## Passed — 2026-11-11 04:26 UTC

Seven tests passed in 49.39 seconds, exit 0. Ordinary eager TPU versus CPU raw-attention output relative L2 error was 0.003026; query/key/value gradient errors were 0.003956, 0.003911, and 0.003077. The original 0.0140759 maximum output discrepancy also occurs in ordinary eager TPU attention, establishing that it is not caused by query chunking.

Chunked versus ordinary eager on TPU: output relative L2 1.1389e-5, query/key/value gradients 2.0091e-5, 2.2318e-5, and 1.1715e-5. Maximum absolute differences were 0.00040624, 0.00069427, 0.00082016, and 0.00033951 respectively. CPU chunking passed atol/rtol 1e-5; TPU chunking passed the original 0.005 thresholds. Full-model CPU-logit and cached-decoding comparisons, adapters/critic isolation, frozen hashes, real disk continuation, sampling-support regression and end-to-end recorder tests all passed.

The transient initialization failure did not recur after fresh idle checks; there was no reset/reboot or process termination. Its root cause remains unknown. Full pretrained resumed-turn memory validation is still required before declaring the longer experiment recovered.
