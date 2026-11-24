# Native TPU check 3 — automatic compilation incompatibility

- Attempt: `native-check3`, 2026-11-11 03:20 UTC, same authorized READY/HEALTHY v5e-4 in `us-west4-a`.
- Exact script: `native-check3.sh`; source archive SHA-256 `c39b38fb8208cf2a889290e664513b90609420d863cfa2e2c998e2ea39490e24`. Source is `source-eager-check1`, a separately preserved repair candidate based on published `b6f9078`.
- Change from check 2: native TPU uses eager attention and 512-token prefill chunks; test fixtures honor the production backend setting; new attention-shape CPU parity and cached generation test added.
- Packages, seeds, dtype, world rules, and sampling settings unchanged. All computation remains on authorized remote compute.
- Result: **5 failed**, exit 1, 58.74 seconds. No world turn completed. The CPU-versus-TPU forward-logit comparison passed before that test's generation stage failed.
- Confirmed boundary: Transformers' chunked-prefill path auto-compiled the model, then TorchTPU rejected a CPU scalar argument with `expected tensor to be on tpu, got cpu`. This is a different failure from the previous SDPA VMEM halt; no repeated VMEM halt was reported in this attempt.
- Evidence: `native-check3-console.txt` retains the traceback and final failure summary; the full remote log is `/tmp/island-community-qwen-v5e4-20261111-024900/native-check3.log`. Initial SCP retrieval failed with `lost connection`; retrieval will be retried without altering remote evidence.
- Repair under verification: pass Transformers' documented `disable_compile=True` during native TPU generation, retaining eager attention, static cache, and prompt chunking. Record this runtime choice. The installed `GenerationConfig` and `_valid_auto_compile_criteria` explicitly support the flag.
- Empty postflight lock timestamp: `2026-11-11 03:20:31.452946299 +0000`; apply the user's explicit cleanup permission only after a fresh all-device idle check.

The full log was successfully retrieved as `native-check3.log` on retry. After fresh READY/HEALTHY metadata and the authorized all-device idle cleanup, check 4 started from `source-eager-check2.tar.gz` (SHA-256 `34cc741b2afbaa675210ae53d9d4fd3791885a125f0934c19bb9f59cdab5719d`), with only `disable_compile=True` added to native generation and its regression test. Exact retry command/environment: `native-check4.sh`.
