# Native TPU check 1 — passed

- Resource: `kmh-tpuvm-v5e-4-spot-myang-ar`, `us-west4-a`, v5litepod-4, 2x2, READY/HEALTHY, Spot; one physical worker, one selected TPU device.
- Start: 2026-11-11 03:00 UTC. Exact command and environment: `native-check1.sh`.
- Source archive SHA-256: `6943336421597985c3d3a9ac3d5da53ac26e0a81a21436dd83581aa45234d009`.
- Python 3.12.16, Torch 2.15.0+cpu, TorchTPU 0.2.0, libtpu 0.0.52, Transformers 5.22.2, NumPy 2.5.4.
- Result: **1 passed, 3 deselected**, 16.23 seconds; exit sentinel `0`. Log preserved as `native-check1.log`.
- Tests actual tiny Qwen architecture with the pinned public tokenizer, float32, seed 42, initialization seed 128, three reasoning tokens and a 1,024-token context. Confirms canonical constrained JSON choices, no trainable frozen parameters or optimizer, unchanged base hash, restored private histories, and identical sampled continuation after checkpoint restoration.
- This is a correctness test with random tiny weights, not the pretrained 0.6B experiment or a throughput estimate for it. Adapter gradients and the full recorder/resume path remain to be tested on TPU.

## Postflight conditions

The successful test emitted 4,336 driver log permission errors because the shared `/tmp/tpu_logs` directory was not writable. No shared permissions were changed. Subsequent checks will set `TPU_LOG_DIR` and `GOOGLE_LOG_DIR` to a task-local directory.

At 03:01:26 UTC the exact-resource preflight found no device or lock holder, but an empty `/tmp/libtpu_lockfile` remained. It belongs to `kristinelu` (UID 2010), with modification time `2026-11-11 03:00:36.970006151 +0000`, during this test's runtime initialization. Test PID 1218440 exited. The file was not removed: explicit cleanup permission was requested because the resource name lacks `kristine`. This is a postflight cleanup blocker, not a failed model test. `full-tests-preflight.json` preserves the evidence.
