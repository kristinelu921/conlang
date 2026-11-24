# Authorized wide read-only TPU sweep — 2026-11-11

Authorization: current user message contains AUTHORIZE WIDE READ-ONLY TPU SWEEP. Acquisition is not authorized by this phrase. A fresh completed kristine-only check_tpus.sh audit ran first and found no idle native-TorchTPU-compatible resource.

Cloud Asset Search returned107 indexed resources across four zones. Current direct TPU list calls covered those zones plus the previously known asia-northeast1-b zone and returned114 resources. No cloud query failed. Exact command: python /tmp/island-community-wide-tpu-sweep-20261111-022500.py. Source, probe, per-zone cloud metadata, and all per-resource results are retained here.

Sweep interval: 02:25:56–02:26:59 UTC. Results:70 idle,17 occupied,4 UNKNOWN due to unheld lockfiles,23 excluded (13 deleting,8 creating,2 without confirmed healthy state). All307 workers belonging to the70 idle resources passed the occupancy probe. Any visit stopped at its first active or unknown worker. No remote writes, signals, claims, workloads, sessions left connected, or resource changes.

Idle inventory:11 v6e-16 and2 v6e-32 in us-east5-b;47 v5e-16,7 v5e-32,1 v5e-4 in us-west4-a;2 v4-8 in us-central2-b. None of the idle resources have kristine in the exact name. Candidate: kmh-tpuvm-v6e-16-kaiminghe-9fe46b, v6e-16/4x4, us-east5-b, READY/HEALTHY/Spot, all4 workers idle. Separate exact acquisition authorization requested. Native runtime and throughput still require validation.

The wider probe also inspected /dev/accel[0-9]*, exposing a real gap in the older VFIO/lock-only discovery and claim checks. The occupied v4 kmh-tpuvm-v4-8-spot-kristine-09-14-38-39-N-us-central2-b had accel-device holder PID899538, owner kristinelu, without a libtpu lock holder. That workload was not touched. The four-thread Qwen CPU process1052507 is separate and continues without initializing TPU devices.

Notification: automatic approval review rejected sending the resource identifier and status to ntfy as metadata egress. A generic approval-required notification containing only the repository name and a request to review Codex was accepted and delivered instead.

Local safety-check repair: check_tpus.sh now inspects accel devices as well as VFIO and lockfile holders. check_tpu_occupancy.sh protects accel holders in both inspection and claim phases; automatic foreign-process termination remains limited to VFIO authorization. Regression test first failed for all three occupied-device probes, then all26 script tests and13 subtests passed after the fix (6.47 seconds). Tests use only fake local commands/devices; no accelerator execution or process termination occurs. Shell syntax and diff whitespace checks passed.

The report is served at http://127.0.0.1:42511/ (PID766222). The actual frozen-Qwen replay is served at http://127.0.0.1:39627/ (PID766324), from the continuously mirrored CPU recording. Servers are local and remain running.

## Candidate disappeared before acquisition, 02:41 UTC

The user supplied the exact acquisition phrase for kmh-tpuvm-v6e-16-kaiminghe-9fe46b. Fresh gcloud describe exited1 with NOT_FOUND before any guest connection, copy, install, claim, or workload. No old IP was contacted. The cause of disappearance is not established. Evidence: candidate-loss-024143.json. The CPU Qwen baseline remains active with its turn2 checkpoint mirrored; no continuation was started elsewhere.
