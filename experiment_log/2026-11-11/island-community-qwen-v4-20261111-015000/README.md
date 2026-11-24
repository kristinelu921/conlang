# Qwen native TPU recovery: 2026-11-11 01:50 UTC

Selected only from the completed filtered check_tpus.sh AVAILABLE table: kmh-tpuvm-v4-8-spot-kristine-09-14-38-39-N-us-central2-b, project he-vision-group, zone us-central2-b. Fresh describe confirmed READY, HEALTHY, Spot, v4-8, topology 2x2x1, one worker. Fresh occupancy helper confirmed AVAILABLE; 41 GiB free local disk. No active job touched or resources created.

Previous authorized non-kristine v6e disappeared before the approved stale-lock cleanup; previous evidence remains under the 2026-11-10 comparison log. No trained checkpoint exists. This is a fresh compatibility attempt, not a resumed completed world.

Task-local environment/source/output root: /tmp/island-community-qwen-v4-20261111-015000. Exact setup command is bash -s < setup.sh over non-interactive SSH to the current verified endpoint. Versions pinned in setup.sh. Four CPU threads retained for tests. No neural work runs on the development machine.

Native TorchTPU actively lists v5e/v5p/v6e/v7x support; v4 is not actively validated (https://torchtpu.readthedocs.io/en/latest/faq/). A minimal remote compatibility test must pass before committing to pretrained inference. Static cache, CPU-side generation tokens, and public single-device launch helper replace the failed manual visibility mask. No change to model weights, precision, or scientific scenario. Status: dependency setup pending.

## CPU compatibility checks and frozen baseline

Native v4 compatibility failed before model creation; see compatibility-failure.md. All four Qwen neural regression tests then passed on the same authorized VM CPU with exactly four threads and TORCH_DEVICE_BACKEND_AUTOLOAD=0, 22.67 seconds. These cover frozen base checksums, independent adapter updates in both modes, inheritance, checkpoint determinism, actual requested/submitted/executed actions, and private history across two runner turns. Real Qwen architecture with small random weights; not pretrained performance. Local 22 action-contract tests passed in 1.33 seconds.

Selected CPU fallback baseline: published commit d119758b12859ec990c8dfbe473c6963d3f66ed8, config island_community/scenarios/qwen_10_frozen.json, seed42, 10 founders, 20 turns, four CPU threads, float32, pinned Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca. Complete effective config is in the committed source and resulting recording. Model downloaded directly from HF on the VM. No dataset or regional checkpoint transfer. Newer runtime is the only baseline runtime change (Transformers5.22.2/Torch2.15.0 instead of Transformers4.51.3/Torch2.6.0; exact versions in dependencies.txt); scenario unchanged. No updates in frozen mode. Exact staging/download/launch command: frozen-cpu.sh. Status: preparing pretrained run.

Pretrained model download completed (7 files, pinned revision). Frozen CPU process PID1052507 (time wrapper1052506), task source d119758, started approximately 02:03 UTC on the one-worker v4 VM CPU. Source archive SHA-256 e1defe2d6595ffdb835c3acd51a29ea1bee610e883ded747296a24a9a8deb7ed. Backend autoload is explicitly disabled so this CPU fallback does not initialize the unsupported TPU. Weights loaded successfully. Recording starts at turn0; first decisions pending.

## First pretrained decisions and continuous monitoring

At 02:05:41 UTC, process1052507 was live (177 seconds elapsed, 344% CPU, RSS4944572KiB). Log confirmed real pretrained decisions for turn1: island0 wait, island1 manufacture. No world turn yet completed; do not treat logged proposals as resolved world actions until all islands act.

A dedicated local read-only monitor (monitor.py, PID621088, monitor.pid) checks the exact resource/API identity, cloud health, this run's process and newest checkpoint every30 seconds. It mirrors recording artifacts with rsync checksum verification, logs monitor.jsonl and monitor-latest.json, and sends milestone/failure notifications. It never launches workloads, claims resources, or touches other runs. Authorized CPU fallback continues while a fresh wider TPU sweep awaits the required current-message phrase.

## Turn2 real-model action and history audit

All10 second-turn choices map exactly to their requested/submitted/world action records. Each of the10 private prompts contains exactly that island's prior action, execution flag, and reward from turn1. No updates occur in frozen mode. Detailed counts and execution outcomes are in turn2-audit.json. The full20-turn baseline remains running; final frozen-weight hash is checked only at completion.

At02:41 UTC the newly authorized replacement kmh-tpuvm-v6e-16-kaiminghe-9fe46b was NOT_FOUND before any guest connection. The CPU run was not interrupted or migrated. A fresh own-pool audit returned no AVAILABLE resource. A smaller previously idle v5e-4 backup is awaiting its separate exact acquisition approval.
