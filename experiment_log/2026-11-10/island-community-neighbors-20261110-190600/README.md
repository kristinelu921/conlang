# Neighborhoods, private history, and flexible trade amounts

Status: completed all 100 requested turns successfully, with exit status 0.
All 287 remote tests passed in 118.28 seconds.

Selected scenario: `island_community/scenarios/neighbors_50.json`, schema 8,
seed 42, 50 founders, 64-unit independent recurrent policies, vmap enabled.
Relative to the preceding schema-7 50-island run, this adds at most five mutual
neighbors, separate grain/water producer access, private eight-turn action/outcome
history, and two-stage selection of all feasible integer trade amounts. Rewards,
gift weights/rates, health, learning rate, and temperature schedule are unchanged.
The complete effective configuration will be retained with the recording.

Authorized compute: CPU only, at most four threads, worker 0 of
`kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`,
project `he-vision-group`, zone `us-west4-a`. Occupancy returned AVAILABLE;
see `preflight.txt`. No TPU device is used by the application.

Validation source archive:
`/tmp/island-community-neighbors-20261110-190600-tests.tar.gz`, SHA-256
`f7795631a65202f0f300dd5ce97ff784e6e079d376477eb3137dcd63e6f8b960`.
The isolated remote extraction has the same name without `.tar.gz`.

Remote validation command, from that extraction:

```bash
PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=/tmp/island-community-neighbors-20261110-190600-tests.xml
```

Local verification executes rule/browser fixtures only; neural tests require the
explicit remote environment variable. The initial quantity fixture needed its
target water reset after the next turn's automatic gift; this was a fixture
correction, not a model or world rule change.

Published source: [9a9cf1b55d9b8061bc775c2e96fcf1e69a00f5b1](https://github.com/kristinelu921/island-community/commit/9a9cf1b55d9b8061bc775c2e96fcf1e69a00f5b1).
The private application repository uses direct commits; no ELF run or PR is involved.
`source.tar.gz` is the exact Git archive, SHA-256
`4532e9d4ea3606206d3d6c993da4936bf35deaf12b9f8dcd4acdde3af00ae0f5`.
The checksum was verified before extraction under
`/tmp/island-community-neighbors-20261110-190600/source` on the remote VM.
The fresh occupancy result is retained in `run-preflight.txt`.

Exact simulation command, from the extracted source:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/time -v -o ../timing.txt /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/scenarios/neighbors_50.json --turns 100 --device cpu --checkpoint-every 10 --output /tmp/island-community-neighbors-20261110-190600/recording
```

Output is redirected to `../run.log`, with the final shell status saved in
`../exit-status.txt`. Both files and `timing.txt` have been retrieved.

## Recorded results

- 101 replay frames, turns 0–100; 50 founders, eight births, 57 deaths, one survivor.
- 2,645 individual learning updates. 489 trade proposals; 28 accepted and all 28
  completed, an acceptance fraction of 5.73%. This is not evidence of convergence.
- Proposed trade amounts covered every integer from 1 through 31. A completed
  example was proposal 45: island 5 offered island 44 eight salt for 16 grain.
- Delivered cargo included 127 grain, 222 water, and 21 meals, plus other goods;
  full totals and event counts are in `summary.json`.
- Every recorded frame passed mutual-neighbor, no-self-contact, maximum-five,
  and maximum-eight-history-entry checks. Every founder had exactly five contacts
  and both required producer types. Missing producer access first appeared at
  turn 85 and is explicitly reported in the replay; all 16 remaining frames show it.
- Wall time: 3 minutes 11.75 seconds. Peak process RSS: 740,868 KiB (723.50 MiB).
  Initially, 50 reserved policy slots used 61,173,600 bytes for weights, Adam,
  clocks, and hidden state. Each policy has 101,934 parameters. Slot capacity
  grows automatically when the population grows; temporary tensors and world
  records are additional memory.

The learning-rule and scenario changes were exercised together in this seeded
demonstration. It is not a controlled comparison attributing survival or trade
changes to any one feature. Training still uses one-step returns and truncates
gradients each turn; the explicit history window is not sequence backpropagation.

## Artifacts and verification

`artifacts.tar.gz` has SHA-256
`743573cd436ba809267200a6ef890b7001c4c5a1f25473711af6bc52f98e1b31`,
verified against the remote archive before extraction. `SHA256SUMS` also records
the checkpoint, configuration, journals, frames, metrics, and manifest. Every
recorded source hash and the effective configuration match the published source.
The complete checkpoint is `recording/checkpoint.pt`.

Local rule verification passed 212 tests; two HTTP fixture tests initially failed
because that invocation could not open sockets, with 64 neural tests deselected.
The subsequent permitted replay/HTTP check passed all 23 tests, including the
new schema-8 reconstruction cases added after the initial local collection.
The complete remote suite passed all 287 tests with no skips or failures.

Replay server: `http://127.0.0.1:35229/`, serving only saved data. Browser validation
initially needed the existing browser-library directory added to `LD_LIBRARY_PATH`;
the failure log is retained as `browser-check-missing-library.log`.
The corrected browser check passed for all 101 frames, including seeking,
play/pause, the 50-island map, neighbor and history inspection, request races,
deceased selection, reward details, and a narrow viewport. See `browser-check.log`.

```bash
LD_LIBRARY_PATH=/tmp/island-replay-browser-check/system-libraries/extracted/usr/lib/x86_64-linux-gnu NODE_PATH=/tmp/island-replay-browser-check/node_modules PLAYWRIGHT_BROWSERS_PATH=/tmp/island-replay-browser-check/browsers node island_community/tests/check_replay_browser.cjs http://127.0.0.1:35229
```

`summarize_recording.py` regenerates the summary from saved JSON only. The replay
server remains running; the remote simulation and test processes completed.
