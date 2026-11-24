# Recorded per-turn model options and logits

Status: testing. The requested visualization will show every valid action and
conditional quantity option with raw logits, adjusted logits, probabilities,
and the sampled choice. Scores are captured during the existing forward passes;
no additional model evaluations or random draws are introduced.

Baseline: `island-community-neighbors-20261110-190600`, source
`9a9cf1b55d9b8061bc775c2e96fcf1e69a00f5b1`. The scenario remains
`island_community/scenarios/neighbors_50.json`, seed 42, 50 founders, 64 recurrent
units, 101,934 parameters per independent agent, eight explicit history turns,
five mutual neighbors, and all-feasible integer trade amounts. No reward,
architecture, optimizer, learning-rate, exploration, allocation, or world-rule
settings are changed. The complete effective configuration will be saved with
the new recording. The planned 100-turn repeat will be compared with the baseline
experience journal to verify that score capture preserved the trajectory.

Authorized compute is the CPU of worker 0 on
`kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`,
project `he-vision-group`, zone `us-west4-a`, limited to four CPU threads. See
`preflight.txt` for the fresh read-only occupancy check.

Test archive `/tmp/island-community-logits-20261110-193900-tests.tar.gz` has SHA-256
`5854a5b23a4ff101718c0d0e23e5421553fe3c11c75bb200cec67874e1745cc8`.
It will be verified before extraction in the isolated remote directory with
the same name minus `.tar.gz`.

Remote test command from that extraction:

```bash
PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=/tmp/island-community-logits-20261110-193900-tests.xml
```

Local checks cover recorded-artifact storage, replay reconstruction, and read-only
HTTP endpoints. Numerical inference and learning checks run only remotely.


The complete remote suite passed **290 tests in 145.09 seconds**. The subsequent
larger-preset test passed (one test, 30 deselected), validating 751,470 parameters,
independent recurrent states, learning, and exact checkpoint continuation.
Its test overlay has SHA-256 `e91ebaa8cce22d7328de1db43bdf807feed89f8c4d9036fb41080fd06144aaae`.
Command: the same remote test environment above, with
`-m pytest island_community/tests/test_policies.py -k gru256 -q`.
Local artifact/storage/replay tests passed all 31 checks.

Published source: `7be9e420ddd3208102db71e34cfdee4e4c1145fd`,
https://github.com/kristinelu921/island-community/commit/7be9e420ddd3208102db71e34cfdee4e4c1145fd .
`source.tar.gz` SHA-256:
`c74c84b40724bc97ffbe0a6e0233ef9f0b85f2abf452a43846a967b120ed526e`.

Exact baseline-repeat command from the isolated remote source directory:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /usr/bin/time -v -o ../timing.txt /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/scenarios/neighbors_50.json --turns 100 --device cpu --checkpoint-every 10 --output /tmp/island-community-logits-20261110-193900/recording
```

The new 256-unit preset is published in the same source, but is run separately
under its own identity. Its only configuration change is `policy.hidden_size`.


## Completed baseline repeat

The simulation completed all 100 turns with exit status 0. Its entire experience
journal is byte-for-byte identical to the previous 64-unit run, SHA-256
`6df49d1f176509e14843a98793f4953bb733526bf34dc086d18b7f72363df2c9`.
Thus score capture preserved all 2,645 decisions, observations, rewards, and
terminal outcomes. The previous 28 completed trades and one final survivor are
unchanged. Wall time: 4:00.35; peak RSS: 766,980 KiB (749.00 MiB).

Remote artifact archive SHA-256:
`ab08f0ff793ce5af3067fe7dd42512fa04f29d478a764d94d7f06862eeb3ac88`.
The archive contains the full checkpoint, score files, journals, source hashes,
configuration, timing, and checksum manifest.

The updated viewer passed the browser interaction suite against the old recording
(all 101 frames), including the explicit historical-logits-unavailable path.
See `browser-legacy.log`. The larger run is logged separately under
`../island-community-gru256-20261110-201500`.


Archive checksum matched after retrieval. The development host uses Python 3.10,
so an initial `hashlib.file_digest` inspection failed before producing a digest;
verification was completed with `sha256sum` instead.
All 2,645 decisions have finite scores, normalized probabilities, and valid
selected indices. Main menus range from 15 to 436 candidates; conditional
quantity menus reach 496 options. The recorded economic and graph summaries
are retained in `summary.json`.

Replay: http://127.0.0.1:42845/ . Browser checks passed against all 101 frames,
including both score stages, filtering, chosen rows, downloads, historical
seeking, pause stability, deceased selection, and narrow layout. See
`browser-check.log` and `replay.png`. This is a read-only local service.


An additional browser inspection verified island 7 at turn 1: all 220 conditional
quantity options render, with the sampled five-grain-for-twelve-herbs offer marked.
See `quantity-panel.png`. Its first inspection attempt timed out waiting for a
collapsed HTML option to be visible; changing the inspection wait to `attached`
resolved it. No application change was needed. All retrieved artifact checksums
passed (`artifact-verification.log`).
