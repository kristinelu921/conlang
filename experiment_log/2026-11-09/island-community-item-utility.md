# Island Community item utility validation

This is a software validation record, not a new economic experiment or a claim
of learned cooperation. The previous 24-frame demonstration is unchanged.

## Changes from the replay baseline

- Rules version 2 introduces one-unit item-use actions costing 0.1 work units.
  Every nonfood item has a health, recovery, comfort, disease-resistance,
  production-speed, or supply-loss-protection effect. The complete configuration
  is saved in `island-community-item-utility-validation/config.json`.
- Temporary effects expire, cannot stack with another unit of the same item,
  and are capped across item types. Items promised in offers remain unavailable.
- Meals carry unused food into later turns. Drinking water is still required
  each turn. Improvements and stored nutrition are not inherited by newborns.
- Private observations include active stats, remaining durations, and nutrition;
  available item-use actions enter the neural candidate list. Resource reward
  accounting, independent policies, and disabled death replay are unchanged.
- Version 1 configuration/checkpoint behavior and neural dimensions are retained.
- Viewer changes add the complete item guide, active effects, expiry turns, and
  saved nutrition. Reconstructed recordings now retain their exact config file.
  The existing replay received its original config only after its SHA-256 matched
  the already-recorded provenance; its frames and outcomes were not regenerated.

## Execution and authorization

Remote machine: `kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`,
project `he-vision-group`, zone `us-west4-a`, worker 0. The user's prior explicit
authorization covers CPU tests on this exact machine. Fresh occupancy checks
returned `AVAILABLE worker=t1v-n-4b587646-w-0` before testing. No TPU accelerator,
foreign machine, or new provisioned resource was used. No processes were stopped.

Python 3.10.18 and PyTorch 2.2.2+cpu; at most four CPU threads. Default world seed
42; each test's deliberate fixture overrides are recorded in the staged source.
The final source archive is `/tmp/island-community-items-final-20261109.tar.gz`,
SHA-256 `9d2c387fb1eab511a597ec2d93dacf9003b629bd49a13a482b4fd361fc676266`.
The archive was copied into a new isolated remote folder and verified remotely.

Exact final test command, working directory
`/tmp/island-community-items-final-20261109`:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
ISLAND_ALLOW_NEURAL_TESTS=1 \
PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies \
/home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest \
  island_community/tests -q --junitxml=final-tests.xml
```

## Validation

- Initial remote suite: 123 passed in 25.80 seconds, including neural item use,
  vmap independence, and exact continuation with births under rule versions 1/2.
- Local rule/storage/viewer suite: 93 passed, 28 neural checks deselected at that
  stage. An initial local attempt could not open HTTP test sockets; rerunning
  with localhost permission passed. No local model execution occurred.
- Final replay/config-preservation and HTTP endpoint checks: 13 passed.
- Chromium: 24-frame replay controls, 14-item guide, explicit legacy-rule label,
  private-stat UI fixture, delayed seek handling, and 390-pixel viewport passed.
  Browser fixtures only inspect data; they do not run agents or simulations.
- A final isolation regression ensures changes to the caller's config cannot
  mutate a running world's item rules. Final remote results are appended below.

Test logs, JUnit results, config, source manifest, and browser screenshot are in
`island-community-item-utility-validation/`. The viewer remains available at
http://127.0.0.1:8769. The item guide labels the earlier recording explicitly;
new mechanics require a new version 2 simulation, not a replay of the old run.

No new population-stability or economic-learning experiment was performed.
The bonus values are illustrative and remain uncalibrated.

## Final result

All **124 tests passed in 25.72 seconds** on the authorized remote CPU.
Final artifacts: `final-tests.log`, `final-tests.xml`, and `final-manifest.json`.
The final browser check passed after the original replay configuration was restored.
Published commit: `95081176bce175d190742d5cdecc5cfe3f20ea32` in the existing private Island Community repository.
Only the documented source and test files were committed; run artifacts remain local.
