# Early exploration and action-menu balancing

Date: November 9, 2026 (UTC). Seed 42; five founders; maximum 40 turns.
Status: validated and published; bounded demonstration pending.

Published source: ba3c94422123f39bb8a2f8f57fe638b798705173 in the private Island Community repository.
Source archive SHA-256: 206b843efa2a161936360481f6225d42320eaa412498ba3071215a0e25de0cef.
Full configuration: selected-config.json; the recording will include its own exact copy and source hashes.

Changes from the previous version 2 run (95081176bce175d190742d5cdecc5cfe3f20ea32):
- Schema version 3 adds action-kind count correction after score temperature scaling.
- Initial temperature 2.0, final 1.0; initial entropy coefficient 0.05, final 0.01.
- Linear annealing over each island's own first 30 completed optimizer updates.
- Entropy regularization is relative to the balanced prior, avoiding a bonus merely for extra trade variants.
- Per-kind candidate counts and exploration metrics are recorded; old configurations/checkpoints preserve original settings.
- No changes to world rules, starting stock, recurrent architecture, reward, or learning rate.

Base LR 0.0003 with the existing unhappiness multiplier; Adam beta1 0.9, beta2 0.999, epsilon 1e-8;
discount 0.99, value coefficient 0.5, gradient clip 1.0, hidden size 64, initial capacity 20.
The complete remaining economic and learning configuration is preserved in selected-config.json.
These exploration settings are initial choices, not empirically calibrated optima.

## Authorized remote execution

Project he-vision-group; zone us-west4-a; worker 0;
VM kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a.
User authorization covers CPU tests and a short recorded simulation, at most four threads.
Occupancy check: AVAILABLE worker=t1v-n-4b587646-w-0. No accelerator execution or provisioning.

Remote tests were staged under /tmp/island-community-exploration-20261109-052623-tests.
Verified test archive SHA-256: f902b8d496d09a5bdec5032812b5e738a1ed205e12e679184ffb02ab4e192fef.
Compared with published source, only README verification prose and test docstrings were added after testing.
Exact test command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=tests.xml

Result: 150 passed in 29.58 seconds. Includes legacy migration, all three configuration versions,
checkpoint continuation, on-policy gradients, count invariance, masks, and independent annealing.
Local non-neural checks: 100 passed; two viewer socket checks initially blocked by local permissions,
then both passed with localhost access. No local model or simulation experiment was executed.

## Demonstration command

Working directory: /tmp/island-community-exploration-20261109-052623/source.

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-exploration-20261109-052623/recording

A separate first-turn inference diagnostic evaluates the same initial network scores under flat temperature 1,
flat temperature 2, and balanced temperature 2. It makes no updates and does not change the demonstration's state.
Its full source and invocation are preserved alongside this report. This diagnostic isolates the immediate
probability transformation; comparing recorded trajectories is not a learning-quality calibration.

## Completed demonstration and diagnostics

Status: completed without runtime failure; stopped naturally at extinction on turn 23.
Five founders, zero births, five deaths; 24 contiguous replay frames; 92 individual decisions/updates.
Per-island updates: 21, 23, 15, 19, 14. All logged learning metrics are finite.
Exact source hashes match both the archived recording source and the published working files.
Recorded config matches selected-config.json, and the final frame equals snapshot.json.

Actions: 11 trade proposals, 16 public requests, 10 private requests, 11 waits,
6 service proposals, 8 teaching proposals, 7 cancellations, 4 rejections,
7 manufacturing commands, 9 item uses, and 3 acceptances.
Nine work orders completed: four cooking, four weaving, one compounding medicine.
Seven began directly and two through accepted service contracts. First completion: turn 2.
Accepted and completed contracts: two services and one bilateral trade.
Item uses: salt twice, fiber twice, stone/clay/iron/fuel/cloth once each.
No tool-building recipe completed in this trajectory.

Previous version 2 seed-42 demo: 78 trade proposals among 94 actions, zero manufacturing,
zero item uses, zero accepted contracts. Both trajectories ended at turn 23.
This shows broader observed behavior in one short example, not stable cooperation or improved survival.
Inventory-based rewards still give no immediate premium for manufacturing or health improvements;
short lives also limit each learner's experience. No reward changes or LR tuning were made.

Initial-state diagnostic: founders had 424–528 trade proposal variants versus 1–2 manufacturing options.
On the identical initial scores, flat temperature 1 gave trade proposals 83.19–87.09 percent
probability and manufacturing 0.16–0.41 percent. Flat temperature 2 still gave trade proposals
83.50–87.11 percent. Balanced temperature 2 gave trade proposals 13.68–14.03 percent and
manufacturing 13.53–15.23 percent. Seven kinds were available initially; within-kind alternatives
remain learned choices. Full per-island measurements: initial-policy-diagnostic.json.

The diagnostic command, from the demonstration source directory, was:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. /home/kristinelu/elf-runs/runtime/venv/bin/python ../inspect_initial_policy.py

Mean recorded temperature ranged 2.0 down to 1.26666665; entropy coefficient 0.05 down to 0.02066667.
No agent reached 30 updates, so none reached the schedule endpoint. Raw entropy is lower than the
old flat-menu entropy because probability is no longer concentrated among hundreds of trade variants;
raw entropy across these different priors is not a measure of economic behavioral breadth.

Artifacts: recording/ contains manifest, full config, source fingerprints, checkpoint, world frames,
events, experiences, and metrics. analysis.json summarizes verified results. tests.log and tests.xml
preserve the 150-test validation. Run output is in run.log.
Results archive SHA-256 verified remotely and locally:
b721aecf9054d33703579d8ef50016e7b9b1a84cd776e77761465f86a292c482.

Published commit: https://github.com/kristinelu921/island-community/commit/ba3c94422123f39bb8a2f8f57fe638b798705173
GitHub confirmed isPrivate=true. Run artifacts were not published.
New replay server: http://127.0.0.1:36899/ (left running). Previous recordings remain unchanged.

Browser verification passed for all 24 frames: playback, seeking, pause stability, selected-island details, item guide and active-stat display, delayed-response handling, and mobile viewport. Screenshot: replay.png.
