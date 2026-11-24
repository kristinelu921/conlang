# More health and explicit survival reward penalties

Date: November 9, 2026, America/New_York. Experiment identity uses UTC timestamp 054927.
Status: implementation complete, remote validation in progress; fresh bounded demonstration pending.

User request: increase starting health and penalize food reaching zero and health decreasing.
Complete configuration: selected-config.json. Seed 42, five founders, independent 64-unit recurrent
policies, vmap enabled, CPU execution, maximum 40 turns. Fresh initial weights, no replay or shared learning.

Changed relative to the prior exploration run at ba3c94422123f39bb8a2f8f57fe638b798705173:
- Configuration schema 4; starting/maximum health increases from 100 to 300 for founders and newborns.
- Reward deduction 5 on each turn ending with no usable food, including repeated empty turns and
  the turn the last ration was consumed. Grain, meals, and saved nutrition count; reserved and
  in-transit stock do not. A surviving parent's food is assessed after inheritance transfers.
- Reward deduction 0.5 per actual net health point lost between the decision and completion of upkeep.
  All causes count; healing offsets losses, and damage cannot be charged past zero health.
- Both explicit terms can apply. They change reward, not physical damage or the food/water requirement.
- Terminal inventory reward remains zero, but terminal penalties survive. Lifetime scores sum learning
  rewards. Unhappiness and resource charts stay based on inventory value to prevent feedback into damage.
- Reward components are recorded in experiences, events, and replay decisions. Old versions/checkpoints
  retain their original rules; no old recordings were modified.

Unchanged: base LR 0.0003 with existing unhappiness multiplier, Adam 0.9/0.999/1e-8,
discount 0.99, value coefficient 0.5, gradient norm clipping 1, independent capacity 20.
Exploration temperature 2 to 1, entropy coefficient 0.05 to 0.01 across 30 own optimizer updates;
action-kind balancing enabled. Inventory reward scale 0.01, currency value zero, original item values.
Physical shortage damage 7 per missing necessity, 1 food and 1 water required each turn.
These are configurable initial settings, not empirically calibrated coefficients.

## Authorization and verification

Previously authorized CPU VM: kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
project he-vision-group, zone us-west4-a, worker 0. Fresh occupancy: AVAILABLE worker=t1v-n-4b587646-w-0.
At most four CPU threads. No local neural execution, accelerator workload, or provisioning.

Local checks: 129 non-neural tests passed, 50 neural checks deselected, in 23.99 seconds.
Remote test source: /tmp/island-community-survival-20261109-054927-tests.
Test archive SHA-256: 459bd4da097f7f4b439cff157fc620854b14a8940eb5fd3ce523837afe875704.

Exact remote validation command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=tests.xml

Exact demonstration command, from /tmp/island-community-survival-20261109-054927/source:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-survival-20261109-054927/recording

Published source, remote test outcome, final metrics, and artifact checksums will be appended after completion.

Initial remote validation: 178 passed, one test fixture failed configuration validation because it requested a one-turn endowment, below the existing five-turn minimum. No application runtime failure occurred. The fixture now retains the valid default endowment and observes twelve bounded turns to capture negative terminal rewards. The failed validation log is preserved; only the corrected test needs rerunning because application code did not change.

## Validated and published source

All 179 tests passed across the initial 178 passing cases and the corrected one-test rerun (2.96 s).
Correction: test_runner.py SHA-256 ef800c4158cb42f7e3dde32203573f33f0a2a22ad0ff8b5df5eb46f1733dffb3.
The rerun used a fresh isolated source directory, /tmp/island-community-survival-20261109-054927-tests-corrected,
and the same environment as the first validation. Exact command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests/test_runner.py::test_survival_penalties_reach_neural_learning_and_terminal_records -q --junitxml=tests.xml

Published source commit: 6e9d39f93e995a43bcb01b6969d830375ac7d7bc.
Source archive SHA-256: fdd97a5c56ebd07ecbb516ba5c5d755fa0fd0af12dddd9b8771af0372daa7603.
Only README validation prose changed after the successful tests.

## Completed demonstration

Completed the requested 40-turn segment without runtime error; 41 contiguous replay frames.
Two founders remain alive, three died, zero births. Remaining health: Island 0 = 61.98717945028512,
Island 1 = 75.98675827260244. The prior 100-health run reached extinction at turn 23.
The larger health budget directly extends survival; this single run does not establish improved policy
quality or isolate the contribution of reward shaping. No extra training or parameter sweep was performed.

There were 169 individual updates: 40, 40, 29, 32, 28 by founder. 110 rewards were negative;
98 transitions incurred the empty-food penalty and 146 incurred some health-loss penalty.
All three terminal rewards retained negative penalties. Complete values: analysis.json.
Four cooking, nine weaving, and one medicine recipe completed; 18 item uses.
Accepted contracts: two trades, four services, three teaching agreements. There were 24 direct trade
proposal actions and two counteroffer actions; the denominator must include counteroffers when reporting
acceptance across every issued trade proposal.

Verification: all source fingerprints match the published source and archived recording, config matches
selected-config.json, frame turns are contiguous, and the final frame equals snapshot.json. Every recorded
reward equals its inventory component plus the two deductions. Each health-loss measurement matches the
actual adjacent-frame health change; terminal inventory components are zero. Lifetime scores equal the
sum of each learner's recorded rewards, and per-turn neural mean rewards match the recorded transitions.
All learning metrics are finite.

Results archive SHA-256 matched remotely and locally:
dc7c20f6c7ceb83206b9d74b89ef365ee1dd1223ce6aa464f5e0cdf85a919fe8.
Artifacts: recording/ (source, config, manifest, checkpoint, frames, events, experiences, metrics),
analysis.json, run.log, initial-tests.log/xml and corrected-test.log/xml. Run artifacts remain local/private;
only application source and tests were published to the existing private GitHub repository.

Browser checks passed for all 41 frames, including replay controls, mobile layout, saved nutrition, active improvements, 300-health scale, negative learning reward, and separate food/health deductions. Screenshot: replay.png. New viewer http://127.0.0.1:41685/ is left running; it was opened in the app.
