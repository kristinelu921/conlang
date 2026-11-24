# Learnable starting preference for grain and water trades

Date: November 9, 2026 UTC. Status: validation in progress.
User request: initialize with more importance on grain and water trading.
Baseline: private GitHub commit 8ced604a8575dddc06d6715768a2b20ceb8de8f2, previous recording
island-community-water-20261109-060224. That recording contains one accepted grain trade
(4 grain from Island 2 to Island 1, accepted turn 6, arrived turn 8) and no water shipments.
Several requests for water targeted non-water producers. A grain/water offer appeared at turn 34
and was countered at turn 38 but not accepted before this 40-turn segment ended.

Complete selected configuration: selected-config.json. Seed 42, five founders, independent 64-unit
recurrent policies, vmap enabled, CPU, maximum four threads, 40-turn fresh demonstration.

Changed relative to baseline: configuration schema 6 introduces trade_initialization with
initial_odds_multiplier=2 and reserve_turns=1. One binary action feature marks qualifying staple
trade proposals, counteroffers, and acceptances. Proposals/counters must request items from their
public producer specialization. The terms must include grain or water (the configured survival bundle
items). Outgoing food/water must leave one turn of the spent necessity in unreserved usable supplies;
food includes stored meal nutrition. Other islands' private inventory is not read by this feature.
Acceptance interprets the outgoing side from the accepting agent's perspective. Reject/cancel,
messages, manufacture, and unrelated trades do not receive the feature. No candidate is removed.

Fresh founder action_bias.weight for this feature receives an additive T_initial * log(2) offset.
Thus the initial odds of a marked vs unmarked action are twice those of an otherwise identical
schema-6 policy without that offset. No permanent logit multiplier or extra reward is added.
Learning updates the weight normally; children inherit it with the existing mutation, without
reapplying the offset. Restoration preserves weights and initialization settings. Versions 1-5
retain old feature dimensions and behavior. Schema-6 defaults have 112 observation features,
109 action features, and 43,566 parameters per 64-unit recurrent agent.

Unchanged: health 300; food/water consumption 1 each; physical shortage damage 7 each;
empty-food and empty-water learning penalties 5 each; health-loss penalty 0.5 per net lost point;
inventory reward scale 0.01; LR 0.0003 with existing unhappiness multiplier; Adam 0.9/0.999/1e-8;
discount 0.99; value coefficient 0.5; gradient clip 1; initial capacity 20; temperature 2 to 1,
entropy coefficient 0.05 to 0.01 over 30 own updates; action-kind balancing. No calibration sweep.
The schema-6 feature changes feature width and seeded weight initialization, so comparisons to
schema 5 are descriptive; the isolated odds test holds architecture and all other weights fixed.

Remote CPU authorization persists for kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
project he-vision-group, zone us-west4-a, worker 0. Fresh occupancy: AVAILABLE worker=t1v-n-4b587646-w-0.
No local neural execution or accelerator workload. Local rule/storage/viewer checks: 157 passed,
59 neural cases deselected, 26.93 seconds. The final integration assertion was added afterward;
it is part of the remote suite below.
Remote test directory: /tmp/island-community-trade-20261109-061718-tests.
Test source archive SHA-256: e9068fd338d2fa2f305472093a97305b222ebb327cecc01a27ede8ae6d46851c.

Exact remote validation command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=tests.xml

Exact demonstration command from /tmp/island-community-trade-20261109-061718/source:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-trade-20261109-061718/recording

Final validation, commit, checksums, and metrics will be appended.

Remote full validation: 216 passed in 45.12 seconds. Tests include actual 2x initial odds in both execution modes, subsequent weight learning, inheritance without a repeated offset, legacy restore, and exact continuation under all six configurations. Only README validation prose changed after testing.

Published private GitHub commit: 918ba38f129a37d4f886ede864de19e83ac58d3d.
Source archive SHA-256: adb7326e470c83f0a068669467cc41e3b2646299a43c87371212d480608dd9a6.
Push to main succeeded. No PR was created for this standalone application update.
Fresh pre-demonstration occupancy: AVAILABLE worker=t1v-n-4b587646-w-0. Source staged in a new
isolated directory and verified by checksum before the short CPU workload.

## Completed demonstration and trade accounting

The fresh run completed 40 turns with 41 contiguous frames, two living founders, three deaths, no births,
and 172 individual updates (40, 40, 31, 32, 29). Six chosen actions had the new feature active: three
proposals, one counteroffer, two acceptances. Surviving health: Island 0 = 132.95594768191015,
Island 1 = 77.97314195030843; mean 105.46454481610928. Complete accounting: analysis.json.

There were 23 issued trade proposals including counters; four were accepted and all four completed.
Two completed contracts involved grain, and one of those also involved water. Grain deliveries:
4 from Island 1 to Island 0 at turn 38, then another 4 at turn 39. The second deal also delivered
4 water from Island 0 to Island 1 at turn 39. Thus necessity shipments delivered 8 grain and 4 water.
This is two distinct necessity-trade contracts, not three. The previous recording had 27 issued
trade proposals including counters, three accepted trades, two completed, and 4 grain/0 water
units delivered in one necessity-trade contract (grain delivered to Island 1 from Island 2).

These are descriptive single-seed results; the schema-6 feature changes the seeded network initialization,
so they are not a controlled estimate of the prior's benefit. Necessity deliveries still occurred late.
No claim is made that this mild initialization has solved survival or established a reliable trading policy.

Verification: source SHA-256 fingerprints match the published source and saved recording; config matches
selected-config.json; all frames are contiguous and the final frame equals snapshot.json; each selected
action has 109 features; actual model size is 43,566 parameters per agent. Rewards equal inventory value
plus all three survival penalties; lifetime totals and neural mean rewards match recorded transitions;
all learning metrics are finite. Only saved artifacts were inspected on the development machine.

Results archive SHA-256 matched remotely and locally:
a75a72b6235cd1f37e3d6e20483194f7129b1d7538373a5a7979b0681d47d5a6.
Artifacts: recording/ (configuration, source, manifest, checkpoint, experiences, frames, metrics, events),
run.log, tests.log, tests.xml, results.tar.gz, selected-config.json, and analysis.json.

Final status: complete. Replay browser checks passed for all 41 frames, including playback, seeking, inspection, mobile layout, and reward breakdown. Screenshot: replay.png. Viewer http://127.0.0.1:33717/ is left running, and opening it is queued in the app.
