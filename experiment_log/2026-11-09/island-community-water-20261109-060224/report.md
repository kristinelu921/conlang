# Separate empty-food and empty-water penalties

Date: November 9, 2026 UTC. Status: validation in progress.
User request: penalize both empty food and empty water.
Baseline source: 6e9d39f93e995a43bcb01b6969d830375ac7d7bc.
Complete configuration: selected-config.json; seed 42, five founders, 64-unit independent recurrent policies,
vmap enabled, CPU, four threads maximum, fresh weights, bounded 40-turn demonstration.

Changed: configuration schema 5 adds empty_water_penalty=5.0. An acting island with no unreserved,
locally usable water at turn end receives -5 in learning reward. Existing empty_food_penalty=5.0 and
health_loss_penalty_per_point=0.5 are unchanged and all three deductions stack. The last consumed
ration/sip triggers its empty-stock deduction; penalties repeat every empty turn. Incoming stock does
not count until delivered; parental supplies are assessed after inheritance. Newborns first learn on
their own decision turn. Terminal penalties are retained. Physical damage remains unchanged.
Old schema-4 configurations and checkpoints retain exactly their original food/health penalties.
The viewer exposes water remaining and its deduction separately; old recordings keep their breakdown.

Unchanged: health 300, daily food/water requirements 1 each, shortage damage 7 each, inventory reward
scale 0.01, currency value 0, Adam LR 0.0003 with the existing unhappiness multiplier, betas 0.9/0.999,
epsilon 1e-8, discount 0.99, value coefficient 0.5, gradient clipping 1, initial capacity 20,
temperature 2 to 1 and entropy coefficient 0.05 to 0.01 across 30 own updates, action-kind balancing.
No coefficient calibration or wider training sweep is part of this change.

Authorized CPU VM: kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
project he-vision-group, zone us-west4-a, worker 0. Fresh occupancy: AVAILABLE worker=t1v-n-4b587646-w-0.
No local neural workload or accelerator execution. Test source directory:
/tmp/island-community-water-20261109-060224-tests.
Test archive SHA-256: f27804b45713484895a18f388c5a1c1534cfc5e27c74a37c6961ab9bc9c3a25d.

Exact remote validation command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=tests.xml

Exact demonstration command from /tmp/island-community-water-20261109-060224/source:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-water-20261109-060224/recording

Test outcomes, published commit, source/result checksums, and final metrics will be appended.

## Completed validation and publication

Local rule/storage/viewer checks: 142 passed, 51 neural cases deselected, 25.15 s.
Remote complete suite: 193 passed in 40.36 s, including neural learning and exact checkpoint continuation
under configurations 1 through 5. Only README verification prose changed after these tests.
Published private GitHub commit: 8ced604a8575dddc06d6715768a2b20ceb8de8f2.
Source archive SHA-256: 331dd9af91160b0a37b994d18861462dd2f98ddbd6d482279f972c046f7bdc6d.
Git push to main succeeded. No PR was created for this standalone application update.

Browser checks against the prior recording passed after starting the updated viewer on port 36377.
The initial browser attempt used the still-running old server on port 41685; its previously imported
configuration validator could not load the newly versioned default rules, so the item guide timed out.
Starting the updated server fixed this without any further source edits. Both legacy and new reward
component displays, seeking/playback, and mobile layout pass. Old recording data remains untouched.

A second fresh occupancy check before the demonstration returned AVAILABLE worker=t1v-n-4b587646-w-0.

## Completed fresh demonstration

The 40-turn demonstration completed successfully with 41 contiguous frames, two living founders,
three deaths, and no births. There were 169 individual updates (40, 40, 29, 32, 28 by founder),
130 negative rewards, 98 empty-food deductions, 105 empty-water deductions, and 146 health-loss
deductions. Complete per-agent values and the final reward breakdown are in analysis.json.

Every recorded reward matches all three deductions plus inventory reward. Empty-necessity charges
match recorded usable balances; actual adjacent-frame net health changes match the health term.
Terminal inventory rewards are zero, lifetime scores sum the true learning rewards, and per-turn
learner mean rewards match the saved transitions. All learning metrics are finite. Source fingerprints
match the exact published commit; recorded configuration matches selected-config.json. The final frame
matches snapshot.json. These checks inspect saved results only, without local inference.

Results archive SHA-256 matched remotely and locally:
236a5ff93c0f4aab6fc43df15c69a31416d9d44581c90491a0b03cb3bfb2849e.
Artifacts include recording/ (config, source, manifest, checkpoint, experiences, frames, metrics, events),
run.log, tests.log, tests.xml, analysis.json, and results.tar.gz.

The short run confirms penalties reach learning; it does not establish that agents learned reliable
necessity trading. Inventory rewards can still exceed the fixed shortage penalties. No change to
physical damage, allocation, trade rules, model size, or learning rate was introduced.

Fresh replay: http://127.0.0.1:36221/ (server left running; opening queued in the app).

Final status: complete. Fresh replay browser checks passed for all 41 frames, including playback, seeking, mobile layout, legacy compatibility, and independent food/water/health deductions. Screenshot: replay.png.
