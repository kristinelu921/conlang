# Fifty islands with proportional producer types and per-island gifts

Date: November 9, 2026 UTC. Status: validation in progress.
User asked about reward adequacy and a 50-island world, then clarified that every island should receive
resource gifts every turn, with different resource types assigned across islands at an appropriate scale.

Selected configuration: selected-config.json, identical to island_community/scenarios/population_50.json.
Baseline source: 918ba38f129a37d4f886ede864de19e83ac58d3d. Baseline population-5 recording:
island-community-trade-20261109-061718 plus continuation island-community-trade-100-20261109-062911.
Seed 42; 50 fresh independent 64-unit RNNs; CPU; vmap; at most four threads; requested 100 turns;
checkpoint every 10 turns; existing extinction rule still applies. No sharing or death replay.

Changed: schema 7; population 5 to 50; policy initial capacity 20 to 50; specialization rule
proportional_single_resource and maximum specialization count 1. Largest-remainder producer quotas
use the existing resource weights (water 4, grain 4, others 1): 13 water, 13 grain, 3 each of eight
other resources. Assignments are shuffled with the world seed. Every agent receives the same baseline
starting 8 grain and 8 water plus 8 of its specialty and 30 currency.

Allocation changes from five global gift draws per turn to one draw per living island per turn
(scope=per_island, recipient_rule=self, allocations_per_minute=1). Each draw is restricted to that
island's producer types. Gift-size probabilities remain unchanged: expected 201/22 = 9.13636 units.
Initially 50 draws yield expected 118.7727 grain and 118.7727 water per turn, against daily consumption
of 50 each. Other resources have expected 27.4091 units each per turn. This is an expectation before
income reductions, manufacturing consumption, losses, reservations, and transport. Each death removes
one producer's gifts; a newborn receives gifts beginning with the next allocation. Quotas are enforced
only at founding, not after deaths/births. Existing global rules and seeded draws remain unchanged.

Unchanged: reward (inventory value *0.01, -5 each empty necessity, -0.5 per actual lost health point),
health 300, daily needs 1 food and 1 water, shortage damage 7, LR 0.0003 and unhappiness multiplier,
Adam 0.9/0.999/1e-8, discount 0.99, value coefficient 0.5, gradient clip 1, temperature 2 to 1 and
entropy 0.05 to 0.01 across 30 updates, 2x learnable initial staple-trade odds, reproduction settings,
trade sizes, topology, and transport. Reward redesign is discussed separately, not silently applied.
The five-island default.json remains unchanged. This scenario changes both population and allocation;
a single run is descriptive, not a controlled estimate of either change's separate effect.

Local validation: recorded seed-42 first-turn global gifts captured as a regression test before edits;
that test passed. Then 170 non-neural checks passed, 60 neural checks deselected, in 26.78 seconds.
Remote CPU VM: kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
project he-vision-group, zone us-west4-a, worker 0. Initial occupancy AVAILABLE worker=t1v-n-4b587646-w-0.
No local neural execution, no provisioning, no accelerator workload.
Test directory: /tmp/island-community-population50-20261109-064546-tests.
Test archive SHA-256: 53574202c532b3c9bde6a64528f1c044c9adabe5dffafe257fef916c32dc2620.

Exact remote validation command:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ISLAND_ALLOW_NEURAL_TESTS=1 PYTHONPATH=/tmp/island-community-verify-20261109/test-dependencies /home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest island_community/tests -q --junitxml=tests.xml

Selected demonstration command from /tmp/island-community-population50-20261109-064546/source:

    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/scenarios/population_50.json --turns 100 --device cpu --checkpoint-every 10 --output /tmp/island-community-population50-20261109-064546/recording

Results, exact published commit, source checksum, and replay verification will be appended.

Validation and source publication:
- Full remote suite: 230 passed in 48.29 seconds; no failures.
- Installable wheel built using the declared isolated build dependencies; the packaged scenario JSON
  matches the selected configuration. An initial no-build-isolation attempt used the development
  machine's obsolete setuptools and produced UNKNOWN metadata; that artifact is not used.
- Published source: 86fe0d28d51330fd372b990f8b6d3c6df878e9df on main of
  https://github.com/kristinelu921/island-community (private).
- Exact git archive: /tmp/island-community-population50-20261109-064546-source.tar.gz.
- Archive SHA-256: d6965a31e9b2aa52b963370b79898004e01cbdcdc44ebcb082597f108970f779.
- Fresh named VM occupancy check before demonstration: AVAILABLE worker=t1v-n-4b587646-w-0.

Replay display adjustment during the run:
- The existing browser check reproduced a real 50-island bug: adjacent text labels intercepted
  clicks on Island 0. A first recheck still used the server's cached old page; restarting that
  local viewer loaded the change.
- Above 20 recorded islands, compact numbered markers replace overlapping labels. Routes and
  cargo shown on the map belong to the selected island; the inspector and full recording retain
  all islands and events. Marker labels do not intercept clicks.
- Updated browser checks passed on the partial 50-island recording (6 frames), including first,
  middle, and last island selection and selected-route counts. The same checks passed against
  the complete previous 5-island recording (73 frames) using the updated viewer.
- Viewer unit tests: 5 passed. The initial restricted attempt could not open local test sockets;
  rerunning with local-server permissions passed. No simulation or neural code was changed.
- Remote run source remains 86fe0d28d51330fd372b990f8b6d3c6df878e9df throughout.

Display publication: c96fa0d36bfe1234630796c6f309a2831323f2fe (private main). This commit changes
only README, viewer.html, and the browser check; it does not change the running world or policies.
The final viewer remains at http://127.0.0.1:35961/.

Interim observations from recorded JSON (not additional simulation):
- At turn 18, all 50 founders remained alive, mean health 202.394/300. Living inventories held
  2,186 water and 2,145 grain, while 37 actors had no usable water and 37 had no usable food.
  These usable counts come from reward components and therefore include nutrition/reservations.
- 900 individual updates and exactly 900 recurring gifts through turn 18. Actual delivered goods
  included 14 grain, 12 water, and 9 meals; all contract kinds combined had 30 acceptances and
  17 completion events. These counts are preliminary, not final trade-only totals.
- At turn 33, 27 founders remained; 23 deaths and zero births. Turn-30 checkpoint was saved.
- Current action menu permits 1- or 4-unit trades only, and one chosen action per island per turn.
  More efficient bulk replenishment is a separate possible mechanism change; none was applied.

Reward-design diagnosis and proposed next comparison (not implemented in this run):
The current reward pays 0.01 times owned inventory value on every turn, then subtracts five each
for empty usable food and water, and half of actual net health lost. Growing producer stock can
therefore offset shortage penalties. Positive reward alone does not imply the actor reinforces a
bad choice: its update depends on an advantage relative to its critic. Nevertheless, the uncapped
wealth term and automatic gifts make the learning target poorly aligned with useful distribution.

Prior five-island evidence: Island 1 accepted a water trade at turn 49, receiving reward 3.96978;
its four water arrived at turn 51, when its current action was an unrelated private clay request
and reward was 12.7. At turn 50 it still had no water, lost about seven health, and earned positive
4.06993 because its inventory term was 12.57. The outcome is rewarded, but the one-step update
must propagate credit indirectly through its value predictions to the earlier agreement.

Recommended next objective: survival/health plus the existing separate shortage penalties, with
bounded/diminishing value for useful reserves, replacing uncapped repeated inventory payouts.
One standard shaping construction is gamma * Phi(next_state) - Phi(current_state), where Phi
measures usable food/water days capped at an initial experimental five-day target and terminal
Phi is zero. This is inspired by potential-based shaping, not a claim of policy-invariance for
this partially observed, changing multi-agent neural setting. Do not pay for proposal/acceptance
counts themselves; circular or useless trades could collect that reward.
Primary reference: Ng, Harada, Russell, Policy invariance under reward transformations: Theory
and application to reward shaping (1999):
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

Use several-turn returns (for example, an initial 8–16-turn rollout) and recurrent training over
those sequences so delayed deliveries can inform earlier choices more directly. No ancestral or
death replay is required. These rollout lengths and reserve targets are tuning proposals, not
calibrated values. The current discount 0.99 already has a relatively long nominal horizon;
merely increasing it does not solve sparse short-run experience or one-step credit propagation.
Primary reference: Mnih et al., Asynchronous Methods for Deep Reinforcement Learning (2016):
https://proceedings.mlr.press/v48/mniha16.pdf

Fifty independent learners do not pool experience. A single seed and at most 100 own updates per
founder cannot establish a trained cooperative equilibrium. Population, gift rule, reward, and
larger trade-quantity changes should be compared separately across multiple seeds.

## Final outcome and verification

Status: completed normally at natural extinction on turn 65 (exit status 0). The requested upper
bound was 100 turns; no empty turns, replacement population, or reward modifications were added.
All 50 founders died; zero births. There are 66 replay frames, covering turns 0–65 without gaps.
The final two surviving islands were water producers 4 and 13; their death turns were 62 and 65.

There were 2,131 individual RNN updates and exactly 2,131 recurring specialty gifts. Every acting
island received exactly one gift on each turn, including income-adjusted draws; no deceased island
received subsequent gifts. Founding quotas exactly match 13 grain, 13 water, and 3 each of the eight
other base types. The world generated 6,099 grain and 6,577 water through recurring gifts alone.
Actual deliveries included 55 grain, 45 water, and 9 meals (the receipt ledger also includes service
output shipments). Thirty-two trade contracts were accepted; 30 completed and 2 failed. Separately,
5 service contracts and 29 teaching contracts were accepted. These are accepted contracts, not
merely selected accept actions, which can fail authoritative validation.

The run took 14 minutes 29.65 seconds on remote CPU, with a maximum resident set of 3,532,292 KiB
(about 3.37 GiB), including the Python world, action menus, policies, and temporary tensors. This is
the measured memory for this CPU run, not a GPU-memory measurement. No local models were executed.

Integrity and accounting checks passed: selected config and manifest match; every recorded source
fingerprint matches the exact published run commit; all frames and event sequences are contiguous;
learner metrics and rewards are finite; reward components, learner mean rewards, and lifetime totals
agree; every gift matches a living actor's specialty; the final snapshot and manifest agree on
extinction. Detailed totals and sampled world stocks are in analysis.json. The read-only validation
program is preserved as validate_recording.py; it was executed as python /tmp/analyze_island_population50.py
before copying that identical program into this artifact directory.

Remote and local SHA-256 hashes agree:
- Results archive: a5fccecdb9c319c0642d53b480aa827f08c4784c0062e296278eb96392aa129b.
- Final checkpoint: 1b5d8a8d9c5c68fb45a77846fcfca18c19d1a7b67bf3860932d32f42a379d7d6.

Artifacts: recording/ (complete logs, frames, configuration, source copies, and checkpoint),
results.tar.gz, run.log, exit-status, tests.log, tests.xml, selected-config.json, analysis.json,
validate_recording.py, replay-preview.png, and replay.png. The private GitHub main branch contains
both the scenario implementation and independently verified display fix. The recording remains
based on the earlier run-source commit; display changes were never injected into the running job.

Interpretation: supply was ample in aggregate but remained concentrated in producers. The larger
population did not establish sustainable cooperation in this seed. Population and allocation both
changed, so this does not isolate the causal effect of either and is not a calibrated reward study.

Final browser verification passed for all 66 recorded frames, including compact-map selection,
route counts, playback, seeking, historical stability, deceased-island inspection, reward breakdowns,
and mobile width. The exact run-source archive is also preserved as source.tar.gz. The local replay
server remains running at http://127.0.0.1:35961/; opening its tab was requested in the app.
