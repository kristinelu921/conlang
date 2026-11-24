# Island Community

A runnable research simulator with five founding islands, a CPU world engine,
and small independent recurrent policies. PyTorch `vmap` evaluates the population
with separate weights. Death and ancestral replay are disabled.

The world engine, learning layer, and read-only viewer are separate modules.
Importing, validating, and viewing do not provision compute or execute a model.

## Quick start

Run commands from the repository root. Install dependencies in your chosen
environment:

```bash
python -m pip install -r island_community/requirements.txt
python -m island_community validate
```

The standalone GitHub checkout also supports an editable installation with
`python -m pip install -e '.[learning,test]'`. Installing without extras provides
the configuration validator and recorded-run viewer using only the standard
library. The installed command is `island-community`.

The repository permits model inference, simulation experiments, and training
only on authorized remote compute. On that machine:

```bash
python -m island_community run \
  --config island_community/default.json \
  --turns 100 \
  --device cpu \
  --output experiment_log/island-community-first-run
```

Use `--device cuda:0` for a GPU-equipped PyTorch environment. `vmap` is enabled
by default; `--no-vmap` selects an equivalent per-island forward loop. `--passive`
provides a wait-only, non-learning control. It still executes a simulation and
belongs on authorized remote compute.

The default output, when `--output` is omitted, is a unique directory under
`experiment_log/YYYY-MM-DD/`. An existing output directory is never overwritten.
Each run is bounded by `--turns`, or ends earlier on total extinction. A completed
run segment does not imply the population reached a stable equilibrium.

Inspect locally recorded or copied results without executing any models:

```bash
python -m island_community view experiment_log/island-community-first-run
```

Open **http://127.0.0.1:8765**. The inspector shows the transport network,
population and resource-value trends, island inventories and skills, active
shipments, and recent events. It polls existing files; it never advances the
simulation. Without a recorded run it displays an explicit empty state.

## Visual replay

New runs save a complete frame at initialization and after every turn. The
viewer opens at the first recorded turn. Use **Play/Pause**, the timeline slider,
previous/next, first/last, and **0.5×–4×** playback speed. Space plays or pauses;
the arrow keys step when a form control is not focused. **Follow newest turn**
follows an ongoing recording without disturbing a paused historical view.

Select an island to inspect its inventory, health, skills, chosen action, and
reward at that turn. Deceased islands remain selectable; their inventory is
explicitly labeled as the archived state before death accounting. Cargo diamonds
denote goods in transit, rather than a simulated physical position along a route.
The chart cursor follows the selected turn.

The item guide lists each item's uses, effects, duration, and manufacturing
recipe. New recordings also show saved nutrition, active improvements, and
expiry turns in island details. Older recordings explicitly label the guide as
a preview of the new default rules; their recorded outcomes are unchanged.

Playback reads saved frames; it never loads neural policies or changes learning.
The server indexes file offsets and small metric records, loads one world frame
per seek, and ignores incomplete trailing writes. The browser caches at most 64
frames. Chart histories over 2,000 turns are sampled across their full span;
individual selected frames remain exact. Full frames add disk usage proportional
to the number and size of recorded turns.

Older founding runs that have an experience journal can be reconstructed once
on authorized compute, using the matching world/config source version:

```bash
python -m island_community rebuild-replay path/to/old-run --output path/to/replay
python -m island_community view path/to/replay
```

Reconstruction uses recorded choices without neural inference. It checks source
hashes, every recorded observation and outcome, and the final world snapshot
before publishing the new directory. Original recordings stay untouched.
Incomplete journals, changed engines, and older continuation segments without
their starting state are rejected. New continuation segments automatically record
frames from their checkpoint turn. A recording without history still opens at its
latest snapshot, with playback disabled.

## Architecture and storage

| Module | Responsibility |
| --- | --- |
| `config.py`, `default.json` | Complete editable rules and validation. |
| `world.py` | Authoritative ownership, proposals, contracts, transport, production, survival, events, and lineage. |
| `policies.py` | Resident independent model tensors, recurrent memory, sampling, per-island learning, and inheritance. |
| `runner.py` | Turn coordination, birth/death synchronization, recording, and checkpoints. |
| `storage.py` | Atomic snapshots, append-only logs, source fingerprints, and versioned checkpoints. |
| `viewer.py`, `viewer.html` | Read-only local inspector with no external browser dependencies. |
| `replay.py` | Verified one-time reconstruction of older founding-run recordings. |

Island identity is permanent. A separate mapping connects that identity to a
reusable model-storage slot. Storage capacity starts at twenty and grows when
needed; it is **not a biological population limit**. Dead slots are cleared
before reuse. Birth creates independent copied and mutated parameters, fresh
optimizer moments and timestep, and empty recurrent memory. It happens before
the parent's update for that turn. Newborns act and consume starting next turn.

The CPU holds economic state and historical records. A single population object
holds each neural layer with an additional island dimension on the selected
device. Each agent receives its own observation, candidate actions, output,
gradient, Adam moments, timestep, and gradient clipping. `vmap` shares execution
code, not trainable weights or experience. It does not average island decisions.
The implementation follows PyTorch's
[functional model batching](https://docs.pytorch.org/tutorials/intermediate/ensembling.html).

Decision histories are appended to disk in `experiences.jsonl`, keyed by permanent
island ID. The journal records observations, selected actions, rewards, terminal
flags, and next observations. It is an audit trail; learning consumes only each
island's current transition. Historical transitions are never replayed on death.

The policy is a small recurrent actor–critic with a candidate-action scorer.
Candidate counts vary with visible offers, resources, capabilities, and partners;
padding is masked. Network widths remain fixed as the population changes.
Each turn applies a one-step temporal-difference update from that island's own
action and reward, with an entropy term. Recurrent state carries forward, while
gradients are truncated at each turn. This is a deliberately simple baseline,
not PPO or a claim of effective learned economic cooperation.

### Early exploration (configuration version 3)

New agents start at temperature **2.0** and entropy coefficient **0.05**.
Both taper linearly over each island's first **30 completed optimizer updates**
to temperature **1.0** and the policy's usual entropy coefficient **0.01**.
Temperature flattens the current choice probabilities; the entropy coefficient
encourages exploration during learning. Neither changes the learning rate.
These starting values are configurable initial choices, not calibrated optima.
Newborns start their own schedule; skipped updates do not advance it. A zero
annealing duration uses the final settings immediately.

Trade offers have many partner, item, and quantity combinations, while a recipe
has far fewer variants. A flat softmax over all candidates therefore assigns
most initial probability to trade proposals. Raising temperature alone cannot
fix that imbalance. The default now uses:

```text
adjusted_score(action) = network_score(action) / temperature
                         - log(available_candidates_of_this_action_kind)
probabilities = softmax(adjusted_scores)
```

Equal network scores give each available action kind equal total probability;
within a kind, equal-scored candidates share its probability. Learned scores
can still favor particular kinds and candidates. Only eligible candidates enter
the distribution, and padded entries remain masked. This balances command kinds
such as manufacture, item use, and trade proposal, not broader economic goals
or individual recipes. It does not force manufacturing or supply ingredients.

The learning bonus is entropy relative to that balanced prior:
`H(probabilities) - sum(probability * log(kind_candidate_count))`. Up to a
state-dependent constant, this is negative KL divergence to the balanced prior.
Ordinary flat entropy would again reward kinds merely for having more variants.
Sampling and the actor loss use the same adjusted scores and cached temperature.
Set `exploration.balance_action_kinds` to false to use flat candidate sampling.
The entropy bonus then becomes ordinary entropy.

Metrics record the temperature, entropy coefficient, raw entropy, corrected
exploration bonus, and candidate count as means over acting islands. Each
experience also records candidate counts by action kind. Checkpoints preserve
the settings, independent update clocks, and any pending sampling distribution.
Version 1 and 2 configurations and existing checkpoints retain flat sampling,
temperature 1, and their original entropy coefficient when resumed. Start a
new version 3 or later run to use these exploration settings.

The next value target uses a cloned preview of the next decision observation,
including scheduled arrivals and events. Previewing leaves the real world's
random state and event log untouched. This prioritizes correctness for the small
initial population; it is an optimization target if long histories become costly.

### Starting preference for staple trades (configuration version 6)

Fresh founders start with **2× the relative odds** for a qualifying grain/water
trade compared with the same initialization without the preference. The feature
applies to trade proposals, counteroffers, and acceptance. A proposal or counter
must request goods from their publicly known producer; acceptance uses the
visible deal terms. The deal must involve a survival-bundle item (grain or water
by default). Giving away food or water earns the preference only when the
island retains one turn of that necessity in unreserved, usable stock. Saved
meal nutrition counts. Other islands' private inventories are never inspected.

This is a trainable starting weight: one binary candidate feature is appended,
and its scalar scoring weight receives `starting_temperature * log(2)` when a
founder is initialized. The same normal sampling and learning paths use this
weight. No reward bonus, forced trade, extra acceptance action, or permanent
probability multiplier is added. Children inherit the learned weight with the
usual mutation; the founder offset is not added again. Checkpoint restoration
also preserves the learned weight without reinitializing it.

Configure `trade_initialization.initial_odds_multiplier` (default 2, set 1 to
disable the initialization offset) and `reserve_turns` (default 1). The odds
ratio is relative to an otherwise identical version-6 policy; it is not a
promise to double the number of trades. Subsequent learning and temperature
changes affect it. Versions 1–5 keep their original feature widths and weights;
use a fresh version-6 run for this initialization. The feature adds 65 parameters
per default 64-unit agent, bringing its total to 43,566.

## Explicit economic choices

`default.json` is a complete configuration, not a partial overlay. Copy and edit
it, then validate before running. Unsupported formula names or transport modes
are rejected rather than silently ignored.
Registry order determines feature layout and seeded sampling order; recorded
configurations preserve that order for reproducibility.

### Health and survival rewards (configuration version 5)

Founders and newborns start at **300 health**, which is also the healing cap.
Physical shortage damage remains 7 health per unmet daily necessity; neither
the food/water requirements nor damage rates were reduced.

Each acting island now receives:

```text
learning reward = inventory reward
                  - 5 if no usable food remains at the end of the turn
                  - 5 if no usable water remains at the end of the turn
                  - 0.5 * max(0, health before actions - health after upkeep)
```

Both empty-stock deductions apply every empty turn, including the turn an island
eats its last ration or drinks its last water. They warn that supplies have run
out even if today's consumption was satisfied. Grain, meals, and saved meal
nutrition count as food. Water is measured in drinkable units from local stock.
Reserved goods and incoming shipments do not count as usable supplies. For
surviving parents, supplies are measured after inheritance transfers too.
Newborns receive no penalties or learning updates until their first decision turn.

The health term measures actual net loss, including shortage, disease, aging,
and unhappiness damage, after healing and recovery. It cannot charge for damage
beyond the remaining health or grant a bonus for healing. All three deductions
stack: no food and no water, with 14 health lost, deducts 17 from the inventory
reward. These deductions do not inflict extra physical damage. The coefficients
live in `reward_shaping` and are configurable, finite,
nonnegative magnitudes. These are initial settings, not calibrated optima.

Inventory reward still values currently owned goods, including commitments and
recipient-owned transit, at scale 0.01. Currency remains zero-valued. Inventory
charts and the unhappiness calculation keep using that resource-only value,
preventing a feedback loop from negative learning rewards into unhappiness
damage. Version 4 and later lifetime scores sum the rewards actually sent to each learner.
On death, the inventory component is zero but the final survival penalties are
retained; versions 1–3 preserve their original zero terminal reward.

Experiences, audit events, and replay decisions record the inventory component,
food and water remaining, health lost, all deductions, and total reward. Select an island
in the replay to inspect the breakdown. Versions 1–3 and their checkpoints keep
their recorded health settings and inventory-only rewards. Version 4 preserves
its food and health penalties without adding the water penalty. Start a fresh
version 5 run for all three penalties; resuming an old checkpoint does not alter its rules.

### Item effects (rules version 2)

Every item now has a survival use or an explicit `use_item` action. Using an
item consumes one uncommitted unit and 0.1 of the turn's work budget. Holding
an item alone grants no bonus. Recipes and their ingredient quantities are
unchanged; materials can either be used directly, manufactured, or traded.

| Item | Effect per unit used | Duration |
| --- | --- | --- |
| Water | Supplies 1 water unit; required every turn. | Automatic upkeep |
| Grain | Supplies 1 food unit. | Automatic upkeep |
| Meal | Supplies 2 food units; surplus nutrition carries into later turns. | Automatic upkeep |
| Timber | Repair shelters: 25% less resource loss from negative events. | 8 turns |
| Stone | Reinforce shelters: 50% less resource loss from negative events. | 12 turns |
| Iron | Improve workshop fittings: +15% production speed. | 6 turns |
| Clay | Hygiene vessels: +1 health recovery per fully fed and watered turn. | 6 turns |
| Fiber | Bedding: 30% less unhappiness damage. | 6 turns |
| Herbs | Restore up to 3 health and shorten current illness by 1 turn. | Immediate |
| Salt | Preserve food: 50% less disease damage; does not remove contamination. | 5 turns |
| Fuel | Warm living quarters: +2 health recovery per fully fed and watered turn. | 3 turns |
| Cloth | Clothing: 60% less unhappiness damage and 20% less disease damage. | 10 turns |
| Tool | Equip workshop: +50% production speed. | 10 turns |
| Medicine | Restore up to 12 health and shorten current illness by 4 turns. | Immediate |

The normal requirement remains 1 food and 1 water per turn. One meal now covers
two turns of food, while water is still required on both turns. Stored nutrition
belongs to the eater, cannot be traded or inherited, and does not earn inventory
reward. Contaminated meals still cause illness when eaten.

Durations include the turn of use. A still-active improvement cannot be reapplied
or stacked with another unit of the same item. Different items add up to capped
bonuses: +100% production speed, +3 fed-turn health recovery, 80% less unhappiness
damage, 80% less disease damage, and 75% less event resource loss. Resource losses
round upward to whole items after protection. Workshop speed is fixed when a job
starts; already-started work is not repriced. Healing is capped at maximum health,
and medicine cannot prevent later reinfection. Improvements do not replace food
or drinking water and are not inherited by newborn islands.

Effects, remaining durations, and saved nutrition enter the island's private
policy observation. Eligible item uses appear in its action candidates. Use,
consumption, and expiry are recorded in the audit trail and replay frames.
Health improvements can help an island survive and protect future resources;
there is no new direct reward bonus for item use. These are illustrative balance
values, not demonstrated learned behavior or a calibrated economy.

Version 1 configurations and checkpoints retain their original food behavior,
action set, and neural feature dimensions. Start a new version 2 run to use the
new mechanics (also included in later versions); resuming an old checkpoint does not
silently change its rules.

- Exactly ten base items exist. The default additionally defines meals, cloth,
  tools, and medicine, with four recipes. Item identifiers are the registry keys.
  Quantities use integer smallest units; change the unit definition when a finer
  physical resolution is needed.
- Five founders collectively cover all allocation specializations. The initial
  survival bundle rounds immediately usable food and water upward on the
  logarithmic quantity scale. Starting specialization stock is separately logged.
- Allocation settings specify **global draws per simulation minute**, not draws
  per island. Quantity weights explicitly define the distribution.
- Default links form a complete, bidirectional graph. Opposite directions share
  per-turn dispatch capacity. Distances are Euclidean, fees are currency paid
  by the accepting party, and arrival takes at least one turn. Disabled channels
  postpone arrivals. Capacity bounds dispatched load per turn, not total cargo
  currently traveling along the channel.
- Pending offers reserve their supplier's promised inputs and any payment
  promised by the proposer. Consumption,
  manufacturing, and births use uncommitted stock. Rejection, cancellation,
  expiry, and failure release the relevant commitments. There is no parent
  survival reserve.
- Accepted bilateral trades dispatch their legs atomically. Cargo becomes the
  recipient's property at dispatch and contributes to that recipient's resource
  reward once, but is unusable until arrival. Trade payment is immediate; service
  payment is held in customer-owned escrow until delivery. Already dispatched
  cargo keeps its ownership if a contract later fails.
- Manufacturing consumes inputs before work begins. Work uses the configured
  time budget and may carry forward. Service suppliers provide inputs, skill,
  and time. Teaching retains the teacher's skill; skill transfer subtracts the
  transferred units from the supplier. Capability transfer is explicit.
- Structured public/private requests and negotiation are supported. New proposals
  are visible for acceptance on the following turn. The neural policy chooses
  one candidate command per turn. The engine additionally accepts multi-command
  `Action("plan", {"actions": [...]})` plans and batch manufacture commands.
- All physical inventory changes have ledger entries. Contaminated quantities
  remain distinguishable and can spread disease through incoming food. Skills
  and productivity deteriorate on the configured schedule.
- Reproduction is automatic when due and funded. The child receives the survival
  bundle plus diminishing additional inheritance. All transferred stock leaves
  the parent. No population cap, compulsory replacement, or artificial killing
  enforces a desired population trajectory.
- Resource rewards are measured after deaths and births. Committed stock and
  recipient-owned transit count; consumed work inputs do not. Currency valuation
  defaults to zero. Version 4 and later subtract the applicable survival penalties described above;
  there are no birth or trade bonuses. Deceased agents retain their final penalties
  and receive no learning after their final own transition. No experience is
  replayed to other agents.
- The default dead-asset rule destroys remaining owned assets, with logged
  accounting. Extinction stops the run. The central experience bank discussed
  earlier is optional future work and is not enabled here.

The recipe validator currently uses a conservative no-value-amplification rule:
output resource reward value cannot exceed input value. This prevents chains
from manufacturing reward through unit conversions, but intentionally excludes
recipes that create a reward premium for labor. Manufacturing can still improve
survival contribution and transport efficiency. Alternate economic rules need
an explicit validator change and tests.

The defaults are illustrative. Their population dynamics and learning quality
have not been calibrated to a population target. A five-turn endowment guarantees
ordinary physical upkeep, not successful learned trading or immunity to the
separately configured unhappiness, disease, aging, or destructive events.

## Reproduction, continuation, and artifacts

Each run directory contains:

- `manifest.json`: exact invocation, resolved configuration, execution settings,
  status, timestamps, dependency versions, and hashes of the copied source.
- `source/`: the exact bytes used for the source fingerprints.
- `config.json`: the complete run configuration.
- `snapshot.json`: the latest complete world snapshot, written atomically.
- `frames.jsonl`: append-only complete world states and compact choices for visual
  replay, including the initial state of each run segment.
- `metrics.jsonl`: append-only per-turn population, resource, health, and learning
  statistics.
- `events.jsonl`: append-only resource and lifecycle audit events for this run
  segment.
- `experiences.jsonl`: each acting island's own decision and transition history.
- `checkpoint.pt`: the latest complete world and policy continuation state.

Checkpoints include all living weights, optimizer moments and clocks, recurrent
memory, random-generator state, identity mappings, pending world obligations,
and the resolved configuration. Restore into a **new** segment directory:

```bash
python -m island_community run \
  --resume experiment_log/island-community-first-run/checkpoint.pt \
  --turns 100 \
  --device cpu \
  --output experiment_log/island-community-continuation
```

Resume uses the checkpoint's configuration. `--turns` means additional turns;
`--config` cannot silently alter a continuation. Exact numerical continuation is
tested on the same backend; changing hardware or execution mode can change
floating-point decisions. A failed run preserves its last complete checkpoint,
which may precede the most recently recorded metrics.

## Verification

The original implementation was validated on November 9, 2026: **77 tests passed** on authorized remote CPU
compute with Python 3.10.18 and PyTorch 2.2.2. The standalone wheel also builds,
installs, and validates rules without installing PyTorch when learning extras
are omitted.

The replay extension additionally passed 60 non-neural checks, two remote
checkpoint/resume integration checks, and headless Chromium interaction checks.
The original example was reconstructed into 24 verified frames (turns 0–23).

The item-utility update passed **124 tests** on the authorized remote CPU,
including a neural item-use transition and exact checkpoint continuation under
both rule versions. Chromium checks cover the complete item guide, active-stat
inspection, legacy-recording labels, replay controls, and a narrow viewport.

The exploration update passed **150 tests** on the authorized remote CPU,
including count-balanced probabilities and entropy gradients, per-island
annealing, matching sampling/training distributions, vmap/loop agreement, and
exact checkpoint continuation under all three configuration versions.

The health and survival-reward update passed **179 tests** across remote
validation and a corrected integration-fixture rerun. Coverage includes all four
configuration versions, usable-food accounting, repeated and terminal penalties,
healing, inheritance, negative rewards reaching neural updates, and checkpoint
continuation. The local non-neural subset also passed all 129 checks.

The separate water-penalty update passed **193 tests** on the authorized remote
CPU and **142 local non-neural checks**. Coverage includes all five configuration
versions, repeated empty-water penalties, penalties after the last sip, both
shortages together, replenishment, reserved/incoming stock, terminal rewards,
and exact checkpoint continuation with independent recurrent learners.

The staple-trade initialization passed **216 tests** on authorized remote CPU,
including an isolated twofold initial-odds check with both vmap and loop execution,
normal learning of the initialized weight, inheritance without adding the offset
again, and checkpoint continuation under all six configuration versions. The
157 local non-neural checks cover producer targeting, private-state independence,
trade direction, saved nutrition, committed supplies, and preserved legal choices.

In the first default-seed example, five untrained founders reached extinction
at turn 23. The defaults have not been calibrated for a stable population of
15–20. The original policy layout used about **9.7 MiB** across twenty slots for weights, Adam
state, and recurrent memory; temporary activations, gradients, allocator/runtime
overhead, and world/history storage are additional. GPU execution and peak
GPU memory have not been measured.

Local configuration, accounting-unit, storage, and viewer tests:

```bash
python -m pytest island_community/tests -m 'not neural' -q
```

Neural and end-to-end checks must run on authorized remote compute. Enable them
explicitly there:

```bash
ISLAND_ALLOW_NEURAL_TESTS=1 python -m pytest island_community/tests -q
```

Neural tests cover vmap/loop agreement, separate updates, Adam agreement,
inheritance, memory growth, slot clearing, and checkpoint continuation. Without
the explicit environment variable, neural tests are skipped. Test fixtures are
synthetic invariants, not reported economic experiments.

Browser interaction checks are optional and require Node.js and Playwright.
Against a viewer serving at least three recorded frames:

```bash
npm install --prefix /tmp/island-browser-tests playwright
PLAYWRIGHT_BROWSERS_PATH=/tmp/island-browser-tests/browsers /tmp/island-browser-tests/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/island-browser-tests/node_modules PLAYWRIGHT_BROWSERS_PATH=/tmp/island-browser-tests/browsers node island_community/tests/check_replay_browser.cjs http://127.0.0.1:8765
```

The checks cover seeking, playback, pause stability during refresh, island
inspection, delayed-response races, and a phone-sized viewport. Minimal Linux
images may also need the system libraries documented by Playwright. These checks
only inspect recordings and do not execute simulation or neural workloads.
