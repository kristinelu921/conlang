# Island Community

A runnable research simulator with five founding islands, a CPU world engine,
and small independent recurrent policies. PyTorch `vmap` evaluates the population
with separate weights. Death and ancestral replay are disabled.

The world engine, learning layer, and read-only viewer are separate modules.
Importing, validating, and viewing do not provision compute or execute a model.

## Qwen agents with private history

`scenarios/qwen_10_frozen.json` starts ten islands using a shared, frozen
`Qwen/Qwen3-0.6B`, pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Each island receives its exact
current inventory, reservations, available currency, health, work, incoming
shipments, permitted proposals/messages, five neighbors' public producer types,
and eight turns of its own actions, execution outcomes, and rewards. Other
islands' inventories and histories are excluded. All choices use the same
pre-action world snapshot; the world resolves them only after everyone chooses.

On authorized remote compute, install `requirements-qwen.txt` and run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m island_community run \
  --config island_community/scenarios/qwen_10_frozen.json \
  --turns 20 --device cpu --checkpoint-every 1 \
  --output experiment_log/qwen-frozen-10-islands
```

Native TPU execution uses a separate environment with
`python -m pip install -r island_community/requirements-qwen-tpu.txt` and
`--device tpu`. The pinned stack is Torch 2.15.0, TorchTPU 0.2.0,
Transformers 5.22.2, and NumPy 2.5.4. The backend selects one TPU device for the
shared float32 model, retains generation token bookkeeping on the CPU, and
uses eager attention with 512-token prompt chunks and a static attention cache
sized by `max_context_tokens`. Chunking keeps every prompt token while limiting
attention memory. Native SDPA produced a VMEM bounds error with the full model's
attention dimensions on v5e. Generation's optional automatic compilation is
disabled because the installed TPU compiler rejects a CPU cache scalar in that
path. Each island still
has its own history and optional adapter; the base model is loaded once.
Set `TPU_LOG_DIR` and `GOOGLE_LOG_DIR` to a writable task-local directory.
Let the backend configure device visibility through TorchTPU's launch helper;
do not set `TPU_VISIBLE_DEVICES` independently. This native backend has been
tested on v5e; the installed TorchTPU version does not discover v4 devices.

Teacher-forcing updates also divide attention into 512-query blocks, each
retaining every key/value token and its original causal mask. Gradient-bearing
blocks use activation checkpointing to recompute attention intermediates during
backpropagation. This avoids retaining full sequence-by-sequence attention
matrices for adapter training. Shared weights remain float32 and the full context is
preserved; CPU/CUDA retain their existing SDPA backend. Tests compare chunked
and ordinary eager outputs and gradients on the same backend, and separately
check model logits and greedy cached decoding against CPU.

The small scenario explicitly permits unassigned producer types: two water,
two grain, and one each of timber, stone, iron, clay, fiber, and herbs.
Salt and fuel have no dedicated producer. This is necessary to retain mutual
five-neighbor grain/water access with only ten single-resource producers.
Older scenarios still require every resource type to have a producer.
All other economic rules, including automatic reproduction, remain in force.

Qwen generates at most 96 reasoning tokens per decision stage. A token-prefix
grammar then restricts its final response to exactly `{"choice":N}`, with N
drawn from the complete zero-based menu shown in that prompt. The parser rejects
extra text, malformed JSON, duplicate fields, booleans, non-integers, and invalid
IDs. The selected menu object is submitted unchanged. For trades, a second
menu selects the exact integer quantities from all feasible combinations. A
parser or context overflow error stops the segment; it never becomes a silent
wait. Context is never silently truncated. Generated reasoning is a model
output, not a verified explanation of its decision.

The authoritative world returns the requested command, its execution status,
and the event sequence IDs it produced. Execution can fail after a valid
decision because simultaneous actions change availability. Both outcomes are
recorded faithfully and enter private history. A successful proposal does not
mean acceptance or delivery. The replay's model panel exposes the exact prompt,
generated reasoning, JSON response, full menu, and execution status. These
records are not per-action GRU logits.

Configuration version 9 also supports `language_model.mode=reward_adapter` or
`value_adapter`. `scenarios/qwen_10_reward_adapter.json` and
`scenarios/qwen_10_value_adapter.json` match the frozen scenario with only the
mode changed, for a controlled three-condition comparison. Each island has
rank-eight LoRA residuals on the last two
layers' query/value projections, with its own Adam state (learning rate 1e-5,
gradient clip 1). The shared pretrained weights remain frozen. `reward_adapter`
uses the island's immediate reward, scaled by 0.1, in a sequence policy-gradient
update. `value_adapter` instead uses a one-step TD advantage with discount 0.99
and its own small observation-to-value MLP (hidden width 64, Adam 1e-3).
Its value targets use the same 0.1 reward scale. This critic predicts future
return; the simulator remains the source of actual rewards.

Policy-gradient updates include the sampled reasoning and answer tokens and
record the exact token sets allowed by temperature/top-k/top-p/grammar sampling.
Training recomputes logits on those recorded sets; it does not rerun a truncation
filter whose boundary can shift numerically between cached decoding and replay.
Forced formatting tokens and a forced reasoning-budget closure are not treated
as unconstrained actions. No experience or optimizer state is shared between
islands. Children inherit their parent's current adapter and optional critic,
with fresh history and optimizer states and no parameter mutation. Frozen
children share the same pinned model and start an empty private history.

Checkpoints store histories, adapter weights, optimizer moments, and the pinned
model reference; they do not duplicate the large pretrained weights. A complete
before/after hash verifies those weights remained unchanged. Decision-level
learning records include rewards, advantages, value targets, gradient norms,
and adapter change norms. The base model is loaded once and island decisions
are processed sequentially on the selected device.

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

For more than twenty recorded islands, the map uses compact numbered markers
and shows routes and cargo for the selected island. Select any marker or use
the island menu to inspect another island; the underlying recording is unchanged.

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

### Fifty islands with proportional resource gifts (configuration version 7)

`scenarios/population_50.json` is a complete alternative configuration. It starts
50 independent agents, each with the usual eight grain and eight water plus
eight units of its specialty. Producer types follow the existing resource
weights using largest-remainder rounding: 13 water, 13 grain, and 3 of each of
the other eight base resources. These assignments are shuffled with the run
seed, so producer types are spread around the island ring reproducibly.

With `allocation.scope = "per_island"`, `allocations_per_minute = 1`, and
`turns_per_minute = 1`, **every living island gets one gift every turn**, from
its own specialization. The weighted gift-size distribution stays unchanged
(1–32 units with positive probability, mean about 9.14). There are initially
50 gift draws per turn; births add producers and deaths remove them. Income
reduction events still affect gift sizes. Newborns first receive a recurring
gift at the next scheduled allocation. Producer proportions can evolve as
islands die or reproduce; they are not forcibly restored.

This preserves roughly the earlier per-capita total supply while spreading it
across producers. The initial expected supply is about 119 grain and 119 water
per turn, against 50 units of each for daily consumption alone. Manufacturing,
transport delays, reservations, and losses can still cause local shortages.
Every island remains dependent on trade for at least one necessity.

The 50-agent scenario reserves 50 policy slots (about 24.94 MiB for weights,
Adam state, and recurrent memory). Candidate menus, temporary neural activations,
Python world state, and histories are additional; the all-to-all partner menu
grows much faster than the policy storage. Rewards, optimizer settings, starting
health, and the learned trade preference are unchanged. More agents do not pool
learning: each policy still learns only from its own experiences.

Run on authorized compute:

```bash
python -m island_community run --config island_community/scenarios/population_50.json --turns 100 --device cpu --output /path/to/new-run
```

The five-island `default.json` remains the version-6 baseline. Versions 1–6 retain
their original global allocation behavior and seeded gift sequence. Version 7
requires an explicit allocation scope; `global` uses `uniform_eligible` recipients,
while `per_island` uses `self`. For islands with multiple specializations, the
per-island draw uses resource weights restricted to that island's own types.

### Five mutual neighbors and private history (configuration version 8)

`scenarios/neighbors_50.json` starts the same 50 producers, with a maximum of
five mutual communication neighbors per island. Every founder has a distinct
grain producer and water producer among its five contacts, including founders
that themselves produce one of those resources. Producers obey the same limit.
These contacts need not be the five geographically closest islands.

Direct messages, public requests, trades, services, and teaching all use this
neighborhood. Public requests reach only the sender's current neighbors, with
no forwarding or global broadcast. Execution checks the same restrictions as
the candidate menu. Only public producer types and locations influence contacts;
private inventories and experiences do not.

After deaths or births, the network is rebuilt with a bounded deterministic
search that prioritizes producer coverage, fills free contacts, and favors
existing links. The five-neighbor maximum always applies. With an odd or small
population, some islands can have fewer than five contacts. If producer supply
or contact capacity no longer permits full coverage, missing access is recorded
and shown in the inspector. No replacement producers or resources are invented.
Pending offers on removed contacts are cancelled and their reservations released.
Already accepted cargo and services retain their original physical delivery routes.
Channel outages can still delay shipping; a producer contact does not guarantee
that the producer will agree to a trade or have unreserved stock.

Each policy still chooses **one action per turn**. Version 8's `all_feasible`
quantity mode first chooses the action, partner, and items, then scores every
feasible pair of positive whole-number quantities for that selected trade.
For example, 3 grain for 7 water is possible; the two amounts are independent.
Purchases can request any feasible whole number, with their price recomputed.
The proposer cannot promise more than its unreserved stock or spend more than
its unreserved currency. Combined transport load must fit the channel; with
unit-load goods and a capacity of 32, a barter can exchange 1–31 on each side
subject to their sum being at most 32, or a purchase can request up to 32.
The partner's hidden stock does not affect these choices; acceptance revalidates
the actual stocks and costs. A policy cannot both propose and accept as its
own chosen action that turn, but several counterparties can accept its outstanding
offers. Thus there is no global one-completed-trade-per-island limit. The engine's
manual `plan` command can submit multiple commands within a time budget, but the
neural policy does not generate plans. Channel capacity remains 32 load units
per edge per turn, and fees, stock, and available time still constrain execution.

Proposals are numeric candidate features, not natural-language prompts. Each
visible incoming proposal supplies separate reject, accept (if feasible), and
counter choices. Features encode the action/proposal type, per-item offered and
requested quantities, price, payer, transport fee and distance, expiry, partner
identity/location and public producer types, and relevant message/skill signals.
Offers become visible on the following turn. Exact proposal IDs are carried in
the executable actions; the network sees their terms and remaining lifetime.
Quantities are scaled with `tanh(quantity / 16)`, rather than tokenized as text.

The conditional quantity choice reuses the same policy weights and current
recurrent state, without advancing memory a second time. Learning sums the
selected template and quantity log probabilities, so the reward trains both
choices. A normalized conditional entropy bonus encourages amount exploration.
Checkpoints preserve both pending choices and sampling state. Experience records
save the actual quantities, both selected indices, and the quantity option count.
Earlier schemas keep their fixed quantity menus and original sampling sequence.

The existing 64-unit recurrent state already carries information across turns
and is saved in checkpoints. Version 8 additionally feeds the last **eight own
completed turns** directly into each observation: chosen action features, realized
reward, health, usable food, and usable water. These records are chronological,
with a presence flag and leading zero padding; newborns start empty. Snapshots
include the latest completed turn, so that record becomes input on the next turn.
History is private to that learner, including after checkpoint restoration.

The observation grows from 112 to 1,024 values. Each 64-unit policy now has
101,934 parameters; fifty reserved slots use about 58.34 MiB for weights, Adam
state, and recurrent memory, before temporary tensors and world/history storage.
The explicit window complements recurrent memory; it does not guarantee useful
long-term recall. Learning is still a one-step actor–critic update with gradients
truncated each turn. This change does not implement multi-step returns, sequence
backpropagation, shared experience, or death replay. Rewards are unchanged, and
older rule versions and checkpoints retain their feature layout.

Run on authorized compute:

```bash
python -m island_community run --config island_community/scenarios/neighbors_50.json --turns 100 --device cpu --output /path/to/new-run
```

### Larger recurrent agents

`scenarios/neighbors_50_gru256.json` increases the recurrent width from 64 to
256 while preserving every other setting in `neighbors_50.json`, including
seed 42, fifty founders, eight explicit history turns, learning rate 0.0003,
and the reward and exploration schedules. Start a fresh run on authorized compute:

```bash
python -m island_community run --config island_community/scenarios/neighbors_50_gru256.json --turns 100 --device cpu --output experiment_log/island-community-gru256
```

The architecture is a 1,024-value observation projected to the chosen width,
one GRU layer of that width, a candidate-scoring head, and a scalar value head.
Each action has 109 features; its embedding is scored against the recurrent
state. The conditional quantity decision reuses the same model. Every island
owns independent weights and optimizer state; `vmap` batches computation.

| GRU width | Parameters per island | Resident tensors for 50 slots |
| --- | ---: | ---: |
| 64 (baseline) | 101,934 | 58.34 MiB |
| 128 | 261,102 | 149.43 MiB |
| 256 (new preset) | 751,470 | 430.04 MiB |
| 512 | 2,420,334 | 1,385.02 MiB |
| 1,024 | 8,510,574 | 4,869.99 MiB |

These are analytical float32 sizes for weights, two Adam moments, recurrent
states, and optimizer clocks. Activations, gradients, pending candidates,
runtime overhead, world state, and diagnostics add memory. Reserved capacity
can grow with births. Changing width requires a fresh run; it cannot resize an
existing checkpoint. The 128/512/1,024 rows are estimates, not measured runs.

Larger models increase capacity, but this preset still uses one-step
actor–critic learning and stops gradients between turns. It does not add
multi-turn backpropagation or PPO. A sensible subsequent experiment is a
recurrent learner trained on 16–32-turn sequences with multi-step advantages;
that would address delayed credit assignment separately from model size.

### Per-turn option counts, logits, and probabilities

New learned runs save the actual sampling scores for every valid option under
`decisions/<turn>.json`, with entries keyed by island ID. Each action stage and
conditional quantity stage records the raw logit, temperature-adjusted logit,
action-type count correction, softmax probability, executable candidate, and
selected index. Padding is excluded. Quantity scores cover the selected template
only; the policy does not evaluate quantity combinations for every other template.
A non-trade has a single unchanged quantity option with probability one.

The replay's **Model choices & logits** panel follows the selected island and
turn. Switch between action and amount stages, filter by goods or action names,
inspect probability bars and the chosen row, or download the full score table.
Rows are ranked by probability. Sampling can select an option other than the
highest-scoring one. The displayed formula is `raw_logit / temperature - log(kind_count)`
for balanced action sampling; quantity sampling has no action-kind correction.

The panel fetches one island on demand from `/api/decision?turn=T&island=I`;
world frames contain only option counts, keeping ordinary playback compact.
These scores are captured before learning from the turn, with no additional
model evaluation or random draws. Capturing scores is tested to preserve actions,
hidden state, gradients, optimizer state, and random-generator continuation.
The observer can inspect scores; other islands do not receive them.

Older recordings did not save logits and show an explicit unavailable message.
Their final checkpoint alone cannot supply the historical pre-update logits.
Recovering those requires rerunning the original initial state and learning
trajectory on compatible authorized compute and checking that the recorded
actions and outcomes match.

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

The proportional per-island allocation update passed **230 tests** on authorized
remote CPU and **170 local non-neural checks**. Coverage includes exact founder
quotas, population-scaled gifts, birth/death handling, income reductions, seeded
legacy compatibility, and checkpoint continuation under all seven rule versions.

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
