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

## Architecture and storage

| Module | Responsibility |
| --- | --- |
| `config.py`, `default.json` | Complete editable rules and validation. |
| `world.py` | Authoritative ownership, proposals, contracts, transport, production, survival, events, and lineage. |
| `policies.py` | Resident independent model tensors, recurrent memory, sampling, per-island learning, and inheritance. |
| `runner.py` | Turn coordination, birth/death synchronization, recording, and checkpoints. |
| `storage.py` | Atomic snapshots, append-only logs, source fingerprints, and versioned checkpoints. |
| `viewer.py`, `viewer.html` | Read-only local inspector with no external browser dependencies. |

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

The next value target uses a cloned preview of the next decision observation,
including scheduled arrivals and events. Previewing leaves the real world's
random state and event log untouched. This prioritizes correctness for the small
initial population; it is an optimization target if long histories become costly.

## Explicit economic choices

`default.json` is a complete configuration, not a partial overlay. Copy and edit
it, then validate before running. Unsupported formula names or transport modes
are rejected rather than silently ignored.

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
  defaults to zero. There are no survival, birth, or trade bonuses. Deceased
  agents receive a zero terminal reward and no further learning after their final
  own transition. No experience is replayed to other agents.
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
