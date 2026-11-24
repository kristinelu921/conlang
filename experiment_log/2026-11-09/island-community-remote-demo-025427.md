# Island Community remote CPU validation and first simulation

## Run Summary

| Field | Value |
|---|---|
| Run ID | island-community-remote-demo-025427 |
| Status | completed |
| Termination | Natural extinction at turn 23; simulator status `extinct` |
| Started | 2026-11-09T02:54:27.552673+00:00 |
| Finished | 2026-11-09T02:54:31.410216+00:00 |
| Baseline | not specified |
| Git revision | `f62c097d85300d0354259be0858eda1abb8e330a` ([private repository](https://github.com/kristinelu921/island-community/tree/f62c097d85300d0354259be0858eda1abb8e330a)) |
| Seed | 42 |
| Hardware | CPU on worker 0 of `kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`, zone `us-west4-a`, project `he-vision-group`; v5litepod-8, topology 2x4, Spot; TPU accelerators unused |
| Runtime | Python 3.10.18; PyTorch 2.2.2+cpu; pytest 9.2.0 |
| Requested run | 40 turns, CPU, vmap enabled, checkpoint every 10 turns |

Exact command, from the source snapshot's root on the authorized VM:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
/home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-verify-20261109/demo
```

## Configuration

Source: [island-community-remote-demo-025427-artifacts/demo/config.json](island-community-remote-demo-025427-artifacts/demo/config.json).
Complete effective configuration:

```json
{
  "schema_version": 1,
  "seed": 42,
  "initialization": {
    "population": 5,
    "specialization_rule": "round_robin_cover",
    "specialization_limit": 2,
    "survival_turns": 5,
    "endowment_rounding": "next_power_of_two",
    "specialization_quantity": 8,
    "currency": 30,
    "location_radius": 2.0,
    "capabilities": [
      "cooking",
      "weaving",
      "building",
      "medicine"
    ]
  },
  "items": {
    "water": {
      "name": "Water",
      "category": "base",
      "unit": "liter",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 1,
      "survival": {
        "water": 1
      },
      "disease_eligible": false
    },
    "grain": {
      "name": "Edible grain",
      "category": "base",
      "unit": "ration",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 2,
      "survival": {
        "food": 1
      },
      "disease_eligible": true
    },
    "timber": {
      "name": "Timber",
      "category": "base",
      "unit": "piece",
      "fractional": false,
      "transport_load": 2,
      "reward_value": 2,
      "survival": {},
      "disease_eligible": false
    },
    "stone": {
      "name": "Stone",
      "category": "base",
      "unit": "block",
      "fractional": false,
      "transport_load": 2,
      "reward_value": 1,
      "survival": {},
      "disease_eligible": false
    },
    "iron": {
      "name": "Iron",
      "category": "base",
      "unit": "ingot",
      "fractional": false,
      "transport_load": 2,
      "reward_value": 4,
      "survival": {},
      "disease_eligible": false
    },
    "clay": {
      "name": "Clay",
      "category": "base",
      "unit": "block",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 1,
      "survival": {},
      "disease_eligible": false
    },
    "fiber": {
      "name": "Fiber",
      "category": "base",
      "unit": "bundle",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 2,
      "survival": {},
      "disease_eligible": false
    },
    "herbs": {
      "name": "Herbs",
      "category": "base",
      "unit": "bundle",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 3,
      "survival": {},
      "disease_eligible": false
    },
    "salt": {
      "name": "Salt",
      "category": "base",
      "unit": "bag",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 2,
      "survival": {},
      "disease_eligible": false
    },
    "fuel": {
      "name": "Fuel",
      "category": "base",
      "unit": "unit",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 3,
      "survival": {},
      "disease_eligible": false
    },
    "meal": {
      "name": "Prepared meal",
      "category": "manufactured",
      "unit": "meal",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 2,
      "survival": {
        "food": 2
      },
      "disease_eligible": true
    },
    "cloth": {
      "name": "Cloth",
      "category": "manufactured",
      "unit": "sheet",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 4,
      "survival": {},
      "disease_eligible": false
    },
    "tool": {
      "name": "Tool",
      "category": "manufactured",
      "unit": "tool",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 6,
      "survival": {},
      "disease_eligible": false
    },
    "medicine": {
      "name": "Medicine",
      "category": "manufactured",
      "unit": "dose",
      "fractional": false,
      "transport_load": 1,
      "reward_value": 5,
      "survival": {},
      "disease_eligible": false
    }
  },
  "recipes": {
    "cook": {
      "inputs": {
        "grain": 2,
        "water": 1
      },
      "outputs": {
        "meal": 2
      },
      "capability": "cooking",
      "skill": "cooking",
      "base_time": 0.7
    },
    "weave": {
      "inputs": {
        "fiber": 2
      },
      "outputs": {
        "cloth": 1
      },
      "capability": "weaving",
      "skill": "weaving",
      "base_time": 1.5
    },
    "build": {
      "inputs": {
        "timber": 1,
        "iron": 1
      },
      "outputs": {
        "tool": 1
      },
      "capability": "building",
      "skill": "building",
      "base_time": 2.0
    },
    "compound": {
      "inputs": {
        "herbs": 2,
        "water": 1
      },
      "outputs": {
        "medicine": 1
      },
      "capability": "medicine",
      "skill": "medicine",
      "base_time": 1.0
    }
  },
  "allocation": {
    "turns_per_minute": 1,
    "allocations_per_minute": 5,
    "quantities": [
      1,
      2,
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024
    ],
    "quantity_weights": [
      1,
      2,
      5,
      8,
      5,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    "resource_weights": {
      "water": 4,
      "grain": 4,
      "timber": 1,
      "stone": 1,
      "iron": 1,
      "clay": 1,
      "fiber": 1,
      "herbs": 1,
      "salt": 1,
      "fuel": 1
    },
    "recipient_rule": "uniform_eligible"
  },
  "actions": {
    "time_budget": 1.0,
    "communication_time": 0.02,
    "agreement_time": 0.02,
    "minimum_work_time": 0.05,
    "work_carries_forward": true,
    "skill_gain": 0.05,
    "skill_speed_factor": 0.2,
    "teaching_time": 0.3,
    "teaching_gain": 0.1,
    "teaching_cost": 0,
    "transfer_time": 0.2,
    "transfer_units": 0.25,
    "transfer_cost": 0,
    "proposal_lifetime": 5,
    "contract_deadline": 15,
    "trade_quantities": [
      1,
      4
    ],
    "service_price": 2,
    "message_lifetime": 10,
    "purchase_price_per_value": 1.0
  },
  "survival": {
    "consumption": {
      "food": 1,
      "water": 1
    },
    "bundle_items": {
      "food": "grain",
      "water": "water"
    },
    "maximum_health": 100.0,
    "death_threshold": 0.0,
    "shortage_damage": 7.0,
    "unhappiness_damage": 1.0,
    "recovery": 1.0,
    "unhappiness_formula": "relative_reward_shortfall_ema",
    "unhappiness_smoothing": 0.2,
    "disease_damage": 3.0,
    "disease_duration": 4,
    "learning_strength_gain": 0.5
  },
  "aging": {
    "start_turn": 60,
    "interval": 10,
    "skill_retention": 0.95,
    "productivity_decay": 0.05,
    "minimum_productivity": 0.2,
    "health_damage": 1.0,
    "event_susceptibility": 0.2
  },
  "transport": {
    "topology": "complete",
    "direction": "bidirectional",
    "distance": "euclidean",
    "capacity": 32,
    "distance_per_turn": 2.0,
    "minimum_transit_turns": 1,
    "fee_per_distance": 1.0,
    "fee_denomination": "currency",
    "congestion": "seeded_priority",
    "ownership": "recipient_at_dispatch",
    "late_failure": "retain_recipient_owned_shipments",
    "cancellation": "pending_only"
  },
  "reproduction": {
    "enabled": true,
    "interval": 20,
    "inheritance_formula": "floor_factor_sqrt_available",
    "additional_inheritance_factor": 0.5,
    "policy_mutation_std": 0.01,
    "location": "near_parent",
    "location_jitter": 0.2,
    "channels": "connect_all",
    "specialization": "inherit_parent",
    "inherit_skills": false,
    "inherit_capabilities": true
  },
  "rewards": {
    "scale": 0.01,
    "currency_value": 0.0,
    "measurement": "after_deaths_and_births",
    "committed_inventory": "included",
    "in_transit": "recipient_owned",
    "work_in_progress": "excluded",
    "lifetime_score": "sum_resource_rewards",
    "terminal_reward": 0.0,
    "recipe_validation": "positive_values_no_amplification"
  },
  "events": {
    "probability_per_turn": 0.05,
    "type_weights": {
      "windfall": 1,
      "loss": 1,
      "channel_outage": 1,
      "income_reduction": 1,
      "disease": 1
    },
    "resource_quantity": 4,
    "duration": 3,
    "affected_items": [
      "water",
      "grain",
      "timber",
      "stone",
      "iron",
      "clay",
      "fiber",
      "herbs",
      "salt",
      "fuel"
    ],
    "windfall_items": [
      "water",
      "grain"
    ],
    "income_multiplier": 0.5,
    "incoming_contamination_probability": 0.02
  },
  "death": {
    "assets": "destroy",
    "extinction": "stop"
  },
  "learning": {
    "death_replay": false,
    "ancestral_replay": false
  },
  "policy": {
    "hidden_size": 64,
    "learning_rate": 0.0003,
    "discount": 0.99,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "gradient_clip": 1.0,
    "initial_capacity": 20,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-08
  }
}
```

## Changed Hyperparameters

No explicit comparison baseline was specified. All simulation parameters equal
the published `island_community/default.json`; there are no overrides to those
parameters. Execution uses 40 requested turns on CPU and the default vmap path.
The deterministic unit-test fixtures use separate synthetic configurations and
are not the configuration of this simulation.

## Results

| Dataset / split | Metric | Value | Turn | Direction | Source |
|---|---|---:|---:|---|---|
| Simulation | Living population | 0 | 23 | Diagnostic | `demo/metrics.jsonl` |
| Simulation | Peak population | 5 | 0–23 | Diagnostic | `demo/metrics.jsonl` |
| Simulation | Cumulative births | 0 | 23 | Diagnostic | `demo/snapshot.json` |
| Simulation | Cumulative deaths | 5 | 23 | Diagnostic | `demo/snapshot.json` |
| Simulation | Total owned resource value | 0.0 | 23 | Reward-related | `demo/snapshot.json` |
| Simulation | Allocated policy capacity | 20 | 23 | Diagnostic | `demo/snapshot.json` |
| Simulation | Parameters per policy | 42156 | 23 | Diagnostic | `demo/snapshot.json` |
| Simulation | Resident policy tensor bytes | 10122720 | 23 | Diagnostic | `demo/snapshot.json` |
| Correctness suite | Passed tests | 77 | not applicable | Higher | `tests-final.xml` |

Twenty policy slots require approximately 9.65 MiB
of resident tensors, including optimizer moments and recurrent state. This
excludes temporary activations, gradients, runtime overhead, and allocator
reserves. No GPU execution or peak GPU-memory measurement was performed.

## Artifacts

All paths below are relative to this report. Their existence was checked.

| Artifact | Path or URL | Verified |
|---|---|---|
| Complete run, source copies, and journals | [island-community-remote-demo-025427-artifacts/demo/](island-community-remote-demo-025427-artifacts/demo/) | yes |
| Full checkpoint | [island-community-remote-demo-025427-artifacts/demo/checkpoint.pt](island-community-remote-demo-025427-artifacts/demo/checkpoint.pt) | yes |
| Run manifest and source checksums | [island-community-remote-demo-025427-artifacts/demo/manifest.json](island-community-remote-demo-025427-artifacts/demo/manifest.json) | yes |
| Console output | [island-community-remote-demo-025427-artifacts/demo-console.log](island-community-remote-demo-025427-artifacts/demo-console.log) | yes |
| Final test output | [island-community-remote-demo-025427-artifacts/tests-final.log](island-community-remote-demo-025427-artifacts/tests-final.log) | yes |
| Final machine-readable test results | [island-community-remote-demo-025427-artifacts/tests-final.xml](island-community-remote-demo-025427-artifacts/tests-final.xml) | yes |
| Initial test failure evidence | [island-community-remote-demo-025427-artifacts/tests.log](island-community-remote-demo-025427-artifacts/tests.log) | yes |
| Local observatory | http://127.0.0.1:8768 | verified separately after startup |
| Tracking service | not used | not applicable |

Downloaded result-archive SHA-256, matching the remote archive:
`898c5d05fe4bde537cbd37a7ee25ef66d6abaceeace5be8cac41aa2c3aa63b96`.
All copied source files match their manifest hashes and the locally published
source before the later documentation-only validation note.

## Failure Details

The simulation completed without a program error. Its natural extinction is
an economic outcome, not a runtime failure. Last completed turn: 23.

The initial correctness suite had one failing test: vmap and loop updates
produced a maximum difference of 0.00012809038162231445 in `action_bias.bias`.
That redundant common logit offset has mathematically zero gradient; float32
roundoff was amplified by Adam. The offset was removed, and the regression was
extended to three updates without weakening tolerances. The final suite passed
all 77 tests, including independent updates, births, and exact resumed histories.

## Observations

- Five untrained founding islands reached extinction at turn 23, before the
  requested 40-turn bound. This run had no births.
- The separate integration fixture exercised funded births and exact checkpoint
  continuation, including observations, selected actions, and experience logs.
- Histories were recorded in ordinary files; policy storage remained resident.
- The complete checkpoint, configuration, metric log, and experience journal
  were retrieved and checksummed. No deceased or ancestral replay occurred.

## Interpretation

This validates an executable research baseline, not learned cooperation or a
stable population of 15–20. The defaults need economic and learning calibration.
The storage measurement supports feasibility of these small policies at that
population size; it does not establish the memory requirements of language
models or peak accelerator memory.

## Updates

- Before execution, filtered discovery reported this exact kristine VM AVAILABLE.
  A stricter check found a holderless zero-byte `/tmp/libtpu_lockfile`, owned by
  `tiangexiang`, last modified 2026-11-08 02:35:26 UTC. The user explicitly
  authorized removal. Fresh checks found no VFIO or lockfile holders and no
  Python `train.py` process; only this lock file was removed. The standard
  occupancy check then reported AVAILABLE. No process was terminated.
- The first cleanup check matched its own shell text and aborted without changes.
  The corrected check restricted trainer matching to Python processes before
  performing the authorized removal.
- The user authorized CPU-only tests and a short simulation. No new cloud
  resources were provisioned, and no non-kristine VM was visited.


## Replay preparation — 2026-11-08T22:24:39.485998-05:00

- Reconstructed all 24 original world frames, turns 0–23, from the saved choices
  using the original world/config source hashes. Verified every observation,
  reward, terminal flag, next observation, and the final snapshot. Source
  recording files remain unchanged.
- Executed on CPU on the same explicitly authorized idle kristine VM. Exact
  command from `/tmp/island-community-replay-20261109`:
  `/home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community rebuild-replay /tmp/island-community-verify-20261109/demo --output /tmp/island-community-replay-20261109/replay`.
  This reused the original seed 42 configuration and recorded actions.
- New derived artifacts:
  [replay directory](island-community-replay-artifacts/replay/),
  [reconstruction provenance](island-community-replay-artifacts/replay/manifest.json),
  [remote integration checks](island-community-replay-artifacts/integration.xml),
  [browser preview](island-community-replay-artifacts/replay-preview.png).
- Replay frame file: 654865 bytes. Download archive SHA-256 matched remotely and
  locally: `a71217a1dafb65ea8c3739541a922c7397ed221a4bcf73a8af5917ec56086478`.
- Verification: 60 non-neural tests passed; two remote checkpoint/resume
  integration tests passed; headless Chromium checks passed for seeking,
  playback/pause, retained paused selection, dead-island inspection, delayed
  request ordering, and a 390-pixel viewport. Standalone wheel builds and
  includes the replay module and browser assets.
- Published source: `b42175d3c02747d61959b30588457f74000927e7` in the private
  [Island Community repository](https://github.com/kristinelu921/island-community).
- Replay viewer is served at http://127.0.0.1:8769. Viewing uses recorded frames
  without neural execution or simulation advancement.
