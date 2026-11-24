# Island Community rerun with item effects

Started November 9, 2026, 12:06 a.m. America/New_York.

Requested retry: one new run of at most 40 turns, using fresh independent
policies and the version 2 item-utility defaults. No learning-rate sweep is
performed. This is a short demonstration, not a calibrated learning experiment.

## Reproducible configuration

- Published source commit: `95081176bce175d190742d5cdecc5cfe3f20ea32`.
- Source archive SHA-256: `e717fbd34e8dbd0e09c4ef0c9fb92ec3c0a87b4c243498c618c43711e2158f47`.
- Complete configuration: `selected-config.json`; seed 42, five founders.
- Changed relative to the original version 1 demo: item-use effects, capped
  temporary stats, meal nutrition carryover, and the policy observation/action
  dimensions needed for those features. Same seed and 40-turn limit.
- Base LR 0.0003, Adam beta1 0.9/beta2 0.999/epsilon 1e-8, discount 0.99,
  entropy coefficient 0.01, value coefficient 0.5, gradient norm clip 1.0.
- Each island's LR multiplier is 1 + 0.5 * its current unhappiness; unhappiness
  uses a 0.2 EMA of positive relative resource-reward shortfalls. Terminal
  transitions use multiplier 1. No LR warmup, decay, or empirical tuning.
- Resource reward is 0.01 times currently owned configured item value,
  including recipient-owned transit. Currency has zero reward value.
- No health/item-use bonus, death replay, or cross-island gradient averaging.

## Authorized compute

Exact VM: `kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a`,
project `he-vision-group`, zone `us-west4-a`, worker 0.
Fresh state: READY, HEALTHY, Spot v5litepod-8, topology 2x4. Occupancy helper:
`AVAILABLE worker=t1v-n-4b587646-w-0`.
Only CPU execution, at most four threads, under the user's existing explicit
tests/short-simulation authorization and current rerun request.
No provisioning, process termination, or foreign-VM inspection.

Exact command from the isolated source directory `/tmp/island-community-items-20261109-000613/source`:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/kristinelu/elf-runs/runtime/venv/bin/python -m island_community run --config island_community/default.json --turns 40 --device cpu --checkpoint-every 10 --output /tmp/island-community-items-20261109-000613/recording
```

Status: staging. Results and artifact checksums will be appended when complete.

## Completed result

The run completed without a runtime error and stopped naturally on extinction at turn 23.
Five founders, zero births, five deaths; 24 complete replay frames, turns 0–23.
There were 94 individual learning updates: 23, 23, 16, 16, and 16 per founding island.
Actions: 78 trade proposals, 11 private requests, 2 public requests, 2 rejections, 1 wait.
No accepted trade, item use, or completed recipe occurred in this short run.
Observed effective LR: 0.0003–0.0003044918092223636. No LR calibration was performed.
Mean pre-clipping gradient norms ranged from 0.87185 to 8.29869; all logged learning metrics were finite.
Short lives, the large trade-action menu, and inventory-based rewards are plausible barriers to useful behavior; this single run does not isolate their causal contributions.
The original and updated demos both ended at turn 23; this does not establish equivalence or show the item changes are ineffective.

Results archive SHA-256 matched remotely and locally: 85f366662c97fa81b28e969e6977ab5a255113c9adeeb409f4f9c6b3865494ba.
Recorded config equals the selected config; frames are contiguous and the final frame equals the final snapshot.
Artifacts: recording/ (source fingerprints, config, manifest, events, experiences, metrics, frames, checkpoint), run.log, analysis.json.
Viewer: http://127.0.0.1:36951. Previous recordings and viewers remain unchanged.
Each policy has 43,501 parameters; capacity 20 reserves 10,445,520 tensor bytes, including weights, Adam state, and recurrent memory but excluding temporary tensors/runtime overhead.

Browser verification passed for all 24 frames, including playback, seeking, the version 2 item guide, saved nutrition and active-stat inspection, and the phone viewport. Screenshot: replay.png.
Architecture inspected: each island already uses its own 64-unit GRU-style recurrent cell. Hidden state persists across turns; actor-critic gradients are truncated to one step. No architecture or LR changes were made for this rerun.
