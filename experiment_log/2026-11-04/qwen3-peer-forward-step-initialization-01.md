# Qwen3 forward-step TPU initialization attempt 01

## Run Summary

- Status: failed (runtime initialization preflight; training never started).
- Recorded at: 2026-11-04T01:34:00.199170+00:00
- Run: forward-step-20261104-attempt-01, source 42f532d1b011758027887bf5dd7fb47fe5bc2ccc, PR https://github.com/kristinelu921/conlang/pull/4.
- Exact TPU: `kmh-tpuvm-v5e-8-spot-kristine-04-01-30-32-976463421-0-us-west4-a`, us-west4-a, v5litepod-8, 2x4 topology, one VM, Spot.
- Cloud creation: 2026-11-04T01:30:37.098297893Z; unchanged hard allocation deadline: 2026-11-04T03:30:37.098297893Z.
- Stage: runtime initialization preflight, before SSH setup or package installation.
- Seed: 20261103 (not used; no experiment started).

## Configuration and Command

Selected experiment config: `experiments/peer_tree_rl/forward_step.json`; full config and baseline are in `qwen3-peer-forward-step.md`.

```sh
ELF_NETRC_PATH=/tmp/smollm2-sft-wandb.netrc ELF_TPU_INITIALIZER=/kmh-nfs-ssd-us-mount/code/kristine/conlang/tpu_scripts/initialize_elf_runtime.sh ELF_RUNTIME_PROFILE=jax-0.6.2 ZONES_OVERRIDE=us-west4-a RETRY_DELAY_SECONDS=60 bash /tmp/qwen-peer-query-tpus-20261103.sh v5e 8 1
```

## Results and Failure Details

The creation request succeeded. Initialization returned exit 4 because the exact
new TPU was READY with health UNHEALTHY_MAINTENANCE. The enclosing single-resource
provisioner exited 1 and reported INITIALIZATION_FAILED; no second TPU was
created. Detailed output: `/tmp/qwen-peer-forward-step-provision.log`.

Last metric step: none. Step 1: not started. Checkpoint: none. W&B run: none.
Source/data staging, package installation and model loading: not started.
Installed package versions: not recorded because initialization did not run.

## Interpretation and Recovery

Direct API health blocked initialization; this is not a model or package failure.
Recheck the same resource without rebooting or creating a replacement. If health
becomes HEALTHY, run the fresh filtered availability audit and initialize only
this selected idle resource. Preserve this failed preflight record separately.

- 2026-11-04T01:35:10.309407+00:00: exact cloud API reports the same creation timestamp and READY/HEALTHY after natural maintenance recovery. No reboot, replacement, process action, or second create. Starting fresh read-only final-table availability audit before retrying runtime initialization.
