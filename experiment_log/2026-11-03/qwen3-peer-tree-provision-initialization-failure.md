# Qwen3 pilot provisioning: maintenance health gate

## Run Summary

- Status: failed (runtime initialization only; created capacity retained).
- Run ID: qwen3-1p7b-peer-tree-pilot-20261103, setup attempt 1.
- Project/zone: he-vision-group/us-west4-a.
- TPU: kmh-tpuvm-v5e-8-spot-kristine-03-22-35-29-776996979-0-us-west4-a.
- Type/topology: v5litepod-8, one host, eight chips, Spot.
- Seed: 20261103 (no experiment executed).
- First step: none. Checkpoint: none. W&B: none.
- Exact command: `ELF_NETRC_PATH=/tmp/smollm2-sft-wandb.netrc ELF_TPU_INITIALIZER=/kmh-nfs-ssd-us-mount/code/kristine/conlang/tpu_scripts/initialize_elf_runtime.sh ZONES_OVERRIDE='us-west4-a us-central1-a' RETRY_DELAY_SECONDS=60 bash /tmp/qwen-peer-query-tpus-20261103.sh v5e 8 1`.

## Configuration

User authorized one new Spot v5e-8 for up to 24 hours. The query helper's
notification topic was changed only in a task-specific copy to kluping.
Source, datasets, training environment, and output directory were not staged.

## Results and Failure Details

Google completed resource creation. The initializer's direct describe returned
state READY and health UNHEALTHY_MAINTENANCE. The initializer failed closed
before runtime installation or accelerator acquisition; launcher exit 1.
The sole created resource counts toward the one-resource capacity limit.

Operation: projects/he-vision-group/locations/us-west4-a/operations/operation-1793745331451-65f66e14c763a-a36eb7c5-11d7f3c8.

## Recovery

Read-only health checks on this exact resource. No second provisioning campaign,
reboot, termination, claim, or deletion. Initialization may retry on the same
resource only after direct health and filtered inventory gates pass.

## Recovery outcome — 2026-11-03T23:35:13.655598+00:00

The same resource became HEALTHY without a reboot. A fresh AVAILABLE audit
passed; runtime initialization succeeded; the float32 pilot subsequently completed
32 training trees and before/after evaluation. See `qwen3-peer-tree-pilot-e225874.md`.
The initial maintenance failure remains a distinct setup record.
