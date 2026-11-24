# Qwen native TPU validation — v5e-4, 2026-11-11

Explicit user acquisition authorization: AUTHORIZE IDLE NON-KRISTINE TPU ACQUISITION kmh-tpuvm-v5e-4-spot-myang-ar. Resource was idle on all workers in the authorized02:26 UTC read-only sweep. Exact resource only, project he-vision-group, zone us-west4-a. Fresh cloud and occupancy checks are required before staging/launch.

Isolated node-local root: /tmp/island-community-qwen-v5e4-20261111-024900. Python3.12, Torch2.15.0+cpu, TorchTPU0.2.0, Transformers5.22.2, NumPy2.5.4, Accelerate1.15.0, pytest8.3.5. Four CPU threads. Source changes retain float32 weights, static attention cache and CPU-side generation tokens, using the public single-device launch helper instead of the invalid TPU_VISIBLE_DEVICES mask.

CPU frozen baseline remains separate, active and monitored on the v4 VM CPU. No scientific TPU run has started; tiny constrained decoding/checkpoint/adapter tests must pass first. No other active job may be touched. Exact commands and each failed attempt will be retained in this directory.

Fresh preflight02:49:11 UTC: READY/HEALTHY/Spot v5litepod-4, topology2x2, one worker IDLE. The VM has112 logical CPUs,188GiB RAM,20GiB free /tmp disk. No existing files removed. Exact preflight metadata/occupancy evidence is in acquisition-preflight.json. Source test archive SHA-2566943336421597985c3d3a9ac3d5da53ac26e0a81a21436dd83581aa45234d009.

## Source staging blocked by automatic approval review

Dependency setup completed successfully. Two attempts to copy the scoped application archive were rejected before execution by automatic approval review. The stated reason is that authorization to use the non-kristine TPU does not explicitly authorize transfer of private repository source to it. No application source was copied. The reviewed payload is /tmp/island-community-qwen-v5e4-check1.tar.gz; its complete file manifest, archive hash, and sizes are in source-payload-manifest.json. It contains only the Island Community application, tests, documentation and synthetic scenarios. No parent ELF code, datasets, credentials, or recorded experiments are included. Explicit authorization for this source transfer is now required.

## Public runtime verification, 02:56 UTC

The public `torch.tpu.distributed.set_tpu_launch_env(nnodes=1, nproc_per_node=1)` helper completed with exit status 0. It configured `TORCH_TPU_TOPOLOGY=1,1,1`; `TPU_VISIBLE_DEVICES` and `TORCH_TPU_DEVICES_PER_HOST` remained unset. Exact package versions and output are saved in `runtime-launch-environment.txt`. This was package and launch-environment inspection only: no tensors, model, inference, or training were created, so TPU execution remains unverified. Application transfer approval is still pending.

The separate CPU baseline completed turn 3 at 02:56:04 UTC with all 10 islands alive. Its recurring monitor recovered from one transient rsync failure on the next check without changing the workload. The first 30 requested actions matched their recorded submitted and executed actions; turn 3's ten prompts each contained exactly that island's two previous actions, execution results, and rewards. Evidence is retained in the baseline report's `turn1-audit.json`, `turn2-audit.json`, and `turn3-audit.json`. The frozen model's final weight checksum remains pending completion of its 20-turn run.

## Transfer authorized and native check 1 started, 03:00 UTC

The user explicitly approved the application transfer with the condition that the zone be correct. Fresh metadata and occupancy at 02:59:52 UTC confirmed the exact authorized resource in `us-west4-a`, READY/HEALTHY/Spot, with its sole worker IDLE. Evidence is in `transfer-preflight-0259.json`. This standalone synthetic simulation has no external training dataset; tokenizer/model downloads use the VM's node-local cache directly. No dataset or checkpoint was copied across regions.

The approved 127,500-byte archive was copied to the task root, and its SHA-256 matched the manifest before extraction. `native-check1.sh` launched the frozen tiny-Qwen correctness test using seed 42, model-initialization seed 128, float32, four CPU threads, 3 reasoning tokens, a 1,024-token context, and native device `tpu`. The exact pytest selector and environment are retained in that script; live output is `native-check1.log` on the VM, with a final `native-check1.exit` sentinel. This tests constrained decoding, immutable shared weights, and checkpoint restoration before any full pretrained TPU experiment.
