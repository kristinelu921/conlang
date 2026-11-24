# Native TorchTPU v4 compatibility failure

Attempt: native-check1, 2026-11-11 UTC. Hardware: kmh-tpuvm-v4-8-spot-kristine-09-14-38-39-N-us-central2-b, us-central2-b, v4-8 2x2x1, READY/HEALTHY/Spot, sole worker AVAILABLE immediately before launch. Exact setup and test commands: setup.sh and native-check1.sh. Node-local source SHA-256: 6943336421597985c3d3a9ac3d5da53ac26e0a81a21436dd83581aa45234d009.

Dependency setup completed with exact versions preserved in dependencies.txt. Test exited 1: one failed, three deselected, 12.73 seconds. Failure stage: device discovery in torch.tpu.distributed.set_tpu_launch_env(nnodes=1,nproc_per_node=1), before model construction/inference. Error: ValueError: set_tpu_launch_env(): expected TPU devices, got none.

The VM exposes /dev/accel0 through /dev/accel3, with only the vfio control node in /dev/vfio. TorchTPU hardware discovery returned no topology. This agrees with official support listing v5e/v5p/v6e/v7x, excluding v4 from active support: https://torchtpu.readthedocs.io/en/latest/faq/. Do not bypass hardware detection or repeat the same unsupported launch. No lockfile remained, no world step completed, and no checkpoint exists. Full failure and dependency logs copied locally and retained remotely. Recovery: select supported idle hardware from a fresh completed kristine-only audit.
