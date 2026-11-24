# v4 accel-device holder omitted from older discovery

Observed during the authorized sweep on 2026-11-11 at02:25:58 UTC. Resource kmh-tpuvm-v4-8-spot-kristine-09-14-38-39-N-us-central2-b, project he-vision-group, us-central2-b, v4-8 topology2x2x1, READY/HEALTHY/Spot. The preceding standard filtered inventory returned AVAILABLE. The expanded read-only probe found PID899538, owner kristinelu, holding /dev/accel0 through3, with no reported VFIO or libtpu-lock holder. No process was changed or signaled. This is a discovery failure, not a failure of the running Qwen CPU job or the separate accelerator job.

Confirmed cause: both check_tpus.sh and check_tpu_occupancy.sh examined /dev/vfio/* and /tmp/libtpu_lockfile but omitted /dev/accel devices. A focused test executing the actual three embedded remote shell probes with fake devices/commands reproduced false availability in discovery, inspection, and claim phases. Before repair: three occupied-device subtests failed.

Repair: include accel devices in discovery and runtime-device checks; detect their holders in both occupancy phases and protect them from automatic termination. Automatic reclamation authorization is not expanded beyond VFIO. No remote files or running jobs were modified.

Verification command: python -m pytest tests/test_query_monitor_tpu_skill.py -q. Result:26 passed,13 subtests passed,6.47seconds. Shell syntax and git diff whitespace checks passed. The tests exercise held and idle fake v4 devices without a libtpu lock, and explicitly fail on any attempted process mutation. Corrected read-only live discovery is recorded separately in corrected-own-pool-audit.txt.

No scientific checkpoint was affected. The separate four-thread Qwen CPU run retains its turn1 checkpoint and continues to be mirrored/monitored under the Qwen run log. The earlier native-v4 attempt failed hardware discovery before model creation; no successful TPU initialization is claimed.
