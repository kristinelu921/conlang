# Authorized v6e resource disappeared before cleanup

At 2026-11-11 01:43 UTC the user approved removal of our empty holderless
lockfile and a corrected-topology retry. The mandatory fresh cloud describe:

```
gcloud compute tpus tpu-vm describe kmh-tpuvm-v6e-16-kaiminghe-070b9f --project=he-vision-group --zone=us-east5-b --format='json(name,state,health,acceleratorType,acceleratorConfig,schedulingConfig,networkEndpoints)'
```

returned NOT_FOUND, exit status 1. This establishes resource disappearance;
the cause (Spot preemption versus deletion) has not been established.
No SSH connection to the stale worker addresses was attempted, no lockfile was
removed, and no retry was launched. No Qwen inference step or world turn had
completed on this resource; no model checkpoint exists there. The initial test
log and exact dependency manifest had already been copied locally and remain
preserved as native-check1.log and dependencies-native.txt.

Recovery initiated: the filtered read-only kristine inventory via
`ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`.
No additional foreign TPU inspection or provisioning is authorized by this
cleanup response. Any replacement will require a fresh source snapshot,
task-local environment, and all-worker occupancy check.
