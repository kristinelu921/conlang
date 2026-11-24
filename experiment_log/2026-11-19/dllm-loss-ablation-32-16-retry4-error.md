# dLLM loss ablation retry 4 recovery error

- Status: failed before TPU initialization or optimizer updates
- Worker: `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, worker 1, `us-west4-a`
- Recovery source: checksum-verified `direct-low-lr` update-100 checkpoint
- Error: the compatibility path for `validation_finecode.manifest.json` was missing, so validation provenance verification raised `FileNotFoundError` before model loading.
- Effect: no checkpoint, optimizer state, training metric, or TPU device state changed.
- Correction: add the missing manifest symlink and relaunch the same recovery identity from the same verified update-100 checkpoint.
