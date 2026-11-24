# Adam curvature comparison

Status: planned. Run identity: nanogpt-shadow-adam-curvature-20261123. Source commit f6fab25; draft PR https://github.com/kristinelu921/modular_norm/pull/5. Seed42, probe seed20261123. Baseline: nanogpt-muon-hessian-alignment-20261123-retry1.

Hardware: kmh-tpuvm-v4-8-spot-kristine-21-13-47-32-N-us-central2-b, us-central2-b, one worker/four v4 devices. Complete filtered discovery returned AVAILABLE; fresh occupancy confirmed AVAILABLE. Snapshot: /home/kristinelu/elf-runs/source/nanogpt-shadow-adam-curvature-20261123-20261123T061036.585227338Z.

Configuration: configs/adam-curvature.json and its unchanged configs/shakespeare-muon.json dependency. Full model defaults are recorded by curvature output/config.json. Change: add shadow Adam beta1=.9, beta2=.95, epsilon=1e-10. Original Muon training replayed unchanged from seed42, using same sampler seed43; shadow steps never affect weights. Exact checkpoint parameters, optimizer state and sampler state are mandatory gates at500,1000,1500,2000. Diagnostics use highest float32 matmul precision, original historical replay uses original default precision. Curvature, Lanczos, damping and solver settings unchanged from baseline.

Commands from snapshot (environment OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES=''):
```
/home/kristinelu/.venvs/modular_norm/bin/python -u shadow_adam.py --config configs/adam-curvature.json --output /home/kristinelu/elf-runs/output/nanogpt-shadow-adam-replay-20261123
/home/kristinelu/.venvs/modular_norm/bin/python -u curvature_alignment.py --config configs/adam-curvature.json --output /home/kristinelu/elf-runs/output/nanogpt-shadow-adam-curvature-20261123
```

Results: pending. Artifacts: remote output paths above (not yet created), local report folder experiment_log/2026-11-23/adam-curvature. Exact command/config, source checkpoint hashes, gradient/update arrays and replay evidence will be retained. Comparison measures counterfactual shadow Adam on Muon-trained weights, not a separately trained Adam model. One seed and one batch per checkpoint, matrix-block Hessians rather than full-network curvature.

Review: correctness b155347, provenance gap fixed f6fab25 with mismatch tests. Cleanup f6fab25, no confirmed findings. Final remote tests pending.

## 2026-11-23 06:12 UTC

Status: running. Full remote suite25/25 CPU and focused TPU3/3 passed. Replay completed all2000steps; parameter and optimizer arrays were bitwise equal at all four original checkpoints (maximum absolute error0.0); sampler states also exactly equal. Shadow-history provenance validated. Supervisor PID692152, curvature evaluator PID693248 on single worker, READY/HEALTHY. Monitor /tmp/adam-monitor.py checks API health, exact process and terminal status every45seconds and remains active until terminal evidence.

## 2026-11-23 06:14 UTC — completed

Status: completed. Single-worker supervisor exited0; COMPLETED.json records32matrices and68.40749097801745seconds for the curvature phase. All96damped block solves accepted. Exact run processes absent at final monitor, TPU READY/HEALTHY. Monitor retired after terminal success. Remote tests25CPU+3TPU passed. Independent saved-array verification confirms Adam directions against NumPy (maximum absolute discrepancy2.0262560207484626e-07), all Adam/Newton cosines, all unchanged baseline metrics exactly, and replay equivalence at all four checkpoints.

| Direction | Median per-case curvature / raw gradient | Mean cosine to damped Newton, ratio .01 |
|---|---:|---:|
| Muon | 0.11217147959704159 | 0.315474123734408 |
| Shadow Adam | 0.41124021279137657 | 0.3260736217203486 |
| Raw gradient | 1.0 | 0.7827589474165799 |
| Momentum | 0.607462716696747 | 0.4740279992251505 |

Adam curvature is lower than raw gradient in32/32cases; Muon lower than Adam in32/32cases. Median Muon/Adam curvature ratio0.27003076005907584. Mean downhill slope per unit Euclidean step is0.038482678920424326(Muon),0.0422446716845076(Adam),0.10739538419742048(raw gradient). Both slope and curvature matter; lower curvature does not alone establish better actual progress. Adam is slightly closer to the damped Newton direction on average at all three tested damping ratios. All observations are conditional on this Muon trajectory and selected Adam betas; they do not isolate the effect of momentum differences or establish a causal explanation of independently trained optimizer performance.

Verified regional archive: gs://kmh-gcp-us-central2/kristine/modular_norm/nanogpt-shadow-adam-curvature-20261123/results.tar.gz. SHA2569522be462c3b4dba2bdcedfccc119925d276a0add163869ac8d632f406756aa4. Local archive and every MANIFEST.sha256 entry verified. Artifacts include complete effective configuration, raw arrays, shadow moment history, exact replay evidence, logs, and all per-case metrics. summary.json, comparison.csv, verification.json, adam-comparison.png and index.html derived locally from saved arrays only; no development-machine model execution. Report served at http://localhost:41831/ (127.0.0.1 only).

A transient SSH transport error while archiving was automatically retried successfully; it did not interrupt the experiment. No failed experiment attempts.


## Corresponding losses (saved-result inspection)
No new experiment. Pre-update next-batch loss shared by both direction probes: {"500": 1.5019348859786987, "1000": 1.394282579421997, "1500": 1.3169941902160645, "2000": 1.3210039138793945}. Units: nats/token. These are neither validation nor post-update losses. Added checkpoint-loss.png and checkpoint-loss.json to the local report.
