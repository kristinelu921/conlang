# nanoGPT Muon–Hessian alignment retry1

Status: planned. Run ID nanogpt-muon-hessian-alignment-20261123-retry1. Source 43c2918; PR https://github.com/kristinelu921/modular_norm/pull/5. This is the same scientific experiment as [the original attempt](nanogpt-muon-hessian-alignment.md), with the JSON boolean serialization repair only. Original output preserved; no source checkpoint or training state modified. All scientific parameters, training seed42, probe seed20261123, 32 cases, 3 damping settings and precision remain unchanged.

Primary config configs/curvature-alignment.json, unchanged dependency configs/shakespeare-muon.json. Full defaults from newton-schulz-resolved-config.json and complete effective run configuration will be saved in output/config.json. Same verified dataset and source checkpoints, same TPU kmh-tpuvm-v4-8-spot-kristine-21-13-47-32-N-us-central2-b, zone us-central2-b, four-device/single-worker v4-8. Most recent monitor READY/HEALTHY; original process exited1 and monitor finished. CPU/TPU regression validation pending, final occupancy before launch required.

Immutable node-local source: /home/kristinelu/elf-runs/source/nanogpt-muon-hessian-alignment-20261123-retry1-20261123T035452.876493522Z

Command from this source:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' /home/kristinelu/.venvs/modular_norm/bin/python -u curvature_alignment.py --config configs/curvature-alignment.json --output /home/kristinelu/elf-runs/output/nanogpt-muon-hessian-alignment-20261123-retry1
```

Artifacts: output alignment.jsonl, config.json, batch_*.npz, step_*/matrix.npz, COMPLETED.json; sibling .console.log, .launch/terminal.status. Metrics and plots not yet recorded. Same block-Hessian, stochastic-batch, one-seed and precision limitations as the original report. Review ledger: correctness a4eba51 (restricted validity fixed3837589); cleanup3837589 (no findings); coordinator verifies runtime fixes with targeted and full remote tests. Exactly two critic passes retained.

## Validation and first progress: 2026-11-22 22:56 EST

Status: running. Full 22-test CPU suite passed in6.909s and all5alignment tests passed on TPU in6.810s, including JSON serialization. Final occupancy AVAILABLE. Exact live evaluator PID655280, supervisor655279, saved-state checkpoint500 loss1.5019348859786987nats/token. First4matrix cases completed in3.39/12.64/10.30/8.64s, all12damped block solves accepted. Original source checkpoint hashes remain input verification gates. Active foreground recurring monitor checks API health, exact PID, metric-count progress and terminal sentinel every45seconds.

## Monitoring transport

The existing aes128-ctr/hmac-sha2-256 SSH transport intermittently fails with Bad packet length/message authentication code incorrect. API health remains READY/HEALTHY. These are monitoring transport failures, not evidence of evaluation failure or preemption; evaluator is detached under the node-local supervisor, and no duplicate or recovery workload is launched. Trying an alternate authenticated SSH cipher for read-only progress.

## Late progress and transport repair

Internal-IP SSH succeeded without cipher errors; use --internal-ip for remaining inspection and artifact transfer. At case25, all75block solves accepted, including all8matrices at checkpoints500/1000/1500. Case times remain approximately7–13seconds after initial compilation. The single foreground monitor remains active; one duplicate local observer was identity-verified and retired without touching the evaluator.

## Completion: 2026-11-22 23:00 EST

Status: completed. All32cases finished in276.9912195389916seconds (4.62minutes). Canonical single-worker supervisor terminal status0, COMPLETED.json present; all96block solves accepted with maximum independently recomputed residual9.761161587795903e-05. All64raw/momentum restricted Hessians passed symmetry (worst relative error6.542045543198696e-08); all192restricted solves passed their dense validity checks. All32estimated minimum eigenvalues were negative; damping was essential. Saved-state source checkpoint SHA-256 values unchanged. Independently recomputed saved Muon/Newton cosines and restricted residuals. Local archive and43individual artifact hashes match.

### Mean cosine across32cases

| Damping ratio | Muon NS + momentum | Exact momentum polar | Exact raw polar | Raw gradient | Nesterov |
|---|---:|---:|---:|---:|---:|
| 0.01 | 0.315474123734 | 0.320280728305 | 0.576301297212 | 0.782758947417 | 0.474027999225 |
| 0.1 | 0.361358102590 | 0.366318827024 | 0.657578847209 | 0.918091152036 | 0.558911434816 |
| 1.0 | 0.365267653159 | 0.369439765520 | 0.661165235355 | 0.989443676591 | 0.599800974747 |

At ratio.01, mean Newton energy outside the raw-gradient SVD span is0.3435488431914494; outside the momentum SVD span0.7649131093938906. Restricted coefficient cosines with uniform weights are0.75649712521181(raw) and0.6962958590849788(momentum). Median raw-basis Hessian off-diagonal Frobenius fraction0.3257685544832652.

Muon normalized directional curvature is lower than the raw gradient in32/32cases. Median per-case Muon/gradient curvature ratio0.11217147959704159 (range0.027146738239194813–0.21678690391900632). This is at equal Euclidean step length; actual loss improvement also depends on first-order gain and step size. Means across matrices are descriptive, not independent-seed confidence estimates.

Interpretation: these probes offer little support for close damped-Newton imitation by the saved-state Muon direction. They do show favorable directional curvature relative to the raw gradient. Exact polar substitution barely changes alignment; removing momentum changes it substantially, but a single-batch reference does not assess momentum's denoising benefit. This is not evidence that curvature explains or fails to explain Muon-versus-Adam training gains; no Adam baseline or intervention was run.

### Verified artifacts

Regional archive: gs://kmh-gcp-us-central2/kristine/modular_norm/nanogpt-muon-hessian-alignment-20261123-retry1/results.tar.gz, 60063785bytes, SHA-256 c45d6c3da5fc0ab153b979096f6d5d3d25b104e4592169f4137076d636a85dbe. Uploaded directly from node-local storage with gsutil. Local full archive/arrays/report at /kmh-nfs-ssd-us-mount/code/kristine/conlang/experiment_log/2026-11-22/muon-hessian-alignment/. Static report and compact metrics also saved under results/alignment/ in this repository. Localhost-only report: http://localhost:34123/ . All complete model defaults, sampled token arrays, source checkpoint hashes, raw gradients, Nesterov matrices, actual updates, block Newton vectors, restricted Hessians and coefficients are retained. No further evaluation or training process remains; run-specific monitor retired after verified terminal success. Results server remains running.
