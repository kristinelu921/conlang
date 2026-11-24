# Independent Adam-only nanoGPT training

Status: planned. Run ID nanogpt-adam-only-shakespeare-20261123. Source commit3af08ee, PR https://github.com/kristinelu921/modular_norm/pull/5. Baseline nanogpt-muon-shakespeare-v4-8-20261123.

Hardware: kmh-tpuvm-v4-8-spot-kristine-21-13-47-32-N-us-central2-b, us-central2-b, single worker/four v4 devices, Spot. Filtered sweep AVAILABLE; initialization probe UNKNOWN resolved by successful remote runtime tests. Occupancy AVAILABLE. Full27CPU and2TPU tests passed. Correctness and cleanup reviews of3af08ee both returned no confirmed findings.

Source snapshot: /home/kristinelu/elf-runs/source/nanogpt-adam-only-shakespeare-20261123-20261123T062951.707866513Z. Config configs/shakespeare-adam.json. All resolved defaults saved in output/config.json at initialization. Model2layers, width128,4heads, vocab65, float32, seed42. Batch32×128tokens, sampler seed43. Dataset checksum86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed, train fraction.9. Fixed validation32768tokens, probe seed1729, every100steps.2000training steps, checkpoints every500.

Change versus baseline: all parameters use upstream Adam(lr=.001,beta1=.9,beta2=.95,epsilon=1e-10), no warmup/warmdown. Hidden matrices previously used Muon(lr=.01,momentum=.95); other parameters already used Adam with matching hyperparameters. Initialization, sampled batches, training length, dataset and validation procedure unchanged. Adam starts from initialization with fresh moments. No Muon optimizer is constructed in this run; no shadow buffers are used. This is one untuned Adam learning rate.

Exact command from snapshot:
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' /home/kristinelu/.venvs/modular_norm/bin/python -u experiment.py --config configs/shakespeare-adam.json --data /home/kristinelu/elf-data/tinyshakespeare/input.txt --output /home/kristinelu/elf-runs/output/nanogpt-adam-only-shakespeare-20261123
```

Results pending. Monitor checks API health, current process, recent metrics and checkpoint markers every45seconds. Step100throughput/finite-validation gate must pass before PROCEED. Existing unrelated jobs remain untouched. Artifacts planned: training.jsonl,validation.jsonl,amplification.jsonl,config.json, four hashed checkpoints, terminal status and completion sentinel. No claim yet about loss or curvature.


## Training completion and independent-checkpoint curvature

Training status: completed. Supervisor terminal0, COMPLETED at2000steps in67.30291502599721seconds, TPU READY/HEALTHY and training process absent. Step100gate passed: validation2.4277897477149963, throughput26866.054295401435examples/s excluding diagnostics. Final validation1.6420904248952866 versus Muon1.5869112759828568; difference0.05517914891242981nats/token. All four Adam checkpoint hashes verified and each optimizer state contains only step/m/v (no Muon). Independent remote comparisons confirm exact sampler equality and matching model/validation configuration at all four checkpoints. Regional training archive SHA2564a8455b258e1e3442e5fceb2fa543303bb672a4f6871748999146af71f6f527c; full local manifest verified.

Additional curvature run planned: nanogpt-adam-only-curvature-20261123, source653214f, configs/adam-only-curvature.json, dependency configs/shakespeare-adam.json. Same32hidden-matrix checkpoint cases, damping ratios.01/.1/1, probe seed20261123, Lanczos32, CG tolerance1e-4/max200, residual acceptance2e-4, highest float32 diagnostic precision. Now restores Adam's actual optimizer state from its own training trajectory; Adam first moment is labeled first_moment, not Nesterov. No Muon optimizer is constructed. Raw and first-moment polar factors are mathematical controls only. Full effective config saved with results. Coordinator reviewed extension directly;28CPU and8TPU tests passed, including pure-Adam routing, correct result labels and TPU Hessian finite differences. Earlier two review passes retained; no third critic.

Curvature command from immutable node-local snapshot (recorded in curvature-source.txt):
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' /home/kristinelu/.venvs/modular_norm/bin/python -u curvature_alignment.py --config configs/adam-only-curvature.json --output /home/kristinelu/elf-runs/output/nanogpt-adam-only-curvature-20261123
```
This comparison uses different optimizer-trained weights; same-step curvature differences are not a controlled same-state intervention or a matched-loss comparison.

Resolved metadata audit: original baseline predates the muon_orthogonalization selector. New config records newton_schulz for this unused Adam field; all other resolved model fields match exactly. All dataset/vocabulary settings match and both first-step losses equal4.174385070800781. All200matrix metric records are labeled Adam. This metadata schema difference does not change Adam training.


## Final completion

Training and curvature status: completed. Curvature supervisor exited0, all32cases completed in271.3563858330017seconds; all96block solves accepted. Canonical single-worker process absent, TPU READY/HEALTHY; both run-specific monitors retired after terminal evidence. Report server remains running at http://localhost:43305/.

On Adam's own trajectory, Adam has lower normalized directional curvature than the raw gradient in32/32cases. Median Adam/raw-gradient ratio0.384890424871684. Mean Adam/Newton cosine0.3113135054145453at damping ratio.01,0.37546453894793386at.1,0.39625187192262445at1.0. Independent NumPy reconstruction from original Adam moments and saved diagnostic gradients matches every saved update within1.4901161193847656e-08maximum absolute discrepancy; all reported Newton cosines independently verified. Diagnostic updates use highest-precision float32 gradients and thus are not bitwise historical next-step updates.

Curvature archive gs://kmh-gcp-us-central2/kristine/modular_norm/nanogpt-adam-only-curvature-20261123/results.tar.gz, SHA25624a7ef0833e2e3ce11ebbf85248a44c65d8e58e56b88e7f1f33a530fcef326c2. Training archive gs://kmh-gcp-us-central2/kristine/modular_norm/nanogpt-adam-only-shakespeare-20261123/results.tar.gz, SHA2564a8455b258e1e3442e5fceb2fa543303bb672a4f6871748999146af71f6f527c. Both downloads and every manifest entry verified. Full effective configs, logs, training checkpoints and curvature arrays are retained locally and regionally. Compact result files committed separately; no checkpoints committed to Git.

No failed training or curvature attempts. One seed, one configured learning rate per optimizer, same-step rather than matched-loss curvature. Lower directional curvature alone does not establish better loss reduction or explain the optimizer loss gap.


## Curvature overlay (saved-result analysis)
Added curvature-overlay.png, CSV and summary with both independent optimizer trajectories. Bold curves show medians over eight matrices per checkpoint; faint curves show individual matrices, not uncertainty intervals. All four saved token/target batches match exactly across Adam and Muon. Median per-case ratio to own raw-gradient curvature: Adam0.384890424871684, Muon0.11217147959704159. Raw directional curvature and gradient-relative ratio are shown separately. Different weights/Hessians preclude causal interpretation of the cross-trajectory gap. No new model evaluations or training jobs.


## Downhill slopes (saved-array analysis)
No new training or model evaluation. Recomputed gradient dot products, update norms and cosine alignments from all64saved matrix/update arrays and checked against recorded values. Both optimizers point downhill in32/32cases each. Median downhill gain per unit Euclidean step: Adam0.07267337119117556, Muon0.03429415625398591nats/token per unit parameter distance. Median cosine to negative gradient: Adam0.3932234003629742, Muon0.3607895075564779. Adam has higher unit-step downhill slope in31/32corresponding cases and higher cosine in24/32; this compares different optimizer-trained weights, not same-state alternatives. Added downhill-slopes.png, CSV, JSON summary and derivation to the local report. Higher initial downhill gain and lower positive curvature are separate local properties; no finite-step loss-decrease measurement is claimed.
