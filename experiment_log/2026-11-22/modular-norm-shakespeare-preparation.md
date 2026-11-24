# modular_norm Shakespeare Muon preparation

## Run Summary

- Status: planned; no training, evaluation, or benchmark has launched.
- Repository: https://github.com/kristinelu921/modular_norm
- Local implementation: /tmp/modular_norm, branch codex/muon-tightness (uncommitted experiment implementation).
- Baseline: not specified.
- Proposed run identity: shakespeare-muon-v4-8-20261123.
- Seed: 42; independent probe seed 1729.
- TPU: kmh-tpuvm-v4-8-spot-kristine-21-13-47-32-N-us-central2-b, he-vision-group, us-central2-b, v4-8, Spot.
- Discovery completed successfully; selected from final AVAILABLE table. Exact describe READY/HEALTHY and occupancy AVAILABLE on sole worker t1v-n-087885ef-w-0.
- No commit/PR or launch command yet; awaiting scientific choices from user.

## Configuration

Draft complete input: /tmp/modular_norm/configs/shakespeare-muon.json.
Proposed: character-level Tiny Shakespeare, vocabulary 65, contiguous 90/10 split; 2 pre-LayerNorm causal Transformer blocks, width 128, 4 heads, ReLU MLP width 512, context 128, batch 32, 2000 updates, fixed LR .01, Muon momentum .95 with Nesterov and 5 quintic Newton–Schulz iterations, float32, no weight decay or dropout. All two-dimensional parameters, including untied token/position embeddings and output, use Muon; LayerNorm vectors use SGD. No weight norm projection. Loss and update constraint remain pending user confirmation.

## Measurements

Every 100 updates: actual single-step W_after - W_before; exact operator norm via SVD; all matrix inputs from the actual gradient-producing batch, and larger fixed dataset subsets (scope pending). Planned larger subsets: 32768 token positions each from training and validation. Also fixed Gaussian probes. Statistics: mean, RMS, median, standard deviation, p95, p99, max; mean/max tightness against exact operator norm; operator norm versus nominal eta times shape scale. Larger-subset maximum is empirical, not the true operator norm.

## Preparation / Validation

- Remote disk: 80 GiB free on node-local root.
- Matching regional bucket kmh-gcp-us-central2 verified US-CENTRAL2; kristine/datasets/tinyshakespeare prefix absent. Direct remote download authorized fallback.
- Isolated runtime /home/kristinelu/.venvs/modular_norm: Python 3.11.18, JAX/JAXLIB 0.4.38, NumPy 1.26.4. Dependencies installed without opening accelerators.
- Remote dataset path /home/kristinelu/elf-data/tinyshakespeare/input.txt.
- Four local non-experiment metric unit tests passed: isometry, rank-one, rectangular orientation/zero input, zero update. Python syntax checks passed.
- Model/optimizer tests authored but not yet run; to run on remote compute.
- SSH default cipher intermittently failed with 'Bad packet length / message authentication code incorrect'. Read-only retries with aes128-ctr and hmac-sha2-256 succeeded. No workload was started or affected.

## Pending

User questions: ordinary Muon versus enforcing nominal update bound; Brier versus cross entropy; batch plus larger train and validation subsets versus train only. Do not launch until choices resolved. Then finalize config, review, commit/push and PR, snapshot, remote tests, fresh occupancy, throughput gate, full run, monitoring through completion, verified artifacts.
