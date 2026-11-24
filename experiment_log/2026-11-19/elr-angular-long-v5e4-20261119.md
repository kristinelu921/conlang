# Longer matched-ELR angular experiment

## Run Summary

- Run ID: `elr-angular-long-v5e4-20261119`; status: completed. Started: 2026-11-19T16:17:52Z; finish observed: 2026-11-19T17:04:17Z.
- Baseline: [600-step pilot](elr-angular-v5e4-20261119-1548.md).
- Commit: a0eeef4; PR: https://github.com/kristinelu921/conlang/pull/6
- Primary config: `configs/elr-angular-long-v5e4-20261119.yaml`.
- Seeds: 1729, 1730. Hardware: single-host four-chip v5e, `kmh-tpuvm-v5e-4-us-west4-spot-jaustin-zg3ocs`, us-west4-a, he-vision-group.
- Exact command (node-local immutable source cwd):
  `JAX_PLATFORMS=tpu CUDA_VISIBLE_DEVICES='' HF_HOME=/home/kristinelu/elf-data/huggingface /home/kristinelu/elf-runs/elr-runtime/bin/python -u -m elr_probe.train --config configs/elr-angular-long-v5e4-20261119.yaml --data /home/kristinelu/elf-data/wikitext-2-gpt2/train.npy --output /home/kristinelu/elf-runs/elr-angular-long-v5e4-20261119`

## Configuration

```yaml
run_id: elr-angular-long-v5e4-20261119
wandb_project: elr-angular-dynamics
width: 128
layers: 4
heads: 4
vocabulary_size: 50304
batch_size: 32
sequence_length: 256
steps: 6000
warmup_steps: 100
ramp_end_step: 600
learning_rate: 0.001
final_norm_multiplier: 2.0
checkpoint_every: 3000
seeds: [1729, 1730]
angular_patterns: true
log_every: 10
pattern_log_every: 10
```

Effective fixed settings: Adam beta1=0.9, beta2=0.95, epsilon=1e-8; no decay, clipping or dropout; float32, highest matmul precision; SwiGLU 4x hidden expansion, tied embeddings, RMS gains and QK normalization, RoPE. Matrix norms controlled after 100 warmup updates; vector parameters unprojected but use each arm's LR. Same minibatches in both arms. Loss EMA decay 0.99. Local WikiText-2 raw train tokenized with GPT-2, 2,415,651 uint16 tokens; SHA256 4b9499e72a3121579e2170810b259654cc216f6d55196e919605f50fe9e673fc. Pinned dataset/tokenizer revisions and runtime match baseline report.

## Changed Hyperparameters

600 → 6000 updates per seed. Ramp still ends at step600, then stays at 2x norm/LR. Checkpoints every200 →3000 updates. Logging every1 →10 updates plus final. Added angular diagnostics; cumulative state updates each optimizer step. Seeds replay from initialization to measure full trajectories; this does not load the pilot endpoint.

## Results

Not recorded; pending launch. Geometry is parameter-space geometry, not functional sensitivity. Repeated training-set exposure limits conclusions about generalization.

## Artifacts

- Node-local output: `/home/kristinelu/elf-runs/elr-angular-long-v5e4-20261119` (pending).
- Regional archive: `gs://kmh-gcp-us-west4/kristine/experiments/elr-angular-long-v5e4-20261119/` (pending).
- W&B: https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/elr-angular-dynamics/runs/fvlkctal (verified initialized and progressing).
- Analytic tests: 13 passed, CPU-only fixtures; no local model workload.

## Failure Details

No training failure. One artifact-download transport error is recorded below.

## Updates

- 2026-11-19: Filtered capacity table had no suitable single-host kristine TPU; full approved wide sweep listed selected v5e4 idle. Exact occupancy AVAILABLE. Data hash verified; runtime reused. Free root disk2.6GB; four expected270MB checkpoints plus logs fit. Original pilot demonstrated compatible topology and throughput; measure diagnostic overhead on this launch.

- Review ledger: correctness head c9beb70: resume without diagnostics failed for matched arm; fixed at a0eeef4 with focused two-arm restoration coverage. Cleanup head a0eeef4: NO_CONFIRMED_FINDINGS. 13 tests passed. Final commit a0eeef4d4dbc19de71d0d642af567454cdb74f27; PR6 ready.
- Immutable snapshot: /kmh-nfs-ssd-us-mount/staging/kristine/elf/elr-angular-long-v5e4-20261119-20261119T161654.153722326Z.

- 2026-11-19T16:17:52Z: supervisor2417366, trainer2417367 started from node-local snapshot. Remote source/config SHA256 matched local files. W&B https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/elr-angular-dynamics/runs/fvlkctal initialized; first-update compilation pending. Minute monitor active (monitor.sh, monitor.log).

- 2026-11-19T16:21Z: status running; seed1729 progressed through420 in mirrored metrics. Early overlapping loss maximum absolute difference from pilot9.5367431640625e-7 nats across both arms. Stable steps120–180:4.520200913 paired updates/sec (batch32 per arm,8192 tokens per update). Original pilot approximately14.8 paired updates/sec: expanded geometry costs about3.3x; accepted for bounded6000-step diagnostic experiment, not a throughput comparison. ELR error early maximum2.4798790887459177e-7.
- Monitoring transport: SCP metric retrieval repeatedly returned `message authentication code incorrect`; the training SSH monitor remained healthy and progressing. Retrieved complete metrics via regional GCS mirror instead. No training restart or data mutation; root cause of transport error unconfirmed. Existing errors skill SCP/SSH interruption guidance applied.

## Verified completion and findings

Both seeds completed6000 updates; trainer exit0, COMPLETED sentinel agrees, and both final checkpoints pass SHA256 verification on the sole worker. Final checkpoint sizes269607270 and269607271 bytes. Canonical worker0 train.log archived locally, SHA256 6efeaed69d0745338b769e756395363cf3c3e910c85214ad61905e6e14168368. Completion ntfy delivered successfully on this status check.

```json
{
  "1729": {
    "last_step": 5999,
    "final_baseline_loss_ema": 3.7565178096885132,
    "final_matched_loss_ema": 3.629021719253862,
    "final_loss_gap_matched_minus_baseline": -0.12749609043465115,
    "max_elr_relative_error": 5.272835490455208e-07,
    "late_rotation_ratio_median": 1.0038927496547208,
    "late_rotation_ratio_p10_p90": [
      0.9824726181712704,
      1.0356411459740658
    ],
    "last_between_arm_measurement_step": 5990,
    "last_median_between_arm_angle_radians": 0.8201043009757996,
    "final_post_warmup_path_efficiency": {
      "baseline": 0.05839404836297035,
      "matched": 0.05705728754401207
    }
  },
  "1730": {
    "last_step": 5999,
    "final_baseline_loss_ema": 3.7854536782892074,
    "final_matched_loss_ema": 3.684933237496609,
    "final_loss_gap_matched_minus_baseline": -0.10052044079259836,
    "max_elr_relative_error": 7.444329443995201e-07,
    "late_rotation_ratio_median": 1.005842250614244,
    "late_rotation_ratio_p10_p90": [
      0.983002221211209,
      1.0416909670578345
    ],
    "last_between_arm_measurement_step": 5990,
    "last_median_between_arm_angle_radians": 0.8287460207939148,
    "final_post_warmup_path_efficiency": {
      "baseline": 0.05775122717022896,
      "matched": 0.05989394336938858
    }
  }
}
```

Source: complete metrics-1729.jsonl and metrics-1730.jsonl from regional mirror. Late rotation ratios pool matrices and recorded steps from the second half of the controlled interval. Between-arm separation last measured at5990, not5999, because its logging cadence is10. Both final checkpoints verified.

Interpretation: instantaneous angular magnitudes remain close, but directions separate substantially (~0.82 radians median across matrices). Equal ELR does not constrain the full directional trajectory. Losses differ by0.10–0.13nats at the endpoint, so this is not evidence for equal-loss/different-angular-dynamics. Training-set repeated exposure limits generalization claims.

Plots: overview.png, patterns-1729.png, patterns-1730.png, served at http://localhost:8769/ .
