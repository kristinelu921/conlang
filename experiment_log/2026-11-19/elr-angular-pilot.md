# Matched ELR and angular dynamics pilot

## Run Summary

| Field | Value |
|---|---|
| Run ID | elr-angular-pilot (planning identity) |
| Status | planned |
| Started / finished | not recorded |
| Baseline | paired fixed-matrix-norm arm after shared warmup |
| Git revision / PR | not recorded; implementation is uncommitted |
| Command | not recorded; no launch issued |
| Seeds | planned: 1729, 1730 |
| Hardware | not recorded; no available authorized TPU |

## Configuration

Paper: [Effective Learning Rate Governs Loss Dynamics in Language Model
Pretraining](https://arxiv.org/abs/2608.24814), August 25, 2026, Appendix B.2.
This is a reduced-scale probe, not an exact reproduction of its 124M/FineWeb
experiment. Source: `elr_probe/train.py` and `elr_probe/dynamics.py`.
No launch YAML has been selected or committed. Intended configuration:

```yaml
width: 128
layers: 4
heads: 4
vocabulary_size: 50304
batch_size: 32
sequence_length: 256
steps: 600
warmup_steps: 100
learning_rate: 0.001
final_norm_multiplier: 2.0
checkpoint_every: 200
seeds: [1729, 1730]
```

Constants: float32; Adam beta1=0.9, beta2=0.95, epsilon=1e-8; zero weight
decay, no clipping or dropout; learned RMSNorm gains and QK normalization;
RoPE; SwiGLU intermediate width 4x; tied embeddings. Loss EMA=0.99.
All two-dimensional parameters, including embeddings, receive norm control.
Vector parameters receive the arm's LR without projection; the paper's exact
vector LR treatment is not independently verified, so this is an explicit
implementation choice.

Intended data: GPT-2-tokenized WikiText-2 raw training split, with EOS between
documents, in a one-dimensional uint16 array. Dataset/tokenizer revisions,
checksum, regional source, and local path: not recorded. Inspect the selected
region's kristine dataset path first; download remotely only if that copy is
absent or invalid. No data download has occurred. This dataset differs from
the paper and limits interpretation.

## Changed Hyperparameters

| Parameter | Baseline A | Matched B |
|---|---|---|
| Initialization, Adam state, data order | shared | shared |
| First 100 updates | common warmup | branches from same state |
| Matrix norms afterward | warmup-end norms | linearly ramp to 2x over 500 updates |
| LR afterward | 0.001 | ramps with the same multiplier |

Projection after each update targets the next step's norm. Pre-update
eta/norm therefore matches separately for every matrix. Rotation is measured
before projection, which preserves direction. Both arms consume identical
batches at each step. Each seed defines an independent pair.

## Results

No experimental results; no training or evaluation has run.
Planned metrics: norm, ELR, rotation in radians, radial/tangential relative
update, unscaled Adam update norm, and R_theta=theta_A/theta_B, for each matrix
and step. Angles use a stable half-angle identity. Ratios with denominator
<=1e-8 radians are null. ELR relative mismatch >2e-5 aborts the pilot. Raw and
EMA cross-entropy use nats/token.

Compare post-warmup loss discrepancy to the two-seed baseline variation and
report seeds separately. Two seeds are exploratory, not a significance test.
Sustained nonunit rotation ratios with small loss discrepancy would support
the proposed distinction in this setting. Failure of this reduced pilot to
show loss collapse is inconclusive about the paper. Functional sensitivity is
not measured in this first pilot.

## Artifacts

| Artifact | Path | Verified |
|---|---|---|
| Pilot source | `elr_probe/train.py` | syntax checked; remote integration pending |
| Diagnostics/controller | `elr_probe/dynamics.py` | analytic tests pass |
| Tests | `tests/test_elr_dynamics.py` | 4 passed |
| Checkpoints / tracking / plots | not recorded | no |
| Capacity watcher | `experiment_log/2026-11-19/monitor-elr-angular-capacity.sh` | activation below |

## Failure Details

- Last known step: not applicable; never launched.
- Capacity checks returned final AVAILABLE=`none`. Candidates were creating,
  deleting, or stale inventory entries. No other owners' resources inspected.
- Initial snap-confine failure resolved by approved external cloud access.
- Remote model compilation, data preparation, checkpoint recovery,
  throughput gate, immutable source commit/PR, and active-run monitor pending.

## Updates

- 2026-11-19T05:05:36Z: Paper identified; protocol inspected; pilot source
  prepared. Four analytic CPU fixture tests pass. No local experiment run.
- 2026-11-19T05:08:00Z: Capacity watcher verified in persistent session 57355;
  first completed audit recorded in `elr-angular-capacity.log`, AVAILABLE=none.
  It checks every 1800 seconds and sends ntfy on available capacity. It does
  not launch automatically: scientific/runtime preflight and source PR remain
  required. Preparation milestone notification delivered successfully.
