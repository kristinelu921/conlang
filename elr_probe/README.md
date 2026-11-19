# Matched ELR angular measurements

The long experiment replays the pilot's seeds and first 600-step schedule,
then continues to 6,000 steps. Norm/LR scaling reaches 2x at step 600 and
stays there; total training duration does not stretch the ramp. Each seed
still starts from the same paired warmup state. This is a small WikiText-2
probe, not a full-scale replication of the ELR paper.

Metrics apply separately to every matrix in every layer. Angles use normalized
weights and a stable half-angle identity, in radians. No weight update is
changed by the diagnostic calculations.

| Metric | Meaning |
|---|---|
| `angle_radians` | Rotation during one optimizer update, before norm projection |
| `path_length_radians` | Sum of actual per-step rotations since initialization |
| `displacement_from_init_radians` | Shortest angular distance from initial weights |
| `post_warmup_path_length_radians` | Angular distance traveled since the arms split |
| `displacement_from_warmup_radians` | Shortest angular distance from the split state |
| `path_efficiency` | Net displacement / path length; near 1 means a direct path, small values mean much movement with limited net displacement |
| `successive_tangent_cosine` | Alignment of consecutive angular directions, after parallel transport into the same tangent plane; +1 continues, -1 reverses |
| `between_arm_angle_radians` | Separation of A and B's matrix directions at the same post-update step |
| `neuron_angle_p10/p50/p90_radians` | Distribution of column-vector rotation in Dense kernels `[input, output]` |
| `head_N_angle_radians` | Q/K/V output-head block or O input-head block rotation |

Path efficiency is a geometric summary, not learning efficiency. Consecutive
directions are compared on the unit sphere of flattened matrix entries; this
does not identify all functionally equivalent neural-network parameterizations.

`successive_tangent_valid=0` marks stationary/near-antipodal steps or the lack
of a previous direction. Its cosine is then a zero placeholder and must be
excluded from aggregation. The warmup-reference metrics are placeholders until
`post_warmup_reference_valid=1`. All cumulative sums and tangent states update
every optimizer step; the long run writes snapshots every 10 steps and at its
last step. Do not sum sparsely logged instantaneous angles to reconstruct the
full path. First-step angles and cumulative path can differ slightly because
path metrics include floating-point projection effects.

The metric state and warmup reference are checkpointed alongside both Adam
states and the batch RNG. Resumption restores them; neither cumulative path
nor direction history resets. Initial references are regenerated from the
same model seed and pinned implementation. Checkpoint checksums and scientific
configuration must match.

Use time-by-layer heatmaps, per-head traces, and distributions over steps to
inspect patterns; aggregate medians alone can hide localized rotation. Future
functional diagnostics could compare held-out logits or activations under
small tangent perturbations, which this experiment does not yet measure.

References: [ELR paper](https://arxiv.org/abs/2608.24814),
[Rotational Equilibrium](https://proceedings.mlr.press/v235/kosson24a.html).
