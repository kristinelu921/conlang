# Full-data question-conditioned sender communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-commdiag` |
| Status | `planned` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-commdiag` |
| Git revision | pending reviewed commit |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-commdiag.yaml` inside the immutable node-local source snapshot |
| Seed | `1730` |
| Hardware | exact same `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-commdiag.yaml`

The run uses the full verified Central2 dataset, batch 256, 2,000 steps, one final 256-example causal evaluation, bfloat16 compute, hidden width 128, output hidden width 256, four message vectors, two layers, untied encoders, no role gates, and W&B enabled.

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---|---|---|
| Sender input | private passage only | question followed by private passage | Shorten the credit path by letting each sender select question-relevant evidence before compressing its passage into four vectors |

All data, seed, optimization, schedule, widths, message length, depth, weight sharing, gates, receiver inputs, hardware, and causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

The passage-only sender must learn a query-independent code for eight facts using only a weak gradient routed through the receiver, while the receiver can cheaply fit question-conditioned answer priors. Conditioning both senders on the question should make the four message vectors query-specific and strengthen the sender learning signal without increasing width, depth, or channel length.

Count this as private-content communication only if intact messages outperform the target-question/wrong-passage control in exact match or joint log probability, with zero messages as a second reference. The wrong-world arm rotates only the two private passages and re-encodes them against each target row's unchanged question, so a gap cannot be attributed merely to question mismatch. Ordinary intact exact match or a gap over zero alone is insufficient. Keep the switch for subsequent matched sharing/width experiments only if it produces that passage-specific causal gap.

## Results

Pending launch and verified completion.

## Updates

- `2026-10-16`: Planned after message length 8 and depth 4 both remained causally flat, and after focused CPU tests confirmed that conditioned messages change when only the question changes.
