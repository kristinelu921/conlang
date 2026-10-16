# Full-data sender-only question-path communication diagnostic

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qsender-norecvq-commdiag` |
| Status | `planned` |
| Baseline | `20261016-conlang-v4-8-full-m4-d2-qsender-commdiag` |
| Git revision | reviewed implementation and configuration closure `bc4630f78867f1983fad2456da6a44a33ed0c333`; immutable launch snapshot pending |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Command | `/home/kristinelu/elf-runs/runtime/venv/bin/python -u train.py --config configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-commdiag.yaml` inside the immutable node-local source snapshot |
| Seed | `1730` |
| Hardware | exact same `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, one host, four TPU devices |

## Configuration

Source: `configs/conlang-jax-v4-8-full-m4-d2-qsender-norecvq-commdiag.yaml`

The run uses the full verified Central2 dataset, batch 256, 2,000 steps, one final 256-example causal evaluation, bfloat16 compute, hidden width 128, output hidden width 256, four message vectors, two layers, untied encoders, no role gates, question-conditioned senders, and W&B enabled.

## Changed Hyperparameters

| Parameter | Baseline | Current | Rationale |
|---|---|---|---|
| Direct receiver question input | enabled | disabled | Remove the confirmed cheap question-only path so the question and passage evidence must reach the receiver through sender messages |

All data, seed, optimization, schedule, widths, message length, depth, sharing, gates, sender inputs, hardware, and causal evaluation arms remain fixed.

## Hypothesis and Decision Rule

Question-conditioned senders did not produce a beneficial passage-specific causal gap while the receiver still saw the question directly. That direct path can continue fitting relation-conditioned answer priors without using messages. This arm masks all receiver question positions; the senders remain question-conditioned, so the question and private evidence can reach the answer decoder only through their eight combined message vectors.

The architecture establishes useful communication only if intact messages outperform both zero messages and target-question/wrong-passages messages. A large intact-versus-zero gap without an intact-versus-wrong-passages gap would show question transport but not useful private-content transport.

## Results

Pending launch and verified completion.

## Review and Verification

- Correctness review passed for teacher-forced and greedy masking, sender conditioning, target-question/wrong-passages semantics, tied/untied compatibility, default parameter-tree compatibility, and the causal decision rule.
- Cleanup review passed after aligning receiver documentation with optional question context.
- The full CPU regression suite passed 52 tests with two existing Pydantic warnings.
- The selected YAML validates and differs from the completed question-conditioned arm only by fresh run identity fields and `receiver_question_input: false`.

## Updates

- `2026-10-16`: Planned after question-conditioned senders showed no beneficial passage-specific gap while the receiver retained a direct question-only shortcut.
