# Overnight communication investigation

Status: `running`

## Current evidence

The current model sends four continuous hidden vectors from each sender. A receiver encoder mixes the eight vectors with the question, and a decoder predicts one answer token plus EOS. Each source world contains eight linked chains and eight questions; every sender passage contains one hop from every chain, while the complementary hop is in the other passage.

| Run family | Key change | Final validation exact match | Final validation joint log probability | Observation |
|---|---|---:|---:|---|
| Full data, hidden 128 | untied, no gates, message 4, depth 2 | 0.0390625 | -4.058863639831543 | Six dominant predictions; near the question-only baseline |
| Full data, hidden 128 | tied encoder, output width 256 | 0.0390625 | -4.04714298248291 | Tying did not improve exact match |
| Full data, hidden 512 | tied encoder, output width 256 | 0.046875 | -4.053905487060547 | Fourfold width did not establish strong generalization |
| Full data, hidden 512 | tied plus 64-channel role gates | 0.0390625 | -4.060502052307129 | Gates did not improve the result |
| 1,024-example prefix | tied, gates 64 | 0.0390625 | -14.307319641113281 | Training loss reached 0.0387 while validation likelihood deteriorated |
| 1,024-example prefix | untied, gates 64 | 0.046875 | -14.934398651123047 | Same memorization/generalization split |
| 32-example prefix | tied / untied | 0.015625 / 0.01171875 | -18.459566354751587 / -17.76997447013855 | Both nearly memorized the prefix but generalized poorly |
| One example | tied / untied | 0.0 / 0.0 | -37.64824861288071 / -35.6571341753006 | Mechanics can memorize; no held-out transfer |

The exact 256-example validation prefix has a 0.03515625 empirical baseline when the answer is guessed from only the relation domain inferred from the question. A full-data question-template lookup reaches 0.0703125. The trained full-data runs' 0.0390625--0.046875 range is therefore consistent with a question-only shortcut. Their final sender-path gradient norms are about 0.004--0.006 while the receiver path is about 0.84--1.18, further motivating a direct causal test rather than another uninstrumented size sweep.

## Ranked hypotheses and tests

1. **Question-only shortcut dominates optimization.** The receiver learns the answer prior for each relation type while assigning little causal weight to the messages. Test intact messages against zero messages and messages from the wrong world. Evidence for communication requires both higher exact match and higher answer log probability with intact messages.
2. **Four query slots are too narrow for eight facts per sender.** Each sender must compress eight independently queryable facts into four vectors. Test message length 8 with every other setting fixed. A larger intact-minus-wrong-world gap would support this bottleneck hypothesis.
3. **Two layers are too shallow for character parsing, fact selection, and a two-hop join.** Test depth 4 with message length fixed at 4. If it improves communication without message expansion, reasoning depth is the stronger bottleneck.
4. **Capacity helps only after the channel and depth are adequate.** Hidden 512 alone already failed to move beyond the shortcut regime. Test width 256 only after message/depth experiments identify a communicating architecture.
5. **Tying and role gates are secondary.** Existing matched small-prefix pairs and full-data runs show no robust benefit. Re-test shared versus unshared weights only on the best communicating architecture, so the comparison measures generalization rather than prior-fitting.
6. **The end-to-end likelihood objective supplies too little sender learning signal.** If wider messages and deeper stacks still show no causal message gap, test an architectural oracle or an auxiliary sender-content objective before stacking more width.

## Controlled sequence

| Order | Intervention | Purpose | Stacking rule |
|---:|---|---|---|
| 1 | Message 4, depth 2 causal baseline | Measure whether the established architecture uses messages at all | No training change |
| 2 | Message 8 only | Test channel capacity | Keep only if causal gap improves |
| 3 | Depth 4 only | Test reasoning depth | Keep only if causal gap improves |
| 4 | Message 8 plus depth 4 | Test complementarity | Run only after individual arms |
| 5 | Hidden 256 on the best prior arm | Test model width after communication is present | Do not repeat hidden 512 blindly |
| 6 | Tied versus untied on the best arm | Test sharing as a generalization constraint | Matched pair, same seed and schedule |

Every arm uses seed 1730, the same Central2 dataset, the same Kristine v4-8, batch 256, 2,000 steps, bfloat16 compute, output hidden width 256, AdamW settings, and the causal intervention evaluation unless a report explicitly records otherwise.

## Current belief

The most likely near-term win is not raw Transformer width. The current evidence favors a shortcut/local-minimum explanation, with the four-slot channel and two-layer reasoning stack as plausible contributors. The causal baseline is the decision gate: if intact, zero, and wrong-world messages are indistinguishable, architecture changes will be judged by whether they create a measurable message-dependence gap, not by ordinary exact match alone.

## Evidence limitations

- Existing comparisons are single-seed experiments on the first 256 validation examples.
- Prior runs did not include causal message interventions, so their message use is inferred from output concentration and gradient imbalance rather than directly measured.
- The read-only W&B reconciliation could enumerate projects but not runs; all values above come from checked-in effective configs, metrics JSONL, outputs JSONL, launcher logs, and reports.
