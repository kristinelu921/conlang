# Overnight communication investigation

Status: `running`

## Current evidence

The current model sends four continuous hidden vectors from each sender. A receiver encoder mixes the eight vectors with the question, and a decoder predicts one answer token plus EOS. Each source world contains eight linked chains and eight questions; every sender passage contains one hop from every chain, while the complementary hop is in the other passage.

| Run family | Key change | Final validation exact match | Final validation joint log probability | Observation |
|---|---|---:|---:|---|
| Full data, hidden 128 | untied, no gates, message 4, depth 2 | 0.0390625 | -4.058863639831543 | All six causal arms are exactly 0.0390625; zero messages are slightly better in likelihood |
| Full data, hidden 128 | untied, no gates, message 8, depth 2 | 0.0390625 | -4.052503585815430 | All six causal arms are exactly 0.0390625; longer messages remain causally unused |
| Full data, hidden 128 | tied encoder, output width 256 | 0.0390625 | -4.04714298248291 | Tying did not improve exact match |
| Full data, hidden 512 | tied encoder, output width 256 | 0.046875 | -4.053905487060547 | Fourfold width did not establish strong generalization |
| Full data, hidden 512 | tied plus 64-channel role gates | 0.0390625 | -4.060502052307129 | Gates did not improve the result |
| 1,024-example prefix | tied, gates 64 | 0.0390625 | -14.307319641113281 | Training loss reached 0.0387 while validation likelihood deteriorated |
| 1,024-example prefix | untied, gates 64 | 0.046875 | -14.934398651123047 | Same memorization/generalization split |
| 32-example prefix | tied / untied | 0.015625 / 0.01171875 | -18.459566354751587 / -17.76997447013855 | Both nearly memorized the prefix but generalized poorly |
| One example | tied / untied | 0.0 / 0.0 | -37.64824861288071 / -35.6571341753006 | Mechanics can memorize; no held-out transfer |

The exact 256-example validation prefix has a 0.03515625 empirical baseline when the answer is guessed from only the relation domain inferred from the question. A full-data question-template lookup reaches 0.0703125. The trained full-data runs' 0.0390625--0.046875 range is therefore consistent with a question-only shortcut. The direct causal baseline confirms it: intact, zero, wrong-world, swapped, agent-1-only, and agent-2-only messages all yield the same 10 correct answers. Zeroing messages improves mean joint log probability by 0.0022974. Final sender-path gradient norms are 0.00452 and 0.00540 while the receiver path is 1.13643, about 210--251 times larger.

## Ranked hypotheses and tests

1. **Question-only shortcut dominates optimization — confirmed for the baseline.** The receiver learns the answer prior for each relation type while assigning no measurable causal weight to the messages. Intact, zero, and wrong-world messages produce identical exact predictions and nearly identical likelihoods.
2. **Four query slots are too narrow for eight facts per sender — ruled out as the primary blocker.** Doubling to eight vectors leaves exact predictions unchanged across all message interventions and makes the sender/receiver gradient imbalance larger.
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

The baseline is definitively in a question-only shortcut/local minimum, not merely weak at communication. Raw Transformer width is unlikely to fix it because hidden 512 already failed, and channel width is no longer a strong explanation because message 8 is equally flat. The two-layer reasoning stack is the remaining planned capacity test. Architecture changes will be judged first by whether they create a measurable intact-minus-zero/wrong-world gap, not by ordinary exact match alone. If depth 4 is also causally flat, the belief shifts strongly toward an objective/credit-assignment fix such as an auxiliary sender-content loss or a receiver design that cannot answer without messages.

## Evidence limitations

- Existing comparisons are single-seed experiments on the first 256 validation examples.
- Prior runs did not include causal message interventions, so their message use is inferred from output concentration and gradient imbalance rather than directly measured.
- The read-only W&B reconciliation could enumerate projects but not runs; all values above come from checked-in effective configs, metrics JSONL, outputs JSONL, launcher logs, and reports.
