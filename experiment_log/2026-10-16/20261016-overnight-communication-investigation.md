# Overnight communication investigation

Status: `running`

## Current evidence

The baseline model sends four continuous hidden vectors from each sender. Its receiver encoder mixes the eight vectors with the question, and a decoder predicts one answer token plus EOS. The current bypass arm masks the receiver's direct question input and exposes the question only to the senders. Each source world contains eight linked chains and eight questions; every sender passage contains one hop from every chain, while the complementary hop is in the other passage.

| Run family | Key change | Final validation exact match | Final validation joint log probability | Observation |
|---|---|---:|---:|---|
| Full data, hidden 128 | untied, no gates, message 4, depth 2 | 0.0390625 | -4.058863639831543 | All six causal arms are exactly 0.0390625; zero messages are slightly better in likelihood |
| Full data, hidden 128 | untied, no gates, message 8, depth 2 | 0.0390625 | -4.052503585815430 | All six causal arms are exactly 0.0390625; longer messages create no beneficial causal gap |
| Full data, hidden 128 | untied, no gates, message 4, depth 4 | 0.0390625 | -4.059023857116699 | Wrong-world reaches 0.04296875 and zero has better likelihood; extra depth creates no beneficial causal gap |
| Full data, hidden 128 | question-conditioned senders, untied, message 4, depth 2 | 0.0390625 | -4.055914878845215 | Target-question/wrong-passages also reaches 0.0390625 with better likelihood; conditioning creates no beneficial passage-specific gap |
| Full data, hidden 128 | question-conditioned senders, receiver question masked, message 4, depth 2 | 0.03515625 | -4.054939270019531 | Zero messages collapse to 0.0078125 / -10.2608, but wrong passages retain 0.03515625; learned messages are necessary, while question identity remains unisolated |
| Full data, hidden 128 | question-conditioned senders, receiver question masked, message 8, depth 2 | 0.03515625 | -4.053813934326172 | Wrong-question falls to 0.0078125 / -10.3296, proving beneficial question transport; wrong passages retain 0.03515625 and slightly better likelihood, so private facts remain unused beneficially |
| Full data, hidden 128 | tied encoder, output width 256 | 0.0390625 | -4.04714298248291 | Tying did not improve exact match |
| Full data, hidden 512 | tied encoder, output width 256 | 0.046875 | -4.053905487060547 | Fourfold width did not establish strong generalization |
| Full data, hidden 512 | tied plus 64-channel role gates | 0.0390625 | -4.060502052307129 | Gates did not improve the result |
| 1,024-example prefix | tied, gates 64 | 0.0390625 | -14.307319641113281 | Training loss reached 0.0387 while validation likelihood deteriorated |
| 1,024-example prefix | untied, gates 64 | 0.046875 | -14.934398651123047 | Same memorization/generalization split |
| 32-example prefix | tied / untied | 0.015625 / 0.01171875 | -18.459566354751587 / -17.76997447013855 | Both nearly memorized the prefix but generalized poorly |
| One example | tied / untied | 0.0 / 0.0 | -37.64824861288071 / -35.6571341753006 | Mechanics can memorize; no held-out transfer |

The exact 256-example validation prefix has a 0.03515625 empirical baseline when the answer is guessed from only the relation domain inferred from the question. A full-data question-template lookup reaches 0.0703125. The direct-question full-data runs' 0.0390625--0.046875 range is therefore consistent with a question-only shortcut. The direct causal baseline confirms it: intact, zero, wrong-world, swapped, agent-1-only, and agent-2-only messages all yield the same 10 correct answers. Zeroing messages improves mean joint log probability by 0.0022974. Final sender-path gradient norms are 0.00452 and 0.00540 while the receiver path is 1.13643, about 210--251 times larger.

## Ranked hypotheses and tests

1. **Question-only shortcut dominates optimization — confirmed for the baseline.** The receiver learns the answer prior for each relation type while assigning no measurable causal weight to the messages. Intact, zero, and wrong-world messages produce identical exact predictions and nearly identical likelihoods.
2. **Four query slots are too narrow for eight facts per sender — ruled out as the primary blocker.** Doubling to eight vectors leaves exact predictions unchanged across all message interventions and makes the sender/receiver gradient imbalance larger.
3. **Two layers are too shallow for character parsing, fact selection, and a two-hop join — ruled out as the primary blocker.** Depth 4 halves throughput but keeps intact messages no better than zero or wrong-world messages.
4. **Capacity helps only after the channel and depth are adequate.** Hidden 512 alone already failed to move beyond the shortcut regime. Test width 256 only after another intervention identifies a communicating architecture.
5. **Tying and role gates are secondary.** Existing matched small-prefix pairs and full-data runs show no robust benefit. Re-test shared versus unshared weights only on the best communicating architecture, so the comparison measures generalization rather than prior-fitting.
6. **Question conditioning alone shortens the path but does not repair credit assignment — ruled out as a sufficient intervention.** Target-question/wrong-passages matches intact exact accuracy and improves likelihood, while sender gradients become even smaller relative to the receiver.
7. **Channel capacity matters only after the shortcut is blocked — ruled out as the private-fact blocker.** Message length 8 makes the active channel's question content explicit: wrong-question re-encoding loses 7/256 exact answers and 6.28 nats. Wrong passages still match intact exact accuracy and slightly improve likelihood, so extra slots do not create private-fact transport.
8. **The end-to-end likelihood objective supplies too little fact-binding signal — leading hypothesis.** The wider active channel remains passage-flat despite transmitting question identity. Test an explicit sender-content objective or sequential exchange before stacking generic width, depth, or weight sharing.

## Controlled sequence

| Order | Intervention | Purpose | Stacking rule |
|---:|---|---|---|
| 1 | Message 4, depth 2 causal baseline | Measure whether the established architecture uses messages at all | No training change |
| 2 | Message 8 only | Test channel capacity | Keep only if causal gap improves |
| 3 | Depth 4 only | Test reasoning depth | Keep only if causal gap improves |
| 4 | Question-conditioned senders, message 4, depth 2 | Shorten the sender credit path and make messages query-specific | Keep only if a causal message gap appears |
| 5 | Mask direct receiver question input | Remove the confirmed cheap shortcut while preserving question-conditioned senders | Keep only if intact beats both zero and wrong passages |
| 6 | Message 8 on the sender-only question path | Test channel capacity after messages become causally active | Completed: question transport confirmed; reject as a private-fact solution because wrong passages remain tied |
| 7 | Explicit sender-content objective or sequential exchange | Supply a direct fact-binding path if the wider active channel remains flat | Design from the stacked result |
| 8 | Hidden 256 and tied versus untied on the best communicating arm | Test capacity and sharing after fact communication is present | Matched comparisons, same seed and schedule |

Every arm uses seed 1730, the same Central2 dataset, the same Kristine v4-8, batch 256, 2,000 steps, bfloat16 compute, output hidden width 256, AdamW settings, and the causal intervention evaluation unless a report explicitly records otherwise.

## Current belief

The baseline is definitively in a question-only shortcut/local minimum, not merely weak at communication. Raw width, message length, depth, and question-conditioned sender selection all failed while the receiver retained its direct question path. Masking that path creates strong learned-message dependence, and the matched message-8 control now identifies what is transmitted: replacing only the sender-side target question with another question from the same world drops exact match from 9/256 to 2/256 and likelihood by 6.28 nats. The agents therefore communicate question identity. They still do not use private passage content beneficially: target-question/wrong-passages remains tied at 9/256 and is marginally better in likelihood. The remaining blocker is fact selection/binding across the complementary hops, not raw channel width. The next intervention should add explicit intermediate supervision or sequential cross-agent exchange before more generic width, depth, or weight-sharing comparisons.

## Evidence limitations

- Existing comparisons are single-seed experiments on the first 256 validation examples.
- The earlier tied-width, role-gate, small-prefix, and one-example runs did not include causal message interventions, so their message use is inferred from output concentration and gradient imbalance rather than directly measured.
- The read-only W&B reconciliation could enumerate projects but not runs; all values above come from checked-in effective configs, metrics JSONL, outputs JSONL, launcher logs, and reports.
