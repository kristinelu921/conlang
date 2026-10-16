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
| Full data, hidden 128 | two sender rounds, receiver question masked, message 4, depth 2 | 0.0390625 | -4.055304527282715 | Wrong-question falls to 0.01171875 / -9.9847, but wrong passages retain 0.0390625 and slightly better likelihood; sequential exchange still does not use private facts beneficially |
| Full data, hidden 128 | bridge-supervised sender, receiver question masked, message 4, depth 2 | 0.06640625 | -4.0323076248168945 | Bridge accuracy reaches 0.125; wrong passages reduce bridge accuracy to 0.0234375, answer exact to 0.046875, and likelihood by 0.0431 nats: the first beneficial private-passage gap |
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
8. **The tested synchronous second exchange supplies the missing join path — ruled out as a sufficient architectural fix.** The exchange gives either sender the other sender's first-round representation, yet wrong passages preserve all 10 exact answers and slightly improve likelihood. This interaction design remains passage-flat.
9. **The end-to-end answer objective supplies too little bridge-selection signal — strongly supported as a material blocker.** Explicit bridge supervision lifts bridge accuracy to 32/256 from 6/256 under either wrong questions or wrong passages. Intact answers beat wrong passages by 5/256 and 0.0431 nats, creating the first beneficial private-passage gap. This establishes that direct bridge credit repairs an important failure mode in this setup, not that it is the unique remaining cause.
10. **Capacity and sharing can now be tested on a communicating arm.** The bridge-supervised model still solves only 17/256 validation examples. Test hidden width 256 first with the current untied architecture, then run a matched tied-encoder arm. Keep changes only when the intact-over-wrong-passages answer gap and bridge accuracy improve, not merely raw exact match.

## Controlled sequence

| Order | Intervention | Purpose | Stacking rule |
|---:|---|---|---|
| 1 | Message 4, depth 2 causal baseline | Measure whether the established architecture uses messages at all | No training change |
| 2 | Message 8 only | Test channel capacity | Keep only if causal gap improves |
| 3 | Depth 4 only | Test reasoning depth | Keep only if causal gap improves |
| 4 | Question-conditioned senders, message 4, depth 2 | Shorten the sender credit path and make messages query-specific | Keep only if a causal message gap appears |
| 5 | Mask direct receiver question input | Remove the confirmed cheap shortcut while preserving question-conditioned senders | Keep only if intact beats both zero and wrong passages |
| 6 | Message 8 on the sender-only question path | Test channel capacity after messages become causally active | Completed: question transport confirmed; reject as a private-fact solution because wrong passages remain tied |
| 7 | Synchronous second sender exchange, message 4 | Let either sender combine an incoming bridge representation with its complementary private hop | Completed and rejected: wrong passages retain all exact answers and slightly better likelihood |
| 8 | Explicit bridge-entity sender objective | Supply direct credit for selecting the private hop needed by the target question | Completed and kept: bridge prediction learns and intact answers beat wrong passages by 5/256 plus 0.0431 nats |
| 9 | Hidden 256 and tied versus untied on the bridge-supervised arm | Test capacity and sharing after fact communication is present | Next: untied width 256 first, then matched tied width 256; same seed and schedule |

Every arm uses seed 1730, the same Central2 dataset, the same Kristine v4-8, batch 256, 2,000 steps, bfloat16 compute, output hidden width 256, AdamW settings, and the causal intervention evaluation unless a report explicitly records otherwise.

## Current belief

The baseline is definitively in a question-only shortcut/local minimum, not merely weak at communication. Masking the receiver question path makes learned messages necessary, and matched wrong-question controls show that those messages carry question identity, but longer messages and a synchronous sender exchange remain passage-flat. Explicit bridge supervision changes that result: bridge accuracy reaches 32/256, wrong-question and wrong-passage bridge controls fall to the 6/256 empirical constant baseline, and intact answers beat wrong passages by 5/256 plus 0.0431 nats. Insufficient bridge-selection credit was therefore a material blocker in this setup. It is not necessarily the only remaining problem—17/256 exact is still low—so width and sharing should now be compared on this causally communicating arm rather than on the old shortcut solution.

## Evidence limitations

- Existing comparisons are single-seed experiments on the first 256 validation examples.
- The earlier tied-width, role-gate, small-prefix, and one-example runs did not include causal message interventions, so their message use is inferred from output concentration and gradient imbalance rather than directly measured.
- The read-only W&B reconciliation could enumerate projects but not runs; all values above come from checked-in effective configs, metrics JSONL, outputs JSONL, launcher logs, and reports.
