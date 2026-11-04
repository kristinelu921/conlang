# Two-peer tree RL pilot

Status: completed. Twenty-two remote tests and the full-size numerical, replay,
backward, memory and checkpoint gates passed after a float32 retry. Across 32
training trees, MATH validation fell from 1/16 to 0/16 and AIME stayed at 0/30.
The pretrained single-agent AIME baseline scored 2/30. This pilot did not show
a collaborative gain. See the [experiment report](../../experiment_log/2026-11-03/qwen3-peer-tree-pilot-e225874.md)
and [verified result summary](../../experiment_log/2026-11-03/qwen3-peer-tree-pilot-e225874-results.json).

The completed forward-step rerun used `forward_step.json` and added an explicit
instruction to advance the solution on each turn and ask a focused follow-up
question in each non-final message. It starts from pretrained
weights with fresh adapters and reuses the original byte-verified train,
validation, and test splits; otherwise, the longer prompt could change the
length filter and seeded question selection. The parser, reward, sampling,
seed, and 32-tree limit are unchanged. A two-hour compute allocation
was used for that completed run. See the
[rerun report](../../experiment_log/2026-11-04/qwen3-peer-forward-step.md).

The first pilot asks whether downstream success can train useful questions and
responses between two symmetric Qwen3-1.7B peers. This is a feasibility study,
not evidence of superiority over a larger model. The user explicitly authorized
provisioning one kristine Spot v5e-8 and using it for up to 24 hours, including
remote tests, compilation, rollouts, updates, and evaluation. At most one TPU may
be active. No development host may execute an experiment.

## Conversation

Both peers use the same collaboration prompt and independent rank-16 LoRA
adapters on the same frozen pretrained checkpoint. A turn receives the complete
original problem, the current turn number, and only the preceding sender's
chosen message and context. There is no scratchpad or automatic transcript
sharing. A message can ask a specific subquestion or answer the partner. Either
peer can submit a final answer; submission ends that trajectory. An eight-message
limit, malformed action, or truncated action without a valid submission yields
zero terminal reward under strict scoring. Prompt tokens remain visible and have no token-level loss.

For the requested looser scoring, set `"answer_matching": "integer_search"` in
the next run's configuration. This searches only the final generated response
for the reference integer, including numeric JSON and free-form or truncated
outputs. It recognizes signed integers and comma-grouped thousands, while
avoiding partial matches such as `2` inside `20`, `-2`, or `2.5`. No LLM call is
used. The default `strict` mode preserves the completed pilots' scoring.

Message routing is unchanged: valid messages continue to the partner, and the
hidden reference answer never decides when to stop a conversation. The search
does not inspect prompts, previous turns, or reference solutions. This is an
answer-presence reward, not semantic verification: “not 42, but 99” still matches
the target 42. Train-tree credit, peer evaluation, and the single-agent baseline
use the same configured rule. Search-mode metrics report `answer_search_rate`
alongside `strict_pass_at_1`, and retain the raw output, routing failure reason,
selected score, strict score, and matching mode. Historical scores and configs
remain unchanged; a new run must opt in explicitly. Focused tests are added;
all 21 selected CPU-only protocol/lifecycle tests passed on the authorized remote VM.
See the [test record](../../experiment_log/2026-11-03/qwen3-peer-answer-search-tests.md).

## Tree and credit

At selected turns 0 and 1, sample four alternative actions from the exact same
state. For each candidate, independently complete two continuations under the
frozen collection policies. A terminal submission is scored directly once and
is not duplicated into fake independent observations. Branching is restricted
to one selected node per tree, so the pilot does not expand an exponential tree.
Turn-1 states come from a separately sampled prefix; selection does not inspect
the answer or future reward. Collection for an entire tree precedes any update.

Each candidate message receives its empirical downstream success rate Q. Its
weight is 2(Q - 0.5), retaining positive and negative credit for unanimous groups.
Continuation messages receive their terminal outcome; the shared prefix is not
repeatedly counted. Each distinct generated message contributes once to its own
agent's update. The initial objective uses a clipped token-ratio surrogate, a
frozen-base KL penalty, one update epoch, and gradient clipping. Sampling and
training both use the same temperature and full token support. This is a pilot
variant, not AT-GRPO and not a claimed unbiased estimate of causal usefulness.

Relevant precedents: [Optima](https://arxiv.org/abs/2410.08115) and
[MALT](https://arxiv.org/abs/2412.01928). This pilot differs by using symmetric
question/reply actions and on-policy gradient updates instead of their offline
or iterative preference-pair training.

## Data and evaluation

Use original MATH training questions whose final boxed answers are integers;
extract only that answer for the verifier. Never expose the reference solution
to either peer or use its reasoning as a training target. Record the filtering
and all exclusions, select a seeded disjoint validation split, and freeze data
and model revisions before download. Audit normalized question overlap against
AIME 2024 and exclude collisions from training/validation. This cannot establish
absence of contamination in the pretrained model.

Before/after evaluation uses the same 30 held-out AIME 2024 problems, seeds,
prompts, and limits, without any test-based checkpoint selection. Preserve raw
questions, outputs, token IDs, sampled token log probabilities, tree ancestry,
terminal rewards, and assigned credit in node-local artifacts and checksummed regional checkpoint evidence. Report pass@1 for one
unbranched conversation per problem, with exact counts; do not present training
tree success as pass@1. Also evaluate a single-agent baseline with the same
maximum generated-token budget and report actual tokens used. The 24-hour cap
may leave some phases incomplete; missing results must remain explicitly missing.

## Required remote gates

Before training: immutable source/config manifest; remote-only execution guard;
all protocol and reward tests; independent loss/gradient checks; pretrained
generation and prompt-mask checks; first backward pass; HBM and measured
throughput gates. Stop before training if the measured work cannot fit the
remaining budget and finalization reserve. No untested TPU backend is assumed
ready. The selected float32 runtime passed these gates; see the report for measured values.

Publish configurations, scalar metrics, execution status, and artifact manifests
to W&B. Save both adapters and optimizer states atomically, with checksums and
regional durable copies. Resume inputs verify every state/evidence hash, restore both
optimizer states and the next tree index, and preserve the original allocation deadline.
Completed evaluation phases are retained; an interrupted tree is repeated from its
last complete checkpoint using the same sampling seeds. Monitor the exact run through completion; never mutate
unrelated resources, jobs, or files.

The initial bfloat16 attempt had a maximum cached/replay log-probability discrepancy
of 1.284767 (mean 0.013022) on its first sampled tree. Its results are retained as a
failed attempt. The float32 retry keeps the same scientific settings, tests the
original gate without relaxing its threshold, and uses a private TPU diagnostic-log
directory. No other owner's log directory is modified.

## Progress-when-able rerun (completed)

`progress_when_able.json` selected a 32-tree run from fresh pretrained
weights. The prompt now says to continue solving whenever capable and asks a
question only when partner help would advance the solution. The original
problem and latest handoff remain the entire per-turn state; history does not
accumulate. Temperature remains 0.7. This run also activates the previously
requested `integer_search` scoring, reporting strict scores separately. Because
both the prompt and reward change, comparison with the completed forward-step
run is not a prompt-only ablation. All 30 remote tests and runtime gates passed. MATH answer-search stayed at
8/16; AIME stayed at 0/30. There was only one valid handoff across all training
trees, so this run did not demonstrate collaborative improvement. See the
[verified report](../../experiment_log/2026-11-03/qwen3-peer-progress-when-able.md).

## Pending prompt revision: one step per turn

The working prompt now requests exactly one concrete calculation, derivation,
or check per turn, followed by a handoff. It removes the instruction to keep
solving while capable. A final submission remains allowed when the answer is
ready. This revision has not been run; the completed progress-when-able results
and saved viewer prompts describe the earlier prompt unchanged.
