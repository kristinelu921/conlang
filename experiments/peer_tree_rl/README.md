# Two-peer tree RL pilot

Status: implementation under remote validation. Twenty protocol, lifecycle, and
independent numerical tests passed on the authorized TPU VM's CPUs. Full-size
model checks, training, and evaluation have not started.

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
zero terminal reward. Prompt tokens remain visible and have no token-level loss.

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
ready. The runtime is implemented; full-size remote verification is still pending.

Publish configurations, scalar metrics, execution status, and artifact manifests
to W&B. Save both adapters and optimizer states atomically, with checksums and
regional durable copies. Resume inputs verify every state/evidence hash, restore both
optimizer states and the next tree index, and preserve the original allocation deadline.
Completed evaluation phases are retained; an interrupted tree is repeated from its
last complete checkpoint using the same sampling seeds. Monitor the exact run through completion; never mutate
unrelated resources, jobs, or files.
