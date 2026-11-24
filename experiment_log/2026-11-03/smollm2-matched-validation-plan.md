# SmolLM2 GSM8K follow-up: matched prompt validation

## Run Summary

| Field | Value |
|---|---|
| Study ID | `smollm2-gsm8k-matched-validation-20261103` |
| Status | planned; additional experiment time not authorized |
| Started / finished | not recorded; no experiment launched |
| Baseline | [Completed 19-trial study](../2026-11-02/smollm2-gsm8k-research.md) |
| Effective run configuration / command / source commit | not recorded; runtime implementation and remote verification remain pending |
| Proposed protocol | [Reviewable JSON specification](smollm2-matched-validation-plan.json); not a launchable configuration |
| Proposed hardware | Existing single dedicated kristine Spot v5e-8 in `us-west4-a`; fresh health/occupancy required |
| Proposed additional budget | Four hours, including remote tests, compilation, all model computation and final evaluation; zero additional seconds currently authorized |
| Seeds | Original split 20261102; optimization 1234; training demonstrations 20261103; validation demonstrations 20261104 |

## Evidence and rationale

The earlier study ended model computation at 2026-11-03T01:29:00Z after about
3h48m. Its original four-hour deadline was 2026-11-03T01:40:51Z. The goal request
did not extend that authorization. This preparation reads saved results and code;
it performs no training, inference, tokenization, benchmark, or accelerator work.

The training implementation in the original `source/research.py` uses zero-shot
validation accuracy as its primary selection criterion. It records the epoch of
minimum NLL but only retains the state selected by accuracy. Trial 04 therefore
saved epoch 4 (validation NLL 0.6421886719537868), while its minimum NLL
0.6045382790792161 occurred at epoch 3. A proposed loss reference must reconstruct
that training-only epoch; the final full-data loss model has trained on the
validation examples and is ineligible for validation-based selection.

Inspection of all 19 saved trial summaries found seven non-dominated checkpoints
under existing zero-shot validation accuracy (higher) and NLL (lower). This is
descriptive analysis of saved metrics, not a new model evaluation:

| Saved checkpoint | Epoch | Strict correct / 747 | Validation NLL (nats/token) |
|---|---:|---:|---:|
| trial-02 | 5 | 47 | 0.6166174625333476 |
| trial-03 | 5 | 51 | 0.6313953449466887 |
| trial-04 | 4 | 66 | 0.6421886719537868 |
| loss-probe-01 | 6 | 76 | 0.7751583827381916 |
| trial-13 | 10 | 93 | 1.1625307081955532 |
| trial-10 | 11 | 95 | 1.300386004568308 |
| trial-08 | 11 | 100 | 1.4551265762959849 |

Source: the `trials` array in the completed study's
[scalar summary](../2026-11-02/smollm2-gsm8k-research-artifacts/report/summary.json).
The dominance rule requires no worse accuracy and NLL, with at least one strict
improvement. Five-shot validation metrics were not collected; the remaining
checkpoints cannot be excluded based on this two-dimensional frontier.

[The Flan Collection](https://arxiv.org/abs/2301.13688) reports benefits from
mixing zero-shot, few-shot and chain-of-thought prompt settings in its instruction
tuning experiments. This motivates a controlled prompt-mixing test, without
assuming its results transfer to SmolLM2-135M or GSM8K.
[Multi-Task Transfer Matters During Instruction-Tuning](https://aclanthology.org/2024.findings-acl.883/)
finds connections between instruction tuning, transfer and in-context learning.
Our observed format sensitivity is consistent with investigating these issues;
it does not prove catastrophic forgetting as a mechanism.

## Proposed configuration and changes

The linked JSON fixes model/data revisions, seeds, optimizer defaults, candidate
order, metric definitions, selection rules, budget gates and data boundaries.
Existing pinned model, tokenizer, GSM8K data and runtime are reused node-locally.

1. Measure both zero- and five-shot validation protocols for existing training-only
   checkpoints. Prefer the seven frontier checkpoints first, then the remainder
   within the measured time limit. Do not access the test split.
2. Reproduce the original loss recipe's epoch-3 reference on 6,726 optimization
   examples, retaining its state. Freeze both-format validation metrics before
   admitting new training candidates. Keep the original 747-example validation
   partition for comparability, acknowledging its repeated use.
3. Run a fixed six-candidate grid: learning rates 1e-4, 3e-4 and 1e-3, each with
   zero-shot-only or 50% five-shot training. Use global batch 32, maximum eight
   epochs, the original 12-epoch schedule horizon, and unchanged optimizer defaults.
   Every query is an optimization target once per epoch. Demonstrations come only
   from optimization rows, exclude the query and normalized duplicate questions,
   and receive zero loss weight. No external or synthetic data is introduced.
4. Score all 747 validation questions in both formats every completed epoch.
   Demonstrations are fixed per validation question and shared across candidates.
   Report each format's unweighted query-answer NLL and strict/flexible accuracy.
   Retain separate minimum-loss and balanced checkpoint states, rather than only
   recording an epoch number.
5. Before test access, lock the selected model, epoch count and full-data refit
   recipe. Refit from base on all 7,473 training examples and record fixed train/test
   loss curves without changing the locked stopping point. Final test evaluation
   uses the historical pinned zero/five-shot benchmark and verifies prompt hashes.

Relative to the previous study, the principal new variables are training prompt
mix and matched two-format validation. Longer prompt widths change cost; sequence
length, HBM headroom and throughput must be measured remotely before launch.
Never silently truncate a query, drop a demonstration, or reduce shot count.
Demonstration sampling must record its IDs and context-length rejection counts.

## Selection and budget

A balanced candidate must match or exceed the fixed reference's correct-answer
count in both formats, with NLL no more than 10% above that reference separately
in either format. Among eligible checkpoints, maximize the smaller correct-answer
count, then their mean; break ties by lower worst relative NLL and earlier epoch.
The 10% guard is a declared research choice, not an established optimal threshold.
Also preserve the validation Pareto frontier and minimum zero-shot NLL model.
If none improves, record that outcome and keep the existing models.

The requested four hours is a cap, not a promise that all candidates will fit.
Reserve one hour for checkpoint/export verification, the fixed refit and final
evaluation. Measure long-context throughput before admitting each candidate,
include compilation time, and use a hard deadline with checkpoint-safe stopping.
Never launch a new epoch or refit whose conservative completion estimate exceeds
the remaining window. Do not expand capacity, switch to a development TPU, or
re-provision without applicable authorization. Save partial progress honestly if
the measured workload cannot finish within the budget.

## Verification and publication

Before a launch, extend a new checksummed source copy and run focused tests only
on the dedicated authorized remote VM. Required checks cover split/demonstration
disjointness, no own-query demonstrations, zero-shot equivalence, demonstration
loss masking, target/EOS boundaries, complete query coverage, deterministic prompts,
NLL/accuracy selection, budget enforcement, checkpoint identities, and independent
long-context TPU/PyTorch parity. These new runtime tests have not been run.

Publish scalar train/validation/test loss curves, both prompt-format accuracies,
timings and public configurations to W&B. Reuse the corrected pagination verifier;
verify unique optimizer-step coverage and final scalar round trips. Keep raw
examples, source, credentials and weights out of W&B. The previously rejected broad
archive upload remains unperformed; this plan does not retry it. Checkpoint paths
and completion manifests must be fixed and recorded before the next launch.

## Results, limitations and next action

No new model result is claimed. Only the saved-metric frontier and this protocol
were produced. The previous test results have already been seen; later test
measurements are exploratory diagnostics, not a fresh blind estimate. Neither
test losses nor test accuracies may feed the search, candidate ranking or stopping.
Repeated validation selection also limits claims of generalization. A single
seed and one dataset cannot establish a globally optimal fine-tune.

Next action: obtain an additional compute budget for the concrete protocol above,
then implement and remotely verify its runner before starting any new model job.
No training or evaluation proceeds merely because time has elapsed while awaiting
that answer.

## Preparation verification — 2026-11-03T01:46Z

The JSON parsed successfully. Inspection checks confirmed zero authorized
additional seconds, one-TPU capacity, all six unique grid combinations, complete
partition counts, test exclusion during search, and the prohibition on broad raw
archive uploads. All seven frontier table rows exactly match the saved scalar
summary; every local report link resolves. No model or training library was
imported for these checks. No new experiment test or model computation ran.

The existing static report server is still live and returned HTTP 200 at
`http://localhost:8769/` when checked from its network namespace. An initial
sandbox-local connection failure did not indicate a stopped server; its original
process handle was verified live before the successful same-namespace check.
The original charts and report were preserved.

The additional-budget question is pending, with four hours recommended and a
two-hour alternative. The requested ntfy approval notification was delivered.
Elapsed waiting time does not authorize a launch. The broader improvement goal
remains active.

## Implementation preparation — 2026-11-03T01:58Z

Prepared a new [source directory](smollm2-matched-validation-source/README.md)
without staging or executing it remotely. It contains deterministic prompt
construction and selection, an auditor for the 19 saved training-only checkpoints,
an independent long-context TPU/PyTorch preflight, and a subprocess deadline
supervisor. Three required legacy source files are byte-identical copies from the
completed study. The previous run's files, checkpoints and reports remain intact.

Prompt construction now specifies a fixed 640-token query-answer reserve before
choosing demonstrations. This avoids a subtle label leak: the held-out answer
cannot alter demonstration selection through context-length resampling. Overflow
after prompt selection fails rather than substituting demonstrations. The protocol
JSON records this implementation detail and a maximum of 10,000 sampling attempts.

The auditor excludes full-data refits, validates checkpoint size/SHA-256/completion
markers, verifies model/data/split identities, and requires reproduction of the
original zero-shot checkpoint result. It also records the exact validation prompt
identity. Both NLL and accuracy selection compare only validation fields. Its
preflight covers short, median and longest sequences and times the largest prompt
shape. Successful runtime parity and live HBM checks are still required.

The supervisor verifies an explicit execution receipt and the exact node-local
source path/remote hostname before model imports. It reserves the declared final
hour and manages only its own child process group, with TERM/KILL grace before
the deadline. A real receipt has not been created. The budget question remains
pending, so no new experiment job or remote test was started.

[Static preparation evidence](smollm2-matched-validation-preparation-checks.json)
records syntax compilation of nine Python files, annotation checks on new
functions, unchanged legacy dependencies, and 31 authored test cases. None of the
tests has run; no model library was imported by the static inspection. Source
manifest SHA-256 is
`5da267057f8b0a96a175bbb66a0c5c8a1b467d0211b3970acd5a50cfcb0c115e`.

This is preparation progress, not a validated model result or a completed study.
The mixed-prompt training/refit controller and scalar publication integration
remain pending, alongside remote tests and the additional-budget answer. The next
experiment action is a fresh approved remote software/parity preflight; it must
not proceed while approval is absent.

## Goal blocker audit — 2026-11-03T01:59Z

The additional-budget question has remained unanswered across three consecutive
goal continuations: protocol preparation, source preparation, and this audit.
The prior continuation made concrete progress by writing the draft implementation
and static-check evidence. The blocking condition itself is unchanged: the
original compute window ended and zero additional seconds are authorized.

Rechecked all ten files in the source manifest, its recorded digest and the
protocol digest; all match. The 31 authored tests remain unexecuted, no additional
remote job is recorded, and no real execution receipt exists. These facts do not
prove model improvement or runtime correctness. They establish the current
preparation state only.

The next necessary step is authorized remote software and long-context runtime
verification. Further training integration must use that evidence rather than
assume the untested draft's throughput, memory use and numerical correctness.
The broader goal is therefore marked blocked pending the existing additional-
budget answer, not complete. No new permission question, provisioning, model
job, or archive upload was issued during this audit. The proposed four-hour
extension and two-hour alternative remain available in the existing question.
