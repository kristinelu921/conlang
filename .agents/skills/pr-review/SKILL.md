---
name: pr-review
description: "Coordinate two-pass reviews of pull requests and proposed PR changes: one correctness pass followed by one cleanup pass. Use when Codex creates, prepares, updates, finalizes, or reviews a PR in this repository."
---

# PR Review

Review code-bearing PRs with exactly two sequential, read-only critic passes:
one correctness pass, then one cleanup pass. Make the coordinating agent the
sole writer. Do not add separate type, design, test, research, or OOD critics;
those concerns belong inside the two defined passes.

## Select the Operating Mode

- Use review-only when the user asks for review, explanation, or findings.
  Inspect and report without changing files, commits, comments, or remote state.
- Use review-and-fix when the user asks to fix or improve the PR, or when Codex
  is creating, preparing, updating, or finalizing it. Let only the coordinator
  implement confirmed in-scope findings.
- Never post comments, approve, merge, push, or otherwise mutate remote PR
  state unless the user authorized that action.

## Freeze the Review Target

Before the first pass:

1. Read all applicable repository instructions.
2. Resolve the base SHA, head SHA, merge base, changed-file list, and complete
   accumulated diff. Record the PR number when one exists.
3. Separate committed PR changes from unrelated working-tree changes and
   preserve all user work.
4. Identify affected public contracts, configuration, callers, producers, and
   relevant tests.
5. Give the critic the immutable base and head with read-only authority.

Inspect full touched functions and relevant unchanged producers, consumers,
configuration, and tests. Report issues introduced by the PR, exposed by it, or
required to make it safe. Label unrelated pre-existing issues instead of
broadening the change.

## Classify Code Items

Inspect content and behavioral effect; do not classify by filename or extension
alone. Code-bearing items include source, tests, executable fixtures or
notebooks, scripts, runtime configuration, feature flags, schemas, migrations,
CI and infrastructure definitions, build and packaging files, dependency
manifests, lockfiles, and generated artifacts that affect behavior.

Treat a PR as no-code only when every changed item is prose documentation,
static media, non-executable release notes, or formatting-only metadata with no
behavioral effect. When uncertain, treat it as code-bearing.

For a verified no-code diff, skip both critic passes, record
SKIPPED_NO_CODE_ITEMS with the rationale, and run relevant lightweight checks
such as formatting, link validation, rendering, or generated-file integrity.

## Run Exactly Two Passes

Never run these passes concurrently. Do not rerun either critic by default.
There are exactly two critic passes total.

### Pass 1: Correctness

Give one read-only correctness critic the immutable raw diff. Do not provide the
coordinator's suspicions or expected answer. Require the critic to:

- Trace behavior through producers, concrete runtime types and shapes, callers,
  configuration defaults and overrides, deserialization, and public inputs.
- Inspect invariants, state ownership, side effects, lifecycle, API
  compatibility, error propagation, partial failure, security boundaries,
  retries, idempotency, concurrency, resource cleanup, and performance.
- Inspect control flow, conversions, fallbacks, array shapes and dtypes, RNG
  ownership, numerical stability, checkpoint semantics, collectives, process
  roles, and multi-host assumptions when relevant.
- Verify that tests reach the intended behavior, use meaningful non-vacuous
  assertions, and would fail for the defect they claim to cover.
- Prefer an oracle independent of the implementation. Consider boundary,
  invalid-input, alternate-configuration, shape, dtype, randomized,
  differential, failure-injection, concurrency, and out-of-distribution cases.
- For scientific tests, inspect baseline comparability, uncertainty, leakage,
  metric validity, confounders, and reproducibility.
- Use public primary sources only when they materially resolve a correctness
  question. Never expose private code, identifiers, data, or credentials.

In review-and-fix mode, validate the findings, establish baseline tests when
needed, apply one cohesive correctness fix batch, and run focused tests. Freeze
the updated raw diff before pass 2. In review-only mode, record the validated
findings without editing and continue to pass 2 on the same immutable diff.

### Pass 2: Cleanup

Give one independent read-only cleanup critic the current raw diff after the
correctness pass and any accepted fixes. Do not reveal the correctness ledger or
the coordinator's expected answer. Require the critic to:

- Find dead code, duplication, redundant state, unnecessary dependencies,
  avoidable nesting, stale references, unrelated files, and multiple sources of
  truth.
- Challenge wrappers, helpers, classes, and flags that do not enforce a useful
  invariant, protocol, lifecycle, public API, compatibility policy, or domain
  boundary.
- Inspect responsibility boundaries, dependency direction, ownership,
  interfaces, configuration design, names, docstrings, comments, and control
  flow clarity.
- Prefer the smallest clear implementation, not the fewest physical lines.
  Preserve useful abstractions and do not compress readable code into code-golf
  expressions.
- Verify that runtime YAMLs contain only intentional overrides and the smallest
  reproducible configuration closure. Reject retry, smoke, or abandoned
  experiment manifests unless the PR intentionally supports them as durable
  entry points.
- Remove test duplication and implementation-detail assertions without
  weakening observable behavior coverage.

In review-and-fix mode, validate the findings, apply one cohesive cleanup batch,
and run focused tests plus the relevant broader suite. In review-only mode,
record the validated findings without editing.

Do not run a third critic pass. If cleanup exposes a correctness problem, the
coordinator must verify it directly, fix it when authorized, and run the
relevant tests without spawning another critic. If the final change is too
large or uncertain to verify safely under this two-pass limit, report NOT_READY
with the exact unresolved risk.

Use prompts shaped like:

    Review base <base-sha> through head <head-sha> in read-only mode. Inspect
    the complete diff and relevant unchanged producers, callers,
    configuration, and tests. Perform the <correctness|cleanup> pass. Return
    evidence-backed findings in the required format, or
    NO_CONFIRMED_FINDINGS. Do not edit files or remote state.

## Apply the Review Rubric

### Concrete Types and Boundaries

- Trace a value through its producer implementation and annotation before
  criticizing type-dependent code.
- Determine the concrete runtime type or shape supported by the producer. A
  type fixed for one experiment is not automatically a repository-wide
  invariant.
- When one type is established, prefer one assertion or validation at the true
  boundary and direct use afterward.
- Reject speculative fallback branches. Preserve genuine polymorphism and
  configuration-dependent variants.

### Compactness and Clarity

- Inline two- or three-line helpers when that neither duplicates logic nor
  hides intent.
- Keep helpers that remove meaningful duplication, isolate complexity, define a
  boundary, or provide a useful domain name.
- Use descriptive, fully spelled-out names. Permit standard domain
  abbreviations.
- Give non-trivial functions a concise purpose docstring.
- Rewrite unclear code before requesting comments. Add comments only for
  non-obvious reasons, constraints, paper discrepancies, or numerical details.
- Prefer one clear success path with validation at boundaries.

### Verification

- Require a focused test for each behavior change or bug fix.
- During a refactor, establish current behavior before editing and rerun the
  same tests afterward.
- Prefer observable behavior over implementation details. Add integration
  coverage when a change crosses module or system boundaries.

## Require Evidence-Backed Findings

Each finding must contain:

- ID and concise title.
- Severity: Critical, Major, Moderate, or Minor.
- Confidence: High or Medium.
- Status: Open, Fixed, Rejected, or Deferred.
- Location: file, line, symbol, and reviewed head SHA.
- Evidence: concrete code, configuration, producer, caller, or runtime
  contract.
- Producer trace for every type or shape claim.
- Problem and observable impact.
- Smallest clean in-scope fix.
- Focused verification that proves resolution.

Use Critical for data loss, security exposure, invalid results, or guaranteed
production failure. Use Major for likely correctness failures, broken
contracts, or serious regressions. Use Moderate for concrete maintainability,
reliability, or performance costs that should normally be fixed in scope. Use
Minor for bounded polish with a concrete benefit.

Return NO_CONFIRMED_FINDINGS when appropriate. Keep preferences and
nonessential cleanup optional. Reject vague claims such as "could be cleaner."

## Synthesize and Gate Readiness

Maintain one coordinator-owned ledger with exactly two rows: correctness and
cleanup. Record each pass's reviewed head, findings, resolutions, and
verification.

1. Deduplicate findings by root cause and preserve the strongest evidence.
2. Verify every Critical and Major finding directly in the repository.
3. Reject unsupported type assumptions, code-golf compression, blanket comment
   requests, flag proliferation, and unrelated architectural churn.
4. Keep critics read-only and do not allow overlapping agent writes.
5. Run narrow tests after behaviorally meaningful fixes and the relevant
   broader suite after cleanup.
6. Perform a final coordinator-owned diff and scope check. This is verification,
   not a third critic pass.

Mark an authorized PR READY only when both passes are current for their recorded
heads, no confirmed Critical or Major finding remains, no newly introduced
Moderate finding remains, accepted findings are Fixed or explicitly Deferred
with rationale, required tests pass or unavailable tests are recorded, and all
edits remain within the PR's intended scope.

In review-only mode or without remote-write authorization, report
READY_RECOMMENDED instead of changing PR state. If either pass is missing,
blocked, or has unresolved material findings, leave the PR draft and report
NOT_READY with the exact gate.

Return findings first, ordered by severity. Then summarize fixes, tests,
deferred items, optional refinements, and the final base and head SHAs. Do not
claim a clean review when material findings remain.
