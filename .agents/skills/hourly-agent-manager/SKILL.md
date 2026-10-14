---
name: hourly-agent-manager
description: Reconcile ELF experiments from authenticated Weights & Biases runs and local evidence into complete, reproducible reports under experiment_log. Use for hourly or recurring experiment logging, finding unreported W&B runs, filling missing report fields, updating running-run status and metrics, or producing a clean findings summary. Coordinate focused subagents when useful, follow the repository report-experiment-logs skill for every report write, keep W&B access read-only, and never overwrite prior experiment records.
---

# Hourly Experiment Log Manager

Reconcile one invocation of experiment state. Treat the external scheduler as responsible for cadence. Load and follow the repository's `report-experiment-logs` skill before creating or updating any experiment report.

## Establish Scope

Use the authenticated W&B account and these repository defaults unless the user overrides them:

- Entity: `kaynelu921-massachusetts-institute-of-technology`
- Primary project: `elf`
- Report root: `experiment_log/`

Inspect all accessible W&B projects under the entity and default to runs created by the authenticated W&B user. Include every currently running run and every historical run missing from `experiment_log/`. Do not assume that a W&B run is an ELF experiment solely because it is accessible; use its project, configuration, tags, program path, Git metadata, and local evidence to determine relevance.

W&B access is read-only. Never edit, delete, stop, resume, tag, or annotate a W&B run. Never download artifacts unless the user explicitly requests it.

## Inventory W&B

Run the bundled inventory utility from the repository root:

```bash
python .agents/skills/hourly-agent-manager/scripts/inventory_wandb_runs.py \
  --entity kaynelu921-massachusetts-institute-of-technology \
  --output /tmp/elf-wandb-run-inventory.json
```

The utility uses the current W&B login, enumerates projects, filters to the authenticated username, redacts secret-like fields, and writes structured run metadata, configuration, summary metrics, and system metadata. Pass `--project` to restrict a run only when the user requests narrower scope. Pass `--all-users` only when the user explicitly asks to include collaborators' runs.

If W&B authentication, network access, or entity permissions fail, continue reconciling local evidence and clearly report the inaccessible scope. Never request or print an API key. Do not describe the inventory as complete when any project could not be queried.

## Inventory Local Evidence

Inspect:

- existing files under `experiment_log/`
- `wandb-run-id` files and local W&B metadata when present
- effective configuration files and command-line overrides
- console output and scheduler logs
- checkpoint and artifact paths
- Git revision and working-tree state recorded by the run

Preserve unrelated user changes. Do not rerun experiments merely to recover missing evidence.

## Reconcile Runs

Match records by stable W&B run ID first, canonical run URL second, and exact local `wandb-run-id` third. Treat display names as hints only. Never merge runs solely because their names are similar.

For each relevant run:

1. Locate its existing report by run ID or URL.
2. Create a report from the `report-experiment-logs` template when none exists.
3. Fill missing fields from traceable W&B or local evidence.
4. Preserve confirmed existing values; record conflicts instead of silently replacing them.
5. Append a timestamped update whenever an existing report gains information.
6. Map W&B states as follows:
   - `running` or `preempting` -> `running`
   - `finished` -> `completed`
   - `crashed` -> `failed`
   - `killed` or `preempted` -> `interrupted`
   - `pending` -> `planned`
7. Mark unavailable required fields as `not recorded`; never infer commands, seeds, baselines, completion, or metric direction.
8. Preserve metric precision and cite the W&B run URL or local source for every value.

Treat configuration from W&B as observed run configuration, but do not call it the complete effective configuration when the training entrypoint did not upload defaults. Cross-check local configs and explicitly identify remaining gaps.

## Delegate Focused Checks

Use subagents only when independent work will materially reduce latency or context noise. Good assignments include W&B inventory review, local artifact discovery, and report completeness auditing. Give each subagent exact scope, read/write authority, required evidence, output format, and stopping condition.

Keep report writes in the manager thread unless files are partitioned so that only one agent owns each report. Do not allow overlapping writes. Wait for required results and send focused follow-ups for missing evidence or conflicting run identity.

Subagents belong to the current invocation. Do not claim access to unrelated Codex chats. Recover continuity on later scheduled invocations from W&B and `experiment_log/`.

## Finish With Clean Findings

Keep the template's `Observations`, `Interpretation`, and `Updates` sections at the bottom of every touched experiment report. Add only concise, relevant findings:

- status transitions and last known progress
- new or changed directly comparable metrics
- failures, stalls, or missing evidence requiring attention
- verified artifact availability
- clearly labeled limitations or recommendations

Do not repeat the full configuration or metric tables in the findings. Separate observations from interpretation and avoid causal claims from uncontrolled comparisons.

Return one consolidated reconciliation summary containing:

- W&B entity, authenticated username, projects queried, and query time
- counts of discovered, created, updated, unchanged, running, completed, failed, and inaccessible runs
- created or updated report paths
- the most relevant findings across experiments
- unresolved missing evidence and recommended next action

If nothing changed, return an explicit no-change report. Do not create a new report merely to record that no experiments changed.

## Scheduled Invocation

Use a durable prompt such as:

```text
Use $hourly-agent-manager to reconcile my W&B experiments with experiment_log.
Inspect all accessible projects under my configured entity, ensure every relevant
run is reported through $report-experiment-logs, fill traceable missing fields,
append clean findings at the bottom, and return one consolidated summary. Keep
W&B read-only and do not rerun experiments.
```
