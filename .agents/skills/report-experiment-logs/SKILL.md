---
name: report-experiment-logs
description: Create, update, summarize, and compare reproducible ELF experiment reports under experiment_log, including configurations, changed hyperparameters, commands, seeds, metrics, statuses, and artifacts. Use when asked to report, document, inspect, summarize, present, or compare training or evaluation run logs and results. Do not use for changing application logging code.
---

# Report Experiment Logs

Produce concise experiment records that distinguish observed evidence from
interpretation and make runs reproducible.

## Gather Evidence

1. Locate the run's effective configuration, command-line overrides, console
   output, metric files or tracking URL, and checkpoint or artifact paths.
2. Capture the exact command, random seed, Git revision, timestamps, hardware,
   and run status when available.
3. Identify the baseline named by the user or run metadata. If no baseline is
   established, record `not specified`; do not choose one silently.
4. Do not rerun an experiment, contact an external service, or download an
   artifact unless the user requests it or it is necessary for the task.
5. Mark unavailable facts as `not recorded`. Never reconstruct or infer missing
   configuration values, commands, metrics, or completion states.
6. Redact credentials, access tokens, and other secrets while preserving the
   names of relevant configuration fields.

## Write The Report

1. Use `assets/experiment-log-template.md` as the report structure. Copy and
   adapt the template; do not modify the template for an individual run.
2. Store reports at `experiment_log/YYYY-MM-DD/<run-name>.md`, using the run's
   start date when known and the current date otherwise.
3. Use a lowercase, hyphen-separated filename. If the path already belongs to a
   different run, add a stable identifier or numeric suffix instead of
   overwriting it.
4. Record the complete effective configuration in the report, including values
   produced by defaults and command-line overrides. Also link the source config
   when one exists.
5. Diff changed hyperparameters against the explicit baseline. Include only
   confirmed differences and preserve value types and units.
6. Record each metric with its dataset or split, step or epoch, optimization
   direction, and source. Preserve the precision emitted by the source.
7. Record artifact paths or URLs and whether each artifact was verified to
   exist. Use paths relative to the repository when practical.
8. For failed or interrupted runs, include the last known step, the concise
   failure message, and the path to detailed output. Do not paste an entire log.
9. Preserve existing records. Append a clearly timestamped update when adding
   information to a previously reported run.

Use only these status values: `planned`, `running`, `completed`, `failed`, or
`interrupted`.

## Compare Runs

- Verify that compared metrics use the same dataset, split, evaluation method,
  and units. Flag differences before drawing conclusions.
- Use a table with one row per run and one column per important changed
  hyperparameter or metric.
- Bold the best metric only when its optimization direction is known and the
  values are directly comparable.
- State the baseline, the absolute change, and the relative change when useful.
  Do not claim causality from an uncontrolled comparison.
- Separate factual observations from hypotheses or recommendations.

## Present Results

When the user asks to see data, create labeled plots and serve them through a
local report. Include axis units, legends, run names, and metric direction where
applicable. Provide the localhost URL and keep the server running. Link the
generated report or plots from the experiment record.

## Check The Record

Before finishing:

1. Confirm the report contains the effective configuration, exact command,
   seed, status, changed hyperparameters, results, and artifacts, or explicitly
   marks each missing field as `not recorded`.
2. Confirm every reported value is traceable to a cited local file, console
   output, artifact, or tracking URL.
3. Confirm no prior record or secret was overwritten or exposed.
4. Report the created or updated paths and any unresolved missing evidence.
