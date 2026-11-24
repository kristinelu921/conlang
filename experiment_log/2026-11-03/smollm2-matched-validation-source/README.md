# Matched GSM8K validation preparation

Status: draft implementation; no new experiment has run and no additional compute
budget is approved. Source preparation and syntax inspection do not establish TPU
correctness. Do not invoke the auditor until the runtime gates below pass.

`matched_protocol.py` defines deterministic zero/five-shot prompts, demonstration
exclusion, query-only loss masks, exact-size training mixtures, fixed validation
identities, the balanced selection criterion, and persisted budget arithmetic.
The demonstration pool is validated/deduplicated once rather than rescanned for
every training query. Held-out answers cannot influence prompt selection: a fixed
640-token reserve is allocated before their length is checked. This reserve is
supported by the prior run's successful untruncated 640-token encoding of every
original training example, including the validation partition.

`audit_checkpoints.py` reuses the established TPU model and scoring implementation
to audit all 19 saved training-only checkpoints. It verifies source/checkpoint
hashes, original partition identity, fixed prompts, completion markers and source
epochs. It measures answer-token NLL and strict/flexible accuracy in both formats
and requires reproduction of each checkpoint's original zero-shot result. Raw
generations remain in the new remote output directory. This command has no cloud
upload or W&B publication path; a later scalar publisher must verify its completed
outputs and exclude raw prompt/generation records.

`legacy/` contains byte-for-byte copies of exactly three source dependencies from
the previous immutable source snapshot. The inherited `research.py` CLI is not
used. Its `StudyEngine` constructor reads only the original training split and its
loss/generation kernels are reused by the auditor. Nothing in the old snapshot is
modified. These legacy files are not a new training launch configuration.

`test_matched_protocol.py`, `test_audit_guards.py` and
`test_deadline_supervisor.py` contain focused software tests. They have not yet
been executed; run them on the authorized dedicated
remote VM after approval, with the time included in the extended budget:

```bash
cd /home/kristinelu/smollm2-gsm8k/matched-validation-20261103/source
/home/kristinelu/smollm2-gsm8k/runtime/venv/bin/python -m unittest -v test_matched_protocol test_audit_guards test_deadline_supervisor
```

Before any model import, the auditor requires an execution receipt with explicit
approval, the approved additional seconds, the fixed first-preflight start time,
the exact TPU identity, and a hostname established by fresh remote preflight.
The source path must be the exact node-local directory above. No execution
receipt exists while approval is pending; unit-test fixtures are synthetic and
cannot authorize a run. Missing receipts, expired time, a different host, and
changed source files cause failure.

The future execution receipt fields are `approval_status`, `study_id`, `hardware`
(project/zone/name/type), `maximum_concurrent_tpus`, `verified_hostname`, `spot`,
`development_machine`, `study_started_unix_seconds`, and `authorized_seconds`.
Record real user approval and real preflight evidence before creating it; do not
infer either from defaults. Its original clock must survive retries.

The auditor's `--preflight-only` mode compares independent PyTorch and TPU NLL and
logits at short, median and longest sequence lengths for each prompt format.
It also measures compilation and warm generation on the 64 longest prompts, so
the measured shape covers the full validation set's largest prompt. These are
draft checks, not results: they have not run. Verify live HBM headroom separately
alongside their results before admitting the full audit.

`run_remote_phase.py` wraps either preflight or audit with an external wall-time
deadline. It records the exact command/PID/process group, signals only its own
child process group, allows TERM then KILL grace within a ten-second margin, and
requires both successful exit and a completion sentinel. Its tests mock the
process handles; they do not start jobs. A new attempt must use a new output and
control identity, preserving interrupted artifacts. The phase's deadline is the
original study start plus the authorized extension minus the finalization reserve;
retries never reset it. This wrapper is also untested and must pass remote tests.

After real approval, staging, fresh health/occupancy checks and software tests,
the proposed preflight command is:

```bash
/home/kristinelu/smollm2-gsm8k/runtime/venv/bin/python run_remote_phase.py --phase preflight --protocol ../research_protocol.json --execution-receipt ../execution-receipt.json --attempt-id 001
```

Those runtime JSON files do not yet exist. The first records the frozen protocol;
the second must record actual approval and actual remote identity evidence.
Passing preflight and HBM checks allows an audit under the same receipt/clock,
using `--phase audit`. Neither command has been launched.

The mixed-prompt training runner, separate minimum-NLL/balanced state retention,
fixed loss-reference recreation, final selection/refit controller and scalar W&B
bridge remain to be integrated and remotely tested. This draft does not claim the
six-recipe study is ready to launch. The JSON protocol in the parent directory
remains the reviewable research specification, not an executable training config.
