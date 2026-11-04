# Answer-search scoring: remote CPU validation

## Run Summary

- Status: completed; code-validation job only, no model experiment.
- User authorization: CPU-only tests for up to five minutes on the existing exact kristine VM, with no inference, training, accelerator use, or provisioning.
- Started: 2026-11-03T21:28:37.737769-05:00 (America/New_York).
- Finished: 2026-11-03T21:28:40.839825-05:00 (America/New_York).
- Resource: `kmh-tpuvm-v4-8-spot-kristine-03-22-29-24-N-us-central2-b`, project `he-vision-group`, zone `us-central2-b`, Spot v4-8, one VM. Accelerator was not used.
- Remote hostname: `t1v-n-0de15714-w-0`; development/control hostname: `t1v-n-21fe7514-w-0`.
- Before/after dedicated occupancy checks: AVAILABLE. Direct API: READY/HEALTHY.
- Source baseline: `9cd6c6e94fda36c8378ea8ec98bccce5dee49386` plus the answer-search working-tree changes. These changes have not been pushed. Exact tested source hashes are preserved in the evidence JSON.

## Configuration and Change

Added optional `answer_matching: integer_search` in `protocol.py` and applied it to all three tree terminal paths, peer evaluation, and the pretrained single-agent evaluator in `run.py`. The strict default preserves completed-run scoring. No completed run configuration or score was changed.

The new rule searches only the final generated response for the reference integer. It accepts prose, numeric JSON, signed integers, leading zeroes, and comma-grouped thousands; it rejects embedded numeric substrings such as 2 in 20, -2, 2.5, or 2,000. Truncated final text can receive search credit if the target integer already appears. Valid intermediate messages still go to the partner: hidden ground truth never affects message routing or early stopping. The search never reads the prompt, previous messages, or reference solution.

This is an answer-presence rule. An incidental or negated occurrence can still receive credit (e.g. target 42 in “not 42, but 99”). Search-mode evaluation reports `answer_search_rate`, while preserving `strict_correct`, `strict_pass_at_1`, the raw generation, routing reason, and matching mode. It does not present the loose score as strict benchmark pass@1.

Tests use scripted generations and mocked dependencies. The protocol fixture reads the committed `experiments/peer_tree_rl/pilot.json` (seed 20261103); evaluation fixtures use seed 17. Both strict and integer_search modes are covered. No dataset, checkpoint, learned policy, or random model sampling is used.

## Exact Command and Snapshot

Executed on the remote VM:

```sh
python3 -I -S /home/kristinelu/peer-tree/answer-search-tests-20261103-01/qwen-peer-answer-search-remote-tests.py > /home/kristinelu/peer-tree/answer-search-tests-20261103-01/console.log 2>&1
```

The preserved runner installs uv 0.8.3 and CPython 3.10.18 only for this test task, verifies all five source-file hashes, and executes its embedded unittest runner using `python3.10 -I -S` from the node-local source snapshot. It applies one 300-second deadline across setup and testing. Environment: `JAX_PLATFORMS=cpu`, empty `CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; uv installation, interpreter, and cache live below the task directory. A module import guard prohibits JAX, JAXLIB, libtpu, Torch, TensorFlow, Transformers, and datasets; the baseline test supplies a temporary fake JAX namespace and removes it afterward.

- Source archive: `/tmp/qwen-peer-answer-search-tests-source.tar.gz` and the same basename in the remote task directory.
- Source archive SHA-256: `17ab374c9e304791937014414da5719b2a37e5ec848088f29de4a616124256a8`.
- Runner SHA-256: `5669370f3b5576416c901dd90cb808b7eba94ef913ecabad6cc513410a167abe`.
- All current implementation/test bytes still match the remotely verified archive after validation. Only README and this test report were updated afterward.

## Results

- **21 tests passed, zero failures, zero errors, zero skips.**
- Scope: all 16 protocol tests and five lifecycle tests covering peer and single-agent scoring, intermediate-answer isolation, strict score retention, failure artifacts, checkpoint integrity, and remote/budget guards.
- Console-reported test execution: 0.022 seconds.
- Loader plus tests: 0.18232732000069518 seconds.
- Entire remote bootstrap and test job: 3.1048882009999943 seconds, below the 300-second authorization.
- Model/accelerator libraries imported: false. Training steps: none. W&B run/checkpoints: none.
- No benchmark re-evaluation was performed. The unchanged model-backend and dataset-loading tests were not rerun; this focused job deliberately has no ML dependencies.

## Preflight and Artifacts

The VM's system Python is 3.8.10, incompatible with the existing test/source syntax. No tests were attempted with it. A private pinned CPython 3.10.18 resolved that preflight dependency without installing the ELF or TPU runtime. The existing VM remains intact; no provisioning, deletion, reboot, process termination, or changes to another user's files were performed.

- Structured evidence: `qwen3-peer-answer-search-tests-evidence.json` beside this report.
- Remote console/result files: `/home/kristinelu/peer-tree/answer-search-tests-20261103-01/console.log`, `test-result.json`, `execution-result.json`.
- Local verified result copies: `/tmp/qwen-peer-answer-search-test-results/`.
- Console SHA-256: `1634a5dc9b893f79ef6c26319abe0cc9706d6513ebce1a06ddd677042fa93e68`.
- Availability and occupancy receipts: `/tmp/qwen-peer-answer-search-approved-availability.log`, `/tmp/qwen-peer-answer-search-final-occupancy.log`, `/tmp/qwen-peer-answer-search-post-test-occupancy.log`.

## Limitations

The new scoring mode is validated on scripted fixtures, not a new training run. A future run must opt in with `answer_matching: integer_search` in its own committed configuration and report loose and strict metrics separately. The answer-search code, focused tests, and README are working-tree changes; historical experiment results remain untouched.

## Bootstrap cleanup

uv also created `/home/kristinelu/.local/bin/python3.10` as a symlink to this task's private interpreter. After verifying its exact target, current-user ownership, and modification time within this job, removed only that new symlink. The task-local interpreter and evidence remain available. Cleanup receipt: `/tmp/qwen-peer-answer-search-python-link-cleanup.log` and remote `bootstrap-link-cleanup.json`.
