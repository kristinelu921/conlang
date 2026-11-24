# SmolLM2 study: W&B history verification pagination failure

- UTC incident: 2026-11-03T01:20Z; study `smollm2-gsm8k-research-20261102`.
- Stage: post-training scalar monitoring/reconciliation, then final publication verification.
- Hardware: `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`,
  `he-vision-group`, `us-west4-a`, Spot v5litepod-8, one VM/eight TPU chips.
  TPU remained READY/HEALTHY. No training process was interrupted or relaunched.
- Original checker command: remote `wandb-runtime/bin/python verify_study_wandb.py`;
  exit 1 at `assert steps == list(range(1, expected_steps + 1)), directory.name`.
  The original finalizer later exited 1 in phase `publish` at the equivalent check.
  Original logs and exit markers are preserved on the node and in the final archive.
- Last confirmed training: accuracy refit 5,148 updates, loss refit 702 updates;
  both completed, with checksummed full optimizer checkpoints mirrored to
  `gs://kmh-gcp-us-west4/kristine/smollm2-gsm8k-research-20261102/`.
  All training/data/output paths were node-local in US-WEST4.
- Installed client: W&B 0.22.2. This was not a dependency import or training failure.

The API returned 5,153 rows for the accuracy refit's 5,148 updates at page size
1,000. Repeated records had identical internal `_step` IDs 1000, 2000, 3000,
4000 and 5000, and identical payloads. Every expected optimizer step was present.
The checker incorrectly treated pagination-boundary repeats as duplicate training
updates. This evidence rejects missing uploads and repeated optimizer execution
as explanations for the observed mismatch.

- [Initial diagnostic](../2026-11-02/smollm2-gsm8k-research-artifacts/wandb-history-diagnostic.log)
- [Internal-ID/payload proof](../2026-11-02/smollm2-gsm8k-research-artifacts/wandb-pagination-proof.log)
- [Five passing focused remote tests](../2026-11-02/smollm2-gsm8k-research-artifacts/wandb-pagination-tests.log)

The versioned fix removes only identical repetitions of the same internal record
ID. Conflicting payloads are rejected; distinct records with duplicate optimizer
steps and missing steps remain visible to strict coverage assertions. Reconciliation
then passed for all 21 completed runs: 80,186 optimizer updates and 241 full-split
loss records. The fix changed neither model weights nor experiment selection nor
uploaded training history. Final scalar publication is retried from versioned
source while preserving the original failed publication log and exit marker.
The recovered publisher completed at 2026-11-03T01:30:43.924918Z and verified
both final W&B summaries, their finished states, scalar values and update counts.

Verification source archive SHA-256:
`c8ea032c1cd71a4753c80ff0afa9ec0fd8ceab57f617e2009d8791c197cf3d39`.

Prevention: include W&B's internal `_step` identifier in paginated reads, collapse
only identical record payloads at overlapping page boundaries, then enforce
complete ordered optimizer-step coverage and compare full-split losses to the
immutable experiment events.
