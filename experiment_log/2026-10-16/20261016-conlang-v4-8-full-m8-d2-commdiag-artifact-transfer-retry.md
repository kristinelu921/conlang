# Message-8 diagnostic artifact-transfer retry

## Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m8-d2-commdiag` |
| Status | `resolved` |
| Failure scope | post-run artifact transport only; training and remote artifacts were unaffected |
| Error | intermittent SSH `message authentication code incorrect` during multi-file SCP |

The failed multi-file attempt copied complete `config.yml` and `metrics.jsonl` files, a 507904-byte partial `outputs.jsonl`, and no launcher log. The partial output hash did not match the remote canonical hash and was quarantined. An isolated output retry eventually returned exit zero; its 2807250-byte payload matched remote SHA-256 `262e4d272bdd6f0b519f5d0a25b2639b26c768e921c0363ef379895a212875f0`. The launcher log then transferred separately and matched its remote hash.

The canonical local archive contains only verified artifacts. The checkpoint remained on node-local storage and passed a CPU-only Flax restore at step 2000. This transport issue had no effect on experiment results.
