# Query-only receiver preflight runtime-probe transport attempt 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` |
| Status | `recovered pre-launch; no trainer started` |
| Started | `2026-10-16T18:54Z` |
| Finished | `2026-10-16T19:07Z` |
| Retry | preflight transport attempt 1; the scientific run identity was not consumed because Python, W&B, and the output directory were never created |
| Git revision | staged source closure `c00dd35e56504a506b9acadb3ea9bc69f391bfc4` |
| Pull request | `https://github.com/kristinelu921/conlang/pull/1` |
| Primary configuration | `configs/conlang-jax-v4-8-full-m4-d2-qquery-msgkv-commdiag.yaml` |
| TPU | `kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b`, `us-central2-b`, Spot `v4-8`, topology `2x2x1`, one host and four TPU devices |
| TPU state | direct API `READY`, `HEALTHY`; exact occupancy helper `AVAILABLE` before source staging |
| Stage reached | source staged and verified; final runtime-readiness inventory probe |
| Last metric step | `not applicable; step 1 did not start` |
| Checkpoint | `none` |

## Failed Preflight Command

```bash
PATH=/snap/google-cloud-cli/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
ELF_TPU_INITIALIZE_MISSING=0 \
ELF_TPU_EXACT_CANDIDATES=kmh-tpuvm-v4-8-spot-kristine-14-23-12-28-N-us-central2-b,us-central2-b \
bash tpu_scripts/check_tpus.sh he-vision-group
```

The command exited `0` and included the selected TPU in its final `AVAILABLE`
table, but its selected-row runtime status was `UNKNOWN` with detail
`ELF runtime probe failed`. The launch stayed blocked because `UNKNOWN` is not
launch-ready.

## Evidence and Diagnosis

- The immutable source had already been copied once to node-local storage. Local and remote trees both contained 60 files and 454098 bytes with relative SHA-256-manifest digest `e218b4847b5bc6fd643a4edec1acd5dc278fe84933a606db0e67d0c504d9ef98`; the selected config hash was `dcdc5b43c04b5665e905042fa8ba8bbdb4ba33633ba7d28f538a36d7077b25f0` and no cache files were present.
- Source, dataset, runtime, output, and logs were node-local under `/home/kristinelu`. The regional dataset manifest hash was `a839b6ac98ba33a008caaebdde40dea7bc1e639106e93aab25ab2af3c29588e8`; all six gzip shards passed integrity checking. The output directory remained absent.
- The standard remote preflight reported 85 GiB free and an idle device. No trainer, W&B run, output directory, or checkpoint was created.
- An exact worker probe using the same runtime requirements passed: the `READY` marker existed; CPU imports of `datasets`, `einops`, `flax`, `huggingface_hub`, JAX, MAUVE, NumPy, Optax, SacreBLEU, Torch, tqdm, Transformers, W&B, and YAML succeeded; Optax exposed Muon; and W&B authentication was present. JAX was `0.6.2`.
- Repeating the official inventory made all otherwise available candidates report the same runtime-probe transport failure, while the selected worker's isolated probe continued to succeed. This rejects a missing-package, missing-marker, Muon, or W&B-authentication cause.
- The lower-level cause of the inventory's failed third SSH reconnect is unconfirmed. It is treated as a control-plane transport failure, not as TPU loss or runtime incompatibility.

## Recovery

The source snapshot and verified node-local data were preserved. No process was
signaled, no lock or output was removed, and no TPU was rebooted. The scientific
run remained eligible for its original identity because the launch fan-out
never opened. A fresh direct `READY`/`HEALTHY` description, a final `AVAILABLE`
selection, an exact successful runtime-readiness probe, and fresh occupancy plus
canonical lockfile gates all passed before the subsequent successful launch.

## Artifacts

| Artifact | Path | State |
|---|---|---|
| Local immutable snapshot | `/kmh-nfs-ssd-us-mount/staging/kristine/elf/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag-20261016T185253.120559935Z` | verified |
| TPU-local source | `/home/kristinelu/elf-runs/source/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` | verified |
| TPU-local output | `/home/kristinelu/elf-runs/20261016-conlang-v4-8-full-m4-d2-qquery-msgkv-commdiag` | absent |
| Detailed command output | controller tool transcript | retained in session; no separate file |
