# dLLM progressive teacher-weighted 32→16, 1,600-update evaluation

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-progressive-teacher-weighted-1600-20261119-evaluation` |
| Status | `running` |
| Started | `2026-11-19 18:49 UTC` |
| Finished | not recorded |
| Git revision | `35a1613e32afe21316ff2f00cc1a2d1593501de9` |
| Hardware | `kmh-tpuvm-v5e-16-kaiminghe-434bf6`, `us-west4-a`, worker 0, four local v5e devices |
| Checkpoint | update 1,600, SHA-256 `54553fbf494bbed7198392dd99f96e921ddafe4402cbcd996f4f362eb42092bf` |
| Command | `python -m reproduction.evaluate_tpu --data <node-local-data> --checkpoint <verified-1600-update-checkpoint> --output <node-local-output> --steps 16 --draws 32 --seed 42 --context 512 --new-tokens 128` |

## Protocol

- EvalPlus `0.3.1`.
- HumanEval+ v0.1.10: 164 tasks, decompressed SHA-256 `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- MBPP+ v0.2.0: 378 tasks, decompressed SHA-256 `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`.
- Deterministic 16-step confidence-remasking sampler, 128 generated tokens, 512-token context.
- Rolling-reference pseudo-perplexity and the 32-draw conditional diffusion NLL bound remain separate metrics.
- Training and evaluation seed: `42`.

## Capacity and Ownership

- The required filtered audit found no usable region-matched `kristine` TPU; its same-region candidate had collaborator-owned, non-writable TPU logs.
- The official wide read-only table listed several same-region candidates. Exact occupancy checks rejected stale entries that had acquired foreign jobs; those jobs and all their state were left untouched.
- `kmh-tpuvm-v5e-16-kaiminghe-434bf6` passed the all-worker occupancy check twice, including immediately before staging. Worker 0 had 30 GB free and writable TPU logs.

## Results

Pending completion.

## Updates

- 2026-11-19 18:49 UTC: The first selective setup attempt exposed a legacy `gcloud storage` temporary-space failure before any accelerator process started. The retry localized only the model, manifest, pinned benchmarks, and final checkpoint required by evaluation. HumanEval+ and MBPP+ hashes and task counts matched the protocol. The 5,929,296,710-byte checkpoint matched SHA-256 `54553fbf494bbed7198392dd99f96e921ddafe4402cbcd996f4f362eb42092bf` and its completion marker at update 1,600.
- 2026-11-19 UTC: The final all-worker ownership check reported every worker `AVAILABLE`. Evaluation launched on worker 0 as PID `3346917`; the initial process and protocol-file check passed. The next inspection is scheduled 30 minutes after launch, matching the requested monitoring cadence.

## Artifacts

- Source snapshot: `experiment_log/2026-11-19/artifacts/dllm-progressive-teacher-weighted-1600-evaluation/source-35a1613.tar.gz`, SHA-256 `dac6e70edfd54b85161b40ee773f5ad4cb8454a30d32912c84ba731b201a49fa`.
- Regional prefix: `gs://kmh-gcp-us-west4/kristine/dllm-progressive-teacher-weighted-1600-20261119/evaluation`.
- W&B run ID: `dllm-progressive-teacher-weighted-1600-20261119-evaluation`.
