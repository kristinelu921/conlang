# SmolLM2 TPU backend verification, attempt 1

## Run Summary

- Status: failed during checkpoint conversion, 2026-11-02T20:50Z.
- TPU: `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`,
  `us-west4-a`, Spot `v5litepod-8`, one worker; latest API READY and HEALTHY.
- Run path: `/home/kristinelu/smollm2-gsm8k/20261102-attempt1`.
- Command: `bash prepare.sh`; Python command: runtime venv `python verify_backend.py`.
- Exit status: 1; no model forward pass, benchmark examples, or checkpoints.
- Configuration: [benchmark report](smollm2-135m-gsm8k-tpu.md).

## Failure Details

Transformers 4.57.6's PyTorch-to-Flax converter accesses `model.params`, which
raises when the model is loaded with `_do_init=False`. The exception occurs
before numerical comparison, so this attempt gives no accuracy or parity result.
Model and dataset downloads passed hash verification; all files are node-local.

- Evidence: [preparation SSH log](smollm2-135m-gsm8k-tpu-artifacts/preparation-ssh.log).
- Immutable source: `smollm2-135m-gsm8k-tpu-artifacts/source-attempt1.tgz`, SHA-256
  `0f3de7075fb846b9c591c06fe0aca801e04beff4ea32c5b7a2f3c73bc26be13d`.
- Repair: use the converter's supported default initialization and read
  `model.params`; preserve checkpoint embeddings for the tied output head.
- Recovery: stage a fresh attempt2 source directory; reuse the verified
  node-local download cache; rerun the same numerical tests before evaluation.
- No hyperparameter, model revision, dataset revision, or scoring changes.
