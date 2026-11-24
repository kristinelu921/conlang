# SmolLM2 fine-tuned checkpoint evaluation, attempt 1

- Status: failed before any test generation, 2026-11-02T21:25Z.
- TPU: `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`,
  dedicated Spot v5e-8, `us-west4-a`, READY/HEALTHY.
- Command: `bash evaluate.sh` from the fine-tune source directory.
- Stage: checkpoint import during `verify_export.py`; exit code 1.
- Training completed 702 updates; the full optimizer state and original trained
  float32 Flax weights are intact and mirrored. No training is repeated.
- Failure: Transformers 4.57.6 resolves the colocated `flax_model.msgpack` before
  `model.safetensors`, even on the `from_pt=True` path, then attempts to read the
  msgpack file with `torch.load`. This raises `Unsupported operand 130`.
- A separate export audit found the PyTorch output has 269,060,552 bytes,
  consistent with the base config's bfloat16 default. Its construction must
  explicitly request float32 to preserve the trained parameters.
- Repair: create a fresh `final-model-fp32` directory containing only the
  PyTorch format and tokenizer, using explicit float32 initialization. Load the
  original trained Flax weights independently for exact tensor comparison.
- Verification: require all saved tensors to match the trained checkpoint,
  then require reference/TPU logit and greedy-generation parity before scoring.
- Evidence: `smollm2-135m-gsm8k-sft-artifacts/evaluation-monitor-history.log` and
  remote `sft-20261102-attempt1/evaluation.log`. Preserve the original failed
  export and evaluation; retry in a fresh evaluation directory.
