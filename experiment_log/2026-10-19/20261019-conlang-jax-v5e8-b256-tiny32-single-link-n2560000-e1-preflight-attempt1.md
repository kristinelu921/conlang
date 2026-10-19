# One-epoch 2.56M run remote model-preflight attempt 1

## Run Summary

| Field | Value |
|---|---|
| Run ID | `20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-preflight-attempt1` |
| Attempt | remote CPU model-preflight attempt 1 |
| Status | `failed` |
| Started | `2026-10-19T03:52:00Z` |
| Finished | `2026-10-19T03:55:33Z` |
| Baseline | parent planned run `20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1`; local CPU validation passed 77 tests |
| Git revision | `1e2808f` source/report head; exact configuration commit `74c6486328e2f3e6a0514943da126ff2e9f63989` |
| Command | exact commands below |
| Seed | `1729` |
| Hardware | `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`, `us-west4-a`, Spot `v5litepod-8`; CPU-only validation did not initialize TPU |

## Configuration

Source: `configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml`

The complete effective configuration is embedded byte-for-byte in the canonical
run report. SHA-256:
`a5bc2c80fcabfa7a2564cc4ff061fdb9caaaea76c7207eaff4c40e403aa32b66`.

## Stage and Commands

Stage reached: remote CPU-only model preflight. The exact immutable source had
already passed archive, content-manifest, and selected-config hash checks.

The first command parsed the exact run config successfully and then attempted
to invoke the focused test through the production runtime:

```bash
cd /home/kristinelu/elf-runs/sources/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-20261019T035122.898087842Z
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
/home/kristinelu/elf-runs/runtime/venv/bin/python -c \
  'from train import parse_arguments; config=parse_arguments(["--config", "configs/conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1.yaml"]); assert config.steps == 10000 and config.batch_size == 256 and config.limit_examples is None; assert config.message_length == 1 and config.hidden_size == 32 and config.sender_hidden_size == 32; assert config.sender_num_layers == 2 and config.num_layers == 2; print(f"config_ok run_id={config.run_id} examples={config.steps * config.batch_size}")'
/home/kristinelu/elf-runs/runtime/venv/bin/python -m pytest -q \
  tests/test_jax_train.py \
  -k tiny_single_link_gate_uses_one_32_dimensional_message_and_two_layers \
  --disable-warnings
```

Exit status: `1`. The config assertion printed
`examples=2560000`; the next command exited with
`No module named pytest` before importing the test or initializing any backend.

A direct replacement assertion initially also exited `1` because it incorrectly
required the production parameter count while constructing the model with the
test's deliberately tiny tokenizer. The diagnostic model facts were valid:
message shape `(2, 2, 32)`, sender/receiver layer-1 paths present, no layer-2
path, and 91,186 parameters for that reduced vocabulary.

The corrected direct validation used `build_task_tokenizer()` and succeeded:

```text
model_preflight_ok messages=(2, 2, 32) parameters=228271 sender_depth=2 receiver_depth=2
```

## Changed Hyperparameters

No scientific or runtime parameter changed during recovery.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| preflight | config examples | 2,560,000 | 0 | exact | config assertion |
| preflight | message shape | `(2, 2, 32)` | 0 | exact | corrected direct validation |
| preflight | production parameters | 228,271 | 0 | exact | corrected direct validation |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Immutable source | `/home/kristinelu/elf-runs/sources/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1-20261019T035122.898087842Z` | 75 files; content hash `ea9f31967eecf44def18bda18f10a614c3c06378b9829427d68006f54a1e8d72` |
| Output | `/home/kristinelu/elf-runs/20261019-conlang-jax-v5e8-b256-tiny32-single-link-n2560000-e1` | absent |
| Checkpoint | configured output checkpoint | absent |
| Tracking run | not created | no |

## Failure Details

- Last known step: none; training was not launched.
- First error: `/home/kristinelu/elf-runs/runtime/venv/bin/python: No module named pytest`.
- Exit status: `1`.
- Detailed output: preserved in this report; no launcher log was created.
- Runtime packages: JAX `0.6.2`, Flax `0.10.5`, Optax `0.2.8`, W&B `0.28.1`, pytest absent.
- Checkpoint: none.

## Observations

- The production runtime intentionally contains training dependencies but not
  the development-only pytest runner.
- Neither failed validation command initialized the TPU backend, created the
  configured output directory, created a launcher log, or started W&B.
- The corrected direct production-tokenizer assertion passed without changing
  the shared runtime.

## Interpretation

- This was a validation-tooling failure, not a model, dataset, configuration,
  or TPU failure. The 91,186 versus 228,271 parameter discrepancy was entirely
  explained by tiny-test versus production vocabulary size.

## Recovery

- Confirmed root cause: pytest is absent from the lean production runtime;
  package discovery reported `pytest_present=False`.
- Alternative causes rejected: exact config parsing, model construction,
  message shape, layer depth, production parameter count, source hashes, and
  dataset verification all passed.
- Repair: do not install pytest into the shared runtime; execute the focused
  assertions directly with production dependencies and the production
  tokenizer.
- Verification: corrected direct assertion reported message shape
  `(2, 2, 32)`, 228,271 parameters, sender depth 2, and receiver depth 2.
- Recovery action: retain the exact planned run identity because no training
  process, output claim, tracking run, TPU backend, or checkpoint was created;
  continue to the final prelaunch availability and occupancy gates.

## Updates

- `2026-10-19T03:55:33Z`: Preserved the failed tooling attempt, proved all run
  artifacts absent, and completed the corrected direct model validation without
  mutating the runtime.
