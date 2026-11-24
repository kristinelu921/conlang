# Native TPU check 1: topology initialization failure

Status: failed at backend initialization, before Qwen inference or training.
TPU: kmh-tpuvm-v6e-16-kaiminghe-070b9f, us-east5-b, v6e-16 / 4x4 / Spot.
Fresh all-worker occupancy gate was IDLE; see v6e-native-check1-preflight.json.
Node-local snapshot: /tmp/island-community-qwen-tpu-20261110-234000/source-check1.
Node-local log: /tmp/island-community-qwen-tpu-20261110-234000/native-check1.log.

Command from source-check1:
`HF_HOME=/tmp/island-community-qwen-tpu-20261110-234000/hf-cache XDG_CACHE_HOME=/tmp/island-community-qwen-tpu-20261110-234000/cache TORCHINDUCTOR_CACHE_DIR=/tmp/island-community-qwen-tpu-20261110-234000/torch-cache TPU_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false ISLAND_ALLOW_NEURAL_TESTS=1 ISLAND_QWEN_TEST_DEVICE=tpu /tmp/island-community-qwen-tpu-20261110-234000/venv-native/bin/python -u -m pytest island_community/tests/test_qwen_backend.py -k 'decode_update_isolation_and_checkpoint and frozen' -vv -s`.

Exit 1: one test failed, three deselected, 11.69 seconds. PyTorch manual_seed
triggered TorchTPU initialization; runtime expected four host devices but the
visibility mask exposed one. Error requested matching deepsea_chips_per_host_bounds
and deepsea_host_bounds. This is a runtime topology mismatch, not model shape
or JSON parsing failure. No inference step completed; no checkpoint exists.

Resolved isolated runtime: Python 3.12.16, torch 2.15.0+cpu, torch-tpu 0.2.0,
libtpu 0.0.52, transformers 5.22.2, numpy 2.5.4. The separate NumPy>=2
dependency environment installed successfully and preserved dependency attempt 1.
Recovery: inspect the installed launch-environment producer and set the supported
single-device topology before retrying under a new test-log identity.
