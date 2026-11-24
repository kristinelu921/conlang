# TorchTPU dependency attempt 1

Status: dependency resolution failed before model import or accelerator use.
UTC: 2026-11-10, following the 23:38 all-worker preflight. Exact TPU:
`kmh-tpuvm-v6e-16-kaiminghe-070b9f`, us-east5-b, v6e-16, topology 4x4,
READY/HEALTHY, Spot. Both required ownership authorization phrases were given.
All four workers were idle before setup; evidence: v6e-runtime-setup-preflight.json.

Exact setup source: `/tmp/island-community-qwen-tpu-setup-20261110.sh`.
Worker 0, task-local root `/tmp/island-community-qwen-tpu-20261110-234000`.
UV 0.13.7 and isolated CPython 3.12.16 installed successfully. The failing command:

```
python3 -m uv pip install --python /tmp/island-community-qwen-tpu-20261110-234000/venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match torch-tpu==0.2.0 transformers==5.22.2 pytest==8.3.5 numpy==1.26.4 accelerate
```

Exit status 1. Resolver evidence: TorchTPU 0.2.0 requires numpy>=2, conflicting
with the prior CPU runtime's numpy==1.26.4 pin. No model was imported, no step
completed, no checkpoint exists, and no accelerator process was launched.
The previous CPU environment is unaffected. The new environment is preserved.
Repair selected: create a distinct TPU environment with numpy>=2, retain the
independent old CPU dependency set, and save the resolved exact versions.
