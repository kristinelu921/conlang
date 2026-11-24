# Isolated Qwen environment setup failures

Date: 2026-11-10 UTC. Stage: dependency setup; no model execution or experiment step occurred.
Authorized host: kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a,
project he-vision-group, us-west4-a, v5e-8; read-only inventory AVAILABLE/READY,
occupancy AVAILABLE worker=t1v-n-4b587646-w-0. CPU only, four threads maximum.

Attempt 1: `python3 -m venv /tmp/island-community-qwen-20261110-224000/venv`.
Exit 1: system Python reports ensurepip unavailable and recommends python3.10-venv.
No shared package or OS mutation was made.

Attempt 2: `/home/kristinelu/elf-runs/runtime/venv/bin/python -m venv /tmp/island-community-qwen-20261110-224000/venv`.
Exit 1: child `venv/bin/python -m ensurepip --upgrade --default-pip` failed.
The partial setup directory remains preserved; no recording/checkpoint exists.

Recovery selected: install virtualenv==20.31.2 into task-local setup-tools using
pip --target, then create a distinct qwen-venv using its bundled pip bootstrap.
Install torch==2.6.0 from the CPU wheel index, transformers==4.51.3,
numpy==1.26.4 and pytest==8.3.5. This leaves the shared runtime untouched.
Verification pending: pip freeze, imports, model construction, remote tests.
