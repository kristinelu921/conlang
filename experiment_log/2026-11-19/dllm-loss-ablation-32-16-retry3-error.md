# dLLM loss ablation retry 3 disk failure

- Status: failed during the fifth arm; four arms completed successfully
- TPU: `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, worker 0, `us-west4-a`
- Exact run: `dllm-loss-ablation-32-16-20261119`, output identity `retry-r3/output-r3`
- Failed arm: `direct-low-lr`
- Last computed update: 200
- Last valid checkpoint: update 100, 5,929,296,470 bytes, SHA-256 `e61a64db2df22724ccbe335f8b1c11b90231dfea1996b79f63664d51f6b8c721`
- Error: worker 0 reached 100% disk use while `pickle.dump` wrote the atomic `latest.tmp` for update 200. The existing `latest.pkl` and `latest.complete.json` remained intact.
- Ownership evidence: the exact current-user trainer PID `283221` was the sole VFIO/lock holder before failure. After the process exited, the occupancy helper reported all four workers `AVAILABLE`.
- Preservation: no checkpoint or output was deleted. The valid update-100 checkpoint was relayed through `experiment_log/2026-11-19/artifacts/direct-recovery-step100/`, and its size and SHA-256 were reverified.
- Recovery: immutable source revision `801b9ec1e3c6c8a1c59dc9a0c8f86560fe81210c`, pinned same-region data, exact dependency lock, and the checkpoint were localized to worker 1. Retry `r5` deterministically reproduced updates 101–200 and completed update 400. Final checkpoint SHA-256: `b597a2e23a05a04b96a0d0c1b010c8889cd0c0c42d3c31e417da4dc27885fff0`.
