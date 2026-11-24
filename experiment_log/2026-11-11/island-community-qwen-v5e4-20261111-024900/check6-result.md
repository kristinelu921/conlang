# Recorded sampling-support regression — passed

All **6 focused TPU tests passed in 48.42 seconds**, exit 0. Exact command/environment: `native-check6.sh`; log: `native-check6.log`; source archive SHA-256: `c99aa6676e64c4d4bbd393599b89b0a614e7165b4a70f53f10bd2e2bacbde13e`.

Generation now saves the exact finite-token supports after its temperature, top-k, top-p, and grammar processing. The gradient pass normalizes differentiable logits over those recorded sets instead of recomputing a truncation boundary. A dedicated regression raises an excluded token's replay score by 1,000; the selected-token log probability stays finite and unchanged on the actual generation support. Other tests cover nonzero independent adapter gradients, unchanged base weights, actual disk/optimizer checkpoint recovery, CPU/TPU attention agreement, and full world recorder/resume behavior.

Runtime metadata identifies `policy_support=recorded_generation_tokens`. The scientific world, rewards, learning rates, model precision, and sampling settings remain unchanged. The full pretrained adapter retry is still required before claiming successful real-model learning.

## Result transport repair

Verbose SSH diagnostics identified `message authentication code incorrect` during a default ChaCha20-Poly1305 recording transfer. The underlying transport defect is not established. Retrying with supported authenticated `aes256-gcm@openssh.com` succeeded without disabling encryption or integrity checks. The bounded recurring monitor then mirrored the complete frozen recording and its local action/base-weight audit passed again. Logs: `transfer-directory-debug.txt`, `frozen-pretrained-gate2-monitor.jsonl`. The monitor limits each recording to 512 MiB and requires free space exceeding three times its size plus 100 MiB.

The development root filesystem remains intermittently full because of activity outside this task; project NFS has several GiB free. This is separate from the authenticated-transfer error. No other user's files or jobs were altered.
