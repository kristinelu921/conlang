# Pretrained frozen TPU gate — failed before the first action

- Attempt: `frozen-pretrained-gate`, 2026-11-11 03:14 UTC; fresh output `frozen-pretrained-gate-recording`.
- Authorized resource: `kmh-tpuvm-v5e-4-spot-myang-ar`, `us-west4-a`, v5litepod-4, topology 2x2, Spot. Fresh preflight 03:14:02 UTC: READY/HEALTHY, sole worker IDLE.
- Published source: `b6f90785210dec206ee344c446f46fe7598a247b`; archive SHA-256 `34952a9c1bfb96715b577b93c704dfac525035b9ef519dd8301c64db38641233`. Private repository: https://github.com/kristinelu921/island-community/commit/b6f90785210dec206ee344c446f46fe7598a247b.
- Exact launcher/environment: `frozen-pretrained-gate.sh`. Scenario: `island_community/scenarios/qwen_10_frozen.json`; seed 42; ten founders; one turn; frozen public Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`; float32; SDPA; static cache 16,384 tokens; 96 reasoning tokens/stage; temperature 0.6, top-p 0.95, top-k 20; four CPU threads. World and reward settings remain the committed scenario's defaults.
- Stage reached: model loading and native TPU initialization succeeded; the first reasoning prefill failed before any action or world turn completed. No valid completed-turn checkpoint.
- Exit: 1 after 42.02 seconds. Detailed log: `frozen-pretrained-gate.log`; timing: `frozen-pretrained-gate-timing.txt`. Host peak RSS 5,090,656 KiB; this is not TPU HBM usage.
- Error: `E0200 RuntimeUnexpectedCoreHalt`, `[user] vmem_address_out_of_range_vld0`, HLO `scaled_dot_product_cudnn_attention.196`. This establishes a compiled attention execution failure. The specific compiler/kernel defect is not identified; the evidence does not establish physical hardware failure or HBM exhaustion.
- Model shape: 28 layers, hidden width 1024, 16 query heads, 8 KV heads, head dimension 128. Earlier passing tests used tiny head dimension 8, so they did not cover the real attention shape.
- Source, model cache, and output were node-local. Model downloaded directly into the selected VM; no cross-region dataset/checkpoint copy.
- Recovery under investigation: use Transformers' parameter-free eager attention backend for native TPU and verify real attention dimensions before a fresh pretrained retry. Preserve float32 weights, prompts, sampling parameters, rewards, and all prior evidence. No reboot or device reset is authorized or attempted.
- Postflight lock: empty current-user file, timestamp `2026-11-11 03:14:22.605273564 +0000`; may be cleaned only after the explicitly authorized all-device idle checks.

Reference: https://openxla.org/xla/errors/error_0200 distinguishes runtime hardware constraints/compiler assertions from infrastructure failures; https://huggingface.co/docs/transformers/main/attention_interface documents selecting eager attention without changing model parameters.
