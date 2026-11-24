# Winning 32→16 evaluation

## Run Summary

| Field | Value |
|---|---|
| Run ID | `dllm-winning-progressive-teacher-weighted-32-16-eval-20261119` |
| Status | `completed` |
| Started | `2026-11-19 UTC` |
| Finished | `2026-11-19 UTC` |
| Git revision | `801b9ec1e3c6c8a1c59dc9a0c8f86560fe81210c` |
| Hardware | `kmh-tpuvm-v5e-16-spot-zande-pool-w4-1`, worker 1, four local v5e devices |
| Checkpoint | `progressive-teacher-weighted`, update 400, SHA-256 `bf752b457b054ce60e93d0cf148b3ef5921c177b57b3bb596f44d884589eecaf` |
| Command | `python -m reproduction.evaluate_tpu --data <node-local-data> --checkpoint <verified-winning-checkpoint> --output <node-local-output> --steps 16 --draws 32 --seed 42 --context 512 --new-tokens 128` |

## Protocol

- EvalPlus `0.3.1`.
- HumanEval+ v0.1.10: 164 tasks, decompressed SHA-256 `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- MBPP+ v0.2.0: 378 tasks, decompressed SHA-256 `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`.
- Deterministic 16-step confidence-remasking sampler, 128 generated tokens, 512-token context.
- Report rolling-reference pseudo-perplexity and the 32-draw conditional diffusion NLL bound separately. Neither is mixed with AR conditional perplexity.
- Training and evaluation seed: `42`.

## Results

| Dataset | Tasks | Rolling pseudo-perplexity | 32-draw bound exponential | Mean latency | Sanitized base pass@1 | Sanitized Plus pass@1 |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval+ | 164 | 6.940964 | 3.838841 | 0.086096 s | 0.6098% | 0.0000% |
| MBPP+ | 378 | 11.780868 | 5.422231 | 0.086083 s | 5.0265% | 3.7037% |

Raw base/Plus pass@1 was 0.6098%/0% on HumanEval+ and 0%/0% on MBPP+. The MBPP sanitation step recovered valid completions that contained surrounding text or formatting. Mean generated lengths were 127.93 tokens on HumanEval+ and 127.43 on MBPP+ against a 128-token cap.

The paper reported 27.12 HumanEval and 35.78 MBPP for its historical 32→16 row, but its rolling implementation and benchmark variants differ. The corrected values above are therefore a new reproducible result, not a like-for-like percentage improvement.

## Artifacts

- Node-local output: `/home/kristinelu/elf-runs/dllm-loss-ablation-32-16-20261119-r4-worker1/eval-winning-progressive-teacher-weighted-32-16/samples`
- Regional prefix: `gs://kmh-gcp-us-west4/kristine/dllm-loss-ablation-32-16-20261119/evaluation-winning-progressive-teacher-weighted-32-16`
- Thirty-minute monitor PID: `231486`
- Grader image: `sha256:23f83f7cc70ce855040a1b60692766099a0c806aaa387255ca668d0a56009746`; canonical and deliberately incorrect fixtures passed their expected checks before grading.
- W&B evaluation: `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/dllm-distillation-reproduction/runs/dllm-loss-ablation-32-16-20261119-winning-evaluation`
- Local dashboard: `http://localhost:8781/`

## Verification

- Evaluation exit status `0`; 164 HumanEval+ and 378 MBPP+ records.
- Regional sample summaries, raw samples, score files, grader image ID, and completion marker verified present.
- All four TPU workers reported `AVAILABLE` after evaluation.
- Dashboard generator compiled, both plots were visually inspected, and the localhost server returned the completed page.
