# SmolLM2-135M on GSM8K

## Run Summary

| Field | Value |
|---|---|
| Run ID | `smollm2-135m-gsm8k-20261102` |
| Status | interrupted |
| Started | 2026-11-02T20:36Z; exact timestamp in `full-start.json` |
| Finished | not recorded |
| Baseline | not specified |
| Git revision | `3cf8cb45f56080c5361dc87929deff4376069061`; pre-existing unrelated worktree changes |
| Command | `bash experiment_log/2026-11-02/smollm2-135m-gsm8k-artifacts/run.sh` |
| Seed | Python: 0; NumPy, PyTorch, few-shot sampler: 1234 |
| Hardware | CPU; AMD EPYC 7B12; 64 inference workers, one thread each; no accelerator |

## Configuration

Source: [run.sh](smollm2-135m-gsm8k-artifacts/run.sh) and
[pinned task](smollm2-135m-gsm8k-artifacts/tasks/gsm8k_pinned.yaml).

```yaml
model: HuggingFaceTB/SmolLM2-135M
model_revision: 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
variant: base
dataset: openai/gsm8k
dataset_configuration: main
dataset_revision: 740312add88f781978c0658806c59bc2815b9866
test_examples: 1319
fewshot_split: train
num_fewshot: 5
harness: lm-eval==0.4.13
task_version: 3.0
device: cpu
dtype: float32
attention: sdpa
batch_size: 8
worker_count: 64
worker_threads: 1
max_gen_toks: 256
do_sample: false
temperature: 0.0
repeats: 1
apply_chat_template: false
stop_strings: ['Question:', '</s>', '<|im_end|>']
seed: [0, 1234, 1234, 1234]
```

The task is an archived copy of the harness's standard GSM8K task, changed only
to pin the dataset revision and give it a distinct task name. Prompt formatting,
few-shot sampling, answer extraction, and metrics are unchanged.

The CPU adapter distributes generation batches only. The harness constructs all
prompts once using its standard random-number sequence, and receives responses
in their original order before applying its own extraction and scoring. Workers
use the unchanged Hugging Face backend. Model context length is 8,192 tokens;
the model contains 134,515,008 parameters. Both model and dataset are pinned by
commit, and downloaded files are recorded in `download-manifest.json` with
SHA-256 hashes. Inference runs entirely offline.

## Changed Hyperparameters

No confirmed changes; no explicit baseline was specified.

## Results

The local full run was interrupted at the user's direction before generation
began: 0/1,319 full-test questions completed. The eight-example preflights are
execution checks, not a benchmark result. Continuation will use remote compute.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Exact command | [run.sh](smollm2-135m-gsm8k-artifacts/run.sh) | yes |
| Task configuration | [gsm8k_pinned.yaml](smollm2-135m-gsm8k-artifacts/tasks/gsm8k_pinned.yaml) | yes |
| Upstream task snapshot | [gsm8k-upstream.yaml](smollm2-135m-gsm8k-artifacts/gsm8k-upstream.yaml) | yes |
| Model | `/tmp/conlang-smollm2-gsm8k-cache/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | yes |
| Tracking run | not recorded | no |

## Failure Details

The first preflight failed before generation because the preinstalled TPU
extension attempted accelerator metadata discovery even though CPU was selected.
Setting `USE_TORCH_XLA=0` and `ACCELERATE_USE_CPU=true` resolved it. The original
trace is retained in `smollm2-135m-gsm8k-artifacts/preflight.log`.

## Observations

The pinned dataset contains 7,473 training examples and 1,319 test examples.
The user's model name is interpreted as the base model, not the Instruct variant.

## Interpretation

This measures prompted arithmetic reasoning using exact final-answer matching.
Scores depend on the model variant, prompt, few-shot examples, token budget,
and extraction rule.

## Updates

- 2026-11-02: Downloaded pinned public model and dataset and prepared offline CPU evaluation.
- 2026-11-02T20:31:39Z: Standard serial eight-example preflight completed; 39 seconds in generation, 57.97 seconds in evaluation. It used `run-serial.sh --limit 8 --output_path experiment_log/2026-11-02/smollm2-135m-gsm8k-artifacts/preflight-cpu`; output was redirected to `preflight-cpu.log`. Its 0/8 score is a smoke check, not the benchmark result.
- 2026-11-02: Two adapter tests passed: out-of-order batch completion restores original response order, and incomplete backend responses raise an error.
- 2026-11-02T20:35:59Z: Parallel-adapter preflight completed in 189 seconds in generation (one active worker for one batch). All eight prompts, raw responses, extracted answers, and both scores matched the standard serial backend exactly; evidence in `preflight-comparison.json`. The configured maximum was 16 workers; only one was needed for eight examples.
- 2026-11-02T20:36Z: Started all 1,319 test examples with 64 workers and one thread per worker. The host exposes 120 physical cores, 240 hardware threads, and approximately 400 GiB RAM. User authorized provisioning a `kristine` Spot v4-8 if needed; no TPU was provisioned.
- 2026-11-02T20:37Z: User instructed never to use a development machine for runs or jobs. Sent TERM to the exact local benchmark process (PID 2105906), verified exit 143, and recorded 0 completed full-test generations in `interrupted.json`. No CPU inference workers had been spawned. All subsequent inference and benchmark execution must be remote, including smoke tests.
