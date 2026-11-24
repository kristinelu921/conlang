# SmolLM2-135M GSM8K remote TPU benchmark

## Run Summary

| Field | Value |
|---|---|
| Run ID | `smollm2-135m-gsm8k-tpu-20261102` |
| Status | completed |
| Started | 2026-11-02T20:53:32Z; provisioning began 2026-11-02T20:38:25Z |
| Finished | 2026-11-02T20:55:31Z |
| Baseline | not specified |
| Git revision | `3cf8cb45f56080c5361dc87929deff4376069061`; benchmark source is archived alongside this report |
| Command | `cd /home/kristinelu/smollm2-gsm8k/20261102-attempt2 && bash run.sh`; full invocation in linked launcher |
| Seed | Python 0; NumPy, PyTorch, few-shot sampler 1234; greedy generation |
| Hardware | One Spot `v5litepod-8`, 8 TPU chips on one VM, `us-west4-a` |

## Configuration

Source: [launcher](smollm2-135m-gsm8k-tpu-artifacts/run.sh),
[backend](smollm2-135m-gsm8k-tpu-artifacts/flax_tpu_lm_eval.py), and
[task](smollm2-135m-gsm8k-tpu-artifacts/tasks/gsm8k_pinned.yaml).

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
backend: hf-flax-tpu
jax: 0.6.2
flax: 0.10.5
transformers: 4.57.6
torch: 2.7.0+cpu
numpy: 1.26.4
device: tpu
dtype: float32
matmul_precision: highest
batch_size_per_device: 8
devices: 8
global_batch_size: 64
max_gen_toks: 256
do_sample: false
temperature: 0.0
repeats: 1
apply_chat_template: false
add_special_tokens: false
stop_strings: ['Question:', '</s>', '<|im_end|>']
seed: [0, 1234, 1234, 1234]
```

The harness constructs all prompts and computes its standard strict and flexible
exact-match metrics. The adapter only generates continuations. The Flax Llama
implementation requires two checkpoint-compatibility corrections: use the
checkpoint's RoPE base of 100,000 instead of the implementation's hard-coded
10,000, and tie the output projection to the input embedding. No model weight is
trained or changed. Remote validation must compare long-position logits and
cached greedy output against the native PyTorch model before promotion.

## Changed Hyperparameters

No confirmed changes against an explicit baseline. Relative to the interrupted
local preflight, generation moves to TPU/Flax and Transformers 4.57.6; checkpoint,
dataset, few-shot seed, prompt, token budget, precision, and scoring are retained.

## Results

All 1,319 GSM8K `main` test examples completed successfully. Higher accuracy is
better. These are measurements of the base checkpoint, not the Instruct model.

| Scoring filter | Correct / total | Exact-match accuracy | Standard error (fraction) |
|---|---:|---:|---:|
| strict-match | 20 / 1,319 | 0.015163002274450341 (1.52%) | 0.0033660229497263316 |
| flexible-extract | 25 / 1,319 | 0.018953752843062926 (1.90%) | 0.003756078341031475 |

Source: [validated summary](smollm2-135m-gsm8k-tpu-artifacts/completed/summary.json)
and [native harness results](smollm2-135m-gsm8k-tpu-artifacts/completed/full/HuggingFaceTB__SmolLM2-135M/results_2026-11-02T20-55-25.807897.json).
Both metrics score the same generations. Strict scoring requires the answer
marker `####`; flexible scoring extracts the last matching number.

Generation took 71.439564195 seconds, including 52.59603215800007 seconds for the
first batch and its compilation. Subsequent batches took approximately 0.9344
seconds each. These times include generation through the 256-token budget before
text stop markers are applied; they are not a latency benchmark against another
backend. Harness evaluation time was 107.97689909300004 seconds; the entire
remote command took 119 seconds, including startup and result writing.

Maximum prompt length was 1,663 tokens, padded to 1,664; no prompt was truncated.
Global batch size was 64, with duplicated padding entries removed from the final
batch. Exactly 1,319 ordered generations and 1,319 unique document IDs per
scoring filter were verified. Sample scores reproduce both aggregate metrics.

[Open the locally accessible chart](http://localhost:8768/) or
[view the standalone plot](smollm2-135m-gsm8k-tpu-artifacts/completed/accuracy.png).
The static page is served on the remote VM through an SSH tunnel; no experiment
or web server runs on the development machine.

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Source and command | [remote artifacts](smollm2-135m-gsm8k-tpu-artifacts/) | yes |
| TPU backend parity check | [verification evidence](smollm2-135m-gsm8k-tpu-artifacts/completed/backend-verification.json) | passed |
| Per-example generations and scores | [samples](smollm2-135m-gsm8k-tpu-artifacts/completed/full/HuggingFaceTB__SmolLM2-135M/samples_gsm8k_pinned_2026-11-02T20-55-25.807897.jsonl) | all 1,319 examples, two filters |
| Canonical worker-0 log | [full.log](smollm2-135m-gsm8k-tpu-artifacts/completed/full.log) | SHA-256 verified |
| Exact installed environment | [requirements.txt](smollm2-135m-gsm8k-tpu-artifacts/completed/requirements.txt) | verified |
| Model and dataset integrity | [download manifest](smollm2-135m-gsm8k-tpu-artifacts/completed/download-manifest.json) | SHA-256 verified |
| Retrieved artifact checksums | [results.sha256](smollm2-135m-gsm8k-tpu-artifacts/completed/results.sha256) | every listed file matches |
| Archived reference comparison | [comparison](smollm2-135m-gsm8k-tpu-artifacts/archived-reference-comparison.json) | 8/8 prompts, responses and strict scores match |
| Interrupted local preflight | [predecessor report](smollm2-135m-gsm8k.md) | yes |
| v4 provisioning log | [tpu-provision.log](smollm2-135m-gsm8k-artifacts/tpu-provision.log) | yes |
| v5e provisioning log | [tpu-provision-v5e.log](smollm2-135m-gsm8k-tpu-artifacts/tpu-provision-v5e.log) | yes |
| Tracking run | not used; local structured results and logs | not applicable |

## Failure Details

Three v4-8 Spot creation attempts failed with TPU API code 8, `no more capacity`
in `us-central2-b`. No model, dataset, or runtime was staged to those targets,
no step completed, and no checkpoint exists. The cause is confirmed by the
create-operation responses, not inferred from a process timeout.

| Attempt | Exact TPU name suffix after `kmh-tpuvm-v4-8-spot-kristine-` | Outcome |
|---|---|---|
| 1 | `02-20-38-25-813120365-0-us-central2-b` | API code 8, capacity unavailable |
| 2 | `02-20-40-21-578291147-0-us-central2-b` | API code 8, capacity unavailable |
| 3 | `02-20-41-36-680704251-0-us-central2-b` | Audit log confirms terminal code 8 at 20:42:03Z; direct describe is NOT_FOUND |

The v4 orchestrator was stopped before starting alternative provisioning. The
third request's terminal audit operation was
`operation-1793652098368-65f512c2c83f0-5451be28-0455ce37`. No unresolved allocation
or active benchmark capacity remained.

## Observations

- The user prohibits all experiment jobs on development machines, including
  smoke runs. Only editing, control-plane orchestration, and result inspection
  are performed locally from this point onward.
- The initial owner-filtered inventory had no AVAILABLE resource.
- The Central2 bucket was verified in `US-CENTRAL2`. Its `kristine/elf/datasets/`
  prefix contains no GSM8K copy, and `kristine/datasets/gsm8k/**` matched no objects.
  A different selected region must be checked independently.

## Interpretation

The base 135M model solves few GSM8K problems in this configuration. Flexible
extraction recovers five additional correct answers from the same outputs;
this is a scoring difference, not a separate model run. The verified TPU adapter
uses standard harness prompts and scoring. Small floating-point differences
across backends remain possible outside the checked examples. Published
model-card scores use a different evaluation tool and are not directly compared.

## Updates

- 2026-11-02T20:38:25Z: Began one Spot v4-8 provisioning campaign with a
  benchmark-specific isolated runtime initializer.
- 2026-11-02T20:42Z: User authorized trying other TPU allocations and explicitly
  approved one Spot v5e-8. Keep at most one active benchmark TPU.
- 2026-11-02T20:43Z: Started `query_tpus.sh v5e 8 1` with
  `ZONES_OVERRIDE='us-west4-a us-central1-a'`, `RETRY_DELAY_SECONDS=60`, and the
  benchmark initializer. No other-owner TPU was inspected.
- 2026-11-02T20:46Z: Allocated
  `kmh-tpuvm-v5e-8-spot-kristine-02-20-42-52-943789975-0-us-west4-a`.
  Direct API confirms READY, HEALTHY, Spot=true; final filtered inventory lists
  the exact target AVAILABLE. No additional benchmark TPU is allocated.
- 2026-11-02T20:46Z: Runtime setup attempt 1 failed before inference because
  Python 3.10 lacked `ensurepip`; preserved separately in
  [runtime failure report](smollm2-135m-gsm8k-runtime-attempt1.md).
  Installed `python3.10-venv` and restarted isolated setup on the same VM.
- 2026-11-02T20:48Z: Staged source snapshot to node-local
  `/home/kristinelu/smollm2-gsm8k/20261102-attempt1`; verified archive SHA-256
  `0f3de7075fb846b9c591c06fe0aca801e04beff4ea32c5b7a2f3c73bc26be13d`
  and all six source files against `source.sha256` on the VM.
  The regional bucket `gs://kmh-gcp-us-west4` is in US-WEST4; checked owned
  dataset locations contain no GSM8K copy. Public pinned data will be fetched
  directly on the TPU VM and verified against known hashes. No dataset is
  copied from the development machine or another region.
- 2026-11-02T20:50Z: Backend attempt1 failed before inference because the
  installed Transformers converter does not support `_do_init=False`; preserved
  [separate failure report](smollm2-135m-gsm8k-backend-attempt1.md).
  Staged the corrected loader at
  `/home/kristinelu/smollm2-gsm8k/20261102-attempt2`, archive SHA-256
  `66a5e8f1c0643a64883fbcdbd8f326987a3fea51e00933633510219e95764331`.
- 2026-11-02T20:53Z: Numerical verification passed on all eight visible TPU
  devices, using one device for comparison against the PyTorch reference on the
  same remote VM. Tested prompt lengths 13, 22, and 1,413 tokens; maximum
  absolute final-token logit difference `3.719329833984375e-05` (threshold 0.02).
  Two left-padded 16-token greedy continuations matched exactly, and both
  stop-string checks passed. The missing stock Flax head is explicitly replaced
  by the checkpoint embedding transpose, and all inference weights are upcast
  to float32. Loader precision/missing-head notices precede these corrections.
- 2026-11-02T20:53Z: A fresh occupancy gate returned AVAILABLE. Started
  `bash run.sh` from attempt2 for the complete 1,319-example evaluation.
- 2026-11-02T20:55:31Z: Full evaluation completed with exit code 0. Worker 0 is
  the only worker and canonical process-zero log. All 1,319 generations and both
  score aggregates are present. The harness's Git metadata probe printed
  `not a git repository` because the immutable source bundle contains no `.git`;
  source provenance is instead recorded by the repository SHA and bundle/file
  checksums. This did not prevent successful evaluation or result writing.
- 2026-11-02T20:56Z: Report parsing initially failed because Python
  `str.splitlines()` splits Unicode separators inside valid JSON strings.
  Reading JSONL by newline-delimited file iteration fixed report processing;
  no inference was rerun or modified. Both the failed processing log and the
  successful coverage verification are preserved in the artifact directory.
- 2026-11-02T20:57Z: Retrieved completed archive SHA-256
  `a7b6d069b5c143cb108418ca5dfee501983df18c740dc66670d2e63ffb9e08a5`;
  archive and all enclosed result hashes match. Comparing saved results only,
  all eight archived reference prompts and full responses match exactly.
  Current API state remains READY, HEALTHY, Spot=true; every TPU device is idle
  and no evaluation process remains. The TPU is still allocated, hosting only
  the static result page. No resource was deleted.
