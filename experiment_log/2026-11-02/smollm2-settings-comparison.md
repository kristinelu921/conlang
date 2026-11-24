# SmolLM2-135M GSM8K settings comparison

## Run Summary

| Field | Value |
|---|---|
| Source study | `smollm2-gsm8k-research-20261102` |
| Status | completed; comparison of existing results only |
| Experiment baseline | Original three-epoch fine-tune; within-search comparisons use the specified matching control |
| Study seeds | Optimization 1234; validation split 20261102 |
| Experiment hardware | Dedicated kristine Spot v5e-8 in us-west4-a, as recorded in the source report |
| New model work | None; renderer reads saved scalar metrics only |
| Source/configuration closure | [Original study report](smollm2-gsm8k-research.md) and [all trial configurations](smollm2-gsm8k-research-artifacts/report/comparison-v1/settings.json) |
| Source summary SHA-256 | `43dd259010de72a7c39fd7346b4efb8e677fa98af7271fa87072f76dc3eb8149` |
| Git revision | Not recorded for this standalone derived report; renderer and artifact checksums are preserved |

## Configuration and Command

Training defaults: full-parameter float32 SFT, 6,726 optimization and 747 held-out
validation examples, answer-only shifted cross entropy including rationale/EOS,
640-token sequences, no truncation or packing, AdamW beta1 0.9/beta2 0.999,
epsilon 1e-8, clipping 1.0, weight decay 0.01, warmup 3%, cosine decay over a
12-epoch horizon ending at 10% of peak LR. Main trials have minimum three epochs
and patience three. The smoothing probe has a six-epoch cap and patience two.
Each row's full configuration and actual completed epochs are in `settings.json`.

Exact rendering command, from repository root:

```bash
python3 experiment_log/2026-11-03/render_smollm2_settings.py --summary experiment_log/2026-11-02/smollm2-gsm8k-research-artifacts/report/summary.json --output experiment_log/2026-11-02/smollm2-gsm8k-research-artifacts/report/comparison-v1
```

## Changed Hyperparameters and Results

The [19-row table](smollm2-gsm8k-research-artifacts/report/comparison-v1/table.md)
lists learning rate, batch size, changes from defaults, maximum zero-shot strict
validation accuracy and its epoch, minimum validation NLL and its epoch, and an
interpretation. The HTML table also includes NLL at the accuracy-selected epoch.
Accuracy and minimum NLL frequently come from different checkpoints; they must
not be interpreted as a jointly attainable point. The scatter plot always pairs
accuracy and NLL from the same checkpoint. Loss units are nats per answer token.

The accuracy recipe is LR 1e-3, batch 16, selected epoch 11, with 100/747 correct
on validation. Its full-data refit obtains 153/1319 zero-shot test answers but only
13/1319 five-shot answers, with test NLL 1.437200022346236. The loss recipe is
LR 1e-4, batch 32, selected epoch 3, with minimum validation NLL
0.6045382790792161. Its full-data refit obtains 96/1319 zero-shot and 64/1319
five-shot answers, with test NLL 0.6010654320418056. Test results did not select
these recipes. Full train/test comparisons and original baseline metrics remain
in the source study report.

## Artifacts and Verification

- [Live sortable table and labeled plot](http://localhost:8769/comparison-v1/).
- [CSV](smollm2-gsm8k-research-artifacts/report/comparison-v1/settings.csv),
  [exact scalar/configuration JSON](smollm2-gsm8k-research-artifacts/report/comparison-v1/settings.json),
  [SVG](smollm2-gsm8k-research-artifacts/report/comparison-v1/accuracy-loss.svg),
  [PNG](smollm2-gsm8k-research-artifacts/report/comparison-v1/accuracy-loss.png),
  and [artifact checksums](smollm2-gsm8k-research-artifacts/report/comparison-v1/manifest.json).
- Verified all 19 configurations, correct-answer counts, both loss fields and both
  epoch identities against the original saved summary. CSV has 19 rows; HTML has
  19 data rows plus one header. All generated file sizes and SHA-256 hashes match
  the manifest. The PNG was visually inspected and the localhost page returned
  HTTP 200. The existing server remains running.
- Original experiment files were preserved. Rendering required no new model
  evaluation, training or remote experiment compute.

## Observations and Limitations

Raising LR at batch 32 improves peak zero-shot accuracy within the tested range,
but late selected checkpoints have higher NLL. Batch 16 wins accuracy at LR1e-3;
batch 8 does worse and batch 64 gives lower NLL with lower accuracy. LR2e-3 is
worse than LR1e-3 at batch 16 on both reported objectives. Weight decay zero yields
99 rather than 100 correct answers, a one-question difference that does not
establish a meaningful effect. Stronger decay, final-answer weighting, longer
warmup and lower terminal LR do not beat the accuracy winner. A 20-epoch schedule
stopped after ten epochs; it must not be described as twenty completed epochs.
Smoothing 0.05 improves peak accuracy relative to trial 04 but worsens minimum NLL
and used a shorter cap. Smoothing 0.1 and alternate shuffle seeds were not run.

These are single-seed, adaptively selected validation observations, not controlled
multi-seed causal estimates or proof of a globally optimal recipe. Five-shot
validation was absent from this completed study. The new proposed mixed-prompt
follow-up has not run and contributes no rows to this comparison.
