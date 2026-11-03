# SmolLM2-135M GSM8K: four retained baselines

These are the four completed refits retained from the two studies, rather than a test-ranked top four. Recipes and stopping epochs were selected on training-only validation; the historical test set was already observed. Each JSON contains the exact refit configuration, metrics, pinned model/data revisions, W&B links, model location and file checksums.

| Run | LR | Batch | Five-shot training | Epochs | Train NLL | Test NLL | Zero-shot | Five-shot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [previous-accuracy](previous-accuracy.json) | 0.001 | 16 | 0% | 11 | 0.000310 | 1.437200 | 11.60% | 0.99% |
| [previous-loss](previous-loss.json) | 0.0001 | 32 | 0% | 3 | 0.357530 | 0.601065 | 7.28% | 4.85% |
| [matched-balanced](matched-balanced.json) | 0.0003 | 32 | 50% | 3 | 0.251852 | 0.634260 | 8.64% | 8.19% |
| [matched-loss](matched-loss.json) | 0.0001 | 32 | 50% | 3 | 0.367189 | 0.600561 | 7.05% | 7.43% |

Accuracy uses all 1,319 test questions, strict exact match and greedy decoding capped at 256 tokens. NLL is in nats per answer token. Both new zero-shot scores reproduce the original fixed-width evaluation exactly.

The matched-balanced recipe improves both prompt-format accuracies over previous-loss at higher NLL. Matched-loss preserves comparable NLL and improves five-shot accuracy, with slightly lower zero-shot accuracy. Previous-accuracy retains the strongest zero-shot score but has substantially worse NLL and five-shot accuracy.

All runs use seed 1234, full-model float32 fine-tuning, AdamW, weight decay 0.01, 3% warmup and a 12-epoch cosine schedule ending at 10% of peak LR. The new mixed-format refits use microbatch 8 with four-step accumulation for effective batch 32. Selection uses 6,726 training questions and 747 held-out training questions; refits use all 7,473.

The four FP32 checkpoints remain at the GCS locations in the JSON records. Source/results archives are preserved locally under `experiment_log/2026-11-03/smollm2-matched-validation-artifacts/`: `completed-results.tgz` for the earlier runs and `matched-validation-preserved-20261103.tgz` for the matched runs. Their hashes are recorded in each run. Model weights, raw archives, unrelated experiments and workspace changes are excluded from this commit.
